"""Parse the JPX daily-report price files (東京証券取引所日報 → 株価情報（過去分）) into one table.

Input: every .pdf under JPX_RAW_DIR, and every .pdf inside any .zip there, so extracting the
monthly archives is optional. The date and code are read from each row, never from file names
(names inside the zips are Shift-JIS and come out garbled).

Layouts handled (columns are found from the header labels on each page, so small shifts are fine):
    2000-2003  daily PDFs: morning/afternoon OHLC, quote, volume, trading unit
    2004-2019  daily PDFs: the same plus off-auction volume
    2020-      monthly PDFs: morning/afternoon OHLC only (no volume)

Only the kept codes (MEMBERSHIP_DIR/kept_codes.csv) and only common stock (5-digit code ending
in 0) are kept.

Outputs:
    JPX_PARSED_DIR/<source>.parquet  one cache file per source PDF (skipped on re-runs)
    DATA_DIR/jpx_prices.parquet      all sources merged: one row per (date, code)
    DATA_DIR/jpx_parse_log.csv       per source: rows read, rows kept, tokens not placed in a column

Run:  python pipeline/jpx.py [raw_dir]
"""
import io
import os
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import fitz  # PyMuPDF
import pandas as pd

from config import DATA_DIR, JPX_RAW_DIR, MEMBERSHIP_DIR

JPX_PARSED_DIR = DATA_DIR / "jpx_parsed"
WORKERS = max(1, (os.cpu_count() or 2) - 2)  # PDFs parsed in parallel

# Header label -> field. Several labels can point to one field (labels wrap over two lines).
LABELS = {
    "前場始値": "m_open", "前場高値": "m_high", "前場安値": "m_low", "前場終値": "m_close",
    "後場始値": "a_open", "後場高値": "a_high", "後場安値": "a_low", "後場終値": "a_close",
    "区分": "quote", "特別気配": "quote", "最終特別": "quote",
    "売買高": "volume", "うち立会外": "offauction_volume",
    "売買単位": "unit", "単位": "unit",
}
GLUED_LABELS = re.compile("|".join(sorted(LABELS, key=len, reverse=True)))
PRICE_FIELDS = {"m_open", "m_high", "m_low", "m_close", "a_open", "a_high", "a_low", "a_close"}
MAX_COLUMN_OFFSET = 30  # pt; values sit 4-16 pt right of their label (23 pt under a glued label)
DATES = [(re.compile(r"\d{8}"), "%Y%m%d"), (re.compile(r"\d{2}/\d{2}/\d{2}"), "%y/%m/%d")]
CODE = re.compile(r"[0-9]{3}[0-9A-Z][0-9]")          # 4-character code + 1 digit (0 = common)
NUMBER = re.compile(r"-?[\d,]+(?:\.\d+)?")
QUOTE = re.compile(r"(\D+)([\d,]+(?:\.\d+)?)")        # e.g. ｶ102: quote type + price
QUOTE_MARK = re.compile(r"[^\d\s,.\-]{1,2}")
MAX_MARK_GAP = 20  # pt between a quote mark and its number (3 pt in 2005, ~15 pt in 2017)


def parse_date(text):
    for pattern, fmt in DATES:
        if pattern.fullmatch(text):
            return datetime.strptime(text, fmt).date()
    return None


def header_columns(words):
    """Return [(right_edge, field)] from the header label words of a page."""
    first_date_y = min((w[1] for w in words if w[0] < 90 and parse_date(w[4])), default=None)
    if first_date_y is None:
        return []
    header = [w for w in words if w[1] < first_date_y - 2]
    offauction = [w for w in header if w[4] == "うち立会外"]
    columns = []
    for x0, _, x1, _, text, *_ in header:
        field = LABELS.get(text)
        # the second 売買高 sits under うち立会外 (off-auction volume), not under the main volume
        if text == "売買高" and any(o[0] <= x1 and x0 <= o[2] for o in offauction):
            field = "offauction_volume"
        if field:
            columns.append((x1, field))
        else:
            # some files glue labels into one word (前場始値前場高値..., 銘柄コード銘柄略称前場始値):
            # place each label at the point of the word's width where it ends in the text
            for m in GLUED_LABELS.finditer(text):
                columns.append((x0 + (x1 - x0) * m.end() / len(text), LABELS[m.group()]))
    return columns


def join_quote_marks(words):
    """Join a lone quote-type mark (e.g. ｶ = bid, ｳ = ask) to the number printed just after it.

    From 2004 the mark is a separate word, which would otherwise land in the close column.
    """
    words = sorted(words, key=lambda w: (round(w[1]), w[0]))
    out = []
    for w in words:
        prev = out[-1] if out else None
        if (prev and QUOTE_MARK.fullmatch(prev[4]) and abs(prev[1] - w[1]) < 2
                and 0 <= w[0] - prev[2] < MAX_MARK_GAP and NUMBER.fullmatch(w[4])):
            out[-1] = (prev[0], prev[1], w[2], w[3], prev[4] + w[4])
        else:
            out.append(tuple(w[:5]))
    return out


def parse_page(page, previous_columns=None):
    """Return (rows, unplaced, columns) for one page; rows are dicts of raw strings.

    Some files print the column header only on the first page of a day, so a page without a
    header reuses the columns of the previous page of the same file (previous_columns).
    """
    words = join_quote_marks(page.get_text("words"))
    has_rows = any(w[0] < 90 and parse_date(w[4]) for w in words)
    own = header_columns(words)
    columns = own if own and not PRICE_FIELDS - {field for _, field in own} else previous_columns
    if not has_rows:
        return [], 0, columns
    if not columns or PRICE_FIELDS - {field for _, field in columns}:
        # no header, or a price column not found: values could slide into a neighbour, so refuse
        # the page and count every word as unplaced so the log shows it
        return [], len(words), columns
    anchors = []  # (y, date, code, x1 of the code word)
    for x0, y0, x1, _, text, *_ in words:
        if x0 < 90 and parse_date(text):
            anchors.append([y0, parse_date(text), None, None])
    for x0, y0, x1, _, text, *_ in words:
        if CODE.fullmatch(text) or re.fullmatch(r"\d{9}", text):
            row = min(anchors, key=lambda a: abs(a[0] - y0), default=None)
            if row and abs(row[0] - y0) < 3 and row[2] is None and x0 > 30:
                row[2], row[3] = text, x1
    anchors = [a for a in anchors if a[2]]
    if not anchors:
        return [], 0, columns
    ys = sorted(a[0] for a in anchors)
    pitch = min((b - a for a, b in zip(ys, ys[1:]) if b - a > 2), default=10)
    first_value_x = min(x for x, _ in columns) - 40
    rows = {id(a): {"date": a[1], "code5": a[2], "name": []} for a in anchors}
    unplaced = 0
    for x0, y0, x1, _, text, *_ in words:
        row_anchor = min(anchors, key=lambda a: abs(a[0] - y0))
        if abs(row_anchor[0] - y0) > 0.6 * pitch or x1 <= row_anchor[3]:
            continue  # header/footer text, or the date/code words themselves
        row = rows[id(row_anchor)]
        if x0 < first_value_x:
            row["name"].append(text)
            continue
        if not (NUMBER.fullmatch(text) or QUOTE.fullmatch(text)):
            continue  # header/footer text that shares a line with a row
        x_col, field = min(columns, key=lambda c: abs(c[0] - x1))
        if abs(x_col - x1) > MAX_COLUMN_OFFSET or field in row:
            unplaced += 1
            continue
        row[field] = text
    out = []
    for row in rows.values():
        row["name"] = "".join(row["name"])
        out.append(row)
    return out, unplaced, columns


def to_number(text):
    if not isinstance(text, str) or not NUMBER.fullmatch(text):
        return None
    return float(text.replace(",", ""))


def clean(rows, kept):
    """Raw row dicts -> typed DataFrame of kept common stocks with daily OHLC."""
    df = pd.DataFrame(rows)
    df = df[df["code5"].str.fullmatch(CODE.pattern) & df["code5"].str.endswith("0")].copy()
    df["code"] = df["code5"].str[:4]
    df = df[df["code"].isin(kept)].copy()
    for field in ["m_open", "m_high", "m_low", "m_close", "a_open", "a_high", "a_low", "a_close",
                  "volume", "offauction_volume", "unit"]:
        df[field] = df[field].map(to_number).astype(float) if field in df else float("nan")
    # some 2002-2003 files print 0 for a session without trades; a price is never 0
    prices = ["m_open", "m_high", "m_low", "m_close", "a_open", "a_high", "a_low", "a_close"]
    df[prices] = df[prices].replace(0, float("nan"))
    quote =df["quote"].str.extract(QUOTE.pattern) if "quote" in df else pd.DataFrame(columns=[0, 1])
    df["quote_type"] = quote[0]
    df["quote_price"] = quote[1].map(to_number).astype(float)
    df["open"] = df["m_open"].fillna(df["a_open"])
    df["high"] = df[["m_high", "a_high"]].max(axis=1)
    df["low"] = df[["m_low", "a_low"]].min(axis=1)
    df["close"] = df["a_close"].fillna(df["m_close"])
    return df[["date", "code", "name", "open", "high", "low", "close", "volume",
               "offauction_volume", "unit", "quote_type", "quote_price"]]


def sources(raw_dir):
    """List (source_name, file_path, zip_member) for every PDF, loose or inside a zip."""
    found = []
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() == ".pdf":
            found.append((str(path.relative_to(raw_dir)), path, None))
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as z:
                for member in sorted(z.namelist()):
                    if member.lower().endswith(".pdf"):
                        # member names are garbled Shift-JIS; the date in them is enough
                        day = re.search(r"\d{8}", member)
                        found.append((f"{path.stem}/{day.group() if day else member}.pdf", path, member))
    return found


def parse_source(name, path, member, kept, cache):
    """Parse one PDF, write its cache file, and return its log entry."""
    if member is None:
        data = path.read_bytes()
    else:
        with zipfile.ZipFile(path) as z:
            data = z.read(member)
    rows, unplaced, columns = [], 0, None
    for page in fitz.open(stream=io.BytesIO(data), filetype="pdf"):
        page_rows, page_unplaced, columns = parse_page(page, columns)
        rows += page_rows
        unplaced += page_unplaced
    df = clean(rows, kept) if rows else pd.DataFrame()
    df.assign(source=name).to_parquet(cache, index=False)
    return {"source": name, "rows_read": len(rows), "rows_kept": len(df), "unplaced_tokens": unplaced}


def main(raw_dir=JPX_RAW_DIR):
    kept = set(pd.read_csv(MEMBERSHIP_DIR / "kept_codes.csv", dtype=str)["code"])
    JPX_PARSED_DIR.mkdir(parents=True, exist_ok=True)
    todo = []
    for name, path, member in sources(raw_dir):
        cache = JPX_PARSED_DIR / (re.sub(r"[^\w.-]", "_", name) + ".parquet")
        if not cache.exists():
            todo.append((name, path, member, kept, cache))
    print(f"{len(todo)} PDFs to parse with {WORKERS} workers")
    log = []
    with ProcessPoolExecutor(WORKERS) as pool:
        for n, done in enumerate(as_completed([pool.submit(parse_source, *t) for t in todo]), 1):
            entry = done.result()
            log.append(entry)
            if entry["unplaced_tokens"] or n % 200 == 0:
                print(f"{n}/{len(todo)} {entry['source']}: {entry['rows_read']} rows, "
                      f"{entry['rows_kept']} kept, {entry['unplaced_tokens']} unplaced tokens")
    if log:
        log_path = DATA_DIR / "jpx_parse_log.csv"
        old = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
        pd.concat([old, pd.DataFrame(log)]).drop_duplicates("source", keep="last").to_csv(log_path, index=False)
    parts = [pd.read_parquet(p) for p in sorted(JPX_PARSED_DIR.glob("*.parquet"))]
    merged = pd.concat([p for p in parts if len(p)], ignore_index=True)
    dup = merged.duplicated(["date", "code"], keep="first")
    if dup.any():
        print(f"{dup.sum()} duplicate (date, code) rows dropped (same day in two sources)")
    merged = merged[~dup].sort_values(["code", "date"]).reset_index(drop=True)
    merged.to_parquet(DATA_DIR / "jpx_prices.parquet", index=False)
    print(f"Merged: {len(merged)} rows, {merged['code'].nunique()} codes, "
          f"{merged['date'].min()} to {merged['date'].max()}")


if __name__ == "__main__":
    from pathlib import Path
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else JPX_RAW_DIR)
