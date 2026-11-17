"""Parse the JPX monthly quotation tables (月間相場表, 株式相場表詳細版) into monthly volume, trading
value and margin/loan status per stock. They fill the volume gap from 2020-01, where the
daily-report files print prices only.

Input: JPX_MONTHLY_DIR/st_YYYYMM.pdf, the detailed version (the short st_YYYYMM-2.pdf has no
volume), downloaded by hand from jpx.co.jp/markets/statistics-equities/price/index.html
(year dropdown バックナンバー; only 2016 onward is online).

Layouts (the columns are found on each page and in each file, so shifts are fine):
    to 2023-02    volume and of-which-ToSTNeT volume; value and of-which-ToSTNeT value; days traded
    from 2023-03  volume = auction (売買立会) + ToSTNeT + off-auction distributions (立会外分売, not
                  printed separately); the same for value
    from 2025-11  the same as 2023-03 on a page 3.57 times larger; coordinates are scaled to A4
Numbers are right-aligned, so each file's numeric columns are the clusters of right edges.
A stock with a 新株落ち event in the month (split, consolidation, spin-off) has two rows:
before and after the ex-date. Status columns (section, 区分 flag, 信用・貸借) are as of month-end.
adv_value divides by the month's trading days, so it understates a stock listed for part of the month
(among index members only the three delisting months in 2020-2022).

Outputs (in DATA_DIR):
    jpx_monthly_rows.parquet            one row per printed row of a kept code, every column
    liquidity_monthly.parquet / .csv    one row per (month, code): value and days summed over the
                                        rows, average daily value, month-end margin/loan status
    jpx_monthly_report.txt              parse counts and the checks against prices_daily

Run:  python pipeline/jpx_monthly.py
"""
import os
import re
from concurrent.futures import ProcessPoolExecutor

import fitz  # PyMuPDF
import pandas as pd

from config import DATA_DIR, JPX_MONTHLY_DIR, MEMBERSHIP_DIR

WORKERS = max(1, (os.cpu_count() or 2) - 2)
A4_WIDTH = 841.92          # pt; pages of other sizes are scaled to this width
ROW_TOLERANCE = 1.0        # pt; words of one printed row share their top edge
COLUMN_GAP = 2.5           # pt; right edges further apart than this are different columns
LABEL_SLACK = 3.0          # pt; a value can start slightly left of its column label
MONTH = re.compile(r"\d{4}/\d{2}")
NUMBER = re.compile(r"-?[\d,]+(?:\.\d+)?")
LABELS = ["銘柄コード", "銘柄属性", "市場区分", "区分", "信用・貸借", "売買単位"]

# Numeric columns from 売買単位 rightwards, by layout (number of right-edge clusters in the file).
NUMERIC = {
    15: ["unit", "open", "open_day", "high", "high_day", "low", "low_day", "close", "close_day",
         "avg_close", "volume", "volume_tostnet", "value", "value_tostnet", "days_traded"],
    17: ["unit", "open", "open_day", "high", "high_day", "low", "low_day", "close", "close_day",
         "avg_close", "volume", "volume_auction", "volume_tostnet", "value", "value_auction",
         "value_tostnet", "days_traded"],
}


def page_labels(words):
    """Left edges of the header labels, or None if the page has no table header."""
    unit = [w for w in words if w[4] == "売買単位"]
    if not unit:
        return None
    header = [w for w in words if w[1] <= unit[0][3]]
    found = {}
    for label in LABELS:
        hits = sorted(w[0] for w in header if w[4] == label)
        if not hits:
            return None
        found[label] = hits[0]
    return found, unit[0][3]


def page_rows(page, labels):
    """Return the raw rows of one page and the labels in force (reused on a page without header)."""
    scale = A4_WIDTH / page.rect.width
    # some files are portrait pages flagged to display rotated 90 degrees (2020-02, 2024-08, ...):
    # words come in the unrotated frame, so map them to the landscape page as displayed
    words = []
    for w in page.get_text("words"):
        r = fitz.Rect(w[:4]) * page.rotation_matrix if page.rotation else fitz.Rect(w[:4])
        words.append((r.x0 * scale, r.y0 * scale, r.x1 * scale, r.y1 * scale, w[4]))
    found = page_labels(words)
    if found:
        labels = found
    if labels is None:
        return [], labels
    lab, header_bottom = labels
    lines = {}
    for w in sorted((w for w in words if w[1] > header_bottom), key=lambda w: (w[1], w[0])):
        key = next((k for k in lines if abs(k - w[1]) <= ROW_TOLERANCE), w[1])
        lines.setdefault(key, []).append(w)
    zones = [("attribute", lab["銘柄属性"], lab["市場区分"]), ("section", lab["市場区分"], lab["区分"]),
             ("flag", lab["区分"], lab["信用・貸借"]), ("margin_loan", lab["信用・貸借"], lab["売買単位"])]
    rows = []
    for line in lines.values():
        line.sort(key=lambda w: w[0])
        if not MONTH.fullmatch(line[0][4]) or line[0][0] > lab["銘柄コード"]:
            continue
        row = {"month": line[0][4], "code": line[1][4] if len(line) > 1 else None,
               "name_jp": line[2][4] if len(line) > 2 else None, "numbers": []}
        for field, left, right in zones:
            # the attribute (e.g. 新株落ち ex-subscription right) starts up to 6 pt left of its label
            slack = 2 * LABEL_SLACK if field == "attribute" else LABEL_SLACK
            texts = [w[4] for w in line if left - slack <= w[0] < right - LABEL_SLACK]
            row[field] = " ".join(texts) if texts else None
        # the 区分 flag (e.g. 整, to be delisted) can start inside the section zone
        if row["section"] and " " in row["section"]:
            row["section"], extra = row["section"].split(" ", 1)
            row["flag"] = " ".join(t for t in (extra, row["flag"]) if t)
        row["numbers"] = [(w[2], w[4]) for w in line
                          if w[0] >= lab["売買単位"] - LABEL_SLACK and NUMBER.fullmatch(w[4])]
        rows.append(row)
    return rows, labels


def column_edges(rows):
    """Cluster the right edges of the numeric tokens into the file's columns."""
    edges = sorted(x for r in rows for x, _ in r["numbers"])
    clusters, current = [], [edges[0]]
    for x in edges[1:]:
        if x - current[-1] > COLUMN_GAP:
            clusters.append(current)
            current = []
        current.append(x)
    clusters.append(current)
    # stray tokens (a digit that wrapped) form tiny clusters; real columns have many values
    return [sum(c) / len(c) for c in clusters if len(c) >= 0.05 * len(rows)]


def to_number(text):
    return float(text.replace(",", "")) if text is not None else None


def parse_file(path):
    """Parse one monthly PDF into a list of row dicts plus a log entry."""
    rows, labels = [], None
    for page in fitz.open(path):
        page_rows_, labels = page_rows(page, labels)
        rows += page_rows_
    if not rows:
        raise ValueError(f"{path.name}: no table rows found")
    edges = column_edges(rows)
    if len(edges) not in NUMERIC:
        raise ValueError(f"{path.name}: {len(edges)} numeric columns found at {[round(e) for e in edges]}")
    fields = NUMERIC[len(edges)]
    unplaced = 0
    for row in rows:
        for x, text in row.pop("numbers"):
            i = min(range(len(edges)), key=lambda k: abs(edges[k] - x))
            if abs(edges[i] - x) > COLUMN_GAP or fields[i] in row:
                unplaced += 1
                continue
            row[fields[i]] = text
        for f in fields:
            row[f] = to_number(row.get(f))
        row["source"] = path.name
    return rows, {"source": path.name, "rows": len(rows), "columns": len(edges), "unplaced_tokens": unplaced}


def clean(rows, kept):
    df = pd.DataFrame(rows)
    df = df[df["code"].isin(kept)].copy()
    df["month"] = pd.to_datetime(df["month"], format="%Y/%m").dt.to_period("M")
    df["part"] = df.groupby(["month", "code"]).cumcount() + 1
    # before 2023-03 only the ToSTNeT part is printed; the rest is auction plus off-auction distributions
    old = df["volume_auction"].isna() & df["volume"].notna()
    df.loc[old, "volume_auction"] = df.loc[old, "volume"] - df.loc[old, "volume_tostnet"].fillna(0)
    df.loc[old, "value_auction"] = df.loc[old, "value"] - df.loc[old, "value_tostnet"].fillna(0)
    df["auction_derived"] = old
    return df


def keep_listed(df, prices, spells):
    """Drop rows of codes reused by another company: keep (month, code) pairs found in prices_daily,
    and after its last month the codes still listed at its end or index members per the spells."""
    present = prices[["month", "code"]].drop_duplicates()
    last_month = prices["month"].max()
    listed_at_end = set(prices.loc[prices["date"] == prices["date"].max(), "code"])
    inside = df.merge(present.assign(_p=True), how="left", on=["month", "code"])["_p"].eq(True).values
    after = df["month"] > last_month
    member_after = df.apply(lambda r: r["month"] > last_month and is_member(spells, r["code"], r["month"]), axis=1)
    keep = (~after & inside) | (after & (df["code"].isin(listed_at_end) | member_after))
    return df[keep.values].copy()


def is_member(spells, code, month):
    start, end = month.start_time, month.end_time
    s = spells[spells["code"] == code]
    return bool(((s["start"].isna() | (s["start"] <= end)) & (s["end"].isna() | (s["end"] > start))).any())


def liquidity_table(df, prices, spells, names):
    """One row per (month, code); a month with two rows (before and after an ex-date) is summed."""
    split_factor = (prices[prices["split"].notna() & (prices["split"] != 1)]
                    .groupby(["month", "code"])["split"].prod())
    g = df.sort_values("part").groupby(["month", "code"])
    out = g.agg(rows=("part", "size"), value=("value", "sum"), value_auction=("value_auction", "sum"),
                value_tostnet=("value_tostnet", "sum"), days_traded=("days_traded", "sum"),
                margin_loan=("margin_loan", "last"), section=("section", "last"), flag=("flag", "last"),
                attribute=("attribute", "first"), unit=("unit", "last")).reset_index()
    # volume in shares as of month-end: the rows before a split are converted at the split ratio
    vol = df.copy()
    factor = vol.set_index(["month", "code"]).index.map(split_factor.to_dict()).astype(float)
    two = vol.groupby(["month", "code"])["part"].transform("size") > 1
    last = vol["part"] == vol.groupby(["month", "code"])["part"].transform("max")
    vol["mult"] = 1.0
    vol.loc[two & ~last, "mult"] = factor[(two & ~last).values]
    # no share change in the daily data (a spin-off): the rows add up as they are; after the daily
    # data ends the ratio is unknown
    known = vol["month"] <= prices["month"].max()
    vol.loc[two & ~last & known & vol["mult"].isna(), "mult"] = 1.0
    for col in ["volume", "volume_auction"]:
        vol[col] = vol[col] * vol["mult"]
    shares = vol.groupby(["month", "code"])[["volume", "volume_auction"]].sum(min_count=1)
    unknown = vol[vol["mult"].isna()].groupby(["month", "code"]).size()
    shares.loc[unknown.index, :] = float("nan")
    out = out.merge(shares.reset_index(), on=["month", "code"], how="left")
    # trading days: the most days any stock traded in the month (liquid names trade every day)
    out["trading_days"] = out.groupby("month")["days_traded"].transform("max")
    out["adv_value"] = out["value"] / out["trading_days"]
    out["adv_value_auction"] = out["value_auction"] / out["trading_days"]
    member_days = prices[prices["in_index"]].groupby(["month", "code"]).size().rename("member_days")
    out = out.merge(member_days.reset_index(), on=["month", "code"], how="left")
    last_month = prices["month"].max()
    after = out["month"] > last_month
    out["in_index"] = out["member_days"].fillna(0) > 0
    out.loc[after, "in_index"] = [is_member(spells, c, m) for c, m in out.loc[after, ["code", "month"]].values]
    out.loc[after, "member_days"] = float("nan")
    out.insert(2, "name", out["code"].map(names))
    cols = ["month", "code", "name", "in_index", "member_days", "margin_loan", "section", "flag", "unit", "rows",
            "attribute", "trading_days", "days_traded", "volume", "volume_auction", "value", "value_auction",
            "value_tostnet", "adv_value", "adv_value_auction"]
    return out[cols].sort_values(["month", "code"]).reset_index(drop=True)


def checks(df, liq, prices):
    """Compare with the daily prices (to 2025-09): days traded, month-end close, high and low."""
    report = []
    traded = prices[prices["close"].notna()].sort_values("date")
    daily = traded.groupby(["month", "code"]).agg(d_days=("close", "size"), d_close=("close", "last"),
                                                  d_high=("high", "max"), d_low=("low", "min")).reset_index()
    calendar = prices.groupby("month")["date"].nunique().rename("calendar_days").reset_index()
    m = liq.merge(daily, on=["month", "code"], how="inner")
    report.append(f"Checks against prices_daily on {len(m)} (month, code) pairs with daily trades:")
    report.append(f"  days traded equal: {(m['days_traded'] == m['d_days']).mean():.4%}")
    for r in m[m["days_traded"] != m["d_days"]].head(10).itertuples():
        report.append(f"    {r.month} {r.code} {r.name}: monthly {r.days_traded:.0f}, daily {r.d_days}")
    last_rows = df.sort_values("part").groupby(["month", "code"]).last().reset_index()
    c = last_rows.merge(daily, on=["month", "code"], how="inner")
    report.append(f"  month-end close equal: {((c['close'] - c['d_close']).abs() < 0.01).mean():.4%}")
    for r in c[(c["close"] - c["d_close"]).abs() >= 0.01].head(10).itertuples():
        report.append(f"    {r.month} {r.code}: monthly {r.close}, daily {r.d_close}")
    one = c[c["part"] == 1]
    hl = ((one["high"] - one["d_high"]).abs() < 0.01) & ((one["low"] - one["d_low"]).abs() < 0.01)
    report.append(f"  high and low equal (one-row months): {hl.mean():.4%}")
    for r in one[~hl].head(10).itertuples():
        report.append(f"    {r.month} {r.code}: high {r.high}/{r.d_high}, low {r.low}/{r.d_low}")
    td = liq.groupby("month")["trading_days"].first().reset_index().merge(calendar, on="month")
    diff = td[td["trading_days"] != td["calendar_days"]]
    report.append(f"  months whose trading days differ from the daily calendar: {len(diff)}")
    for r in diff.itertuples():
        report.append(f"    {r.month}: monthly max days traded {r.trading_days:.0f}, daily calendar {r.calendar_days}")
    return report


def main():
    files = sorted(JPX_MONTHLY_DIR.glob("st_??????.pdf"))
    kept = set(pd.read_csv(MEMBERSHIP_DIR / "kept_codes.csv", dtype=str)["code"])
    names = pd.read_csv(MEMBERSHIP_DIR / "kept_codes.csv", dtype={"code": str}).set_index("code")["name"]
    spells = pd.read_csv(MEMBERSHIP_DIR / "nk225_spells.csv", dtype={"code": str}, parse_dates=["start", "end"])
    prices = pd.read_parquet(DATA_DIR / "prices_daily.parquet",
                             columns=["date", "code", "in_index", "close", "high", "low", "split"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices["month"] = prices["date"].dt.to_period("M")

    rows, log = [], []
    with ProcessPoolExecutor(WORKERS) as pool:
        for file_rows, entry in pool.map(parse_file, files):
            rows += file_rows
            log.append(entry)
    report = [f"Files: {len(files)}, {files[0].name} to {files[-1].name}"]
    layouts = pd.DataFrame(log).groupby("columns")["source"].agg(["size", "min", "max"])
    for n, r in layouts.iterrows():
        report.append(f"  layout with {n} numeric columns: {r['size']} files, {r['min']} to {r['max']}")
    report.append(f"Rows read: {len(rows)}; numeric tokens not placed in a column: "
                  f"{sum(e['unplaced_tokens'] for e in log)}")
    for e in log:
        if e["unplaced_tokens"]:
            report.append(f"  {e['source']}: {e['unplaced_tokens']}")

    df = clean(rows, kept)
    wrong_month = df["month"].astype(str).str.replace("-", "") != df["source"].str[3:9]
    report.append(f"Rows of kept codes: {len(df)}; rows whose month differs from the file name: {wrong_month.sum()}")
    before = len(df)
    df = keep_listed(df, prices, spells)
    report.append(f"Rows dropped as another company under a reused code: {before - len(df)}")
    multi = df[df["part"] > 1][["month", "code", "attribute"]]
    report.append(f"(month, code) pairs printed on two or more rows: {len(multi)} "
                  f"(attributes: {multi['attribute'].value_counts().to_dict()})")
    df.to_parquet(DATA_DIR / "jpx_monthly_rows.parquet", index=False)

    liq = liquidity_table(df, prices, spells, names)
    report += checks(df, liq, prices)

    members = liq[liq["in_index"]]
    report.append(f"Index members per month: min {members.groupby('month').size().min()}, "
                  f"max {members.groupby('month').size().max()}")
    report.append(f"Margin/loan status of index-member months: {members['margin_loan'].value_counts(dropna=False).to_dict()}")
    for r in members[members["margin_loan"] != "貸借"].itertuples():
        report.append(f"  not 貸借: {r.month} {r.code} {r.name}: {r.margin_loan}, flag {r.flag}, "
                      f"traded {r.days_traded:.0f} of {r.trading_days:.0f} days")
    report.append(f"Month-end volume unknown (two-row month without a split ratio): {liq['volume'].isna().sum()}")
    adv = members.groupby(members["month"].dt.year)["adv_value"].describe(percentiles=[0.05, 0.5])
    report.append("Index members' average daily trading value, ¥bn (all trades):")
    for y, r in adv.iterrows():
        report.append(f"  {y}: min {r['min'] / 1e9:.2f}, 5th pct {r['5%'] / 1e9:.2f}, median {r['50%'] / 1e9:.2f}")

    out = liq.assign(month=liq["month"].astype(str))
    out.to_parquet(DATA_DIR / "liquidity_monthly.parquet", index=False)
    out.to_csv(DATA_DIR / "liquidity_monthly.csv", index=False, encoding="utf-8")
    report.append(f"Written: liquidity_monthly ({len(out)} rows, {out['code'].nunique()} codes, "
                  f"{out['month'].min()} to {out['month'].max()}) and jpx_monthly_rows ({len(df)} rows)")
    (DATA_DIR / "jpx_monthly_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
