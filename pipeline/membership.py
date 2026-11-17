"""Rebuild Nikkei 225 membership at every date.

Nikkei publishes a PDF listing every constituent change (date, deleted code,
added code). Starting from today's member list and undoing the changes backwards gives the members
on any past date (D5: each date's actual members, including companies later removed).

Outputs (in MEMBERSHIP_DIR):
    <index>_changes.csv  one row per change: date, deleted, added ("--" = none), and their English names
    <index>_spells.csv   one row per membership spell: code, start, end (member on start <= d < end;
                         empty start = member before the history begins, empty end = still a member)
    kept_codes.csv       codes that were members at any time on or after KEPT_SINCE, with an English
                         name (today's member list, else the latest change naming the code)

Run:  python pipeline/membership.py [--refresh]
"""
import re
import sys
from collections import defaultdict
from datetime import date, datetime
from html import unescape

import fitz  # PyMuPDF
import pandas as pd
import requests

from config import INDICES, KEPT_SINCE, MEMBERSHIP_DIR, REFERENCE_DIR, USER_AGENT

# x-ranges in PDF points of the two code columns:
DELETED_CODE_X = (290, 320)
ADDED_CODE_X = (515, 545)
DATE_MAX_X = 80        # dates sit in the left-most column
HALF_ROW = 7.3         # half the row pitch (rows are ~14.6 pt apart)
MAX_FIT_ERROR = 3.0    # pt; a block whose centre misses its date by more is reported
CODE_RE = re.compile(r"[0-9]{3}[0-9A-Z]|--")  # "--" = no company
DELETED_NAME_X = (100, 290)  # company-name columns
ADDED_NAME_X = (330, 515)


def download(url, path, refresh=False):
    if path.exists() and not refresh:
        return path
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return path


def current_members(html_path):
    """Today's members as {code: English name}."""
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    rows = re.findall(r"<td[^>]*>\s*([0-9]{3}[0-9A-Z])\s*</td>\s*<td[^>]*>(.*?)</td>", html, flags=re.S)
    return {code: unescape(re.sub(r"<[^>]+>", "", name)).strip() for code, name in rows}


def _page_rows(page, date_format):
    """Dates (y, date), row y positions, the deleted/added code on each row, and name words."""
    words = page.get_text("words")
    dates, deleted, added, names = [], {}, {}, []
    for x0, y0, _, _, text, *_ in words:
        if x0 < DATE_MAX_X:
            try:
                dates.append((y0, datetime.strptime(text, date_format).date()))
            except ValueError:
                pass
        elif CODE_RE.fullmatch(text):
            y = round(y0, 1)
            if DELETED_CODE_X[0] <= x0 <= DELETED_CODE_X[1]:
                deleted[y] = text
            elif ADDED_CODE_X[0] <= x0 <= ADDED_CODE_X[1]:
                added[y] = text
        elif DELETED_NAME_X[0] <= x0 < DELETED_NAME_X[1]:
            names.append(("deleted", y0, x0, text))
        elif ADDED_NAME_X[0] <= x0 < ADDED_NAME_X[1]:
            names.append(("added", y0, x0, text))
    ys = sorted(set(deleted) | set(added))
    return sorted(dates), ys, deleted, added, names


def parse_change_history(pdf_path, date_format):
    """Return (changes DataFrame, list of layout problems).

    Each date is printed at the vertical centre of its block of rows, and rows can be taller than
    usual when a company name wraps. So rows are measured by their extent (midpoints between
    neighbouring rows) and each block is cut where its centre lands on its date. A block running
    past the bottom of a page continues, without a date, at the top of the next page; that next
    page is then cut from the bottom up and its leftover top rows go to the carried date.
    """
    changes, problems = [], []
    carried = None
    for pno, page in enumerate(fitz.open(pdf_path)):
        dates, ys, deleted, added, names = _page_rows(page, date_format)
        n = len(ys)
        if not n:
            continue
        # row k spans bounds[k] .. bounds[k + 1]
        bounds = [ys[0] - HALF_ROW] + [(a + b) / 2 for a, b in zip(ys, ys[1:])] + [ys[-1] + HALF_ROW]
        centre = lambda i, j: (bounds[i] + bounds[j + 1]) / 2
        owner = [None] * n
        if carried is None:
            i = 0
            for k, (y_date, d) in enumerate(dates):
                if k == len(dates) - 1:
                    j = n - 1
                    if 2 * y_date - bounds[i] > bounds[n] + HALF_ROW:
                        carried = d  # centring would need rows beyond this page
                    elif abs(centre(i, j) - y_date) > MAX_FIT_ERROR:
                        problems.append((pno, d, round(abs(centre(i, j) - y_date), 1)))
                else:
                    err, j = min((abs(centre(i, j) - y_date), j) for j in range(i, n))
                    if err > MAX_FIT_ERROR:
                        problems.append((pno, d, round(err, 1)))
                owner[i:j + 1] = [d] * (j + 1 - i)
                i = j + 1
        else:
            j = n - 1
            for y_date, d in reversed(dates):
                err, i = min((abs(centre(i, j) - y_date), i) for i in range(0, j + 1))
                if err > MAX_FIT_ERROR:
                    problems.append((pno, d, round(err, 1)))
                owner[i:j + 1] = [d] * (j + 1 - i)
                j = i - 1
            owner[:j + 1] = [carried] * (j + 1)
            carried = None
        for k, y in enumerate(ys):
            # a name (possibly wrapped over two lines) belongs to the row whose extent holds it
            name = lambda side: " ".join(t for s, yn, _, t in sorted(names, key=lambda n: (round(n[1]), n[2]))
                                         if s == side and bounds[k] <= yn < bounds[k + 1] and t != "--")
            changes.append((owner[k], deleted.get(y, "??"), added.get(y, "??"), name("deleted"), name("added")))
    df = pd.DataFrame(changes, columns=["date", "deleted", "added", "deleted_name", "added_name"])
    unreadable = df[(df["deleted"] == "??") | (df["added"] == "??") | df["date"].isna()]
    problems += [("unreadable row", r.date, r.deleted, r.added) for r in unreadable.itertuples()]
    return df.sort_values("date", kind="stable").reset_index(drop=True), problems


def rebuild(changes, current, size):
    """Undo the changes backwards from today's members; return (spells, problems)."""
    by_date = defaultdict(list)
    for r in changes.itertuples():
        by_date[r.date].append((r.deleted, r.added))
    members, problems = set(current), []
    if len(members) != size:
        problems.append(("today", f"{len(members)} members listed, expected {size}"))
    for d in sorted(by_date, reverse=True):
        for dele, add in by_date[d]:
            if add != "--":
                if add not in members:
                    problems.append((d, f"added {add} but it is not a member afterwards"))
                members.discard(add)
        for dele, add in by_date[d]:
            if dele != "--":
                if dele in members:
                    problems.append((d, f"deleted {dele} but it is still a member afterwards"))
                members.add(dele)
    # Walk forward from the oldest state to cut membership spells.
    open_spells = {code: None for code in members}
    spells = []
    for d in sorted(by_date):
        for dele, add in by_date[d]:
            if dele != "--" and dele in open_spells:
                spells.append((dele, open_spells.pop(dele), d))
        for dele, add in by_date[d]:
            if add != "--":
                open_spells[add] = d
    spells += [(code, start, None) for code, start in open_spells.items()]
    df = pd.DataFrame(spells, columns=["code", "start", "end"])
    return df.sort_values(["code", "start"], na_position="first").reset_index(drop=True), problems


def english_names(changes, current):
    """{code: English name}: today's list for current members, else the latest change naming the code."""
    names = {}
    for r in changes.sort_values("date").itertuples():
        if r.deleted != "--" and r.deleted_name:
            names[r.deleted] = r.deleted_name.upper()
        if r.added != "--" and r.added_name:
            names[r.added] = r.added_name.upper()
    names.update(current)
    return names


def main(refresh=False):
    MEMBERSHIP_DIR.mkdir(parents=True, exist_ok=True)
    kept, names = [], {}
    for idx, spec in INDICES.items():
        pdf = download(spec["history_url"], REFERENCE_DIR / f"{idx}_change_history.pdf", refresh)
        html = download(spec["current_url"], REFERENCE_DIR / f"{idx}_current_members.html", refresh)
        changes, layout_problems = parse_change_history(pdf, spec["date_format"])
        current = current_members(html)
        spells, problems = rebuild(changes, current, spec["size"])
        names.update(english_names(changes, current))
        changes.to_csv(MEMBERSHIP_DIR / f"{idx}_changes.csv", index=False)
        spells.to_csv(MEMBERSHIP_DIR / f"{idx}_spells.csv", index=False)
        print(f"{spec['name']}: {len(changes)} changes on {changes['date'].nunique()} dates "
              f"({changes['date'].min()} to {changes['date'].max()}), {len(spells)} spells")
        for p in layout_problems + problems:
            print("  PROBLEM:", *p)
        in_window = spells[spells["end"].isna() | (spells["end"] > KEPT_SINCE)]
        kept.append(in_window.assign(index=idx))
    kept = pd.concat(kept)
    kept["start"] = kept["start"].fillna(date(1900, 1, 1))
    summary = (kept.groupby("code")
                   .agg(indices=("index", lambda s: "+".join(sorted(set(s)))),
                        first_start=("start", "min"),
                        spells=("start", "size"))
                   .reset_index())
    summary.loc[summary["first_start"] == date(1900, 1, 1), "first_start"] = None
    summary.insert(1, "name", summary["code"].map(names))
    summary.to_csv(MEMBERSHIP_DIR / "kept_codes.csv", index=False)
    print(f"Kept codes (member of {' or '.join(INDICES)} on or after {KEPT_SINCE}): {len(summary)}, "
          f"{summary['name'].isna().sum()} without an English name")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
