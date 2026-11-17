"""Build the clean daily price file for the kept Nikkei 225 companies.

Input: DATA_DIR/jpx_prices.parquet (from jpx.py) and the membership files (from membership.py).

Steps:
    1. Code reuse: a code can belong to different companies over time (e.g. 8729). JPX lists every
       listed stock every trading day, so a gap of more than MAX_LISTING_GAP trading days starts a new
       listing. Only listings that overlap a Nikkei 225 membership spell are kept; the whole listing is
       kept, including the years before joining, which D8's training window uses.
    2. Checks: duplicates, OHLC consistency, and days a listed stock is missing.
    3. Adjusted closes and daily log returns (splits and dividends; see adjust.py).
    4. English names replace the Japanese short names, and in_index flags index membership per day.

Outputs (in DATA_DIR):
    prices_daily.parquet / prices_daily.csv   one row per (date, code)
    corporate_actions.csv                     every split and dividend applied, with its source
    prices_daily_report.txt                   what the checks found

Run:  python pipeline/build_prices.py
"""
import pandas as pd

from adjust import add_adjusted
from config import DATA_DIR, MEMBERSHIP_DIR

MAX_LISTING_GAP = 20  # trading days a listed stock can be absent before it counts as a new listing
COLUMNS = ["date", "code", "name", "in_index", "open", "high", "low", "close", "adj_close", "log_return",
           "volume", "offauction_volume", "unit", "quote_type", "quote_price",
           "split", "dividend", "dividend_source", "dividends_known"]


def listing_spells(df, calendar):
    """Label each row with a listing number per code (a new number after a long absence)."""
    pos = df["date"].map(calendar)
    gap = pos.groupby(df["code"]).diff()
    return (gap.isna() | (gap > MAX_LISTING_GAP)).groupby(df["code"]).cumsum()


def membership_flag(df, spells):
    """True where the code was a Nikkei 225 member on that date (start <= date < end)."""
    flag = pd.Series(False, index=df.index)
    for s in spells.itertuples():
        rows = df["code"] == s.code
        if pd.notna(s.start):
            rows &= df["date"] >= s.start
        if pd.notna(s.end):
            rows &= df["date"] < s.end
        flag |= rows
    return flag


def main():
    report = []
    df = pd.read_parquet(DATA_DIR / "jpx_prices.parquet")
    df["date"] = pd.to_datetime(df["date"])
    spells = pd.read_csv(MEMBERSHIP_DIR / "nk225_spells.csv", dtype={"code": str},
                         parse_dates=["start", "end"])
    names = pd.read_csv(MEMBERSHIP_DIR / "kept_codes.csv", dtype={"code": str}).set_index("code")["name"]
    report.append(f"Parsed rows: {len(df)}, codes: {df['code'].nunique()}, "
                  f"{df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")

    dup = df.duplicated(["date", "code"])
    report.append(f"Duplicate (date, code) rows dropped: {dup.sum()}")
    df = df[~dup].sort_values(["code", "date"]).reset_index(drop=True)

    # 1. Keep only the listings that overlap a membership spell.
    calendar = {d: i for i, d in enumerate(sorted(df["date"].unique()))}
    df["listing"] = listing_spells(df, calendar)
    df["in_index"] = membership_flag(df, spells)
    member_listings = df.loc[df["in_index"], ["code", "listing"]].drop_duplicates()
    before = len(df)
    dropped = df.merge(member_listings, how="left", indicator=True)["_merge"].eq("left_only").values
    for (code, listing), g in df[dropped].groupby(["code", "listing"]):
        report.append(f"  dropped listing of {code} ({g['name'].iloc[0]}): "
                      f"{g['date'].min():%Y-%m-%d} to {g['date'].max():%Y-%m-%d}, {len(g)} rows "
                      f"(not a member during these dates: code reused, or joins after the data ends)")
    df = df[~dropped]
    report.append(f"Rows from listings that are not index members in the data period dropped: {before - len(df)}")
    never_found = sorted(set(names.index) - set(df["code"]))
    report.append(f"Kept codes with no rows (member only before 2000, or only after the data ends): "
                  f"{len(never_found)} {never_found}")

    # 2. Checks.
    traded = df["close"].notna()
    bad_ohlc = traded & ((df["low"] > df[["open", "close"]].min(axis=1)) |
                         (df["high"] < df[["open", "close"]].max(axis=1)))
    report.append(f"Traded rows: {traded.sum()}, no-trade rows: {(~traded).sum()} "
                  f"(of which with a special quote: {(~traded & df['quote_price'].notna()).sum()})")
    report.append(f"OHLC-inconsistent rows: {bad_ohlc.sum()}")
    for r in df[bad_ohlc].head(20).itertuples():
        report.append(f"  {r.date:%Y-%m-%d} {r.code} O={r.open} H={r.high} L={r.low} C={r.close}")
    # a listed stock missing from a day's report (between its first and last day in its listing)
    expected = df.groupby(["code", "listing"])["date"].agg(["min", "max", "size"])
    expected["days"] = expected.apply(lambda r: calendar[r["max"]] - calendar[r["min"]] + 1, axis=1)
    missing = (expected["days"] - expected["size"]).sum()
    report.append(f"Days a listed stock is absent from the report: {missing}")
    rows_per_day = df.groupby("date").size()
    report.append(f"Stocks per day: min {rows_per_day.min()}, median {int(rows_per_day.median())}, "
                  f"max {rows_per_day.max()}")
    members_per_day = df[df["in_index"]].groupby("date").size()
    low = members_per_day[members_per_day < 220]
    report.append(f"Days with fewer than 220 index members found: {len(low)}")
    for d, n in low.head(20).items():
        report.append(f"  {d:%Y-%m-%d}: {n}")

    # 3. Adjusted closes and log returns.
    df, actions, adjust_report = add_adjusted(df)
    report += adjust_report
    actions.insert(1, "name", actions["code"].map(names))
    actions.assign(date=actions["date"].dt.date).to_csv(DATA_DIR / "corporate_actions.csv", index=False)

    # 4. English names; write.
    df["name"] = df["code"].map(names)
    df["date"] = df["date"].dt.date
    out = df[COLUMNS].sort_values(["date", "code"]).reset_index(drop=True)
    out.to_parquet(DATA_DIR / "prices_daily.parquet", index=False)
    out.to_csv(DATA_DIR / "prices_daily.csv", index=False, encoding="utf-8")
    report.append(f"Written: {len(out)} rows, {out['code'].nunique()} codes, "
                  f"{out['date'].min()} to {out['date'].max()}")
    (DATA_DIR / "prices_daily_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
