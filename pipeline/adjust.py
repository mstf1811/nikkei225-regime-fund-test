"""Adjusted closes and daily log returns (D9 step 4): splits and dividends applied to the JPX raw closes.

Splits
    Where Yahoo has the stock, JPX raw close / Yahoo close is a step function: Yahoo's closes are
    adjusted for every later split, so the ratio jumps on each split's ex-date. This also finds the
    splits missing from Yahoo's split list (e.g. SoftBank 2000). A step counts only if its ratio is
    a clean split ratio and the JPX raw price moves by that ratio on that day, because Yahoo's close
    series has a few bogus steps (e.g. Keyence 2005-09-29, Tokio Marine 2004-2005). When Yahoo has
    smoothed over an event so the step is not a split ratio, the ratio in Yahoo's split list is used
    if the JPX price confirms it (Haseko 2002-08-02: a 5:1 consolidation on which holders lost 26%).
    Elsewhere (names Yahoo lacks, days before Yahoo's history starts, Yahoo series too noisy to use),
    splits are detected from the JPX prices alone: a move matching a split ratio of 2:1 or more; or
    of 1.5:1 or more on one of the 2nd-4th last trading days of a month (the ex-date of a month-end
    record date, where 89% of the confirmed splits fall); or of 1.5:1 or more on a day the trading
    unit changes the same way. Tested on the days Yahoo covers, these rules find 298 of the 314
    confirmed splits, and fire on no day without an event (twice they get the ratio wrong, e.g. a
    split on the same day as a large dividend). Smaller splits (1.1, 1.2) can't be told apart from
    ordinary moves; the report lists the large moves left unexplained.
Dividends
    Yahoo only. Yahoo's amounts are adjusted for later splits, so they are converted back to yen per
    share as traded on the ex-date with the confirmed splits. A dividend above CHECK_YIELD of the
    price must show as a drop in the JPX price on its ex-date, which catches Yahoo errors such as
    Nippon Steel 2015-09-28 (10x too high).
    Days outside Yahoo's coverage (dividends_known = False; mostly companies later taken private or
    moved to a new code) get imputed dividends on the March and September ex-dates, and so do the
    Yahoo dividends dropped as errors (see impute_dividends). Leaving them at zero would understate
    these stocks by about 2% a year.
    Spin-offs (SPINOFFS, listed by hand) count as a dividend worth the new shares at their JPX close
    on the ex-date. dividend_source says where each day's dividend comes from.
Returns
    log_return = ln(split * (close + dividend) / previous close): the total return from the last
    traded close, with the dividend counted on its ex-date (dividend in yen per share after that
    day's split). It is empty on a listing's first day, on days without a trade, and when the
    previous trade is more than MAX_RETURN_GAP calendar days back (e.g. across the missing 2016-06).
    adj_close compounds the same returns, scaled so that a listing's last adj_close equals its last
    raw close.
"""
import numpy as np
import pandas as pd

from config import DATA_DIR, YAHOO_DIR

STEP_MIN = 0.02           # log size of a ratio step / level difference that counts as a change
SEGMENT_MIN = 3           # Yahoo observations needed on each side of a step (fewer only at a series end)
USABLE_FIT = 0.9          # share of Yahoo days within STEP_MIN of their segment level for the series to be used
MIN_SPLIT = np.log(1.08)  # smallest split ratio considered (the usual smallest is 1.1:1)
# Split ratios looked for in the JPX prices alone (new shares per old share; < 1 is a consolidation).
SPLIT_RATIOS = np.array([1.1, 1.2, 1.25, 1.5, 2, 3, 4, 5, 10, 20, 25, 50, 100, 200, 300, 400, 500, 1000,
                         1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 10, 1 / 1000])
JPX_MAX_RESID = 0.10      # max |excess return after the split| for a price-only detection
JPX_UNIT_MAX_RESID = 0.15  # the same on a month-end ex-date or when the trading unit changes that day
JPX_MIN_PRICE = 30        # yen; below this, 1-yen ticks of distressed stocks look like 2:1 splits
JPX_MIN_SPLIT_PRICE = 500  # yen; the lowest pre-split price among the confirmed splits is 850
MONTH_END_DAYS = {1, 2, 3}  # trading days before a month's last one: the ex-dates of month-end record dates
UNEXPLAINED = 0.25        # |excess log move| listed in the report when no split explains it
MAX_DIVIDEND_YIELD = 0.5  # dividends above this share of the previous close are Yahoo errors
CHECK_YIELD = 0.08        # dividends above this share of the price must show in the ex-date price
MAX_RETURN_GAP = 14       # calendar days; a longer gap to the previous trade gives no daily return
IMPUTE_EVENTS = 6         # past ex-dates (3 years) behind a stock's own imputed-dividend estimate
IMPUTE_MIN_EVENTS = 4     # with fewer, the imputed yield is the market median alone
# (parent code, ex-date, new code, new shares per parent share). Sony gave one Sony Financial Group
# share per share; Yahoo's JPX/Yahoo ratio confirms the ex-date.
SPINOFFS = [("6758", "2025-09-29", "8729", 1.0)]


def snap(q):
    """Nearest plausible split ratio, and the log distance to it: a whole number (or its inverse for a
    consolidation), or below 3 a ratio in steps of 0.1 or 0.25 (1.1, 1.25, 2.5, ...)."""
    x = 1 / q if q < 1 else q
    for d in (1, 2, 4, 5, 10) if x < 3 else (1,):
        f = max(round(x * d), d) / d
        if abs(np.log(x / f)) < 0.01:
            return (1 / f if q < 1 else f), abs(np.log(x / f))
    return q, np.inf


def ratio_segments(x):
    """Split a log JPX/Yahoo ratio series into constant segments: [[start, end, level], ...]."""
    med = pd.Series(x).rolling(11, center=True, min_periods=3).median().to_numpy()
    cut = [0, *(np.flatnonzero(np.abs(np.diff(med)) > STEP_MIN) + 1), len(x)]
    segments = []
    for a, b in zip(cut[:-1], cut[1:]):
        if b <= a:
            continue
        if segments and abs(np.median(x[a:b]) - segments[-1][2]) < STEP_MIN:
            segments[-1][1] = b
            segments[-1][2] = np.median(x[segments[-1][0]:b])
        else:
            segments.append([a, b, np.median(x[a:b])])
    # the centred median can't see a step in the last (or first) few days: split off a run at the
    # series end that sits at another level
    a, b, lvl = segments[-1]
    run = b - 1
    while run > a and abs(x[run - 1] - x[b - 1]) < STEP_MIN:
        run -= 1
    if run > a and abs(x[b - 1] - lvl) > STEP_MIN:
        segments[-1] = [a, run, np.median(x[a:run])]
        segments.append([run, b, np.median(x[run:b])])
    a, b, lvl = segments[0]
    run = a + 1
    while run < b and abs(x[run] - x[a]) < STEP_MIN:
        run += 1
    if run < b and abs(x[a] - lvl) > STEP_MIN:
        segments[0] = [run, b, np.median(x[run:b])]
        segments.insert(0, [a, run, np.median(x[a:run])])
    return segments


def yahoo_splits(df, yahoo, yahoo_list):
    """Splits confirmed by the JPX/Yahoo ratio, the rejected steps, and Yahoo's coverage per listing."""
    both = df[df["traded"]].merge(yahoo, on=["code", "date"])
    both = both[both["yclose"] > 0].sort_values(["key", "date"])
    # stale Yahoo bars repeat a close while the JPX price moves; they bend the ratio into fake steps
    # (Okuma 6103 sat at 1865 from 2016-04-07 to 06-09, which read as a 1.1 split on 05-02). The first
    # bar of a repeated run goes too: JPX Group 8697's series opens with six bars at a placeholder 215,
    # and its first one alone read as a 1.1 consolidation on 2013-01-11.
    by_key = both.groupby("key")
    stale = ((by_key["yclose"].diff().eq(0) & by_key["close"].diff().ne(0))
             | (by_key["yclose"].diff(-1).eq(0) & by_key["close"].diff(-1).ne(0)))
    both = both[~stale]
    splits, rejected, coverage = [], [], []
    for key, g in both.groupby("key"):
        lst = df[df["key"] == key]
        x = np.log(g["close"] / g["yclose"]).to_numpy()
        dates = g["date"].to_numpy()
        segments = ratio_segments(x)
        fit = np.concatenate([np.abs(x[a:b] - lvl) < STEP_MIN for a, b, lvl in segments]).mean()
        usable = fit >= USABLE_FIT and len(x) >= 20
        coverage.append({"key": key, "yahoo_first": dates[0], "yahoo_last": dates[-1], "yahoo_days": len(x),
                         "fit": fit, "usable": usable, "level_end": segments[-1][2]})
        if not usable:
            continue
        code = g["code"].iloc[0]
        listed = yahoo_list[yahoo_list["code"] == code]
        for i in range(1, len(segments)):
            a = segments[i][0]
            q, err = snap(np.exp(segments[i - 1][2] - segments[i][2]))
            source = "yahoo ratio"
            if err > 0.01:
                # not a split ratio: Yahoo may have smoothed over the event; use its split list if it has one
                near = listed[(listed["date"] > dates[a - 1] - pd.Timedelta(days=5))
                              & (listed["date"] <= dates[a] + pd.Timedelta(days=5))]
                if len(near):
                    q, err, source = near["split"].prod(), 0.0, "yahoo split list"
            # the ex-date: the JPX day between the two Yahoo days whose move best matches the ratio
            days = lst[(lst["date"] > dates[a - 1]) & (lst["date"] <= dates[a]) & lst["r"].notna()]
            step = {"code": code, "key": key, "date": pd.Timestamp(dates[a]), "split": q}
            if days.empty:
                rejected.append({**step, "why": "no JPX trade between the Yahoo days"})
                continue
            resid = (days["e"] + np.log(q)).abs()
            day = days.loc[resid.idxmin()]
            step.update(date=day["date"], r=day["r"], resid=resid.min())
            short = segments[i - 1][1] - segments[i - 1][0] < SEGMENT_MIN or segments[i][1] - a < SEGMENT_MIN
            at_end = i == 1 or i == len(segments) - 1
            if err > 0.01 or abs(np.log(q)) < MIN_SPLIT:
                why = "ratio is not a split ratio"
            elif resid.min() >= abs(day["e"]):
                why = "JPX price does not move by the ratio"
            elif short and not (at_end and resid.min() < JPX_MAX_RESID):
                why = "Yahoo glitch (step lasts under 3 days)"
            else:
                splits.append({**step, "source": source})
                continue
            rejected.append({**step, "why": why})
    return pd.DataFrame(splits), pd.DataFrame(rejected), pd.DataFrame(coverage)


def jpx_splits(df):
    """Splits detected from the JPX prices alone (for every day; the caller keeps the uncovered ones)."""
    c = df[df["e"].notna()].copy()
    best = np.abs(c["e"].to_numpy()[:, None] + np.log(SPLIT_RATIOS)[None, :]).argmin(axis=1)
    c["split"] = SPLIT_RATIOS[best]
    lq = np.log(c["split"])
    c["resid"] = (c["e"] + lq).abs()
    unit_change = np.log(c["unit"] / c["unit_prev"])
    plausible = ((np.minimum(c["close"], c["close_prev"]) >= JPX_MIN_PRICE)
                 & ~((lq > 0) & (c["close_prev"] < JPX_MIN_SPLIT_PRICE))
                 & ~(c["r_next"].abs() >= 0.5 * lq.abs()))  # not undone the next day
    by_price = plausible & (lq.abs() >= np.log(2) - 0.01) & (c["resid"] <= JPX_MAX_RESID)
    by_month_end = (plausible & c["to_month_end"].isin(MONTH_END_DAYS)
                    & (lq.abs() >= np.log(1.5) - 0.01) & (c["resid"] <= JPX_UNIT_MAX_RESID))
    by_unit = ((lq.abs() >= np.log(1.5) - 0.01) & (c["resid"] <= JPX_UNIT_MAX_RESID)
               & (np.sign(unit_change) == np.sign(lq)) & (unit_change != 0))
    c["source"] = np.select([by_unit, by_month_end], ["jpx price + unit change", "jpx price on a month-end ex-date"],
                            "jpx price")
    return c[by_price | by_month_end | by_unit][["code", "key", "date", "split", "r", "resid", "source"]]


def convert_dividends(df, dividends, splits, coverage):
    """Yahoo dividends in yen per share as traded on the ex-date, for the listings Yahoo covers."""
    cov = coverage[coverage["usable"]]
    rows = df[["key", "code", "date", "close_prev"]].merge(cov, on="key")
    # factor from Yahoo's later-split-adjusted units back to the units traded on the day:
    # exp(level of the last ratio segment) x every confirmed split after the day
    later = splits.sort_values("date", ascending=False)
    later["log_after"] = np.log(later["split"]).groupby(later["key"]).cumsum() - np.log(later["split"])
    rows = rows.sort_values("date")
    rows = pd.merge_asof(rows, later.sort_values("date")[["key", "date", "split", "log_after"]],
                         on="date", by="key", direction="forward", allow_exact_matches=False)
    rows["log_after"] = rows["log_after"].fillna(0) + np.log(rows["split"]).fillna(0)
    rows["factor"] = np.exp(rows["level_end"] + rows["log_after"])

    d = dividends.rename(columns={"ex_date": "date"}).sort_values("date")
    # an ex-date on a day missing from the JPX data counts on the listing's next day
    d = pd.merge_asof(d, rows[["code", "date", "key", "yahoo_first", "yahoo_last", "factor", "close_prev"]],
                      on="date", by="code", direction="forward", tolerance=pd.Timedelta(days=MAX_RETURN_GAP))
    d = d[d["key"].notna() & (d["ex_date_orig"] > d["yahoo_first"]) & (d["ex_date_orig"] <= d["yahoo_last"])]
    d["dividend"] = d["amount"] * d["factor"]
    return d


def impute_dividends(df, dropped):
    """Dividends for the days Yahoo does not cover, and for the Yahoo dividends dropped as errors.

    On each March and September ex-date (the day most known dividends of that month go ex; 86% of
    dividends are paid in those two months), the imputed yield is the average of:
      - the median yield that day of the index members with known dividends (0 for those paying
        nothing), and
      - the stock's own excess price drop on ex-dates (its price return minus the market's total
        return), averaged over its previous IMPUTE_EVENTS ex-dates and floored at 0.
    The median alone is used while the stock has fewer than IMPUTE_MIN_EVENTS earlier ex-dates.
    The result is scaled up by the members' total dividends / their dividends on these ex-dates
    (about 1.17), so the dividends paid in other months (June/December payers, other record dates)
    are not lost: without it the annual yield came out 0.3% too low.
    Apart from that one constant, only past and same-day data are used. Tested in U5 (tracker):
    unbiased and the closest to each stock's long-run yield. Returns the imputed dividends, the
    scale, and the same method's error on the members whose dividends are known.
    """
    known = df["dividends_known"] & df["traded"]
    price = df["split"] * df["close"] / df["close_prev"]
    total = np.log(price + df["split"] * df["dividend"] / df["close_prev"])
    market = total[known].groupby(df.loc[known, "date"]).median()
    paid = df[known & (df["dividend"] > 0) & df["date"].dt.month.isin([3, 9])]
    ex_dates = paid.groupby(paid["date"].dt.to_period("M"))["date"].agg(lambda s: s.mode().iloc[0])

    ev = df[df["date"].isin(ex_dates)].copy()
    ev["year"] = ev["date"].dt.year
    ev["yield"] = ev["dividend"] * ev["split"] / ev["close_prev"]
    ev["drop"] = ev["date"].map(market) - np.log(price[ev.index])
    members = ev["dividends_known"] & ev["in_index"] & ev["traded"]
    ev["median"] = ev["date"].map(ev[members].groupby("date")["yield"].median())
    own = ev.groupby("key")["drop"].transform(
        lambda s: s.shift().rolling(IMPUTE_EVENTS, min_periods=IMPUTE_MIN_EVENTS).mean()).clip(lower=0)
    pay = df[known & (df["dividend"] > 0)]
    pay_yield = pay["dividend"] * pay["split"] / pay["close_prev"]
    member_pay = pay_yield[pay["in_index"]]
    scale = member_pay.sum() / member_pay[pay.loc[pay["in_index"], "date"].isin(ex_dates)].sum()
    ev["imputed_yield"] = scale * np.where(own.notna(), 0.5 * own + 0.5 * ev["median"], ev["median"])
    ev["imputed_dividend"] = ev["imputed_yield"] * ev["close_prev"] / ev["split"]

    # the method on the known dividends: annual yield per member, imputed vs paid (all payments that year)
    t = ev[members].groupby(["key", "year"]).agg(imputed=("imputed_yield", "sum"), n=("date", "size"))
    paid_yield = pay_yield.groupby([pay["key"], pay["date"].dt.year]).sum()
    t["paid"] = paid_yield.reindex(t.index).fillna(0).to_numpy()
    t = t[t["n"] == 2]  # member with known dividends on both ex-dates of the year
    err = t["imputed"] - t["paid"]
    test = {"n": len(t), "mean": err.mean(), "typical": err.abs().mean(), "paid": t["paid"].mean(), "scale": scale}

    was_dropped = ev.set_index(["key", "date"]).index.isin(dropped.set_index(["key", "date"]).index)
    missing = (~ev["dividends_known"] | was_dropped) & ev["close_prev"].notna() & (ev["imputed_dividend"] > 0)
    return ev[missing][["key", "code", "date", "imputed_dividend", "imputed_yield"]], test


def add_adjusted(df):
    """Add split, dividend, dividends_known, adj_close and log_return to the price table.

    df needs date, code, listing, close and unit, one row per (date, code). Returns the table, the
    list of corporate actions applied, and the report lines.
    """
    report = ["", "Adjusted closes and log returns (pipeline/adjust.py)"]
    df = df.sort_values(["code", "listing", "date"]).reset_index(drop=True)
    df["key"] = df["code"] + "_" + df["listing"].astype(str)
    df["traded"] = df["close"] > 0
    grp = df.groupby("key")
    prev = grp["close"].transform(lambda s: s.ffill().shift())  # last traded close before the day
    df["close_prev"] = prev
    df["r"] = np.log(df["close"] / prev).where(df["traded"])
    df["e"] = df["r"] - df.groupby("date")["r"].transform("median")  # move beyond the market's
    df["r_next"] = grp["r"].transform(lambda s: s.bfill().shift(-1)).where(df["traded"])
    df["unit_prev"] = grp["unit"].transform(lambda s: s.ffill().shift())
    days = pd.Series(sorted(df["date"].unique()))
    to_end = days.groupby(days.dt.to_period("M")).cumcount(ascending=False)  # 0 = the month's last trading day
    df["to_month_end"] = df["date"].map(dict(zip(days, to_end)))

    yahoo = pd.read_parquet(YAHOO_DIR / "check_prices.parquet", columns=["code", "date", "close"])
    yahoo = yahoo.rename(columns={"close": "yclose"}).assign(date=lambda y: pd.to_datetime(y["date"]))
    yahoo_list = pd.read_csv(YAHOO_DIR / "splits.csv", dtype={"code": str}, parse_dates=["date"])
    yahoo_list["split"] = yahoo_list["numerator"] / yahoo_list["denominator"]
    dividends = pd.read_csv(YAHOO_DIR / "dividends.csv", dtype={"code": str}, parse_dates=["ex_date"])

    # --- splits ---
    ysplits, rejected, coverage = yahoo_splits(df, yahoo, yahoo_list)
    cov = df[["key", "date"]].merge(coverage, on="key", how="left")
    covered = (cov["usable"].fillna(False).astype(bool)
               & (cov["date"] > cov["yahoo_first"]) & (cov["date"] <= cov["yahoo_last"])).to_numpy()
    detected = jpx_splits(df)
    in_cov = detected.index.isin(df.index[covered])
    jsplits = detected[~in_cov]
    splits = pd.concat([ysplits, jsplits], ignore_index=True)

    unusable = coverage[~coverage["usable"]]
    report.append(f"Yahoo closes overlap {len(coverage)} listings; {len(unusable)} too noisy to use "
                  f"({', '.join(unusable['key'])}).")
    from_list = (ysplits["source"] == "yahoo split list").sum()
    report.append(f"Splits: {len(ysplits)} from the JPX/Yahoo ratio (of which {from_list} with the ratio from "
                  f"Yahoo's split list), {len(jsplits)} from the JPX prices alone (days Yahoo does not cover).")
    # Yahoo's own split list, for comparison
    yl = yahoo_list.merge(df[["code", "date", "key"]], on=["code", "date"], how="left")
    yl = yl.merge(coverage[coverage["usable"]][["key", "yahoo_first", "yahoo_last"]], on="key")
    yl = yl[(yl["date"] > yl["yahoo_first"]) & (yl["date"] <= yl["yahoo_last"])]
    near = yl.merge(ysplits[["key", "date", "split"]], on="key", suffixes=("", "_found"))
    near = near[((near["date"] - near["date_found"]).dt.days.abs() <= 5)
                & (np.abs(np.log(near["split"] / near["split_found"])) < 0.01)]
    found = near[["key", "date_found"]].drop_duplicates()
    near = near[["key", "date"]].drop_duplicates()
    report.append(f"  Yahoo's split list has {len(yl)} splits on covered days; {len(near)} match a ratio step. "
                  f"{len(ysplits) - len(found)} ratio steps are missing from Yahoo's list.")
    for r in yl.merge(near, how="left", indicator=True).query("_merge == 'left_only'").itertuples():
        report.append(f"    in Yahoo's list but no ratio step: {r.code} {r.date:%Y-%m-%d} {r.split:g}")
    report.append(f"  Ratio steps rejected: {len(rejected)}")
    for r in rejected.itertuples():
        report.append(f"    {r.code} {r.date:%Y-%m-%d} ratio {r.split:.4g}: {r.why}")
    # the JPX-only detector, scored on the days Yahoo covers
    test = detected[in_cov].merge(ysplits[["key", "date", "split"]], on=["key", "date"], how="left",
                                  suffixes=("", "_y"))
    hit = np.abs(np.log(test["split"] / test["split_y"])) < 0.01
    report.append(f"  JPX-only detector tested on Yahoo-covered days: finds {hit.sum()} of {len(ysplits)} "
                  f"confirmed splits, with {(~hit).sum()} false alarms:")
    for r in test[~hit].itertuples():
        report.append(f"    {r.code} {r.date:%Y-%m-%d} ratio {r.split:g} ({r.source})")
    for r in jsplits.itertuples():
        report.append(f"    JPX-only split: {r.code} {r.date:%Y-%m-%d} ratio {r.split:g} "
                      f"(raw move {r.r:+.3f}, left after the split {r.resid:.3f}, {r.source})")
    split_days = df.index.isin(df.reset_index().merge(splits[["key", "date"]], on=["key", "date"])["index"])
    big = df[~covered & ~split_days & (df["e"].abs() > UNEXPLAINED)]
    penny = np.minimum(big["close"], big["close_prev"]) < JPX_MIN_PRICE
    report.append(f"  Moves above {UNEXPLAINED:.0%} (log, beyond the market) on days Yahoo does not cover, "
                  f"not explained by a split: {len(big)}, of which {penny.sum()} are stocks under {JPX_MIN_PRICE} yen "
                  f"(1-yen ticks of distressed names; not listed). The others:")
    for r in big[~penny].itertuples():
        report.append(f"    {r.code} {r.date:%Y-%m-%d} close {r.close_prev:g} -> {r.close:g}"
                      f" ({r.to_month_end} trading days before the month's last)")

    # --- dividends ---
    dividends["ex_date_orig"] = dividends["ex_date"]
    divs = convert_dividends(df, dividends, ysplits, coverage)
    # yield on the ex-date's share basis (a split the same day changes the basis of the previous close)
    same_day = splits.groupby(["key", "date"])["split"].prod().rename("q").reset_index()
    divs = divs.merge(same_day, on=["key", "date"], how="left").merge(df[["key", "date", "e"]], on=["key", "date"])
    divs["q"] = divs["q"].fillna(1.0)
    divs["moved"] = divs["e"] + np.log(divs["q"])

    def failed(dividend):
        """Too large, or large without showing in the ex-date price (at least half of it, beyond the market)."""
        y = dividend * divs["q"] / divs["close_prev"]
        return (y > MAX_DIVIDEND_YIELD) | ((y > CHECK_YIELD) & (divs["moved"] > np.log(1 - y / 2)))

    # with a split the same day, Yahoo gives the amount per new share (Toyota 2021-09-29) or per old
    # share (Nippon Steel 2025-09-29); the per-new-share reading is kept unless the price rejects it
    per_old = (divs["q"] != 1) & failed(divs["dividend"]) & ~failed(divs["dividend"] / divs["q"])
    divs.loc[per_old, "dividend"] /= divs.loc[per_old, "q"]
    divs["yield"] = divs["dividend"] * divs["q"] / divs["close_prev"]
    bad = failed(divs["dividend"])
    report.append(f"Dividends: {len(dividends)} in Yahoo's list, {len(divs)} on days Yahoo covers. "
                  f"{per_old.sum()} on a split day read as per old share. {bad.sum()} dropped as Yahoo errors (above {MAX_DIVIDEND_YIELD:.0%} of the price, or above "
                  f"{CHECK_YIELD:.0%} with no matching drop in the ex-date price):")
    for r in divs[bad].rename(columns={"yield": "dy"}).itertuples():
        report.append(f"    {r.code} {r.date:%Y-%m-%d}: {r.dividend:,.1f} yen = {r.dy:.3g} x the price; "
                      f"price moved {np.expm1(r.moved):+.1%} beyond the market (imputed instead)")
    dropped = divs[bad][["key", "date"]]
    divs = divs[~bad]
    report.append(f"  Dividend yield per payment: median {divs['yield'].median():.2%}, 99th percentile "
                  f"{divs['yield'].quantile(0.99):.2%}. Kept above {CHECK_YIELD:.0%} (each confirmed by the price):")
    for r in divs[divs["yield"] > CHECK_YIELD].rename(columns={"yield": "dy"}).itertuples():
        report.append(f"    {r.code} {r.date:%Y-%m-%d}: {r.dividend:,.1f} yen ({r.dy:.1%} of the price; "
                      f"price moved {np.expm1(r.moved):+.1%} beyond the market)")
    divs["source"] = "yahoo"
    # spin-offs: the new shares at their close on the ex-date (the new code's rows are not in df,
    # as it is not an index member, so its price comes from the full JPX table)
    jpx = pd.read_parquet(DATA_DIR / "jpx_prices.parquet", columns=["code", "date", "close"])
    for parent, day, child, ratio in SPINOFFS:
        day = pd.Timestamp(day)
        price = jpx.loc[(jpx["code"] == child) & (pd.to_datetime(jpx["date"]) == day), "close"]
        row = df[(df["code"] == parent) & (df["date"] == day)]
        if price.empty or row.empty:
            report.append(f"  Spin-off {parent} -> {child} on {day:%Y-%m-%d}: price not found, not applied")
            continue
        value = ratio * price.iloc[0]
        divs = pd.concat([divs, pd.DataFrame([{"code": parent, "key": row["key"].iloc[0], "date": day,
                                               "dividend": value, "source": f"spin-off of {child}"}])])
        report.append(f"  Spin-off {parent} -> {child} on {day:%Y-%m-%d}: {ratio:g} new share(s) at "
                      f"{price.iloc[0]:,.1f} yen = {value:,.1f} yen per share")

    # --- events into the table; dividends imputed where Yahoo has none ---
    df = df.merge(splits.groupby(["key", "date"])["split"].prod().reset_index(), on=["key", "date"], how="left")
    df = df.merge(divs.groupby(["key", "date"])["dividend"].sum().reset_index(), on=["key", "date"], how="left")
    df["split"] = df["split"].fillna(1.0)
    df["dividend"] = df["dividend"].fillna(0.0)
    df["dividends_known"] = covered
    imputed, test = impute_dividends(df, dropped)
    df = df.merge(imputed[["key", "date", "imputed_dividend"]], on=["key", "date"], how="left")
    df["dividend"] += df.pop("imputed_dividend").fillna(0.0)
    imputed = imputed.rename(columns={"imputed_dividend": "dividend"}).assign(source="imputed")
    divs = pd.concat([divs, imputed])
    source = divs.groupby(["key", "date"])["source"].agg(", ".join).rename("dividend_source").reset_index()
    df = df.merge(source, on=["key", "date"], how="left")
    df["dividend_source"] = df["dividend_source"].fillna("")
    report.append(f"  Imputed dividends (days Yahoo does not cover, and the dropped Yahoo errors): {len(imputed)} "
                  f"on {imputed['code'].nunique()} stocks, median {imputed['imputed_yield'].median():.2%} per payment "
                  f"(scaled by {test['scale']:.3f} for the dividends paid outside March and September).")
    report.append(f"  The same method on the index members with known dividends ({test['n']} stock-years, paying "
                  f"{test['paid']:.2%} a year on average): annual yield error {test['mean']:+.2%} on average, "
                  f"{test['typical']:.2%} typical (leaving them at zero: {-test['paid']:+.2%}).")

    # --- returns ---
    # events on a day without a trade count on the listing's next traded day
    df["slot"] = df.index.where(df["traded"])
    df["slot"] = df.groupby("key")["slot"].transform(lambda s: s.bfill())
    lost = df[df["slot"].isna() & ((df["split"] != 1) | (df["dividend"] != 0))]
    if len(lost):
        report.append(f"  Events after a listing's last trade, not applied: {len(lost)}")
    ev = df.groupby("slot").agg(q=("split", "prod"), d=("dividend", "sum"))
    ev.index = ev.index.astype(int)
    q = df.index.map(ev["q"]).to_numpy(dtype=float)
    d = df.index.map(ev["d"]).to_numpy(dtype=float)
    gross = q * (df["close"] + d) / df["close_prev"]
    df["gross"] = gross.where(df["traded"])
    last_trade = df["date"].where(df["traded"]).groupby(df["key"]).transform(lambda s: s.ffill().shift())
    gap = (df["date"] - last_trade).dt.days > MAX_RETURN_GAP
    df["log_return"] = np.log(df["gross"]).where(~gap)
    # total-return index (gross is 1 on days without a trade), scaled to end at the last raw close
    index = df["gross"].fillna(1.0).groupby(df["key"]).cumprod()
    scale = df.groupby("key")["close"].transform("last") / index.groupby(df["key"]).transform("last")
    df["adj_close"] = (index * scale).where(df["traded"])

    report.append(f"Returns: {df['log_return'].notna().sum()} daily log returns; empty on "
                  f"{(~df['traded']).sum()} days without a trade, {df['key'].nunique()} listing first days, and "
                  f"{(gap & df['traded']).sum()} days more than {MAX_RETURN_GAP} days after the previous trade.")
    known = df[df["in_index"]].groupby(df["date"].dt.year)["dividends_known"].mean()
    report.append("  Share of index-member days with Yahoo dividend data (the rest imputed), by year: "
                  + ", ".join(f"{y}: {v:.0%}" for y, v in known.items()))

    actions = pd.concat([
        splits.assign(event="split", value=splits["split"])[["code", "date", "event", "value", "source"]],
        divs.assign(event="dividend", value=divs["dividend"])[["code", "date", "event", "value", "source"]],
    ]).sort_values(["code", "date", "event"]).reset_index(drop=True)
    drop = ["key", "traded", "close_prev", "r", "e", "r_next", "unit_prev", "to_month_end", "slot", "gross"]
    return df.drop(columns=drop), actions, report
