"""TSE daily price limits (制限値幅) and the days each stock hit them.

The rule
    Each day a stock may trade only within base price +/- a yen amount set by the band of the base
    price (基準値段). Two tables cover 2000-2025:
      LIMITS_2000  used until 2009-12-30 (the table before arrowhead; source: broker notices of the
                   2010 change, e.g. Yutaka Securities, with the old and new tables side by side);
      LIMITS_2010  from 2010-01-04, when arrowhead started (JPX's current table, page checked 2026-10-02).
    Base price: the previous day's close, or its final special quote (最終気配) if the session ended on
    a special quote; carried over a day with neither. On an ex-date it is cut to the theoretical
    price: divided by the split ratio, minus the expected dividend (we use the dividend applied in
    prices_daily, which is the paid or imputed amount, not the forecast TSE uses).
    The stop price can sit up to one tick beyond base +/- limit when that is not a valid price, so a
    hit is read as "close at or beyond base +/- limit" and only moves beyond it by more than a tick
    are treated as anomalies.
    Widening (制限値幅の拡大): after STOP_RUN consecutive stop days in one direction, that side of the
    limit is multiplied from the next day until the stock trades at another price. Until 2020-07-31:
    3 days, x2; from 2020-08-03: 2 days, x4 (JPX notice of 2020-06-30). A stop day is a day without
    trades ending on a special quote at the stop price, or one traded only at the stop price at the
    close; the daily file has no intraday times, so the latter is read as open = high = low = close
    = stop price. When the x2 rule started is not verified (after the April 2000 Hikari Tsushin run).
    Temporary change: limits halved on 2001-09-12 to 09-14 (after 9/11).
    Not modelled: the removed lower limit for stocks designated for delisting (整理銘柄) and the
    first-day rules for new listings. The report lists any move beyond the computed limit.

Each row gets one hit type:
    close_up / close_down   traded and closed at the stop price: the close-to-close return is censored
    quote_up / quote_down   no trade, final special quote at the stop price: censored and missing
    touch_up / touch_down   the high (low) reached the stop price but the close is inside: not censored
    (empty)                 none of the above
    beyond                  close beyond the stop price by more than a tick: a rule we don't model
                            (or a data error); listed in the report. A day traded at a single price
                            beyond it is instead read as a stop day whose base we can't reproduce
                            (off_base = True, also listed), e.g. TSE deducting a forecast dividend

Outputs (DATA_DIR):
    price_limits_daily.parquet   date, code, in_index, base, limit, upper, lower, widened_up,
                                 widened_down, hit, off_base, width_up, width_down (log distance from
                                 the base to the normal stop prices)
    price_limits_report.txt      counts and checks

Run:  python pipeline/price_limits.py
"""
import bisect
from datetime import date

import numpy as np
import pandas as pd

from config import DATA_DIR

# (upper bound of the base-price band in yen, exclusive; limit in yen). The last band has no bound.
LIMITS_2000 = [(100, 30), (200, 50), (500, 80), (700, 100), (1_000, 100), (1_500, 200), (2_000, 300),
               (3_000, 400), (5_000, 500), (7_000, 1_000), (10_000, 1_000), (15_000, 2_000),
               (20_000, 2_000), (30_000, 3_000), (50_000, 4_000), (70_000, 5_000), (100_000, 10_000),
               (150_000, 20_000), (200_000, 30_000), (300_000, 40_000), (500_000, 50_000),
               (700_000, 100_000), (1_000_000, 100_000), (1_500_000, 200_000), (2_000_000, 300_000),
               (3_000_000, 400_000), (5_000_000, 500_000), (7_000_000, 1_000_000),
               (10_000_000, 1_000_000), (15_000_000, 2_000_000), (20_000_000, 3_000_000),
               (30_000_000, 4_000_000), (50_000_000, 5_000_000), (np.inf, 10_000_000)]
LIMITS_2010 = [(100, 30), (200, 50), (500, 80), (700, 100), (1_000, 150), (1_500, 300), (2_000, 400),
               (3_000, 500), (5_000, 700), (7_000, 1_000), (10_000, 1_500), (15_000, 3_000),
               (20_000, 4_000), (30_000, 5_000), (50_000, 7_000), (70_000, 10_000), (100_000, 15_000),
               (150_000, 30_000), (200_000, 40_000), (300_000, 50_000), (500_000, 70_000),
               (700_000, 100_000), (1_000_000, 150_000), (1_500_000, 300_000), (2_000_000, 400_000),
               (3_000_000, 500_000), (5_000_000, 700_000), (7_000_000, 1_000_000),
               (10_000_000, 1_500_000), (15_000_000, 3_000_000), (20_000_000, 4_000_000),
               (30_000_000, 5_000_000), (50_000_000, 7_000_000), (np.inf, 10_000_000)]
TABLE_2010_FROM = date(2010, 1, 4)

# Tick sizes, used only as the tolerance for rounding the stop price to a valid price: the coarser
# (non-TOPIX100) ticks, before and after arrowhead. (upper bound of the price, inclusive; tick)
# Before 2010 the ticks of high-priced stocks were coarser earlier in the decade (stop-high closes
# such as 1,090,000 on a base of 986,000 in 2004 imply a 10,000 tick above 1,000,000), so the
# coarsest tick seen is used.
TICKS_2000 = [(2_000, 1), (3_000, 5), (30_000, 10), (50_000, 50), (100_000, 100), (1_000_000, 1_000),
              (20_000_000, 10_000), (30_000_000, 50_000), (np.inf, 100_000)]
TICKS_2010 = [(3_000, 1), (5_000, 5), (30_000, 10), (50_000, 50), (300_000, 100), (500_000, 500),
              (3_000_000, 1_000), (5_000_000, 5_000), (30_000_000, 10_000), (50_000_000, 50_000),
              (np.inf, 100_000)]

HALVED = (date(2001, 9, 12), date(2001, 9, 14))   # limits halved after 9/11
WIDENING_2020 = date(2020, 8, 3)                  # new widening rule: 2 stop days -> x4
MAX_BASE_GAP = 14  # calendar days; with a longer gap to the previous row the base price is unknown
EPS = 1e-6


def band_value(table, price):
    bounds = [b for b, _ in table]
    return table[bisect.bisect_right(bounds, price)][1]


def tick_value(table, price):
    bounds = [b for b, _ in table]
    return table[bisect.bisect_left(bounds, price)][1]


def daily_limit(base, day):
    limit = band_value(LIMITS_2010 if day >= TABLE_2010_FROM else LIMITS_2000, base)
    if HALVED[0] <= day <= HALVED[1]:
        limit /= 2
    return limit


def tick(price, day):
    return tick_value(TICKS_2010 if day >= TABLE_2010_FROM else TICKS_2000, price)


def widening(day):
    """(stop days needed, multiplier) under the rule in force on a qualifying day."""
    return (2, 4) if day >= WIDENING_2020 else (3, 2)


def classify_listing(rows):
    """Base price, limits and hit type for one code's rows (sorted by date)."""
    out = []
    base = None          # today's base, before adjustment
    prev_day = None
    run = {"up": 0, "down": 0}      # consecutive stop days
    mult = {"up": 1, "down": 1}     # active widening multiplier
    pending = {"up": 1, "down": 1}  # multiplier for tomorrow once a run completes
    for r in rows:
        day = r.date
        if prev_day is None or (day - prev_day).days > MAX_BASE_GAP or base is None:
            b = None
        else:
            b = base / r.split - r.dividend
        rec = {"base": np.nan, "limit": np.nan, "upper": np.nan, "lower": np.nan,
               "widened_up": 1, "widened_down": 1, "hit": "", "off_base": False}
        traded = not np.isnan(r.close)
        if b is not None and b > 0:
            lim = daily_limit(b, day)
            upper = b + lim * mult["up"]
            lower = max(b - lim * mult["down"], 1.0)
            tol = tick(max(upper, b), day)
            rec.update(base=b, limit=lim, upper=upper, lower=lower,
                       widened_up=mult["up"], widened_down=mult["down"])
            stop = {"up": False, "down": False}
            if traded:
                flat = r.open == r.high == r.low == r.close
                if r.close > upper + tol + EPS or r.close < b - lim * mult["down"] - tick(b, day) - EPS:
                    # A day traded at one price beyond our limit is locked at a stop price whose base
                    # we can't reproduce: on ex-dates TSE deducts the *forecast* dividend (TEPCO
                    # 2011-03-29: forecast 30 yen, later cancelled, so prices_daily has none).
                    if flat:
                        rec["hit"] = "close_up" if r.close > b else "close_down"
                        rec["off_base"] = True
                    else:
                        rec["hit"] = "beyond"
                elif r.close >= upper - EPS:
                    rec["hit"] = "close_up"
                elif r.close <= lower + EPS:
                    rec["hit"] = "close_down"
                elif r.high >= upper - EPS:
                    rec["hit"] = "touch_up"
                elif r.low <= lower + EPS:
                    rec["hit"] = "touch_down"
                stop["up"] = flat and rec["hit"] == "close_up"
                stop["down"] = flat and rec["hit"] == "close_down"
            elif r.quote_type == "ｶ" and r.quote_price >= upper - EPS:
                rec["hit"] = "quote_up"
                stop["up"] = True
            elif r.quote_type == "ｳ" and r.quote_price <= lower + EPS:
                rec["hit"] = "quote_down"
                stop["down"] = True
            # widening state for tomorrow
            for side in ("up", "down"):
                if stop[side]:
                    run[side] += 1
                    need, factor = widening(day)
                    if run[side] >= need:
                        pending[side] = max(mult[side], factor)
                else:
                    run[side] = 0
                    if traded:            # a trade at another price ends the widening
                        pending[side] = 1
                    else:
                        pending[side] = mult[side]
            mult = dict(pending)
        else:   # base unknown: start the widening state afresh
            run = {"up": 0, "down": 0}
            mult = {"up": 1, "down": 1}
            pending = {"up": 1, "down": 1}
        out.append(rec)
        # tomorrow's base: the final special quote, else the close, else carried over
        if not np.isnan(r.quote_price) and r.quote_type in ("ｶ", "ｳ"):
            base = r.quote_price
        elif traded:
            base = r.close
        else:
            base = b
        prev_day = day
    return out


def build():
    cols = ["date", "code", "name", "in_index", "open", "high", "low", "close", "log_return",
            "quote_type", "quote_price", "split", "dividend"]
    df = pd.read_parquet(DATA_DIR / "prices_daily.parquet", columns=cols)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    recs = []
    for _, g in df.groupby("code", sort=False):
        recs += classify_listing(g.itertuples(index=False))
    res = pd.concat([df[["date", "code", "name", "in_index", "close", "log_return", "dividend"]],
                     pd.DataFrame(recs)], axis=1)
    res["width_up"] = np.log1p(res["limit"] / res["base"])
    res["width_down"] = -np.log1p(-(res["limit"] / res["base"]).clip(upper=0.999))
    return res


def report(res):
    lines = []
    m = res[res["in_index"]].copy()
    m["year"] = pd.to_datetime(m["date"]).dt.year
    for label, sel in [("Whole file 2000-01 to 2025-09", m),
                       ("Test period 2005-01 to 2025-09", m[m["year"] >= 2005])]:
        known = sel["base"].notna()
        lines.append(f"{label}: {len(sel)} member days, {known.sum()} with a known base price")
        counts = sel.loc[known, "hit"].replace("", "none").value_counts()
        for k, v in counts.items():
            lines.append(f"  {k:<11} {v:>9}  {v / known.sum():.4%}")
        cens = sel["hit"].isin(["close_up", "close_down", "quote_up", "quote_down"])
        withret = sel["log_return"].notna()
        lines.append(f"  censored (close or quote at the stop): {cens.sum()}, "
                     f"{cens.sum() / known.sum():.4%} of member days; "
                     f"{(cens & withret).sum()} of them have a log return "
                     f"({(cens & withret).sum() / withret.sum():.4%} of member returns)")
        lines.append(f"  stocks with at least one censored day: {sel.loc[cens, 'code'].nunique()}")
    t = m[m["year"] >= 2005]
    lines.append("Width of the limit in log-return terms, member days 2005+ (normal limit):")
    for side in ("width_up", "width_down"):
        q = t[side].quantile([0, 0.01, 0.1, 0.5, 0.9]).round(3).to_dict()
        lines.append(f"  {side}: min {q[0]}, 1% {q[0.01]}, 10% {q[0.1]}, median {q[0.5]}, 90% {q[0.9]}")
    w = t[(t["widened_up"] > 1) | (t["widened_down"] > 1)]
    lines.append(f"Member days 2005+ with a widened limit: {len(w)}")
    for r in w.itertuples():
        lines.append(f"  {r.date} {r.code} {r.name}: x{max(r.widened_up, r.widened_down)} -> {r.hit or 'inside'}")
    o = res[res["off_base"]]
    lines.append(f"Days locked at one price beyond our limit, counted as stop days (all rows): {len(o)}")
    for r in o.itertuples():
        lines.append(f"  {r.date} {r.code} {r.name} member={r.in_index}: base {r.base:.1f}, "
                     f"limit {r.limit:.0f}, close {r.close:.1f}, dividend in prices_daily {r.dividend:g}")
    b = res[res["hit"] == "beyond"]
    lines.append(f"Closes beyond the stop price by more than a tick (all rows): {len(b)}")
    for r in b.itertuples():
        lines.append(f"  {r.date} {r.code} {r.name} member={r.in_index}: base {r.base:.1f}, "
                     f"limit {r.limit:.0f}, close {r.close:.1f}")
    return "\n".join(lines)


def main():
    res = build()
    out = res.drop(columns=["close", "log_return", "dividend"])
    out.to_parquet(DATA_DIR / "price_limits_daily.parquet", index=False)
    text = report(res)
    (DATA_DIR / "price_limits_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
