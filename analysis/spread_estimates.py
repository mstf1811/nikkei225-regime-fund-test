"""Bid-ask spreads of Nikkei 225 members, estimated from daily close, high and low prices.

There is no free quote history for Japan, so the spread is estimated with Abdi & Ranaldo (2017),
"A Simple Estimation of Bid-Ask Spreads from Daily Close, High, and Low Prices", RFS 30(12):
    s^2 = 4 E[(c_t - eta_t)(c_t - eta_{t+1})],  c = log close,  eta = (log high + log low) / 2.
Per stock-month: the mean of the two-day products over the member days of the month (day t a
member, day t+1 the stock's next trading day), with at least 10 products; s = sqrt(max(mean, 0)).
Prices are scaled by adj_close / close, so splits and dividends don't show up as price moves.

The estimate is the full quoted spread as a share of the price; a trade pays about half of it.
The tick-size floor (one tick / price) is reported next to it: with the ¥1 tick of the 2000s, a
¥135 stock could not trade at a spread below 0.74%.

Result (2026-10-07): NOT USABLE for Nikkei 225 members. Their spreads are far below the estimator's
noise: 45% of stock-months have a negative mean product, and even pooled over all member days the
yearly estimate jumps between 0 (2005, 2011, 2018-2020, 2023) and 72 bps (2024), following
market-wide return autocorrelation rather than liquidity. Kept as the record of why the cost model
does not use daily-data spread estimates.

Input: data/prices_daily.parquet. Outputs (analysis/output/costs/): spread_monthly.parquet
(one row per member stock-month), spread_by_year.csv, summary.txt.

Run:  python analysis/spread_estimates.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import DATA_DIR, PROJECT_DIR  # noqa: E402

OUT = PROJECT_DIR / "analysis" / "output" / "costs"
MIN_PRODUCTS = 10


def tick_size(price, date):
    """TSE tick for a price (yen). Before 2014-01-14 the old table; from then the TOPIX100 names had
    finer ticks, which this function ignores (so the floor is an upper bound for them after 2014)."""
    bands = [(3000, 1), (5000, 5), (30000, 10), (50000, 50), (300000, 100), (500000, 500),
             (3000000, 1000), (5000000, 5000), (30000000, 10000), (50000000, 50000)]
    for top, tick in bands:
        if price <= top:
            return tick
    return 100000


def load():
    px = pd.read_parquet(DATA_DIR / "prices_daily.parquet",
                         columns=["date", "code", "in_index", "high", "low", "close", "adj_close"])
    px["date"] = pd.to_datetime(px["date"])
    px = px.dropna(subset=["high", "low", "close", "adj_close"])
    px = px[(px["low"] > 0) & (px["high"] >= px["low"])]
    return px.sort_values(["code", "date"]).reset_index(drop=True)


def two_day_products(px):
    f = px["adj_close"] / px["close"]
    c = np.log(px["close"] * f)
    eta = 0.5 * (np.log(px["high"] * f) + np.log(px["low"] * f))
    nxt = eta.groupby(px["code"]).shift(-1)
    # the next row must be the stock's next trading day, not a later listing spell
    gap = px.groupby("code")["date"].shift(-1) - px["date"]
    ok = nxt.notna() & (gap <= pd.Timedelta(days=14))
    out = px.loc[ok & px["in_index"], ["date", "code", "close"]].copy()
    out["prod"] = 4 * (c - eta)[out.index] * (c - nxt)[out.index]
    return out


def monthly(prod):
    prod["month"] = prod["date"].dt.to_period("M")
    g = prod.groupby(["code", "month"])
    m = g.agg(mean_prod=("prod", "mean"), n=("prod", "size"), price=("close", "median"),
              last=("date", "max")).reset_index()
    m = m[m["n"] >= MIN_PRODUCTS].copy()
    m["spread"] = np.sqrt(m["mean_prod"].clip(lower=0))
    m["negative"] = m["mean_prod"] < 0
    m["tick_floor"] = [tick_size(p, d) / p for p, d in zip(m["price"], m["last"])]
    return m


def by_year(m):
    m = m.assign(year=m["month"].dt.year)
    q = lambda p: (lambda x: x.quantile(p))  # noqa: E731
    t = m.groupby("year").agg(
        stock_months=("spread", "size"),
        median=("spread", "median"), p10=("spread", q(0.10)), p90=("spread", q(0.90)),
        mean=("spread", "mean"), share_negative=("negative", "mean"),
        tick_floor_median=("tick_floor", "median"))
    return t


def pooled_by_year(prod):
    """One estimate per year from the mean product over all member days (least noisy version)."""
    g = prod.groupby(prod["date"].dt.year)["prod"].mean()
    return (1e4 * np.sqrt(g.clip(lower=0))).round(1).rename("pooled_bps")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px = load()
    prod = two_day_products(px)
    pooled = pooled_by_year(prod)
    m = monthly(prod)
    m.drop(columns="last").assign(month=m["month"].astype(str)).to_parquet(
        OUT / "spread_monthly.parquet", index=False)
    t = by_year(m)
    t.to_csv(OUT / "spread_by_year.csv", float_format="%.6f")

    bp = t.copy()
    for col in ["median", "p10", "p90", "mean", "tick_floor_median"]:
        bp[col] = (1e4 * bp[col]).round(1)
    bp["share_negative"] = (100 * bp["share_negative"]).round(1)
    test = m[(m["month"] >= pd.Period("2005-01", "M")) & (m["month"] <= pd.Period("2025-09", "M"))]
    lines = [
        "Abdi-Ranaldo (2017) spread estimates, Nikkei 225 members, one value per stock-month",
        f"(>= {MIN_PRODUCTS} two-day products). Full quoted spread in bps of the price; a trade pays about half.",
        "",
        bp.rename(columns={"share_negative": "neg_%"}).join(pooled).to_string(),
        "",
        f"Test period 2005-01 -> 2025-09: {len(test):,} stock-months, median {1e4 * test['spread'].median():.1f} bps, "
        f"mean {1e4 * test['spread'].mean():.1f} bps; {100 * test['negative'].mean():.1f}% of months "
        "have a negative mean product (set to 0).",
        "Stock-months whose estimate is below the tick floor: "
        f"{100 * (test['spread'] < test['tick_floor']).mean():.1f}% "
        "(the estimator is noisy month by month; average over months before using it per stock).",
        "",
        "Verdict: not usable. pooled_bps = one estimate per year from all member days; it swings between 0",
        "and 72 bps with market-wide return autocorrelation, so the spread signal is below the noise.",
    ]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
