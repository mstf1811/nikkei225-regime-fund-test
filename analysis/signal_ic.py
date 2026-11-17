"""Do price-based signals predict which Nikkei 225 members do better next month? (U7, D12)

The six signals are fixed in advance from the literature (D12); this test only describes them.
At each month-end t, for the index members on that day:
    rev     log return over month t                           (literature: reversal, IC < 0)
    mom     log return over months t-11 .. t-1 (12-1)          (momentum, IC > 0; weak in Japan)
    vol     std of daily log returns, last 252 days, annualised (low-volatility effect, IC < 0)
    beta    on the equal-weighted member return, last 252 days  (low-beta effect, IC <= 0)
    dy      dividends over the last 12 months / price at ex-date (yield, IC > 0); imputed dividends
            (D10) included, the Sony spin-off excluded; needs 12 months of returns
    illiq   log of the 12-month mean of monthly Amihud ratios: mean |daily return| / average daily
            trading value in yen (to 2019 close x volume from the daily file; from 2020 the JPX
            monthly tables, data/liquidity_monthly.parquet); needs 6 months  (illiquidity, IC > 0)
vol and beta need 120 daily returns in the window; mom needs all 11 months.

Forward returns are total returns from prices_daily (a member delisted during the month keeps its
return up to its last day). Signal months 2004-12 to 2025-08, so return months 2005-01 to 2025-09.

Measures (h = 1 unless stated):
    IC            Spearman correlation across members between the signal and the next month's return
    t-stats       Newey-West, h + 2 lags
    quintiles     equal-weighted next-month simple returns by signal quintile; Q5 - Q1
    FM slope      next-month return on the signal's cross-sectional z-score (of its rank): the return
                  per one standard deviation of the signal; multivariate with all six signals
    persistence   Spearman correlation of the signal with itself one month later (high = low turnover)
    decay         IC of the signal with the return of month t+k alone, k = 1, 2, 3, 6, 12

Dividend-yield robustness: dividends lagged two months (March amounts are confirmed around May,
so the baseline uses an amount the market only roughly knows at the ex-date); windows without an
imputed dividend; payers only.

Outputs (analysis/output/signals/): summary.txt, ic_monthly.csv, signal_summary.csv,
subperiods.csv, decay.csv, fama_macbeth.csv, signal_corr.csv, quintiles.csv, coverage.csv,
dy_robustness_ic.csv

Run:  python analysis/signal_ic.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import DATA_DIR, PROJECT_DIR  # noqa: E402

OUT = PROJECT_DIR / "analysis" / "output" / "signals"
FIRST_SIGNAL, LAST_SIGNAL = pd.Period("2004-12", "M"), pd.Period("2025-08", "M")
SIGNALS = ["rev", "mom", "vol", "beta", "dy", "illiq"]
EXPECTED = {"rev": "-", "mom": "+ (weak in Japan)", "vol": "-", "beta": "- or 0", "dy": "+", "illiq": "+"}
WINDOW, MIN_DAYS = 252, 120
MIN_STOCKS = 50
SUBPERIODS = [("2005-2011", "2005-01", "2011-12"), ("2012-2018", "2012-01", "2018-12"),
              ("2019-2025", "2019-01", "2025-09")]
DECAY = [1, 2, 3, 6, 12]


def nw_t(x, lags):
    """Mean, Newey-West standard error and t-stat of a monthly series."""
    x = pd.Series(x).dropna().to_numpy()
    n = len(x)
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        var += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    se = np.sqrt(var / n)
    return x.mean(), se, x.mean() / se


def zrank(s):
    """Cross-sectional z-score of the ranks (robust to outliers)."""
    r = s.rank()
    return (r - r.mean()) / r.std()


def load():
    px = pd.read_parquet(DATA_DIR / "prices_daily.parquet",
                         columns=["date", "code", "in_index", "close", "log_return", "volume", "dividend",
                                  "dividend_source"])
    px["date"] = pd.to_datetime(px["date"])
    liq = pd.read_parquet(DATA_DIR / "liquidity_monthly.parquet", columns=["month", "code", "value"])
    return px, liq


def build_signals(px, liq):
    """Month x code panels of the signals and the forward returns."""
    wide = lambda col: px.pivot(index="date", columns="code", values=col)  # noqa: E731
    r = wide("log_return")
    member = wide("in_index").eq(True)
    dates = r.index
    month = dates.to_period("M")
    month_end = pd.Series(dates, index=dates).groupby(month).max()
    days_in_month = pd.Series(1, index=dates).groupby(month).sum()

    rm = r.groupby(month).sum(min_count=1)                 # monthly log returns
    s = {"rev": rm}
    s["mom"] = rm.shift(1).rolling(11, min_periods=11).sum()

    n = r.notna().rolling(WINDOW, min_periods=1).sum()
    vol = r.rolling(WINDOW, min_periods=MIN_DAYS).std() * np.sqrt(252)
    s["vol"] = vol.loc[month_end.values].set_axis(month_end.index)

    mkt = r.where(member).mean(axis=1)                      # equal-weighted member return
    m = r.notna().mul(mkt, axis=0).where(r.notna())         # the market on the stock's own days
    sx, sy = r.rolling(WINDOW, min_periods=1).sum(), m.rolling(WINDOW, min_periods=1).sum()
    sxy, syy = (r * m).rolling(WINDOW, min_periods=1).sum(), (m * m).rolling(WINDOW, min_periods=1).sum()
    beta = ((sxy - sx * sy / n) / (syy - sy ** 2 / n)).where(n >= MIN_DAYS)
    s["beta"] = beta.loc[month_end.values].set_axis(month_end.index)

    div = px[(px["dividend"] > 0) & (px["dividend_source"] != "spin-off of 8729")]
    y = (div["dividend"] / div["close"]).groupby([div["date"].dt.to_period("M"), div["code"]]).sum()
    ym = y.unstack().reindex(index=rm.index, columns=rm.columns).fillna(0.0)
    full_year = rm.notna().rolling(12, min_periods=12).sum() == 12
    s["dy"] = ym.rolling(12, min_periods=12).sum().where(full_year)
    # robustness: dividends known two months later (March amounts are confirmed around May), and
    # whether the window holds an imputed dividend (D10)
    s["dy_lag2"] = s["dy"].shift(2).where(full_year)
    imp = div[div["dividend_source"] == "imputed"]
    im = imp.groupby([imp["date"].dt.to_period("M"), imp["code"]]).size().unstack()
    s["dy_imputed"] = (im.reindex(index=rm.index, columns=rm.columns).fillna(0).rolling(12, min_periods=1).sum() > 0)

    value_daily = (px["close"] * px["volume"]).groupby([px["date"].dt.to_period("M"), px["code"]]).sum(min_count=1)
    value = value_daily.unstack().reindex(index=rm.index, columns=rm.columns)
    lq = liq.assign(month=pd.PeriodIndex(liq["month"], freq="M")).pivot(index="month", columns="code", values="value")
    lq = lq.reindex(index=rm.index, columns=rm.columns)
    monthly_table = value.index >= pd.Period("2020-01", "M")   # the daily file has no volume from 2020
    value.loc[monthly_table] = lq.loc[monthly_table]
    adv = value.div(days_in_month, axis=0)
    abs_r = r.abs().groupby(month).mean()
    amihud = (abs_r / adv.where(adv > 0)) * 1e9             # mean |r| per ¥1bn traded a day
    s["illiq"] = np.log(amihud.rolling(12, min_periods=6).mean())

    fwd = {k: rm.shift(-k) for k in DECAY}                  # return of month t+k alone
    members = member.loc[month_end.values].set_axis(month_end.index)
    return s, fwd, members, rm


def panel(s, fwd, members):
    months = [p for p in members.index if FIRST_SIGNAL <= p <= LAST_SIGNAL]
    frames = []
    for name, df in {**s, **{f"f{k}": v for k, v in fwd.items()}}.items():
        frames.append(df.loc[months].where(members.loc[months]).stack().rename(name))
    p = pd.concat(frames, axis=1)
    p.index.names = ["month", "code"]
    return p[p.index.get_level_values("code").isin(members.columns)].dropna(subset=["f1"], how="all")


def monthly_ic(p, signal, target):
    def one(g):
        g = g[[signal, target]].dropna()
        return g[signal].rank().corr(g[target].rank()) if len(g) >= MIN_STOCKS else np.nan
    return p.groupby(level="month").apply(one)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    px, liq = load()
    s, fwd, members, rm = build_signals(px, liq)
    p = panel({k: v.astype(float) for k, v in s.items()}, fwd, members)
    p = p[p["f1"].notna()]
    p["ret1"] = np.expm1(p["f1"])
    lines = [f"Panel: {len(p):,} member-months with a next-month return, "
             f"{p.index.get_level_values('month').min()} to {p.index.get_level_values('month').max()} (signal months)"]

    cov = p[SIGNALS].notna().groupby(level="month").sum()
    cov.to_csv(OUT / "coverage.csv")
    lines.append("Members with each signal per month (mean, min): " +
                 ", ".join(f"{c} {cov[c].mean():.0f}/{cov[c].min():.0f}" for c in SIGNALS))

    ic = pd.DataFrame({c: monthly_ic(p, c, "f1") for c in SIGNALS})
    ic.to_csv(OUT / "ic_monthly.csv")

    # persistence: the members' signal ranks against the same stocks' ranks one month later
    persist = {}
    for c in SIGNALS:
        corrs = []
        for mth in ic.index:
            a, b = s[c].loc[mth].where(members.loc[mth]), s[c].loc[mth + 1]
            ok = a.notna() & b.notna()
            corrs.append(a[ok].rank().corr(b[ok].rank()) if ok.sum() >= MIN_STOCKS else np.nan)
        persist[c] = np.nanmean(corrs)

    sigma_cs = p.groupby(level="month")["ret1"].std().median()
    rows, quint = [], []
    for c in SIGNALS:
        mean_ic, _, t_ic = nw_t(ic[c], 3)
        g = p[[c, "ret1"]].dropna()
        g["q"] = g.groupby(level="month")[c].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1)
        qm = g.groupby([pd.Grouper(level="month"), "q"])["ret1"].mean().unstack()
        spread = qm[5] - qm[1]
        sp_mean, _, sp_t = nw_t(spread, 3)
        g["z"] = g.groupby(level="month")[c].transform(zrank)
        slope = g.groupby(level="month").apply(lambda d: np.polyfit(d["z"], d["ret1"], 1)[0])
        fm_mean, _, fm_t = nw_t(slope, 3)
        rows.append({"signal": c, "expected_sign": EXPECTED[c], "months": ic[c].notna().sum(),
                     "mean_ic": mean_ic, "ic_t": t_ic, "ic_std": ic[c].std(),
                     "ic_ir_annual": mean_ic / ic[c].std() * np.sqrt(12), "share_ic_pos": (ic[c] > 0).mean(),
                     "q5_minus_q1_pa": sp_mean * 12, "q5_minus_q1_t": sp_t,
                     "fm_slope_pa": fm_mean * 12, "fm_slope_t": fm_t,
                     "ic_x_sigma_pa": mean_ic * sigma_cs * 12, "persistence": persist[c]})
        quint.append((qm.mean() * 12).rename(c))
    summ = pd.DataFrame(rows).set_index("signal")
    summ.to_csv(OUT / "signal_summary.csv")
    pd.concat(quint, axis=1).T.add_prefix("Q").to_csv(OUT / "quintiles.csv")
    lines.append(f"Median cross-sectional std of next-month returns: {sigma_cs:.2%} a month")
    lines.append("\nMain table (h = 1; returns per year; t-stats Newey-West, 3 lags):")
    lines.append(summ.round(4).to_string())

    sub = []
    for label, a, b in SUBPERIODS:
        # return months a..b, so signal months one earlier
        part = ic[(ic.index >= pd.Period(a, "M") - 1) & (ic.index <= pd.Period(b, "M") - 1)]
        for c in SIGNALS:
            mean_ic, _, t_ic = nw_t(part[c], 3)
            sub.append({"period": label, "signal": c, "months": part[c].notna().sum(), "mean_ic": mean_ic, "t": t_ic})
    sub = pd.DataFrame(sub).pivot(index="signal", columns="period", values=["mean_ic", "t"])
    sub.to_csv(OUT / "subperiods.csv")
    lines.append("\nIC by sub-period (return months):")
    lines.append(sub.round(3).to_string())

    decay = []
    for k in DECAY:
        for c in SIGNALS:
            mean_ic, _, t_ic = nw_t(monthly_ic(p, c, f"f{k}"), k + 2)
            decay.append({"k": k, "signal": c, "mean_ic": mean_ic, "t": t_ic})
    decay = pd.DataFrame(decay).pivot(index="signal", columns="k", values=["mean_ic", "t"])
    decay.to_csv(OUT / "decay.csv")
    lines.append("\nIC with the return of month t+k alone (decay):")
    lines.append(decay.round(3).to_string())

    full = p[SIGNALS + ["ret1"]].dropna()
    z = full[SIGNALS].groupby(level="month").transform(zrank)
    coefs = []
    for mth, zz in z.groupby(level="month"):
        X = np.column_stack([np.ones(len(zz)), zz.to_numpy()])
        coefs.append(np.linalg.lstsq(X, full.loc[zz.index, "ret1"].to_numpy(), rcond=None)[0][1:])
    coefs = pd.DataFrame(coefs, columns=SIGNALS)
    fm = pd.DataFrame([{"signal": c, "slope_pa": nw_t(coefs[c], 3)[0] * 12, "t": nw_t(coefs[c], 3)[2]}
                       for c in SIGNALS]).set_index("signal")
    fm.to_csv(OUT / "fama_macbeth.csv")
    lines.append(f"\nFama-MacBeth, all six signals together ({len(coefs)} months, "
                 f"{len(full) / len(coefs):.0f} stocks a month with all six):")
    lines.append(fm.round(4).to_string())

    rob = {"dy (baseline)": ic["dy"],
           "dy, dividends lagged 2 months": monthly_ic(p, "dy_lag2", "f1"),
           "dy, windows without imputed dividends": monthly_ic(p[p["dy_imputed"] == 0], "dy", "f1"),
           "dy, payers only (dy > 0)": monthly_ic(p[p["dy"] > 0], "dy", "f1")}
    nonpay = p[p["dy"].notna()].assign(payer=lambda d: d["dy"] > 0)
    share = (~nonpay["payer"]).groupby(level="month").mean().mean()
    gap = nonpay.groupby([pd.Grouper(level="month"), "payer"])["ret1"].mean().unstack()
    g_mean, _, g_t = nw_t(gap[True] - gap[False], 3)
    lines.append("\nDividend-yield robustness (h = 1):")
    for label, series in rob.items():
        m_, _, t_ = nw_t(series, 3)
        lines.append(f"  {label}: mean IC {m_:.4f}, t {t_:.2f}, months {series.notna().sum()}")
    lines.append(f"  non-payers: {share:.1%} of member-months; payers minus non-payers {g_mean * 12:.2%} a year (t {g_t:.2f})")
    pd.DataFrame(rob).to_csv(OUT / "dy_robustness_ic.csv")

    corr = sum(p[SIGNALS].loc[mth].rank().corr() for mth in ic.index) / len(ic.index)
    corr.to_csv(OUT / "signal_corr.csv")
    lines.append("\nAverage cross-sectional Spearman correlation between signals:")
    lines.append(corr.round(2).to_string())

    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
