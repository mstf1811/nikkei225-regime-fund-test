"""How often Nikkei 225 members hit the daily price limit in the test period, and whether a Tobit
AR(1) (PFE §3) gives different expected returns from plain OLS on our data.

Inputs: data/prices_daily.parquet and data/price_limits_daily.parquet (pipeline/price_limits.py).
Outputs (analysis/output/price_limits/): summary.txt, censored_by_year.csv, top_days.csv,
top_stocks.csv, tobit_vs_ols.csv.

Tobit check. For each test year 2005-2025: the members on its first trading day, each with its
returns over the previous 5 years (D8, history from before joining included). AR(1) pairs need both
r_t and r_{t-1} (PFE fn. 5), and at least 40 pairs. r_t is upper (lower) censored when the day
closed at the upper (lower) stop price; the censoring point is the observed return itself, since the
close is the stop price. Where a window has no censored day, the Tobit likelihood is the Gaussian
one and its estimate is OLS exactly, so only windows with a censored day are fitted.

Run:  python analysis/price_limits_tobit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import DATA_DIR, PROJECT_DIR  # noqa: E402

OUT = PROJECT_DIR / "analysis" / "output" / "price_limits"
TEST_START, TEST_END = pd.Timestamp("2005-01-01"), pd.Timestamp("2025-09-30")
WINDOW_YEARS = 5
MIN_PAIRS = 40
PFE_L = np.log(1.06)        # BVMT limit used in the PFE
PFE_CENSORED = 0.008        # PFE: ~0.8% of non-zero return days censored
CENSORED = ["close_up", "close_down", "quote_up", "quote_down"]


def load():
    px = pd.read_parquet(DATA_DIR / "prices_daily.parquet",
                         columns=["date", "code", "name", "in_index", "close", "log_return"])
    pl = pd.read_parquet(DATA_DIR / "price_limits_daily.parquet",
                         columns=["date", "code", "base", "limit", "hit", "width_up", "width_down"])
    df = px.merge(pl, on=["date", "code"], how="left")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    df["lag"] = df.groupby("code")["log_return"].shift()
    df["cens"] = np.select([df.hit.isin(["close_up", "quote_up"]), df.hit.isin(["close_down", "quote_down"])],
                           [1, -1], 0)
    return df


def frequency(df, lines):
    m = df[df.in_index & df.date.between(TEST_START, TEST_END) & df.base.notna()].copy()
    m["year"] = m.date.dt.year
    n = len(m)
    cens = m.hit.isin(CENSORED)
    lines.append(f"Member days in the test period with a known base price: {n:,}")
    lines.append(f"Censored (closed at, or quoted at, the stop price): {cens.sum()} = {cens.mean():.4%} "
                 f"(up {m.hit.isin(['close_up', 'quote_up']).sum()}, "
                 f"down {m.hit.isin(['close_down', 'quote_down']).sum()})")
    lines.append(f"  vs the PFE's ~{PFE_CENSORED:.1%} on the BVMT: {PFE_CENSORED / cens.mean():.0f} times rarer")
    nz = m.log_return.notna() & (m.log_return != 0)
    lines.append(f"  share of non-zero return days censored (the PFE's measure): {(cens & nz).sum() / nz.sum():.4%}")
    touch = m.hit.isin(["touch_up", "touch_down"])
    lines.append(f"Touched the stop price intraday but closed inside: {touch.sum()} = {touch.mean():.4%}")
    lines.append(f"Stocks with a censored day: {m.loc[cens, 'code'].nunique()} of {m.code.nunique()}")
    lines.append(f"Days with at least one censored member: {m.loc[cens, 'date'].nunique()} "
                 f"of {m.date.nunique()} trading days")

    by_year = m.groupby("year").agg(member_days=("code", "size"),
                                    up=("hit", lambda s: s.isin(["close_up", "quote_up"]).sum()),
                                    down=("hit", lambda s: s.isin(["close_down", "quote_down"]).sum()))
    by_year["rate"] = (by_year.up + by_year.down) / by_year.member_days
    by_year.to_csv(OUT / "censored_by_year.csv")

    days = m[cens].groupby("date").agg(n=("code", "size"), up=("hit", lambda s: s.str.endswith("up").sum()),
                                       names=("name", lambda s: ", ".join(s.str.title().str[:24].head(6))))
    days = days.sort_values("n", ascending=False)
    days.head(25).to_csv(OUT / "top_days.csv")
    top = days.head(10)
    lines.append("Days with the most censored members:")
    for d, r in top.iterrows():
        lines.append(f"  {d:%Y-%m-%d}: {r.n} ({r.up} up) e.g. {r.names}")
    share_top = days.n.head(10).sum() / days.n.sum()
    lines.append(f"  the 10 busiest days hold {share_top:.0%} of all censored member days")

    st = m[cens].groupby(["code", "name"]).agg(n=("hit", "size"), up=("hit", lambda s: s.str.endswith("up").sum()),
                                                first=("date", "min"), last=("date", "max"))
    st = st.sort_values("n", ascending=False)
    st.head(25).to_csv(OUT / "top_stocks.csv")
    lines.append("Stocks with the most censored days:")
    for (c, nm), r in st.head(10).iterrows():
        lines.append(f"  {c} {nm.title()[:30]}: {r.n} ({r.up} up), {r['first']:%Y-%m} to {r['last']:%Y-%m}")

    # Where the limits sit
    wu, wd = m.width_up, m.width_down
    lines.append(f"Distance to the stop price in log return: up median {wu.median():.3f} (1% quantile "
                 f"{wu.quantile(.01):.3f}), down median {wd.median():.3f} (1% {wd.quantile(.01):.3f}); "
                 f"PFE L = {PFE_L:.4f}")
    r = m.log_return.dropna()
    lines.append(f"Member returns beyond +/-{PFE_L:.4f} (where a BVMT-style limit would bind): "
                 f"{(r.abs() >= PFE_L).mean():.2%}; beyond 10%: {(r.abs() >= 0.10).mean():.3%}")
    # near misses: closes within 1% of the base inside the stop price
    near = m[(m.hit == "") & m.close.notna() & (
        (m.close >= m.base + m.limit - 0.01 * m.base) | (m.close <= m.base - m.limit + 0.01 * m.base))]
    lines.append(f"Closes within 1% of the base price inside a stop price (not counted): {len(near)}")
    return m, by_year


def ols(y, x):
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta[0], beta[1], np.sqrt(resid @ resid / (len(y) - 2))


def tobit(y, x, c, start):
    """Two-sided Tobit AR(1); c = +1 / -1 marks upper / lower censored y (censoring point = y).
    Fitted in percent so the three parameters have similar scales; returns (a, b, sigma, converged)."""
    u, up, lo = c == 0, c == 1, c == -1
    y, x = 100 * y, 100 * x

    def nll(p):
        a, b, ls = p
        s = np.exp(ls)
        z = (y - a - b * x) / s
        ll = (-0.5 * z[u] ** 2 - 0.5 * np.log(2 * np.pi) - ls).sum()
        ll += log_ndtr(-z[up]).sum() + log_ndtr(z[lo]).sum()
        return -ll

    p0 = [100 * start[0], start[1], np.log(100 * start[2])]
    res = minimize(nll, p0, method="BFGS", options={"gtol": 1e-6})
    if not res.success:   # BFGS can stop on precision loss at the optimum: confirm with Nelder-Mead
        res2 = minimize(nll, res.x, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
        res = res2 if res2.fun <= res.fun else res
        res.success = res2.success
    a, b, ls = res.x
    return a / 100, b, np.exp(ls) / 100, bool(res.success)


def tobit_check(df, lines):
    rows = []
    trading_days = df.date.drop_duplicates().sort_values()
    for year in range(TEST_START.year, TEST_END.year + 1):
        tau = trading_days[trading_days >= pd.Timestamp(year, 1, 1)].iloc[0]
        members = df.loc[(df.date == tau) & df.in_index, "code"]
        lo = tau - pd.DateOffset(years=WINDOW_YEARS)
        w = df[df.code.isin(members) & (df.date >= lo) & (df.date < tau)
               & df.log_return.notna() & df.lag.notna()]
        for code, g in w.groupby("code"):
            if len(g) < MIN_PAIRS:
                continue
            y, x, c = g.log_return.to_numpy(), g.lag.to_numpy(), g.cens.to_numpy()
            a, b, s = ols(y, x)
            rec = {"year": year, "code": code, "pairs": len(y), "censored": int((c != 0).sum()),
                   "a_ols": a, "b_ols": b, "s_ols": s, "a_tob": a, "b_tob": b, "s_tob": s, "ok": True}
            if rec["censored"]:
                rec["a_tob"], rec["b_tob"], rec["s_tob"], rec["ok"] = tobit(y, x, c, (a, b, s))
            rows.append(rec)
    t = pd.DataFrame(rows)
    t["mu_ols"] = t.a_ols / (1 - t.b_ols) * 252
    t["mu_tob"] = t.a_tob / (1 - t.b_tob) * 252
    t["dmu"] = t.mu_tob - t.mu_ols
    # standard error of the annualised mean return over the window, for scale
    t["se_mu"] = t.s_ols / np.sqrt(t.pairs) * 252
    t["cs_sd"] = t.groupby("year").mu_ols.transform("std")
    t.to_csv(OUT / "tobit_vs_ols.csv", index=False)

    cw = t[t.censored > 0]
    lines.append(f"Tobit vs OLS, 5-year windows, test years {TEST_START.year}-{TEST_END.year}: "
                 f"{len(t)} stock-windows")
    lines.append(f"  with no censored day (Tobit = OLS exactly): {(t.censored == 0).mean():.1%}")
    lines.append(f"  with a censored day: {len(cw)} ({len(cw) / len(t):.1%}); censored share of their pairs: "
                 f"median {(cw.censored / cw.pairs).median():.3%}, max {(cw.censored / cw.pairs).max():.2%}; "
                 f"failed fits {(~cw.ok).sum()}")
    ad = cw.dmu.abs()
    lines.append(f"  |mu_Tobit - mu_OLS| (annualised): median {ad.median():.4%}, 90% {ad.quantile(.9):.3%}, "
                 f"max {ad.max():.2%}")
    lines.append(f"  as a share of the mean's standard error: median {(ad / cw.se_mu).median():.3f}, "
                 f"max {(ad / cw.se_mu).max():.2f}")
    lines.append(f"  as a share of the cross-sectional sd of mu that year: median {(ad / cw.cs_sd).median():.3f}, "
                 f"max {(ad / cw.cs_sd).max():.2f}")
    lines.append(f"  over all stock-windows: mean |dmu| {t.dmu.abs().mean():.4%} a year; "
                 f"mean dmu {t.dmu.mean():+.4%} a year")
    lines.append(f"  |b_Tobit - b_OLS|: median {(cw.b_tob - cw.b_ols).abs().median():.4f}, "
                 f"max {(cw.b_tob - cw.b_ols).abs().max():.3f}; sigma ratio Tobit/OLS: "
                 f"median {(cw.s_tob / cw.s_ols).median():.4f}, max {(cw.s_tob / cw.s_ols).max():.3f}")
    rank_corr = t.groupby("year").apply(lambda g: g.mu_tob.rank().corr(g.mu_ols.rank()), include_groups=False)
    lines.append(f"  rank correlation of mu across stocks, Tobit vs OLS, by year: min {rank_corr.min():.5f}")
    big = cw.reindex(ad.sort_values(ascending=False).index).head(5)
    lines.append("  largest differences:")
    for r in big.itertuples():
        lines.append(f"    {r.year} {r.code}: {r.censored} censored of {r.pairs}; mu OLS {r.mu_ols:+.2%}, "
                     f"Tobit {r.mu_tob:+.2%}; b {r.b_ols:+.3f} -> {r.b_tob:+.3f}")
    return t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    lines = []
    frequency(df, lines)
    tobit_check(df, lines)
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
