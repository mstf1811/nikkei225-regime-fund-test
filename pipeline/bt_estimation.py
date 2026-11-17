"""Monthly estimation for the optimizer (E5): beta (D15 (2)), covariance (D16), the Fama-MacBeth
signal slopes and their empirical-Bayes shrinkage (D15 (4)), alpha (D14, D15 (7), D18 (6)), and the
C5 comparison model (D14's baseline: the shrunk historical mean per stock, the PFE without the Tobit).

Ported from analysis/optimizer_check.py; bt_tests.py checks the 2005-01 inputs against it.
Every function takes a month-end date and reads data up to that date only.
"""
import numpy as np
import pandas as pd

from bt_data import MONTH_DAYS
from bt_signals import zrank

WINDOW_YEARS = 5                 # D8
MIN_BETA_DAYS = 250              # D15 (2)
COVERAGE = 0.90                  # a full 5-year window, else single-index rows (D16 (3))
FM_MIN_STOCKS = 50
FM_WINDOW = 60                   # D15 (4)
FM_MIN_MONTHS = 12               # [ours] fewer slope months -> gamma = 0 (never binds from 2005-01)
ALPHA_TRIM = 3.0                 # D18 (6)


# ----------------------------------------------------------------------------------------------
# beta (D15 (2))

def ols_beta(y, x):
    """Per-column OLS slope of y on x (both excess), on each column's own days, with its s.e."""
    ok = y.notna()
    xm = ok.mul(x, axis=0).where(ok)
    n = ok.sum()
    sx, sy = xm.sum(), y.sum()
    sxx = (xm ** 2).sum() - sx ** 2 / n
    sxy = (xm * y).sum() - sx * sy / n
    syy = (y ** 2).sum() - sy ** 2 / n
    b = sxy / sxx
    resid = (syy - b * sxy) / (n - 2)
    return b.where(n >= MIN_BETA_DAYS), np.sqrt(resid / sxx).where(n >= MIN_BETA_DAYS), resid, n


def vasicek(b, se):
    ok = b.notna() & se.notna()
    prior_var = max(b[ok].var() - (se[ok] ** 2).mean(), 1e-4)
    return (prior_var * b + se ** 2 * b[ok].mean()) / (prior_var + se ** 2)


def window(simple, rf, date):
    lo = date - pd.DateOffset(years=WINDOW_YEARS)
    return (simple.index > lo) & (simple.index <= date)


def betas_at(simple, rf, mkt, date, codes):
    idx = window(simple, rf, date)
    y = simple.loc[idx, codes].sub(rf[idx], axis=0)
    x = (mkt - rf).loc[y.index]
    b, se, resid, n = ols_beta(y, x)
    return vasicek(b, se), b, resid, n


# ----------------------------------------------------------------------------------------------
# covariance (D16)

def cov_market(x):
    """Ledoit & Wolf (2003) shrinkage toward the single-index target (their covMarket code)."""
    t, n = x.shape
    x = x - x.mean(axis=0)
    xm = x.mean(axis=1, keepdims=True)
    full = np.cov(np.hstack([x, xm]), rowvar=False) * (t - 1) / t
    covm, varm = full[:n, n], full[n, n]
    sample = full[:n, :n]
    prior = np.outer(covm, covm) / varm
    np.fill_diagonal(prior, np.diag(sample))
    c = np.linalg.norm(sample - prior, "fro") ** 2
    y = x ** 2
    p = (y.T @ y).sum() / t - (sample ** 2).sum()
    rdiag = (y ** 2).sum() / t - (np.diag(sample) ** 2).sum()
    z = x * xm
    v1 = y.T @ z / t - covm[:, None] * sample
    roff1 = (v1 * covm[None, :]).sum() / varm - (np.diag(v1) * covm).sum() / varm
    v3 = z.T @ z / t - varm * sample
    roff3 = (v3 * np.outer(covm, covm)).sum() / varm ** 2 - (np.diag(v3) * covm ** 2).sum() / varm ** 2
    r = rdiag + 2 * roff1 - roff3
    k = (p - r) / c
    delta = max(0.0, min(1.0, k / t))
    return delta * prior + (1 - delta) * sample, delta


def covariance(simple, rf, mkt, date, codes, beta_ols, resid):
    """Daily covariance of excess returns (D16); members without a full window get single-index
    rows; repaired to positive definite if needed."""
    idx = window(simple, rf, date)
    y = simple.loc[idx, codes].sub(rf[idx], axis=0)
    y = y[y.notna().any(axis=1)]
    cover = y.notna().mean()
    full = cover[cover >= COVERAGE].index
    short = [c for c in codes if c not in full]
    S, delta = cov_market(y[full].fillna(0.0).to_numpy())
    sig = pd.DataFrame(S, index=full, columns=full)
    if short:
        m = (mkt - rf).loc[y.index]
        vm = m.var()
        b_all = beta_ols.reindex(codes)
        bvec = pd.Series({k: (b_all[k] if pd.notna(b_all[k]) else 1.0) for k in codes})
        sig = sig.reindex(index=codes, columns=codes)
        for c in short:
            bc = bvec[c]
            rv = resid[c] if pd.notna(resid.get(c, np.nan)) else y[c].var()
            sig.loc[c, :] = bc * vm * bvec
            sig.loc[:, c] = sig.loc[c, :]
            sig.loc[c, c] = bc ** 2 * vm + rv
    sig = sig.loc[codes, codes]
    sig = (sig + sig.T) / 2
    ev = np.linalg.eigvalsh(sig.to_numpy())
    if ev.min() <= 0:
        vals, vecs = np.linalg.eigh(sig.to_numpy())
        sig = pd.DataFrame(vecs @ np.diag(np.maximum(vals, 1e-10)) @ vecs.T, index=codes, columns=codes)
    return sig, delta, len(full), short


# ----------------------------------------------------------------------------------------------
# signal slopes (D15 (4)) and alpha

class Monthly:
    """Monthly totals of the daily panels, for the Fama-MacBeth dependent variable."""

    def __init__(self, data):
        month = data.dates.to_period("M")
        self.gross = (1 + data.simple).groupby(month).prod(min_count=1) - 1
        self.rf = (1 + data.rf).groupby(month).prod() - 1
        self.mkt = (1 + data.mkt).groupby(month).prod() - 1


def fm_slope(m, signals, members_me, beta_m, monthly, names):
    """The cross-sectional slope of signal month m: next month's market-adjusted return on the
    signals' rank z-scores (members at the end of m; beta known at the end of m). None if a
    signal covers fewer than 50 members or the month has fewer than 50 returns."""
    codes = members_me.loc[m][members_me.loc[m]].index
    z = pd.DataFrame({k: signals[k].loc[m, codes] for k in names}).apply(zrank)
    if z.notna().sum().min() < FM_MIN_STOCKS:
        return None
    nxt = m + 1
    bv = beta_m.reindex(codes)
    y = monthly.gross.loc[nxt, codes] - monthly.rf.loc[nxt] - bv * (monthly.mkt.loc[nxt] - monthly.rf.loc[nxt])
    d = z.fillna(0.0).assign(y=y).dropna()
    if len(d) < FM_MIN_STOCKS:
        return None
    X = np.column_stack([np.ones(len(d)), d[names].to_numpy()])
    coef = np.linalg.lstsq(X, d["y"].to_numpy(), rcond=None)[0]
    return pd.Series(coef[1:], index=names, name=m)


def eb_shrink(mean, se):
    """Empirical Bayes toward 0 across the signals (D15 (4))."""
    tau2 = max(0.0, (mean ** 2).mean() - (se ** 2).mean())
    return mean * tau2 / (tau2 + se ** 2), tau2


def gamma_rolling(slopes):
    """D15 (4) as written: mean of the last 60 slope months, then empirical Bayes."""
    g = slopes.iloc[-FM_WINDOW:]
    if len(g) < FM_MIN_MONTHS:
        return pd.Series(0.0, index=slopes.columns), 0.0
    gbar, se = g.mean(), g.std(ddof=1) / np.sqrt(len(g))
    return eb_shrink(gbar, se)


def alpha(signals, m, codes, gamma):
    """alpha = z . gamma on the month-end members' rank z-scores (missing signal -> 0), trimmed at
    +-3 cross-sectional standard deviations (D18 (6)). Monthly units."""
    z = pd.DataFrame({k: signals[k].loc[m, codes] for k in gamma.index}).apply(zrank).fillna(0.0)
    a = z @ gamma
    sd = a.std()
    return a.clip(-ALPHA_TRIM * sd, ALPHA_TRIM * sd) if sd > 0 else a, a


# ----------------------------------------------------------------------------------------------
# C5: the shrunk historical mean per stock (the PFE without the Tobit; PFE ch. 5.3, our sec. 3.2-3.6)

def hist_mean(r, rf, date, codes):
    """Per stock, over the 5-year window (D8, chained returns): OLS AR(1) on consecutive non-missing
    daily log returns (>= 40 pairs), mu = a / (1 - b); James-Stein toward the mean of the strongly
    identified stocks (|t_b| >= 1.96) with w = t_b^2 / (t_b^2 + 1) for the others; capped above at
    the mean of the annualised values beyond the 95th percentile, and [ours, shorts are allowed]
    below at the mean of those under the 5th. Returns monthly excess returns (log mean minus the
    window's mean cash return, x 245/12, as the PFE feeds log means to the optimizer)."""
    idx = window(r, rf, date)
    R = r.loc[idx, codes]
    rows = {}
    for c in codes:
        x = R[c].to_numpy()
        ok = ~np.isnan(x[1:]) & ~np.isnan(x[:-1])
        if ok.sum() < 40:
            continue
        y0, y1 = x[:-1][ok], x[1:][ok]
        X = np.column_stack([np.ones(len(y0)), y0])
        coef, *_ = np.linalg.lstsq(X, y1, rcond=None)
        e = y1 - X @ coef
        s2 = e @ e / (len(y1) - 2)
        se_b = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
        a, b = coef
        if abs(b) >= 1:
            continue
        rows[c] = (a / (1 - b), b / se_b)
    df = pd.DataFrame(rows, index=["mu", "t"]).T
    strong = df["t"].abs() >= 1.96
    grand = df.loc[strong, "mu"].mean() if strong.any() else df["mu"].mean()
    w = np.where(strong, 1.0, df["t"] ** 2 / (df["t"] ** 2 + 1))
    mu = w * df["mu"] + (1 - w) * grand
    ann = mu * 245
    hi = ann[ann > ann.quantile(0.95)].mean() / 245
    lo = ann[ann < ann.quantile(0.05)].mean() / 245
    mu = mu.clip(lo, hi)
    rf_d = np.log1p(rf[idx]).mean()
    return ((mu - rf_d) * MONTH_DAYS).reindex(codes).fillna(0.0)
