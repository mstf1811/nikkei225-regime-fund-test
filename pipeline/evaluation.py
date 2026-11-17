"""Evaluation toolkit for the JMP backtest (D20). Each function follows a named source.

    metrics          DeMiguel, Garlappi & Uppal (2009): Sharpe ratio, certainty-equivalent return
                     (risk aversion 1), turnover; plus drawdown, Calmar, Sortino and expected
                     shortfall (as Shu, Yu & Mulvey 2024 report)
    nw_mean          Newey & West (1987) t-test of a mean, lags by the Newey-West (1994) rule
    sr_se_hac        Lo (2002) standard error of a Sharpe ratio for non-IID returns (delta method,
                     HAC covariance of the first two moments)
    psr, dsr         Bailey & Lopez de Prado (2012, 2014): probabilistic and deflated Sharpe ratio
    lw_sharpe_test   Ledoit & Wolf (2008): studentized circular block bootstrap test of a Sharpe
                     ratio difference, block size by their calibration (Algorithm 3.1), HAC
                     standard error with the prewhitened QS kernel (Andrews & Monahan 1992)
    holm             Holm (1979) step-down adjustment for a family of tests
    power_*          what the test period can detect (normal approximations; Memmel 2003 for the
                     variance of a Sharpe ratio difference)

All Sharpe ratios inside the tests are per period (monthly); annualise with sqrt(12) for reports.
"""
import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649015329


# ----------------------------------------------------------------------------------------------
# metrics

def to_monthly(daily):
    return (1 + daily).groupby(daily.index.to_period("M")).prod() - 1


def metrics(ret_d, rf_d, weights=None, year_days=245):
    """Performance summary from daily returns (fractions). Sharpe and CEQ from monthly returns."""
    rm, rfm = to_monthly(ret_d), to_monthly(rf_d.reindex(ret_d.index).fillna(0))
    ex = rm - rfm
    nav = (1 + ret_d).cumprod()
    years = len(ret_d) / year_days
    dd = nav / nav.cummax() - 1
    down = ex.clip(upper=0)
    out = {"CAGR": nav.iloc[-1] ** (1 / years) - 1,
           "vol": ret_d.std() * np.sqrt(year_days),
           "Sharpe": ex.mean() / ex.std() * np.sqrt(12),
           "Sortino": ex.mean() / np.sqrt((down ** 2).mean()) * np.sqrt(12),
           "CEQ": (ex.mean() - 0.5 * ex.var()) * 12,          # DeMiguel et al. (2009), gamma = 1
           "MDD": dd.min(),
           "Calmar": (nav.iloc[-1] ** (1 / years) - 1) / abs(dd.min()) if dd.min() < 0 else np.nan,
           "ES 5% (daily)": ret_d[ret_d <= ret_d.quantile(0.05)].mean(),
           "skew (monthly)": stats.skew(ex), "kurtosis (monthly)": stats.kurtosis(ex, fisher=False)}
    if weights is not None:
        out["turnover/yr"] = weights.diff().abs().sum(axis=1).sum() / 2 / years if weights.ndim == 2 \
            else weights.diff().abs().sum() / years
    return out


# ----------------------------------------------------------------------------------------------
# single-series inference

def nw_lags(T):
    return int(np.floor(4 * (T / 100) ** (2 / 9)))


def nw_mean(x, lags=None):
    """Mean, Newey-West standard error, t, one-sided p (H0: mean <= 0)."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    T = len(x)
    lags = nw_lags(T) if lags is None else lags
    e = x - x.mean()
    v = e @ e / T
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / T
    se = np.sqrt(v / T)
    t = x.mean() / se
    return {"mean": x.mean(), "se": se, "t": t, "p_one_sided": 1 - stats.norm.cdf(t), "lags": lags}


def _hac_qs_prewhitened(y):
    """Long-run covariance of the rows of y (T x k), prewhitened VAR(1) + QS kernel with the
    Andrews (1991) AR(1) plug-in bandwidth (Andrews & Monahan 1992), with LW's T/(T-4) factor."""
    T, k = y.shape
    y = y - y.mean(axis=0)
    X, Y = y[:-1], y[1:]
    A = np.linalg.lstsq(X, Y, rcond=None)[0].T                 # y_t = A y_{t-1} + e_t
    eig = np.abs(np.linalg.eigvals(A))
    if eig.max() > 0.97:                                       # Andrews-Monahan eigenvalue guard
        A = A * 0.97 / eig.max()
    e = Y - X @ A.T
    n = len(e)
    num = den = 0.0
    for a in range(k):
        x0, x1 = e[:-1, a], e[1:, a]
        rho = (x0 @ x1) / (x0 @ x0)
        s2 = np.mean((x1 - rho * x0) ** 2)
        num += 4 * rho ** 2 * s2 ** 2 / (1 - rho) ** 8
        den += s2 ** 2 / (1 - rho) ** 4
    alpha2 = num / den if den > 0 else 0.0
    S = 1.3221 * (alpha2 * n) ** 0.2 if alpha2 > 0 else 1.0
    psi = e.T @ e / n
    for j in range(1, n):
        x = 6 * np.pi * (j / S) / 5
        kq = 25 / (12 * np.pi ** 2 * (j / S) ** 2) * (np.sin(x) / x - np.cos(x))
        G = e[j:].T @ e[:-j] / n
        psi += kq * (G + G.T)
    B = np.linalg.inv(np.eye(k) - A)
    return B @ psi @ B.T * T / (T - 4)


def sr_se_hac(x):
    """Lo (2002) non-IID standard error of the per-period Sharpe ratio (delta method + HAC)."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    T = len(x)
    a, c = x.mean(), (x ** 2).mean()
    grad = np.array([c / (c - a * a) ** 1.5, -0.5 * a / (c - a * a) ** 1.5])
    psi = _hac_qs_prewhitened(np.column_stack([x, x ** 2]))
    return a / np.sqrt(c - a * a), float(np.sqrt(grad @ psi @ grad / T))


def psr(sr, T, skew, kurt, sr0=0.0):
    """Probabilistic Sharpe ratio: P(true SR > sr0); per-period sr; kurtosis not in excess."""
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) /
                                np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)))


def dsr(sr, T, skew, kurt, trial_srs):
    """Deflated Sharpe ratio: the PSR at the expected maximum Sharpe ratio of N unskilled trials."""
    trial_srs = np.asarray(trial_srs, dtype=float)
    N = len(trial_srs)
    if N < 2:
        return psr(sr, T, skew, kurt, 0.0), 0.0
    v = trial_srs.var(ddof=1)
    sr0 = np.sqrt(v) * ((1 - EULER) * stats.norm.ppf(1 - 1 / N) + EULER * stats.norm.ppf(1 - 1 / (N * np.e)))
    return psr(sr, T, skew, kurt, sr0), float(sr0)


# ----------------------------------------------------------------------------------------------
# Ledoit & Wolf (2008)

def _delta_and_grad(r1, r2):
    a, b = r1.mean(axis=-1), r2.mean(axis=-1)
    c, d = (r1 ** 2).mean(axis=-1), (r2 ** 2).mean(axis=-1)
    delta = a / np.sqrt(c - a * a) - b / np.sqrt(d - b * b)
    grad = np.stack([c / (c - a * a) ** 1.5, -d / (d - b * b) ** 1.5,
                     -0.5 * a / (c - a * a) ** 1.5, 0.5 * b / (d - b * b) ** 1.5], axis=-1)
    return delta, grad


def _boot_stats(r1, r2, b, M, rng):
    """M circular-block bootstrap draws: the studentized |delta* - delta| / s(delta*)."""
    T = len(r1)
    l = T // b
    starts = rng.integers(0, T, size=(M, l))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % T     # M x l x b
    x1, x2 = r1[idx], r2[idx]
    flat1, flat2 = x1.reshape(M, -1), x2.reshape(M, -1)
    delta_s, grad = _delta_and_grad(flat1, flat2)
    y = np.stack([x1, x2, x1 ** 2, x2 ** 2], axis=-1)                # M x l x b x 4
    y = y - y.reshape(M, -1, 4).mean(axis=1)[:, None, None, :]
    f = y.sum(axis=2) / np.sqrt(b)                                    # M x l x 4
    psi = np.einsum("mlk,mlj->mkj", f, f) / l
    se = np.sqrt(np.einsum("mk,mkj,mj->m", grad, psi, grad) / (l * b))
    return delta_s, se


def lw_sharpe_test(r1, r2, blocks=(1, 2, 4, 6, 8, 10), M=4999, K=500, alpha=0.05, seed=0):
    """Two-sided test of H0: SR1 = SR2 (per-period excess returns). Returns the difference, the HAC
    standard error, the calibrated block size and the bootstrap p-value (LW eq. 9)."""
    rng = np.random.default_rng(seed)
    r1, r2 = np.asarray(r1, float), np.asarray(r2, float)
    T = len(r1)
    delta, grad = _delta_and_grad(r1, r2)
    psi = _hac_qs_prewhitened(np.column_stack([r1, r2, r1 ** 2, r2 ** 2]))
    se = float(np.sqrt(grad @ psi @ grad / T))

    # calibration (Algorithm 3.1): VAR(1) fitted to the pair, residuals resampled by the stationary
    # bootstrap (mean block 5); pick b whose bootstrap interval covers the truth closest to 1-alpha
    Y = np.column_stack([r1, r2])
    X = np.column_stack([np.ones(T - 1), Y[:-1]])
    B = np.linalg.lstsq(X, Y[1:], rcond=None)[0]
    res = Y[1:] - X @ B
    pseudo = []
    for _ in range(K):
        # stationary bootstrap of the residuals (Politis & Romano 1994), then the fitted VAR(1)
        out, pos = np.empty((T + 50, 2)), rng.integers(0, len(res))
        for t in range(T + 50):
            if rng.random() < 1 / 5:
                pos = rng.integers(0, len(res))
            out[t] = res[pos]
            pos = (pos + 1) % len(res)
        ysim = np.empty((T + 50, 2))
        ysim[0] = Y.mean(axis=0)
        for t in range(1, T + 50):
            ysim[t] = B[0] + ysim[t - 1] @ B[1:] + out[t]
        ysim = ysim[50:]
        d_k, g_k = _delta_and_grad(ysim[:, 0], ysim[:, 1])
        psi_k = _hac_qs_prewhitened(np.column_stack([ysim, ysim ** 2]))
        pseudo.append((ysim, d_k, np.sqrt(g_k @ psi_k @ g_k / T)))
    cover = {}
    for b in blocks:
        hits = 0
        for ysim, d_k, se_k in pseudo:
            ds, ses = _boot_stats(ysim[:, 0], ysim[:, 1], b, 199, rng)
            q = np.quantile(np.abs(ds - d_k) / ses, 1 - alpha)
            hits += abs(d_k - delta) <= q * se_k
        cover[b] = hits / K
    b_star = min(blocks, key=lambda b: abs(cover[b] - (1 - alpha)))

    ds, ses = _boot_stats(r1, r2, b_star, M, rng)
    d0 = abs(delta) / se
    p = (np.sum(np.abs(ds - delta) / ses >= d0) + 1) / (M + 1)
    return {"delta": float(delta), "se_hac": se, "p_hac": float(2 * stats.norm.cdf(-d0)),
            "block": b_star, "coverage": cover, "p_boot": float(p)}


def holm(pvals):
    """Holm (1979) adjusted p-values for a dict {name: p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


# ----------------------------------------------------------------------------------------------
# power (normal approximations)

def power_min_sr(years, alpha=0.05, power=0.8):
    """Smallest annual Sharpe ratio detected with the given power by a one-sided test."""
    return (stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)) / np.sqrt(years)


def power_min_sr_diff(years, rho, sr=0.5, alpha=0.05, power=0.8):
    """Smallest annual Sharpe difference detected (two-sided) for strategies with correlation rho
    and Sharpe ratios around sr (Memmel 2003 variance, IID normal)."""
    m = sr / np.sqrt(12)
    T = years * 12
    var = (2 - 2 * rho + 0.5 * (m ** 2 + m ** 2 - 2 * m * m * rho ** 2)) / T
    return (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * np.sqrt(var) * np.sqrt(12)
