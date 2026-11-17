"""Regime detection (E1, D17): the statistical jump model of Shu, Yu & Mulvey (2024) on ^N225 excess
returns, with the paper's 8-year cross-validation of the jump penalty, and the paper's HMM benchmark
(the D20 (8) sensitivity). Ported from analysis/regime_jm.py with the data end as a parameter;
bt_tests.py checks it reproduces the replication's labels and lambda path to 2004.

Everything is causal (D1): a fit uses the 3,000 days before its period start, a day's online label
uses data up to that day, and the lambda used in month p is chosen on the 8 years to the end of p-1.
Results are cached in data/backtest/cache/ by data end (and this file's hash).

Output (RegimeResult):
    f          ^N225 frame: r, rf, x (excess), logx, the three features
    labels     the cross-validated online label, 1 = bear, from 1990-01 (JM, or HMM if asked)
    path       month -> chosen lambda (or HMM k) and its validation Sharpe
    fits       every JM fit: period start, lambda, training bull/bear means and vols (daily)
    by_month   month -> the D17 numbers of the fit in force: bull_mean, bear_mean, bull_vol
"""
import hashlib
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from bt_data import YEAR_DAYS, load_rate
from config import DATA_DIR

warnings.filterwarnings("ignore")
CACHE = DATA_DIR / "backtest" / "cache"
FIRST_REFIT = "1982-01-01"
OOS_START = "1990-01-01"
TRAIN_DAYS = 3000
VALID_YEARS = 8
COST = 0.0010
BURN_IN = 120
N_INIT = 10
LAMBDAS = [0.0] + list(np.round(np.logspace(0, np.log10(150), 11), 2))
HMM_K = [1, 3, 6, 10, 15, 21]
SEED = 20261012
FEATURES = ["DD10", "Sortino20", "Sortino60"]


def load_index(end):
    n = pd.read_csv(DATA_DIR / "market" / "nikkei225_index.csv", parse_dates=["date"])
    n = n[n["date"] <= pd.Timestamp(end)].sort_values("date").set_index("date")["close"]
    r = n.pct_change().dropna()
    rate = load_rate(r.index)
    gap = pd.Series(r.index, index=r.index).diff().dt.days.fillna(1)
    rf = rate.shift(1).bfill() / 100 * gap / 365
    x = r - rf
    f = pd.DataFrame({"r": r, "rf": rf, "x": x, "logx": np.log1p(r) - np.log1p(rf)})
    down = (x.clip(upper=0) ** 2)
    dd = {h: np.sqrt(down.ewm(halflife=h).mean()) for h in (10, 20, 60)}
    f["DD10"] = dd[10]
    f["Sortino20"] = x.ewm(halflife=20).mean() / dd[20]
    f["Sortino60"] = x.ewm(halflife=60).mean() / dd[60]
    return f.iloc[BURN_IN:]


def standardise(train, other):
    mu, sd = train.mean(), train.std()
    z = lambda a: ((a - mu) / sd).clip(-3, 3).to_numpy()  # noqa: E731
    return z(train), z(other)


# ----------------------------------------------------------------------------------------------
# jump model

def dp_states(loss, lam):
    T, K = loss.shape
    V = np.empty((T, K))
    back = np.empty((T, K), dtype=np.int64)
    V[0] = loss[0]
    for t in range(1, T):
        prev = V[t - 1]
        j = int(prev.argmin())
        move = prev[j] + lam
        back[t] = np.where(prev <= move, np.arange(K), j)
        V[t] = loss[t] + np.minimum(prev, move)
    s = np.empty(T, dtype=np.int64)
    s[-1] = int(V[-1].argmin())
    for t in range(T - 1, 0, -1):
        s[t - 1] = back[t, s[t]]
    return s, float(V[-1].min())


def forward_labels(loss, lam):
    T, K = loss.shape
    V = loss[0].copy()
    out = np.empty(T, dtype=np.int64)
    out[0] = int(V.argmin())
    for t in range(1, T):
        V = loss[t] + np.minimum(V, V.min() + lam)
        out[t] = int(V.argmin())
    return out


def sq_loss(X, theta):
    return 0.5 * ((X[:, None, :] - theta[None, :, :]) ** 2).sum(axis=2)


def kmeanspp(X, K, rng):
    c = [X[rng.integers(len(X))]]
    for _ in range(1, K):
        d = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(axis=2), axis=1)
        c.append(X[rng.choice(len(X), p=d / d.sum())])
    return np.array(c)


def fit_jm(X, lam, K=2, n_init=N_INIT, max_iter=100, seed=SEED):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        theta = kmeanspp(X, K, rng)
        s_prev = None
        for _ in range(max_iter):
            s, val = dp_states(sq_loss(X, theta), lam)
            if s_prev is not None and np.array_equal(s, s_prev):
                break
            for k in range(K):
                if (s == k).any():
                    theta[k] = X[s == k].mean(axis=0)
            s_prev = s
        if best is None or val < best[0]:
            best = (val, theta.copy(), s.copy())
    return best[1], best[2]


def refit_dates(index, end):
    years = range(pd.Timestamp(FIRST_REFIT).year, pd.Timestamp(end).year + 1)
    dates = [pd.Timestamp(f"{y}-{m:02d}-01") for y in years for m in (1, 7)]
    return [d for d in dates if pd.Timestamp(FIRST_REFIT) <= d <= index[-1]]


def jm_period(f, start, end, lam):
    """Fit on the 3,000 days before `start`; online labels for [start, end). 0 = bull."""
    pos = f.index.searchsorted(start)
    tr = f.iloc[max(0, pos - TRAIN_DAYS):pos]
    ap = f.iloc[pos:f.index.searchsorted(end)]
    Xtr, Xall = standardise(tr[FEATURES], pd.concat([tr, ap])[FEATURES])
    theta, s = fit_jm(Xtr, lam)
    cum = [tr["x"][s == k].sum() for k in range(2)]
    order = np.argsort(cum)[::-1]
    lab = forward_labels(sq_loss(Xall, theta[order]), lam)[len(tr):]
    ins = order.argsort()[s]
    stats = {"start": start, "lam": lam, "train_days": len(tr),
             "bull_mean": tr["x"][ins == 0].mean(), "bear_mean": tr["x"][ins == 1].mean(),
             "bull_vol": tr["x"][ins == 0].std(), "bear_vol": tr["x"][ins == 1].std(),
             "bear_share": (ins == 1).mean(), "switches": int((np.diff(ins) != 0).sum())}
    return pd.Series(lab, index=ap.index), stats


def jm_all(f, lam, end):
    ds = refit_dates(f.index, end) + [pd.Timestamp(end) + pd.Timedelta(days=1)]
    parts, stats = [], []
    for a, b in zip(ds[:-1], ds[1:]):
        lab, st = jm_period(f, a, b, lam)
        parts.append(lab)
        stats.append(st)
    return pd.concat(parts), stats


# ----------------------------------------------------------------------------------------------
# HMM benchmark

def hmm_period(f, start, end, k_list):
    from hmmlearn.hmm import GaussianHMM
    pos = f.index.searchsorted(start)
    tr = f.iloc[max(0, pos - TRAIN_DAYS):pos]
    ap = f.iloc[pos:f.index.searchsorted(end)]
    y = tr[["logx"]].to_numpy()
    best = None
    for i in range(N_INIT):
        m = GaussianHMM(2, covariance_type="diag", n_iter=200, random_state=SEED + i)
        try:
            m.fit(y)
            sc = m.score(y)
        except Exception:
            continue
        if best is None or sc > best[0]:
            best = (sc, m)
    m = best[1]
    bear = int(np.argmax(m.covars_.ravel()))
    yall = pd.concat([tr, ap])[["logx"]].to_numpy()
    logb = np.column_stack([-0.5 * np.log(2 * np.pi * m.covars_.ravel()[k])
                            - 0.5 * (yall[:, 0] - m.means_.ravel()[k]) ** 2 / m.covars_.ravel()[k]
                            for k in range(2)])
    logA = np.log(m.transmat_)
    d = np.log(m.startprob_ + 1e-300) + logb[0]
    last = np.empty(len(yall), dtype=np.int64)
    last[0] = d.argmax()
    for t in range(1, len(yall)):
        d = (d[:, None] + logA).max(axis=0) + logb[t]
        d -= d.max()
        last[t] = d.argmax()
    raw = pd.Series((last == bear).astype(float), index=pd.concat([tr, ap]).index)
    return {k: (raw.rolling(k).mean() > 0.5).astype(int).loc[ap.index] for k in k_list}


def hmm_all(f, end):
    ds = pd.date_range(FIRST_REFIT, end, freq="MS").tolist() + [pd.Timestamp(end) + pd.Timedelta(days=1)]
    res = Parallel(n_jobs=-1)(delayed(hmm_period)(f, a, b, HMM_K) for a, b in zip(ds[:-1], ds[1:]))
    return {k: pd.concat([r[k] for r in res]) for k in HMM_K}


# ----------------------------------------------------------------------------------------------
# the 0/1 strategy and the cross-validation

def strategy(f, bear, delay=1):
    w = (1 - bear).shift(delay + 1)
    w = w.reindex(f.index)
    ret = w * f["r"] + (1 - w) * f["rf"] - COST * w.diff().abs()
    return ret.dropna(), w


def sharpe(ret, rf):
    ex = ret - rf.reindex(ret.index)
    return ex.mean() / ex.std() * np.sqrt(YEAR_DAYS) if ex.std() > 0 else -np.inf


def cross_validate(f, labels_by):
    rets = {k: strategy(f, lab)[0] for k, lab in labels_by.items()}
    month_ends = f.loc[OOS_START:].index.to_series().groupby(f.loc[OOS_START:].index.to_period("M")).first()
    chosen, pieces = [], []
    for p, first_day in month_ends.items():
        cut = f.index[f.index.searchsorted(first_day) - 1]
        lo = cut - pd.DateOffset(years=VALID_YEARS)
        scores = {k: sharpe(r.loc[lo:cut], f["rf"]) for k, r in rets.items()}
        best = max(scores, key=scores.get)
        chosen.append((str(p), best, scores[best]))
        nxt = (p + 1).to_timestamp()
        pieces.append(labels_by[best].loc[first_day:nxt - pd.Timedelta(days=1)])
    return pd.concat(pieces), pd.DataFrame(chosen, columns=["month", "choice", "valid_sharpe"])


# ----------------------------------------------------------------------------------------------

@dataclass
class RegimeResult:
    kind: str
    f: pd.DataFrame
    labels: pd.Series
    path: pd.DataFrame
    fits: pd.DataFrame
    by_month: pd.DataFrame


def _d17_by_month(path, fits):
    """For each month p: the fit in force (its 6-month period start) at the lambda chosen for p."""
    rows = []
    for m, lam in zip(path["month"], path["choice"]):
        p = pd.Period(m, "M")
        start = pd.Timestamp(f"{p.year}-{1 if p.month <= 6 else 7:02d}-01")
        row = fits[(fits["start"] == start) & np.isclose(fits["lam"], lam)].iloc[0]
        rows.append({"month": m, "lam": lam, "bull_mean": row["bull_mean"], "bear_mean": row["bear_mean"],
                     "bull_vol": row["bull_vol"]})
    return pd.DataFrame(rows).set_index("month")


def _source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def run(end, kind="jm", cache=True):
    """Labels and D17 numbers to `end`. kind: 'jm' (B0) or 'hmm' (the D20 (8) sensitivity; the D17
    numbers still come from the jump model)."""
    end = pd.Timestamp(end)
    path_c = CACHE / f"regime_{kind}_{end:%Y%m%d}_{_source_hash()}.pkl"
    if cache and path_c.exists():
        return pickle.loads(path_c.read_bytes())
    f = load_index(end)
    res = Parallel(n_jobs=-1)(delayed(jm_all)(f, lam, end) for lam in LAMBDAS)
    jm_by = {lam: r[0] for lam, r in zip(LAMBDAS, res)}
    fits = pd.DataFrame([s for r in res for s in r[1]])
    labels, path = cross_validate(f, jm_by)
    by_month = _d17_by_month(path, fits)
    if kind == "hmm":
        labels, path = cross_validate(f, hmm_all(f, end))
    out = RegimeResult(kind, f, labels, path, fits, by_month)
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        path_c.write_bytes(pickle.dumps(out))
    return out
