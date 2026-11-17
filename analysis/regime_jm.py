"""Regime detection (E1) as in the recent literature: the statistical jump model (JM) of
Shu, Yu & Mulvey (2024, J. Asset Management 25(5) 493-507; arXiv 2402.05272), replicated on data
from BEFORE the test period only (Nikkei 225, 1970-2004; out-of-sample 1990-2004).

The paper tests exactly this on the Nikkei 225 (with the S&P 500 and DAX), out of sample
1990-2023. Every setting below is the paper's unless marked [ours]; the 2005-2025 test period is
not touched.

Data
    ^N225 daily close (Yahoo, price index) [ours: the paper uses a total-return index; the
    Nikkei's dividend yield (about 0.5-2%/yr) is 0.002-0.008%/day, negligible for these features].
    Risk-free: BOJ call rate, daily from 1998, monthly averages before (pipeline/market.py)
    [ours: the paper uses a Japanese 3-month T-bill from GFD, which is paid data].
    Excess return x_t = r_t - rf_t (simple daily returns; rf accrues over calendar days).

Features (the paper's three, from the excess returns, exponentially weighted)
    DD(10)        downside deviation, halflife 10: sqrt(EWM of x^2 * 1{x<0})
    Sortino(20)   EWM mean / EWM downside deviation, halflife 20
    Sortino(60)   the same, halflife 60
    Standardised on each training window [ours: z-score, then clipped at +-3; the paper says
    "standardized" without details].

Jump model (K = 2 states)
    minimise  sum_t 1/2 ||x_t - theta_{s_t}||^2  +  lambda * sum_t 1{s_t != s_t-1}
    by coordinate descent: dynamic programming for the states, cluster means for theta; 10 starts
    (k-means++ centres), keeping the lowest objective. The state with the higher cumulative excess
    return over the training window is "bull".
    Training window: the last 3,000 trading days, refitted every 6 months (1 January, 1 July).
    Online labels (causal, D1): with the frozen parameters, a forward DP from the window start;
    the day's label is the state that ends the cheapest path through that day.

Jump penalty lambda (time-series cross-validation, the paper's)
    Every month-end, for each lambda on the grid, the online labels over the last 8 years drive
    the "0/1 strategy" (100% index when bull, 100% cash when bear; signal at the close of t is
    traded at the close of t+1, so it earns from t+2; 10 bp one-way cost). The lambda with the
    highest Sharpe ratio is used for the next month. Grid [ours: the paper does not give it]:
    0 and 11 log-spaced values from 1 to 150 (the range of the paper's Table 3).

HMM benchmark (the paper's)
    2-state Gaussian HMM on daily log excess returns, 3,000-day window; label = last state of the
    Viterbi path; a "bear" signal when the mean of the last k labels exceeds 0.5, with k chosen
    by the same cross-validation (grid 1, 3, 6, 10, 15, 21). Bear = the high-volatility state.
    [ours: refitted monthly, not daily, to keep the run short.]

Outputs (analysis/output/regime_jm/): summary.txt, labels.parquet, lambda_path.csv,
    regime_means.csv, fig_regimes.png

Run:  python analysis/regime_jm.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import DATA_DIR, PROJECT_DIR  # noqa: E402

warnings.filterwarnings("ignore")
OUT = PROJECT_DIR / "analysis" / "output" / "regime_jm"
DATA_END = "2004-12-31"
FIRST_REFIT = "1982-01-01"        # online labels from 1982: 8 validation years before 1990
OOS_START = "1990-01-01"          # the paper's out-of-sample start
TRAIN_DAYS = 3000
VALID_YEARS = 8
COST = 0.0010                     # one-way, on each change of the index weight
YEAR_DAYS = 245                   # trading days a year in Japan (243-251 in 1970-2004)
BURN_IN = 120                     # days dropped before the features are trusted
N_INIT = 10
LAMBDAS = [0.0] + list(np.round(np.logspace(0, np.log10(150), 11), 2))
HMM_K = [1, 3, 6, 10, 15, 21]
SEED = 20261012


# ----------------------------------------------------------------------------------------------
# data and features

def load_rf(dates):
    """Call rate in % p.a. on each date: the daily series from 1998, monthly averages before."""
    d = pd.read_csv(DATA_DIR / "market" / "call_rate_daily.csv", parse_dates=["date"])
    m = pd.read_csv(DATA_DIR / "market" / "call_rate_monthly.csv")
    month = pd.Series(dates.strftime("%Y-%m"), index=dates)
    rate = month.map(m.set_index("month")["call_rate"]).astype(float)
    daily = d.set_index("date")["call_rate"].reindex(dates).ffill()
    rate[dates >= d["date"].min()] = daily[dates >= d["date"].min()]
    return rate


def load():
    n = pd.read_csv(DATA_DIR / "market" / "nikkei225_index.csv", parse_dates=["date"])
    n = n[n["date"] <= DATA_END].sort_values("date").set_index("date")["close"]
    r = n.pct_change().dropna()
    rate = load_rf(r.index)
    gap = pd.Series(r.index, index=r.index).diff().dt.days.fillna(1)
    rf = rate.shift(1).bfill() / 100 * gap / 365          # accrues at the previous day's rate
    x = r - rf
    f = pd.DataFrame({"r": r, "rf": rf, "x": x, "logx": np.log1p(r) - np.log1p(rf)})
    down = (x.clip(upper=0) ** 2)
    dd = {h: np.sqrt(down.ewm(halflife=h).mean()) for h in (10, 20, 60)}
    f["DD10"] = dd[10]
    f["Sortino20"] = x.ewm(halflife=20).mean() / dd[20]
    f["Sortino60"] = x.ewm(halflife=60).mean() / dd[60]
    return f.iloc[BURN_IN:]


FEATURES = ["DD10", "Sortino20", "Sortino60"]


def standardise(train, other):
    mu, sd = train.mean(), train.std()
    z = lambda a: ((a - mu) / sd).clip(-3, 3).to_numpy()
    return z(train), z(other)


# ----------------------------------------------------------------------------------------------
# jump model

def dp_states(loss, lam):
    """Optimal state path for a (T, K) loss matrix with a jump penalty lam (Viterbi-like)."""
    T, K = loss.shape
    V = np.empty((T, K))
    back = np.empty((T, K), dtype=np.int64)
    V[0] = loss[0]
    for t in range(1, T):
        prev = V[t - 1]
        j = int(prev.argmin())
        stay = prev
        move = prev[j] + lam
        back[t] = np.where(stay <= move, np.arange(K), j)
        V[t] = loss[t] + np.minimum(stay, move)
    s = np.empty(T, dtype=np.int64)
    s[-1] = int(V[-1].argmin())
    for t in range(T - 1, 0, -1):
        s[t - 1] = back[t, s[t]]
    return s, float(V[-1].min())


def forward_labels(loss, lam):
    """Online labels: the state ending the cheapest path through each day (causal)."""
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


def refit_dates(index):
    years = range(pd.Timestamp(FIRST_REFIT).year, pd.Timestamp(DATA_END).year + 1)
    dates = [pd.Timestamp(f"{y}-{m:02d}-01") for y in years for m in (1, 7)]
    return [d for d in dates if d >= pd.Timestamp(FIRST_REFIT) and d <= index[-1]]


def jm_period(f, start, end, lam):
    """Fit on the 3,000 days before `start`; online labels for [start, end). 0 = bull."""
    pos = f.index.searchsorted(start)
    tr = f.iloc[max(0, pos - TRAIN_DAYS):pos]
    ap = f.iloc[pos:f.index.searchsorted(end)]
    Xtr, Xall = standardise(tr[FEATURES], pd.concat([tr, ap])[FEATURES])
    theta, s = fit_jm(Xtr, lam)
    cum = [tr["x"][s == k].sum() for k in range(2)]
    order = np.argsort(cum)[::-1]                      # bull first
    lab = forward_labels(sq_loss(Xall, theta[order]), lam)[len(tr):]
    ins = order.argsort()[s]                           # in-sample labels, bull = 0
    stats = {"start": start, "lam": lam, "train_days": len(tr),
             "bull_mean": tr["x"][ins == 0].mean(), "bear_mean": tr["x"][ins == 1].mean(),
             "bull_vol": tr["x"][ins == 0].std(), "bear_vol": tr["x"][ins == 1].std(),
             "bear_share": (ins == 1).mean(), "switches": int((np.diff(ins) != 0).sum())}
    return pd.Series(lab, index=ap.index), stats


def jm_all(f, lam):
    ds = refit_dates(f.index) + [pd.Timestamp(DATA_END) + pd.Timedelta(days=1)]
    parts, stats = [], []
    for a, b in zip(ds[:-1], ds[1:]):
        lab, st = jm_period(f, a, b, lam)
        parts.append(lab)
        stats.append(st)
    return pd.concat(parts), stats


# ----------------------------------------------------------------------------------------------
# HMM benchmark

def hmm_period(f, start, end, k_list):
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
    # last state of the Viterbi path ending each day = argmax of the forward max-product
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


def hmm_all(f):
    ds = pd.date_range(FIRST_REFIT, DATA_END, freq="MS").tolist() + [pd.Timestamp(DATA_END)
                                                                       + pd.Timedelta(days=1)]
    res = Parallel(n_jobs=-1)(delayed(hmm_period)(f, a, b, HMM_K) for a, b in zip(ds[:-1], ds[1:]))
    return {k: pd.concat([r[k] for r in res]) for k in HMM_K}


# ----------------------------------------------------------------------------------------------
# the 0/1 strategy and the cross-validation

def strategy(f, bear, delay=1):
    """Daily returns of the 0/1 strategy. Signal at the close of t, traded at the close of
    t + delay, earns from t + delay + 1."""
    w = (1 - bear).shift(delay + 1)
    w = w.reindex(f.index)
    ret = w * f["r"] + (1 - w) * f["rf"] - COST * w.diff().abs()
    return ret.dropna(), w


def sharpe(ret, rf):
    ex = ret - rf.reindex(ret.index)
    return ex.mean() / ex.std() * np.sqrt(YEAR_DAYS) if ex.std() > 0 else -np.inf


def cross_validate(f, labels_by):
    """Each month-end from 1989-12, the candidate with the best 8-year Sharpe is used next month."""
    rets = {k: strategy(f, lab)[0] for k, lab in labels_by.items()}
    month_ends = f.loc[OOS_START:].index.to_series().groupby(
        f.loc[OOS_START:].index.to_period("M")).first()
    chosen, pieces = [], []
    for p, first_day in month_ends.items():
        cut = f.index[f.index.searchsorted(first_day) - 1]       # last day of the previous month
        lo = cut - pd.DateOffset(years=VALID_YEARS)
        scores = {k: sharpe(r.loc[lo:cut], f["rf"]) for k, r in rets.items()}
        best = max(scores, key=scores.get)
        chosen.append((str(p), best, scores[best]))
        nxt = (p + 1).to_timestamp()
        pieces.append(labels_by[best].loc[first_day:nxt - pd.Timedelta(days=1)])
    # the labels from 1990 come from the chosen candidates; keep 1982-89 of the median one for plots
    lab = pd.concat(pieces)
    return lab, pd.DataFrame(chosen, columns=["month", "choice", "valid_sharpe"])


# ----------------------------------------------------------------------------------------------
# reporting

def perf(ret, w, rf):
    ex = ret - rf.reindex(ret.index)
    nav = (1 + ret).cumprod()
    years = len(ret) / YEAR_DAYS
    return {"CAGR": nav.iloc[-1] ** (1 / years) - 1,
            "vol": ret.std() * np.sqrt(YEAR_DAYS),
            "Sharpe": ex.mean() / ex.std() * np.sqrt(YEAR_DAYS),
            "MDD": (nav / nav.cummax() - 1).min(),
            "turnover/yr": w.diff().abs().sum() / years,
            "invested": w.mean()}


def spells(bear):
    ch = bear.ne(bear.shift()).cumsum()
    lengths = bear.groupby(ch).size()
    return lengths


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = load()
    print(f"data {f.index[0]:%Y-%m-%d} .. {f.index[-1]:%Y-%m-%d}, {len(f)} days")

    # JM for every lambda (parallel over the grid)
    res = Parallel(n_jobs=-1)(delayed(jm_all)(f, lam) for lam in LAMBDAS)
    jm_by = {lam: r[0] for lam, r in zip(LAMBDAS, res)}
    stats = pd.DataFrame([s for r in res for s in r[1]])
    jm_lab, jm_path = cross_validate(f, jm_by)
    print("JM done")

    hmm_by = hmm_all(f)
    hmm_lab, hmm_path = cross_validate(f, hmm_by)
    print("HMM done")

    oos = f.loc[OOS_START:]
    rf = f["rf"]
    rows, delays = {}, {}
    bh_w = pd.Series(1.0, index=oos.index)
    rows["Buy and hold"] = perf(oos["r"], bh_w, rf)
    for name, lab in (("HMM 0/1", hmm_lab), ("JM 0/1", jm_lab)):
        for dl in (1, 5, 10):
            ret, w = strategy(f, lab, dl)
            ret, w = ret.loc[OOS_START:], w.loc[OOS_START:]
            p = perf(ret, w, rf)
            if dl == 1:
                rows[name] = p
            delays[(name, dl)] = p["Sharpe"]
    table = pd.DataFrame(rows).T
    sub = {}
    for a, b in (("1990", "1997"), ("1998", "2004")):
        sub[("Buy and hold", a)] = perf(oos.loc[a:b, "r"], bh_w.loc[a:b], rf)["Sharpe"]
        for name, lab in (("HMM 0/1", hmm_lab), ("JM 0/1", jm_lab)):
            ret, w = strategy(f, lab, 1)
            sub[(name, a)] = perf(ret.loc[a:b], w.loc[a:b], rf)["Sharpe"]

    # regime-conditional excess returns: in-sample (training-window labels at the chosen lambda)
    # vs out-of-sample (the day after an online label, i.e. what a forecast can earn)
    lam_used = jm_path.assign(lam=jm_path["choice"]).set_index("month")["lam"]
    ins = stats[stats["start"] >= pd.Timestamp(OOS_START)].copy()
    ins = ins[[l == lam_used.get(f"{s:%Y-%m}") for s, l in zip(ins["start"], ins["lam"])]]
    nxt = oos["x"].shift(-1)
    regime_means = pd.DataFrame({
        "in-sample mean x (%/yr)": [ins["bull_mean"].mean() * YEAR_DAYS * 100,
                                    ins["bear_mean"].mean() * YEAR_DAYS * 100],
        "in-sample vol (%/yr)": [ins["bull_vol"].mean() * np.sqrt(YEAR_DAYS) * 100,
                                 ins["bear_vol"].mean() * np.sqrt(YEAR_DAYS) * 100],
        "out-of-sample next-day mean x (%/yr)": [nxt[jm_lab == 0].mean() * YEAR_DAYS * 100,
                                                 nxt[jm_lab == 1].mean() * YEAR_DAYS * 100],
        "out-of-sample next-day vol (%/yr)": [nxt[jm_lab == 0].std() * np.sqrt(YEAR_DAYS) * 100,
                                              nxt[jm_lab == 1].std() * np.sqrt(YEAR_DAYS) * 100],
        "out-of-sample share of days": [(jm_lab == 0).mean(), (jm_lab == 1).mean()],
    }, index=["bull", "bear"])
    # the t-statistic of the bull - bear difference out of sample
    a, b = nxt[jm_lab == 0].dropna(), nxt[jm_lab == 1].dropna()
    t_diff = (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))

    sp_jm, sp_hmm = spells(jm_lab), spells(hmm_lab)
    yrs = len(oos) / YEAR_DAYS
    lam_counts = jm_path["choice"].value_counts().sort_index()

    # save
    labels = pd.DataFrame({"jm_bear": jm_lab, "hmm_bear": hmm_lab}).join(f[FEATURES + ["x"]])
    labels.to_parquet(OUT / "labels.parquet")
    jm_path.to_csv(OUT / "lambda_path.csv", index=False)
    hmm_path.to_csv(OUT / "hmm_k_path.csv", index=False)
    regime_means.to_csv(OUT / "regime_means.csv")
    stats.to_csv(OUT / "jm_fits.csv", index=False)

    L = []
    L.append("E1 replication of Shu, Yu & Mulvey (2024) on ^N225, pre-test data only")
    L.append(f"data {f.index[0]:%Y-%m-%d} .. {f.index[-1]:%Y-%m-%d}; out of sample "
             f"{oos.index[0]:%Y-%m-%d} .. {oos.index[-1]:%Y-%m-%d} ({yrs:.1f} years)")
    L.append(f"first training windows: {stats['train_days'].min()} days (data start), "
             f"then {TRAIN_DAYS}")
    L.append("")
    L.append("0/1 strategy, 1-day delay, 10 bp one-way, 1990-2004")
    L.append(table.to_string(float_format=lambda v: f"{v:.3f}"))
    L.append("")
    L.append("Sharpe by sub-period (1-day delay)")
    L.append(pd.Series(sub).unstack().to_string(float_format=lambda v: f"{v:.2f}"))
    L.append("")
    L.append("Sharpe by trading delay (days)")
    L.append(pd.Series(delays).unstack().to_string(float_format=lambda v: f"{v:.2f}"))
    L.append("")
    L.append("Regime labels 1990-2004 (online, after cross-validation)")
    for name, lab, sp in (("JM", jm_lab, sp_jm), ("HMM", hmm_lab, sp_hmm)):
        L.append(f"  {name}: {(lab.diff().abs() > 0).sum() / yrs:.1f} switches a year, median spell "
                 f"{sp.median():.0f} days, bear share {lab.mean():.0%}")
    L.append("")
    L.append("Lambda chosen by the 8-year cross-validation (months)")
    L.append(lam_counts.to_string())
    L.append("")
    L.append("HMM smoothing k chosen (months)")
    L.append(hmm_path["choice"].value_counts().sort_index().to_string())
    L.append("")
    L.append("Regime-conditional excess returns of the index (JM, chosen lambda)")
    L.append(regime_means.to_string(float_format=lambda v: f"{v:.2f}"))
    L.append(f"  out-of-sample bull - bear next-day mean: t = {t_diff:.2f}")
    L.append("")
    L.append("JM training fits (all lambdas): bear share of training days, switches per fit")
    L.append(stats.groupby("lam")[["bear_share", "switches"]].mean().to_string(
        float_format=lambda v: f"{v:.2f}"))
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    plot(f, jm_lab, hmm_lab)


def plot(f, jm_lab, hmm_lab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    px = (1 + f["r"]).cumprod().loc[OOS_START:]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, lab, name in ((axes[0], jm_lab, "Statistical jump model"),
                          (axes[1], hmm_lab, "HMM on daily returns (benchmark)")):
        ax.plot(px.index, px.values, color="#1f3b57", lw=1)
        ax.set_yscale("log")
        b = lab.reindex(px.index).fillna(0)
        ax.fill_between(px.index, px.min(), px.max(), where=b.values > 0.5, color="#c0504d",
                        alpha=0.25, lw=0, label="bear (online label)")
        ax.set_title(f"{name}: Nikkei 225, 1990-2004 (out of sample, causal labels)", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_regimes.png", dpi=150)


if __name__ == "__main__":
    main()
