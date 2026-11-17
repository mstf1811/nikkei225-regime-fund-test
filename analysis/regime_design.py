"""Regime detection (E1) design study, on data from BEFORE the test period only.

The test period is 2005-2025 (D9). To keep it clean, the E1 design choices are compared on the
Nikkei 225 index (Yahoo ^N225, daily close, price index) from 1970 to 2004. The study walks forward
like the real pipeline: the model is refitted on 1 January of each year 1980-2004 on the training
window before it, frozen, and run over the year.

Candidates
    features  pfe    the PFE's two features (§4.3): 60-day rolling mean log return and 60-day
                     rolling volatility, annualised; diagonal Gaussian emissions
              daily  the daily log return itself; Gaussian emissions (Hamilton 1989; Nystrup et
                     al. 2015)
    states    2, 3
    window    5y and 10y rolling, and expanding from 1970
Every fit is EM with N_STARTS random starts, keeping the highest likelihood (as the PFE).

Decoding (D1). Causal forward filter with the frozen parameters: P(state_t | data up to t), run
from the start of the training window through the year. The label is the most likely state.
The PFE's block Viterbi over the year is computed too, only to show how often it disagrees.
States are ordered by volatility (0 = calmest).

Measures (structural, never portfolio performance)
    degenerate    a fit in which some state holds < MIN_OCCUPANCY of the training days
    vol ratio     highest / lowest state volatility (separation)
    param drift   variation across the yearly refits of the calmest and the most turbulent state's
                  volatility (stable state definitions = low drift)
    switches      label changes per year (each one triggers a rebalance, D7) and median spell length
    vs Viterbi    share of days where the block-Viterbi label differs from the causal one
    vol forecast  the model's forecast of the next 21 days' volatility against the realised one
                  (correlation and mean squared error of log volatility); benchmarks: trailing
                  21-day and 60-day volatility, and EWMA (lambda 0.94)
    by label      the next 21 days' return, volatility and return/volatility by today's label
                  (descriptive)

Outputs (analysis/output/regime/): summary.txt, configs.csv, by_label.csv, labels.parquet,
yearly_shares.csv, fig_labels.png. `--plots` redraws the last two from labels.parquet.

Run:  python analysis/regime_design.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import DATA_DIR, PROJECT_DIR  # noqa: E402

warnings.filterwarnings("ignore")
OUT = PROJECT_DIR / "analysis" / "output" / "regime"
FIRST_YEAR, LAST_YEAR = 1980, 2004
DATA_END = "2004-12-31"
N_STARTS = 8
MIN_OCCUPANCY = 0.03
ROLL = 60
H = 21                      # forecast horizon, trading days
YEAR_DAYS = 245             # trading days a year in Japan (data: 243-251 a year in 1970-2004)
WINDOWS = {"5y": 5, "10y": 10, "expanding": None}


def load():
    n = pd.read_csv(DATA_DIR / "market" / "nikkei225_index.csv", parse_dates=["date"])
    n = n[n["date"] <= DATA_END].sort_values("date").set_index("date")
    r = np.log(n["close"]).diff().dropna()
    f = pd.DataFrame({"r": r})
    f["m60"] = r.rolling(ROLL).mean() * YEAR_DAYS
    f["v60"] = r.rolling(ROLL).std() * np.sqrt(YEAR_DAYS)
    f["v21"] = r.rolling(H).std() * np.sqrt(YEAR_DAYS)
    var = np.zeros(len(r))
    var[0] = r.iloc[:H].var()
    for i in range(1, len(r)):
        var[i] = 0.94 * var[i - 1] + 0.06 * r.iloc[i - 1] ** 2
    # EWMA variance known at the close of day t (uses r up to t)
    f["ewma"] = np.sqrt(np.r_[var[1:], 0.94 * var[-1] + 0.06 * r.iloc[-1] ** 2] * YEAR_DAYS)
    # realised volatility of the next H days (t+1 .. t+H), the target of the forecasts
    f["fwd_vol"] = r[::-1].rolling(H).std()[::-1].shift(-1) * np.sqrt(YEAR_DAYS)
    f["fwd_ret"] = r[::-1].rolling(H).sum()[::-1].shift(-1) * YEAR_DAYS / H
    return f


def features(f, kind):
    return (f[["r"]] * 100).dropna() if kind == "daily" else f[["m60", "v60"]].dropna()


def fit(x, k, seed0):
    best = None
    for s in range(N_STARTS):
        m = GaussianHMM(n_components=k, covariance_type="diag", n_iter=500, tol=1e-4,
                        random_state=seed0 + s, init_params="stmc")
        try:
            m.fit(x)
            ll = m.score(x)
        except Exception:
            continue
        if best is None or ll > best[0]:
            best = (ll, m)
    return best[1]


def order_by_vol(m, kind):
    """State order by volatility: the daily return's sd, or the mean of the rolling-vol feature."""
    if kind == "daily":
        key = np.sqrt(m.covars_[:, 0, 0])
    else:
        key = m.means_[:, 1]
    return np.argsort(key)


def log_emission(m, x):
    mu, var = m.means_, np.array([np.diag(c) for c in m.covars_])
    return -0.5 * (np.log(2 * np.pi * var)[None] + (x[:, None, :] - mu[None]) ** 2 / var[None]).sum(axis=2)


def forward_filter(m, x):
    """P(state_t | x_1..t) for every t (causal)."""
    le = log_emission(m, x)
    A = m.transmat_
    out = np.empty_like(le)
    a = m.startprob_ * np.exp(le[0] - le[0].max())
    out[0] = a / a.sum()
    for t in range(1, len(x)):
        a = (out[t - 1] @ A) * np.exp(le[t] - le[t].max())
        out[t] = a / a.sum()
    return out


def vol_forecast(m, kind, p):
    """Forecast of the next H days' annualised volatility from today's filtered probabilities."""
    A = m.transmat_
    if kind == "daily":
        mu = m.means_[:, 0] / 100
        var = m.covars_[:, 0, 0] / 1e4
        tot = np.zeros(len(p))
        q = p.copy()
        for _ in range(H):
            q = q @ A
            tot += (q * (var + mu ** 2)).sum(axis=1) - ((q * mu).sum(axis=1)) ** 2
        return np.sqrt(tot / H * YEAR_DAYS)
    q = p.copy()
    for _ in range(H):
        q = q @ A
    return q @ m.means_[:, 1]


def run_config(f, kind, k, window):
    x_all = features(f, kind)
    rows, years = [], []
    for y in range(FIRST_YEAR, LAST_YEAR + 1):
        start = x_all.index.min() if WINDOWS[window] is None else pd.Timestamp(f"{y - WINDOWS[window]}-01-01")
        train = x_all[(x_all.index >= start) & (x_all.index < f"{y}-01-01")]
        span = x_all[(x_all.index >= start) & (x_all.index <= f"{y}-12-31")]
        m = fit(train.values, k, seed0=1000 * y)
        order = order_by_vol(m, kind)
        rank = np.empty(k, int)
        rank[order] = np.arange(k)
        p = forward_filter(m, span.values)
        lab = rank[p.argmax(axis=1)]
        ntr = len(train)
        occ = np.bincount(lab[:ntr], minlength=k) / ntr
        test = span.index >= f"{y}-01-01"
        vit = rank[m.predict(span.values[test])]                 # the PFE's block Viterbi
        sd = (np.sqrt(m.covars_[:, 0, 0]) / 100 * np.sqrt(YEAR_DAYS) if kind == "daily"
              else m.means_[:, 1])[order]
        years.append({"year": y, "degenerate": bool(occ.min() < MIN_OCCUPANCY),
                      "vol_low": sd[0], "vol_high": sd[-1], "vol_ratio": sd[-1] / sd[0],
                      "viterbi_differs": float((vit != lab[test]).mean())})
        fc = vol_forecast(m, kind, p[test])
        prob_top = p[test][:, order[-1]]
        rows.append(pd.DataFrame({"label": lab[test], "vol_fc": fc, "p_top": prob_top}, index=span.index[test]))
    lab = pd.concat(rows)
    return lab, pd.DataFrame(years)


def spells(labels):
    change = labels.ne(labels.shift())
    ids = change.cumsum()
    return labels.groupby(ids).size()


def score_forecast(fc, realised):
    ok = fc.notna() & realised.notna() & (fc > 0) & (realised > 0)
    a, b = np.log(fc[ok]), np.log(realised[ok])
    return np.corrcoef(a, b)[0, 1], ((a - b) ** 2).mean()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = load()
    test = f[(f.index >= f"{FIRST_YEAR}-01-01")]
    configs, by_label, label_paths = [], [], {}
    for kind in ["pfe", "daily"]:
        for k in [2, 3]:
            for window in WINDOWS:
                name = f"{kind}-{k}-{window}"
                lab, yrs = run_config(f, kind, k, window)
                label_paths[name] = lab["label"]
                sp = spells(lab["label"])
                corr, mse = score_forecast(lab["vol_fc"], test["fwd_vol"].reindex(lab.index))
                configs.append({
                    "config": name, "features": kind, "states": k, "window": window,
                    "degenerate_fits": int(yrs["degenerate"].sum()),
                    "vol_ratio_median": yrs["vol_ratio"].median(),
                    "drift_low": yrs["vol_low"].std() / yrs["vol_low"].mean(),
                    "drift_high": yrs["vol_high"].std() / yrs["vol_high"].mean(),
                    "switches_per_year": (lab["label"].diff().fillna(0) != 0).sum() / (LAST_YEAR - FIRST_YEAR + 1),
                    "median_spell_days": sp.median(),
                    "viterbi_differs": yrs["viterbi_differs"].mean(),
                    "fc_corr": corr, "fc_mse": mse,
                    "share_top_state": (lab["label"] == k - 1).mean(),
                })
                d = test.reindex(lab.index).assign(label=lab["label"])
                g = d.groupby("label").agg(days=("r", "size"), fwd_ret=("fwd_ret", "mean"),
                                           fwd_vol=("fwd_vol", "mean"))
                g["fwd_ret_over_vol"] = g["fwd_ret"] / g["fwd_vol"]
                by_label.append(g.reset_index().assign(config=name))
                print(name, "done", flush=True)
    configs = pd.DataFrame(configs)
    bench = []
    for col, nm in [("v21", "trailing 21d"), ("v60", "trailing 60d"), ("ewma", "EWMA 0.94")]:
        c, e = score_forecast(test[col], test["fwd_vol"])
        bench.append({"benchmark": nm, "fc_corr": c, "fc_mse": e})
    bench = pd.DataFrame(bench)
    by_label = pd.concat(by_label)
    configs.to_csv(OUT / "configs.csv", index=False, float_format="%.4f")
    by_label.to_csv(OUT / "by_label.csv", index=False, float_format="%.4f")
    pd.DataFrame(label_paths).to_parquet(OUT / "labels.parquet")

    pd.set_option("display.width", 200)
    lines = [
        f"E1 design study on ^N225 (price index), walk-forward refits {FIRST_YEAR}-{LAST_YEAR}, data to {DATA_END} only.",
        "Causal forward-filter labels; states ordered by volatility (0 = calmest).",
        "",
        configs.round(3).to_string(index=False),
        "",
        "Volatility-forecast benchmarks (next 21 days, log-vol correlation and MSE), same days:",
        bench.round(3).to_string(index=False),
        "",
        "Next 21 days by today's label (annualised; price returns, no cash rate):",
        by_label.round(3).to_string(index=False),
    ]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    yearly(pd.DataFrame(label_paths))
    plot(pd.DataFrame(label_paths))

def yearly(paths):
    """Share of each year in the most turbulent and the calmest state, next to the year's realised
    volatility. A model stuck in one state through a whole era shows here (the PFE's failure mode)."""
    n = pd.read_csv(DATA_DIR / "market" / "nikkei225_index.csv", parse_dates=["date"]).set_index("date")["close"]
    r = np.log(n).diff()
    t = pd.DataFrame({"vol": (r.groupby(r.index.year).std() * np.sqrt(YEAR_DAYS)).loc[FIRST_YEAR:LAST_YEAR]})
    for c in paths:
        k = paths[c].max() + 1
        t[c + " top"] = (paths[c] == k - 1).groupby(paths.index.year).mean()
        t[c + " calm"] = (paths[c] == 0).groupby(paths.index.year).mean()
    t.round(3).to_csv(OUT / "yearly_shares.csv")
    return t


def plot(paths):
    """Index on a log scale and the label paths of the main candidates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = pd.read_csv(DATA_DIR / "market" / "nikkei225_index.csv", parse_dates=["date"]).set_index("date")["close"]
    show = [c for c in ["pfe-3-5y", "pfe-3-10y", "pfe-3-expanding", "daily-3-10y"] if c in paths]
    fig, axes = plt.subplots(len(show) + 1, 1, figsize=(11, 2 + 1.3 * len(show)), sharex=True,
                             gridspec_kw={"height_ratios": [2.5] + [1] * len(show)})
    axes[0].plot(n[f"{FIRST_YEAR}":DATA_END], color="#333", lw=0.8)
    axes[0].set_yscale("log")
    axes[0].set_title("Nikkei 225 (log scale) and causal regime labels (dark = more turbulent)", fontsize=10)
    for ax, c in zip(axes[1:], show):
        s = paths[c]
        ax.imshow(s.values[None, :], aspect="auto", cmap="Greys", interpolation="nearest",
                  extent=[matplotlib.dates.date2num(s.index[0]), matplotlib.dates.date2num(s.index[-1]), 0, 1])
        ax.set_yticks([])
        ax.set_ylabel(c, rotation=0, ha="right", va="center", fontsize=8)
    axes[-1].xaxis_date()
    fig.tight_layout()
    fig.savefig(OUT / "fig_labels.png", dpi=150)


if __name__ == "__main__":
    if "--plots" in sys.argv:
        saved = pd.read_parquet(OUT / "labels.parquet")
        yearly(saved)
        plot(saved)
    else:
        main()
