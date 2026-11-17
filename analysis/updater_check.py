"""E3 (updating component) as in the recent literature, checked on data from BEFORE the test period.

D14 says the updater revises the few common parameters, the market premium pi and the signal weights
gamma, not 225 stock means. The literature's updater for return-prediction coefficients is a
Kalman filter with time-varying coefficients and a discount factor (a dynamic linear model, West &
Harrison 1997), with Bayesian averaging over the discount factor ("dynamic model averaging",
Raftery, Karny & Ettler 2010; Koop & Korobilis 2012); Dangl & Halling (2012, JFE) apply it to
monthly stock-return prediction and find time-varying coefficients beat constant ones out of sample.
It nests the PFE's scalar random-walk Kalman filter (its q plays the role of the discount factor).

Model (one scalar filter per parameter; F = 1, random-walk state)
    observation  y_t = theta_t + e_t,  e_t ~ N(0, V k_t)     V unknown, inverse-gamma (W&H 4.6)
    state        theta_t = theta_t-1 + w_t,  Var(w_t) = (1 - delta)/delta * C_t-1
    discount grid delta in {0.95, 0.98, 0.99, 1.00} (Koop & Korobilis 2012; 1.00 = constant
    coefficient), averaged with weights from the one-step Student-t predictive likelihood; no
    forgetting on the weights (alpha = 1, as Dangl & Halling 2012).
    Prior: worth one month of data (unit-information prior, Kass & Wasserman 1995) [ours].

Part A: the bull-regime premium pi_bull (D17), on ^N225 1990-2004 (pre-test)
    y_t = mean daily excess return of the month over the days the fund holds the bull exposure
          (online jump-model label of t-2 is bull; D17 timing), k_t = 1 / number of such days.
    Prior on 1990-01: the D17 value then (the training bull-day mean) with variance S/21.
    Forecast for month t = the averaged posterior mean at the end of t-1, floored at 0
    (Campbell & Thompson 2008: a negative premium forecast is set to zero).
    Compared out of sample with: the D17 rule (training bull-day mean at the latest refit), zero,
    and the historical average excess return of all days (Campbell & Thompson's benchmark).
    Loss: squared error weighted by the number of bull days (the exposure scales with them).

Part B: the six signal weights gamma (D15 (4)), mechanics only (2001-2004 has 46 slope months)
    y_t = the month's Fama-MacBeth slope; first 12 months set V and the prior (mean 0); the D15
    empirical-Bayes shrinkage toward 0 is then applied to the filtered mean and variance.

Outputs (analysis/output/updater/): summary.txt, pi_path.csv, gamma_dlm.csv

Run:  python analysis/updater_check.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROJECT_DIR  # noqa: E402
import regime_jm as rj  # noqa: E402

warnings.filterwarnings("ignore")
OUT = PROJECT_DIR / "analysis" / "output" / "updater"
REG = PROJECT_DIR / "analysis" / "output" / "regime_jm"
DELTAS = [0.95, 0.98, 0.99, 1.00]
PRIOR_DAYS = 21                  # unit-information prior: one month of daily data
YEAR_DAYS = rj.YEAR_DAYS


class DLM:
    """Scalar local-level DLM with discount delta and unknown constant variance (West & Harrison)."""

    def __init__(self, delta, m, C, S, n):
        self.delta, self.m, self.C, self.S, self.n = delta, m, C, S, n

    def step(self, y=None, k=1.0):
        """Advance one period; update with y (variance scale k) if given. Returns the predictive
        log density of y (0 when there is no observation)."""
        R = self.C / self.delta
        if y is None:
            self.C = R
            return 0.0
        Q = R + k * self.S
        e = y - self.m
        logp = stats.t.logpdf(e, df=self.n, scale=np.sqrt(Q))
        A = R / Q
        n_new = self.n + 1
        S_new = self.S + self.S / n_new * (e * e / Q - 1)
        self.m = self.m + A * e
        self.C = S_new / self.S * (R - A * A * Q)
        self.S, self.n = S_new, n_new
        return logp


class DMA:
    """Averaging over the discount grid, weights from the predictive likelihood (alpha = 1)."""

    def __init__(self, m, C, S, n):
        self.models = [DLM(d, m, C, S, n) for d in DELTAS]
        self.logw = np.zeros(len(DELTAS))

    def weights(self):
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def mean(self):
        return float(self.weights() @ [d.m for d in self.models])

    def var(self):
        w, mu = self.weights(), np.array([d.m for d in self.models])
        c = np.array([d.C for d in self.models])
        return float(w @ (c + mu ** 2) - (w @ mu) ** 2)

    def step(self, y=None, k=1.0):
        self.logw += np.array([d.step(y, k) for d in self.models])


# ----------------------------------------------------------------------------------------------
# Part A: pi_bull

def part_a():
    f = rj.load()
    labels = pd.read_parquet(REG / "labels.parquet")
    fits = pd.read_csv(REG / "jm_fits.csv", parse_dates=["start"])
    path = pd.read_csv(REG / "lambda_path.csv").set_index("month")["choice"]
    hold = (1 - labels["jm_bear"]).shift(2)                 # exposure on day d: label of d-2
    x = f["x"].reindex(hold.index)
    months = pd.period_range("1990-01", "2004-12", freq="M")

    def d17(m):
        start = pd.Timestamp(f"{m.year}-{1 if m.month <= 6 else 7:02d}-01")
        lam = path[str(m)]
        row = fits[(fits["start"] == start) & np.isclose(fits["lam"], lam)].iloc[0]
        return row["bull_mean"], row["bull_vol"] ** 2

    pi0, s0 = d17(months[0])
    dma = DMA(m=pi0, C=s0 / PRIOR_DAYS, S=s0, n=PRIOR_DAYS)
    rows = []
    for m in months:
        in_m = x.index.to_period("M") == m
        days = in_m & (hold == 1)
        y, n = x[days].mean(), int(days.sum())
        pi_d17, _ = d17(m)
        hist = f["x"].loc[:m.to_timestamp() - pd.Timedelta(days=1)].mean()
        rows.append({"month": str(m), "bull_days": n, "realised": y if n else np.nan,
                     "E3": max(dma.mean(), 0.0), "E3 raw": dma.mean(), "D17": pi_d17,
                     "historical average": hist, "zero": 0.0,
                     **{f"w{d}": w for d, w in zip(DELTAS, dma.weights())}})
        dma.step(y if n else None, 1.0 / n if n else 1.0)
    p = pd.DataFrame(rows).set_index("month")
    final = {"E3 pi_bull for 2005-01": max(dma.mean(), 0.0), "sd": np.sqrt(dma.var()),
             "weights": dict(zip(DELTAS, dma.weights()))}
    return p, final


def score(p, cols):
    ok = p["bull_days"] > 0
    w, y = p.loc[ok, "bull_days"], p.loc[ok, "realised"]
    sse = {c: float((w * (y - p.loc[ok, c]) ** 2).sum()) for c in cols}
    return sse


# ----------------------------------------------------------------------------------------------
# Part B: gamma

def part_b(names=None):
    """names: the signal list (D12's six by default; D21's eight from analysis/signals_v2_check.py)."""
    import optimizer_check as oc
    names = names or oc.SIGNALS
    px, liq, ch, rate = oc.load()
    r, simple, member, rf, mkt = oc.daily_panels(px, ch, rate)
    signals, members_me = oc.signal_panels(px, liq, ch, simple, rf, mkt, names)
    g = oc.fm_slopes(simple, rf, mkt, signals, members_me, names)

    # D15 (4) as written: Fama-MacBeth mean and empirical Bayes
    gbar, se = g.mean(), g.std(ddof=1) / np.sqrt(len(g))
    tau2 = max(0.0, (gbar ** 2).mean() - (se ** 2).mean())
    fm_eb = gbar * tau2 / (tau2 + se ** 2)

    # E3: the filter from month 13, after a 12-month burn-in that sets V
    burn = g.iloc[:12]
    res = {}
    for k in g.columns:
        s0 = burn[k].var()
        dma = DMA(m=0.0, C=s0, S=s0, n=12)
        for v in g[k].iloc[12:]:
            dma.step(v)
        res[k] = {"mean": dma.mean(), "sd": np.sqrt(dma.var()),
                  **{f"w{d}": w for d, w in zip(DELTAS, dma.weights())}}
    d = pd.DataFrame(res).T
    tau2_k = max(0.0, (d["mean"] ** 2).mean() - (d["sd"] ** 2).mean())
    d["E3 (EB)"] = d["mean"] * tau2_k / (tau2_k + d["sd"] ** 2)
    d["D15 FM mean"] = gbar
    d["D15 FM se"] = se
    d["D15 (EB)"] = fm_eb
    return d, len(g), g.index[0], g.index[-1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p, final = part_a()
    p.to_csv(OUT / "pi_path.csv")
    cols = ["E3", "D17", "historical average", "zero"]
    sse = score(p, cols)
    ann = lambda v: v * YEAR_DAYS * 100  # noqa: E731
    L = ["E3 check on pre-test data", "",
         "Part A: bull-regime premium on ^N225, forecasts for 1990-01 .. 2004-12 (out of sample)"]
    ok = p["bull_days"] > 0
    L.append(f"months with bull exposure: {ok.sum()} of {len(p)}; bull days {p['bull_days'].sum()}")
    L.append(f"realised mean excess return on bull days: "
             f"{ann((p.loc[ok, 'realised'] * p.loc[ok, 'bull_days']).sum() / p['bull_days'].sum()):.1f}%/yr")
    L.append("mean forecast (%/yr) and out-of-sample R^2 against each benchmark (bull-day weighted)")
    for c in cols:
        r2 = {b: 1 - sse[c] / sse[b] for b in cols if b != c}
        L.append(f"  {c:20s} mean {ann(p[c].mean()):6.1f}   " +
                 "  ".join(f"R2 vs {b}: {v:+.4f}" for b, v in r2.items()))
    L.append("  (R^2 is small for any return forecast at this horizon; Campbell & Thompson 2008 call "
             "0.5% a month economically meaningful)")
    for a, b in (("1990-01", "1997-12"), ("1998-01", "2004-12")):
        q = p.loc[a:b]
        s = score(q, cols)
        L.append(f"  {a[:4]}-{b[:4]}: R2 of E3 vs D17 {1 - s['E3'] / s['D17']:+.4f}; mean forecast "
                 f"E3 {ann(q['E3'].mean()):.1f}, D17 {ann(q['D17'].mean()):.1f}, realised "
                 f"{ann((q.loc[q.bull_days > 0, 'realised'] * q.loc[q.bull_days > 0, 'bull_days']).sum() / max(q['bull_days'].sum(), 1)):.1f} %/yr")
    wcols = [f"w{d}" for d in DELTAS]
    L.append("discount-factor weights, average over the months: " +
             ", ".join(f"{d}: {p[c].mean():.2f}" for d, c in zip(DELTAS, wcols)))
    L.append(f"at 2004-12 (prior for the test): pi_bull = {ann(final['E3 pi_bull for 2005-01']):.1f}%/yr "
             f"(sd {ann(final['sd']):.1f}); weights " +
             ", ".join(f"{d}: {w:.2f}" for d, w in final["weights"].items()))
    L.append(f"months where the E3 forecast was floored at 0: {(p['E3 raw'] < 0).sum()}")

    d, n, first, last = part_b()
    d.to_csv(OUT / "gamma_dlm.csv")
    L.append("")
    L.append(f"Part B: signal weights, monthly units; {n} slope months {first} .. {last}; "
             f"filter from month 13 (mechanics only)")
    L.append(d[["D15 FM mean", "D15 FM se", "D15 (EB)", "mean", "sd", "E3 (EB)"] + wcols].to_string(
        float_format=lambda v: f"{v:.5f}"))
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
