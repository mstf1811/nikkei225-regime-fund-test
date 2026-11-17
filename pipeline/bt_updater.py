"""The updating component (E3, D19): scalar dynamic linear models with a random-walk state and a
discount factor (West & Harrison 1997), averaged over the discount grid by predictive likelihood
(dynamic model averaging, Raftery et al. 2010; Koop & Korobilis 2012; alpha = 1 as Dangl & Halling
2012). Ported from analysis/updater_check.py; bt_tests.py checks the pi path against it.

    pi_path()      the market premium for each month: the E3 bull premium (floored at 0, Campbell &
                   Thompson 2008), the D17 rule (training bull-day mean, the no-E3 comparison), the
                   bear mean and its -10 bp/day bound (the D17 (9) sensitivity), and the historical
                   average (C1). All in daily units; the value for month M uses data to the end of M-1.
    GammaFilters   one filter per signal on the monthly Fama-MacBeth slopes, after a 12-month burn-in,
                   then the D15 empirical-Bayes shrinkage across the signals.
"""
import numpy as np
import pandas as pd
from scipy import stats

from bt_estimation import eb_shrink

DELTAS = [0.95, 0.98, 0.99, 1.00]
PRIOR_DAYS = 21
GAMMA_BURN = 12
BEAR_CAP = -0.0010               # Shu, Yu & Mulvey (2025): bear forecasts capped at -10 bp a day


class DLM:
    def __init__(self, delta, m, C, S, n):
        self.delta, self.m, self.C, self.S, self.n = delta, m, C, S, n

    def step(self, y=None, k=1.0):
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
    def __init__(self, m, C, S, n, deltas=DELTAS):
        self.deltas = list(deltas)
        self.models = [DLM(d, m, C, S, n) for d in self.deltas]
        self.logw = np.zeros(len(self.deltas))

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


def pi_path(regime, deltas=DELTAS, last_month=None):
    """Monthly premium inputs from 1990-01 to `last_month` (default: the month after the data end)."""
    f, labels, d17 = regime.f, regime.labels, regime.by_month
    hold = (1 - labels).shift(2)
    x = f["x"].reindex(hold.index)
    first = pd.Period(labels.index[0], "M")
    last = last_month or pd.Period(f.index[-1], "M") + 1
    months = pd.period_range(first, last, freq="M")
    pi0, bull_vol = d17.loc[str(first), "bull_mean"], d17.loc[str(first), "bull_vol"]
    s0 = bull_vol ** 2
    dma = DMA(m=pi0, C=s0 / PRIOR_DAYS, S=s0, n=PRIOR_DAYS, deltas=deltas)
    rows = []
    for m in months:
        row = {"month": str(m), "e3": max(dma.mean(), 0.0), "e3_raw": dma.mean(), "e3_sd": np.sqrt(dma.var())}
        if str(m) in d17.index:
            row.update(d17=d17.loc[str(m), "bull_mean"], bear=d17.loc[str(m), "bear_mean"])
        row["bear_capped"] = max(row.get("bear", np.nan), BEAR_CAP) if "bear" in row else np.nan
        row["hist"] = f["x"].loc[:m.to_timestamp() - pd.Timedelta(days=1)].mean()
        rows.append(row)
        in_m = x.index.to_period("M") == m
        if not in_m.any():
            continue
        days = in_m & (hold == 1)
        n = int(days.sum())
        dma.step(x[days].mean() if n else None, 1.0 / n if n else 1.0)
    return pd.DataFrame(rows).set_index("month")


class GammaFilters:
    """E3 for the signal weights (D19 (4)): fed one Fama-MacBeth slope row per month."""

    def __init__(self, names, deltas=DELTAS):
        self.names, self.deltas = list(names), list(deltas)
        self.rows, self.dma = [], None

    def add(self, slope):
        self.rows.append(slope[self.names].to_numpy(dtype=float))
        if len(self.rows) == GAMMA_BURN:
            burn = pd.DataFrame(self.rows, columns=self.names)
            self.dma = {k: DMA(m=0.0, C=burn[k].var(), S=burn[k].var(), n=GAMMA_BURN, deltas=self.deltas)
                        for k in self.names}
        elif len(self.rows) > GAMMA_BURN:
            for k, v in zip(self.names, self.rows[-1]):
                self.dma[k].step(v)

    def gamma(self):
        if self.dma is None:
            return pd.Series(0.0, index=self.names), {}
        mean = pd.Series({k: d.mean() for k, d in self.dma.items()})
        sd = pd.Series({k: np.sqrt(d.var()) for k, d in self.dma.items()})
        g, tau2 = eb_shrink(mean, sd)
        return g, {"mean": mean, "sd": sd, "tau2": tau2}
