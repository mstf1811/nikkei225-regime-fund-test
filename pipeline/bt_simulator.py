"""Daily NAV simulation of the fund (D2, D7, D13, D17 (7), D18, D19 (5), D20 (1)).

Each trading day t:
  1. overnight: cash earns the call rate of t-1 over the calendar days elapsed; shorts are charged
     the borrow fee plus the call rate on their value at the close of t-1 (the short proceeds earn
     0%, D13; with `proceeds_earn` only the fee); held positions move by their price factor, longs
     receive and shorts pay the day's dividends (D13; imputed ones included, D10); a holding whose
     last price day has passed is closed at its last value [ours];
  2. at the first trading day of a month, the estimator processes the previous month-end: beta,
     the new Fama-MacBeth slope, the E3 filters, covariance and alpha for this month's decisions;
  3. at the close: index leavers are sold (bought back) at up to 10% of the day's volume (D18 (5));
     on a decision day the optimizer is solved and its trades executed; costs are charged with the
     optimizer's own model (D18 (3)); a decision is re-solved on the next days with the updated
     holdings, up to 5 days, until a day's turnover is below 0.5% of NAV (D18 (8));
  4. risk checks after the close trigger a decision the next day (D7).

Decisions (D7, D17 (7), D18 (8)): scheduled (the first trading day of each month; weekly or quarterly
in the sensitivities), a regime trigger when the label used changes (the label of t - delay), and a
risk trigger after a close with a long above 10%, a short beyond -5%, or the long or short side above
100% of NAV, each by more than the band below [ours: D7's "a name above 10%, short squeeze, exposure
drift"]. Inputs for a month are
estimated at the previous month-end only. Benchmarks (1/N, quintile sort) trade to their targets at
the monthly decision, without the fund's caps, with the same costs.

Risk triggers fire only on a material breach [ours]: a limit exceeded by more than 10% of itself
(a long above 11%, a short beyond -5.5%, a side above 110% of NAV). Without the band, a short held
at its -5% bound drifts past it the next day and re-triggers a full decision every day (found in the
pre-test mechanics run of C3, 2026-10-12). The optimizer still enforces the exact limits.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

import bt_estimation as be
import bt_optimizer as bo
from bt_data import MONTH_DAYS
from bt_signals import SIGNALS_V2, build
from bt_updater import DELTAS, GammaFilters


@dataclass(frozen=True)
class Config:
    name: str = "B0"
    signals: tuple = tuple(SIGNALS_V2)
    regime: str = "jm"            # jm | hmm | off (C1: historical-average premium, no regime triggers)
    pi_bear: str = "zero"         # zero (D17) | capped (D17 (9): bear mean bounded at -10 bp/day)
    updater: str = "e3"           # e3 (D19) | off (C3: D17 pi_bull, D15 rolling Fama-MacBeth)
    deltas: tuple = tuple(DELTAS)
    return_model: str = "d14"     # d14 | hist (C5)
    g_risk: float = 10.0
    g_trade: float = 1.0
    g_hold: float = 1.0
    fund_rules: bool = True       # False = C4
    cost_a: float = 0.0015
    borrow: float = 0.0115
    proceeds_earn: bool = False
    short_months: int = 0         # 6 = shorts closed and re-opened every 6 months (D13 (3) sensitivity)
    rebalance: str = "monthly"    # monthly | weekly | quarterly
    delay: int = 1                # trading days between the label and the trade (D17 (7))
    nav0: float = 10e9
    imputed_dividends: bool = True
    benchmark: str = ""           # "" | "1/N" | "quintile"
    exec_days: int = 5
    exec_stop: float = 0.005
    risk_band: float = 0.10       # [ours] a risk trigger needs a limit exceeded by > 10% of itself

    def params(self):
        return bo.Params(g_risk=self.g_risk, g_trade=self.g_trade, g_hold=self.g_hold, a=self.cost_a,
                         borrow=self.borrow, proceeds_earn=self.proceeds_earn, fund_rules=self.fund_rules)

    def diff(self, base=None):
        base = asdict(base or Config())
        return {k: v for k, v in asdict(self).items() if base.get(k) != v and k != "name"}


# ----------------------------------------------------------------------------------------------

class Estimator:
    """Month-end processing (D8, D15, D16, D19 (5)); inputs[M] holds what decisions in month M use."""

    def __init__(self, data, cfg, pis):
        self.d, self.cfg, self.pis = data, cfg, pis
        self.names = list(cfg.signals)
        self.signals, self.members_me = build(data, self.names)
        self.monthly = be.Monthly(data)
        self.month_end = data.month_ends()
        self.beta, self.slopes, self.inputs = {}, [], {}
        self.gf = GammaFilters(self.names, cfg.deltas)

    def process(self, m, full):
        d, me = self.d, self.month_end[m]
        codes = list(self.members_me.loc[m][self.members_me.loc[m]].index)
        bv, b_ols, resid, _ = be.betas_at(d.simple, d.rf, d.mkt, me, codes)
        self.beta[m] = bv
        if m - 1 in self.beta:
            sl = be.fm_slope(m - 1, self.signals, self.members_me, self.beta[m - 1], self.monthly, self.names)
            if sl is not None:
                self.slopes.append(sl)
                self.gf.add(sl)
        if not full:
            return
        bv = bv.fillna(bv.mean())
        sig, delta, n_full, short = be.covariance(d.simple, d.rf, d.mkt, me, codes, b_ols, resid)
        slopes = pd.DataFrame(self.slopes)
        if self.cfg.updater == "e3":
            gamma, info = self.gf.gamma()
            tau2 = info.get("tau2", 0.0)
        else:
            gamma, tau2 = be.gamma_rolling(slopes) if len(slopes) else (pd.Series(0.0, index=self.names), 0.0)
        a_trim, a_raw = be.alpha(self.signals, m, codes, gamma)
        adv = d.adv.loc[m].reindex(codes) if m in d.adv.index else pd.Series(np.nan, index=codes)
        adv = adv.fillna(adv.median()).clip(lower=1.0)
        row = self.pis.loc[str(m + 1)] if str(m + 1) in self.pis.index else None
        inp = {"codes": codes, "beta": bv.reindex(codes).to_numpy(), "Sigma": sig.to_numpy() * MONTH_DAYS,
               "alpha": a_trim.reindex(codes).to_numpy(), "adv": adv.to_numpy(), "gamma": gamma,
               "tau2": tau2, "lw_delta": delta, "n_short_hist": len(short), "alpha_sd": float(a_raw.std()),
               "n_slopes": len(slopes)}
        if self.cfg.return_model == "hist":
            inp["mu_hist"] = be.hist_mean(d.r, d.rf, me, codes).to_numpy()
        if row is not None:
            inp["pi_bull"] = (row["e3"] if self.cfg.updater == "e3" else row["d17"]) * MONTH_DAYS
            inp["pi_bear"] = (row["bear_capped"] if self.cfg.pi_bear == "capped" else 0.0) * MONTH_DAYS
            inp["pi_hist"] = row["hist"] * MONTH_DAYS
        self.inputs[m + 1] = inp


# ----------------------------------------------------------------------------------------------

class Simulator:
    def __init__(self, data, cfg, est, labels):
        self.d, self.cfg, self.est, self.p = data, cfg, est, cfg.params()
        self.codes = list(data.r.columns)
        self.col = {c: i for i, c in enumerate(self.codes)}
        A = lambda df: df.reindex(columns=self.codes).to_numpy(dtype=float)  # noqa: E731
        self.GP, self.DY = A(data.gp), A(data.dy)
        self.MEM = data.member.reindex(columns=self.codes).to_numpy(dtype=bool)
        self.TRD = data.traded.reindex(columns=self.codes).to_numpy(dtype=bool)
        self.NOBUY = data.no_buy.reindex(columns=self.codes).fillna(False).to_numpy(dtype=bool)
        self.NOSELL = data.no_sell.reindex(columns=self.codes).fillna(False).to_numpy(dtype=bool)
        self.SIG = np.nan_to_num(A(data.sigma_d), nan=np.nanmedian(A(data.sigma_d)))
        self.last = data.last_day.reindex(self.codes).to_numpy()
        self.labels = labels.reindex(data.dates, method="ffill") if labels is not None else None

    # -- helpers ---------------------------------------------------------------------------------
    def _sched(self, dates):
        s = pd.Series(dates, index=dates)
        if self.cfg.rebalance == "weekly":
            first = s.groupby(dates.to_period("W")).min()
        else:
            first = s.groupby(dates.to_period("M")).min()
            if self.cfg.rebalance == "quarterly":
                first = first[first.dt.month.isin([1, 4, 7, 10])]
        return set(first.values)

    def _label(self, i):
        j = i - self.cfg.delay
        if self.labels is None or j < 0:
            return 0
        v = self.labels.iloc[j]
        return 0 if pd.isna(v) else int(v)

    def _exec(self, idx, z, i, nav, liq):
        """Trade z (fractions of NAV) in names idx at the close of day i; returns turnover, costs."""
        if len(idx) == 0:
            return 0.0, 0.0, 0.0
        sd = self.SIG[i - 1, idx] if i > 0 else self.SIG[i, idx]
        lin = self.p.a * np.abs(z)
        imp = self.p.b * sd / np.sqrt(liq) * np.abs(z) ** 1.5
        self.pos[idx] += z * nav
        self.cash -= (z * nav).sum() + (lin + imp).sum() * nav
        return float(np.abs(z).sum()), float(lin.sum() * nav), float(imp.sum() * nav)

    # -- main loop -------------------------------------------------------------------------------
    def run(self, start, end):
        d, cfg, est = self.d, self.cfg, self.est
        dates = d.dates
        i0 = dates.searchsorted(pd.Timestamp(start))
        i1 = dates.searchsorted(pd.Timestamp(end), side="right")
        months = pd.period_range(dates[0].to_period("M"), dates[i0].to_period("M") - 1, freq="M")
        for k, m in enumerate(months):                         # warm-up: no trading
            est.process(m, full=(k == len(months) - 1))
        sched = self._sched(dates[i0:i1])
        N = len(self.codes)
        self.pos, self.cash = np.zeros(N), float(cfg.nav0)
        short_since = np.full(N, np.datetime64("NaT"), dtype="datetime64[ns]")
        rows, snaps, decisions = [], {}, []
        risk_flag, exec_day, exec_ctx = False, 0, None
        rate = d.rate.to_numpy()
        for i in range(i0, i1):
            t = dates[i]
            m_now = t.to_period("M")
            rec = {"date": t, "interest": 0.0, "fee": 0.0, "dividends": 0.0, "price_pnl": 0.0, "delisted": 0.0,
                   "turnover": 0.0, "cost_lin": 0.0, "cost_imp": 0.0, "event": "", "status": ""}
            if i > i0:
                gap = (t - dates[i - 1]).days
                rec["interest"] = self.cash * d.rf.iloc[i]
                short_val = -self.pos[self.pos < 0].sum()
                fee_rate = cfg.borrow + (0.0 if cfg.proceeds_earn else rate[i - 1] / 100)
                rec["fee"] = short_val * fee_rate * gap / 365
                held = np.nonzero(self.pos)[0]
                rec["dividends"] = float((self.pos[held] * self.DY[i, held]).sum())
                rec["price_pnl"] = float((self.pos[held] * (self.GP[i, held] - 1)).sum())
                self.pos[held] *= self.GP[i, held]
                self.cash += rec["interest"] - rec["fee"] + rec["dividends"]
                gone = held[self.last[held] < t]
                if len(gone):
                    rec["delisted"] = float(self.pos[gone].sum())
                    self.cash += self.pos[gone].sum()
                    self.pos[gone] = 0.0
            if i > i0 and m_now != dates[i - 1].to_period("M"):
                est.process(m_now - 1, full=True)
            nav = self.cash + self.pos.sum()
            inp = est.inputs.get(m_now)
            U = np.array([self.col[c] for c in inp["codes"]]) if inp else np.array([], int)
            inU = np.zeros(N, bool)
            inU[U] = True
            adv_all = np.full(N, np.nan)
            if inp:
                adv_all[U] = inp["adv"]
            med = np.nanmedian(inp["adv"]) if inp else 1.0
            adv_all = np.where(np.isnan(adv_all), med, adv_all)
            liq_all = adv_all / nav
            can_buy = self.TRD[i] & ~self.NOBUY[i]
            can_sell = self.TRD[i] & ~self.NOSELL[i]

            # index leavers and other holdings outside this month's universe: sold down (D18 (5))
            act = U[self.MEM[i, U]] if len(U) else U
            out = np.nonzero((self.pos != 0) & ~(inU & self.MEM[i]))[0]
            if len(out):
                w = self.pos[out] / nav
                cap = self.p.participation * liq_all[out] if cfg.fund_rules and not cfg.benchmark else np.full(len(out), np.inf)
                z = np.clip(-w, -cap, cap)
                z = np.where((z < 0) & ~can_sell[out], 0.0, z)
                z = np.where((z > 0) & ~can_buy[out], 0.0, z)
                tv, cl, ci = self._exec(out, z, i, nav, liq_all[out])
                rec["turnover"] += tv
                rec["cost_lin"] += cl
                rec["cost_imp"] += ci
            nav = self.cash + self.pos.sum()

            # decisions
            lab = self._label(i)
            regime_trig = cfg.regime != "off" and i > 0 and lab != self._label(i - 1) and not cfg.benchmark
            scheduled = dates[i] in sched
            event = "scheduled" if scheduled else "regime" if regime_trig else "risk" if (risk_flag and not cfg.benchmark) else ""
            if inp and len(act) and (event or exec_day):
                if event:
                    exec_day = 1
                    pi = inp.get("pi_hist", 0.0) if cfg.regime == "off" else (inp.get("pi_bear", 0.0) if lab else inp.get("pi_bull", 0.0))
                    mu_full = inp["mu_hist"] if cfg.return_model == "hist" else inp["beta"] * pi + inp["alpha"]
                    exec_ctx = {"pi": pi, "label": lab, "mu": dict(zip(inp["codes"], mu_full))}
                    rec["event"] = event
                else:
                    exec_day += 1
                    rec["event"] = f"exec{exec_day}"
                mu = np.array([exec_ctx["mu"][self.codes[j]] for j in act])
                held_out = self.pos.copy()
                held_out[act] = 0.0
                fixed_long = held_out[held_out > 0].sum() / nav
                fixed_short = -held_out[held_out < 0].sum() / nav
                w_prev = self.pos[act] / nav
                if cfg.benchmark:
                    w_new = self._benchmark_targets(act, mu)
                    w_new = np.where((w_new > w_prev) & ~can_buy[act], w_prev, w_new)
                    w_new = np.where((w_new < w_prev) & ~can_sell[act], w_prev, w_new)
                    status = "target"
                    exec_day = cfg.exec_days
                else:
                    where_U = {j: k for k, j in enumerate(U)}
                    pos_in_U = np.array([where_U[j] for j in act])
                    Sig = inp["Sigma"][np.ix_(pos_in_U, pos_in_U)]
                    rf_y = rate[i - 1] / 100 if i > 0 else rate[i] / 100
                    sd = self.SIG[i - 1, act] if i > 0 else self.SIG[i, act]
                    w_new, status = bo.solve(mu, Sig, w_prev, sd, liq_all[act], rf_y, self.p,
                                             can_buy[act], can_sell[act], fixed_long, fixed_short)
                z = w_new - w_prev
                diag = self._audit(w_prev, w_new, liq_all[act], can_buy[act], can_sell[act], fixed_long, fixed_short)
                tv, cl, ci = self._exec(act, z, i, nav, liq_all[act])
                rec["turnover"] += tv
                rec["cost_lin"] += cl
                rec["cost_imp"] += ci
                rec["status"] = status
                decisions.append({"date": t, "event": rec["event"], "label": exec_ctx["label"],
                                  "pi_m": exec_ctx["pi"], "turnover": tv, "status": status,
                                  "n_names": len(act), **diag})
                if tv < cfg.exec_stop or exec_day >= cfg.exec_days:
                    exec_day = 0

            # the 6-month short rule (D13 (3) sensitivity): close and re-open at this close
            if cfg.short_months:
                is_short = self.pos < 0
                short_since[~is_short] = np.datetime64("NaT")
                new = is_short & np.isnat(short_since)
                short_since[new] = np.datetime64(t)
                due = is_short & (short_since <= np.datetime64(t - pd.DateOffset(months=cfg.short_months)))
                if due.any():
                    j = np.nonzero(due)[0]
                    nav = self.cash + self.pos.sum()
                    w = np.abs(self.pos[j]) / nav
                    sd = self.SIG[i - 1, j]
                    c = 2 * (self.p.a * w + self.p.b * sd / np.sqrt(liq_all[j]) * w ** 1.5) * nav
                    self.cash -= c.sum()
                    rec["cost_lin"] += float(2 * self.p.a * w.sum() * nav)
                    rec["cost_imp"] += float(c.sum() - 2 * self.p.a * w.sum() * nav)
                    short_since[j] = np.datetime64(t)

            nav = self.cash + self.pos.sum()
            w_all = self.pos / nav
            k = 1 + cfg.risk_band                   # drift at a binding limit is not a breach
            risk_flag = bool((w_all.sum(where=w_all > 0) > k * self.p.long_max) or
                             (-w_all.sum(where=w_all < 0) > k * self.p.short_max) or
                             (cfg.fund_rules and ((w_all > k * self.p.w_max).any() or
                                                  (w_all < k * self.p.w_min).any())))
            beta_exp = float(inp["beta"] @ w_all[U]) if inp else 0.0
            rec.update(nav=nav, cash=self.cash, long=float(w_all[w_all > 0].sum()),
                       short=float(-w_all[w_all < 0].sum()), n_long=int((w_all > 1e-6).sum()),
                       n_short=int((w_all < -1e-6).sum()), beta_exp=beta_exp, label=lab,
                       risk_flag=risk_flag)
            rows.append(rec)
            if i + 1 == i1 or dates[i + 1].to_period("M") != m_now:
                snaps[t] = pd.Series(w_all, index=self.codes)[np.abs(w_all) > 1e-9]
        daily = pd.DataFrame(rows).set_index("date")
        daily["ret"] = daily["nav"].pct_change()
        daily.loc[daily.index[0], "ret"] = daily["nav"].iloc[0] / cfg.nav0 - 1
        weights = pd.DataFrame(snaps).T.fillna(0.0) if snaps else pd.DataFrame()
        return daily, weights, pd.DataFrame(decisions)

    def _audit(self, w_prev, w_new, liq, can_buy, can_sell, fixed_long, fixed_short, tol=1e-6):
        """Constraint checks of one decision (counts of names breaking a rule in a way the day's
        trade cap does not explain; 0 everywhere is the expected result)."""
        p, z = self.p, w_new - w_prev
        out = {"long_after": float(w_new[w_new > 0].sum() + fixed_long),
               "short_after": float(-w_new[w_new < 0].sum() + fixed_short),
               "blocked_trades": int((((z > tol) & ~can_buy) | ((z < -tol) & ~can_sell)).sum())}
        if not p.fund_rules or self.cfg.benchmark:
            return out
        tcap = p.participation * liq
        cap = p.liq_days * tcap
        could_sell = can_sell & (z > -tcap + tol)          # could have sold (covered) more today
        could_buy = can_buy & (z < tcap - tol)
        out.update(trade_cap_breaches=int((np.abs(z) > tcap + tol).sum()),
                   w_max_breaches=int(((w_new > p.w_max + tol) & could_sell).sum()),
                   w_min_breaches=int(((w_new < p.w_min - tol) & could_buy).sum()),
                   liq_breaches=int((((w_new > cap + tol) & could_sell) | ((w_new < -cap - tol) & could_buy)).sum()))
        return out

    def _benchmark_targets(self, act, mu):
        n = len(act)
        if self.cfg.benchmark == "1/N":
            return np.full(n, 1.0 / n)
        k = max(1, int(round(0.2 * n)))
        w = np.zeros(n)
        if np.ptp(mu) < 1e-12:            # [ours] nothing to sort (e.g. bear month with alpha = 0): cash
            return w
        order = np.argsort(mu, kind="stable")
        w[order[-k:]] = 1.0 / k
        w[order[:k]] = -1.0 / k
        return w
