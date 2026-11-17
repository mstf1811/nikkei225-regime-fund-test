"""Checks of the backtest code, on data to 2004-12-31 only (the test period stays unread).

Data:
    data         exp(return) = price factor + dividend exactly; the no-imputed-dividend load removes
                 exactly the imputed part (D20 (8) sensitivity)
Reproduction (the pipeline ports must give the analysis checks' numbers):
    regime       labels and lambda path = analysis/regime_jm.py (D17 replication)
    pi_path      the E3 and D17 bull premia 1990-2004 = analysis/updater_check.py part A; 4.9%/yr for 2005-01
    signals      the six and eight signal panels = analysis/optimizer_check.signal_panels
    inputs       beta, alpha, ADV and the slope count for 2005-01 = analysis/optimizer_check.py (six
                 signals) and analysis/signals_v2_check.py (eight); the E3 signal weights = their
                 gamma_dlm.csv
    optimizer    the day-1 portfolio from cash at 2005-01 = the D18 check (long 0.300, short 0.095)
Simulation:
    accounting   every day, NAV change = interest - fee + dividends + price P&L - costs (to 1e-9 of NAV);
                 every decision passes the constraint audit (no unexplained breach, no blocked trade)
    cash_only    with mu = 0 the fund holds cash and its NAV compounds the call rate exactly
    truncation   a run to 2004-06-30 and one to 2004-12-31 have the same inputs and labels (to
                 rounding) and the same NAV (to the solver's tolerance) over their common days: no
                 look-ahead anywhere in data, signals, regime, estimation or the simulation

Run:  python pipeline/bt_tests.py [test names]
"""
import sys
import time

import numpy as np
import pandas as pd

import bt_data
import bt_optimizer as bo
import bt_regime
import bt_updater
from bt_signals import SIGNALS_V1, SIGNALS_V2
from bt_simulator import Config, Estimator, Simulator
from config import PROJECT_DIR

END = "2004-12-31"
A = PROJECT_DIR / "analysis" / "output"
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def read_inputs(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


def estimator_at(cfg, data, pis, last=pd.Period("2004-12", "M")):
    est = Estimator(data, cfg, pis)
    for m in pd.period_range(data.dates[0].to_period("M"), last, freq="M"):
        est.process(m, full=(m == last))
    return est


# ----------------------------------------------------------------------------------------------

def t_regime():
    res = bt_regime.run(END)
    old = pd.read_parquet(A / "regime_jm" / "labels.parquet")["jm_bear"].dropna()
    common = old.index.intersection(res.labels.index)
    mism = int((old.loc[common] != res.labels.loc[common]).sum())
    check("regime labels", mism == 0 and len(common) == len(old), f"{len(common)} days, {mism} mismatches")
    op = pd.read_csv(A / "regime_jm" / "lambda_path.csv").merge(res.path, on="month", suffixes=("_o", "_n"))
    mism = int((~np.isclose(op["choice_o"], op["choice_n"])).sum())
    check("lambda path", mism == 0 and len(op) == 180, f"{len(op)} months, {mism} mismatches")


def t_pi_path():
    p = bt_updater.pi_path(bt_regime.run(END))
    old = pd.read_csv(A / "updater" / "pi_path.csv", index_col="month")
    j = old.join(p, how="inner")
    e3 = float((j["E3"] - j["e3"]).abs().max())
    d17 = float((j["D17"] - j["d17"]).abs().max())
    check("pi path E3 and D17 1990-2004", len(j) == 180 and e3 < 1e-12 and d17 < 1e-12,
          f"{len(j)} months, max diff {e3:.1e} / {d17:.1e}")
    v = p.loc["2005-01", "e3"] * bt_data.YEAR_DAYS
    check("pi_bull for 2005-01 = 4.9%/yr (D19)", abs(v - 0.049) < 0.0005, f"{v:.4f}")


def t_signals():
    sys.path.insert(0, str(PROJECT_DIR / "analysis"))
    import optimizer_check as oc
    from bt_signals import build
    data = bt_data.load(END)
    px, liq, ch, rate = oc.load()
    r, simple, member, rf, mkt = oc.daily_panels(px, ch, rate)
    for names in (SIGNALS_V1, SIGNALS_V2):
        s_old, m_old = oc.signal_panels(px, liq, ch, simple, rf, mkt, names)
        s_new, m_new = build(data, names)
        worst = 0.0
        for k in names:
            a, b = s_old[k], s_new[k].reindex_like(s_old[k])
            same_nan = (a.isna() == b.isna()).all().all()
            worst = max(worst, float((a - b).abs().max().max()) if same_nan else np.inf)
        check(f"signal panels ({len(names)} signals)", worst < 1e-12 and m_old.equals(m_new.reindex_like(m_old)),
              f"max diff {worst:.1e}")


def t_inputs():
    data = bt_data.load(END)
    pis = bt_updater.pi_path(bt_regime.run(END))
    for names, ref, n_slopes in ((SIGNALS_V1, A / "optimizer", 46), (SIGNALS_V2, A / "signals_v2" / "optimizer", 23)):
        est = estimator_at(Config(signals=tuple(names), updater="off"), data, pis)
        inp = est.inputs[pd.Period("2005-01", "M")]
        old = read_inputs(ref / "inputs_2005-01.csv").reindex(inp["codes"])
        diffs = {"beta": np.abs(inp["beta"] - old["beta_vasicek"].to_numpy()).max(),
                 "alpha": np.abs(inp["alpha"] - old["alpha_trim_m"].to_numpy()).max(),
                 "adv": np.abs(inp["adv"] / 1e9 - old["adv_bn"].to_numpy()).max()}
        check(f"2005-01 inputs ({len(names)} signals, D15 gamma)",
              max(diffs.values()) < 1e-10 and inp["n_slopes"] == n_slopes,
              f"{len(inp['codes'])} names, {inp['n_slopes']} slope months, max diffs " +
              ", ".join(f"{k} {v:.1e}" for k, v in diffs.items()))
        est3 = estimator_at(Config(signals=tuple(names), updater="e3"), data, pis)
        g = est3.inputs[pd.Period("2005-01", "M")]["gamma"]
        ref_g = pd.read_csv((A / "updater" if len(names) == 6 else A / "signals_v2") / "gamma_dlm.csv",
                            index_col=0)["E3 (EB)"]
        dg = float((g - ref_g.reindex(g.index)).abs().max())
        check(f"E3 signal weights at 2005-01 ({len(names)} signals)", dg < 1e-12, f"max diff {dg:.1e}")


def t_optimizer():
    data = bt_data.load(END)
    pis = bt_updater.pi_path(bt_regime.run(END))
    inp = estimator_at(Config(signals=tuple(SIGNALS_V1), updater="off"), data, pis).inputs[pd.Period("2005-01", "M")]
    f = bt_regime.load_index(END)
    _, st = bt_regime.jm_period(f, pd.Timestamp("2005-01-01"), pd.Timestamp("2005-01-02"), 33.36)
    pi = st["bull_mean"] * bt_data.MONTH_DAYS
    mu = inp["beta"] * pi + inp["alpha"]
    nav = 10e9
    sd = data.sigma_d.loc["2004-12-30", inp["codes"]].to_numpy()
    sd = np.where(np.isnan(sd), np.nanmedian(sd), sd)
    ones = np.ones(len(mu), bool)
    w, status = bo.solve(mu, inp["Sigma"], np.zeros(len(mu)), sd, inp["adv"] / nav,
                         data.rate.iloc[-1] / 100, bo.Params(), ones, ones)
    long, short = w[w > 0].sum(), -w[w < 0].sum()
    check("day-1 portfolio from cash = D18 check (0.300 / 0.095 / 0.204)",
          round(long, 3) == 0.300 and round(short, 3) == 0.095 and round(long - short, 3) == 0.204,
          f"long {long:.4f}, short {short:.4f}, pi_bull {pi * 12:.3f}/yr, {status}")


def reconcile(daily, nav0):
    prev = daily["nav"].shift(1).fillna(nav0)
    pnl = daily["interest"] - daily["fee"] + daily["dividends"] + daily["price_pnl"] - daily["cost_lin"] - daily["cost_imp"]
    return float(((daily["nav"] - prev - pnl).abs() / prev).max())


ACCOUNTING = {"T_v1_D15": Config(name="T_v1_D15", signals=tuple(SIGNALS_V1), updater="off"),
              "T_B0": Config(name="T_B0"),
              "T_C4": Config(name="T_C4", signals=tuple(SIGNALS_V1), updater="off", fund_rules=False),
              "T_6m": Config(name="T_6m", signals=tuple(SIGNALS_V1), updater="off", short_months=6),
              "T_quintile": Config(name="T_quintile", signals=tuple(SIGNALS_V1), updater="off", benchmark="quintile")}


def t_accounting(only=None):
    for cfg in (ACCOUNTING[k] for k in (only or ACCOUNTING)):
        t0 = time.time()
        sim, daily, dec = simulate(cfg, "2002-01-01", END)
        err = reconcile(daily, cfg.nav0)
        cols = [c for c in ("trade_cap_breaches", "w_max_breaches", "w_min_breaches", "liq_breaches",
                            "blocked_trades") if c in dec]
        breaches = {c: int(dec[c].sum()) for c in cols}
        box = float(max(dec["long_after"].max(), dec["short_after"].max()))
        fails = int((~dec["status"].isin(["optimal", "target", "scs:optimal", "scs:optimal_inaccurate"])).sum())
        check(f"accounting and audit: {cfg.name}", err < 1e-9 and sum(breaches.values()) == 0 and box < 1 + 1e-6
              and fails == 0,
              f"NAV recon err {err:.1e}; breaches {breaches}; max box {box:.4f}; solver failures {fails}; "
              f"{len(dec)} solves; avg long/short {daily['long'].mean():.2f}/{daily['short'].mean():.2f}; "
              f"{time.time() - t0:.0f}s")


def simulate(cfg, start, end, est_cls=Estimator):
    data = bt_data.load(end, imputed_dividends=cfg.imputed_dividends)
    reg = bt_regime.run(end)
    est = est_cls(data, cfg, bt_updater.pi_path(reg, deltas=cfg.deltas))
    sim = Simulator(data, cfg, est, reg.labels)
    daily, weights, dec = sim.run(start, end)
    return sim, daily, dec


class ZeroMu(Estimator):
    def process(self, m, full):
        super().process(m, full)
        if full:
            inp = self.inputs[m + 1]
            inp["alpha"] = np.zeros_like(inp["alpha"])
            inp["pi_bull"] = inp["pi_bear"] = 0.0


def t_cash_only():
    cfg = Config(name="T_cash")
    sim, daily, dec = simulate(cfg, "2004-01-01", END, ZeroMu)
    data = sim.d
    rf = data.rf.reindex(daily.index)
    expected = cfg.nav0 * (1 + rf.iloc[1:]).cumprod()
    err = float((daily["nav"].iloc[1:] / expected - 1).abs().max())
    check("cash only: NAV = call-rate compounding", err < 1e-12 and daily["long"].max() == 0 and daily["short"].max() == 0,
          f"max rel err {err:.1e}; final NAV {daily['nav'].iloc[-1]:,.0f}")


def t_truncation():
    """Inputs must agree to rounding (1e-12): the shorter cut has fewer stock columns, so row sums
    run in a different order (differences of ~1e-16). The NAVs then agree to the solver's tolerance
    (SCS at 1e-7, D18 (9)), not bit for bit."""
    cfg = Config(name="T_trunc", signals=tuple(SIGNALS_V1), updater="off")
    sa, a, _ = simulate(cfg, "2004-01-01", "2004-06-30")
    sb, b, _ = simulate(cfg, "2004-01-01", END)
    worst = 0.0
    for M in pd.period_range("2004-01", "2004-06", freq="M"):
        x, y = sa.est.inputs[M], sb.est.inputs[M]
        same = x["codes"] == y["codes"] and x["pi_bull"] == y["pi_bull"] and x["pi_bear"] == y["pi_bear"]
        worst = max(worst, *(float(np.abs(x[k] - y[k]).max()) for k in ("beta", "alpha", "Sigma", "adv"))) if same else np.inf
    check("truncation: inputs to 2004-06 identical to rounding", worst < 1e-12, f"max diff {worst:.1e}")
    lab = sa.labels.loc[:"2004-06-30"].equals(sb.labels.loc[:"2004-06-30"])
    check("truncation: regime labels identical", lab)
    common = a.index.intersection(b.index)
    diff = float((a.loc[common, "nav"] / b.loc[common, "nav"] - 1).abs().max())
    check("truncation: NAV agrees to the solver tolerance", diff < 1e-6 and len(common) == len(a),
          f"{len(common)} common days, max rel diff {diff:.1e}")


def t_data():
    a, b = bt_data.load(END), bt_data.load(END, imputed_dividends=False)
    lr = lambda d: np.exp(d.px.pivot(index="date", columns="code", values="log_return"))  # noqa: E731
    ga, gb = lr(a), lr(b).reindex_like(lr(a))
    e1 = float(np.nanmax(np.abs((ga - (a.gp + a.dy)).values)))
    check("returns split exactly into price factor + dividend", e1 < 1e-12 and bool((a.dy >= -1e-12).all().all()),
          f"max err {e1:.1e}")
    e2 = float(np.nanmax(np.abs((gb - (ga - a.dy_imp)).values)))
    check("no-imputed-dividend data: imputed part removed exactly", e2 < 1e-12 and float(b.dy_imp.abs().sum().sum()) == 0
          and bool((b.dy >= -1e-12).all().all()), f"max err {e2:.1e}")


TESTS = {"data": t_data, "regime": t_regime, "pi_path": t_pi_path, "signals": t_signals, "inputs": t_inputs,
         "optimizer": t_optimizer, "accounting": t_accounting, "cash_only": t_cash_only,
         "truncation": t_truncation, "accounting_C4": lambda: t_accounting(["T_C4"])}

if __name__ == "__main__":
    names = sys.argv[1:] or list(TESTS)
    for n in names:
        print(n)
        TESTS[n]()
    print(f"\n{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
    sys.exit(1 if FAILS else 0)
