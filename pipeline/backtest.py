"""Run the JMP backtest (D20): the baseline B0, the comparison runs C1-C5, the two simulated
benchmarks and the 15 pre-declared sensitivities (N = 23), under the Arnott, Harvey & Markowitz
(2019) protocol.

Protocol guard (D20 (2)):
  - a run that reaches the test period (any date from 2005-01-01) is refused unless the
    specification has been frozen (`--freeze`, which records the date and a hash of the backtest
    code in data/backtest/FREEZE.json);
  - every test-period run is appended to analysis/output/backtest/trials.csv with its config
    difference from B0, the code hash and whether it matches the frozen hash. A run whose code
    differs from the frozen code is a new trial (it counts in N for the deflated Sharpe ratio);
  - runs that stop before 2005 are mechanics checks: they go to pretest_runs.csv, not the trials.

Outputs per run (analysis/output/backtest/{pretest,runs}/<name>/): daily.parquet (NAV, P&L parts,
exposures, events), weights.parquet (month-end weights), decisions.csv, summary.json.

Run:  python pipeline/backtest.py --list
      python pipeline/backtest.py --runs B0 --start 2002-01-01 --end 2004-12-31     (pre-test)
      python pipeline/backtest.py --runs all --start 2002-01-01 --end 2004-12-31 --jobs 6
      python pipeline/backtest.py --freeze                                          (the user's call)
      python pipeline/backtest.py --runs all --jobs 6                               (after the freeze)
"""
import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

import bt_data
import bt_regime
import bt_updater
import evaluation as ev
from bt_simulator import Config, Estimator, Simulator
from config import DATA_DIR, PROJECT_DIR

TEST_START = bt_data.TEST_START
TEST_END = pd.Timestamp("2025-09-30")
OUT = PROJECT_DIR / "analysis" / "output" / "backtest"
FREEZE = DATA_DIR / "backtest" / "FREEZE.json"
CODE = ["bt_data.py", "bt_signals.py", "bt_regime.py", "bt_estimation.py", "bt_updater.py",
        "bt_optimizer.py", "bt_simulator.py", "backtest.py", "bt_report.py", "evaluation.py", "config.py"]

B0 = Config()
RUNS = {r.name: r for r in [
    B0,
    # comparison runs, one per component (D20 (4))
    replace(B0, name="C1_regime_off", regime="off"),
    replace(B0, name="C2_no_cost_terms", g_trade=0.0, g_hold=0.0),
    replace(B0, name="C3_updater_off", updater="off"),
    replace(B0, name="C4_fund_rules_relaxed", fund_rules=False),
    replace(B0, name="C5_hist_mean", return_model="hist"),
    # simulated benchmarks (D20 (5))
    replace(B0, name="BM_1N", benchmark="1/N"),
    replace(B0, name="BM_quintile", benchmark="quintile"),
    # sensitivities (D20 (8))
    replace(B0, name="S_borrow_035", borrow=0.0035, proceeds_earn=True),
    replace(B0, name="S_short_6m", short_months=6),
    replace(B0, name="S_cost_010", cost_a=0.0010),
    replace(B0, name="S_cost_025", cost_a=0.0025),
    replace(B0, name="S_weekly", rebalance="weekly"),
    replace(B0, name="S_quarterly", rebalance="quarterly"),
    replace(B0, name="S_pi_bear_capped", pi_bear="capped"),
    replace(B0, name="S_delay_5", delay=5),
    replace(B0, name="S_delay_10", delay=10),
    replace(B0, name="S_hmm", regime="hmm"),
    replace(B0, name="S_grisk_5", g_risk=5.0),
    replace(B0, name="S_grisk_20", g_risk=20.0),
    replace(B0, name="S_nav_100bn", nav0=100e9),
    replace(B0, name="S_delta_1", deltas=(1.0,)),
    replace(B0, name="S_no_imputed_div", imputed_dividends=False),
]}


def code_hash():
    h = hashlib.sha256()
    for f in CODE:
        h.update((Path(__file__).parent / f).read_bytes())
    return h.hexdigest()


def frozen():
    return json.loads(FREEZE.read_text(encoding="utf-8")) if FREEZE.exists() else None


def guard(end):
    """Refuse a test-period run before the freeze (D20 (2))."""
    if pd.Timestamp(end) >= TEST_START and frozen() is None:
        raise SystemExit("Refused: this run reaches the test period (2005+) and the specification is not "
                         "frozen. Freeze first (python pipeline/backtest.py --freeze), on the user's decision.")


def monthly_inputs(est, first, last):
    """The optimizer inputs of each simulated month: per stock (alpha, beta, the C5 mean) and per
    month (premia, signal weights, shrinkage), for the D20 (7) secondary analyses."""
    rows, meta = [], {}
    for M in pd.period_range(first.to_period("M"), last.to_period("M"), freq="M"):
        inp = est.inputs.get(M)
        if inp is None:
            continue
        df = pd.DataFrame({"month": str(M), "code": inp["codes"], "alpha": inp["alpha"], "beta": inp["beta"]})
        if "mu_hist" in inp:
            df["mu_hist"] = inp["mu_hist"]
        rows.append(df)
        meta[str(M)] = {"pi_bull": inp.get("pi_bull"), "pi_bear": inp.get("pi_bear"), "pi_hist": inp.get("pi_hist"),
                        "tau2": inp["tau2"], "lw_delta": inp["lw_delta"], "alpha_sd": inp["alpha_sd"],
                        "n_slopes": inp["n_slopes"], **{f"gamma_{k}": v for k, v in inp["gamma"].items()}}
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), pd.DataFrame(meta).T


def run(cfg, start, end, write=True, verbose=True, log=True):
    guard(end)
    t0 = time.time()
    end = min(pd.Timestamp(end), TEST_END)
    data = bt_data.load(end, imputed_dividends=cfg.imputed_dividends)
    reg = bt_regime.run(end, kind="hmm" if cfg.regime == "hmm" else "jm")
    pis = bt_updater.pi_path(reg, deltas=cfg.deltas)
    est = Estimator(data, cfg, pis)
    sim = Simulator(data, cfg, est, reg.labels)
    daily, weights, decisions = sim.run(start, end)
    inputs, meta = monthly_inputs(est, daily.index[0], daily.index[-1])
    rf = data.rf.reindex(daily.index)
    m = ev.metrics(daily["ret"], rf, weights=None)
    years = len(daily) / bt_data.YEAR_DAYS
    summary = {"name": cfg.name, "start": f"{daily.index[0]:%Y-%m-%d}", "end": f"{daily.index[-1]:%Y-%m-%d}",
               **{k: float(v) for k, v in m.items()},
               "turnover/yr": float(daily["turnover"].sum() / years),
               "costs %/yr": float((daily["cost_lin"] + daily["cost_imp"]).sum() / daily["nav"].mean() / years),
               "avg long": float(daily["long"].mean()), "avg short": float(daily["short"].mean()),
               "avg beta exposure": float(daily["beta_exp"].mean()),
               "decisions": int((decisions["event"].isin(["scheduled", "regime", "risk"])).sum()) if len(decisions) else 0,
               "solver failures": int((~decisions["status"].isin(["optimal", "target", "scs:optimal",
                                                                 "scs:optimal_inaccurate"])).sum()) if len(decisions) else 0,
               "seconds": round(time.time() - t0, 1)}
    if write:
        test = pd.Timestamp(end) >= TEST_START
        d = OUT / ("runs" if test else "pretest") / cfg.name
        d.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(d / "daily.parquet")
        weights.to_parquet(d / "weights.parquet")
        decisions.to_csv(d / "decisions.csv", index=False)
        inputs.to_parquet(d / "inputs.parquet")
        meta.to_csv(d / "inputs_meta.csv")
        (d / "summary.json").write_text(json.dumps({"summary": summary, "config": asdict(cfg)}, indent=1,
                                                   default=str), encoding="utf-8")
        if log:
            write_log(cfg, summary)
    if verbose:
        print(json.dumps(summary, indent=1))
    return daily, weights, decisions, summary


def write_log(cfg, summary):
    """One row per run in trials.csv (test period) or pretest_runs.csv (D20 (2)); written by one
    process only."""
    row = {"time": pd.Timestamp.now().isoformat(timespec="seconds"), "name": cfg.name,
           "start": summary["start"], "end": summary["end"], "config_diff": json.dumps(cfg.diff(B0), default=str),
           "code_sha256": code_hash()[:16],
           "matches_freeze": (frozen() or {}).get("code_sha256", "")[:16] == code_hash()[:16],
           **{k: summary[k] for k in ("CAGR", "vol", "Sharpe", "MDD", "turnover/yr", "costs %/yr",
                                      "decisions", "solver failures", "seconds")}}
    f = OUT / ("trials.csv" if pd.Timestamp(summary["end"]) >= TEST_START else "pretest_runs.csv")
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(f, mode="a", header=not f.exists(), index=False)


def _worker(args):
    name, start, end = args
    return name, run(RUNS[name], start, end, log=False, verbose=False)[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="B0", help="comma-separated run names, or 'all'")
    ap.add_argument("--start", default=str(TEST_START.date()))
    ap.add_argument("--end", default=str(TEST_END.date()))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="runs in parallel processes")
    a = ap.parse_args()
    if a.list:
        for k, c in RUNS.items():
            print(f"{k:24s} {c.diff(B0)}")
        return
    if a.freeze:
        if FREEZE.exists():
            raise SystemExit(f"Already frozen: {FREEZE.read_text(encoding='utf-8')}")
        FREEZE.parent.mkdir(parents=True, exist_ok=True)
        info = {"frozen_at": pd.Timestamp.now().isoformat(timespec="seconds"), "decisions": "D1-D21",
                "code_sha256": code_hash(), "runs": list(RUNS)}
        FREEZE.write_text(json.dumps(info, indent=1), encoding="utf-8")
        print(json.dumps(info, indent=1))
        return
    names = list(RUNS) if a.runs == "all" else a.runs.split(",")
    guard(a.end)
    if a.jobs <= 1:
        for n in names:
            run(RUNS[n], a.start, a.end)
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futs = {pool.submit(_worker, (n, a.start, a.end)): n for n in names}
        for fu in as_completed(futs):
            n = futs[fu]
            try:
                _, s = fu.result()
            except Exception as e:                       # logged, the batch goes on
                print(f"{n}: FAILED {e!r}", flush=True)
                continue
            write_log(RUNS[n], s)
            print(f"{n}: Sharpe {s['Sharpe']:.2f}, CAGR {s['CAGR']:.2%}, {s['decisions']} decisions, "
                  f"{s['solver failures']} solver failures, {s['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
