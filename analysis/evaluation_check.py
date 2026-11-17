"""D20 evaluation toolkit, tried on data from BEFORE the test period, plus the power of the test
period. Nothing here reads a return after 2004.

Part A  The E1 replication's strategies on ^N225, 1990-2004 (analysis/regime_jm.py): buy-and-hold,
        the HMM 0/1 and the jump-model 0/1 strategy. The D20 statistics are computed on their
        monthly excess returns: Newey-West t, Lo (2002) standard error, probabilistic and deflated
        Sharpe ratios, the Ledoit-Wolf (2008) bootstrap test of Sharpe differences, Holm.
        This checks the code and shows what 15 years can and cannot establish.
Part B  Power of the 2005-01 -> 2025-09 test (20.75 years, monthly): the smallest Sharpe ratio
        and Sharpe difference a one-sided / two-sided 5% test detects with 80% power; and the
        Sharpe ratio a deflated-Sharpe test needs when N variants are run.

Outputs (analysis/output/evaluation/): summary.txt

Run:  python analysis/evaluation_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROJECT_DIR  # noqa: E402
import evaluation as ev  # noqa: E402
import regime_jm as rj  # noqa: E402

OUT = PROJECT_DIR / "analysis" / "output" / "evaluation"
TEST_YEARS = 249 / 12            # 2005-01 .. 2025-09


def part_a():
    f = rj.load()
    lab = pd.read_parquet(PROJECT_DIR / "analysis" / "output" / "regime_jm" / "labels.parquet")
    oos = f.loc["1990":]
    rets = {"buy and hold": oos["r"]}
    weights = {"buy and hold": pd.Series(1.0, index=oos.index)}
    for name, col in (("HMM 0/1", "hmm_bear"), ("jump model 0/1", "jm_bear")):
        r, w = rj.strategy(f, lab[col])
        rets[name], weights[name] = r.loc["1990":], w.loc["1990":]
    rf = f["rf"]
    rows, ex_m = {}, {}
    for name, r in rets.items():
        m = ev.metrics(r, rf, weights[name], year_days=rj.YEAR_DAYS)
        x = ev.to_monthly(r) - ev.to_monthly(rf.reindex(r.index))
        ex_m[name] = x
        t = ev.nw_mean(x)
        sr_m, se_m = ev.sr_se_hac(x)
        m.update({"NW t (mean excess)": t["t"], "SR s.e. (Lo, annual)": se_m * np.sqrt(12),
                  "PSR (SR > 0)": ev.psr(sr_m, len(x), stats.skew(x), stats.kurtosis(x, fisher=False))})
        rows[name] = m
    table = pd.DataFrame(rows).T
    tests = {}
    for a, b in (("jump model 0/1", "buy and hold"), ("jump model 0/1", "HMM 0/1")):
        d = pd.concat([ex_m[a], ex_m[b]], axis=1).dropna()
        tests[f"{a} vs {b}"] = ev.lw_sharpe_test(d.iloc[:, 0], d.iloc[:, 1], M=4999, K=500, seed=1)
        tests[f"{a} vs {b}"]["corr"] = float(d.corr().iloc[0, 1])
    adj = ev.holm({k: v["p_boot"] for k, v in tests.items()})
    # DSR of the jump model if both regime models count as trials (N = 2), and if 25 had been run
    x = ex_m["jump model 0/1"]
    sr_m = x.mean() / x.std()
    trial = [ex_m[k].mean() / ex_m[k].std() for k in ("HMM 0/1", "jump model 0/1")]
    dsr2, sr0_2 = ev.dsr(sr_m, len(x), stats.skew(x), stats.kurtosis(x, fisher=False), trial)
    return table, tests, adj, (dsr2, sr0_2)


def part_b():
    L = []
    L.append(f"Test period {TEST_YEARS:.2f} years of monthly returns")
    L.append(f"  smallest annual Sharpe ratio detected (one-sided 5%, power 80%): "
             f"{ev.power_min_sr(TEST_YEARS):.2f}; 50% power: {ev.power_min_sr(TEST_YEARS, power=0.5):.2f}")
    for sr in (0.3, 0.5, 0.7, 1.0):
        L.append(f"  expected t for a true Sharpe of {sr}: {sr * np.sqrt(TEST_YEARS):.2f}")
    L.append("  smallest annual Sharpe difference detected (two-sided 5%, power 80%), by correlation "
             "of the paired strategies:")
    for rho in (0.5, 0.8, 0.9, 0.95, 0.99):
        L.append(f"    rho {rho:.2f}: {ev.power_min_sr_diff(TEST_YEARS, rho):.2f}")
    L.append("  Sharpe ratio needed for a deflated Sharpe ratio of 0.95 (normal returns), N variants "
             "with a cross-variant s.d. of Sharpe ratios of 0.2 a year:")
    T = int(TEST_YEARS * 12)
    for N in (1, 2, 5, 10, 25, 50):                # N = 2: signal lists v1 and v2 (D12, D21)
        trials = np.random.default_rng(0).normal(0, 0.2 / np.sqrt(12), N) if N > 1 else [0.0]
        if N > 1:                                   # set the cross-variant s.d. exactly
            trials = (trials - trials.mean()) / trials.std(ddof=1) * 0.2 / np.sqrt(12)
        grid = np.linspace(0, 3, 3001) / np.sqrt(12)
        need = next(s for s in grid if ev.dsr(s, T, 0.0, 3.0, trials)[0] >= 0.95)
        L.append(f"    N = {N:2d}: {need * np.sqrt(12):.2f}")
    return L


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table, tests, adj, (dsr2, sr0_2) = part_a()
    L = ["D20 evaluation toolkit on pre-test data, and the power of the test period", "",
         "Part A: E1 replication strategies, ^N225, 1990-01 .. 2004-12 (monthly excess returns)"]
    cols = ["CAGR", "vol", "Sharpe", "SR s.e. (Lo, annual)", "NW t (mean excess)", "PSR (SR > 0)",
            "Sortino", "CEQ", "MDD", "Calmar", "ES 5% (daily)", "turnover/yr"]
    with pd.option_context("display.width", 200):
        L.append(table[cols].to_string(float_format=lambda v: f"{v:.3f}"))
    L.append("")
    L.append("Ledoit-Wolf (2008) test of the Sharpe difference (monthly SR units; M = 4999, K = 500)")
    for k, v in tests.items():
        L.append(f"  {k}: difference {v['delta'] * np.sqrt(12):+.2f}/yr, correlation {v['corr']:.2f}, "
                 f"block {v['block']} (coverage {', '.join(f'{b}:{c:.2f}' for b, c in v['coverage'].items())}), "
                 f"p HAC {v['p_hac']:.3f}, p bootstrap {v['p_boot']:.3f}, Holm-adjusted {adj[k]:.3f}")
    L.append(f"Deflated Sharpe ratio of the jump model with N = 2 (both regime models): {dsr2:.3f} "
             f"(hurdle {sr0_2 * np.sqrt(12):.2f}/yr)")
    L.append("")
    L.append("Part B: power of the test period")
    L += part_b()
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
