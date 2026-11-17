"""D21 (signal list v2) checked on data from BEFORE the test period (to 2004-12-30).

Adds residual momentum and residual reversal (analysis/residual_signals.py) to the six D12 signals.
No return after 2004 is read, and no forward-return performance of the new signals is measured: under
D12 they stay in whatever they show, and the test period stays clean (D20 (2)).

Part A  construction: coverage of the members, average cross-sectional rank correlation with the
        six D12 signals, and the papers' main claim, that residual signals carry much smaller factor
        tilts than total-return ones (Blitz, Huij & Martens 2011; Blitz et al. 2013): each month,
        the rank correlation across members between the signal and the stocks' 36-month loadings
        on the market, SMB and HML.
Part B  the signal weights at the 2005-01 start with eight signals: D15 (rolling Fama-MacBeth +
        empirical Bayes) and D19 (E3 filters), as in analysis/updater_check.py part B.
Part C  the first portfolio of 2005 with eight signals (analysis/optimizer_check.py, D18 settings).

Outputs (analysis/output/signals_v2/): summary.txt, coverage.csv, signal_corr.csv, factor_tilts.csv,
gamma_dlm.csv, optimizer/ (summary.txt, inputs_2005-01.csv, weights_2005-01.csv)

Run:  python analysis/signals_v2_check.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROJECT_DIR  # noqa: E402
import optimizer_check as oc  # noqa: E402
import updater_check as uc  # noqa: E402
from residual_signals import SIGNALS_V2, residual_signals  # noqa: E402

warnings.filterwarnings("ignore")
OUT = PROJECT_DIR / "analysis" / "output" / "signals_v2"
FIRST = pd.Period("2003-01", "M")       # first month with a 36-month window (data from 2000-02)
LAST = pd.Period("2004-12", "M")
PAIRS = [("mom", "resmom"), ("rev", "resrev")]
LOADINGS = ["b_mkt", "b_smb", "b_hml"]


def spearman(a, b):
    ok = a.notna() & b.notna()
    return a[ok].rank().corr(b[ok].rank()) if ok.sum() >= oc.FM_MIN_STOCKS else np.nan


def part_a():
    px, liq, ch, rate = oc.load()
    r, simple, member, rf, mkt = oc.daily_panels(px, ch, rate)
    signals, members_me = oc.signal_panels(px, liq, ch, simple, rf, mkt, SIGNALS_V2)
    res = residual_signals(simple, rf, mkt)
    months = [m for m in signals["rev"].index if FIRST <= m <= LAST]

    cov = pd.DataFrame({k: [signals[k].loc[m][members_me.loc[m]].notna().sum() for m in months]
                        for k in SIGNALS_V2}, index=months)
    cov.insert(0, "members", [int(members_me.loc[m].sum()) for m in months])

    corr = sum(pd.DataFrame({k: signals[k].loc[m][members_me.loc[m]] for k in SIGNALS_V2})
               .rank().corr() for m in months) / len(months)

    tilt = []
    for m in months:
        mem = members_me.loc[m]
        for sig in [s for p in PAIRS for s in p]:
            for b in LOADINGS:
                load = res[b].reindex(columns=signals[sig].columns).loc[m][mem]
                tilt.append({"month": m, "signal": sig, "loading": b,
                             "corr": spearman(signals[sig].loc[m][mem], load)})
    tilt = pd.DataFrame(tilt)
    tilt_sum = tilt.groupby(["signal", "loading"])["corr"].agg(
        mean="mean", mean_abs=lambda c: c.abs().mean(), sd="std", min="min", max="max").unstack("loading")
    return cov, corr, tilt_sum


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cov, corr, tilt = part_a()
    cov.to_csv(OUT / "coverage.csv")
    corr.to_csv(OUT / "signal_corr.csv")
    tilt.to_csv(OUT / "factor_tilts.csv")
    L = ["D21 check on pre-test data (to 2004-12-30)", "",
         f"Part A: construction, signal months {FIRST} .. {LAST}",
         "members with each signal (mean / min / at 2004-12): " +
         ", ".join(f"{k} {cov[k].mean():.0f}/{cov[k].min()}/{cov[k].iloc[-1]}" for k in cov.columns), "",
         "average cross-sectional Spearman correlation between the signals:",
         corr.round(2).to_string(), "",
         "factor tilts: Spearman correlation across members between the signal and the stocks' 36-month",
         "loadings (mean |corr| is the size of the tilt whatever its sign; sd is how much it moves):"]
    with pd.option_context("display.width", 200):
        L.append(tilt[["mean_abs", "mean", "sd"]].round(2).to_string())

    d, n, first, last = uc.part_b(SIGNALS_V2)
    d.to_csv(OUT / "gamma_dlm.csv")
    wcols = [f"w{x}" for x in uc.DELTAS]
    L += ["", f"Part B: signal weights at the 2005-01 start, eight signals, monthly units; {n} slope months "
          f"{first} .. {last} (months where all eight cover >= {oc.FM_MIN_STOCKS} members)",
          d[["D15 FM mean", "D15 FM se", "D15 (EB)", "mean", "sd", "E3 (EB)"] + wcols].to_string(
              float_format=lambda v: f"{v:.5f}")]

    res, fm = oc.main(names=SIGNALS_V2, out=OUT / "optimizer")
    keep = ["long", "short", "net", "beta exposure", "n long", "n short", "ex-ante vol",
            "E[excess] market", "E[excess] alpha", "trade cost", "at liquidity cap"]
    rows = [i for i in res.index if "| built" in i and not i.startswith("bull premium")]
    L += ["", "Part C: first portfolio of 2005 with eight signals (D18 settings; full output in optimizer/)",
          res.loc[rows, keep].to_string(float_format=lambda v: f"{v:.3f}")]
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
