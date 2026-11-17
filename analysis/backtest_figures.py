"""Figures for JMP_Backtest_Report.docx, from the frozen test-period runs (2005-01-04 .. 2025-09-30).

    fig1_cumulative.png   growth of 1 yen: B0, the four comparison runs and the two simulated benchmarks
    fig2_exposure.png     B0's long, short and beta exposure, with the jump model's bear days shaded
    fig3_sensitivity.png  gross Sharpe ratio of every logged run (D20 (8)), B0 marked

Run:  python analysis/backtest_figures.py
"""
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS = PROJECT_DIR / "analysis" / "output" / "backtest" / "runs"
OUT = PROJECT_DIR / "analysis" / "output" / "backtest" / "figures"
INK, GREY, BLUE, ORANGE, GREEN, RED, PURPLE = "#1f1e1d", "#8a8985", "#2a6fdb", "#e08a1e", "#3a9a5b", "#c8423b", "#7b5cc4"

plt.rcParams.update({"font.family": "Calibri", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})


def daily(name):
    d = pd.read_parquet(RUNS / name / "daily.parquet")
    d.index = pd.to_datetime(d.index)
    return d


def cumulative():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    wide = {"BM_1N": ("1/N (long only)", GREEN), "BM_quintile": ("Quintile sort", ORANGE),
            "C5_hist_mean": ("C5 historical mean", PURPLE), "B0": ("B0 baseline", BLUE)}
    for name, (label, colour) in wide.items():
        a.plot((1 + daily(name).ret).cumprod(), label=label, color=colour, lw=1.4 if name == "B0" else 1)
    a.set_yscale("log")
    a.set_title("B0 against the simulated benchmarks (log scale)", fontsize=9.5, loc="left")
    a.legend(frameon=False, fontsize=8)
    close = {"B0": ("B0 baseline", BLUE), "C1_regime_off": ("C1 regime off", ORANGE),
             "C2_no_cost_terms": ("C2 no cost terms", RED), "C3_updater_off": ("C3 updater off", GREEN)}
    for name, (label, colour) in close.items():
        b.plot((1 + daily(name).ret).cumprod(), label=label, color=colour, lw=1.4 if name == "B0" else 1)
    b.set_title("B0 against its comparison runs", fontsize=9.5, loc="left")
    b.legend(frameon=False, fontsize=8)
    for ax in (a, b):
        ax.set_ylabel("growth of ¥1, gross of fund fee")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_cumulative.png", dpi=200)


def exposure():
    d = daily("B0")
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    bear = d.label.eq(1)
    ax.fill_between(d.index, 0, 1, where=bear, transform=ax.get_xaxis_transform(), color=GREY, alpha=0.18,
                    lw=0, label="bear label (jump model)")
    ax.plot(d.long, color=BLUE, lw=1, label="long")
    ax.plot(-d.short, color=RED, lw=1, label="short")
    ax.plot(d.beta_exp, color=INK, lw=1, label="beta exposure")
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_ylabel("share of NAV")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "fig2_exposure.png", dpi=200)


def sensitivity():
    t = pd.read_csv(PROJECT_DIR / "analysis" / "output" / "backtest" / "trials.csv").set_index("name")
    s = t.Sharpe.sort_values()
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    colours = [BLUE if n == "B0" else (GREEN if n.startswith("BM") else (ORANGE if n.startswith("C") else GREY))
               for n in s.index]
    ax.barh(s.index, s.values, color=colours)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel("Sharpe ratio, gross of fund fee (monthly excess returns, annualised)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_sensitivity.png", dpi=200)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cumulative()
    exposure()
    sensitivity()
    print(f"figures in {OUT}")
