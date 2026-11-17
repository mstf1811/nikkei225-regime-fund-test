"""Figures for the PBFJ paper, from the frozen test-period runs (2005-01-04 .. 2025-09-30).

Same data and plots as backtest_figures.py, re-labelled for the paper. Borsa Istanbul Review asks for
one file per figure, TIFF or JPEG, at least 500 dpi for line art with colour, in Arial or a similar font.
Each figure is saved twice: Figure_n.tif (500 dpi, LZW, the file to upload) and Figure_n.png (400 dpi,
embedded in the Word manuscript by analysis/build_submission.py).

    Figure_1   growth of 1 yen: B0, the comparison runs and the simulated benchmarks
    Figure_2   B0's long, short and beta exposure, with the jump model's bear days shaded
    Figure_3   gross Sharpe ratio of every logged run, with readable run names

Run:  python analysis/paper_figures.py
"""
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS = PROJECT_DIR / "analysis" / "output" / "backtest" / "runs"
TRIALS = PROJECT_DIR / "analysis" / "output" / "backtest" / "trials.csv"
OUT = PROJECT_DIR / "analysis" / "output" / "paper"
DPI_PNG, DPI_TIFF = 400, 500
INK, GREY, BLUE, ORANGE, GREEN, RED, PURPLE = "#1f1e1d", "#8a8985", "#2a6fdb", "#e08a1e", "#3a9a5b", "#c8423b", "#7b5cc4"

plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})

# Readable names for every logged run (trials.csv `name` -> label in Figure 3).
LABELS = {
    "B0": "B0 baseline",
    "C1_regime_off": "C1 without regime model",
    "C2_no_cost_terms": "C2 without cost terms",
    "C3_updater_off": "C3 without updater",
    "C4_fund_rules_relaxed": "C4 fund rules relaxed",
    "C5_hist_mean": "C5 shrunk historical mean",
    "BM_1N": "1/N, long only",
    "BM_quintile": "Quintile sort on B0's expected returns",
    "S_borrow_035": "Borrow fee 0.35%, proceeds earn call rate",
    "S_short_6m": "Shorts held at most 6 months",
    "S_quarterly": "Quarterly rebalancing",
    "S_weekly": "Weekly rebalancing",
    "S_cost_010": "Trading cost 0.10% one-way",
    "S_cost_025": "Trading cost 0.25% one-way",
    "S_pi_bear_capped": "Negative bear premium",
    "S_delay_5": "Regime label acted on 5 days late",
    "S_delay_10": "Regime label acted on 10 days late",
    "S_hmm": "Hidden Markov model instead of jump model",
    "S_grisk_5": "Risk aversion 5 (baseline 10)",
    "S_grisk_20": "Risk aversion 20 (baseline 10)",
    "S_nav_100bn": "Fund size ¥100bn (baseline ¥10bn)",
    "S_delta_1": "Updater with constant coefficients (δ = 1)",
    "S_no_imputed_div": "No imputed dividends",
}


def save(fig, n):
    fig.savefig(OUT / f"Figure_{n}.png", dpi=DPI_PNG)
    fig.savefig(OUT / f"Figure_{n}.tif", dpi=DPI_TIFF, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def daily(name):
    d = pd.read_parquet(RUNS / name / "daily.parquet")
    d.index = pd.to_datetime(d.index)
    return d


def figure_1():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    wide = {"BM_1N": ("1/N, long only", GREEN), "BM_quintile": ("Quintile sort", ORANGE),
            "C5_hist_mean": ("C5 shrunk historical mean", PURPLE), "B0": ("B0 baseline", BLUE)}
    for name, (label, colour) in wide.items():
        a.plot((1 + daily(name).ret).cumprod(), label=label, color=colour, lw=1.4 if name == "B0" else 1)
    a.set_yscale("log")
    a.set_title("(a) B0 against the simulated benchmarks (log scale)", fontsize=9.5, loc="left")
    a.legend(frameon=False, fontsize=8)
    close = {"B0": ("B0 baseline", BLUE), "C1_regime_off": ("C1 without regime model", ORANGE),
             "C2_no_cost_terms": ("C2 without cost terms", RED), "C3_updater_off": ("C3 without updater", GREEN)}
    for name, (label, colour) in close.items():
        b.plot((1 + daily(name).ret).cumprod(), label=label, color=colour, lw=1.4 if name == "B0" else 1)
    b.set_title("(b) B0 against its comparison runs", fontsize=9.5, loc="left")
    b.legend(frameon=False, fontsize=8)
    for ax in (a, b):
        ax.set_ylabel("Growth of ¥1, gross of fund fee")
    fig.tight_layout()
    save(fig, 1)


def figure_2():
    d = daily("B0")
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    bear = d.label.eq(1)
    ax.fill_between(d.index, 0, 1, where=bear, transform=ax.get_xaxis_transform(), color=GREY, alpha=0.18,
                    lw=0, label="Bear label (jump model)")
    ax.plot(d.long, color=BLUE, lw=1, label="Long")
    ax.plot(-d.short, color=RED, lw=1, label="Short")
    ax.plot(d.beta_exp, color=INK, lw=1, label="Beta exposure")
    ax.axhline(0, color=GREY, lw=0.6)
    ax.set_ylabel("Share of net assets")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, 2)


def figure_3():
    t = pd.read_csv(TRIALS).set_index("name")
    missing = set(t.index) - set(LABELS)
    assert not missing, f"no label for {missing}"
    s = t.Sharpe.sort_values()
    colours = [BLUE if n == "B0" else (GREEN if n.startswith("BM") else (ORANGE if n.startswith("C") else GREY))
               for n in s.index]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.barh([LABELS[n] for n in s.index], s.values, color=colours)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_xlabel("Sharpe ratio, gross of fund fee (monthly excess returns, annualized)")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.legend(handles=[Patch(color=BLUE, label="Baseline (B0)"), Patch(color=ORANGE, label="Comparison runs"),
                       Patch(color=GREEN, label="Simulated benchmarks"), Patch(color=GREY, label="Sensitivities")],
              frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, 3)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure_1()
    figure_2()
    figure_3()
    print(f"figures in {OUT}")
