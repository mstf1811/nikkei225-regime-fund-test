"""Figures for the signal-test report (run after analysis/signal_ic.py).

fig1_mean_ic.png        mean monthly IC per signal with its 95% interval (Newey-West, 3 lags)
fig2_cumulative_ic.png  cumulative monthly IC over 2005-2025, one panel per signal, shared scale

Run:  python analysis/signal_ic_figures.py
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signal_ic import OUT, SIGNALS, nw_t  # noqa: E402

SERIES = "#2a78d6"                        # categorical slot 1 (validated, light surface)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#ffffff"
LABELS = {"rev": "1-month reversal", "mom": "12-1 momentum", "vol": "Volatility (1 year)",
          "beta": "Beta (1 year)", "dy": "Dividend yield (12 months)", "illiq": "Amihud illiquidity"}

plt.rcParams.update({"font.family": "Calibri", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})


def fig1(ic):
    stats = [nw_t(ic[c], 3) for c in SIGNALS]
    fig, ax = plt.subplots(figsize=(6.3, 2.6), dpi=200, facecolor=SURFACE)
    y = np.arange(len(SIGNALS))[::-1]
    for yi, (mean, se, t), c in zip(y, stats, SIGNALS):
        ax.plot([mean - 1.96 * se, mean + 1.96 * se], [yi, yi], color=SERIES, lw=1.5, solid_capstyle="round")
        ax.plot(mean, yi, "o", ms=6, color=SERIES, mec=SURFACE, mew=1.5, zorder=3)
        ax.text(mean + 1.96 * se + 0.002, yi, f"{mean:+.3f}  (t = {t:.1f})", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [LABELS[c] for c in SIGNALS], color=INK)
    ax.set_xlim(-0.045, 0.075)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Mean monthly IC: rank correlation with next month's return (dot), with 95% interval")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_mean_ic.png", facecolor=SURFACE)
    plt.close(fig)


def fig2(ic):
    cum = ic.cumsum()
    t = cum.index.to_timestamp()
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 3.3), dpi=200, facecolor=SURFACE, sharex=True, sharey=True)
    for ax, c in zip(axes.flat, SIGNALS):
        ax.axhline(0, color=INK2, lw=0.8)
        ax.plot(t, cum[c], color=SERIES, lw=1.4)
        ax.set_title(LABELS[c], fontsize=9, color=INK, loc="left")
        ax.text(t[-1], cum[c].iloc[-1], f" {cum[c].iloc[-1]:+.1f}", va="center", fontsize=8, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("Cumulative IC")
    for ax in axes[1]:
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(10))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT / "fig2_cumulative_ic.png", facecolor=SURFACE)
    plt.close(fig)


def main():
    ic = pd.read_csv(OUT / "ic_monthly.csv", index_col="month")
    ic.index = pd.PeriodIndex(ic.index, freq="M")
    fig1(ic)
    fig2(ic)


if __name__ == "__main__":
    main()
