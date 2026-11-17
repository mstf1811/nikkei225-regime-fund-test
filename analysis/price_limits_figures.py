"""Figures for the price-limit report (run after analysis/price_limits_tobit.py).

fig1_censored_by_year.png   limit closes per 10,000 member days, by year and direction, with the
                            PFE's BVMT rate for reference
fig2_return_tail.png        share of member returns beyond each size, with Tunisia's limit and the
                            range of Japan's limits marked

Run:  python analysis/price_limits_figures.py
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import DATA_DIR, PROJECT_DIR  # noqa: E402

OUT = PROJECT_DIR / "analysis" / "output" / "price_limits"
UP, DOWN = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated, light surface)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SURFACE = "#ffffff"
PFE_RATE = 0.008
PFE_L = np.log(1.06)

plt.rcParams.update({"font.family": "Calibri", "font.size": 9, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False})


def fig1():
    y = pd.read_csv(OUT / "censored_by_year.csv", index_col="year")
    up = y.up / y.member_days * 1e4
    down = y.down / y.member_days * 1e4
    fig, ax = plt.subplots(figsize=(6.3, 2.5), dpi=200, facecolor=SURFACE)
    x = np.arange(len(y))
    ax.bar(x, down, width=0.62, color=DOWN, edgecolor=SURFACE, linewidth=1, label="closed limit-down")
    ax.bar(x, up, width=0.62, bottom=down, color=UP, edgecolor=SURFACE, linewidth=1, label="closed limit-up")
    ax.axhline(PFE_RATE * 1e4, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
    ax.text(len(y) - 0.6, PFE_RATE * 1e4 + 2, "PFE (Tunisia, ±6% limit): ≈80", ha="right", va="bottom",
            color=INK2, fontsize=8)
    ax.axvline(x[list(y.index).index(2010)] - 0.5, color=INK2, linewidth=0.8, linestyle=":")
    ax.text(x[list(y.index).index(2010)] - 0.4, 58, "wider limits\nfrom 2010-01-04", color=INK2, fontsize=7.5,
            va="top")
    total = up + down
    for yr in (2008, 2011, 2024):
        i = list(y.index).index(yr)
        ax.text(x[i], total[yr] + 1.5, f"{total[yr]:.0f}", ha="center", va="bottom", color=INK, fontsize=8)
    ax.set_xticks(x, [str(v) if v in (2005, 2008, 2011, 2014, 2017, 2020, 2024) else "" for v in y.index],
                  fontsize=8)
    ax.set_ylabel("per 10,000 member days")
    ax.set_ylim(0, 90)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=2, bbox_to_anchor=(0.0, 1.02))
    fig.tight_layout()
    fig.savefig(OUT / "fig1_censored_by_year.png", facecolor=SURFACE)
    plt.close(fig)


def fig2():
    px = pd.read_parquet(DATA_DIR / "prices_daily.parquet", columns=["date", "in_index", "log_return"])
    pl = pd.read_parquet(DATA_DIR / "price_limits_daily.parquet", columns=["width_up", "width_down", "in_index", "date"])
    px["date"] = pd.to_datetime(px["date"])
    r = px.loc[px.in_index & (px.date >= "2005-01-01"), "log_return"].dropna().abs().to_numpy()
    pl["date"] = pd.to_datetime(pl["date"])
    w = pl[pl.in_index & (pl.date >= "2005-01-01") & (pl.date >= "2010-01-04")]
    lo, hi = w.width_up.quantile(0.01), w.width_down.quantile(0.99)
    med = (w.width_up.median() + w.width_down.median()) / 2
    grid = np.linspace(0, 0.30, 301)
    surv = np.array([(r >= g).mean() for g in grid])
    fig, ax = plt.subplots(figsize=(6.3, 2.4), dpi=200, facecolor=SURFACE)
    ax.axvspan(lo * 100, hi * 100, color=UP, alpha=0.10, linewidth=0)
    ax.text(lo * 100 + 0.4, 0.55, "range of Japan's limits since 2010 (1st–99th percentile)", ha="left",
            va="top", color=INK2, fontsize=7.5, zorder=3,
            bbox=dict(facecolor="#e9f1fb", edgecolor="none", pad=1.5))
    ax.plot(grid * 100, surv, color=INK, linewidth=1.6)
    for xv, label in [(PFE_L, f"Tunisia ±6%: {(r >= PFE_L).mean():.2%} of days beyond"),
                      (med, f"Japan median ≈{med:.0%}: {(r >= med).mean():.3%} beyond")]:
        ax.axvline(xv * 100, color=INK2, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.annotate(label, (xv * 100, (r >= xv).mean()), xytext=(6, 6), textcoords="offset points",
                    color=INK, fontsize=8)
        ax.plot([xv * 100], [(r >= xv).mean()], "o", color=INK, markersize=4)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1)
    ax.set_xlim(0, 30)
    ax.set_xlabel("size of the daily move, |log return| (%)")
    ax.set_ylabel("share of member days\nwith a larger move")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, which="major")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_return_tail.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print("figures written to", OUT)
