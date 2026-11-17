"""Build JMP_Signal_Test_Report.docx (run after signal_ic.py and signal_ic_figures.py).

Tables are filled from analysis/output/signals/*.csv; the prose quotes the 2026-10-06 run.

Run:  python analysis/signal_report.py
"""
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor, Twips

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signal_ic import OUT, PROJECT_DIR, SIGNALS  # noqa: E402

REPORT = PROJECT_DIR / "JMP_Signal_Test_Report.docx"
GREY = RGBColor(0x52, 0x51, 0x4E)
ANSWER_FILL, HEADER_FILL, BORDER = "EEF4FC", "EFEEEA", "BFBEB8"
NAMES = {"rev": "1-month reversal", "mom": "12-1 momentum", "vol": "Volatility", "beta": "Beta",
         "dy": "Dividend yield", "illiq": "Amihud illiquidity"}


def shade(element, fill):
    pr = element.get_or_add_pPr() if hasattr(element, "get_or_add_pPr") else element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pr.append(shd)


def para(doc, text="", size=10, bold=False, color=None, after=4, before=0, align=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.space_before = Pt(before)
    if align:
        p.alignment = align
    if text:
        add(p, text, size, bold, color)
    return p


def add(p, text, size=10, bold=False, color=None):
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.bold = bold
    if color:
        r.font.color.rgb = color
    return r


def heading(doc, text):
    para(doc, text, size=12, bold=True, before=10, after=4).paragraph_format.keep_with_next = True


def caption(doc, text):
    para(doc, text, size=8.5, color=GREY, after=8)


def figure(doc, path, width_cm):
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].paragraph_format.keep_with_next = True   # keep the caption on the same page


def bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(2)
        if isinstance(item, tuple):
            add(p, item[0], bold=True)
            add(p, item[1])
        else:
            add(p, item)


def table(doc, header, rows, widths, align_right_from=1):
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = OxmlElement(f"w:{edge}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "4")
        b.set(qn("w:color"), BORDER)
        borders.append(b)
    t._tbl.tblPr.append(borders)
    for i, row in enumerate([header] + rows):
        for j, value in enumerate(row):
            cell = t.cell(i, j)
            cell.width = Cm(widths[j])
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            if j >= align_right_from:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            add(p, str(value), size=9, bold=(i == 0))
            if i == 0:
                shade(cell._tc, HEADER_FILL)
    return t


def f(x, d=3, sign=True):
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def build():
    summ = pd.read_csv(OUT / "signal_summary.csv", index_col="signal")
    sub = pd.read_csv(OUT / "subperiods.csv", header=[0, 1], index_col=0)
    rob = pd.read_csv(OUT / "dy_robustness_ic.csv", index_col=0)

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Twips(1020)
    sec.left_margin = sec.right_margin = Twips(1134)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Yu Gothic")

    para(doc, "Which price signals predict Nikkei 225 returns?", size=17, bold=True, after=2)
    para(doc, "JMP · 6 October 2026 · Nikkei 225 members, return months January 2005 – September 2025",
         size=9, color=GREY, after=10)

    p = para(doc, after=10)
    shade(p._p, ANSWER_FILL)
    add(p, "Answer: only dividend yield predicts. ", bold=True)
    add(p, "Members with a higher trailing dividend yield tend to do better the next month: the monthly IC is "
           "+0.032 (t = 3.8), positive in 59% of months. It survives a two-month lag of the dividend, dropping the "
           "imputed dividends and dropping the non-payers. But much of the evidence is recent: 62% of its cumulative "
           "IC comes from 2021–2025, and over 2005–2020 the IC is +0.016 (t = 2.1). The other five signals "
           "(reversal, momentum, volatility, beta, illiquidity) are indistinguishable from zero. Under D12 all six "
           "stay in the model; the data says their weights should be near zero, except dividend yield's.")

    heading(doc, "1. What was tested")
    para(doc, "The question is whether any of six price-based signals, computed at a month-end from data "
              "available that day, ranks the index members' returns over the next month. The six were fixed in "
              "advance from the literature (decision D12), so this test describes them; it does not choose among "
              "them. Choosing by performance over 2005–2025 and then backtesting on the same years would flatter "
              "the backtest.")
    table(doc, ["Signal", "Definition at month-end t", "The literature finds"], [
        ["1-month reversal", "Log total return over month t", "Negative IC: last month's losers rebound"],
        ["12-1 momentum", "Log return over months t−11 to t−1", "Positive, but weak or absent in Japan"],
        ["Volatility", "Std of daily log returns, last 252 days", "Negative: calmer stocks do better"],
        ["Beta", "On the equal-weighted member return, last 252 days", "Negative or zero (betting against beta)"],
        ["Dividend yield", "Dividends with ex-dates in the last 12 months, each over its ex-date price",
         "Positive (a value signal)"],
        ["Amihud illiquidity", "Log of the 12-month mean of: mean |daily return| ÷ average daily yen traded",
         "Positive: an illiquidity premium"],
    ], widths=[3.2, 7.6, 6.2], align_right_from=99)
    caption(doc, "Table 1. The six signals (D12). Volatility and beta need 120 daily returns in the window, momentum "
                 "all 11 months, dividend yield 12 months of returns. Trading value: close × volume from the daily "
                 "file to 2019, the JPX monthly tables from 2020. Dividend yield includes the imputed dividends (D10).")
    para(doc, "Sample: each month-end from December 2004 to August 2025, the members on that day (D5), with their "
              "total return over the next month; a member delisted during the month keeps its return up to its last "
              "day. That is 249 months and 56,011 member-months, and every signal is available for at least 221 "
              "members each month. The main measure is the IC: the rank correlation across members between the "
              "signal and next month's return, computed each month and then averaged. Its t-statistic uses "
              "Newey-West standard errors. In practice a mean IC of 0.02–0.05 makes a useful equity signal, and "
              "below 0.01 is noise.")

    heading(doc, "2. Results")
    rows = []
    for c in SIGNALS:
        r = summ.loc[c]
        rows.append([NAMES[c], f(r.mean_ic), f"{r.ic_t:.1f}", f"{r.share_ic_pos:.0%}",
                     f"{r.q5_minus_q1_pa * 100:+.1f} ({r.q5_minus_q1_t:.1f})",
                     f"{r.fm_slope_pa * 100:+.1f} ({r.fm_slope_t:.1f})", f"{r.persistence:.2f}"])
    table(doc, ["Signal", "Mean IC", "t", "IC > 0", "Q5 − Q1, %/yr (t)", "Per 1σ, %/yr (t)", "Persistence"],
          rows, widths=[3.4, 1.8, 1.2, 1.6, 3.2, 3.2, 2.2])
    caption(doc, "Table 2. Each signal against next month's total return, 249 months. Q5 − Q1: equal-weighted "
                 "top-quintile minus bottom-quintile return. Per 1σ: the monthly cross-sectional regression slope on "
                 "the signal's standardised rank (Fama-MacBeth), i.e. the extra return per standard deviation of the "
                 "signal. Persistence: rank correlation of the signal with itself a month later (near 1 = slow-moving, "
                 "low turnover). t-statistics: Newey-West, 3 lags.")
    figure(doc, OUT / "fig1_mean_ic.png", 15)
    caption(doc, "Figure 1. Mean monthly IC per signal with its 95% interval (Newey-West). Only dividend yield's "
                 "interval excludes zero.")
    para(doc, "Dividend yield is the only signal whose IC is clearly positive. The other five sit within ±0.012 of "
              "zero, with |t| ≤ 1.1. Momentum's absence matches what the literature reports for Japan. Reversal has "
              "the expected sign but is weak among these large, liquid stocks. There is no low-risk effect: by rank, "
              "volatility and beta predict nothing, and by average return the high-volatility and high-beta "
              "quintiles did better (13–14% a year against 8–9%). That comes from a few large winners in those "
              "quintiles, not from a consistent ranking, and it is not significant.")
    para(doc, "The return-based measures are weaker than the IC for every signal, dividend yield included (Q5 − Q1 "
              "+2.4% a year, t = 1.1). Monthly returns are fat-tailed and move together by industry, which adds "
              "noise to averages of returns; ranks are much less affected, so the IC is the more powerful test here. "
              "Dividend yield's quintiles are not a smooth staircase: the top two earned about 12% a year and the "
              "bottom three about 9%.")

    heading(doc, "3. Stability over time")
    periods = ["2005-2011", "2012-2018", "2019-2025"]
    rows = [[NAMES[c]] + [f"{f(sub.loc[c, ('mean_ic', per)])} ({sub.loc[c, ('t', per)]:.1f})" for per in periods]
            for c in SIGNALS]
    table(doc, ["Signal", "2005–2011", "2012–2018", "2019–2025"], rows, widths=[3.6, 3.4, 3.4, 3.4])
    caption(doc, "Table 3. Mean monthly IC (t) by period of the return month. 2019–2025 runs to September 2025.")
    figure(doc, OUT / "fig2_cumulative_ic.png", 13)
    caption(doc, "Figure 2. Cumulative monthly IC, 2005–2025, same scale in every panel. A signal with no "
                 "predictive power wanders around zero; a steady signal climbs in a straight line.")
    para(doc, "Dividend yield's IC is positive in all three periods but uneven: strong in 2005–2011, weak in "
              "2012–2018, and strongest in 2019–2025. Figure 2 shows the steepest climb starts in 2021: 62% of its "
              "cumulative IC over the 21 years comes from the last 57 months (IC +0.087, t = 4.0), against +0.016 "
              "(t = 2.1) for 2005–2020. So the effect holds over the full period, but much of its strength is recent. "
              "A model that re-estimates the signal's weight on a rolling window (D8) learns this only as it "
              "happens, which is the honest way to backtest it.")
    para(doc, "No other signal is consistent. Beta and volatility change sign between periods, and illiquidity "
              "drifts from positive to negative.")

    heading(doc, "4. Overlap, horizon and robustness")
    para(doc, "Overlap. Volatility and beta are largely the same signal (average rank correlation 0.75). Dividend "
              "yield is negatively related to volatility (−0.29): high-yield members are calmer stocks. The other "
              "pairs are close to zero. In one regression with all six, no signal is significant on its own "
              "(dividend yield +0.9% a year per σ, t = 1.3; reversal −1.1%, t = −1.6), so a combined model will "
              "rest mostly on the yield tilt.")
    para(doc, "Horizon. Dividend yield predicts each later month almost as well as the next one: IC +0.032, +0.030, "
              "+0.026, +0.024 and +0.027 for the return 1, 2, 3, 6 and 12 months ahead, all with t ≥ 2.9. With a "
              "persistence of 0.98, it is a slow signal: monthly rebalancing (D7) is more than enough for it, and it "
              "gives the daily updater (E3) little to track at the stock level. Reversal's IC with the return 12 "
              "months ahead (−0.023, t = −2.8) is one of 30 horizon results and is treated as chance; D12 rules out "
              "adding signals found this way.")
    labels = {"dy (baseline)": "Baseline",
              "dy, dividends lagged 2 months": "Dividends lagged two months",
              "dy, windows without imputed dividends": "Windows without an imputed dividend (D10)",
              "dy, payers only (dy > 0)": "Payers only (yield > 0)"}
    from signal_ic import nw_t  # noqa: E402
    rows = []
    for col, label in labels.items():
        m, _, t = nw_t(rob[col], 3)
        rows.append([label, f(m), f"{t:.1f}"])
    table(doc, ["Dividend yield variant", "Mean IC", "t"], rows, widths=[8.0, 2.2, 1.6])
    caption(doc, "Table 4. Dividend-yield checks, next-month IC, 249 months. The lag tests timing: Japanese year-end "
                 "dividends are confirmed only around May, so the baseline uses an amount the market only roughly "
                 "knew at the ex-date. Non-payers are 3.7% of member-months.")
    para(doc, "The lag costs about a fifth of the IC, so part of the baseline benefits from knowing the exact "
              "amount early; the model should use the lagged version. The imputed dividends do not create the "
              "effect, and the few non-payers do not drive it.")

    heading(doc, "5. What this means for return estimation (U7)")
    bullets(doc, [
        ("Weights. ", "Under D12 all six signals stay in. The evidence says how much each should count: dividend "
                      "yield clearly, the other five at or near zero. A model that estimates each weight from past "
                      "data and shrinks weak weights toward zero does this by itself."),
        ("Size. ", "An IC of 0.032 with a cross-sectional spread of 6.9% a month is worth about 2.6% a year per "
                   "standard deviation of yield by rank, or 1.2% by the regression slope. Between a +1σ and a −1σ "
                   "stock that is a 2.5–5% a year gap: small against the 16% a year standard error of a single "
                   "stock's 5-year mean, which is why pooling across stocks is needed. This supports the proposal in "
                   "U7: μᵢ = r_f + βᵢ·π + αᵢ, with αᵢ from the signals and their weights re-estimated on the rolling "
                   "5-year window (D8)."),
        ("Timing. ", "Use the dividend yield lagged two months (Table 4)."),
        ("Open points. ", "(1) Much of the yield effect is in 2021–2025. (2) It may be partly an industry tilt; we "
                          "have not checked, and industry codes are available only from 2020 (JPX monthly tables). "
                          "(3) Volatility and beta overlap (0.75), so the model should not count them as two "
                          "independent pieces of evidence."),
    ])
    para(doc, "Files. Code: analysis/signal_ic.py, signal_ic_figures.py and signal_report.py. Results: "
              "analysis/output/signals/ (summary.txt and one CSV per table).", size=8.5, color=GREY, before=6)
    doc.save(REPORT)
    print(f"Written {REPORT}")


if __name__ == "__main__":
    build()
