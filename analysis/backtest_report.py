"""Build JMP_Backtest_Report.docx: the write-up of the frozen backtest (run after bt_report.py and
backtest_figures.py).

Tables are filled from analysis/output/backtest/runs/report/*.csv and trials.csv; the prose quotes the
2026-10-31 test-period batch (specification frozen 2026-10-31, D1-D22) and the 2026-11-01 diagnosis of
the no-imputed-dividend sensitivity (JMP_TRACKER.md, "Build the backtest").

Run:  python analysis/backtest_report.py
"""
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, Twips

sys.path.insert(0, str(Path(__file__).resolve().parent))
from signal_report import (ANSWER_FILL, GREY, add, bullets, caption, figure, heading,  # noqa: E402
                           para, shade, table)

PROJECT_DIR = Path(__file__).resolve().parent.parent
BT = PROJECT_DIR / "analysis" / "output" / "backtest"
REP, FIG = BT / "runs" / "report", BT / "figures"
REPORT = PROJECT_DIR / "JMP_Backtest_Report.docx"

LABELS = {"B0": "B0 baseline", "C1_regime_off": "C1 regime off (E1)", "C2_no_cost_terms": "C2 no cost terms",
          "C3_updater_off": "C3 updater off (E3)", "C4_fund_rules_relaxed": "C4 fund rules relaxed",
          "C5_hist_mean": "C5 shrunk historical mean (E5)", "BM_1N": "1/N, long only (simulated)",
          "BM_quintile": "Quintile sort on B0's μ (simulated)", "cash": "Cash (call rate)",
          "EW member market": "EW member market", "^N225 (price)": "^N225 (price index)"}


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def build():
    m = pd.read_csv(REP / "metrics.csv", index_col=0)
    h2 = pd.read_csv(REP / "h2.csv", index_col=0)
    h3 = pd.read_csv(REP / "h3.csv", index_col=0)

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Twips(1020)
    sec.left_margin = sec.right_margin = Twips(1134)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "Yu Gothic")

    para(doc, "Does a regime-aware long-short fund beat cash in Japan?", size=17, bold=True, after=2)
    para(doc, "A pre-registered backtest on free data, Nikkei 225 members, January 2005 – September 2025",
         size=11, after=2)
    para(doc, "JMP · Mustapha Aziz Laroussi · 31 October 2026 · specification frozen 2026-10-31 (D1–D22), "
              "23 logged runs", size=9, color=GREY, after=10)

    b0 = m.loc["B0"]
    p = para(doc, after=10)
    shade(p._p, ANSWER_FILL)
    add(p, "Answer: no. ", bold=True)
    add(p, f"The baseline fund (B0) earned {pct(b0.CAGR)}/yr at {pct(b0.vol)} volatility, a Sharpe ratio of "
           f"{b0.Sharpe:.2f} gross of fees (Newey-West t 0.67, p 0.25). After a typical Japanese long-short trust "
           f"fee its return is {pct(b0['CAGR net'])}/yr. Its factor alpha is −0.3%/yr (t −1.3), and its deflated Sharpe "
           "ratio is 0.12 once all 23 logged runs are counted. Of the four components tested in pairs, only the "
           "trading-cost terms add significant value (Holm p 0.014); the regime model, the Kalman updater and the "
           "cross-sectional return model do not. A long-only equal-weighted portfolio of the same stocks had a "
           "Sharpe ratio of 0.54. The fund sat mostly in cash: its fixed risk aversion held market exposure near "
           "7% of NAV, and the jump model, which labelled 59% of days bear, separated volatility but not returns.")

    # ------------------------------------------------------------------ 1
    heading(doc, "1. The question and why it is asked this way")
    para(doc, "The project ports a regime-aware portfolio pipeline, first built for the Tunisian market (the "
              "author's end-of-studies project), to Japan, as an investment fund would run it. Five components "
              "were fixed from the start: regime detection, return estimation, an updating component, "
              "optimisation, and the rules of a Japanese public investment trust. Every design choice was taken "
              "from recent published work and replicated rather than invented, and the whole specification was "
              "frozen before the first run on the test period. The question is narrow: does the fund, with "
              "realistic costs and Japanese fund rules, beat cash, and does each component earn its place?")
    para(doc, "This design trades the chance of a striking result for one that can be believed. Backtests are "
              "usually tuned on the period they report; here nothing was tuned on 2005–2025, all 23 runs are "
              "reported, and the statistics account for how many were tried (Arnott, Harvey & Markowitz 2019).")

    # ------------------------------------------------------------------ 2
    heading(doc, "2. Data, from free sources only")
    bullets(doc, [
        ("Universe: ", "the Nikkei 225 as it stood on each date, 394 codes since 1996, including companies "
                       "later removed (no survivorship bias)."),
        ("Prices: ", "JPX daily reports for every stock, parsed from the raw files: 1,891,508 rows, 2000-01-04 → "
                     "2025-09-30, 222–225 members every day, no OHLC inconsistency. 99.7% of closes match Yahoo "
                     "within 1%; the mismatches examined are Yahoo's errors."),
        ("Corporate actions: ", "337 splits, 314 confirmed by the step in the JPX/Yahoo price ratio and 23 "
                                "detected from JPX prices alone; 11,393 Yahoo dividends converted back to yen paid; "
                                "one spin-off (Sony → Sony Financial Group, 2025). Total returns are computed by "
                                "the project, not taken from a vendor."),
        ("Missing dividends (D10): ", "Yahoo has no dividends for codes that no longer exist (11% of member "
                                      "days). They are imputed on the March and September ex-dates from past data "
                                      "only: half the members' median yield, half the stock's own past ex-date "
                                      "price drops."),
        ("Short histories: ", "32 successors of mergers and holding-company reorganisations are chained to "
                              "their predecessors' returns, cutting member month-ends with less than five years "
                              "of history from 3.10% to 0.88%."),
        ("Liquidity and shortability: ", "daily volume from the JPX reports to 2019 and the JPX monthly detailed "
                                         "tables from 2020; every member was loan-eligible (貸借) in 2020–2025."),
        ("Rates and factors: ", "the BOJ call rate; Kenneth French's Japan factors (SMB, HML, RMW, CMA, WML)."),
    ])
    para(doc, "Known gaps, accepted before the freeze (D22): splits of 1.1–1.25 cannot be detected on days "
              "Yahoo does not cover (about 1–2 expected); and three of 1,783 imputed dividends, each just before "
              "a 10:1 consolidation, are implausibly large (8–11% of the price). Neither was corrected after "
              "the freeze.", after=6)

    # ------------------------------------------------------------------ 3
    heading(doc, "3. The fund")
    para(doc, "Each component reproduces a published method; settings the sources do not state are flagged "
              "in the project tracker as the project's own.")
    table(doc, ["Component", "Method", "Main source"], [
        ["E1 Regime detection", "2-state statistical jump model on ^N225 excess returns; downside deviation and "
         "two Sortino features; 3,000-day window refitted every 6 months; penalty λ by 8-year cross-validation; "
         "causal online labels", "Shu, Yu & Mulvey (2024), tested on the Nikkei 225"],
        ["E5 Return estimation", "μ_i = r_f + β_i·π(regime) + α_i. β by Vasicek shrinkage; α from eight "
         "signals (reversal, momentum, volatility, beta, dividend yield, illiquidity, residual momentum, "
         "residual reversal), weights by Fama-MacBeth shrunk toward 0; π_bear = 0",
         "Fama & MacBeth (1973); Blitz et al. (2011, 2013); D12, D14–D15, D21"],
        ["E5 Covariance", "Ledoit-Wolf shrinkage toward the single-index target, 5-year daily window, "
         "monthly", "Ledoit & Wolf (2003)"],
        ["E3 Updating", "Kalman filters (dynamic linear models) on π_bull and the signal weights, averaged "
         "over four discount factors", "West & Harrison (1997); Dangl & Halling (2012)"],
        ["E2 Optimisation", "Single-period convex optimisation with linear + 3/2-power impact costs, "
         "borrow cost and a risk term; γ_risk 10, γ_trade 1, γ_hold 1, fixed", "Boyd et al. (2017); "
         "Shu, Yu & Mulvey (2025)"],
        ["Fund rules", "Long ≤ 100% and short ≤ 100% of NAV; long ≤ 10% per name (JITA); short ≥ −5%; "
         "every position sellable in 3 days at 10% of volume; trades ≤ 10% of a day's volume; no trade "
         "against a limit close", "JITA rules; D13, D18"],
    ], [3.2, 9.6, 4.6], align_right_from=9)
    para(doc, "", after=4)
    para(doc, "Trading: monthly decisions at the first close of the month, plus decisions triggered by a "
              "regime change or a risk-limit breach, each executed over up to five days. Costs: 0.15% one-way "
              "plus square-root market impact on a ¥10bn fund; shorts pay 1.15%/yr plus the call rate and every "
              "dividend, and their proceeds earn nothing. Weights drift with prices between decisions. The NAV "
              "is simulated daily and reconciles to interest, fees, dividends, price P&L and costs to 5e-16.")

    # ------------------------------------------------------------------ 4
    heading(doc, "4. Protocol")
    bullets(doc, [
        ("Frozen: ", "2026-10-31 16:41, before any run on 2005 or later. The code hash is recorded, and all 23 "
                     "runs carry the frozen hash. Estimation histories start earlier and stay causal."),
        ("Statistics: ", "249 monthly returns, 2005-01 → 2025-09 (20.75 years)."),
        ("H1, B0 beats cash: ", "one-sided Newey-West t on the mean monthly excess return; Lo's (2002) standard "
                                "error; the probabilistic Sharpe ratio."),
        ("H2, each component adds value: ", "Ledoit-Wolf (2008) bootstrap test of the Sharpe difference between "
                                            "B0 and each of C1 (no regime), C2 (no cost terms in the objective), "
                                            "C3 (no updater) and C5 (the original pipeline's shrunk historical "
                                            "mean), Holm-adjusted across the four."),
        ("H3, positive alpha: ", "Newey-West t of the intercept on the yen market and the five French Japan "
                                 "factors."),
        ("Overfitting: ", "deflated Sharpe ratio (Bailey & López de Prado 2014) with N = 2 signal lists and "
                          "N = 23 runs; the t ≥ 3 hurdle of Harvey, Liu & Zhu (2016)."),
        ("Sensitivities: ", "15 pre-declared runs (costs, borrow fee, rebalancing frequency, delays, risk "
                            "aversion, fund size, the HMM instead of the jump model, a negative bear premium, a "
                            "6-month short limit, no imputed dividends). They are reported, never selected."),
    ])

    # ------------------------------------------------------------------ 5
    heading(doc, "5. Results")
    rows = []
    for k in ["B0", "C1_regime_off", "C2_no_cost_terms", "C3_updater_off", "C4_fund_rules_relaxed",
              "C5_hist_mean", "BM_1N", "BM_quintile", "EW member market", "^N225 (price)", "cash"]:
        r = m.loc[k]
        sharpe = "—" if pd.isna(r.Sharpe) else f"{r.Sharpe:.2f}"
        net = "—" if pd.isna(r["Sharpe net"]) else f"{r['Sharpe net']:.2f}"
        gross = "—" if pd.isna(r["avg gross"]) else pct(r["avg gross"], 0)
        rows.append([LABELS[k], pct(r.CAGR), pct(r.vol), sharpe, net, pct(r.MDD, 0), gross])
    table(doc, ["Run", "CAGR", "Volatility", "Sharpe", "Sharpe net", "Max drawdown", "Avg gross"], rows,
          [5.6, 1.8, 1.9, 1.6, 1.9, 2.4, 2.2])
    caption(doc, "Table 1. Annualised, gross of the fund fee unless 'net' (2.24%/yr trust fee). Sharpe from "
                 "monthly excess returns over the call rate. Avg gross = average long + short as a share of NAV.")

    figure(doc, FIG / "fig1_cumulative.png", 17.4)
    caption(doc, "Figure 1. Growth of ¥1, gross of the fund fee. Left: B0 against the benchmarks on a log "
                 "scale. Right: B0 against its comparison runs.")

    para(doc, "H1, B0 against cash: not supported. ", bold=True, after=2).paragraph_format.keep_with_next = True
    para(doc, "The mean excess return is 0.17%/yr gross (NW t 0.67, p 0.25; Sharpe 0.16 with Lo standard error "
              "0.24; PSR 0.77). Net of the trust fee it is −2.06%/yr (t −8.1). Over 20.75 years the smallest "
              "Sharpe ratio this test could detect was about 0.55, so a true Sharpe near 0.16 could not have "
              "been confirmed, but nothing here suggests it is larger.")

    para(doc, "H2, the components: only the cost terms add significant value.", bold=True, after=2)
    table(doc, ["B0 minus", "Sharpe difference", "s.e.", "Correlation", "p (bootstrap)", "p (Holm)"],
          [[LABELS[k], f"{r['SR diff /yr']:+.2f}", f"{r['se /yr']:.2f}", f"{r['corr']:.2f}", f"{r['p boot']:.3f}",
            f"{r['p Holm']:.3f}"] for k, r in h2.iterrows()], [5.6, 2.6, 1.6, 2.2, 2.4, 2.0])
    caption(doc, "Table 2. Ledoit-Wolf (2008) studentised circular block bootstrap, two-sided, K = 5000, "
                 "M = 4999, block size calibrated; Holm across the four comparisons.")
    para(doc, "Without the cost terms (C2) the optimiser trades 4.9× NAV a year and loses 1.2%/yr; the cost "
              "model is what keeps the fund from destroying value. Every other difference is positive but well "
              "within noise. C4, which relaxes the Japanese fund rules, is identical to B0: the rules never bind "
              "for a fund this small in exposure.")

    para(doc, "H3, factor alpha: none.", bold=True, after=2).paragraph_format.keep_with_next = True
    names = {"alpha": "Alpha (per year)", "mkt_yen": "Yen market", "smb5": "SMB", "hml5": "HML", "rmw": "RMW",
             "cma": "CMA", "wml": "WML"}
    table(doc, ["", "Coefficient", "NW s.e.", "t"],
          [[names[k], f"{r.coef:+.4f}", f"{r['NW se']:.4f}", f"{r.t:+.2f}"] for k, r in h3.iterrows()],
          [4.0, 2.6, 2.6, 2.0])
    caption(doc, "Table 3. B0's monthly excess return on the yen market (EW members − call rate) and French's "
                 "Japan factors, 249 months. The intercept is annualised (monthly × 12); the "
                 "factor loadings are unitless.")

    para(doc, "Overfitting checks.", bold=True, after=2)
    para(doc, "The deflated Sharpe ratio is 0.59 with N = 2 (the two signal lists) and 0.12 with N = 23 (every "
              "logged run); B0's t of 0.67 is far from the t ≥ 3 hurdle. With so little to deflate, these "
              "mostly confirm that the result is not an artefact of trying many variants.")

    # ------------------------------------------------------------------ 6
    heading(doc, "6. Why the fund stayed near cash")
    figure(doc, FIG / "fig2_exposure.png", 17.4)
    caption(doc, "Figure 2. B0's long and short exposure and its beta exposure, with the jump model's bear "
                 "labels shaded.")
    bullets(doc, [
        ("The regime labels carried volatility information, not return information. ", "The jump model labelled "
         "59% of test days bear and switched 5.6 times a year. On the days the fund acted on a bear label, ^N225 "
         "returned 6.1%/yr at 25% volatility; on bull days, 7.3%/yr at 19%. Setting the bear premium to zero "
         "(D17) therefore cut market exposure on days whose returns were nearly as high. It cost little only "
         "because the exposure was small anyway (next point): the no-regime run C1 did no better (Sharpe −0.11)."),
        ("Risk aversion kept even the bull exposure small. ", "With γ_risk = 10 in monthly units, the optimal "
         "market weight is roughly π / (2·γ_risk·σ²). With π_bull averaging 6.6%/yr from the updater and "
         "market variance near 0.4% a month, that is about 7% of NAV, which is what B0 held (average beta "
         "exposure 0.066 in bull, 0.016 in bear). Halving γ_risk roughly doubled the exposure but left the Sharpe ratio "
         "at 0.16: the level of exposure is not why the Sharpe ratio is low."),
        ("The cross-sectional model found little. ", "The signal weights started at zero in 2005 and had any "
         "dispersion in only 47 of 249 months. The monthly information coefficient of α was 0.0035 (t 0.16). "
         "This matches the pre-test signal study, where only dividend yield predicted returns, mostly in "
         "2021–2025."),
        ("Drawdowns were avoided. ", "From September 2008 to March 2009 B0 lost 0.6% while the equal-weighted "
         "member market lost 36%, and in February–March 2020 it lost 0.8% against 21%. But B0 held so little of "
         "the market that this says more about its size than about its timing: C1, without the regime model, "
         "lost 1.5% and 1.8% in the same episodes."),
    ])

    # ------------------------------------------------------------------ 7
    heading(doc, "7. Sensitivities")
    figure(doc, FIG / "fig3_sensitivity.png", 16.5)
    caption(doc, "Figure 3. Gross Sharpe ratio of all 23 logged runs (B0 blue, comparison runs orange, "
                 "simulated benchmarks green, sensitivities grey). None is selected.")
    para(doc, "Every sensitivity of the fund lies between −0.07 and 0.53, and none survives the fund fee. Two "
              "need comment.")
    bullets(doc, [
        ("No imputed dividends (Sharpe 0.53) is a look-ahead artefact. ", "Yahoo lacks dividends exactly for "
         "codes that later disappeared, so whether a stock's dividends are missing depends on its future. With "
         "the imputation removed, those stocks show a dividend yield of zero and understated past returns; the "
         "dividend-yield weight rises about fourfold, gross exposure rises from 16% to 52% of NAV, and 31% of "
         "the run's short weight (against 4% in B0) lands on the 8% of stock-months whose dividends are missing. "
         "The best contributors are stocks later absorbed in mergers (Sumitomo Metal, CSK, Sumitomo Trust, "
         "Mitsubishi Rayon, Nippon Paper Group). The run is in effect told which stocks will disappear. Investors "
         "at the time saw these dividends, so the result supports the past-only imputation of B0."),
        ("A negative bear premium (−0.07) made things worse. ", "Going net short on bear labels, as the 2025 "
         "source caps it, lost money: the bear labels did not forecast falling markets."),
    ])

    # ------------------------------------------------------------------ 8
    heading(doc, "8. Limitations")
    bullets(doc, [
        "Free data: total returns are constructed, not vendor-supplied. Dividends for 11% of member days are "
        "imputed, a few small splits may be missing, and no free value-weighted total-return index exists, so "
        "the market proxy is equal-weighted.",
        "The jump-model evidence comes mostly from one research group. The project's own pre-2005 check agreed "
        "with it; in the test period B0 beat the no-regime run by 0.27 Sharpe, which is not significant.",
        "π_bear = 0 is the project's long-short translation of the sources' long-only exit; it does not test "
        "their exact strategy.",
        "The optimiser's knobs were fixed from a source on multi-asset portfolios, not tuned for this fund. That "
        "keeps the test honest but may not suit a single-country stock fund.",
        "Power is limited: a true Sharpe ratio below about 0.5 cannot be confirmed in 21 years.",
        "The trust fee (2.24%/yr) comes from one Japanese long-short fund's filing; the gross results do not "
        "depend on it.",
    ])

    # ------------------------------------------------------------------ 9
    heading(doc, "9. Conclusion")
    para(doc, "Built from published methods, run under Japanese fund rules with realistic costs, and tested once "
              "on a frozen specification, the regime-aware long-short fund does not beat cash over 2005–2025, and "
              "none of its forecasting components adds measurable value. The cost model is the one component that "
              "matters. The regime model identified high-volatility periods but not low-return ones, and the fixed "
              "risk aversion kept market exposure near 7% of NAV even in bull periods, so the fund stayed close "
              "to cash throughout. A long-only "
              "equal-weighted portfolio of the same stocks did better on every risk-adjusted measure except "
              "drawdown. The contribution is the test itself: a free, survivorship-free Nikkei 225 dataset, a "
              "pre-registered protocol, and an honest negative result.")
    para(doc, "Any change made after seeing these results, such as a different bear premium or risk aversion, "
              "would be a new trial and must be counted as one.", color=GREY)

    # ------------------------------------------------------------------ appendix
    heading(doc, "References")
    refs = [
        "Arnott, R., Harvey, C. R. & Markowitz, H. (2019). A Backtesting Protocol in the Era of Machine Learning. "
        "Journal of Financial Data Science 1(1), 64–74.",
        "Bailey, D. H. & López de Prado, M. (2012). The Sharpe ratio efficient frontier. Journal of Risk 15(2), 3–44.",
        "Bailey, D. H. & López de Prado, M. (2014). The Deflated Sharpe Ratio. Journal of Portfolio Management "
        "40(5), 94–107.",
        "Blitz, D., Huij, J. & Martens, M. (2011). Residual momentum. Journal of Empirical Finance 18(3), 506–521.",
        "Blitz, D., Huij, J., Lansdorp, S. & Verbeek, M. (2013). Short-term residual reversal. Journal of "
        "Financial Markets 16(3), 477–504.",
        "Boyd, S., Busseti, E., Diamond, S., Kahn, R. N., Koh, K., Nystrup, P. & Speth, J. (2017). Multi-Period "
        "Trading via Convex Optimization. Foundations and Trends in Optimization 3(1), 1–76.",
        "Campbell, J. Y. & Thompson, S. B. (2008). Predicting Excess Stock Returns Out of Sample. Review of "
        "Financial Studies 21(4), 1509–1531.",
        "Dangl, T. & Halling, M. (2012). Predictive regressions with time-varying coefficients. Journal of "
        "Financial Economics 106(1), 157–181.",
        "DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal Versus Naive Diversification. Review of "
        "Financial Studies 22(5), 1915–1953.",
        "Fama, E. F. & MacBeth, J. D. (1973). Risk, Return, and Equilibrium: Empirical Tests. Journal of "
        "Political Economy 81(3), 607–636.",
        "Fama, E. F. & French, K. R. (2017). International tests of a five-factor asset pricing model. Journal of "
        "Financial Economics 123(3), 441–463.",
        "Harvey, C. R., Liu, Y. & Zhu, H. (2016). … and the Cross-Section of Expected Returns. Review of "
        "Financial Studies 29(1), 5–68.",
        "Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of "
        "Statistics 6(2), 65–70.",
        "Iwanaga, Y. (2024). Revisiting the residual momentum in Japan. International Review of Financial "
        "Analysis 93, 103190.",
        "Koop, G. & Korobilis, D. (2012). Forecasting Inflation Using Dynamic Model Averaging. International "
        "Economic Review 53(3), 867–886.",
        "Ledoit, O. & Wolf, M. (2003). Improved estimation of the covariance matrix of stock returns. Journal of "
        "Empirical Finance 10(5), 603–621.",
        "Ledoit, O. & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. Journal of "
        "Empirical Finance 15(5), 850–859.",
        "Lo, A. W. (2002). The Statistics of Sharpe Ratios. Financial Analysts Journal 58(4), 36–52.",
        "Newey, W. K. & West, K. D. (1987). A Simple, Positive Semi-definite, Heteroskedasticity and "
        "Autocorrelation Consistent Covariance Matrix. Econometrica 55(3), 703–708.",
        "Shu, Y., Yu, C. & Mulvey, J. M. (2024). Downside risk reduction using regime-switching signals: a "
        "statistical jump model approach. Journal of Asset Management 25(5), 493–507.",
        "Shu, Y., Yu, C. & Mulvey, J. M. (2025). Dynamic asset allocation with asset-specific regime forecasts. "
        "Annals of Operations Research 346(1), 285–318.",
        "West, M. & Harrison, J. (1997). Bayesian Forecasting and Dynamic Models, 2nd ed. Springer.",
    ]
    for r in refs:
        para(doc, r, size=9, after=2)

    doc.save(REPORT)
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    build()
