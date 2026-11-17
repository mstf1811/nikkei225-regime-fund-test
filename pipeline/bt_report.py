"""The D20 evaluation of the backtest runs, written before any test-period run (D20 (2): the analysis
is fixed in advance). With --pretest it runs on the pre-test mechanics runs, as a check of the code.

    metrics      D20 (6) for every run and benchmark, gross of fund fees and net of a 2.2356%/yr
                 trust fee (the Bayview filing, F11); cash, the EW member market and ^N225 as
                 benchmarks next to the simulated 1/N and quintile sort (D20 (5))
    H1           B0 beats cash: one-sided Newey-West t on the mean monthly excess return, with Lo's
                 (2002) standard error and the probabilistic Sharpe ratio [ours: on gross-of-fee
                 returns; net-of-fee reported alongside]
    H2           each component adds value: Ledoit-Wolf (2008) bootstrap test of the Sharpe
                 difference B0 - C1, C2, C3, C5 (block size calibrated on {1, 2, 4, 6, 8, 10},
                 K = 5000, M = 4999), Holm-adjusted across the four; C4 reported, not tested
    H3           B0's alpha: Newey-West t of the intercept on the yen market factor (EW members -
                 call rate) and French's Japan SMB, HML, RMW, CMA (5-factor file) and WML
    overfitting  deflated Sharpe ratio with N = 2 (signal lists v1 and v2, D21) and with N = all
                 logged trials, the trial Sharpe variance taken from the logged runs [ours]; the
                 t >= 3 hurdle (Harvey, Liu & Zhu 2016)
    secondary    monthly IC of alpha and mu with Newey-West t; transfer coefficient (Clarke, de Silva
                 & Thorley 2002); sub-periods; the fixed crisis episodes; sensitivity table; exposures,
                 costs and the constraint audit (D20 (7))

Run:  python pipeline/bt_report.py            (test-period runs, analysis/output/backtest/runs/)
      python pipeline/bt_report.py --pretest  (mechanics runs, analysis/output/backtest/pretest/)
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

import bt_data
import bt_regime
import evaluation as ev
from bt_signals import load_ff
from config import PROJECT_DIR

OUT = PROJECT_DIR / "analysis" / "output" / "backtest"
FEE = 0.022356
H2_RUNS = ["C1_regime_off", "C2_no_cost_terms", "C3_updater_off", "C5_hist_mean"]
SUBPERIODS = [("2005-11", "2005-01", "2011-12"), ("2012-18", "2012-01", "2018-12"), ("2019-25", "2019-01", "2025-09")]
EPISODES = [("2008-09", "2009-03"), ("2011-03", "2011-03"), ("2013-05", "2013-06"), ("2015-08", "2016-02"),
            ("2018-10", "2018-12"), ("2020-02", "2020-03"), ("2024-07", "2024-08")]


def load_runs(folder):
    runs = {}
    for d in sorted(folder.iterdir()) if folder.exists() else []:
        if (d / "daily.parquet").exists():
            runs[d.name] = {"daily": pd.read_parquet(d / "daily.parquet"),
                            "decisions": pd.read_csv(d / "decisions.csv", parse_dates=["date"]),
                            "weights": pd.read_parquet(d / "weights.parquet"),
                            "inputs": pd.read_parquet(d / "inputs.parquet") if (d / "inputs.parquet").exists() else None,
                            "meta": json.loads((d / "summary.json").read_text(encoding="utf-8"))}
    return runs


def monthly(ret_d):
    return (1 + ret_d).groupby(ret_d.index.to_period("M")).prod() - 1


def ols_nw(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    T, L = len(y), ev.nw_lags(len(y))
    Xe = X * e[:, None]
    S = Xe.T @ Xe / T
    for k in range(1, L + 1):
        G = Xe[k:].T @ Xe[:-k] / T
        S += (1 - k / (L + 1)) * (G + G.T)
    Q = np.linalg.inv(X.T @ X / T)
    se = np.sqrt(np.diag(Q @ S @ Q / T))
    return b, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretest", action="store_true")
    ap.add_argument("--K", type=int, default=5000)
    a = ap.parse_args()
    folder = OUT / ("pretest" if a.pretest else "runs")
    runs = load_runs(folder)
    if "B0" not in runs:
        raise SystemExit(f"no B0 run in {folder}")
    rep = folder / "report"
    rep.mkdir(parents=True, exist_ok=True)
    b0 = runs["B0"]["daily"]
    idx = b0.index
    end = idx[-1]
    data = bt_data.load(end)
    rf_d = data.rf.reindex(idx)
    rf_m = monthly(rf_d)
    L = [f"JMP backtest report ({'pre-test mechanics' if a.pretest else 'test period'}): "
         f"{idx[0]:%Y-%m-%d} .. {end:%Y-%m-%d}, {len(rf_m)} months", ""]

    # --- metrics, gross and net of the trust fee (D20 (6)) ---------------------------------------
    series = {k: r["daily"]["ret"] for k, r in runs.items()}
    series["cash"] = rf_d
    series["EW member market"] = data.mkt.reindex(idx)
    n225 = bt_regime.load_index(end)["r"].reindex(idx).fillna(0.0)
    series["^N225 (price)"] = n225
    rows = {}
    for k, s in series.items():
        g = ev.metrics(s, rf_d)
        net = ev.metrics(s - FEE / bt_data.YEAR_DAYS, rf_d) if k in runs else {}
        extra = {}
        if k in runs:
            d = runs[k]["daily"]
            yrs = len(d) / bt_data.YEAR_DAYS
            extra = {"Sharpe net": net["Sharpe"], "CAGR net": net["CAGR"],
                     "turnover/yr": d["turnover"].sum() / yrs,
                     "costs %/yr": (d["cost_lin"] + d["cost_imp"]).sum() / d["nav"].mean() / yrs,
                     "fees %/yr": d["fee"].sum() / d["nav"].mean() / yrs,
                     "avg gross": (d["long"] + d["short"]).mean(), "avg net": (d["long"] - d["short"]).mean(),
                     "avg beta exp": d["beta_exp"].mean()}
        rows[k] = {**{x: g[x] for x in ("CAGR", "vol", "Sharpe", "Sortino", "CEQ", "MDD", "Calmar", "ES 5% (daily)")},
                   **extra}
    table = pd.DataFrame(rows).T
    table.to_csv(rep / "metrics.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        L += ["Metrics (annualised; Sharpe from monthly excess returns; 'net' after the 2.2356%/yr trust fee)",
              table.to_string(float_format=lambda v: f"{v:.3f}"), ""]

    # --- H1 ---------------------------------------------------------------------------------------
    ex = {k: monthly(r["daily"]["ret"]) - rf_m for k, r in runs.items()}
    x = ex["B0"].to_numpy()
    nw = ev.nw_mean(x)
    sr, se = ev.sr_se_hac(x)
    p_sr = ev.psr(sr, len(x), stats.skew(x), stats.kurtosis(x, fisher=False))
    xn = (ex["B0"] - FEE / 12).to_numpy()
    nwn = ev.nw_mean(xn)
    L += ["H1: B0 beats cash (one-sided)",
          f"  gross: mean {nw['mean'] * 12:.2%}/yr, NW t {nw['t']:.2f} (lags {nw['lags']}), p {nw['p_one_sided']:.4f}; "
          f"Sharpe {sr * np.sqrt(12):.2f} (Lo s.e. {se * np.sqrt(12):.2f}); PSR {p_sr:.3f}",
          f"  net of fee: mean {nwn['mean'] * 12:.2%}/yr, NW t {nwn['t']:.2f}, p {nwn['p_one_sided']:.4f}", ""]

    # --- H2 ---------------------------------------------------------------------------------------
    pv, h2 = {}, []
    for c in H2_RUNS:
        if c not in ex:
            continue
        j = pd.concat([ex["B0"], ex[c]], axis=1).dropna()
        t = ev.lw_sharpe_test(j.iloc[:, 0].to_numpy(), j.iloc[:, 1].to_numpy(), K=a.K)
        pv[c] = t["p_boot"]
        h2.append({"comparison": c, "SR diff /yr": t["delta"] * np.sqrt(12), "se /yr": t["se_hac"] * np.sqrt(12),
                   "corr": float(j.corr().iloc[0, 1]), "block": t["block"], "p boot": t["p_boot"], "p HAC": t["p_hac"]})
    if h2:
        h2 = pd.DataFrame(h2).set_index("comparison")
        h2["p Holm"] = pd.Series(ev.holm(pv))
        h2.to_csv(rep / "h2.csv")
        L += ["H2: B0 vs each comparison run (Ledoit-Wolf 2008, two-sided; Holm across the four)",
              h2.to_string(float_format=lambda v: f"{v:.3f}"), ""]

    # --- H3 ---------------------------------------------------------------------------------------
    ff = load_ff()
    mkt_m = monthly(data.mkt.reindex(idx)) - rf_m
    F = pd.DataFrame({"mkt_yen": mkt_m}).join(ff[["smb5", "hml5", "rmw", "cma", "wml"]], how="left")
    j = pd.concat([ex["B0"].rename("y"), F], axis=1).dropna()
    if len(j) > 12:
        b, se_b = ols_nw(j["y"].to_numpy(), j.drop(columns="y").to_numpy())
        names = ["alpha"] + list(F.columns)
        h3 = pd.DataFrame({"coef": b, "NW se": se_b, "t": b / se_b}, index=names)
        h3.loc["alpha", ["coef", "NW se"]] *= 12
        h3.to_csv(rep / "h3.csv")
        L += [f"H3: B0 monthly excess on the yen market and French's Japan factors ({len(j)} months; alpha per year)",
              h3.to_string(float_format=lambda v: f"{v:.3f}"), ""]

    # --- overfitting ------------------------------------------------------------------------------
    trial_sr = np.array([ex[k].mean() / ex[k].std() for k in ex if not k.startswith("BM_")])
    v = trial_sr.var(ddof=1) if len(trial_sr) > 1 else (0.2 / np.sqrt(12)) ** 2
    sk, ku = stats.skew(x), stats.kurtosis(x, fisher=False)
    log = OUT / ("pretest_runs.csv" if a.pretest else "trials.csv")
    n_logged = len(pd.read_csv(log)) if log.exists() else len(ex)
    for N in (2, n_logged):
        trials = np.random.default_rng(0).normal(0, 1, N)
        trials = (trials - trials.mean()) / trials.std(ddof=1) * np.sqrt(v)
        d, sr0 = ev.dsr(sr, len(x), sk, ku, trials)
        L.append(f"Deflated Sharpe ratio, N = {N}: {d:.3f} (expected max Sharpe of N unskilled trials "
                 f"{sr0 * np.sqrt(12):.2f}/yr; trial Sharpe s.d. {np.sqrt(v) * np.sqrt(12):.2f}/yr)")
    L += [f"Harvey, Liu & Zhu (2016) hurdle t >= 3: B0's NW t = {nw['t']:.2f}", ""]

    # --- secondary (D20 (7)) ----------------------------------------------------------------------
    inp = runs["B0"]["inputs"]
    if inp is not None and len(inp):
        gross = (1 + data.simple).groupby(data.dates.to_period("M")).prod(min_count=1) - 1
        ic = []
        for M, g in inp.groupby("month"):
            M = pd.Period(M, "M")
            if M not in gross.index:
                continue
            r = gross.loc[M, g["code"]].to_numpy()
            ok = ~np.isnan(r)
            if ok.sum() < 50 or np.std(g["alpha"].to_numpy()[ok]) == 0:
                continue
            ic.append({"month": str(M), "IC alpha": stats.spearmanr(g["alpha"].to_numpy()[ok], r[ok])[0]})
        ic = pd.DataFrame(ic)
        if len(ic):
            t_ic = ev.nw_mean(ic["IC alpha"].to_numpy())
            L.append(f"IC of alpha with the same month's return: mean {t_ic['mean']:.4f}, NW t {t_ic['t']:.2f}, "
                     f"{len(ic)} months with alpha dispersion")
        w = runs["B0"]["weights"]
        tc = []
        for t_, row in w.iterrows():
            M = str(pd.Timestamp(t_).to_period("M"))
            g = inp[inp["month"] == M]
            if len(g) < 50 or g["alpha"].std() == 0:
                continue
            tc.append(np.corrcoef(g["alpha"], row.reindex(g["code"]).fillna(0.0))[0, 1])
        if tc:
            L.append(f"Transfer coefficient (alpha vs month-end weights): mean {np.nanmean(tc):.2f} over {len(tc)} months")
        L.append("")

    sub = {}
    for name, lo, hi in SUBPERIODS:
        for k in ("B0", "C1_regime_off", "BM_1N"):
            if k in ex:
                e = ex[k].loc[lo:hi]
                if len(e) > 2:
                    sub[(name, k)] = e.mean() / e.std() * np.sqrt(12)
    if sub:
        L += ["Sharpe by sub-period", pd.Series(sub).unstack().to_string(float_format=lambda v: f"{v:.2f}"), ""]
    ep = []
    for lo, hi in EPISODES:
        for k, s in (("B0", series["B0"]), ("C1_regime_off", series.get("C1_regime_off")),
                     ("EW member market", series["EW member market"])):
            if s is None:
                continue
            s = s.loc[lo:pd.Period(hi, "M").end_time]
            if len(s) < 2:
                continue
            nav = (1 + s).cumprod()
            ep.append({"episode": f"{lo}..{hi}", "series": k, "return": nav.iloc[-1] - 1,
                       "MDD": (nav / nav.cummax() - 1).min()})
    if ep:
        L += ["Crisis episodes (fixed in D20 (7))",
              pd.DataFrame(ep).pivot(index="episode", columns="series").to_string(float_format=lambda v: f"{v:.3f}"), ""]
    sens = table.loc[[k for k in table.index if k.startswith("S_") or k == "B0"], ["Sharpe", "Sharpe net", "CAGR", "MDD", "turnover/yr"]]
    if len(sens) > 1:
        L += ["Sensitivities (reported, never selected; D20 (8))", sens.to_string(float_format=lambda v: f"{v:.3f}"), ""]
    audit = []
    for k, r in runs.items():
        dcs = r["decisions"]
        cols = [c for c in ("trade_cap_breaches", "w_max_breaches", "w_min_breaches", "liq_breaches", "blocked_trades") if c in dcs]
        dd = r["daily"]
        prev = dd["nav"].shift(1).fillna(r["meta"]["config"]["nav0"])
        pnl = dd["interest"] - dd["fee"] + dd["dividends"] + dd["price_pnl"] - dd["cost_lin"] - dd["cost_imp"]
        audit.append({"run": k, "NAV recon err": float(((dd["nav"] - prev - pnl).abs() / prev).max()),
                      "decisions": len(dcs), **{c: int(dcs[c].sum()) for c in cols},
                      "solver failures": int((~dcs["status"].isin(["optimal", "target", "scs:optimal",
                                                                   "scs:optimal_inaccurate"])).sum())})
    L += ["Constraint audit (counts of unexplained breaches; 0 expected)",
          pd.DataFrame(audit).set_index("run").to_string(), ""]
    txt = "\n".join(L)
    (rep / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
