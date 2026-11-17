"""E2 (optimizer) construction check at the first rebalance of the test period, using only data up
to the last trading day of 2004. Structural only: no return after 2004 is read.

The optimizer settings follow the literature (D18):
    objective   Boyd, Busseti, Diamond, Kahn, Koh, Nystrup & Speth (2017), single-period
                optimisation (SPO, eq. 4.9), with the risk and trade aversion of Shu, Yu & Mulvey
                (2025): gamma_risk = 10, gamma_trade = 1, gamma_hold = 1.
                    maximise  mu'w - g_trade*phi_trade(w - w_prev) - g_hold*phi_hold(w) - g_risk*w'Sw
    costs       phi_trade = a|z| + b*sigma*|z|^1.5 / (V/v)^0.5  (Boyd et al. eq. 2.3), a = 0.15%
                (D13), b = 1, sigma = 10-day mean of |ln open - ln close| (Boyd et al. 7.1),
                V = the previous month's average daily trading value [ours: the 2020+ volume is
                monthly], v = NAV;
                phi_hold = s'(w)_-, s = 1.15%/yr borrow fee + the cash rate (the short proceeds earn
                0%, D13).
    constraints long <= 100% and short <= 100% of NAV (D13); w_i <= 10% (JITA); w_i >= -5% (Boyd
                et al. 4.4 example) [ours for the short level]; JITA high-liquidity tier: every
                position sellable or coverable within 3 business days at 10% of the daily volume,
                |w_i| v <= 0.3 V_i; a rebalance trades at most 10% of a day's volume, |z_i| v <= 0.1 V_i.
    inputs      D15 / D16 / D17 built here at the 2004 year-end: Vasicek beta (5 years, EW member
                proxy), the six D12 signals (dividends lagged 2 months), Fama-MacBeth gamma with
                empirical-Bayes shrinkage, alpha trimmed at 3 cross-sectional standard deviations
                (Grinold & Kahn), Ledoit-Wolf (2003) covariance toward the single-index target, and
                the regime premium (D17: pi_bull = training bull-day mean, pi_bear = 0).
    fund size   NAV 10 bn yen [ours: Boyd et al.'s smaller example is $100M; real Japanese public
                long-short funds are about 1 bn yen], robustness 100 bn yen.

Units: one period = one month (the D7 rebalance); returns and covariance are monthly, costs are
one-off fractions of NAV, so gamma_trade = 1 amortises a trade over one month (Boyd et al. 4.8).

Outputs (analysis/output/optimizer/): summary.txt, inputs_2005-01.csv, weights_2005-01.csv

Run:  python analysis/optimizer_check.py
"""
import sys
import warnings
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, MEMBERSHIP_DIR, PROJECT_DIR  # noqa: E402
import regime_jm as rj  # noqa: E402
from signal_ic import SIGNALS, build_signals, zrank  # noqa: E402

warnings.filterwarnings("ignore")
OUT = PROJECT_DIR / "analysis" / "output" / "optimizer"
CUT = pd.Timestamp("2004-12-31")
YEAR_DAYS = 245
MONTH_DAYS = YEAR_DAYS / 12
WINDOW_YEARS = 5                 # D8
MIN_BETA_DAYS = 250              # D15 (2)
COVERAGE = 0.90                  # a full 5-year window (else rows from the single-index target)
FM_MIN_STOCKS = 50

# D18 settings
G_RISK, G_TRADE, G_HOLD = 10.0, 1.0, 1.0
A_COST = 0.0015                  # one-way linear cost (D13)
B_IMPACT = 1.0
BORROW = 0.0115                  # per year (D13)
LONG_MAX, SHORT_MAX = 1.0, 1.0   # D13
W_MAX, W_MIN = 0.10, -0.05
PARTICIPATION, LIQ_DAYS = 0.10, 3
NAV = 10e9
ALPHA_TRIM = 3.0


# ----------------------------------------------------------------------------------------------
# data

def load():
    px = pd.read_parquet(DATA_DIR / "prices_daily.parquet",
                         columns=["date", "code", "in_index", "open", "close", "log_return", "volume",
                                  "dividend", "dividend_source"])
    px["date"] = pd.to_datetime(px["date"])
    px = px[px["date"] <= CUT]
    liq = pd.read_parquet(DATA_DIR / "liquidity_monthly.parquet", columns=["month", "code", "value"])
    ch = pd.read_parquet(MEMBERSHIP_DIR / "successor_returns.parquet")
    ch["date"] = pd.to_datetime(ch["date"])
    ch = ch[ch["date"] <= CUT]
    rate = rj.load_rf(pd.DatetimeIndex(sorted(px["date"].unique())))
    return px, liq, ch, rate


def daily_panels(px, ch, rate):
    r = px.pivot(index="date", columns="code", values="log_return")
    # D15 (6): chained predecessor returns before a successor's first day
    rc = ch.pivot(index="date", columns="code", values="log_return").reindex(index=r.index)
    for c in rc.columns:
        if c in r.columns:
            r[c] = r[c].fillna(rc[c])
    member = px.pivot(index="date", columns="code", values="in_index").reindex_like(r).eq(True)
    simple = np.expm1(r)
    gap = pd.Series(r.index, index=r.index).diff().dt.days.fillna(1)
    rf = rate.shift(1).bfill() / 100 * gap / 365
    mkt = simple.where(member).mean(axis=1)                     # D15 (1): EW member proxy
    return r, simple, member, rf, mkt


# ----------------------------------------------------------------------------------------------
# beta (D15 (2)) and covariance (D16)

def ols_beta(y, x):
    """Per-column OLS slope of y on x (both excess), on each column's own days, with its s.e."""
    ok = y.notna()
    xm = ok.mul(x, axis=0).where(ok)
    n = ok.sum()
    sx, sy = xm.sum(), y.sum()
    sxx = (xm ** 2).sum() - sx ** 2 / n
    sxy = (xm * y).sum() - sx * sy / n
    syy = (y ** 2).sum() - sy ** 2 / n
    b = sxy / sxx
    resid = (syy - b * sxy) / (n - 2)
    return b.where(n >= MIN_BETA_DAYS), np.sqrt(resid / sxx).where(n >= MIN_BETA_DAYS), resid, n


def vasicek(b, se):
    ok = b.notna() & se.notna()
    prior_var = max(b[ok].var() - (se[ok] ** 2).mean(), 1e-4)
    return (prior_var * b + se ** 2 * b[ok].mean()) / (prior_var + se ** 2)


def betas_at(simple, rf, mkt, date, codes):
    lo = date - pd.DateOffset(years=WINDOW_YEARS)
    y = simple.loc[(simple.index > lo) & (simple.index <= date), codes].sub(rf, axis=0)
    x = (mkt - rf).loc[y.index]
    b, se, resid, n = ols_beta(y, x)
    return vasicek(b, se), b, resid, n


def cov_market(x):
    """Ledoit & Wolf (2003) shrinkage toward the single-index target (their covMarket code);
    x is T x N excess returns without missing values; the market is the EW mean of x."""
    t, n = x.shape
    x = x - x.mean(axis=0)
    xm = x.mean(axis=1, keepdims=True)
    full = np.cov(np.hstack([x, xm]), rowvar=False) * (t - 1) / t
    covm, varm = full[:n, n], full[n, n]
    sample = full[:n, :n]
    prior = np.outer(covm, covm) / varm
    np.fill_diagonal(prior, np.diag(sample))
    c = np.linalg.norm(sample - prior, "fro") ** 2
    y = x ** 2
    p = (y.T @ y).sum() / t - (sample ** 2).sum()
    rdiag = (y ** 2).sum() / t - (np.diag(sample) ** 2).sum()
    z = x * xm
    v1 = y.T @ z / t - covm[:, None] * sample
    roff1 = (v1 * covm[None, :]).sum() / varm - (np.diag(v1) * covm).sum() / varm
    v3 = z.T @ z / t - varm * sample
    roff3 = (v3 * np.outer(covm, covm)).sum() / varm ** 2 - (np.diag(v3) * covm ** 2).sum() / varm ** 2
    r = rdiag + 2 * roff1 - roff3
    k = (p - r) / c
    delta = max(0.0, min(1.0, k / t))
    return delta * prior + (1 - delta) * sample, delta


def covariance(simple, rf, mkt, date, codes, beta_ols, resid):
    """D16: 5-year daily excess returns; days with no member trade dropped; isolated no-trade days
    set to 0; members without a full window get single-index rows (beta, residual variance)."""
    lo = date - pd.DateOffset(years=WINDOW_YEARS)
    y = simple.loc[(simple.index > lo) & (simple.index <= date), codes].sub(rf, axis=0)
    y = y[y.notna().any(axis=1)]
    cover = y.notna().mean()
    full = cover[cover >= COVERAGE].index
    short = [c for c in codes if c not in full]
    S, delta = cov_market(y[full].fillna(0.0).to_numpy())
    sig = pd.DataFrame(S, index=full, columns=full)
    if short:
        m = (mkt - rf).loc[y.index]
        vm = m.var()
        b_all = beta_ols.reindex(codes)
        sig = sig.reindex(index=codes, columns=codes)
        for c in short:
            bc = b_all[c] if pd.notna(b_all[c]) else 1.0
            rv = resid[c] if pd.notna(resid.get(c, np.nan)) else y[c].var()
            sig.loc[c, :] = bc * vm * pd.Series({k: (b_all[k] if pd.notna(b_all[k]) else 1.0)
                                                 for k in codes})
            sig.loc[:, c] = sig.loc[c, :]
            sig.loc[c, c] = bc ** 2 * vm + rv
    sig = sig.loc[codes, codes]
    sig = (sig + sig.T) / 2
    ev = np.linalg.eigvalsh(sig.to_numpy())
    if ev.min() <= 0:                               # D16 (3): the final matrix must be PD
        vals, vecs = np.linalg.eigh(sig.to_numpy())
        sig = pd.DataFrame(vecs @ np.diag(np.maximum(vals, 1e-10)) @ vecs.T, index=codes, columns=codes)
    return sig, delta, len(full), short, ev.min()


# ----------------------------------------------------------------------------------------------
# alpha (D15 (3)-(4), (7))

def signal_panels(px, liq, ch, simple, rf, mkt, names=SIGNALS):
    """Month x code signal panels: the returns-based ones with the chained history (D15 (6)), the
    dividend and illiquidity ones without; dividend yield lagged 2 months (D15 (3)). The D21
    residual signals are added when `names` asks for them."""
    extra = ch[["code", "date", "log_return"]]
    extra = extra[extra["date"].isin(px["date"].unique())]
    extra = extra[~extra.set_index(["date", "code"]).index.isin(px.set_index(["date", "code"]).index)]
    px_aug = pd.concat([px, extra.assign(in_index=False, dividend=0.0, dividend_source="none")],
                       ignore_index=True)
    s_chain, _, _, _ = build_signals(px_aug, liq)
    s_plain, _, members_me, _ = build_signals(px, liq)
    signals = {k: s_chain[k] for k in ("rev", "mom", "vol", "beta")}
    signals["dy"] = s_plain["dy_lag2"]
    signals["illiq"] = s_plain["illiq"]
    if {"resmom", "resrev"} & set(names):
        from residual_signals import residual_signals
        res = residual_signals(simple, rf, mkt)
        for k in ("resmom", "resrev"):
            signals[k] = res[k].reindex(index=signals["rev"].index, columns=signals["rev"].columns)
    members_me = members_me.reindex(columns=signals["rev"].columns).fillna(False)
    return signals, members_me


def fama_macbeth(px, simple, member, rf, mkt, signals, members_me, names=SIGNALS):
    g = fm_slopes(simple, rf, mkt, signals, members_me, names)
    gbar, se = g.mean(), g.std(ddof=1) / np.sqrt(len(g))
    tau2 = max(0.0, (gbar ** 2).mean() - (se ** 2).mean())
    shrunk = gbar * tau2 / (tau2 + se ** 2)
    return pd.DataFrame({"gamma_bar": gbar, "se": se, "t": gbar / se, "gamma_eb": shrunk}), len(g), tau2


def fm_slopes(simple, rf, mkt, signals, members_me, names=SIGNALS):
    """Monthly cross-sectional slopes (D15 (4)); row = signal month, slope on next month's return.
    A month enters only when every signal covers at least FM_MIN_STOCKS members."""
    month = simple.index.to_period("M")
    gross = (1 + simple).groupby(month).prod(min_count=1) - 1
    rf_m = (1 + rf).groupby(month).prod() - 1
    mkt_m = (1 + mkt).groupby(month).prod() - 1
    months = [m for m in signals["rev"].index if m < pd.Period(CUT, "M")]
    rows = []
    for m in months:
        me = simple.index[month == m].max()
        codes = members_me.loc[m][members_me.loc[m]].index
        z = pd.DataFrame({k: signals[k].loc[m, codes] for k in names}).apply(zrank)
        if z.notna().sum().min() < FM_MIN_STOCKS:
            continue
        bv, *_ = betas_at(simple, rf, mkt, me, list(codes))
        nxt = m + 1
        y = gross.loc[nxt, codes] - rf_m.loc[nxt] - bv * (mkt_m.loc[nxt] - rf_m.loc[nxt])
        d = z.fillna(0.0).assign(y=y).dropna()
        if len(d) < FM_MIN_STOCKS:
            continue
        X = np.column_stack([np.ones(len(d)), d[names].to_numpy()])
        coef = np.linalg.lstsq(X, d["y"].to_numpy(), rcond=None)[0]
        rows.append(pd.Series(coef[1:], index=names, name=m))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# regime premium (D17)

def regime_premium():
    f = rj.load()
    lo, hi = CUT - pd.DateOffset(years=rj.VALID_YEARS), CUT
    refits = [d for d in pd.date_range("1996-07-01", "2004-07-01", freq="6MS")
              if d.month in (1, 7)]
    ends = refits[1:] + [CUT + pd.Timedelta(days=1)]
    best, scores = None, {}
    from joblib import Parallel, delayed
    labs = Parallel(n_jobs=-1)(delayed(lambda lam: pd.concat(
        [rj.jm_period(f, a, b, lam)[0] for a, b in zip(refits, ends)]))(lam) for lam in rj.LAMBDAS)
    for lam, lab in zip(rj.LAMBDAS, labs):
        ret, _ = rj.strategy(f, lab)
        scores[lam] = rj.sharpe(ret.loc[lo:hi], f["rf"])
    best = max(scores, key=scores.get)
    _, st = rj.jm_period(f, pd.Timestamp("2005-01-01"), pd.Timestamp("2005-01-02"), best)
    labels = pd.read_parquet(PROJECT_DIR / "analysis" / "output" / "regime_jm" / "labels.parquet")
    bear_now = int(labels["jm_bear"].loc[:CUT].iloc[-1])
    return {"lambda": best, "valid_sharpe": scores[best], "pi_bull_m": st["bull_mean"] * MONTH_DAYS,
            "bear_mean_m": st["bear_mean"] * MONTH_DAYS, "bear_now": bear_now,
            "label_date": labels.loc[:CUT].index[-1]}


# ----------------------------------------------------------------------------------------------
# optimizer (D18)

def solve(mu_ex, Sigma, w_prev, sigma_d, adv, rf_y, nav=NAV, g_risk=G_RISK, g_trade=G_TRADE,
          g_hold=G_HOLD):
    n = len(mu_ex)
    w = cp.Variable(n)
    z = w - w_prev
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(n))
    liq = adv / nav
    trade_cost = A_COST * cp.sum(cp.abs(z)) + cp.sum(
        cp.multiply(B_IMPACT * sigma_d / np.sqrt(liq), cp.power(cp.abs(z), 1.5)))
    s = (BORROW + rf_y) / 12                    # fee + the cash rate the short proceeds forgo
    hold_cost = s * cp.sum(cp.neg(w))
    risk = cp.sum_squares(L.T @ w)
    # in basis points of NAV, for the solver's accuracy (the terms are ~1e-3 in fractions)
    obj = cp.Maximize(1e4 * (mu_ex @ w - g_trade * trade_cost - g_hold * hold_cost - g_risk * risk))
    cons = [cp.sum(cp.pos(w)) <= LONG_MAX, cp.sum(cp.neg(w)) <= SHORT_MAX,
            w <= W_MAX, w >= W_MIN,
            cp.abs(w) <= LIQ_DAYS * PARTICIPATION * liq,
            cp.abs(z) <= PARTICIPATION * liq]
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.CLARABEL)
    if prob.status != cp.OPTIMAL:   # Clarabel flags the 3/2-power cone as inaccurate; SCS confirms
        prob.solve(solver=cp.SCS, eps=1e-7, max_iters=200000)
    wv = np.where(np.abs(w.value) < 1e-6, 0.0, w.value)
    tc = A_COST * np.abs(wv - w_prev).sum() + (B_IMPACT * sigma_d / np.sqrt(liq)
                                               * np.abs(wv - w_prev) ** 1.5).sum()
    return wv, prob.status, tc


def describe(name, w, mu_mkt, alpha, beta, Sigma, adv, tc, nav=NAV):
    liq_cap = LIQ_DAYS * PARTICIPATION * adv / nav
    vol = np.sqrt(w @ Sigma @ w * 12)
    return {"case": name,
            "long": w[w > 0].sum(), "short": -w[w < 0].sum(), "net": w.sum(),
            "beta exposure": beta @ w, "n long": int((w > 0).sum()), "n short": int((w < 0).sum()),
            "ex-ante vol": vol,
            "E[excess] market": (mu_mkt @ w) * 12, "E[excess] alpha": (alpha @ w) * 12,
            "trade cost": tc,
            "at +10%": int((w >= W_MAX - 1e-5).sum()), "at -5%": int((w <= W_MIN + 1e-5).sum()),
            "at liquidity cap": int(((np.abs(w) >= liq_cap - 1e-6) & (np.abs(w) > 1e-6)).sum())}


def main(names=SIGNALS, out=OUT):
    """names: the signal list (D12's six by default; D21's eight from analysis/signals_v2_check.py)."""
    out.mkdir(parents=True, exist_ok=True)
    px, liq, ch, rate = load()
    r, simple, member, rf, mkt = daily_panels(px, ch, rate)
    C = r.index.max()
    print(f"data to {C:%Y-%m-%d}; {r.shape[1]} codes")

    signals, members_me = signal_panels(px, liq, ch, simple, rf, mkt, names)
    fm, n_months, tau2 = fama_macbeth(px, simple, member, rf, mkt, signals, members_me, names)
    print("Fama-MacBeth done")

    m_now = pd.Period(C, "M")
    codes = list(members_me.loc[m_now][members_me.loc[m_now]].index)
    z = pd.DataFrame({k: signals[k].loc[m_now, codes] for k in names}).apply(zrank).fillna(0.0)
    alpha = z @ fm["gamma_eb"]
    sd = alpha.std()
    alpha_trim = alpha.clip(-ALPHA_TRIM * sd, ALPHA_TRIM * sd)

    bv, b_ols, resid, nobs = betas_at(simple, rf, mkt, C, codes)
    bv = bv.fillna(bv.mean())
    Sigma_d, delta, n_full, short, ev_min = covariance(simple, rf, mkt, C, codes, b_ols, resid)
    Sigma = Sigma_d.to_numpy() * MONTH_DAYS
    print("covariance done")

    reg = regime_premium()
    print("regime done")
    pi_bull = reg["pi_bull_m"]
    pi_now = 0.0 if reg["bear_now"] else pi_bull
    lo = C - pd.DateOffset(years=WINDOW_YEARS)
    pi_unc = ((mkt - rf).loc[lo:C]).mean() * MONTH_DAYS      # D15 (5): unconditional premium

    # liquidity and impact inputs (previous month = December 2004)
    dec = px[(px["date"].dt.to_period("M") == m_now) & px["code"].isin(codes)]
    adv = (dec["close"] * dec["volume"]).groupby(dec["code"]).sum() / dec["date"].nunique()
    last10 = px[px["code"].isin(codes) & (px["date"] > r.index[-11])]
    sig_d = (np.log(last10["open"]) - np.log(last10["close"])).abs().groupby(last10["code"]).mean()
    adv = adv.reindex(codes).fillna(adv.median())
    sig_d = sig_d.reindex(codes).fillna(sig_d.median())
    rf_y = rate.loc[:C].iloc[-1] / 100

    beta = bv.reindex(codes).to_numpy()
    a = alpha_trim.reindex(codes).to_numpy()
    w0 = np.zeros(len(codes))
    cases = [("baseline (regime now)", pi_now, {}),
             ("bull premium", pi_bull, {}), ("bear (pi = 0)", 0.0, {}),
             ("bull, pi = unconditional (D17 sensitivity)", pi_unc, {}),
             ("bull, gamma_risk 5", pi_bull, {"g_risk": 5.0}),
             ("bull, gamma_risk 20", pi_bull, {"g_risk": 20.0}),
             ("bull, gamma_trade 0", pi_bull, {"g_trade": 0.0}),
             ("bull, NAV 100 bn yen", pi_bull, {"nav": 100e9}),
             ("alpha only (no market term)", 0.0, {})]
    rows, weights, paths = [], {}, {}
    for name, pi, kw in cases:
        mu_mkt = beta * pi
        mu = mu_mkt + a
        w, status, tc = solve(mu, Sigma, w0, sig_d.to_numpy(), adv.to_numpy(), rf_y, **kw)
        d = describe(name + " | day 1", w, mu_mkt, a, beta, Sigma, adv.to_numpy(), tc, kw.get("nav", NAV))
        d["status"] = status
        rows.append(d)
        # build-up: repeated rebalance days at unchanged prices until trading stops
        tc_sum, days, turn = tc, 1, [np.abs(w).sum() / 2]
        for days in range(2, 21):
            w_new, status, tc = solve(mu, Sigma, w, sig_d.to_numpy(), adv.to_numpy(), rf_y, **kw)
            tc_sum += tc
            turn.append(np.abs(w_new - w).sum() / 2)
            done = np.abs(w_new - w).sum() / 2 < 0.005         # turnover below 0.5% of NAV
            w = w_new
            if done:
                break
        d = describe(name + f" | built ({days} days)", w, mu_mkt, a, beta, Sigma, adv.to_numpy(),
                     tc_sum, kw.get("nav", NAV))
        d["status"] = status
        rows.append(d)
        weights[name] = w
        paths[name] = np.cumsum(turn) / np.sum(turn)
    res = pd.DataFrame(rows).set_index("case")

    inputs = pd.DataFrame({"beta_vasicek": bv.reindex(codes), "beta_ols": b_ols.reindex(codes),
                           "alpha_m": alpha.reindex(codes), "alpha_trim_m": alpha_trim.reindex(codes),
                           "adv_bn": adv / 1e9, "sigma_daily": sig_d,
                           "liq_cap": LIQ_DAYS * PARTICIPATION * adv / NAV}).join(z)
    inputs.to_csv(out / "inputs_2005-01.csv")
    pd.DataFrame(weights, index=codes).to_csv(out / "weights_2005-01.csv")

    L = []
    L.append(f"E2 construction check: first rebalance of 2005, data to {C:%Y-%m-%d}")
    L.append(f"members {len(codes)}; full 5-year window {n_full}; single-index rows {len(short)} "
             f"({', '.join(map(str, short))})")
    L.append(f"Ledoit-Wolf shrinkage intensity toward the single-index target: {delta:.2f}; "
             f"smallest eigenvalue before repair {ev_min:.2e}")
    L.append(f"beta (Vasicek): median {bv.median():.2f}, 5-95% {bv.quantile(.05):.2f}-{bv.quantile(.95):.2f}")
    L.append("")
    L.append(f"Fama-MacBeth over {n_months} months (signal months to 2004-11), monthly units; "
             f"EB tau^2 = {tau2:.2e}")
    L.append(fm.to_string(float_format=lambda v: f"{v:.5f}"))
    L.append(f"alpha (monthly): sd {sd*100:.3f}% ({sd*12*100:.2f}%/yr); trimmed at +-{ALPHA_TRIM} sd "
             f"({(alpha.abs() > ALPHA_TRIM*sd).sum()} names)")
    L.append("")
    L.append(f"Regime (D17): lambda {reg['lambda']} (8-year validation Sharpe {reg['valid_sharpe']:.2f}); "
             f"label on {reg['label_date']:%Y-%m-%d}: {'bear' if reg['bear_now'] else 'bull'}")
    L.append(f"  pi_bull = {reg['pi_bull_m']*12*100:.1f}%/yr (training bull-day mean); training "
             f"bear-day mean {reg['bear_mean_m']*12*100:.1f}%/yr; unconditional (EW proxy, 5y) "
             f"{pi_unc*12*100:.1f}%/yr")
    L.append(f"cash rate {rf_y*100:.3f}%; NAV {NAV/1e9:.0f} bn yen; ADV median {adv.median()/1e9:.2f} bn, "
             f"5% {adv.quantile(.05)/1e9:.2f} bn; liquidity cap below 10% for "
             f"{(LIQ_DAYS*PARTICIPATION*adv/NAV < W_MAX).sum()} names")
    L.append("")
    L.append("Portfolios from cash (weights in NAV; returns and vol annualised; trade cost one-off)")
    short_names = {"beta exposure": "beta", "ex-ante vol": "vol", "E[excess] market": "E mkt",
                   "E[excess] alpha": "E alpha", "trade cost": "cost", "at +10%": "@+10%",
                   "at -5%": "@-5%", "at liquidity cap": "@liq", "n long": "#L", "n short": "#S"}
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        L.append(res.rename(columns=short_names).to_string(float_format=lambda v: f"{v:.3f}"))
    L.append("")
    L.append("Build-up from cash: share of the total trading done by day 1, 2, 3, 5 (unchanged prices)")
    for name, cum in paths.items():
        L.append(f"  {name}: " + ", ".join(f"{cum[min(d, len(cum)) - 1]:.0%}" for d in (1, 2, 3, 5)))
    txt = "\n".join(L)
    (out / "summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return res, fm


if __name__ == "__main__":
    main()
