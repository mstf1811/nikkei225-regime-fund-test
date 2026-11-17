"""The return signals of the alpha model (D12, D15 (3), D21): month x code panels at month-ends.

Ported from analysis/signal_ic.py (build_signals), analysis/optimizer_check.py (signal_panels) and
analysis/residual_signals.py; bt_tests.py checks that the panels are identical to theirs.

    rev     log return over month m                                   (chained history, D15 (6))
    mom     log return over months m-11 .. m-1 (12-1)                 (chained)
    vol     std of daily log returns over 252 days, >= 120 returns    (chained)
    beta    on the EW member return, 252 days, >= 120 returns         (chained)
    dy      dividends over 12 months / price at ex-date, lagged 2 months (D15 (3)); own listing
    illiq   log 12-month mean of monthly Amihud ratios, >= 6 months; own listing
    resmom  residual momentum, Blitz, Huij & Martens (2011)            (D21; chained)
    resrev  residual reversal, Blitz, Huij, Lansdorp & Verbeek (2013)  (D21; chained)
"""
import numpy as np
import pandas as pd

from config import DATA_DIR

SIGNALS_V1 = ["rev", "mom", "vol", "beta", "dy", "illiq"]            # D12
SIGNALS_V2 = SIGNALS_V1 + ["resmom", "resrev"]                        # D21
WINDOW, MIN_DAYS = 252, 120
RES_WINDOW, RES_FORM = 36, 11


def zrank(s):
    """Cross-sectional z-score of the ranks (robust to outliers)."""
    r = s.rank()
    return (r - r.mean()) / r.std()


def _basic(px, liq):
    """signal_ic.build_signals, the six D12 signals (plus dy lagged 2 months) and month-end members."""
    wide = lambda col: px.pivot(index="date", columns="code", values=col)  # noqa: E731
    r = wide("log_return")
    member = wide("in_index").eq(True)
    dates = r.index
    month = dates.to_period("M")
    month_end = pd.Series(dates, index=dates).groupby(month).max()
    days_in_month = pd.Series(1, index=dates).groupby(month).sum()

    rm = r.groupby(month).sum(min_count=1)
    s = {"rev": rm}
    s["mom"] = rm.shift(1).rolling(11, min_periods=11).sum()
    n = r.notna().rolling(WINDOW, min_periods=1).sum()
    vol = r.rolling(WINDOW, min_periods=MIN_DAYS).std() * np.sqrt(252)
    s["vol"] = vol.loc[month_end.values].set_axis(month_end.index)
    mkt = r.where(member).mean(axis=1)
    m = r.notna().mul(mkt, axis=0).where(r.notna())
    sx, sy = r.rolling(WINDOW, min_periods=1).sum(), m.rolling(WINDOW, min_periods=1).sum()
    sxy, syy = (r * m).rolling(WINDOW, min_periods=1).sum(), (m * m).rolling(WINDOW, min_periods=1).sum()
    beta = ((sxy - sx * sy / n) / (syy - sy ** 2 / n)).where(n >= MIN_DAYS)
    s["beta"] = beta.loc[month_end.values].set_axis(month_end.index)

    div = px[(px["dividend"] > 0) & (px["dividend_source"] != "spin-off of 8729")]
    y = (div["dividend"] / div["close"]).groupby([div["date"].dt.to_period("M"), div["code"]]).sum()
    ym = y.unstack().reindex(index=rm.index, columns=rm.columns).fillna(0.0)
    full_year = rm.notna().rolling(12, min_periods=12).sum() == 12
    s["dy"] = ym.rolling(12, min_periods=12).sum().where(full_year)
    s["dy_lag2"] = s["dy"].shift(2).where(full_year)

    value_daily = (px["close"] * px["volume"]).groupby([px["date"].dt.to_period("M"), px["code"]]).sum(min_count=1)
    value = value_daily.unstack().reindex(index=rm.index, columns=rm.columns)
    lq = liq.assign(month=pd.PeriodIndex(liq["month"], freq="M")).pivot(index="month", columns="code", values="value")
    lq = lq.reindex(index=rm.index, columns=rm.columns)
    monthly_table = value.index >= pd.Period("2020-01", "M")
    value.loc[monthly_table] = lq.loc[monthly_table]
    adv = value.div(days_in_month, axis=0)
    abs_r = r.abs().groupby(month).mean()
    amihud = (abs_r / adv.where(adv > 0)) * 1e9
    s["illiq"] = np.log(amihud.rolling(12, min_periods=6).mean())
    members = member.loc[month_end.values].set_axis(month_end.index)
    return s, members


def load_ff():
    ff = pd.read_csv(DATA_DIR / "market" / "ff_japan" / "ff_japan_monthly.csv")
    return ff.set_index(pd.PeriodIndex(ff["month"], freq="M"))


def residual(simple, rf, mkt, end):
    """D21: residual momentum and residual reversal (and the 36-month factor loadings)."""
    month = simple.index.to_period("M")
    gross = (1 + simple).groupby(month).prod(min_count=1) - 1
    rf_m = (1 + rf).groupby(month).prod() - 1
    mkt_m = (1 + mkt).groupby(month).prod() - 1
    y = gross.sub(rf_m, axis=0).iloc[1:]                   # 2000-01 lacks its first day's return
    ff = load_ff()
    ff = ff[ff.index <= pd.Period(end, "M")]
    f = pd.DataFrame({"mkt": mkt_m - rf_m, "smb": ff["smb3"], "hml": ff["hml3"]}).reindex(y.index)
    names = ["resmom", "resrev", "b_mkt", "b_smb", "b_hml"]
    out = {k: pd.DataFrame(np.nan, index=y.index, columns=y.columns) for k in names}
    for i in range(RES_WINDOW - 1, len(y)):
        Y, F = y.iloc[i - RES_WINDOW + 1: i + 1], f.iloc[i - RES_WINDOW + 1: i + 1]
        ok = Y.notna().all()
        if F.isna().any().any() or not ok.any():
            continue
        Yv, Fv = Y.loc[:, ok].to_numpy(), F.to_numpy()
        B = np.linalg.lstsq(np.column_stack([np.ones(RES_WINDOW), Fv]), Yv, rcond=None)[0]
        e = Yv - Fv @ B[1:]                                # the intercept stays in the residual
        form = e[-1 - RES_FORM:-1]
        m, cols = y.index[i], Y.columns[ok]
        out["resmom"].loc[m, cols] = form.sum(axis=0) / form.std(axis=0, ddof=1)
        out["resrev"].loc[m, cols] = e[-1] / e.std(axis=0, ddof=1)
        for j, k in enumerate(["b_mkt", "b_smb", "b_hml"], start=1):
            out[k].loc[m, cols] = B[j]
    return out


def build(data, names=SIGNALS_V2):
    """Signal panels (month x code) and the month-end member table, from bt_data.Data."""
    px, ch = data.px, data.chain
    extra = ch[["code", "date", "log_return"]]
    extra = extra[extra["date"].isin(px["date"].unique())]
    extra = extra[~extra.set_index(["date", "code"]).index.isin(px.set_index(["date", "code"]).index)]
    px_aug = pd.concat([px, extra.assign(in_index=False, dividend=0.0, dividend_source="none")],
                       ignore_index=True)
    s_chain, _ = _basic(px_aug, data.liq)
    s_plain, members = _basic(px, data.liq)
    signals = {k: s_chain[k] for k in ("rev", "mom", "vol", "beta")}
    signals["dy"] = s_plain["dy_lag2"]
    signals["illiq"] = s_plain["illiq"]
    if {"resmom", "resrev"} & set(names):
        res = residual(data.simple, data.rf, data.mkt, data.end)
        for k in ("resmom", "resrev"):
            signals[k] = res[k].reindex(index=signals["rev"].index, columns=signals["rev"].columns)
    members = members.reindex(columns=signals["rev"].columns).fillna(False)
    return {k: signals[k] for k in names}, members
