"""Backtest data: every stock-level input of the simulation, loaded once and cut at `end`.

Nothing after `end` is read, so a run that stops before the test period (2005-01-01) cannot see it
(D20 (2)); the regime model loads its own index series (bt_regime.py) with the same cut.

Panels (date x code unless stated):
    r          daily log total returns with the chained predecessor returns before a successor's
               first day (D15 (6)); used for estimation only
    simple     expm1(r)
    gp         price factor of a held position on its own listing: split * close / previous close;
               1 on days without a return (no trade: the position keeps its last close, JITA, F6)
    dy         dividend paid on the day per yen held at the previous close: split * dividend /
               previous close (imputed dividends included, D10); 0 when there is no return
    dy_imp     the imputed part of dy (for the D20 (8) sensitivity "imputed dividends set to 0")
    member     index member that day (D5)
    traded     the stock has a return that day (it traded; it can be traded at the close)
    no_buy     closed at limit-up (close_up / quote_up): no buying, no short covering (D11, D18 (5))
    no_sell    closed at limit-down (close_down / quote_down): no selling, no shorting
    sigma_d    10-day mean of |ln open - ln close| (Boyd et al. 7.1, D18 (3)), on the stock's days
    adv        month x code: the month's average daily trading value in yen (to 2019 close x volume
               from the daily file, from 2020 the JPX monthly tables), used for the next month (D18)
    last_day   per code: the last day with a price row
Series: rate (call rate, % p.a.), rf (the cash return earned on day t: rate of t-1 x days / 365),
mkt (D15 (1) equal-weighted member return, simple).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import DATA_DIR, MEMBERSHIP_DIR

TEST_START = pd.Timestamp("2005-01-01")
YEAR_DAYS = 245
MONTH_DAYS = YEAR_DAYS / 12


def load_rate(dates):
    """Call rate in % p.a. on each date: the daily series from 1998, monthly averages before."""
    d = pd.read_csv(DATA_DIR / "market" / "call_rate_daily.csv", parse_dates=["date"])
    m = pd.read_csv(DATA_DIR / "market" / "call_rate_monthly.csv")
    month = pd.Series(dates.strftime("%Y-%m"), index=dates)
    rate = month.map(m.set_index("month")["call_rate"]).astype(float)
    daily = d.set_index("date")["call_rate"].reindex(dates).ffill()
    rate[dates >= d["date"].min()] = daily[dates >= d["date"].min()]
    return rate


def accrual(rate, index):
    """Cash return of each day: the previous day's rate over the calendar days elapsed."""
    gap = pd.Series(index, index=index).diff().dt.days.fillna(1)
    return rate.shift(1).bfill() / 100 * gap / 365


@dataclass
class Data:
    end: pd.Timestamp
    dates: pd.DatetimeIndex
    r: pd.DataFrame
    simple: pd.DataFrame
    gp: pd.DataFrame
    dy: pd.DataFrame
    dy_imp: pd.DataFrame
    member: pd.DataFrame
    traded: pd.DataFrame
    no_buy: pd.DataFrame
    no_sell: pd.DataFrame
    sigma_d: pd.DataFrame
    adv: pd.DataFrame
    last_day: pd.Series
    rate: pd.Series
    rf: pd.Series
    mkt: pd.Series
    px: pd.DataFrame = field(repr=False)        # long price rows (for the signals)
    liq: pd.DataFrame = field(repr=False)       # JPX monthly liquidity rows
    chain: pd.DataFrame = field(repr=False)     # chained predecessor rows (D15 (6))

    def month_ends(self):
        s = pd.Series(self.dates, index=self.dates)
        return s.groupby(self.dates.to_period("M")).max()


def load(end, imputed_dividends=True):
    """imputed_dividends=False is the D20 (8) sensitivity: imputed dividends (D10) are removed from
    the returns and the dividend-yield signal alike, as if they had never been paid."""
    end = pd.Timestamp(end)
    px = pd.read_parquet(DATA_DIR / "prices_daily.parquet",
                         columns=["date", "code", "in_index", "open", "close", "log_return", "volume",
                                  "split", "dividend", "dividend_source"])
    px["date"] = pd.to_datetime(px["date"])
    px = px[px["date"] <= end]
    liq = pd.read_parquet(DATA_DIR / "liquidity_monthly.parquet", columns=["month", "code", "value", "adv_value"])
    liq = liq[pd.PeriodIndex(liq["month"], freq="M") <= pd.Period(end, "M")]
    ch = pd.read_parquet(MEMBERSHIP_DIR / "successor_returns.parquet")
    ch["date"] = pd.to_datetime(ch["date"])
    ch = ch[ch["date"] <= end]
    lim = pd.read_parquet(DATA_DIR / "price_limits_daily.parquet", columns=["date", "code", "hit"])
    lim["date"] = pd.to_datetime(lim["date"])
    lim = lim[lim["date"] <= end]

    wide = lambda col: px.pivot(index="date", columns="code", values=col)  # noqa: E731
    lr = wide("log_return")
    dates, codes = lr.index, lr.columns
    w = lambda d: d.reindex(index=dates, columns=codes)  # noqa: E731

    # holdings: price factor and dividend cash, on the stock's own listing only. A split or dividend
    # dated on a no-trade day is carried to the next traded day, as the stored return does
    # (pipeline/adjust.py), so exp(log_return) = gp + dy exactly.
    close = wide("close")
    prev = close.ffill().shift(1)
    traded = lr.notna()
    since_trade = lambda c: c - c.where(traded).ffill().shift(1).fillna(0.0)  # noqa: E731
    split = np.exp(since_trade(np.log(w(wide("split")).fillna(1.0)).cumsum()))
    gp = (split * close / prev).where(traded, 1.0)
    dy = (np.exp(lr) - gp).where(traded, 0.0)
    div = since_trade(w(wide("dividend")).fillna(0.0).cumsum())
    imp = px["dividend_source"].eq("imputed")
    div_imp = since_trade(w(px[imp].pivot(index="date", columns="code", values="dividend")).fillna(0.0).cumsum())
    dy_imp = (dy * (div_imp / div.where(div > 0))).where(traded & (div > 0), 0.0)
    if not imputed_dividends:
        # remove the imputed part of each day's dividend from the return, here and in the price rows
        # the signals read (the dividend-yield signal then sees no imputed dividend)
        lr = np.log(np.exp(lr) - dy_imp).where(traded)
        dy, dy_imp = dy - dy_imp, dy_imp * 0.0
        new = lr.stack().rename("lr_new")
        px = px.join(new, on=["date", "code"])
        px["log_return"] = px["lr_new"].where(px["log_return"].notna())
        px = px.drop(columns="lr_new")
        px.loc[imp, "dividend"] = 0.0
        px.loc[imp, "dividend_source"] = "imputed (removed)"

    # estimation returns: own returns, chained predecessor returns before the successor's first day
    r = lr.copy()
    rc = ch.pivot(index="date", columns="code", values="log_return").reindex(index=dates)
    for c in rc.columns:
        if c in r.columns:
            r[c] = r[c].fillna(rc[c])
    simple = np.expm1(r)

    member = w(wide("in_index")).eq(True)
    hit = w(lim.pivot(index="date", columns="code", values="hit"))
    no_buy = hit.isin(["close_up", "quote_up"])
    no_sell = hit.isin(["close_down", "quote_down"])

    oc = (np.log(wide("open")) - np.log(close)).abs()
    sigma_d = oc.rolling(10, min_periods=1).mean()

    month = dates.to_period("M")
    days = pd.Series(1, index=dates).groupby(month).sum()
    value = (close * wide("volume")).groupby(month).sum(min_count=1).div(days, axis=0)
    lq = liq.assign(month=pd.PeriodIndex(liq["month"], freq="M")).pivot(index="month", columns="code",
                                                                       values="adv_value")
    lq = lq.reindex(index=value.index, columns=codes)
    monthly_table = value.index >= pd.Period("2020-01", "M")   # the daily file has no volume from 2020
    value.loc[monthly_table] = lq.loc[monthly_table]

    rate = load_rate(dates)
    rf = accrual(rate, dates)
    mkt = simple.where(member).mean(axis=1)
    last_day = px.groupby("code")["date"].max().reindex(codes)
    return Data(end=end, dates=dates, r=r, simple=simple, gp=gp, dy=dy, dy_imp=dy_imp, member=member,
                traded=traded, no_buy=no_buy, no_sell=no_sell, sigma_d=sigma_d, adv=value,
                last_day=last_day, rate=rate, rf=rf, mkt=mkt, px=px, liq=liq, chain=ch)
