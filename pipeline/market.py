"""Download the market-level series: the cash rate and the Nikkei 225 index.

- Cash rate (D4 baseline): Bank of Japan, uncollateralised overnight call rate, daily average
  (series FM01 / STRDCLUCON, from 1998-01-05), through the BOJ's public API (no login, launched
  2026-02-18). Values are in percent per annum; the BOJ leaves weekends and holidays empty.
- Cash rate before 1998 (added 2026-10-12, for the regime model's training windows): BOJ monthly
  averages (db FM02), the uncollateralised overnight call rate (STRACLUCON) from 1985-07 and,
  before it, the collateralised overnight call rate (STRACLCOON, from 1960-01).
- Nikkei 225 index: Yahoo `^N225`, daily from 1970. It is a price index (no dividends) and only a
  *candidate* regime input; the HMM's inputs are not decided.

Outputs (in MARKET_DIR):
    raw/boj_call_rate.json, raw/n225.json   untouched API responses
    call_rate_daily.csv                     date, call_rate (% p.a.), business days only
    call_rate_monthly.csv                   month, call_rate (% p.a., monthly average), source
    nikkei225_index.csv                     date, open, high, low, close, volume

Run:  python pipeline/market.py              (everything)
      python pipeline/market.py --monthly    (only the monthly call rates)
"""
import json
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

from config import DATA_DIR, USER_AGENT

MARKET_DIR = DATA_DIR / "market"
BOJ_URL = "https://www.stat-search.boj.or.jp/api/v1/getDataCode"
BOJ_PARAMS = {"format": "json", "lang": "en", "db": "FM01", "code": "STRDCLUCON"}
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EN225"
YAHOO_PARAMS = {"period1": 0, "period2": 9999999999, "interval": "1d"}


def get_json(url, params, name):
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT,
                                                  "Accept-Encoding": "gzip"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    (MARKET_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (MARKET_DIR / "raw" / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def call_rate():
    """Daily call rate. The API pages long answers through NEXTPOSITION."""
    dates, values, params = [], [], dict(BOJ_PARAMS)
    while True:
        data = get_json(BOJ_URL, params, "boj_call_rate")
        if data.get("STATUS") != 200:
            raise RuntimeError(f"BOJ API error: {data.get('MESSAGE')}")
        v = data["RESULTSET"][0]["VALUES"]
        dates += v["SURVEY_DATES"]
        values += v["VALUES"]
        if not data.get("NEXTPOSITION"):
            break
        params["startPosition"] = data["NEXTPOSITION"]
    df = pd.DataFrame({"date": pd.to_datetime([str(d) for d in dates]), "call_rate": values})
    return df.dropna().drop_duplicates("date").sort_values("date")


def call_rate_monthly():
    """Monthly averages: uncollateralised from 1985-07, collateralised before it."""
    out = []
    for code, source in (("STRACLUCON", "uncollateralised"), ("STRACLCOON", "collateralised")):
        data = get_json(BOJ_URL, {"format": "json", "lang": "en", "db": "FM02", "code": code},
                        f"boj_call_rate_monthly_{code}")
        if data.get("STATUS") != 200:
            raise RuntimeError(f"BOJ API error: {data.get('MESSAGE')}")
        v = data["RESULTSET"][0]["VALUES"]
        out.append(pd.DataFrame({"month": [str(d) for d in v["SURVEY_DATES"]],
                                 "call_rate": v["VALUES"], "source": source}).dropna())
    unc, col = out
    df = pd.concat([col[col.month < unc.month.min()], unc])
    df["month"] = pd.to_datetime(df["month"], format="%Y%m").dt.strftime("%Y-%m")
    return df.sort_values("month")


def nikkei225():
    r = get_json(YAHOO_URL, YAHOO_PARAMS, "n225")["chart"]["result"][0]
    offset = r["meta"].get("gmtoffset", 9 * 3600)
    q = r["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": [datetime.fromtimestamp(t + offset, tz=timezone.utc).date() for t in r["timestamp"]],
        "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"], "volume": q["volume"],
    })
    return df.dropna(subset=["close"]).drop_duplicates("date", keep="last").sort_values("date")


def main():
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    cm = call_rate_monthly()
    cm.to_csv(MARKET_DIR / "call_rate_monthly.csv", index=False)
    print(f"monthly call rate: {cm.month.iloc[0]} to {cm.month.iloc[-1]}, "
          + ", ".join(f"{k} from {g.month.iloc[0]}" for k, g in cm.groupby("source", sort=False)))
    if "--monthly" in sys.argv:
        return
    cr = call_rate()
    cr.to_csv(MARKET_DIR / "call_rate_daily.csv", index=False, date_format="%Y-%m-%d")
    print(f"call rate: {len(cr)} days, {cr.date.min():%Y-%m-%d} to {cr.date.max():%Y-%m-%d}, "
          f"min {cr.call_rate.min()}%, max {cr.call_rate.max()}%")
    nk = nikkei225()
    nk.to_csv(MARKET_DIR / "nikkei225_index.csv", index=False)
    print(f"^N225: {len(nk)} days, {nk.date.min()} to {nk.date.max()}")


if __name__ == "__main__":
    main()
