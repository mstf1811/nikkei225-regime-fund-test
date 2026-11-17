"""Download Yahoo Finance corporate actions (splits, dividends) for the kept codes, plus Yahoo's
daily prices, which serve only as a check on the JPX data (D9).

Yahoo quirks (see tracker): dividend amounts and the "close" series are already adjusted for later
splits, and the split list can be incomplete. A code may also belong to a different company than the
index member (codes get reused), so first/last dates are kept for later steps to compare.

Outputs (in YAHOO_DIR):
    raw/<code>.json       untouched API response (re-used unless --refresh)
    splits.csv            code, date, numerator, denominator
    dividends.csv         code, ex_date, amount (split-adjusted, as Yahoo gives it)
    check_prices.parquet  code, date, open, high, low, close, adjclose, volume
    coverage.csv          code, first_date, last_date, rows (empty dates = Yahoo has nothing)

Run:  python pipeline/yahoo.py [--refresh]
"""
import json
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from config import MEMBERSHIP_DIR, USER_AGENT, YAHOO_DIR

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{code}.T"
PARAMS = {"period1": 0, "period2": 9999999999, "interval": "1d", "events": "div,split"}
PAUSE = 0.5  # seconds between requests


def fetch(code, refresh=False):
    path = YAHOO_DIR / "raw" / f"{code}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(4):
        try:
            r = requests.get(CHART_URL.format(code=code), params=PARAMS,
                             headers={"User-Agent": USER_AGENT}, timeout=30)
            data = r.json()
            # Yahoo answers "Not Found" (HTTP 404) for codes it has no data for: a valid answer.
            if r.ok or data.get("chart", {}).get("error"):
                break
        except (requests.RequestException, ValueError):
            pass
        time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"Yahoo request for {code} failed 4 times")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(PAUSE)
    return data


def parse(code, data):
    """Return (prices, splits, dividends) DataFrames for one code."""
    result = (data.get("chart") or {}).get("result") or []
    if not result or not result[0].get("timestamp"):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    r = result[0]
    offset = r["meta"].get("gmtoffset", 9 * 3600)  # Tokyo time
    to_date = lambda ts: datetime.fromtimestamp(ts + offset, tz=timezone.utc).date()

    quote = r["indicators"]["quote"][0]
    adjclose = (r["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    prices = pd.DataFrame({
        "code": code,
        "date": [to_date(t) for t in r["timestamp"]],
        "open": quote.get("open"), "high": quote.get("high"), "low": quote.get("low"),
        "close": quote.get("close"), "adjclose": adjclose, "volume": quote.get("volume"),
    })
    # Yahoo occasionally sends non-numbers such as "-Infinity"; treat them as missing
    for col in ["open", "high", "low", "close", "adjclose", "volume"]:
        prices[col] = pd.to_numeric(prices[col], errors="coerce").replace([float("inf"), float("-inf")], None)
    prices = prices.dropna(subset=["close"])

    events = r.get("events", {})
    splits = pd.DataFrame([
        {"code": code, "date": to_date(e["date"]),
         "numerator": e["numerator"], "denominator": e["denominator"]}
        for e in events.get("splits", {}).values()])
    dividends = pd.DataFrame([
        {"code": code, "ex_date": to_date(e["date"]), "amount": e["amount"]}
        for e in events.get("dividends", {}).values()])
    return prices, splits, dividends


def main(refresh=False):
    codes = pd.read_csv(MEMBERSHIP_DIR / "kept_codes.csv", dtype=str)["code"].tolist()
    prices, splits, dividends, coverage = [], [], [], []
    for n, code in enumerate(codes, 1):
        p, s, d = parse(code, fetch(code, refresh))
        prices.append(p), splits.append(s), dividends.append(d)
        coverage.append({"code": code,
                         "first_date": p["date"].min() if len(p) else None,
                         "last_date": p["date"].max() if len(p) else None,
                         "rows": len(p)})
        if n % 50 == 0:
            print(f"  {n}/{len(codes)}")
    sort = lambda frames, by: pd.concat(frames, ignore_index=True).sort_values(by)
    sort(prices, ["code", "date"]).to_parquet(YAHOO_DIR / "check_prices.parquet", index=False)
    sort(splits, ["code", "date"]).to_csv(YAHOO_DIR / "splits.csv", index=False)
    sort(dividends, ["code", "ex_date"]).to_csv(YAHOO_DIR / "dividends.csv", index=False)
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(YAHOO_DIR / "coverage.csv", index=False)
    has = coverage["rows"] > 0
    print(f"Yahoo has data for {has.sum()} of {len(codes)} codes; "
          f"{sum(map(len, splits))} splits, {sum(map(len, dividends))} dividends")


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
