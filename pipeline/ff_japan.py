"""Download the Fama-French Japan factors from Kenneth French's data library (free).

Used by D21 (the residual signals regress on the yen market plus SMB and HML) and D20 (3) (H3: the
alpha of B0 on the yen market plus SMB, HML, RMW, CMA and WML).

The library's Japan factors are monthly from 1990-07, in US dollars, in percent. SMB, HML, RMW, CMA
and WML are long-short, so their dollar and yen returns differ only by a cross term (long-short
return x the month's yen move), which is second order; the market factor in dollars is not used
(D15's yen equal-weighted member market replaces it).

Files (unchanged zips in MARKET_DIR/ff_japan/raw/):
    Japan_3_Factors_CSV.zip     Mkt-RF, SMB, HML, RF
    Japan_5_Factors_CSV.zip     Mkt-RF, SMB, HML, RMW, CMA, RF
    Japan_Mom_Factor_CSV.zip    WML
Output:
    MARKET_DIR/ff_japan/ff_japan_monthly.csv    month, smb3, hml3, mkt_rf_usd, rf_usd (3-factor
                                                file); smb5, hml5, rmw, cma (5-factor file); wml.
                                                Decimals, not percent.

Run:  python pipeline/ff_japan.py
"""
import io
import zipfile

import pandas as pd
import requests

from config import DATA_DIR, USER_AGENT

FF_DIR = DATA_DIR / "market" / "ff_japan"
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {"3": "Japan_3_Factors_CSV.zip", "5": "Japan_5_Factors_CSV.zip", "mom": "Japan_Mom_Factor_CSV.zip"}


def fetch(name):
    r = requests.get(BASE + name, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    (FF_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (FF_DIR / "raw" / name).write_bytes(r.content)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        return z.read(z.namelist()[0]).decode("latin-1")


def monthly_block(text):
    """The first table of the file: rows 'YYYYMM, v1, v2, ...' under a header line; it stops at the
    first blank line (the annual table follows)."""
    lines = text.splitlines()
    head = next(i for i, l in enumerate(lines) if l.strip().startswith(","))
    cols = [c.strip() for c in lines[head].split(",")[1:]]
    rows = []
    for l in lines[head + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) < 2 or not parts[0].isdigit() or len(parts[0]) != 6:
            break
        rows.append([parts[0]] + [float(p) for p in parts[1:]])
    df = pd.DataFrame(rows, columns=["month"] + cols)
    df["month"] = pd.to_datetime(df["month"], format="%Y%m").dt.strftime("%Y-%m")
    return df.set_index("month") / 100


def main():
    f3 = monthly_block(fetch(FILES["3"])).rename(columns={"Mkt-RF": "mkt_rf_usd", "SMB": "smb3",
                                                         "HML": "hml3", "RF": "rf_usd"})
    f5 = monthly_block(fetch(FILES["5"])).rename(columns={"SMB": "smb5", "HML": "hml5", "RMW": "rmw",
                                                         "CMA": "cma"})[["smb5", "hml5", "rmw", "cma"]]
    mom = monthly_block(fetch(FILES["mom"])).rename(columns={"WML": "wml"})[["wml"]]
    df = f3.join(f5, how="outer").join(mom, how="outer")
    df.to_csv(FF_DIR / "ff_japan_monthly.csv", float_format="%.6f")
    for c in df.columns:                       # coverage only: no test-period averages (D20 (2))
        s = df[c].dropna()
        print(f"{c:10s} {s.index[0]} to {s.index[-1]} ({len(s)} months)")


if __name__ == "__main__":
    main()
