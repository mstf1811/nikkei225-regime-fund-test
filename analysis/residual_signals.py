"""The two D21 signals (signal list v2): residual momentum and residual reversal.

Both rank stocks on returns net of their Fama-French three-factor exposures, so the signal is not a
hidden bet on the market, size or value (the papers' reason: total-return momentum and reversal carry
large time-varying factor exposures).

    resmom  Blitz, Huij & Martens (2011, JEF 18(3)), section 3, read in full:
            each month-end m, regress the stock's monthly excess returns on the three factors over
            the 36 months m-35 .. m (with an intercept; complete 36-month history required); the
            residual is e_k = r_k - b'f_k, WITHOUT subtracting the intercept (their footnote: the
            intercept is mostly fitted on months outside the formation period and would bring in
            long-term reversal); the signal is the sum of e over months m-11 .. m-1 (12-1M) divided by
            the standard deviation of those 11 residuals.
    resrev  Blitz, Huij, Lansdorp & Verbeek (2013, JFM 16(3)); full text not read, construction from
            summaries [S]: the same rolling 36-month three-factor regression; the signal is month m's
            residual divided by the residuals' standard deviation over the 36 months. Intercept
            treated as in resmom [ours].

Factors [ours, translation]: the market is D15's yen equal-weighted member return minus the call
rate; SMB and HML are Kenneth French's Japan factors (3-factor file, USD; long-short, so the currency
enters only as a cross term). French publishes a month about three weeks after it ends; the factor
returns are realised by the month-end, so a live fund would compute them itself.

Chained predecessor returns (D15 (6)) count toward the 36-month history. The first month of data
(2000-01) lacks its first day's return and is dropped, so the first signals are for 2003-01.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR  # noqa: E402
from signal_ic import SIGNALS  # noqa: E402

SIGNALS_V2 = SIGNALS + ["resmom", "resrev"]      # D21
WINDOW, FORM = 36, 11


def load_ff():
    ff = pd.read_csv(DATA_DIR / "market" / "ff_japan" / "ff_japan_monthly.csv")
    return ff.set_index(pd.PeriodIndex(ff["month"], freq="M"))


def residual_signals(simple, rf, mkt):
    """simple: daily simple returns (date x code, chained histories included); rf, mkt: daily cash
    and EW member returns. Returns month x code panels resmom, resrev and the factor loadings."""
    month = simple.index.to_period("M")
    gross = (1 + simple).groupby(month).prod(min_count=1) - 1
    rf_m = (1 + rf).groupby(month).prod() - 1
    mkt_m = (1 + mkt).groupby(month).prod() - 1
    y = gross.sub(rf_m, axis=0).iloc[1:]
    ff = load_ff()
    f = pd.DataFrame({"mkt": mkt_m - rf_m, "smb": ff["smb3"], "hml": ff["hml3"]}).reindex(y.index)
    names = ["resmom", "resrev", "b_mkt", "b_smb", "b_hml"]
    out = {k: pd.DataFrame(np.nan, index=y.index, columns=y.columns) for k in names}
    for i in range(WINDOW - 1, len(y)):
        Y, F = y.iloc[i - WINDOW + 1: i + 1], f.iloc[i - WINDOW + 1: i + 1]
        ok = Y.notna().all()
        if F.isna().any().any() or not ok.any():
            continue
        Yv, Fv = Y.loc[:, ok].to_numpy(), F.to_numpy()
        B = np.linalg.lstsq(np.column_stack([np.ones(WINDOW), Fv]), Yv, rcond=None)[0]
        e = Yv - Fv @ B[1:]                       # residual with the intercept kept
        form = e[-1 - FORM:-1]                    # months m-11 .. m-1
        m, cols = y.index[i], Y.columns[ok]
        out["resmom"].loc[m, cols] = form.sum(axis=0) / form.std(axis=0, ddof=1)
        out["resrev"].loc[m, cols] = e[-1] / e.std(axis=0, ddof=1)
        for j, k in enumerate(["b_mkt", "b_smb", "b_hml"], start=1):
            out[k].loc[m, cols] = B[j]
    return out
