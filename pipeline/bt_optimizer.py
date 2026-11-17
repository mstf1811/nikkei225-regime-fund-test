"""The optimizer (E2, D18): Boyd et al. (2017) single-period optimisation in monthly units,

    maximise  mu'w - g_trade*phi_trade(w - w_prev) - g_hold*phi_hold(w) - g_risk*w'Sigma*w
    phi_trade = a|z| + b*sigma*|z|^1.5 / (V/v)^0.5        phi_hold = s*sum((w)_-) / 12

with the D13/D18 constraints. Ported from analysis/optimizer_check.py (solve), with two additions
for the daily simulation [ours]:
  - names the fund cannot trade today (no trade, limit-up/limit-down close, D11) get z bounds of 0
    on the blocked side (Boyd et al. 4.4's no-buy / no-sell restrictions);
  - per-name bounds are intersected with what the day's trade cap can reach, so a position that
    drifted past a limit is moved toward it as fast as the cap allows instead of making the
    problem infeasible; the long and short boxes are widened to what the forced bounds imply when
    drift made them unreachable today. The box limits are net of positions held outside the
    optimizer (index leavers being sold, `fixed_long` / `fixed_short`).
Solver: Clarabel; SCS at 1e-7 when Clarabel does not report optimal; no trade if both fail (D18 (9)).
"""
from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass
class Params:
    g_risk: float = 10.0
    g_trade: float = 1.0
    g_hold: float = 1.0
    a: float = 0.0015            # D13: one-way linear cost
    b: float = 1.0               # D18 (3): square-root impact coefficient
    borrow: float = 0.0115       # D13: per year on the short value
    proceeds_earn: bool = False  # D13: short proceeds earn 0% (the holding cost adds the cash rate)
    long_max: float = 1.0        # D13
    short_max: float = 1.0
    w_max: float = 0.10          # JITA 10%, longs only (D13)
    w_min: float = -0.05         # D18 (5)
    participation: float = 0.10  # D18 (5)
    liq_days: int = 3            # JITA high-liquidity tier (D18 (5))
    fund_rules: bool = True      # False = C4: only the long/short box


def trade_cost(z, sigma_d, liq, p):
    """One-off cost of trades z (fractions of NAV): linear + 3/2-power impact (Boyd et al. eq. 2.3)."""
    z = np.abs(z)
    return p.a * z + p.b * sigma_d / np.sqrt(liq) * z ** 1.5


def bounds(w_prev, liq, can_buy, can_sell, p):
    """Per-name bounds on the post-trade weights (see the module note)."""
    n = len(w_prev)
    if p.fund_rules:
        cap = p.liq_days * p.participation * liq
        lo, hi = np.maximum(p.w_min, -cap), np.minimum(p.w_max, cap)
        tcap = p.participation * liq
    else:
        lo, hi, tcap = np.full(n, -p.short_max), np.full(n, p.long_max), np.full(n, np.inf)
    zlo, zhi = np.where(can_sell, -tcap, 0.0), np.where(can_buy, tcap, 0.0)
    L = np.clip(lo, w_prev + zlo, w_prev + zhi)
    U = np.clip(hi, w_prev + zlo, w_prev + zhi)
    return L, U


def solve(mu, Sigma, w_prev, sigma_d, liq, rf_y, p, can_buy, can_sell, fixed_long=0.0, fixed_short=0.0):
    """Target weights for the names in mu (monthly excess returns); returns (w, status)."""
    n = len(mu)
    L_, U_ = bounds(w_prev, liq, can_buy, can_sell, p)
    long_cap = max(p.long_max - fixed_long, np.maximum(L_, 0).sum())
    short_cap = max(p.short_max - fixed_short, np.maximum(-U_, 0).sum())
    w = cp.Variable(n)
    z = w - w_prev
    chol = np.linalg.cholesky(Sigma + 1e-12 * np.eye(n))
    trade = p.a * cp.sum(cp.abs(z)) + cp.sum(cp.multiply(p.b * sigma_d / np.sqrt(liq), cp.power(cp.abs(z), 1.5)))
    s = (p.borrow + (0.0 if p.proceeds_earn else rf_y)) / 12
    hold = s * cp.sum(cp.neg(w))
    risk = cp.sum_squares(chol.T @ w)
    obj = cp.Maximize(1e4 * (mu @ w - p.g_trade * trade - p.g_hold * hold - p.g_risk * risk))
    cons = [cp.sum(cp.pos(w)) <= long_cap, cp.sum(cp.neg(w)) <= short_cap, w >= L_, w <= U_]
    prob = cp.Problem(obj, cons)
    status = "failed"
    try:
        prob.solve(solver=cp.CLARABEL)
        status = prob.status
    except cp.SolverError:
        pass
    if status != cp.OPTIMAL:
        try:
            prob.solve(solver=cp.SCS, eps=1e-7, max_iters=200000)
            status = "scs:" + prob.status
        except cp.SolverError:
            status = "failed"
    if w.value is None or status not in (cp.OPTIMAL, "scs:" + cp.OPTIMAL, "scs:" + cp.OPTIMAL_INACCURATE):
        return w_prev.copy(), status
    wv = np.clip(np.where(np.abs(w.value) < 1e-6, 0.0, w.value), L_, U_)
    # SCS meets the boxes only to its tolerance: scale the overshooting side back onto them exactly
    lp, sp = wv[wv > 0].sum(), -wv[wv < 0].sum()
    if lp > long_cap:
        wv[wv > 0] *= long_cap / lp
    if sp > short_cap:
        wv[wv < 0] *= short_cap / sp
    return wv, status
