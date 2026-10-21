"""Deflated Sharpe maths, pure and dependency-free so the proposer can compute the bar
without importing the judge."""

from __future__ import annotations

import math
from statistics import NormalDist

EULER_GAMMA = 0.5772156649
_N = NormalDist()


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """SR0: the daily Sharpe the best of n_trials unskilled attempts reaches by luck."""
    if n_trials < 2 or var_trials <= 0:
        return 0.0
    return math.sqrt(var_trials) * ((1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials)
                                    + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * math.e)))


def dsr_from_moments(sr: float, t: int, skew: float, kurt: float, n_trials: int, var_trials: float) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)."""
    if t < 3:
        return 0.0
    den = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if den <= 0:
        return 0.0
    sr0 = expected_max_sharpe(n_trials, var_trials)
    return float(_N.cdf((sr - sr0) * math.sqrt(t - 1) / math.sqrt(den)))


def sharpe_bar(n_trials: int, t_days: int = 2191, dsr_min: float = 0.95) -> float:
    """Annualised OOS Sharpe at which a normal-returns track (skew 0, kurtosis 3) of t_days
    reaches dsr_min after n_trials, with the no-edge spread 1/(t_days-1) the judge uses."""
    var_trials = 1.0 / (t_days - 1)
    lo, hi = 0.0, 0.5
    for _ in range(100):
        mid = (lo + hi) / 2
        if dsr_from_moments(mid, t_days, 0.0, 3.0, n_trials, var_trials) < dsr_min:
            lo = mid
        else:
            hi = mid
    return hi * math.sqrt(365)
