"""Judge gates: pure functions, no I/O. Resampling is seeded, so a judgement is reproducible."""

from __future__ import annotations

import hashlib
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from perp_lab.engine.metrics import sharpe

EULER_GAMMA = 0.5772156649
_N = NormalDist()


def seed_from(run_id: str) -> int:
    return int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16)


def moments(r: pd.Series) -> tuple[float, int, float, float]:
    """Daily Sharpe (ddof=1), T, skew and Pearson kurtosis (population moments)."""
    x = r.dropna().to_numpy(dtype=float)
    t = len(x)
    if t < 3 or x.std(ddof=1) == 0:
        return 0.0, t, 0.0, 3.0
    d = x - x.mean()
    m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()
    return float(x.mean() / x.std(ddof=1)), t, float(m3 / m2 ** 1.5), float(m4 / m2 ** 2)


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


def deflated_sharpe(r: pd.Series, n_trials: int, var_trials: float | None = None) -> dict:
    """var_trials defaults to the no-edge sampling variance of a daily Sharpe over this track,
    1/(T-1): the spread N unskilled strategies would show by luck alone. A spread measured across
    past trials is inflated by any trial whose true Sharpe is far from zero (e.g. one ruined by
    costs), which would make the bar unreachable for every later strategy."""
    sr, t, skew, kurt = moments(r)
    if var_trials is None:
        var_trials = 1.0 / (t - 1) if t > 1 else 0.0
    flat = t < 3 or float(r.dropna().std(ddof=1)) == 0
    return {"dsr": 0.0 if flat else dsr_from_moments(sr, t, skew, kurt, n_trials, var_trials),
            "sr0": expected_max_sharpe(n_trials, var_trials), "sr": sr, "t": t, "skew": skew, "kurt": kurt}


def block_indices(n: int, length: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, length) moving-block bootstrap indices into a series of n points."""
    block = max(1, min(block, n))
    k = -(-length // block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, k))
    return (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, k * block)[:, :length]


def _row_sharpes(a: np.ndarray) -> np.ndarray:
    sd = a.std(axis=1, ddof=1)
    ok = sd > 0
    return np.where(ok, a.mean(axis=1) / np.where(ok, sd, 1.0), 0.0)


def beats_parent(child: pd.Series, parent: pd.Series, *, block: int, n_boot: int, seed: int) -> float:
    """Fraction of paired block-bootstrap resamples in which the child's Sharpe is strictly higher.

    Days are resampled jointly so the two series stay matched. Ties are not wins, so identical
    series score 0. Fewer than two blocks of overlapping days scores 0."""
    both = pd.concat({"c": child, "p": parent}, axis=1, join="inner").dropna()
    n = len(both)
    if n < 2 * block:
        return 0.0
    idx = block_indices(n, n, block, n_boot, np.random.default_rng(seed))
    c, p = both.c.to_numpy()[idx], both.p.to_numpy()[idx]
    return float((_row_sharpes(c) > _row_sharpes(p)).mean())


def holdout_consistent(holdout: pd.Series, n_trades: int, wf_oos: pd.Series, *, min_trades: int,
                       floor_pct: float, block: int, n_boot: int, seed: int) -> tuple[bool, list[str], dict]:
    """Pass iff enough trades, Sharpe > 0, and the holdout's cumulative return is at least the
    floor_pct percentile of same-length cumulative returns resampled from the walk-forward OOS record."""
    h = holdout.dropna()
    w = wf_oos.dropna().to_numpy(dtype=float)
    detail = {"days": len(h), "trades": int(n_trades), "sharpe": sharpe(h),
              "cum_return": float((1 + h).prod() - 1), "floor": None}
    reasons = []
    if n_trades < min_trades:
        reasons.append(f"not enough evidence: {n_trades} holdout trades < {min_trades}")
    if detail["sharpe"] <= 0:
        reasons.append(f"holdout Sharpe {detail['sharpe']:.2f} <= 0")
    if len(h) == 0 or len(w) < block:
        reasons.append("not enough data to compare with the walk-forward record")
    else:
        idx = block_indices(len(w), len(h), block, n_boot, np.random.default_rng(seed))
        detail["floor"] = float(np.percentile(np.prod(1 + w[idx], axis=1) - 1, floor_pct))
        if detail["cum_return"] < detail["floor"]:
            reasons.append(f"holdout return {detail['cum_return']:.1%} below the walk-forward "
                           f"p{floor_pct:g} for {len(h)} days ({detail['floor']:.1%})")
    return not reasons, reasons, detail
