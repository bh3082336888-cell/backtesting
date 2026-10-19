"""Per-bar slippage from trailing volume: floor + quoted half-spread, which falls as a power of ADV."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# 1: flat costs.slippage_bps for every coin and date; 2: per coin and bar (costs.slippage_model)
COST_MODEL_VERSION = 2
DAY_MS = 86_400_000
ADV_REF_USD = 1e8


def trailing_adv(open_time_ms: np.ndarray, quote_volume: np.ndarray, days: int) -> np.ndarray:
    """Mean daily quote volume over the `days` UTC days before each bar's day; NaN if none has data."""
    day = np.asarray(open_time_ms, dtype="int64") // DAY_MS
    if len(day) == 0:
        return np.array([], dtype=float)
    daily = pd.Series(np.asarray(quote_volume, dtype=float)).groupby(day).sum(min_count=1)
    full = daily.reindex(np.arange(day.min(), day.max() + 1))   # gap days stay NaN, so are skipped
    adv = full.rolling(days, min_periods=1).mean().shift(1)
    return adv.reindex(day).to_numpy()


def slippage_bps(adv_usd: np.ndarray, floor_bps: float, model: dict) -> np.ndarray:
    """min(max_bps, floor + spread_at_100m * (adv / 100m) ** -alpha); max_bps where adv is unknown or <= 0."""
    cap = float(model["max_bps"])
    if cap < floor_bps:
        raise ValueError(f"slippage max_bps {cap} is below the floor {floor_bps}")
    adv = np.asarray(adv_usd, dtype=float)
    out = np.full(adv.shape, cap)
    ok = np.isfinite(adv) & (adv > 0)
    out[ok] = np.minimum(cap, floor_bps + model["spread_bps_at_100m"]
                         * (adv[ok] / ADV_REF_USD) ** -model["alpha"])
    return out


def half_spread_bps(bid: np.ndarray, ask: np.ndarray, t_ms: np.ndarray, end_ms: int) -> float:
    """Time-weighted mean of (ask - bid) / (ask + bid) in bp over [t_ms[0], end_ms); quotes sorted by time."""
    bid, ask, t = (np.asarray(a, dtype=float) for a in (bid, ask, t_ms))
    keep = (bid > 0) & (ask > bid)
    bid, ask, t = bid[keep], ask[keep], t[keep]
    if len(t) == 0:
        return float("nan")
    w = np.diff(np.append(t, end_ms))
    return float(np.average((ask - bid) / (ask + bid) * 1e4, weights=w))


def fit_spread_curve(adv_usd: np.ndarray, half_spread_bps: np.ndarray, pct: float = 90.0) -> dict:
    """OLS of log half-spread on log ADV; alpha is the slope, the intercept is lifted to the pct residual."""
    adv, hs = np.asarray(adv_usd, dtype=float), np.asarray(half_spread_bps, dtype=float)
    ok = np.isfinite(adv) & np.isfinite(hs) & (adv > 0) & (hs > 0)
    if ok.sum() < 10:
        raise ValueError(f"need at least 10 finite positive points, got {int(ok.sum())}")
    x, y = np.log(adv[ok] / ADV_REF_USD), np.log(hs[ok])
    b = np.polyfit(x, y, 1)[0]
    if b >= 0:
        raise ValueError("the spread does not fall as volume rises")
    alpha = round(-float(b), 2)
    e = y + alpha * x
    return {"alpha": alpha,
            "spread_bps_at_100m": math.ceil(math.exp(float(np.percentile(e, pct))) * 100) / 100,
            "n": int(ok.sum())}
