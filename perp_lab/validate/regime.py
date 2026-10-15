"""Regime labels: trending / ranging / high_vol / transition.

Computed on native 4h bars for every timeframe so labels are consistent across
TFs and pairs (and Phase 3's paper regime-change flag uses the same buckets).
A 4h label becomes visible only after that 4h bar closes.

    high_vol   : realized-vol trailing percentile rank > vol_high_pct  (takes precedence)
    trending   : ADX > adx_trend
    ranging    : ADX < adx_range
    transition : in between
    unknown    : warmup
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from perp_lab.indicators import adx

REGIMES = ["trending", "ranging", "high_vol", "transition", "unknown"]


def regime_frame(df4h: pd.DataFrame, rcfg: dict) -> pd.DataFrame:
    a = adx(df4h, rcfg["adx_n"])
    lr = np.log(df4h.close).diff()
    vol = lr.rolling(rcfg["vol_n"], min_periods=rcfg["vol_n"]).std()
    w = rcfg["vol_rank_window"]
    # Trailing percentile rank of current vol within the last w values (causal).
    vrank = vol.rolling(w, min_periods=w // 6).rank(pct=True)
    lab = np.full(len(df4h), "transition", dtype=object)
    lab[(a > rcfg["adx_trend"]).values] = "trending"
    lab[(a < rcfg["adx_range"]).values] = "ranging"
    lab[(vrank > rcfg["vol_high_pct"]).values] = "high_vol"
    lab[(a.isna() | vrank.isna()).values] = "unknown"
    return pd.DataFrame({"close_time": df4h.close_time.values, "adx": a.values,
                         "vol_rank": vrank.values, "regime": lab}, index=df4h.index)


def attach_regime(cfg, bars: pd.DataFrame, symbol: str, start=None, end=None, holdout=None) -> pd.DataFrame:
    from perp_lab.data.store import load_candles

    rcfg = cfg["regime"]
    df4 = load_candles(cfg, symbol, rcfg["timeframe"], start, end, holdout)
    rf = regime_frame(df4, rcfg)
    known = rf.close_time.values
    j = np.searchsorted(known, bars.close_time.values, side="right") - 1
    jj = np.clip(j, 0, max(len(rf) - 1, 0))
    out = bars.copy()
    out["regime"] = np.where(j >= 0, rf.regime.values[jj], "unknown") if len(rf) else "unknown"
    out["regime_adx"] = np.where(j >= 0, rf.adx.values[jj], np.nan) if len(rf) else np.nan
    return out
