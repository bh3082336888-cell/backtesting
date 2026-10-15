"""Causal indicators. Every output at index t uses data <= t only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df.close.shift(1)
    return pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder(true_range(df), n)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df.high.diff()
    dn = -df.low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = wilder(true_range(df), n)
    pdi = 100 * wilder(plus_dm, n) / tr
    mdi = 100 * wilder(minus_dm, n) / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return wilder(dx, n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    g = wilder(d.clip(lower=0), n)
    l = wilder(-d.clip(upper=0), n)
    rs = g / l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def bollinger(close: pd.Series, n: int, k: float):
    mid = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid, mid + k * sd, mid - k * sd


def rolling_vwap(df: pd.DataFrame, n: int) -> pd.Series:
    tp = (df.high + df.low + df.close) / 3
    pv = (tp * df.volume).rolling(n, min_periods=n).sum()
    v = df.volume.rolling(n, min_periods=n).sum()
    return pv / v.replace(0, np.nan)


def donchian_prior(df: pd.DataFrame, n: int):
    """Highest high / lowest low of the n bars *before* t (excludes bar t)."""
    return (df.high.rolling(n, min_periods=n).max().shift(1),
            df.low.rolling(n, min_periods=n).min().shift(1))
