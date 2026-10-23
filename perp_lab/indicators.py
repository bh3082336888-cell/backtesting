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


_DAY_MS = 86_400_000


def _first_sunday_on_or_after(day: np.ndarray) -> np.ndarray:
    """Epoch-day of the first Sunday on or after `day` (1970-01-01 was a Thursday)."""
    return day + (3 - day) % 7


def ny_offset_ms(t_ms: np.ndarray) -> np.ndarray:
    """New York UTC offset in ms (-4h in US daylight time, else -5h); no tz database."""
    t = np.asarray(t_ms, dtype=np.int64)
    year = t.astype("datetime64[ms]").astype("datetime64[Y]")
    mar1 = year.astype("datetime64[M]") + np.timedelta64(2, "M")
    nov1 = year.astype("datetime64[M]") + np.timedelta64(10, "M")
    mar1 = mar1.astype("datetime64[D]").astype(np.int64)
    nov1 = nov1.astype("datetime64[D]").astype(np.int64)
    start = (_first_sunday_on_or_after(mar1) + 7) * _DAY_MS + 7 * 3_600_000   # 2nd Sun Mar 07:00 UTC
    end = _first_sunday_on_or_after(nov1) * _DAY_MS + 6 * 3_600_000           # 1st Sun Nov 06:00 UTC
    return np.where((t >= start) & (t < end), -14_400_000, -18_000_000).astype(np.int64)


def ny_weekday_minute(t_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """New York local weekday (Mon=0) and minute of day."""
    t = np.asarray(t_ms, dtype=np.int64)
    local = t + ny_offset_ms(t)
    day = local // _DAY_MS
    return (day + 3) % 7, (local - day * _DAY_MS) // 60_000


def ny_session_day(t_ms: np.ndarray) -> np.ndarray:
    """Trading date (epoch days) of a 18:00->18:00 New York session; 18:00 starts the next date."""
    t = np.asarray(t_ms, dtype=np.int64)
    return (t + ny_offset_ms(t) + 6 * 3_600_000) // _DAY_MS
