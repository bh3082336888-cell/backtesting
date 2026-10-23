"""One fixed synthetic market that plugins at every timeframe trade, for `lab preflight`'s overlap.

Each symbol is a single 5m path from `synthetic_bars` (fixed seeds and start date, never the
day-pinned holdout span, so the market is the same every day), resampled to each coarser timeframe.
A 1h plugin and its 4h twin then see the same prices, and their positions on the hourly grid can be
compared. Positions measured here are cached by MARKET_VERSION: bump it whenever the bars change.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from perp_lab.config import TF_MS
from perp_lab.validate.causality import synthetic_bars, with_moving_columns

MARKET_VERSION = 2
SYMBOLS = ("SYN1", "SYN2", "SYN3", "SYN4")
DAYS = 180
BASE_MS = TF_MS["5m"]
_SUMMED = ("volume", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "funding_charge")


class NoMarket(ValueError):
    """A timeframe the 5m path can't be resampled to."""


def _path(i: int) -> pd.DataFrame:
    bars = synthetic_bars(n=DAYS * 86_400_000 // BASE_MS, seed=100 + i, start="2023-01-01", funding_every=96,
                          tf_ms=BASE_MS)                              # funding every 8 hours
    return with_moving_columns(bars, seed=200 + i)


def _resample(bars: pd.DataFrame, ms: int) -> pd.DataFrame:
    """OHLC as candles combine; volumes, trade counts, taker flow and funding charged add up; the
    bar's own open time; every other column (funding rate and interval, open interest, regime, the
    close time) is its state at the bar's close."""
    how = {c: "last" for c in bars.columns}
    how.update({c: "sum" for c in _SUMMED})
    how.update(open_time="first", open="first", high="max", low="min", close="last")
    out = bars.resample(pd.Timedelta(milliseconds=ms), label="left", closed="left").agg(how)
    return out[bars.columns]


@lru_cache(maxsize=None)
def market(tf: str) -> dict[str, pd.DataFrame]:
    """{symbol: bars at tf}. Built once per timeframe and shared: callers must not change the frames."""
    ms = TF_MS.get(tf)
    if ms is None or ms < BASE_MS or ms % BASE_MS:
        raise NoMarket(f"no synthetic market at {tf}: its bars are 5m, resampled to multiples of 5m")
    out = {}
    for i, s in enumerate(SYMBOLS):
        bars = _path(i)
        out[s] = bars if ms == BASE_MS else _resample(bars, ms)
    return out
