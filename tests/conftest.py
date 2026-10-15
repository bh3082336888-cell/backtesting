import numpy as np
import pandas as pd
import pytest

from perp_lab.strategy import HOLD, Signal, Strategy

H = 3_600_000


def make_bars(n=3000, seed=0, start="2023-01-01", funding_every=8, tf_ms=H):
    """Synthetic 1h perp bars with regimes, trends and funding extremes."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 0.002, n // 200 + 1), 200)[:n]
    r = drift + rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[100.0, close[:-1]]
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    t0 = int(pd.Timestamp(start, tz="UTC").value // 10**6)
    ot = t0 + np.arange(n) * tf_ms
    fc = np.zeros(n)
    idx = np.arange(0, n, funding_every)
    fc[idx] = 0.0001 + rng.normal(0, 0.00005, len(idx))
    spikes = rng.choice(idx, size=max(len(idx) // 15, 1), replace=False)
    fc[spikes] = rng.choice([-1, 1], size=len(spikes)) * rng.uniform(0.0008, 0.002, len(spikes))
    df = pd.DataFrame({
        "open_time": ot, "open": open_, "high": hi, "low": lo, "close": close,
        "volume": rng.uniform(100, 1000, n), "quote_volume": np.nan, "trades": np.nan,
        "taker_buy_volume": np.nan, "taker_buy_quote_volume": np.nan, "close_time": ot + tf_ms - 1,
        "funding_charge": fc, "funding_last": pd.Series(np.where(fc != 0, fc, np.nan)).ffill().values,
        "funding_interval_h": 8.0, "oi": np.nan, "oi_value": np.nan,
        "regime": rng.choice(["trending", "ranging", "high_vol"], n), "regime_adx": 20.0,
    })
    df.index = pd.to_datetime(df.open_time, unit="ms", utc=True)
    return df


def flat_bars(n=50, price=100.0, tf_ms=H, start="2023-01-01"):
    t0 = int(pd.Timestamp(start, tz="UTC").value // 10**6)
    ot = t0 + np.arange(n) * tf_ms
    df = pd.DataFrame({"open_time": ot, "open": price, "high": price, "low": price, "close": price,
                       "volume": 1.0, "close_time": ot + tf_ms - 1, "funding_charge": 0.0,
                       "regime": "ranging"})
    df.index = pd.to_datetime(df.open_time, unit="ms", utc=True)
    return df


class Scripted(Strategy):
    """Test helper: emits signals from a {bar_index: Signal} schedule."""
    name = "scripted"
    params = {}

    def __init__(self, schedule):
        self.p = {}
        self.schedule = schedule

    def prepare(self, bars):
        return bars[[]]

    def on_bar(self, ctx):
        return self.schedule.get(ctx.i, HOLD)


@pytest.fixture
def bars():
    return make_bars()
