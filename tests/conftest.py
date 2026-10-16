import numpy as np
import pandas as pd
import pytest

from perp_lab.strategy import HOLD, Signal, Strategy
from perp_lab.validate.causality import synthetic_bars as make_bars  # noqa: F401  (tests import it from here)

H = 3_600_000


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
