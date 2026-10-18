import numpy as np
import pandas as pd

from perp_lab.engine.metrics import daily_returns

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def test_daily_returns_of_an_empty_curve_has_a_utc_index():
    r = daily_returns(np.array([]), np.array([], dtype="int64"))
    assert r.empty and isinstance(r.index, pd.DatetimeIndex) and str(r.index.tz) == "UTC"
    assert r[r.index >= T("2025-01-01")].empty            # what walkforward._window does
