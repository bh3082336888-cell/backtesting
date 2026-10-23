from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from perp_lab.session_window import check_session, in_session

LCE = {"tz": "America/New_York", "days": [0, 1, 2, 3, 4], "start": "09:40", "end": "15:45"}
NY = ZoneInfo("America/New_York")


def _ms(y, mo, d, h, mi):
    return int(datetime(y, mo, d, h, mi, tzinfo=NY).timestamp() * 1000)


@pytest.mark.parametrize("month", [1, 7])
def test_in_session_edges(month):
    # a Monday in January 2024 is the 8th; in July 2024 the 8th too
    closes = np.array([_ms(2024, month, 8, h, m) for h, m in [(9, 39), (9, 40), (15, 45), (15, 46)]], dtype=np.int64)
    assert in_session(closes, LCE).tolist() == [False, True, True, False]
    sat = np.array([_ms(2024, month, 13, 10, 0)], dtype=np.int64)
    assert in_session(sat, LCE).tolist() == [False]


def test_empty_session_is_all_true():
    out = in_session(np.array([0, 10**12, 2 * 10**12], dtype=np.int64), {})
    assert out.dtype == bool and out.tolist() == [True] * 3


@pytest.mark.parametrize("bad", [
    {**LCE, "extra": 1},
    {**LCE, "days": [7]},
    {**LCE, "days": []},
    {**LCE, "days": [1, 0]},
    {**LCE, "start": "9:40"},
    {**LCE, "start": "15:46"},
    {**LCE, "tz": "Not/AZone"},
    {"tz": "UTC", "days": [0], "start": "09:00"},
])
def test_check_session_refuses(bad):
    with pytest.raises(ValueError):
        check_session(bad)


def test_check_session_accepts():
    check_session({})
    check_session(LCE)
