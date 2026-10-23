from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from perp_lab.indicators import ny_offset_ms, ny_session_day, ny_weekday_minute

NY = ZoneInfo("America/New_York")
HOUR = 3_600_000


def _ms(*args) -> int:
    return int(datetime(*args, tzinfo=timezone.utc).timestamp() * 1000)


def test_offset_matches_zoneinfo_every_hour_2019_2028():
    t = np.arange(_ms(2019, 1, 1), _ms(2029, 1, 1), HOUR, dtype=np.int64)
    got = ny_offset_ms(t)
    want = np.array([int(datetime.fromtimestamp(x / 1000, NY).utcoffset().total_seconds() * 1000)
                     for x in t], dtype=np.int64)
    assert set(np.unique(got)) <= {-14_400_000, -18_000_000}
    assert (got == want).all()
    # +-1 minute around every switch
    switches = t[1:][want[1:] != want[:-1]]
    near = np.concatenate([switches + d * 60_000 for d in (-60, -1, 0, 1, 60)])
    want_near = np.array([int(datetime.fromtimestamp(x / 1000, NY).utcoffset().total_seconds() * 1000)
                          for x in near], dtype=np.int64)
    assert (ny_offset_ms(near) == want_near).all()


def test_weekday_minute_dst_mondays():
    t = np.array([_ms(2024, 3, 11, 13, 40), _ms(2024, 11, 4, 14, 40)], dtype=np.int64)
    wd, mi = ny_weekday_minute(t)
    assert wd.tolist() == [0, 0]
    assert mi.tolist() == [580, 580]


def test_session_day_boundary():
    t = np.array([_ms(2024, 1, 7, 22, 59, 59), _ms(2024, 1, 7, 23, 0)], dtype=np.int64)
    d = ny_session_day(t)
    assert d[1] - d[0] == 1
    assert d[1] == _ms(2024, 1, 8) // 86_400_000
