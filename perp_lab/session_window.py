"""A plugin's declared trading session: the shape check and the bar mask. Trusted code only.

`session` is `{}` (none declared) or {"tz": IANA name, "days": sorted weekdays 0-6 (Mon=0),
"start": "HH:MM", "end": "HH:MM"}. A bar is in the session when the local weekday of its close is in
`days` and start <= local HH:MM of the close <= end (both ends inclusive). Uses `zoneinfo`, which a
sealed worker can't; plugins never import this.
"""

from __future__ import annotations

import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_KEYS = {"tz", "days", "start", "end"}
_HHMM = re.compile(r"([01]\d|2[0-3]):([0-5]\d)")


def _minutes(s, key: str) -> int:
    if not isinstance(s, str) or not _HHMM.fullmatch(s):
        raise ValueError(f"session {key} must be 'HH:MM', got {s!r}")
    return int(s[:2]) * 60 + int(s[3:])


def check_session(d: dict) -> None:
    """`{}` is allowed. Otherwise exactly tz, days, start, end, well-formed; ValueError if not."""
    if not isinstance(d, dict):
        raise ValueError("session must be a dict")
    if not d:
        return
    if set(d) != _KEYS:
        raise ValueError(f"session keys must be exactly {sorted(_KEYS)}, got {sorted(map(str, d))}")
    tz = d["tz"]
    try:
        if not isinstance(tz, str):
            raise ValueError
        ZoneInfo(tz)
    except Exception:
        raise ValueError(f"session tz {tz!r} is not an IANA time zone") from None
    days = d["days"]
    if (not isinstance(days, list) or not days
            or not all(type(x) is int and 0 <= x <= 6 for x in days) or days != sorted(set(days))):
        raise ValueError("session days must be a non-empty sorted list of distinct ints 0-6")
    if _minutes(d["start"], "start") > _minutes(d["end"], "end"):
        raise ValueError("session start must not be after end")


def in_session(close_ms: np.ndarray, session: dict) -> np.ndarray:
    """Bool mask over bar close times (UTC epoch ms): True where the bar closes inside `session`."""
    close_ms = np.asarray(close_ms, dtype=np.int64)
    if not session:
        return np.ones(len(close_ms), dtype=bool)
    check_session(session)
    local = pd.DatetimeIndex(pd.to_datetime(close_ms, unit="ms", utc=True)).tz_convert(session["tz"])
    minutes = np.asarray(local.hour * 60 + local.minute)
    return (np.isin(np.asarray(local.weekday), session["days"])
            & (minutes >= _minutes(session["start"], "start")) & (minutes <= _minutes(session["end"], "end")))
