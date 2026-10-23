"""How alike two strategies are, judged on what they hold rather than when they enter.

Exact entry bars are brittle: a one-bar delay or a parameter tweak drops the match to near 0.
So each strategy is reduced to the (symbol, hour) cells it was in a position, on an hourly grid
shared by every timeframe, and compared as sets. Pure: no I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd

H_MS = 3_600_000
_NAMES = ["symbol", "hour"]


def positions(trades: pd.DataFrame) -> pd.Series:
    """(symbol, hour) -> side (int8). A trade holds every hour it touched; on a clash, the later trade wins."""
    if len(trades) == 0:
        idx = pd.MultiIndex.from_arrays([pd.Index([], dtype=object), pd.Index([], dtype="int64")], names=_NAMES)
        return pd.Series([], index=idx, dtype="int8")
    t = trades.sort_values(["entry_time", "exit_time"], kind="stable")
    first = t["entry_time"].to_numpy(dtype="int64") // H_MS
    last = (t["exit_time"].to_numpy(dtype="int64") - 1) // H_MS
    n = np.maximum(last - first + 1, 1)  # a zero-length trade still holds its entry hour
    total = int(n.sum())
    # hour = first[i] + (0..n[i]-1), built flat: a global counter minus each trade's start offset
    offsets = np.cumsum(n) - n
    hours = np.repeat(first, n) + (np.arange(total, dtype="int64") - np.repeat(offsets, n))
    df = pd.DataFrame(
        {
            "symbol": np.repeat(t["symbol"].to_numpy(dtype=object), n),
            "hour": hours,
            "side": np.repeat(t["side"].to_numpy(dtype="int8"), n),
        }
    )
    df = df.drop_duplicates(["symbol", "hour"], keep="last")  # rows are in trade order, so last = later trade
    idx = pd.MultiIndex.from_arrays([df["symbol"].to_numpy(dtype=object), df["hour"].to_numpy()], names=_NAMES)
    return pd.Series(df["side"].to_numpy(dtype="int8"), index=idx)


def _agreement(a: pd.Series, b: pd.Series, sign: int) -> float:
    union = a.index.union(b.index)
    if len(union) == 0:
        return 0.0
    shared = a.index.intersection(b.index)
    if len(shared) == 0:
        return 0.0
    same = int((a.loc[shared].to_numpy() * sign == b.loc[shared].to_numpy()).sum())
    return same / len(union)


def overlap(a: pd.Series, b: pd.Series) -> float:
    """Cells held on the same side by both, over cells held by either. 0.0 when neither holds anything."""
    return _agreement(a, b, 1)


def mirror_overlap(a: pd.Series, b: pd.Series) -> float:
    """Like overlap, with the sides opposite: a strategy that is another one flipped."""
    return _agreement(a, b, -1)


def return_corr(a: pd.Series, b: pd.Series, min_days: int = 30) -> float | None:
    """Pearson correlation of daily returns on shared days; None when too few days or no variance."""
    both = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(both) < min_days:
        return None
    x, y = both.iloc[:, 0].to_numpy(dtype=float), both.iloc[:, 1].to_numpy(dtype=float)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    return r if np.isfinite(r) else None


def nearest(
    pos: pd.Series,
    daily: pd.Series | None,
    others: dict[tuple[str, str], tuple[pd.Series, pd.Series | None]],
    k: int = 3,
) -> dict:
    """The k closest earlier runs by position overlap, mirror overlap and return correlation.

    `others` maps (strategy, timeframe) to (positions, daily returns or None). Zero or None values are dropped."""
    rows: dict[str, list[tuple[float, str, str, float]]] = {"position": [], "mirror": [], "correlation": []}
    for (name, tf), (opos, odaily) in others.items():
        o, m = overlap(pos, opos), mirror_overlap(pos, opos)
        rows["position"].append((o, name, tf, o))
        rows["mirror"].append((m, name, tf, m))
        c = return_corr(daily, odaily) if daily is not None and odaily is not None else None
        if c is not None:
            rows["correlation"].append((abs(c), name, tf, c))
    out = {}
    for key, items in rows.items():
        items = [r for r in items if round(r[3], 3) != 0]
        items.sort(key=lambda r: -r[0])
        out[key] = [{"strategy": n, "timeframe": tf, "value": round(v, 3)} for _, n, tf, v in items[:k]]
    return out
