import numpy as np
import pandas as pd
import pytest

from perp_lab.overlap import H_MS, mirror_overlap, nearest, overlap, positions, return_corr


def _ms(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").value // 1_000_000)


def hour(ts: str) -> int:
    return _ms(ts) // H_MS


def trades(rows):
    return pd.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "entry_time": np.array([_ms(r[1]) for r in rows], dtype="int64"),
            "exit_time": np.array([_ms(r[2]) for r in rows], dtype="int64"),
            "side": [r[3] for r in rows],
        }
    )


def daily(n, seed=0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.Series(np.random.default_rng(seed).normal(0, 0.01, n), index=idx)


def test_identical_trades_overlap_fully():
    p = positions(trades([("S", "2024-01-01 00:00", "2024-01-01 12:00", 1)]))
    assert overlap(p, p) == 1.0 and mirror_overlap(p, p) == 0.0


def test_a_one_hour_delay_keeps_most_of_the_overlap():
    a = positions(trades([("S", "2024-01-01 00:00", "2024-01-01 12:00", 1)]))
    b = positions(trades([("S", "2024-01-01 01:00", "2024-01-01 13:00", 1)]))
    assert overlap(a, b) == pytest.approx(11 / 13)


def test_a_flipped_copy_is_a_full_mirror():
    a = positions(trades([("S", "2024-01-01 00:00", "2024-01-01 06:00", 1)]))
    b = positions(trades([("S", "2024-01-01 00:00", "2024-01-01 06:00", -1)]))
    assert mirror_overlap(a, b) == 1.0 and overlap(a, b) == 0.0


def test_a_15m_trade_inside_one_hour_holds_that_hour():
    p = positions(trades([("S", "2024-01-01 10:15", "2024-01-01 10:45", 1)]))
    assert list(p.index) == [("S", hour("2024-01-01 10:00"))]


def test_timeframes_compare_on_the_same_hours():
    four_h = positions(trades([("S", "2024-01-01 08:00", "2024-01-01 12:00", 1)]))
    one_h = positions(
        trades([("S", f"2024-01-01 {h:02d}:00", f"2024-01-01 {h + 1:02d}:00", 1) for h in range(8, 12)])
    )
    assert overlap(four_h, one_h) == 1.0


def test_empty_positions_overlap_zero():  # Review Focus 3
    e = positions(pd.DataFrame(columns=["symbol", "entry_time", "exit_time", "side"]))
    assert overlap(e, e) == 0.0 and mirror_overlap(e, e) == 0.0
    assert list(e.index.names) == ["symbol", "hour"]


def test_later_trade_wins_when_both_sides_hold_one_hour():
    p = positions(
        trades(
            [("S", "2024-01-01 00:00", "2024-01-01 03:00", 1), ("S", "2024-01-01 02:00", "2024-01-01 04:00", -1)]
        )
    )
    assert p[("S", hour("2024-01-01 02:00"))] == -1 and p[("S", hour("2024-01-01 01:00"))] == 1
    assert p.dtype == np.int8 and len(p) == 4


def test_symbols_are_separate():
    a = positions(trades([("A", "2024-01-01 00:00", "2024-01-01 04:00", 1)]))
    b = positions(trades([("B", "2024-01-01 00:00", "2024-01-01 04:00", 1)]))
    assert overlap(a, b) == 0.0


def test_correlation_needs_30_shared_days():
    assert return_corr(daily(29), daily(29)) is None and return_corr(daily(60), daily(60)) == pytest.approx(1.0)


def test_correlation_of_a_constant_series_is_none():
    flat = daily(60) * 0
    assert return_corr(flat, daily(60)) is None


def test_nearest_ranks_and_names_runs():
    pos = positions(trades([("S", "2024-01-01 00:00", "2024-01-01 10:00", 1)]))
    near = positions(trades([("S", "2024-01-01 00:00", "2024-01-01 11:00", 1)]))  # 10/11 ~ 0.909
    some = positions(trades([("S", "2024-01-01 08:00", "2024-01-01 20:00", 1)]))  # 2/20 = 0.1
    none = positions(trades([("S", "2024-02-01 00:00", "2024-02-01 10:00", 1)]))
    flipped = positions(trades([("S", "2024-01-01 00:00", "2024-01-01 10:00", -1)]))
    me = daily(60, 1)
    others = {
        ("near", "1h"): (near, daily(60, 1)),
        ("some", "4h"): (some, -daily(60, 1)),
        ("none", "1h"): (none, None),
        ("flip", "1h"): (flipped, daily(60, 2)),
    }
    out = nearest(pos, me, others)
    assert [e["strategy"] for e in out["position"]] == ["near", "some"]
    assert out["position"][0] == {"strategy": "near", "timeframe": "1h", "value": 0.909}
    assert [e["strategy"] for e in out["mirror"]] == ["flip"]
    corr = out["correlation"]
    assert [e["strategy"] for e in corr[:2]] == ["near", "some"]
    assert corr[0]["value"] == 1.0 and corr[1]["value"] == -1.0
    assert all(e["strategy"] != "none" for e in corr)


def test_nearest_keeps_only_k():
    pos = positions(trades([("S", "2024-01-01 00:00", "2024-01-01 10:00", 1)]))
    others = {(f"s{i}", "1h"): (pos, None) for i in range(5)}
    assert len(nearest(pos, None, others, k=3)["position"]) == 3
