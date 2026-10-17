"""flow_shock_fade: each rule on small hand-built bars, the bracket, the time exit, warmup, NaN taker data."""

import numpy as np
import pandas as pd
import pytest

from perp_lab.engine.backtest import EngineParams, run_backtest
from perp_lab.strategy import EXIT, HOLD
from perp_lab.validate.causality import synthetic_bars, with_moving_columns
from tests.conftest import flat_bars
from tests.sandbox_fixtures import load_class

N = 400
SHOCK = 250   # well past the 168-row volume warmup


@pytest.fixture(scope="module")
def cls():
    return load_class("flow_shock_fade")


def make(shocks=None, n=N):
    """Quiet bars: volume 1, balanced flow, a 2.0 high-low range (ATR 2.0 from the first full window).
    shocks: {row: (taker_buy_volume, close_minus_open)} overrides on those rows."""
    b = flat_bars(n=n, price=100.0)
    b["open"] = 100.0
    b["high"] = 101.0
    b["low"] = 99.0
    b["close"] = 100.0
    b["volume"] = 1.0
    b["taker_buy_volume"] = 0.5
    for i, (tb, body) in (shocks or {}).items():
        b.iloc[i, b.columns.get_loc("taker_buy_volume")] = tb
        b.iloc[i, b.columns.get_loc("close")] = 100.0 + body
        b.iloc[i, b.columns.get_loc("high")] = max(101.0, 100.0 + body)
        b.iloc[i, b.columns.get_loc("low")] = min(99.0, 100.0 + body)
    return b


def signals(cls, bars, **p):
    s = cls(**p) if p else cls()
    r = run_backtest(s, bars, s.prepare(bars), symbol="X", tf="1h", ep=EngineParams(), record_signals=True)
    return r, [x for x in r.signals if x[1] in ("long", "short")]


def test_prepare_columns_formulas_and_warmup(cls):
    b = make({SHOCK: (0.0, -3.0)})
    f = cls().prepare(b)
    assert list(f.columns) == ["flow", "mv", "atr"] and f.index.equals(b.index)
    assert f.flow.iloc[:168].isna().all() and f.flow.iloc[168:].notna().all()
    assert f.mv.iloc[:14].isna().all() and f.mv.iloc[14:].notna().all()
    assert f.atr.iloc[:13].isna().all() and f.atr.iloc[13:].notna().all()
    assert f.flow.iloc[SHOCK] == pytest.approx(-1.0)          # (2*0 - 1) / median 1
    assert f.flow.iloc[SHOCK + 1] == pytest.approx(0.0)
    assert f.mv.iloc[SHOCK] == pytest.approx(-3.0 / f.atr.iloc[SHOCK - 1])


def test_flow_is_base_volume_and_uses_only_the_previous_median(cls):
    b = make({SHOCK: (0.0, -3.0)})
    b.iloc[SHOCK, b.columns.get_loc("volume")] = 50.0          # bar t's own volume must not enter the median
    f = cls().prepare(b)
    assert f.flow.iloc[SHOCK] == pytest.approx((0.0 - 50.0) / 1.0)


def test_long_fires_exactly_on_the_down_shock(cls):
    r, sig = signals(cls, make({SHOCK: (0.0, -3.0)}))
    assert [(i, a) for i, a, *_ in sig] == [(SHOCK, "long")]


def test_short_fires_exactly_on_the_up_shock(cls):
    r, sig = signals(cls, make({SHOCK: (1.0, 3.0)}))
    assert [(i, a) for i, a, *_ in sig] == [(SHOCK, "short")]


@pytest.mark.parametrize("tb,body", [
    (0.0, -0.5),    # flow enough, move too small
    (0.0, +3.0),    # flow down but price up: no long
    (1.0, -3.0),    # flow up but price down: no short
    (0.3, -3.0),    # flow -0.4, above the -0.5 line
    (0.7, +3.0),    # flow +0.4, below the +0.5 line
])
def test_no_signal_unless_both_flow_and_move_qualify(cls, tb, body):
    _, sig = signals(cls, make({SHOCK: (tb, body)}))
    assert sig == []


def test_comparisons_are_inclusive(cls):
    b = make({SHOCK: (0.25, 0.0)})            # flow exactly -0.5
    b.iloc[SHOCK, b.columns.get_loc("close")] = 100.0
    a = cls().prepare(b).atr.iloc[SHOCK - 1]
    b.iloc[SHOCK, b.columns.get_loc("close")] = 100.0 - a          # mv exactly -1.0
    b.iloc[SHOCK, b.columns.get_loc("low")] = 100.0 - a
    f = cls().prepare(b)
    assert f.flow.iloc[SHOCK] == -0.5 and f.mv.iloc[SHOCK] == -1.0
    _, sig = signals(cls, b)
    assert [(i, a_) for i, a_, *_ in sig] == [(SHOCK, "long")]


def test_flow_x_param_moves_the_threshold(cls):
    b = make({SHOCK: (0.0, -3.0)})            # flow -1.0
    assert len(signals(cls, b, flow_x=1.0)[1]) == 1
    b2 = make({SHOCK: (0.1, -3.0)})           # flow -0.8
    assert len(signals(cls, b2, flow_x=1.0)[1]) == 0 and len(signals(cls, b2)[1]) == 1


def test_nothing_fires_in_warmup(cls):
    for row in (5, 100, 167):
        _, sig = signals(cls, make({row: (0.0, -3.0)}))
        assert sig == [], row
    _, sig = signals(cls, make({168: (0.0, -3.0)}))
    assert [i for i, *_ in sig] == [168]


def test_nan_taker_volume_never_fires(cls):
    b = make({SHOCK: (0.0, -3.0)})
    b.iloc[SHOCK, b.columns.get_loc("taker_buy_volume")] = np.nan
    assert signals(cls, b)[1] == []
    b["taker_buy_volume"] = np.nan
    assert signals(cls, b)[1] == []


def test_zero_volume_median_gives_nan_flow_and_no_signal(cls):
    b = make({SHOCK: (0.0, -3.0)})
    b["volume"] = 0.0
    f = cls().prepare(b)
    assert f.flow.isna().all()
    assert signals(cls, b)[1] == []


@pytest.mark.parametrize("side", [1, -1])
def test_bracket_is_three_atr_stop_and_no_target(cls, side):
    s = cls()
    stop, target = s.bracket({"close": 100.0, "atr": 2.0}, side)
    assert stop == pytest.approx(100.0 - side * 6.0) and target is None


def test_signal_stop_is_three_atr_from_signal_close(cls):
    b = make({SHOCK: (0.0, -3.0)})
    _, sig = signals(cls, b)
    atr_t = cls().prepare(b).atr.iloc[SHOCK]
    (_, action, stop, target), = sig
    assert action == "long" and target is None
    assert stop == pytest.approx(97.0 - 3.0 * atr_t)


def test_manage_exits_at_hold_bars_and_never_moves_the_stop(cls):
    s = cls()

    class Pos:
        pass

    class C:
        position = Pos()

    c = C()
    for held, want in [(1, HOLD), (11, HOLD), (12, EXIT), (13, EXIT)]:
        c.position.bars_held = held
        assert s.manage(c) == want


@pytest.mark.parametrize("hold,bars_held", [(6, 7), (12, 13), (24, 25)])
def test_time_exit_timing_and_trade_record(cls, hold, bars_held):
    # Held for exactly hold_bars bars; the trade record counts the exit bar too, so hold + 1.
    # Each grid value is checked, so a hold hard-coded in manage fails here.
    b = make({SHOCK: (0.0, -3.0)})
    r, sig = signals(cls, b, hold_bars=hold)
    assert len(r.trades) == 1
    t = r.trades[0]
    assert t["exit_reason"] == "signal" and t["bars_held"] == bars_held
    open_ms = b.open_time.to_numpy()
    assert t["entry_time"] == open_ms[SHOCK + 1] and t["exit_time"] == open_ms[SHOCK + 1 + hold]


def test_opposite_trigger_while_in_position_is_ignored(cls):
    b = make({SHOCK: (0.0, -3.0), SHOCK + 4: (1.0, 3.0)})
    r, sig = signals(cls, b)
    assert [(i, a) for i, a, *_ in sig if a in ("long", "short")] == [(SHOCK, "long")]
    assert len(r.trades) == 1


def test_next_entry_possible_at_close_of_exit_fill_bar(cls):
    e = SHOCK + 1
    b = make({SHOCK: (0.0, -3.0), e + 12: (1.0, 3.0)})
    _, sig = signals(cls, b)
    assert [(i, a) for i, a, *_ in sig] == [(SHOCK, "long"), (e + 12, "short")]


def test_synthetic_bars_behaviour(cls):
    bars = with_moving_columns(synthetic_bars(seed=1))
    n = {h: len(signals_trades(cls, bars, hold_bars=h)) for h in (6, 12, 24)}
    assert all(v > 0 for v in n.values())
    assert signals_trades(cls, bars, flow_x=1.0) == []
    assert signals_trades(cls, synthetic_bars(seed=1)) == []   # raw bars: NaN taker columns


def signals_trades(cls, bars, **p):
    s = cls(**p) if p else cls()
    return run_backtest(s, bars, s.prepare(bars), symbol="X", tf="1h", ep=EngineParams()).trades


def test_literals_and_hypothesis(cls):
    assert cls.params == {"flow_x": 0.5, "move_k": 1.0, "hold_bars": 12, "stop_atr": 3.0, "atr_n": 14, "vol_n": 168}
    assert cls.grid == {"flow_x": [0.5, 1.0], "hold_bars": [6, 12, 24]}
    assert cls.timeframes == ["1h"]
    assert cls.hypothesis().startswith("An hour of extreme one-sided taker flow that moves price")
