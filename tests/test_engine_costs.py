"""Fill timing, fees, slippage, funding, stops/targets."""

import math

import numpy as np
import pandas as pd
import pytest

from perp_lab.engine.backtest import EngineParams, run_backtest
from perp_lab.strategy import EXIT, Signal
from tests.conftest import Scripted, flat_bars, market_config

EP = EngineParams(taker=0.0005, maker=0.0002, slippage=0.0002, equity0=10_000, leverage=3,
                  risk=0.01, mmr=0.004)


def ramp_bars(n=30):
    b = flat_bars(n)
    o = 100.0 + np.arange(n)
    b["open"], b["high"], b["low"], b["close"] = o, o + 0.5, o - 0.5, o + 0.2
    return b


def run(bars, sched, ep=EP):
    s = Scripted(sched)
    return run_backtest(s, bars, s.prepare(bars), symbol="X", tf="1h", ep=ep)


def test_signal_at_close_fills_next_open_with_fees_and_slippage():
    b = ramp_bars()
    r = run(b, {5: Signal("long", stop=90.0), 10: EXIT})
    t = r.trades[0]
    fill_in = 106.0 * 1.0002
    assert t["entry_time"] == b.open_time.iloc[6]
    assert t["entry_price"] == pytest.approx(fill_in)
    qty = 100.0 / (fill_in - 90.0)
    assert t["qty"] == pytest.approx(qty)
    fill_out = 111.0 * (1 - 0.0002)
    assert t["exit_time"] == b.open_time.iloc[11] and t["exit_price"] == pytest.approx(fill_out)
    fees = 0.0005 * qty * fill_in + 0.0005 * qty * fill_out
    assert t["fees"] == pytest.approx(fees)
    assert t["pnl"] == pytest.approx(qty * (fill_out - fill_in) - fees)
    assert r.equity[-1] == pytest.approx(10_000 + t["pnl"])
    assert t["exit_reason"] == "signal"


def test_no_fill_on_signal_bar_itself():
    b = ramp_bars()
    r = run(b, {5: Signal("long", stop=90.0)})
    assert r.trades[0]["entry_price"] != pytest.approx(b.close.iloc[5])


def test_funding_charged_on_position_held_into_settlement_open():
    b = flat_bars(40)
    b.loc[b.index[[8, 16, 24]], "funding_charge"] = 0.001
    # enter at open of 8 (settlement bar → not charged), exit at open 16 (charged), 24 untouched
    r = run(b, {7: Signal("long", stop=90.0), 15: EXIT})
    t = r.trades[0]
    assert t["funding"] == pytest.approx(t["qty"] * 100.0 * 0.001)
    # short receives positive funding
    r2 = run(b, {7: Signal("short", stop=110.0), 15: EXIT})
    assert r2.trades[0]["funding"] == pytest.approx(-r2.trades[0]["qty"] * 100.0 * 0.001)


def test_funding_reduces_isolated_margin_and_moves_liq():
    from perp_lab.engine.backtest import liq_price
    q, e, m = 10.0, 100.0, 333.0
    before = liq_price(1, e, q, m, 0.004, 0.0)
    after = liq_price(1, e, q, m - 5.0, 0.004, 0.0)
    assert after > before


def test_stop_gap_fills_at_open():
    b = flat_bars(30)
    b.loc[b.index[12], ["open", "high", "low", "close"]] = [94.0, 95.0, 93.0, 94.0]
    r = run(b, {5: Signal("long", stop=97.0)})
    t = r.trades[0]
    assert t["exit_reason"] == "stop" and t["exit_price"] == pytest.approx(94.0 * (1 - 0.0002))


def test_intrabar_stop_fills_at_stop_minus_slippage():
    b = flat_bars(30)
    b.loc[b.index[12], ["low"]] = 96.0
    r = run(b, {5: Signal("long", stop=97.0)})
    assert r.trades[0]["exit_price"] == pytest.approx(97.0 * (1 - 0.0002))


def test_same_bar_stop_and_target_assumes_stop_first():
    b = flat_bars(30)
    b.loc[b.index[12], ["high", "low"]] = [110.0, 90.0]
    r = run(b, {5: Signal("long", stop=97.0, target=105.0)})
    assert r.trades[0]["exit_reason"] == "stop"


def test_target_is_maker_and_needs_trade_through():
    b = flat_bars(30)
    b.loc[b.index[10], "high"] = 105.0      # touch only: no fill
    b.loc[b.index[12], "high"] = 105.5      # through: fill
    r = run(b, {5: Signal("long", stop=97.0, target=105.0)})
    t = r.trades[0]
    assert t["exit_reason"] == "target" and t["exit_time"] == b.open_time.iloc[12]
    fill_in = 100.0 * 1.0002
    assert t["fees"] == pytest.approx(0.0005 * t["qty"] * fill_in + 0.0002 * t["qty"] * 105.0)


def test_trailing_stop_only_ratchets():
    b = ramp_bars(30)
    sched = {5: Signal("long", stop=90.0), 8: Signal("hold", stop=100.0), 9: Signal("hold", stop=95.0)}
    b.loc[b.index[15], "low"] = 85.0
    r = run(b, sched)
    assert r.trades[0]["exit_reason"] == "stop"
    assert r.trades[0]["exit_price"] == pytest.approx(100.0 * (1 - 0.0002))


def test_risk_sizing_and_notional_cap():
    b = flat_bars(30)
    r = run(b, {5: Signal("long", stop=99.9)})
    t = r.trades[0]
    # 1% risk to a 0.1% stop would be ~10x notional → capped at equity*leverage
    assert t["notional"] == pytest.approx(10_000 * 3 * (1 - 0.0), rel=1e-3)


def test_reversal_exits_then_enters():
    b = ramp_bars()
    r = run(b, {5: Signal("long", stop=90.0), 10: Signal("short", stop=120.0)})
    assert r.trades[0]["exit_time"] == b.open_time.iloc[11]
    assert r.trades[1]["side"] == -1 and r.trades[1]["entry_time"] == b.open_time.iloc[11]


def test_end_of_window_flattens():
    b = flat_bars(20)
    r = run(b, {5: Signal("long", stop=90.0)})
    assert r.trades[-1]["exit_reason"] == "eow"


def levels(t):
    return t["level_time"], t["level_stop"], t["level_target"]


def test_level_path_records_each_change_from_the_next_open():
    b = flat_bars(30)
    T = b.open_time
    sched = {5: Signal("long", stop=90.0),
             8: Signal("hold", stop=95.0),                   # ratchets up: in force from the open of 9
             10: Signal("hold", stop=93.0),                  # lower: ignored, so not a change
             12: Signal("hold", target=110.0),               # a target: in force from the open of 13
             14: Signal("hold", stop=95.0, target=110.0),    # the levels already in force
             16: EXIT}
    t = run(b, sched).trades[0]
    assert t["exit_reason"] == "signal" and t["exit_time"] == T.iloc[17]
    lt, ls, lg = levels(t)
    assert lt == [T.iloc[6], T.iloc[9], T.iloc[13]] and all(type(x) is int for x in lt)
    assert lt[0] == t["entry_time"]
    assert ls == [90.0, 95.0, 95.0]
    np.testing.assert_array_equal(lg, [math.nan, math.nan, 110.0])


@pytest.mark.parametrize("target", [110.0, None])
def test_level_path_of_a_trade_whose_levels_never_move_is_its_entry(target):
    b = flat_bars(30)
    t = run(b, {5: Signal("long", stop=90.0, target=target), 9: Signal("hold", stop=90.0), 16: EXIT}).trades[0]
    lt, ls, lg = levels(t)
    assert (lt, ls) == ([t["entry_time"]], [t["stop_init"]])
    np.testing.assert_array_equal(lg, [t["target"]])
    assert len(lg) == 1


def test_level_path_of_a_short_mirrors_the_ratchet():
    b = flat_bars(30)
    T = b.open_time
    sched = {5: Signal("short", stop=110.0),
             8: Signal("hold", stop=105.0),                  # lower: ratchets for a short
             10: Signal("hold", stop=107.0),                 # higher: ignored
             12: Signal("hold", stop=103.0, target=90.0),
             14: Signal("hold", target=math.nan),            # the target withdrawn
             16: EXIT}
    t = run(b, sched).trades[0]
    assert t["side"] == -1 and t["exit_reason"] == "signal"
    lt, ls, lg = levels(t)
    assert lt == [T.iloc[6], T.iloc[9], T.iloc[13], T.iloc[15]]
    assert ls == [110.0, 105.0, 103.0, 103.0]
    np.testing.assert_array_equal(lg, [math.nan, math.nan, 90.0, math.nan])


def test_level_path_survives_a_run_file_round_trip(tmp_path):
    from perp_lab.runs import write_run
    cfg = market_config(tmp_path)
    b = flat_bars(40)
    r = run(b, {5: Signal("long", stop=90.0), 8: Signal("hold", stop=95.0), 10: Signal("hold", target=110.0),
                12: EXIT, 20: Signal("short", stop=110.0, target=90.0), 30: EXIT})
    assert len(r.trades) == 2
    d = write_run(cfg, {"run_id": "r1", "strategy": "scripted", "universe": ["X"]}, {"oos_trades": pd.DataFrame(r.trades)})
    back = pd.read_parquet(d / "oos_trades.parquet")
    assert len(back) == 2 and [len(t["level_time"]) for t in r.trades] == [3, 1]
    for want, (_, got) in zip(r.trades, back.iterrows()):
        for k in ("level_time", "level_stop", "level_target"):
            assert isinstance(got[k], np.ndarray)
            np.testing.assert_array_equal(got[k], want[k])


def test_level_records_of_the_same_trade_compare_equal_whichever_nan_withdrew_the_target():
    # The sandbox replay checks its trades against the reference run's with ==, and its NaN is not the
    # strategy's NaN object; a list compares NaN by identity.
    bars = flat_bars(30)

    def trades(nan):
        return run(bars, {5: Signal("long", stop=90.0, target=110.0), 8: Signal("hold", target=nan), 16: EXIT}).trades

    a, b, c = (trades(nan) for nan in (math.nan, float("nan"), np.float64("nan")))
    assert len(a[0]["level_target"]) == 2
    assert a == b == c


def test_fills_use_each_bars_own_slippage():
    b = ramp_bars()
    b["slippage"] = 0.0002
    b.loc[b.index[6], "slippage"] = 0.001
    b.loc[b.index[11], "slippage"] = 0.003
    t = run(b, {5: Signal("long", stop=90.0), 10: EXIT}).trades[0]
    assert t["entry_price"] == pytest.approx(106.0 * 1.001)
    assert t["exit_price"] == pytest.approx(111.0 * (1 - 0.003))


def test_a_stop_fills_with_its_bars_slippage():
    b = ramp_bars()
    b["slippage"] = 0.0002
    b.loc[b.index[8], ["low", "slippage"]] = [103.0, 0.002]
    t = run(b, {5: Signal("long", stop=104.0)}).trades[0]
    assert t["exit_reason"] == "stop" and t["exit_price"] == pytest.approx(104.0 * (1 - 0.002))


def _slip_bars(n=30, slip=0.001):
    b = ramp_bars(n)
    b["slippage"] = slip
    return b


def test_a_zero_slippage_scale_fills_at_the_price_before_slippage():
    b = _slip_bars()
    ep = EngineParams(taker=0.0005, maker=0.0002, slippage=0.0002, equity0=10_000, leverage=3,
                      risk=0.01, mmr=0.004, slippage_scale=0.0)
    t = run(b, {5: Signal("long", stop=90.0), 10: EXIT}, ep).trades[0]
    assert t["entry_price"] == b.open.iloc[6]
    assert t["exit_price"] == b.open.iloc[11]
    assert t["slippage"] == 0.0


def test_a_trade_records_its_slippage_cost():
    b = ramp_bars()
    b["slippage"] = 0.0002
    b.loc[b.index[10], "low"] = 80.0     # bar 10 trades through the stop at 90
    r = run(b, {5: Signal("long", stop=90.0)})
    t = r.trades[0]
    assert t["exit_reason"] == "stop"
    open_px = b.open.iloc[6]
    assert t["slippage"] == pytest.approx(t["qty"] * open_px * 0.0002 + t["qty"] * 90.0 * 0.0002)


def test_slippage_scale_one_changes_nothing():
    b = _slip_bars()
    sched = {5: Signal("long", stop=90.0), 10: EXIT}
    base = run(b, sched)
    one = run(b, sched, EngineParams(**{**EP.__dict__, "slippage_scale": 1.0}))
    assert one.trades == base.trades
    assert np.array_equal(one.equity, base.equity)
