import pytest

from perp_lab.engine.backtest import EngineParams, liq_price, run_backtest
from perp_lab.strategy import Signal
from tests.conftest import Scripted, flat_bars


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("lev,mmr,cum", [(3, 0.004, 0.0), (20, 0.01, 0.0), (50, 0.005, 25.0)])
def test_liq_price_is_where_equity_equals_maintenance(side, lev, mmr, cum):
    e, q = 100.0, 7.0
    m = q * e / lev
    p = liq_price(side, e, q, m, mmr, cum)
    equity = m + side * q * (p - e)
    maint = q * p * mmr - cum
    assert equity == pytest.approx(maint, rel=1e-9, abs=1e-9)


def test_liq_side_and_distance():
    assert liq_price(1, 100, 1, 100 / 10, 0.004, 0) == pytest.approx(90.0 / 0.996)
    assert liq_price(-1, 100, 1, 100 / 10, 0.004, 0) == pytest.approx(110.0 / 1.004)


def _run(bars, sched, ep):
    s = Scripted(sched)
    return run_backtest(s, bars, s.prepare(bars), symbol="X", tf="1h", ep=ep)


def test_entry_rejected_when_liq_inside_stop():
    ep = EngineParams(leverage=50, risk=0.5, mmr=0.004)
    b = flat_bars(30)
    r = _run(b, {5: Signal("long", stop=95.0)}, ep)   # liq ≈ 98.4 > stop 95
    assert not r.trades and r.rejected["liq_inside_stop"] == 1


def test_gap_through_liq_loses_isolated_margin():
    ep = EngineParams(leverage=50, risk=0.01, mmr=0.004, slippage=0.0)
    b = flat_bars(30)
    b.loc[b.index[12], ["open", "high", "low", "close"]] = [90.0, 90.0, 89.0, 90.0]
    r = _run(b, {5: Signal("long", stop=99.0)}, ep)
    t = r.trades[0]
    assert t["exit_reason"] == "liq"
    assert t["pnl_gross"] == pytest.approx(-t["margin"])


def test_funding_can_push_liq_through_stop():
    """Margin eaten by funding → liq hit before the stop."""
    ep = EngineParams(leverage=50, risk=0.01, mmr=0.004, slippage=0.0)
    b = flat_bars(60)
    b.loc[b.index[8:40:2], "funding_charge"] = 0.002
    b.loc[b.index[45], "low"] = 98.0
    r = _run(b, {5: Signal("long", stop=98.5)}, ep)
    assert r.trades[0]["exit_reason"] == "liq"
