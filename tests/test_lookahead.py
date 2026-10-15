"""Lookahead test — runs automatically on EVERY plugin in ./strategies.

For several cut points k: features computed on bars[:k] must equal features
computed on the full series for rows < k, and the engine must emit identical
signals for bars < k-1. Any use of future data breaks one of the two.
"""

import numpy as np
import pandas as pd
import pytest

from perp_lab.engine.backtest import EngineParams, run_backtest
from perp_lab.indicators import atr
from perp_lab.strategy import HOLD, RandomEntry, Strategy, enter, list_plugins, load_strategy_class
from tests.conftest import make_bars

PLUGINS = list_plugins()
CUTS = [400, 1100, 1777, 2500]


def assert_causal(strat_factory, bars):
    full = strat_factory()
    f_full = full.prepare(bars)
    assert f_full.index.equals(bars.index), "prepare() must return features on the bars index"
    r_full = run_backtest(full, bars, f_full, symbol="X", tf="1h", ep=EngineParams(), record_signals=True)
    for k in CUTS:
        cut = bars.iloc[:k]
        s = strat_factory()
        f_cut = s.prepare(cut)
        pd.testing.assert_frame_equal(f_full.iloc[:k], f_cut, check_exact=False, rtol=1e-9, atol=1e-12,
                                      obj=f"features at cut {k}")
        r_cut = run_backtest(s, cut, f_cut, symbol="X", tf="1h", ep=EngineParams(), record_signals=True)
        a = [x for x in r_full.signals if x[0] < k - 1]
        b = [x for x in r_cut.signals if x[0] < k - 1]
        assert a == b, f"signals diverge before cut {k}"


def test_plugins_discovered():
    assert {"donchian_atr", "bb_vwap_reversion", "funding_fade"} <= set(PLUGINS)


@pytest.mark.parametrize("name", PLUGINS)
def test_plugin_has_one_sentence_hypothesis(name):
    h = load_strategy_class(name).hypothesis()
    assert h and len(h) > 20, f"{name}: module docstring must start with a one-sentence hypothesis"


@pytest.mark.parametrize("name", PLUGINS)
def test_plugin_no_lookahead(name):
    cls = load_strategy_class(name)
    bars = make_bars(seed=1)
    assert_causal(lambda: cls(), bars)
    # also every grid corner
    for k, vals in cls.grid.items():
        for v in (vals[0], vals[-1]):
            assert_causal(lambda: cls(**{k: v}), bars)


@pytest.mark.parametrize("name", PLUGINS)
def test_plugin_trades_on_synthetic_data(name):
    """A plugin that never trades can't fail the lookahead test — make sure ours do."""
    cls = load_strategy_class(name)
    bars = make_bars(seed=1)
    s = cls()
    r = run_backtest(s, bars, s.prepare(bars), symbol="X", tf="1h", ep=EngineParams())
    assert len(r.trades) > 0


def test_random_entry_wrapper_is_causal():
    cls = load_strategy_class("donchian_atr")
    bars = make_bars(seed=2)
    assert_causal(lambda: RandomEntry(cls(), 0.01, seed=3), bars)


class Cheater(Strategy):
    """Peeks at the next close."""
    params = {}

    def prepare(self, bars):
        return bars[[]].assign(nxt=bars.close.shift(-1), atr=atr(bars, 14))

    def bracket(self, ctx, side):
        return ctx["close"] - side * 2 * ctx["atr"], None

    def entry(self, ctx):
        n = ctx["nxt"]
        if n != n:
            return HOLD
        return enter(1 if n > ctx["close"] else -1, *self.bracket(ctx, 1 if n > ctx["close"] else -1))


def test_lookahead_check_catches_a_cheater():
    with pytest.raises(AssertionError):
        assert_causal(lambda: Cheater(), make_bars(seed=4))
