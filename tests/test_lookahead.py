"""Lookahead test — runs automatically on EVERY plugin in ./strategies.

For several cut points k: features computed on bars[:k] must equal features
computed on the full series for rows < k, and the engine must emit identical
signals for bars < k-1. Any use of future data breaks one of the two.
"""

import pandas as pd
import pytest

from perp_lab.engine.backtest import EngineParams, run_backtest
from perp_lab.plugins import PluginSource
from perp_lab.strategy import RandomEntry, list_plugins
from tests.sandbox_fixtures import load_class
from perp_lab.validate.causality import CUTS, causality_failures, with_moving_columns
from tests.conftest import make_bars
from tests.test_store import cfg  # noqa: F401

PLUGINS = list_plugins()


def test_plugins_discovered():
    assert {"donchian_atr", "bb_vwap_reversion", "funding_fade"} <= set(PLUGINS)


@pytest.mark.parametrize("name", PLUGINS)
def test_plugin_has_one_sentence_hypothesis(name):
    h = PluginSource.read(name).meta.hypothesis
    assert h and len(h) > 20, f"{name}: module docstring must start with a one-sentence hypothesis"


@pytest.mark.parametrize("name", PLUGINS)
def test_plugin_no_lookahead(cfg, name):
    # the default params and every grid corner, each compared run in its own worker
    assert causality_failures(cfg, PluginSource.read(name), make_bars(seed=1)) == []


@pytest.mark.parametrize("name", PLUGINS)
def test_plugin_trades_on_synthetic_data(name):
    """A plugin that never trades can't fail the lookahead test — make sure ours do."""
    cls = load_class(name)
    bars = with_moving_columns(make_bars(seed=1))
    s = cls()
    r = run_backtest(s, bars, s.prepare(bars), symbol="X", tf="1h", ep=EngineParams())
    assert len(r.trades) > 0


def _assert_causal_in_process(strat_factory, bars):
    """The old in-process check, kept for the RandomEntry wrapper (a trusted class, not a plugin)."""
    full = strat_factory()
    f_full = full.prepare(bars)
    r_full = run_backtest(full, bars, f_full, symbol="X", tf="1h", ep=EngineParams(), record_signals=True)
    for k in CUTS:
        s = strat_factory()
        f_cut = s.prepare(bars.iloc[:k])
        pd.testing.assert_frame_equal(f_full.iloc[:k], f_cut, check_exact=False, rtol=1e-9, atol=1e-12)
        r_cut = run_backtest(s, bars.iloc[:k], f_cut, symbol="X", tf="1h", ep=EngineParams(), record_signals=True)
        assert [x for x in r_full.signals if x[0] < k - 1] == [x for x in r_cut.signals if x[0] < k - 1]


def test_random_entry_wrapper_is_causal():
    cls = load_class("donchian_atr")
    _assert_causal_in_process(lambda: RandomEntry(cls(), 0.01, seed=3), make_bars(seed=2))


CHEATER_TEXT = '''"""Peeks at the next close."""
from perp_lab.indicators import atr
from perp_lab.strategy import HOLD, Strategy, enter


class Cheater(Strategy):
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
'''
CHEATER = PluginSource.from_text("cheater", CHEATER_TEXT)


def test_lookahead_check_catches_a_cheater(cfg):
    fails = causality_failures(cfg, CHEATER, make_bars(seed=4))
    assert len(fails) == 1 and fails[0].startswith("defaults: AssertionError")
