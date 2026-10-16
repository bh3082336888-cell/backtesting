import pytest

from perp_lab.indicators import atr
from perp_lab.strategy import HOLD, Strategy, enter, load_strategy_class
from perp_lab.validate.causality import causality_failures
from tests.conftest import REAL_PLUGINS, REPO
from tests.test_lookahead import Cheater

EXCHANGE_COLUMNS = ["oi", "oi_value", "quote_volume", "trades", "taker_buy_volume",
                    "taker_buy_quote_volume", "regime_adx", "funding_interval_h"]


def test_real_plugins_are_found():
    assert {"donchian_atr", "donchian_atr_wide", "bb_vwap_reversion", "funding_fade"} <= set(REAL_PLUGINS)


@pytest.mark.parametrize("name", REAL_PLUGINS)
def test_every_real_plugin_has_no_causality_failures(name, monkeypatch):
    monkeypatch.setattr("perp_lab.strategy.STRATEGY_DIR", REPO / "strategies")
    assert causality_failures(load_strategy_class(name)) == []


def test_cheater_is_reported_not_raised():
    fails = causality_failures(Cheater)
    assert len(fails) == 1 and fails[0].startswith("defaults: AssertionError")


def test_plugin_that_crashes_is_reported():
    class Broken(Cheater):
        def prepare(self, bars):
            raise KeyError("no such column")

    assert causality_failures(Broken) == ["defaults: KeyError: 'no such column'"]


def peeker(col):
    class Peeker(Strategy):
        """Goes long when the next bar's value of one column is higher (e.g. next OI)."""
        params = {}

        def prepare(self, bars):
            return bars[[]].assign(nxt=bars[col].shift(-1), now=bars[col], atr=atr(bars, 14))

        def bracket(self, ctx, side):
            return ctx["close"] - side * 2 * ctx["atr"], None

        def entry(self, ctx):
            return enter(1, *self.bracket(ctx, 1)) if ctx["nxt"] > ctx["now"] else HOLD
    return Peeker


@pytest.mark.parametrize("col", EXCHANGE_COLUMNS)
def test_a_plugin_peeking_at_any_column_is_reported(col):
    fails = causality_failures(peeker(col))
    assert fails and fails[0].startswith("defaults: AssertionError")


def test_a_plugin_that_never_signals_cannot_be_verified():
    class Silent(Strategy):
        """Never trades, so a lookahead check proves nothing about it."""
        params = {}

        def prepare(self, bars):
            return bars[[]]

        def bracket(self, ctx, side):
            return None, None

        def entry(self, ctx):
            return HOLD

    assert causality_failures(Silent) == ["defaults: no signals on synthetic data; causality cannot be verified"]
