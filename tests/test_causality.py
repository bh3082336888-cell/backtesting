from perp_lab.strategy import load_strategy_class
from perp_lab.validate.causality import causality_failures
from tests.test_lookahead import Cheater


def test_real_plugin_has_no_causality_failures():
    assert causality_failures(load_strategy_class("donchian_atr")) == []


def test_cheater_is_reported_not_raised():
    fails = causality_failures(Cheater)
    assert len(fails) == 1 and fails[0].startswith("defaults: AssertionError")


def test_plugin_that_crashes_is_reported():
    class Broken(Cheater):
        def prepare(self, bars):
            raise KeyError("no such column")

    assert causality_failures(Broken) == ["defaults: KeyError: 'no such column'"]
