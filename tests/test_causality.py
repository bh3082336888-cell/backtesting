import threading
import time

import pytest

from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client
from perp_lab.sandbox.errors import PluginError
from perp_lab.validate.causality import causality_failures
from tests.conftest import REAL_PLUGINS, REPO
from tests.test_lookahead import CHEATER, CHEATER_TEXT
from tests.test_store import cfg  # noqa: F401

EXCHANGE_COLUMNS = ["oi", "oi_value", "quote_volume", "trades", "taker_buy_volume",
                    "taker_buy_quote_volume", "regime_adx", "funding_interval_h"]


def test_real_plugins_are_found():
    assert {"donchian_atr", "donchian_atr_wide", "bb_vwap_reversion", "funding_fade"} <= set(REAL_PLUGINS)


@pytest.mark.parametrize("name", REAL_PLUGINS)
def test_every_real_plugin_has_no_causality_failures(cfg, name, monkeypatch):
    monkeypatch.setattr("perp_lab.strategy.STRATEGY_DIR", REPO / "strategies")
    assert causality_failures(cfg, PluginSource.read(name)) == []


def test_cheater_is_reported_not_raised(cfg):
    fails = causality_failures(cfg, CHEATER)
    assert len(fails) == 1 and fails[0].startswith("defaults: AssertionError")


def test_plugin_that_crashes_is_reported(cfg):
    text = CHEATER_TEXT.replace("return bars[[]].assign(nxt=bars.close.shift(-1), atr=atr(bars, 14))",
                                'raise KeyError("no such column")')
    assert causality_failures(cfg, PluginSource.from_text("broken", text)) == ["defaults: KeyError: 'no such column'"]


PEEKER = '''"""Goes long when the next bar's value of one column is higher (e.g. next OI)."""
from perp_lab.indicators import atr
from perp_lab.strategy import HOLD, Strategy, enter


class Peeker(Strategy):
    params = {}

    def prepare(self, bars):
        return bars[[]].assign(nxt=bars[COL].shift(-1), now=bars[COL], atr=atr(bars, 14))

    def bracket(self, ctx, side):
        return ctx["close"] - side * 2 * ctx["atr"], None

    def entry(self, ctx):
        return enter(1, *self.bracket(ctx, 1)) if ctx["nxt"] > ctx["now"] else HOLD
'''


def peeker(col) -> PluginSource:
    return PluginSource.from_text("peeker", PEEKER.replace("COL", repr(col)))


@pytest.mark.parametrize("col", EXCHANGE_COLUMNS)
def test_a_plugin_peeking_at_any_column_is_reported(cfg, col):
    fails = causality_failures(cfg, peeker(col))
    assert fails and fails[0].startswith("defaults: AssertionError")


SILENT = '''"""Never trades, so a lookahead check proves nothing about it."""
from perp_lab.strategy import HOLD, Strategy


class Silent(Strategy):
    params = {}

    def prepare(self, bars):
        return bars[[]]

    def bracket(self, ctx, side):
        return None, None

    def entry(self, ctx):
        return HOLD
'''


def test_a_plugin_that_never_signals_cannot_be_verified(cfg):
    src = PluginSource.from_text("silent", SILENT)
    assert causality_failures(cfg, src) == ["defaults: no signals on synthetic data; causality cannot be verified"]


def test_waves_never_overlap_a_longer_run(cfg, monkeypatch):
    real, lock, runs = client.fresh_run, threading.Lock(), []

    def spy(cfg, src, symbol, tf, bars, params, ep, *, features=False):
        start = time.monotonic()
        time.sleep(0.02)
        try:
            return real(cfg, src, symbol, tf, bars, params, ep, features=features)
        finally:
            with lock:
                runs.append((start, time.monotonic(), len(bars)))

    monkeypatch.setattr(client, "fresh_run", spy)
    monkeypatch.setattr(client, "max_workers", lambda cfg: 8)
    src = PluginSource.read("donchian_atr")
    assert causality_failures(cfg, src, cuts=[400, 1100, 1777]) == []
    lengths = sorted({n for _, _, n in runs})
    assert lengths[:3] == [400, 1100, 1777] and len(runs) > len(lengths)     # variants share a wave
    for _, end, n in runs:
        assert all(end <= start2 for start2, _, m in runs if m > n)


MEMOISER = '''"""Remembers the longest series it was shown and replays it truncated on shorter ones."""
from perp_lab.indicators import atr
from perp_lab.strategy import HOLD, Strategy, enter

_CACHE = None


class Memoiser(Strategy):
    params = {}

    def prepare(self, bars):
        global _CACHE
        if _CACHE is None or len(_CACHE) < len(bars):
            _CACHE = bars[[]].assign(nxt=bars.close.shift(-1), atr=atr(bars, 14))
        return _CACHE.iloc[:len(bars)]

    def bracket(self, ctx, side):
        return ctx["close"] - side * 2 * ctx["atr"], None

    def entry(self, ctx):
        n = ctx["nxt"]
        if n != n:
            return HOLD
        side = 1 if n > ctx["close"] else -1
        return enter(side, *self.bracket(ctx, side))
'''


@pytest.mark.sandbox
def test_a_memoising_cheater_is_caught(cfg):
    fails = causality_failures(cfg, PluginSource.from_text("memoiser", MEMOISER))
    assert len(fails) == 1 and fails[0].startswith("defaults: AssertionError: features at cut")


def test_a_cut_run_that_dies_is_reported_for_its_variant(cfg, monkeypatch):
    real = client.fresh_run

    def dying(cfg, src, symbol, tf, bars, params, ep, *, features=False):
        if len(bars) == 1100:
            raise PluginError("died")
        return real(cfg, src, symbol, tf, bars, params, ep, features=features)

    monkeypatch.setattr(client, "fresh_run", dying)
    honest = CHEATER_TEXT.replace("shift(-1)", "shift(1)")
    assert causality_failures(cfg, PluginSource.from_text("honest", honest)) == ["defaults: plugin_error: died"]
