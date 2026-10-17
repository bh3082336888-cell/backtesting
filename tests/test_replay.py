import math

import numpy as np
import pytest

from perp_lab.engine.backtest import EngineParams, run_backtest
from perp_lab.sandbox.decisions import Decisions, Recorder, decide_last
from perp_lab.sandbox.errors import PluginError, ProtocolError
from perp_lab.sandbox.replay import check_fingerprint, random_draw, replay
from perp_lab.strategy import HOLD, Signal, Strategy, RandomEntry
from perp_lab.validate.causality import with_moving_columns
from tests.conftest import REAL_PLUGINS, make_bars
from tests.sandbox_fixtures import load_class


class Fixture(Strategy):
    name = "fixture"
    params = {}

    def __init__(self, fn, feats=None):
        self.p = {}
        self.fn = fn
        self.feats = feats

    def prepare(self, bars):
        return bars[[]] if self.feats is None else self.feats(bars)

    def on_bar(self, ctx):
        return self.fn(ctx)


def run_ref(s, bars, **kw):
    ep = EngineParams()
    return run_backtest(s, bars, s.prepare(bars), symbol="X", tf="1h", ep=ep, **kw), ep


@pytest.mark.parametrize("name", REAL_PLUGINS)
def test_replaying_the_log_reproduces_the_run(name):
    bars = with_moving_columns(make_bars(seed=1))
    s = Recorder(load_class(name)())
    ref = run_backtest(s, bars, s.inner.prepare(bars), symbol="X", tf="1h", ep=EngineParams())
    res = replay(bars, Decisions.from_log(s.log), symbol="X", tf="1h", ep=EngineParams())
    assert res.trades == ref.trades and np.array_equal(res.equity, ref.equity)
    check_fingerprint(res, (len(ref.trades), float(ref.equity[-1])), EngineParams())


def _enter_then_hold_signals(hold_target):
    def fn(ctx):
        if ctx.position is None:
            return Signal("long", ctx["close"] * 0.9, ctx["close"] * 1.5) if ctx.i % 7 == 0 else HOLD
        return Signal("hold", target=math.nan if ctx.i % 2 else hold_target)
    return fn


def test_nan_and_none_targets_replay_identically():
    bars = make_bars(seed=3)
    s = Recorder(Fixture(_enter_then_hold_signals(None)))
    ref, ep = run_ref(s, bars)
    assert len(ref.trades) > 0
    dec = Decisions.from_log(s.log)
    assert dec.has_target.any() and np.isnan(dec.target[dec.has_target == 1]).any()
    res = replay(bars, dec, symbol="X", tf="1h", ep=ep)
    assert res.trades == ref.trades and np.array_equal(res.equity, ref.equity)
    # the two mean different things to the engine: a NaN target must not be turned into None
    flipped = Decisions(dec.i, dec.action, dec.stop, dec.target, dec.has_stop, np.zeros_like(dec.has_target))
    assert not dec.same(flipped)


def test_a_forged_fingerprint_is_a_mismatch():
    bars = make_bars(seed=1)
    s = Recorder(load_class("donchian_atr")())
    ref, ep = run_ref(s, bars)
    res = replay(bars, Decisions.from_log(s.log), symbol="X", tf="1h", ep=ep)
    good = (len(ref.trades), float(ref.equity[-1]))
    for bad in ((good[0] + 1, good[1]), (good[0], good[1] + 1e-9)):
        with pytest.raises(PluginError) as e:
            check_fingerprint(res, bad, ep)
        assert e.value.kind == "mismatch"


def test_a_feature_named_close_is_a_mismatch():
    bars = make_bars(seed=4)

    def fn(ctx):
        if ctx.position is None:
            return Signal("long", ctx["close"] * 0.95) if ctx.i % 50 == 0 else HOLD
        return HOLD
    s = Recorder(Fixture(fn, feats=lambda b: b[[]].assign(close=b.close * 1.01)))
    ref, ep = run_ref(s, bars)
    assert len(ref.trades) > 0
    res = replay(bars, Decisions.from_log(s.log), symbol="X", tf="1h", ep=ep)
    with pytest.raises(PluginError) as e:
        check_fingerprint(res, (len(ref.trades), float(ref.equity[-1])), ep)
    assert e.value.kind == "mismatch"


def test_random_draw_matches_random_entry():
    f = RandomEntry(load_class("donchian_atr")(), 0.02, seed=7).prepare(make_bars(seed=2))
    fire, side = random_draw(len(f), 7, 0.02)
    assert np.array_equal(np.where(fire, side, 0), f["_rand"].to_numpy())


def _random_log():
    bars = make_bars(seed=2)
    re_ = RandomEntry(load_class("donchian_atr")(), 0.02, seed=7)
    s = Recorder(re_)
    ref, ep = run_ref(s, bars)
    return bars, s, ref, ep


def test_random_entry_replay_with_the_draw_matches():
    bars, s, ref, ep = _random_log()
    draw = random_draw(len(bars), 7, 0.02)
    res = replay(bars, Decisions.from_log(s.log), symbol="X", tf="1h", ep=ep, draw=draw)
    assert res.trades == ref.trades


def test_random_entry_log_off_the_draw_is_a_mismatch():
    bars, s, ref, ep = _random_log()
    draw = random_draw(len(bars), 7, 0.02)
    j = int(np.flatnonzero(~draw[0])[0])
    log = sorted(s.log + [(j, Signal("long", float(bars.close.iloc[j]) * 0.9))], key=lambda e: e[0])
    with pytest.raises(PluginError) as e:
        replay(bars, Decisions.from_log(log), symbol="X", tf="1h", ep=ep, draw=draw)
    assert e.value.kind == "mismatch" and "off the draw" in e.value.message


def test_a_reversal_from_manage_passes_random_entry_replay():
    bars = make_bars(seed=5)
    draw = random_draw(len(bars), 9, 0.5)
    k = int(np.flatnonzero(draw[0])[0])
    side = int(draw[1][k])
    c = float(bars.close.iloc[k])
    entry = Signal("long" if side > 0 else "short", c * (0.9 if side > 0 else 1.1))
    rev = Signal("short" if side > 0 else "long", c * (1.2 if side > 0 else 0.8))
    log = [(k, entry), (k + 3, rev)]
    # k+3 is not on the draw, but we are in a position there (the entry fills at k+1)
    replay(bars, Decisions.from_log(log), symbol="X", tf="1h", ep=EngineParams(), draw=draw)


def test_validate_rejects_bad_logs():
    ok = Decisions.from_log([(1, Signal("long", 1.0)), (5, Signal("exit"))])
    ok.validate(10)

    def bad(**over):
        d = {k: getattr(ok, k) for k in ("i", "action", "stop", "target", "has_stop", "has_target")}
        d.update(over)
        return Decisions(**d)

    cases = [
        (bad(i=np.array([5, 1], "<i8")), 10),
        (ok, 5),
        (bad(i=np.array([-1, 5], "<i8")), 10),
        (bad(action=np.array([4, 3], "<i1")), 10),
        (bad(has_stop=np.array([2, 0], "<i1")), 10),
        (bad(stop=np.array([1.0, 2.0], "<f4")), 10),
        (bad(target=np.array([1.0], "<f8")), 10),
    ]
    for d, n in cases:
        with pytest.raises(ProtocolError):
            d.validate(n)


def test_decisions_helpers():
    d = Decisions.from_log([(1, Signal("long", 1.0)), (2, Signal("hold", target=math.nan)), (5, Signal("exit"))])
    assert d.n_actions() == 2 and len(d.before(5).i) == 2 and d.same(Decisions.from_log(
        [(1, Signal("long", 1.0)), (2, Signal("hold", target=math.nan)), (5, Signal("exit"))]))
    assert Decisions.empty().n_actions() == 0 and set(d.by_bar()) == {1, 2, 5}


def test_bare_holds_are_not_logged():
    s = Recorder(Fixture(lambda ctx: Signal("hold", stop=1.0) if ctx.i == 3 else HOLD))
    run_ref(s, make_bars(n=10))
    assert [i for i, _ in s.log] == [3]


@pytest.mark.parametrize("ret", [("long", 1.0), Signal("buy", 1.0), Signal("long", "1")])
def test_a_non_signal_return_is_a_type_error(ret):
    s = Recorder(Fixture(lambda ctx: ret))
    with pytest.raises(TypeError):
        run_ref(s, make_bars(n=10))


def test_decide_last_checks_the_signal():
    bars = make_bars(n=50)
    assert decide_last(Fixture(lambda ctx: Signal("long", 1.0)), "X", bars, None) == Signal("long", 1.0)
    with pytest.raises(TypeError):
        decide_last(Fixture(lambda ctx: None), "X", bars, None)
