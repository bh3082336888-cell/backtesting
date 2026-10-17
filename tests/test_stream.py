"""The streamed exam: per symbol, a fresh worker is sent the research bars (the warm-up), then one
holdout bar at a time; the trusted engine asks it for each decision with the engine's own position,
and the symbols' streams advance in lockstep. Unmarked tests run on the in-process twin; the
`sandbox` ones spawn real workers."""

import contextlib
import io
import json
import math
import random
import threading
import time

import numpy as np
import pandas as pd
import pytest

from perp_lab.data.store import load_bars
from perp_lab.engine.backtest import EngineParams, run_backtest
from perp_lab.engine.metrics import daily_returns, portfolio_returns, sharpe
from perp_lab.holdout import issue_holdout_token
from perp_lab.judge.exam import run_exam
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client, session
from perp_lab.sandbox.decisions import Decisions
from perp_lab.sandbox.errors import PluginError, ProtocolError, WorkerLost
from perp_lab.sandbox.protocol import signal_from
from perp_lab.sandbox.stream import Aborted, Lockstep, StreamedStrategy
from perp_lab.strategy import HOLD, Signal
from tests.conftest import REAL_PLUGINS, REPO, make_bars, market_config
from tests.sandbox_fixtures import InProcessWorker, load_class
from tests.test_worker import header, sandboxed

END = pd.Timestamp("2026-10-01", tz="UTC")          # 92 holdout days on the synthetic market
SHORT_END = pd.Timestamp("2026-07-11", tz="UTC")    # 10 days (60 bars), for tests that need a few steps
HOLDOUT_MS = pd.Timestamp("2026-07-01", tz="UTC").value // 10**6
BAR_MS = 4 * 3_600_000


def token(tmp_path):
    return issue_holdout_token("fam", "stream test", tmp_path / "test_log.jsonl")


def real_source(name):
    return PluginSource.from_text(name, (REPO / "strategies" / f"{name}.py").read_bytes())


def batch_exam(cfg, cls, tf, symbols, params, tok, end):
    """3a's exam: one in-process batch run per symbol over research + holdout bars, then sliced."""
    start = cfg.holdout_start
    daily, parts = {}, []
    for s in symbols:
        bars = load_bars(cfg, s, tf, end=end, holdout=tok)
        strat = cls(**params)
        res = run_backtest(strat, bars, strat.prepare(bars), symbol=s, tf=tf, ep=EngineParams.from_config(cfg, s))
        dr = daily_returns(res.equity, res.time)
        daily[s] = dr[(dr.index >= start) & (dr.index < end)] if len(dr) else dr
        t = pd.DataFrame(res.trades)
        if len(t):
            st = pd.to_datetime(t.signal_time, unit="ms", utc=True)
            parts.append(t[(st >= start) & (st < end)])
    trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return daily, portfolio_returns(daily), trades


def assert_same_exam(res, daily, port, trades):
    assert list(res.daily) == list(daily)
    for s in daily:
        pd.testing.assert_series_equal(res.daily[s], daily[s], check_exact=True)
    pd.testing.assert_series_equal(res.portfolio, port, check_exact=True)
    pd.testing.assert_frame_equal(res.trades, trades, check_exact=True)


def plugin(name, body, *, head=""):
    """A fixture plugin; `body` runs in entry(self, ctx) before its `return HOLD`."""
    text = f'''"""Fixture plugin {name}."""
import os
import time

from perp_lab.strategy import HOLD, Strategy

HOLDOUT_MS = {HOLDOUT_MS}
BAR_MS = {BAR_MS}
{head}

class Fixture(Strategy):
    params = {{}}

    def prepare(self, bars):
        return bars[[]]

    def entry(self, ctx):
        {body}
        return HOLD
'''
    return PluginSource.from_text(name, text)


# ---------------------------------------------------------------- streamed == batch for causal plugins

@pytest.mark.parametrize("name", REAL_PLUGINS)
def test_streamed_exam_equals_the_batch_exam(tmp_path, name):
    cfg = market_config(tmp_path)
    tok = token(tmp_path)
    src = real_source(name)
    res = run_exam(cfg, src, "4h", cfg.symbols, src.meta.params, tok, END)
    assert (res.start, res.end, res.params) == (cfg.holdout_start, END, src.meta.params)
    assert_same_exam(res, *batch_exam(cfg, load_class(name), "4h", cfg.symbols, src.meta.params, tok, END))


ORACLE = f'''"""Knows the next close: in the holdout, long iff it is above this one."""
from perp_lab.strategy import EXIT, HOLD, Strategy, enter


class Oracle(Strategy):
    params = {{}}

    def prepare(self, bars):
        return bars[[]].assign(nxt=bars.close.shift(-1))

    def entry(self, ctx):
        if ctx["open_time"] >= {HOLDOUT_MS} and ctx["nxt"] > ctx["close"]:
            return enter(1, 0.9 * ctx["close"])
        return HOLD

    def manage(self, ctx):
        return HOLD if ctx["nxt"] > ctx["close"] else EXIT
'''


def test_an_oracle_plugin_gains_nothing_streamed(tmp_path):
    cfg = market_config(tmp_path)
    tok = token(tmp_path)
    src = PluginSource.from_text("oracle", ORACLE)
    _, port, trades = batch_exam(cfg, session.load_plugin("oracle", src.text), "4h", cfg.symbols, {}, tok, END)
    assert sharpe(port) > 5 and len(trades) > 50              # in batch it sees the next bar
    res = run_exam(cfg, src, "4h", cfg.symbols, {}, tok, END)
    assert abs(sharpe(res.portfolio)) < 1                     # streamed there is no next bar to see
    assert len(res.trades) == 0 and (res.portfolio == 0).all()


# ---------------------------------------------------------------- lockstep

class FakeWorker:
    """Stands in for a stream worker holding the first k0 bars: logs each step's start and end
    (phase, symbol, bar time), sleeps a little at random in between, and holds."""

    def __init__(self, symbol, bars, k0, log, lock):
        self.symbol, self.t, self.n, self.log, self.lock = symbol, bars.open_time.to_numpy(), k0, log, lock

    def step(self, position, *, append=None, window=None):
        assert window is None
        if append is not None:
            assert len(append) == 1 and int(append.open_time.iloc[0]) == self.t[self.n]
            self.n += 1
        t = int(self.t[self.n - 1])
        with self.lock:
            self.log.append(("start", self.symbol, t))
        time.sleep(random.random() * 0.002)
        with self.lock:
            self.log.append(("end", self.symbol, t))
        return HOLD


def stream_threads(series: dict, k0: int) -> list:
    """Each symbol's engine run with a StreamedStrategy over a FakeWorker, one thread each, all in
    one lockstep; the step log."""
    lockstep = Lockstep(list(series))
    log, lock, errors = [], threading.Lock(), []

    def one(s):
        try:
            w = FakeWorker(s, series[s], k0, log, lock)
            run_backtest(StreamedStrategy(w, Decisions.empty(), k0, series[s], s, lockstep), series[s],
                         series[s][[]], symbol=s, tf="1h", ep=EngineParams())
        except BaseException as e:
            errors.append(e)
            lockstep.abort()
        finally:
            lockstep.finish(s)

    threads = [threading.Thread(target=one, args=(s,), daemon=True) for s in series]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=60)
    assert not any(th.is_alive() for th in threads), "the lockstep deadlocked"
    assert errors == []
    return log


def assert_lockstep(log):
    """No step at a later bar starts before every step at an earlier bar has ended."""
    start, end = {}, {}
    for k, (phase, s, t) in enumerate(log):
        (start if phase == "start" else end)[s, t] = k
    assert set(start) == set(end)
    for (s, t), k in start.items():
        for (s2, t2), e in end.items():
            if t2 < t:
                assert e < k, f"{s} started bar {t} before {s2} finished bar {t2}"


def steps_by_symbol(log):
    out = {}
    for phase, s, _ in log:
        out[s] = out.get(s, 0) + (phase == "start")
    return out


def test_lockstep_orders_steps_across_symbols():
    bars = make_bars(n=80, seed=3)
    log = stream_threads({"A": bars, "B": bars, "C": bars}, k0=30)
    assert_lockstep(log)
    assert steps_by_symbol(log) == {"A": 50, "B": 50, "C": 50}   # bars 29 (the last warm-up bar) to 78


def test_lockstep_with_a_symbol_missing_bars():
    bars = make_bars(n=80, seed=3)
    gappy = bars.drop(bars.index[[40, 41, 55, 60, 79]])            # its last bar is missing too
    log = stream_threads({"A": bars, "B": gappy, "C": bars}, k0=30)
    assert_lockstep(log)
    assert steps_by_symbol(log) == {"A": 50, "B": 45, "C": 50}
    assert {t for _, s, t in log if s == "B"}.isdisjoint(int(t) for t in bars.open_time.iloc[[40, 41, 55, 60]])


def test_no_step_until_every_warm_up_is_done():
    lockstep, passed = Lockstep(["A", "B"]), threading.Event()

    def a():
        lockstep.turn("A", 100)
        passed.set()
    ta = threading.Thread(target=a, daemon=True)
    ta.start()
    assert not passed.wait(0.2)                  # B is still warming up
    tb = threading.Thread(target=lambda: lockstep.turn("B", 100), daemon=True)
    tb.start()
    assert passed.wait(5)
    ta.join(5)
    tb.join(5)
    assert not ta.is_alive() and not tb.is_alive()


def test_abort_wakes_a_waiting_turn_and_every_later_one():
    lockstep, raised = Lockstep(["A", "B"]), []

    def a():
        try:
            lockstep.turn("A", 100)
        except Aborted:
            raised.append("A")
    th = threading.Thread(target=a, daemon=True)
    th.start()
    time.sleep(0.1)
    lockstep.abort()
    th.join(5)
    assert raised == ["A"] and not th.is_alive()
    with pytest.raises(Aborted):
        lockstep.turn("B", 100)


def test_a_finished_symbol_no_longer_holds_the_others_back():
    lockstep = Lockstep(["A", "B"])
    lockstep.finish("B")
    lockstep.turn("A", 100)                      # returns at once


# ---------------------------------------------------------------- the exam's threads and workers

class Spy(InProcessWorker):
    """The twin, counting each symbol's steps and the workers alive at once."""

    steps: dict = {}
    alive: set = set()
    most = 0
    spawned = 0
    lock = threading.Lock()

    def __init__(self, cfg, src):
        super().__init__(cfg, src)
        with Spy.lock:
            Spy.spawned += 1
            Spy.alive.add(id(self))
            Spy.most = max(Spy.most, len(Spy.alive))

    def bars(self, symbol, tf, df):
        self.symbol = symbol
        super().bars(symbol, tf, df)

    def step(self, position, **kw):
        with Spy.lock:
            Spy.steps[self.symbol] = Spy.steps.get(self.symbol, 0) + 1
        return super().step(position, **kw)

    def close(self):
        with Spy.lock:
            Spy.alive.discard(id(self))
        super().close()


@pytest.fixture
def spy(monkeypatch):
    monkeypatch.setattr(Spy, "steps", {})
    monkeypatch.setattr(Spy, "alive", set())
    monkeypatch.setattr(Spy, "most", 0)
    monkeypatch.setattr(Spy, "spawned", 0)
    monkeypatch.setattr(client, "spawn", Spy)
    return Spy


def test_a_failing_symbol_aborts_the_others(tmp_path, spy):
    cfg = market_config(tmp_path)
    src = plugin("fails_on_eth", 'if ctx.symbol == "ETHUSDT" and ctx["open_time"] >= HOLDOUT_MS + 10 * BAR_MS:\n'
                                 '            raise ValueError("boom")')
    before = threading.active_count()
    with pytest.raises(PluginError) as e:
        run_exam(cfg, src, "4h", cfg.symbols, {}, token(tmp_path), END)
    assert (e.value.kind, e.value.type, e.value.message) == ("exception", "ValueError", "boom")
    assert spy.steps["ETHUSDT"] == 12          # the last research bar, then holdout bars 0..10
    assert spy.steps["BTCUSDT"] <= 12          # not the 552 it would have streamed alone
    assert spy.spawned == 2 and spy.alive == set()      # a plugin error is not retried
    assert threading.active_count() == before


def test_a_lost_worker_restarts_the_whole_batch_once(tmp_path, spy, monkeypatch):
    cfg = market_config(tmp_path)
    src = real_source("donchian_atr")
    tok = token(tmp_path)
    lost, counted = [], Spy.step

    def flaky(self, position, **kw):
        if self.symbol == "BTCUSDT" and not lost and spy.steps.get("BTCUSDT") == 5:
            lost.append(1)
            raise WorkerLost("died", "worker exited with code -9")
        return counted(self, position, **kw)
    monkeypatch.setattr(Spy, "step", flaky)
    res = run_exam(cfg, src, "4h", cfg.symbols, src.meta.params, tok, SHORT_END)
    assert lost == [1] and spy.spawned == 4 and spy.alive == set()   # both symbols again, in fresh workers
    assert_same_exam(res, *batch_exam(cfg, load_class("donchian_atr"), "4h", cfg.symbols, src.meta.params, tok,
                                      SHORT_END))


def test_a_worker_lost_twice_fails_the_exam(tmp_path, spy, monkeypatch):
    cfg = market_config(tmp_path)

    def lost(self, position, **kw):
        if self.symbol == "ETHUSDT":
            raise WorkerLost("timeout", "over 30 s of wall-clock time")
        return InProcessWorker.step(self, position, **kw)
    monkeypatch.setattr(Spy, "step", lost)
    with pytest.raises(PluginError) as e:
        run_exam(cfg, real_source("donchian_atr"), "4h", cfg.symbols, {}, token(tmp_path), SHORT_END)
    assert e.value.kind == "timeout" and spy.spawned == 4 and spy.alive == set()


DICE = f'''"""Enters with a random stop on holdout bars: a fresh draw on every run."""
import random

from perp_lab.strategy import HOLD, Strategy, enter


class Dice(Strategy):
    params = {{}}

    def prepare(self, bars):
        return bars[[]]

    def entry(self, ctx):
        if ctx["open_time"] >= {HOLDOUT_MS}:
            return enter(1, ctx["close"] * (0.8 - 0.1 * random.random()))
        return HOLD
'''


def losing_btc_at(spy, monkeypatch, step):
    """BTC's worker is lost once, before its step number `step` (the first is the last research bar)."""
    lost, counted = [], Spy.step

    def flaky(self, position, **kw):
        if self.symbol == "BTCUSDT" and not lost and spy.steps.get("BTCUSDT") == step - 1:
            lost.append(1)
            raise WorkerLost("died", "worker exited with code -9")
        return counted(self, position, **kw)
    monkeypatch.setattr(Spy, "step", flaky)
    return lost


def test_a_rerun_must_repeat_the_lost_runs_decisions(tmp_path, spy, monkeypatch):
    # The worker is lost after the plugin decided on holdout bars. The rerun would be a second draw
    # for a plugin that decides at random there (or that saw its first results), so it must decide
    # exactly as the lost run did, up to where each symbol stopped: otherwise a mismatch.
    cfg = market_config(tmp_path)
    lost = losing_btc_at(spy, monkeypatch, 6)
    with pytest.raises(PluginError) as e:
        run_exam(cfg, PluginSource.from_text("dice", DICE), "4h", cfg.symbols, {}, token(tmp_path), SHORT_END)
    assert e.value.kind == "mismatch"
    assert lost == [1] and spy.spawned == 4 and spy.alive == set()


def test_an_honest_plugin_whose_worker_is_lost_once_passes_on_the_rerun(tmp_path, spy, monkeypatch):
    cfg = market_config(tmp_path)
    src = PluginSource.from_text("watcher", WATCHER)          # trades in the holdout, deterministically
    ref = run_exam(cfg, src, "4h", cfg.symbols, {}, token(tmp_path), SHORT_END)
    assert len(ref.trades) > 0
    monkeypatch.setattr(Spy, "spawned", 0)
    monkeypatch.setattr(Spy, "steps", {})
    lost = losing_btc_at(spy, monkeypatch, 30)
    res = run_exam(cfg, src, "4h", cfg.symbols, {}, token(tmp_path), SHORT_END)
    assert lost == [1] and spy.spawned == 4 and spy.alive == set()
    assert_same_exam(res, ref.daily, ref.portfolio, ref.trades)


def test_a_thread_that_cannot_start_stops_the_ones_that_did(tmp_path, spy, monkeypatch):
    # BTC's thread is up and waits in its first turn for ETH, whose thread never starts: it must
    # be stopped and joined, its worker closed, not left blocked with a live worker.
    cfg = market_config(tmp_path)
    real = threading.Thread.start

    def start(self):
        if self.name == "stream-ETHUSDT":
            raise RuntimeError("can't start new thread")
        return real(self)
    monkeypatch.setattr(threading.Thread, "start", start)
    before = threading.active_count()
    with pytest.raises(RuntimeError, match="can't start new thread"):
        run_exam(cfg, real_source("donchian_atr"), "4h", cfg.symbols, {}, token(tmp_path), SHORT_END)
    assert threading.active_count() == before        # BTC's thread was stopped and joined
    assert spy.spawned == 1 and spy.alive == set()   # after closing its worker


def test_symbols_go_in_batches_of_max_workers(tmp_path, spy, monkeypatch):
    cfg = market_config(tmp_path)
    src = real_source("donchian_atr")
    tok = token(tmp_path)
    monkeypatch.setattr(client, "max_workers", lambda cfg: 1)
    res = run_exam(cfg, src, "4h", cfg.symbols, src.meta.params, tok, SHORT_END)
    assert spy.most == 1 and spy.spawned == 2
    assert_same_exam(res, *batch_exam(cfg, load_class("donchian_atr"), "4h", cfg.symbols, src.meta.params, tok,
                                      SHORT_END))


def test_each_workers_stderr_goes_to_the_stderr_dir(tmp_path):
    cfg = market_config(tmp_path)
    src = plugin("loud", 'if ctx["open_time"] >= HOLDOUT_MS:\n            print("deciding", ctx.symbol)')
    run_exam(cfg, src, "4h", cfg.symbols, {}, token(tmp_path), SHORT_END, stderr_dir=tmp_path / "err")
    files = sorted(p.name for p in (tmp_path / "err").iterdir())
    assert [f.split("-")[0] for f in files] == ["BTCUSDT", "ETHUSDT"]
    assert "deciding ETHUSDT" in (tmp_path / "err" / files[1]).read_text()


# ---------------------------------------------------------------- what a step sends and gets back

def test_a_step_sends_only_the_next_bar(tmp_path, spy, monkeypatch):
    cfg = market_config(tmp_path)
    sent = []

    def watching(self, position, *, append=None, window=None):
        assert window is None
        sent.append(None if append is None else [int(t) for t in append.open_time])
        return InProcessWorker.step(self, position, append=append, window=window)
    monkeypatch.setattr(Spy, "step", watching)
    src = real_source("donchian_atr")
    run_exam(cfg, src, "4h", ["BTCUSDT"], src.meta.params, token(tmp_path), SHORT_END)
    assert sent[0] is None                                   # the last research bar: already held
    assert sent[1:] == [[HOLDOUT_MS + k * BAR_MS] for k in range(59)]   # never the window's last bar


WATCHER = f'''"""Prints the position it is shown from the last research bar on; goes long when flat in the holdout."""
from perp_lab.strategy import HOLD, Strategy, enter

SLOTS = ("side", "entry_price", "entry_i", "stop", "target", "qty", "best_close", "bars_held")


class Watcher(Strategy):
    params = {{}}

    def prepare(self, bars):
        return bars[[]]

    def on_bar(self, ctx):
        p = ctx.position
        if ctx["open_time"] >= {HOLDOUT_MS - BAR_MS}:
            print("pos", ctx.i, None if p is None else repr(tuple(float(getattr(p, k)) for k in SLOTS)))
        if p is None and ctx["open_time"] >= {HOLDOUT_MS}:
            return enter(1, 0.8 * ctx["close"], 1.05 * ctx["close"])
        return HOLD
'''


def test_the_plugin_sees_the_engines_position_at_every_step(tmp_path):
    cfg = market_config(tmp_path)
    tok = token(tmp_path)
    src = PluginSource.from_text("watcher", WATCHER)
    run_exam(cfg, src, "4h", ["BTCUSDT"], {}, tok, SHORT_END, stderr_dir=tmp_path / "err")
    [f] = (tmp_path / "err").iterdir()
    streamed = [ln for ln in f.read_text().splitlines() if ln.startswith("pos ")]
    bars = load_bars(cfg, "BTCUSDT", "4h", end=SHORT_END, holdout=tok)
    strat, out = session.load_plugin("watcher", src.text)(), io.StringIO()
    with contextlib.redirect_stdout(out):
        run_backtest(strat, bars, strat.prepare(bars), symbol="BTCUSDT", tf="4h",
                     ep=EngineParams.from_config(cfg, "BTCUSDT"))
    batch = [ln for ln in out.getvalue().splitlines() if ln.startswith("pos ")]
    assert streamed == batch and len(batch) == 60
    assert sum("None" not in ln for ln in batch) > 10        # shown a position on many steps


@pytest.mark.parametrize("sig", [
    {"action": "buy", "stop": 1.0, "target": None, "has_stop": True, "has_target": False},
    {"action": "long", "stop": True, "target": None, "has_stop": True, "has_target": False},
    {"action": "long", "stop": "1", "target": None, "has_stop": True, "has_target": False},
    {"action": "long", "stop": 1.0, "target": None, "has_stop": False, "has_target": False},
    {"action": "long", "stop": None, "target": None, "has_stop": True, "has_target": False},
    {"action": "long", "stop": 1.0, "target": None, "has_stop": 1, "has_target": False},
    {"action": "long", "stop": 10 ** 400, "target": None, "has_stop": True, "has_target": False},
    {"action": "long", "stop": 1.0, "target": None, "has_stop": True},
    {"action": "long", "stop": 1.0, "target": None, "has_stop": True, "has_target": False, "extra": 1},
    ["long", 1.0],
])
def test_a_malformed_signal_is_a_protocol_error(sig):
    with pytest.raises(ProtocolError):
        signal_from({"ok": True, "signal": sig}, {})


@pytest.mark.parametrize("key, value", [("stop", math.inf), ("stop", -math.inf), ("target", math.inf),
                                        ("target", -math.inf)])
def test_an_infinite_stop_or_target_in_a_step_is_a_protocol_error(key, value):
    # Pinned from the final review: a hold with target +inf on a short (-inf on a long) closed at an
    # infinite price in the parent's engine, and the exam dropped the symbol's remaining days.
    sig = {"action": "hold", "stop": None, "target": None, "has_stop": False, "has_target": False,
           key: value, f"has_{key}": True}
    with pytest.raises(ProtocolError, match="finite"):
        signal_from(json.loads(json.dumps({"ok": True, "signal": sig})), {})


OVERFLOW = f'''"""Shorts once in the holdout, then moves its target far above the price: finite, but the
close there overflows equity to -inf."""
from perp_lab.strategy import HOLD, Signal, Strategy, enter


class Overflow(Strategy):
    params = {{}}

    def prepare(self, bars):
        return bars[[]]

    def entry(self, ctx):
        if ctx["open_time"] >= {HOLDOUT_MS} + 5 * {BAR_MS}:
            return enter(-1, 1.1 * ctx["close"])
        return HOLD

    def manage(self, ctx):
        return Signal("hold", None, 1.7e308)
'''


def test_a_streamed_run_whose_equity_is_not_finite_fails_the_exam(tmp_path):
    # Not finite equity would drop days from the exam's portfolio, never count against it.
    cfg = market_config(tmp_path)
    with pytest.raises(PluginError) as e:
        run_exam(cfg, PluginSource.from_text("overflow", OVERFLOW), "4h", cfg.symbols, {}, token(tmp_path), END)
    assert e.value.kind == "mismatch" and "finite" in e.value.message


def test_a_signal_with_sections_is_a_protocol_error():
    sig = {"action": "hold", "stop": None, "target": None, "has_stop": False, "has_target": False}
    with pytest.raises(ProtocolError):
        signal_from({"ok": True, "signal": sig}, {"x": np.zeros(1)})


def test_a_well_formed_signal_round_trips():
    assert signal_from({"ok": True, "signal": {"action": "hold", "stop": None, "target": None,
                                               "has_stop": False, "has_target": False}}, {}) == HOLD
    s = signal_from({"ok": True, "signal": {"action": "long", "stop": 95, "target": float("nan"),
                                            "has_stop": True, "has_target": True}}, {})
    assert s.action == "long" and s.stop == 95.0 and type(s.stop) is float and math.isnan(s.target)
    assert signal_from({"ok": True, "signal": {"action": "exit", "stop": None, "target": None,
                                               "has_stop": False, "has_target": False}}, {}) == Signal("exit")


# ---------------------------------------------------------------- real workers

@pytest.mark.sandbox
def test_sleeping_in_a_stream_step_fails_the_exam(tmp_path, monkeypatch):
    cfg = sandboxed(market_config(tmp_path), stream_step_s=1)
    started, real = [], client.spawn

    def counting(*a):
        started.append(1)
        return real(*a)
    monkeypatch.setattr(client, "spawn", counting)
    src = plugin("sleeper", 'if ctx["open_time"] >= HOLDOUT_MS:\n            time.sleep(5)')
    with pytest.raises(PluginError) as e:
        run_exam(cfg, src, "4h", ["BTCUSDT"], {}, token(tmp_path), SHORT_END)
    assert e.value.kind == "timeout" and len(started) == 2        # retried once in a fresh worker


@pytest.mark.sandbox
@pytest.mark.parametrize("step_s", [1, 2])
def test_a_busy_loop_in_a_stream_step_is_cpu_limit(tmp_path, monkeypatch, step_s):
    # A step's CPU and wall-clock limits are equal, so a spinning step trips the wall clock first
    # (with a little less CPU, the time it took to start). It is the plugin's busy loop, not lost
    # time: cpu_limit, not retried.
    cfg = sandboxed(market_config(tmp_path), stream_step_s=step_s)
    started, real = [], client.spawn

    def counting(*a):
        started.append(1)
        return real(*a)
    monkeypatch.setattr(client, "spawn", counting)
    src = plugin("spinner", 'if ctx["open_time"] >= HOLDOUT_MS:\n            while True:\n                pass')
    with pytest.raises(PluginError) as e:
        run_exam(cfg, src, "4h", ["BTCUSDT"], {}, token(tmp_path), SHORT_END)
    assert e.value.kind == "cpu_limit" and len(started) == 1


@pytest.mark.sandbox
def test_a_forged_signal_is_a_protocol_error(tmp_path):
    cfg = market_config(tmp_path)
    bad = ('{"ok": True, "signal": {"action": "long", "stop": True, "target": None, "has_stop": True, '
           '"has_target": False}}')
    head = f"\nimport json, struct\n\n\ndef forge():\n    os.write(4, {header(bad)})\n    time.sleep(30)\n"
    src = plugin("forger", 'if ctx["open_time"] >= HOLDOUT_MS:\n            forge()', head=head)
    with pytest.raises(PluginError) as e:
        run_exam(cfg, src, "4h", ["BTCUSDT"], {}, token(tmp_path), SHORT_END)
    assert (e.value.kind, e.value.message) == ("protocol", "signal stop must be a number")


@pytest.mark.sandbox
def test_streamed_exam_through_real_workers(tmp_path):
    cfg = market_config(tmp_path)
    tok = token(tmp_path)
    src = real_source("donchian_atr")
    res = run_exam(cfg, src, "4h", cfg.symbols, src.meta.params, tok, END)
    assert len(res.trades) > 0
    assert_same_exam(res, *batch_exam(cfg, load_class("donchian_atr"), "4h", cfg.symbols, src.meta.params, tok, END))
