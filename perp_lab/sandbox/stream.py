"""Streams: a plugin decides on bar t holding bars up to t only, for the holdout exam.

Per symbol, one thread spawns a fresh worker, sends it the research bars (`bars[:k0]`, where k0 is
the first holdout bar) and has it run them in batch (the warm-up, `stream_open`). The trusted
engine then runs over research and holdout bars with a `StreamedStrategy`: the warm-up's decisions
up to the bar before the last research bar, and from the last research bar on one `step` per bar,
which sends that bar (none for the first step, the worker holds it already) and the engine's own
position, and gets the plugin's signal back. The engine makes no decision on a run's last bar, so
the worker never holds that one. For a causal plugin this gives exactly the decisions of one batch
run over all the bars.

`Lockstep`: the symbols streamed together decide bar t before any of them is sent a later bar, so
no worker holds a bar later than one another worker is still deciding on (none can pass another a
later bar through CPU timing), and no step starts before every warm-up is done. Symbols with
missing bars just skip those times.

Failures: any error in one symbol's thread aborts the lockstep, so the others stop at their next
turn; once every thread has joined, the first error is raised. A lost worker (`WorkerLost`)
restarts the whole batch once in fresh workers, a second is `PluginError(kind)`
(`client._retry_once`). One symbol's stream can't be restarted alone: the others already hold
later bars.

Each thread spawns, uses and closes its own worker (a worker dies with the thread that spawned
it), and writes the worker's stderr, if any, to `<stderr_dir>/<symbol>-<pid>.txt`.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import pandas as pd

from perp_lab.engine.backtest import BacktestResult, EngineParams, run_backtest
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client
from perp_lab.sandbox.decisions import Decisions
from perp_lab.sandbox.replay import check_finite
from perp_lab.strategy import HOLD, Ctx, Signal, Strategy

# load(symbol) -> (bars: research then holdout, k0: index of the first holdout bar, engine params)
Loader = Callable[[str], tuple[pd.DataFrame, int, EngineParams]]


class Aborted(Exception):
    """Another symbol's stream failed: this one stops at its next turn."""


class Lockstep:
    """Before a step at bar time t, a symbol waits until every other symbol still streaming has
    reached t (`turn`). A symbol is `None` until its first turn, so no step happens before every
    warm-up is done; `finish` takes a symbol out; after `abort`, every turn raises `Aborted`."""

    def __init__(self, symbols):
        self._cv = threading.Condition()
        self._at: dict[str, int | None] = {s: None for s in symbols}
        self._aborted = False

    def _clear(self, symbol: str, t: int) -> bool:
        return all(v is not None and v >= t for s, v in self._at.items() if s != symbol)

    def turn(self, symbol: str, t: int) -> None:
        with self._cv:
            if not self._aborted:
                prev = self._at[symbol]             # KeyError: not in this lockstep, or finished
                if prev is not None and t < prev:
                    raise ValueError(f"{symbol}: bar time {t} is before {prev}")
                self._at[symbol] = t
                self._cv.notify_all()
                self._cv.wait_for(lambda: self._aborted or self._clear(symbol, t))
            if self._aborted:
                raise Aborted(f"{symbol} stopped: another symbol's stream failed")

    def finish(self, symbol: str) -> None:
        with self._cv:
            self._at.pop(symbol, None)
            self._cv.notify_all()

    def abort(self) -> None:
        with self._cv:
            self._aborted = True
            self._cv.notify_all()


class StreamedStrategy(Strategy):
    """The engine's strategy for one symbol's stream: the warm-up's decisions before the last
    research bar (k0 - 1), then one worker step per bar, in lockstep with the other symbols."""

    name = "streamed"

    def __init__(self, worker, warm: Decisions, k0: int, bars: pd.DataFrame, symbol: str, lockstep: Lockstep):
        self.p = {}
        self.worker, self.k0, self.bars, self.symbol, self.lockstep = worker, k0, bars, symbol, lockstep
        self._warm = warm.by_bar()
        self._t = bars["open_time"].to_numpy()

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        return bars[[]]

    def on_bar(self, ctx: Ctx) -> Signal:
        i = ctx.i
        if i < self.k0 - 1:
            return self._warm.get(i, HOLD)
        self.lockstep.turn(self.symbol, int(self._t[i]))
        return self.worker.step(ctx.position, append=self.bars.iloc[[i]] if i >= self.k0 else None)


def _close(w, symbol: str, stderr_dir: Path | None) -> None:
    w.close()
    err = w.stderr
    if err and stderr_dir is not None:
        stderr_dir.mkdir(parents=True, exist_ok=True)
        with open(stderr_dir / f"{symbol}-{w.pid}.txt", "ab") as f:
            f.write(err)


def stream_symbol(cfg, src: PluginSource, symbol: str, tf: str, bars: pd.DataFrame, k0: int, params: dict,
                  ep: EngineParams, lockstep: Lockstep, stderr_dir: Path | None = None) -> BacktestResult:
    """One symbol's streamed run, in a fresh worker that this thread spawns and closes."""
    w = client.spawn(cfg, src)
    try:
        w.bars(symbol, tf, bars.iloc[:k0])
        warm = w.stream_open(params, ep, k0)
        res = run_backtest(StreamedStrategy(w, warm, k0, bars, symbol, lockstep), bars, bars[[]],
                           symbol=symbol, tf=tf, ep=ep)
        lockstep.finish(symbol)             # done deciding: the others need not wait for the close
        check_finite(res)                   # NaN days would be dropped from the exam, not counted
        return res
    finally:
        _close(w, symbol, None if stderr_dir is None else Path(stderr_dir))


def _streams(cfg, src: PluginSource, tf: str, params: dict, symbols: list[str], load: Loader,
             stderr_dir: Path | None) -> dict[str, BacktestResult]:
    """One attempt: every symbol in its own thread and worker, all in one lockstep."""
    lockstep = Lockstep(symbols)
    results: dict[str, BacktestResult] = {}
    errors: list[BaseException] = []
    lock = threading.Lock()

    def one(symbol: str) -> None:
        try:
            bars, k0, ep = load(symbol)
            results[symbol] = stream_symbol(cfg, src, symbol, tf, bars, k0, params, ep, lockstep, stderr_dir)
        except Aborted:
            pass                            # the error that caused it is recorded
        except BaseException as e:
            with lock:
                errors.append(e)
            lockstep.abort()
        finally:
            lockstep.finish(symbol)

    threads = [threading.Thread(target=one, args=(s,), name=f"stream-{s}", daemon=True) for s in symbols]
    for th in threads:
        th.start()
    try:
        for th in threads:
            th.join()
    except BaseException:                   # e.g. Ctrl-C: the threads stop at their next turn
        lockstep.abort()
        for th in threads:
            th.join()
        raise
    if errors:
        raise errors[0]
    return results


def run_batch(cfg, src: PluginSource, tf: str, params: dict, symbols: list[str], load: Loader,
              stderr_dir: Path | None = None) -> dict[str, BacktestResult]:
    """Each symbol's streamed run, all at once in lockstep; the trusted engine's result per symbol.
    A lost worker restarts the whole batch once, in fresh workers."""
    return client._retry_once(lambda: _streams(cfg, src, tf, params, list(symbols), load, stderr_dir))
