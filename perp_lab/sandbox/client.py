"""The trusted side of the sandbox: starts confined workers, enforces their limits from outside,
and turns what goes wrong into the errors the harness records.

Callers go through the module (`from perp_lab.sandbox import client`, then `client.spawn`,
`client.check`, `client.Batch`, `client.fresh_run`, `client.max_workers`), so tests can put the
in-process twin in place of `spawn`.

Nothing a worker sends is unpickled or parsed as Arrow: its replies are read by
`read_frame(..., from_worker=True)`, which accepts only JSON and raw int64, int8 and float64
arrays sized against `sandbox.max_output_mb`. Every string kept from a worker (an error's type
and message, its stderr) is cut to printable ASCII here, whatever the worker says it did.

Limits: each request may use `sandbox.task_cpu_s` of the worker's CPU time (read from
/proc/<pid>/stat) and `task_cpu_s * wall_factor` of wall-clock time; a stream step (one `prepare`
and one decision) may use `sandbox.stream_step_s` of each. Over the CPU limit the worker is killed
and the request is `PluginError("cpu_limit")`; over the wall-clock limit it is killed and
`WorkerLost("timeout")`; a worker that goes away is `WorkerLost("died")`. `Batch` and `fresh_run`
retry a `WorkerLost` once in a fresh worker; a second is `PluginError(kind)`. Streams are run by
`stream.py`.

A worker dies with its parent: it is SIGKILLed when the thread that spawned it exits (the
kernel's parent-death signal belongs to that thread, not to the process). So a worker must be
closed by the thread that spawned it, never handed to another thread that outlives this one.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from perp_lab.engine.backtest import BacktestResult, EngineParams
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import confine
from perp_lab.sandbox.decisions import Decisions
from perp_lab.sandbox.errors import (PluginError, ProtocolError, SandboxUnavailable, WorkerLost, ascii_bytes,
                                     ascii_text)
from perp_lab.sandbox.protocol import (Arrow, decisions_from, position_json, read_frame, signal_from, table_bytes,
                                       warmup_decisions, write_frame)
from perp_lab.sandbox.replay import check_fingerprint, random_draw, replay
from perp_lab.strategy import PositionView, Signal

# Exactly this environment, nothing inherited (no secrets). PYTHONHASHSEED makes hash() of a
# string agree across workers, which is why the worker is started with -s -P and not -I.
ENV = {"LANG": "C.UTF-8", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
STDERR_CAP = 64 << 10                   # kept per worker, for humans
STDERR_TAIL = 2 << 10                   # carried on PluginError.stderr, for the judge's log
READY_S = 120                           # interpreter start, confinement and the numpy/pandas/pyarrow imports
CLOSE_S = 2                             # after stdin closes, before the kill
SLICE_S = 0.25                          # how often a waiting request checks the limits
# The only error kinds a worker may report: "exception", and "features" on a run that asked for
# them. Every other kind (cpu_limit, died, timeout, protocol, mismatch, metadata) is the parent's
# to decide; a worker claiming one is a protocol violation, so it can't earn itself a retry.
WORKER_KINDS = ("exception", "features")
_DECISION_KEYS = ("i", "action", "stop", "target", "has_stop", "has_target")
_META = ("params", "grid", "timeframes")
_CLK_TCK = os.sysconf("SC_CLK_TCK")


@dataclass(frozen=True, eq=False)
class RunOutput:
    decisions: Decisions
    fingerprint: tuple[int, float]      # the worker's own trade count and final equity
    features: pd.DataFrame | None       # float64, on the run's bars index; None unless asked for


# ---------------------------------------------------------------- the worker's threads

class _Tracked:
    """The worker's stdout, noting whether a read came back short (end of stream)."""

    def __init__(self, stream):
        self.stream, self.ended = stream, False

    def read(self, n: int) -> bytes:
        b = self.stream.read(n)
        if len(b) < n:
            self.ended = True
        return b


class _Truncated(Exception):
    """The stream ended inside a frame: the worker died mid-reply, or closed its descriptor."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _read_frames(stream, frames: queue.Queue, max_bytes: int, stop: threading.Event) -> None:
    """Puts each reply frame, or the exception that ended the stream, on `frames` (one at a time)."""
    tracked = _Tracked(stream)
    while True:
        try:
            item = read_frame(tracked, max_bytes=max_bytes, from_worker=True)
        except ProtocolError as e:
            item = _Truncated(e.message) if tracked.ended else e
        except BaseException as e:       # EOFError, or the pipe closed under us
            item = e
        while not stop.is_set():
            try:
                frames.put(item, timeout=SLICE_S)
                break
            except queue.Full:
                pass
        if stop.is_set() or isinstance(item, BaseException):
            return


def _drain(stream, kept: bytearray, tail: bytearray, lock: threading.Lock) -> None:
    """Reads the worker's stderr until EOF: the first STDERR_CAP bytes, and a rolling tail."""
    while True:
        try:
            chunk = stream.read1(65536)
        except (OSError, ValueError):
            return
        if not chunk:
            return
        with lock:
            kept += chunk[:max(0, STDERR_CAP - len(kept))]
            tail += chunk
            del tail[:-STDERR_TAIL]


def _write(stream, header: dict, sections: dict | None, box: list) -> None:
    try:
        write_frame(stream, header, sections)
    except BaseException as e:
        box.append(e)


def _reap(proc: subprocess.Popen, stop: threading.Event) -> None:
    """Kill and wait. Also the finalizer of a Worker dropped without close(), and at exit."""
    stop.set()
    try:
        proc.kill()
    except OSError:
        pass
    proc.wait()


# ---------------------------------------------------------------- one worker

class Worker:
    """One confined worker process. One request at a time; not for use from several threads.
    Close it from the thread that spawned it: it is killed when that thread exits."""

    def __init__(self, cfg, proc: subprocess.Popen):
        sb = cfg["sandbox"]
        self.pid = proc.pid
        self.abi: int | None = None
        self.last_cpu_s = 0.0           # CPU seconds the last request used, measured from outside
        self._proc = proc
        self._cpu_limit = float(sb["task_cpu_s"])
        self._wall_limit = float(sb["task_cpu_s"]) * float(sb["wall_factor"])
        self._step_limit = float(sb["stream_step_s"])     # CPU and wall clock, each
        self._index: pd.Index | None = None
        self._dead = self._closed = False
        self._lock = threading.Lock()
        self._kept, self._tail = bytearray(), bytearray()
        self._frames: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._writer: threading.Thread | None = None
        self._finalizer = weakref.finalize(self, _reap, proc, self._stop)
        self._reader = threading.Thread(target=_read_frames, daemon=True, name=f"sandbox-{proc.pid}-out",
                                        args=(proc.stdout, self._frames, int(sb["max_output_mb"]) << 20, self._stop))
        self._drainer = threading.Thread(target=_drain, daemon=True, name=f"sandbox-{proc.pid}-err",
                                         args=(proc.stderr, self._kept, self._tail, self._lock))
        self._reader.start()
        self._drainer.start()

    @property
    def stderr(self) -> bytes:
        """The first 64 KiB the worker wrote to stderr (a plugin's print goes there), printable."""
        with self._lock:
            return ascii_bytes(self._kept)

    def _stderr_tail(self) -> bytes:
        with self._lock:
            return ascii_bytes(self._tail)

    # -- lifecycle

    def _ready(self, min_abi: int) -> None:
        deadline = time.monotonic() + READY_S
        while True:
            try:
                item = self._frames.get(timeout=SLICE_S)
                break
            except queue.Empty:
                if time.monotonic() > deadline:
                    self._kill()
                    raise SandboxUnavailable(f"worker not ready within {READY_S} s") from None
        if isinstance(item, BaseException):
            self._kill()
            last = [ln for ln in self._stderr_tail().decode().splitlines() if ln.strip()]
            raise SandboxUnavailable(f"worker exited before it was ready (code {self._proc.returncode})"
                                     + (f": {ascii_text(last[-1])}" if last else ""))
        header, _ = item
        err = header.get("error")
        if isinstance(err, dict):
            self._kill()
            raise SandboxUnavailable(ascii_text(err.get("message")) or ascii_text(err.get("type"))
                                     or "the worker could not confine itself")
        abi = header.get("abi")
        if header.get("ok") is not True or header.get("ready") is not True or type(abi) is not int:
            self._kill()
            raise SandboxUnavailable("the worker did not report ready")
        if abi < min_abi:
            self._kill()
            raise SandboxUnavailable(f"Landlock ABI {abi} < {min_abi}")
        self.abi = abi

    def _kill(self) -> None:
        if not self._dead:
            self._dead = True
            self._finalizer()
            self._drainer.join(timeout=2)   # so the stderr tail is complete

    def _stream_failure(self, item: BaseException) -> Exception:
        """What the end of the reply stream means. A stream cut mid-frame by a worker that has
        exited is the worker dying (retried); cut by a live one, a protocol violation."""
        if isinstance(item, _Truncated):
            try:
                self._proc.wait(timeout=1)      # its pipe closes a moment before it is reaped
            except subprocess.TimeoutExpired:
                return self._fail(ProtocolError(ascii_text(item.message)))
        elif isinstance(item, ProtocolError):   # its message quotes what the worker sent
            return self._fail(ProtocolError(ascii_text(item.message)))
        return self._fail(WorkerLost("died", f"worker exited with code {self._proc.poll()}"))

    def _fail(self, exc: Exception) -> Exception:
        """Kill the worker; the exception, carrying the stderr tail, for the caller to raise."""
        self._kill()
        exc.stderr = self._stderr_tail()
        return exc

    def close(self) -> None:
        """Close stdin (the worker exits on EOF), wait CLOSE_S, then kill."""
        if self._closed:
            return
        self._closed = True
        if not self._dead and (self._writer is None or not self._writer.is_alive()):
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            try:
                self._proc.wait(timeout=CLOSE_S)
            except subprocess.TimeoutExpired:
                pass
        self._kill()
        self._reader.join(timeout=5)
        if self._writer is not None:
            self._writer.join(timeout=2)
        for stream, thread in ((self._proc.stdin, self._writer), (self._proc.stdout, self._reader),
                               (self._proc.stderr, self._drainer)):
            if thread is None or not thread.is_alive():
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass

    def __enter__(self) -> Worker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- requests

    def _cpu(self) -> float | None:
        """This worker's CPU seconds so far (user + system, all threads), or None if it's gone."""
        try:
            with open(f"/proc/{self.pid}/stat", "rb") as f:
                stat = f.read()
        except OSError:
            return None
        fields = stat[stat.rfind(b")") + 2:].split()    # after the name, which the plugin can set
        return (int(fields[11]) + int(fields[12])) / _CLK_TCK

    def _request(self, header: dict, sections: dict | None = None, *,
                 limits: tuple[float, float] | None = None) -> tuple[dict, dict]:
        """One request and its reply. `limits` is (CPU s, wall-clock s); the batch limits by default."""
        cpu_limit, wall_limit = limits or (self._cpu_limit, self._wall_limit)
        if self._closed or self._dead:
            raise RuntimeError(f"worker {self.pid} is {'closed' if self._closed else 'dead'}")
        try:                            # nothing may arrive between requests
            early = self._frames.get_nowait()
        except queue.Empty:
            early = None
        if isinstance(early, tuple):
            raise self._fail(ProtocolError("worker sent a frame outside a request"))
        if early is not None:
            raise self._stream_failure(early)

        cpu0 = self._cpu() or 0.0
        t0 = time.monotonic()
        box: list = []
        self._writer = threading.Thread(target=_write, args=(self._proc.stdin, header, sections, box),
                                        daemon=True, name=f"sandbox-{self.pid}-in")
        self._writer.start()
        used = 0.0
        try:
            while True:
                try:
                    item = self._frames.get(timeout=SLICE_S)
                except queue.Empty:
                    item = None
                now = self._cpu()
                if now is not None:
                    used = now - cpu0
                if used > cpu_limit:        # finished or not: the limit is on CPU used
                    raise self._fail(PluginError("cpu_limit", "", f"over {cpu_limit:g} s of CPU"))
                if item is not None:
                    break
                if box and not isinstance(box[0], OSError):
                    raise box[0]        # our request could not be framed
                if time.monotonic() - t0 > wall_limit:
                    raise self._fail(WorkerLost("timeout", f"over {wall_limit:g} s of wall-clock time"))
        except BaseException:
            self._kill()
            raise
        finally:
            self.last_cpu_s = used

        if isinstance(item, BaseException):
            raise self._stream_failure(item)
        self._writer.join(timeout=5)
        if self._writer.is_alive() or box:
            raise self._fail(ProtocolError("worker replied before reading its request"))
        reply, secs = item
        if "error" in reply:
            features = header.get("op") == "run" and header.get("features") is True
            e = reply_error(reply["error"], self._stderr_tail(), features=features)
            raise self._fail(e) if e.kind == "protocol" else e
        if reply.get("ok") is not True:
            raise self._fail(ProtocolError("reply is neither ok nor an error"))
        return reply, secs

    def _load(self, src: PluginSource) -> None:
        reply, _ = self._request({"op": "load", "name": src.name}, {"source": src.text})
        check_meta(src, reply)

    def bars(self, symbol: str, tf: str, df: pd.DataFrame) -> None:
        """Replace the bars the worker holds (one symbol's, ever)."""
        reply, _ = self._request({"op": "bars", "symbol": symbol, "tf": tf}, {"table": Arrow(table_bytes(df))})
        if type(reply.get("n")) is not int or reply["n"] != len(df):
            raise self._fail(ProtocolError("worker holds a different number of bars"))
        self._index = df.index

    def run(self, params: dict, ep: EngineParams, *, k: int | None = None, start_i: int = 0,
            random: tuple[int, float] | None = None, features: bool = False) -> RunOutput:
        """One run on the held bars (the first k, when k is set). `random` is (seed, entry_prob)
        for the random-entry benchmark. Features, if asked for, are indexed by the bars' index."""
        if self._index is None:
            raise RuntimeError("send the bars before a run")
        reply, secs = self._request(run_request(params, ep, k=k, start_i=start_i, random=random, features=features))
        return self._parsed("run", lambda: run_output(reply, secs, self._index if k is None else self._index[:k],
                                                      features))

    def prepare(self, params: dict) -> float:
        """The plugin's `prepare` (and the engine's arrays) on the held bars, with these params: the
        CPU seconds the worker spent on it, measured from outside (/proc). For the exam's cost probe."""
        if self._index is None:
            raise RuntimeError("send the bars before a prepare")
        _, secs = self._request(prepare_request(params))
        if secs:
            raise self._fail(ProtocolError("a prepare reply must have no sections"))
        return self.last_cpu_s

    def stream_open(self, params: dict, ep: EngineParams, warmup_k: int) -> Decisions:
        """Start a stream on the held bars: the worker keeps a new strategy instance and returns its
        decisions from a batch run over the first warmup_k bars (none when warmup_k is 0)."""
        if self._index is None:
            raise RuntimeError("send the bars before a stream")
        if not 0 <= warmup_k <= len(self._index):
            raise ValueError(f"warmup_k {warmup_k} is outside the {len(self._index)} bars held")
        reply, secs = self._request(stream_open_request(params, ep, warmup_k))
        return self._parsed("stream_open", lambda: warmup_decisions(secs, warmup_k))

    def step(self, position: PositionView | None, *, append: pd.DataFrame | None = None,
             window: pd.DataFrame | None = None) -> Signal:
        """The stream's signal on the last bar held, after adding `append`'s rows to the held bars
        or replacing them with `window`; `position` is the trusted engine's. Limited to
        `stream_step_s` of CPU and of wall-clock time."""
        if append is not None and window is not None:
            raise ValueError("a step either appends or replaces the bars, not both")
        if self._index is None:
            raise RuntimeError("send the bars before a stream")
        mode, table = ("append", append) if append is not None else ("window", window) if window is not None \
            else ("none", None)
        sections = None if table is None else {"table": Arrow(table_bytes(table))}
        reply, secs = self._request(step_request(position, mode), sections, limits=(self._step_limit,) * 2)
        sig = self._parsed("step", lambda: signal_from(reply, secs))
        self._index = self._index.append(table.index) if mode == "append" else table.index if mode == "window" \
            else self._index
        return sig

    def _parsed(self, what: str, parse):
        """`parse()` of a reply; whatever it raises is the worker's doing: a protocol violation."""
        try:
            return parse()
        except ProtocolError as e:
            raise self._fail(ProtocolError(ascii_text(e.message))) from None
        except Exception as e:
            raise self._fail(ProtocolError(f"malformed {what} reply ({type(e).__name__})")) from None


# ---------------------------------------------------------------- what crosses the pipe

def run_request(params: dict, ep: EngineParams, *, k=None, start_i=0, random=None, features=False) -> dict:
    return {"op": "run", "params": dict(params), "ep": dataclasses.asdict(ep),
            "k": None if k is None else int(k), "start_i": int(start_i),
            "random": None if random is None else {"seed": int(random[0]), "prob": float(random[1])},
            "features": bool(features)}


def prepare_request(params: dict) -> dict:
    return {"op": "prepare", "params": dict(params)}


def stream_open_request(params: dict, ep: EngineParams, warmup_k: int) -> dict:
    return {"op": "stream_open", "params": dict(params), "ep": dataclasses.asdict(ep), "warmup_k": int(warmup_k)}


def step_request(position: PositionView | None, mode: str) -> dict:
    return {"op": "step", "position": position_json(position), "mode": mode}


def run_output(reply: dict, sections: dict, index: pd.Index, features: bool) -> RunOutput:
    """A run reply, checked: decisions within the bars, a fingerprint, and the features asked for."""
    n = len(index)
    fp = reply.get("fingerprint")
    if not (isinstance(fp, list) and len(fp) == 2 and type(fp[0]) is int and fp[0] >= 0
            and type(fp[1]) in (int, float)):
        raise ProtocolError("malformed fingerprint")
    try:
        equity = float(fp[1])
    except OverflowError:
        raise ProtocolError("fingerprint equity is out of range") from None
    if not math.isfinite(equity):
        raise ProtocolError("fingerprint equity is not a finite number")
    names = reply.get("features")
    expected = set(_DECISION_KEYS)
    if features:
        if not isinstance(names, list) or not all(type(x) is str for x in names):
            raise ProtocolError("malformed feature names")
        names = [ascii_text(x) for x in names]
        if len(set(names)) != len(names):
            raise ProtocolError("feature names must be unique")
        expected |= {f"f{j}" for j in range(len(names))}
    elif names is not None:
        raise ProtocolError("features sent without being asked for")
    if set(sections) != expected:
        raise ProtocolError("reply sections are not the ones expected")
    decisions = decisions_from(sections, n)
    feats = None
    if features:
        for j in range(len(names)):
            a = sections[f"f{j}"]
            if a.dtype != np.dtype("<f8") or len(a) != n:
                raise ProtocolError(f"feature {j} must be {n} float64 values")
        feats = pd.DataFrame({name: sections[f"f{j}"] for j, name in enumerate(names)}, index=index, copy=True)
    return RunOutput(decisions, (fp[0], equity), feats)


def reply_error(err, stderr: bytes, *, features: bool = False) -> PluginError:
    """A worker's error reply as a PluginError, its text cut to printable ASCII again here. A kind
    the worker may not report (see WORKER_KINDS) is a ProtocolError. `features`: the request was a
    run that asked for features."""
    if not isinstance(err, dict):
        return ProtocolError("malformed error reply", stderr)
    kind = err.get("kind")
    if kind not in (WORKER_KINDS if features else WORKER_KINDS[:1]):
        return ProtocolError(ascii_text(f"worker reported a {ascii_text(kind, 40) or '?'} error: "
                                        f"{ascii_text(err.get('message'))}"), stderr)
    return PluginError(kind, ascii_text(err.get("type")), ascii_text(err.get("message")), stderr)


def _canon(value) -> str | None:
    try:
        return json.dumps(json.loads(json.dumps(value, allow_nan=True)), sort_keys=True, allow_nan=True)
    except (TypeError, ValueError, RecursionError):
        return None


def check_meta(src: PluginSource, reply: dict) -> None:
    """The plugin's params, grid and timeframes at runtime must equal what its source says."""
    for key in _META:
        want = _canon(getattr(src.meta, key))
        if want is None or _canon(reply.get(key)) != want:
            raise PluginError("metadata", "", f"{key} at runtime differs from the source")


# ---------------------------------------------------------------- starting workers

def _start(cfg) -> Worker:
    """A confined worker, ready, with no plugin loaded; SandboxUnavailable if it can't be had."""
    sb = cfg["sandbox"]
    cmd = [sys.executable, "-s", "-P", "-m", "perp_lab.sandbox.worker",
           "--deny-probe", str(Path(cfg.path).resolve()),       # the config: readable here, never there
           "--memory-mb", str(int(sb["memory_mb"])),
           "--cpu-s", str(math.ceil(sb["exam_cpu_hours"] * 3600 * 2)),   # lifetime backstop only
           "--parent-pid", str(os.getpid())]                            # it dies with this process
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=dict(ENV), cwd="/", close_fds=True, start_new_session=True)
    except OSError as e:
        raise SandboxUnavailable(f"cannot start a worker: {e}") from e
    w = Worker(cfg, proc)
    try:
        w._ready(int(sb["min_landlock_abi"]))
    except BaseException:
        w.close()
        raise
    return w


def check(cfg) -> int:
    """Start a worker, wait for it to confine itself and pass its self-test, close it; its Landlock
    ABI. SandboxUnavailable, with the reason, if this machine can't run plugin code."""
    reason = confine.probe(int(cfg["sandbox"]["min_landlock_abi"]))
    if reason is not None:
        raise SandboxUnavailable(reason)
    w = _start(cfg)
    w.close()
    return w.abi


def spawn(cfg, src: PluginSource) -> Worker:
    """A worker with the plugin loaded. SandboxUnavailable before the plugin is sent;
    PluginError("metadata") if its runtime params, grid or timeframes differ from the source.
    The calling thread must close it: the worker is killed when that thread exits."""
    w = _start(cfg)
    try:
        w._load(src)
    except BaseException:
        w.close()
        raise
    return w


def _physical_mb() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") >> 20


def max_workers(cfg) -> int:
    """Concurrent workers this machine holds: workers x memory_mb within 3/4 of physical memory."""
    return max(1, math.floor(0.75 * _physical_mb() / cfg["sandbox"]["memory_mb"]))


# ---------------------------------------------------------------- batch runs

def _retry_once(fn):
    try:
        return fn()
    except WorkerLost:
        pass                            # transient machine trouble gets one more go, in a fresh worker
    try:
        return fn()
    except WorkerLost as e:
        raise PluginError(e.kind, "", e.message, e.stderr) from e


class Batch:
    """Batch runs of one plugin on one symbol's bars, each replayed by the trusted engine.

    Keeps one worker and sends it the bars once. A worker that fails is closed and not reused; a
    lost one (died, timeout) is replaced once, bars and all. Each worker's stderr, if any, goes to
    `<stderr_dir>/<symbol>-<pid>.txt` when it closes.
    """

    def __init__(self, cfg, src: PluginSource, symbol: str, tf: str, bars: pd.DataFrame,
                 stderr_dir: Path | None = None):
        self.cfg, self.src, self.symbol, self.tf, self.bars = cfg, src, symbol, tf, bars
        self.stderr_dir = None if stderr_dir is None else Path(stderr_dir)
        self._w: Worker | None = None

    def __enter__(self) -> Batch:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        w, self._w = self._w, None
        if w is not None:
            w.close()
            err = w.stderr
            if err and self.stderr_dir is not None:
                self.stderr_dir.mkdir(parents=True, exist_ok=True)
                with open(self.stderr_dir / f"{self.symbol}-{w.pid}.txt", "ab") as f:
                    f.write(err)

    def _run(self, params: dict, ep: EngineParams, random, start_i: int) -> RunOutput:
        try:
            if self._w is None:
                self._w = spawn(self.cfg, self.src)
                self._w.bars(self.symbol, self.tf, self.bars)
            return self._w.run(params, ep, random=random, start_i=start_i)
        except BaseException:
            self.close()
            raise

    def backtest(self, params: dict, ep: EngineParams, *, random: tuple[int, float] | None = None,
                 start_i: int = 0) -> BacktestResult:
        """The plugin's run, as the trusted replay of its decisions. `random` is (seed, entry_prob)."""
        out = _retry_once(lambda: self._run(params, ep, random, start_i))
        draw = None if random is None else random_draw(len(self.bars), int(random[0]), float(random[1]))
        res = replay(self.bars, out.decisions, symbol=self.symbol, tf=self.tf, ep=ep, start_i=start_i, draw=draw)
        check_fingerprint(res, out.fingerprint, ep)
        return res


def fresh_run(cfg, src: PluginSource, symbol: str, tf: str, bars: pd.DataFrame, params: dict,
              ep: EngineParams, *, features: bool = False) -> RunOutput:
    """One run in a new worker that is sent only these bars (and a second one if it is lost)."""
    def once() -> RunOutput:
        with spawn(cfg, src) as w:
            w.bars(symbol, tf, bars)
            return w.run(params, ep, features=features)
    return _retry_once(once)
