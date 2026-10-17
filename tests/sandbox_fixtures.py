"""Shared helpers for sandbox tests: the in-process twin of a worker, and a real plugin loader.

Tests without the `sandbox` marker get `InProcessWorker` in place of `client.spawn` (see the
autouse fixture in conftest.py). It runs plugin code unconfined, in the test process, so it exists
only here, under tests/.
"""

import contextlib
import io
import json
import os
import signal
import threading
import time
from pathlib import Path

from perp_lab import strategy as _strategy
from perp_lab.sandbox import client, protocol
from perp_lab.sandbox.errors import ascii_bytes
from perp_lab.sandbox.protocol import table_bytes, table_frame
from tests.conftest import REPO

# The real functions, captured before any test patches the module.
REAL_CHECK = client.check
REAL_SPAWN = client.spawn

# redirect_stdout swaps sys.stdout for the whole process, so twins run their plugin calls one at
# a time (tests may run several on threads).
_CAPTURE = threading.RLock()


def load_class(name):
    """Load a real plugin in-process, from the repo's strategies/ (Task 12 re-points this)."""
    old = _strategy.STRATEGY_DIR
    _strategy.STRATEGY_DIR = REPO / "strategies"
    try:
        return _strategy.load_strategy_class(name)
    finally:
        _strategy.STRATEGY_DIR = old


class InProcessWorker:
    """`client.Worker`'s interface, served by a `session.Session` in this process.

    Same metadata check, same error conversion (`session.error_reply`), and the same `RunOutput`
    (built by `client.run_output` from the reply a real worker would send); stream replies are
    checked by the same `protocol.warmup_decisions` and `protocol.signal_from`. Bars go through
    Arrow and requests through JSON, as they do on the pipe. `stderr` is what the plugin printed.
    No limits are enforced.
    """

    def __init__(self, cfg, src):
        from perp_lab.sandbox import session
        self._session = session
        self.pid = os.getpid()
        self.abi = None
        self.last_cpu_s = 0.0
        self._out = io.StringIO()
        self._index = None
        self._s = session.Session()                   # a fresh plugin module per instance
        client.check_meta(src, self._call(self._s.load, src.name, src.text))

    @property
    def stderr(self) -> bytes:
        return ascii_bytes(self._out.getvalue().encode("utf-8", "replace")[:client.STDERR_CAP])

    def _call(self, fn, *args, features=False):
        t0 = time.process_time()
        try:
            with _CAPTURE, contextlib.redirect_stdout(self._out), contextlib.redirect_stderr(self._out):
                return fn(*args)
        except (Exception, SystemExit) as e:
            raise client.reply_error(self._session.error_reply(e)["error"], self.stderr, features=features) from e
        finally:
            self.last_cpu_s = time.process_time() - t0

    def bars(self, symbol, tf, df):
        self._call(self._s.bars, symbol, tf, table_frame(table_bytes(df)))
        self._index = df.index

    def run(self, params, ep, *, k=None, start_i=0, random=None, features=False):
        if self._index is None:
            raise RuntimeError("send the bars before a run")
        req = json.loads(json.dumps(client.run_request(params, ep, k=k, start_i=start_i, random=random,
                                                       features=features), allow_nan=True))
        result = self._call(self._s.run, req["params"], req["ep"], req["k"], req["start_i"], req["random"],
                            req["features"], features=req["features"])
        reply, sections = self._session.run_reply(result, req["features"])
        reply = json.loads(json.dumps(reply, allow_nan=True))
        return client.run_output(reply, sections, self._index if k is None else self._index[:k], features)

    def prepare(self, params):
        if self._index is None:
            raise RuntimeError("send the bars before a prepare")
        req = json.loads(json.dumps(client.prepare_request(params), allow_nan=True))
        self._call(self._s.prepare, req["params"])
        return self.last_cpu_s

    def stream_open(self, params, ep, warmup_k):
        if self._index is None:
            raise RuntimeError("send the bars before a stream")
        if not 0 <= warmup_k <= len(self._index):
            raise ValueError(f"warmup_k {warmup_k} is outside the {len(self._index)} bars held")
        req = json.loads(json.dumps(client.stream_open_request(params, ep, warmup_k), allow_nan=True))
        dec = self._call(self._s.stream_open, req["params"], req["ep"], req["warmup_k"])
        _, sections = self._session.stream_open_reply(dec)
        return protocol.warmup_decisions(sections, warmup_k)

    def step(self, position, *, append=None, window=None):
        if append is not None and window is not None:
            raise ValueError("a step either appends or replaces the bars, not both")
        if self._index is None:
            raise RuntimeError("send the bars before a stream")
        mode, table = ("append", append) if append is not None else ("window", window) if window is not None \
            else ("none", None)
        req = json.loads(json.dumps(client.step_request(position, mode), allow_nan=True))
        sent = None if table is None else table_frame(table_bytes(table))
        signal = self._call(self._s.step, req["position"], req["mode"], sent)
        reply, sections = self._session.step_reply(signal)
        sig = protocol.signal_from(json.loads(json.dumps(reply, allow_nan=True)), sections)
        self._index = self._index.append(table.index) if mode == "append" else table.index if mode == "window" \
            else self._index
        return sig

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def running_workers() -> list[int]:
    """Pids of this process's children that are sandbox workers and still running."""
    pids = set()
    for tid in os.listdir("/proc/self/task"):
        with contextlib.suppress(OSError):
            pids.update(int(p) for p in Path(f"/proc/self/task/{tid}/children").read_text().split())
    out = []
    for pid in sorted(pids):
        with contextlib.suppress(OSError):
            if b"perp_lab.sandbox.worker" in Path(f"/proc/{pid}/cmdline").read_bytes():
                out.append(pid)
    return out


def kill_workers(pids):
    for pid in pids:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
