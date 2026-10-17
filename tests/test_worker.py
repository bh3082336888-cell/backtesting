"""The confined worker and its trusted client: batch runs, limits, failures and confinement seen
through a real worker. Tests marked `sandbox` spawn real workers; the rest run on the twin."""

import ctypes
import errno
import os
import select
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from perp_lab.config import Config
from perp_lab.data.store import load_candles
from perp_lab.engine.backtest import EngineParams, run_backtest
from perp_lab.holdout import issue_holdout_token
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client, confine, session, worker
from perp_lab.sandbox.errors import PluginError, SandboxUnavailable
from perp_lab.validate.causality import with_moving_columns
from tests import sandbox_fixtures
from tests.conftest import REAL_PLUGINS, REPO, make_bars, market_config
from tests.sandbox_fixtures import InProcessWorker, load_class

IPC_CREAT, IPC_RMID = 0o1000, 0


@pytest.fixture
def cfg(tmp_path):
    return market_config(tmp_path)


def sandboxed(cfg, **over):
    """A copy of cfg with some `sandbox:` values changed."""
    return Config(raw={**cfg.raw, "sandbox": {**cfg["sandbox"], **over}}, path=cfg.path)


def plugin(name, prepare="return bars[[]]", *, params="{}", tail=""):
    """A fixture plugin from source text; `prepare` is the body of prepare(self, bars)."""
    text = f'''"""Fixture plugin {name}."""
import os
import time

import numpy as np
import pandas as pd

from perp_lab.strategy import HOLD, Strategy


class Fixture(Strategy):
    params = {params}

    def prepare(self, bars):
{textwrap.indent(textwrap.dedent(prepare).strip(), " " * 8)}

    def entry(self, ctx):
        return HOLD
{textwrap.dedent(tail)}
'''
    return PluginSource.from_text(name, text)


def real_source(name):
    return PluginSource.from_text(name, (REPO / "strategies" / f"{name}.py").read_bytes())


def batch_run(cfg, src, bars=None):
    with client.Batch(cfg, src, "X", "1h", make_bars() if bars is None else bars) as b:
        return b.backtest(src.meta.params, EngineParams())


def plugin_error(fn, *args, **kw) -> PluginError:
    with pytest.raises(PluginError) as e:
        fn(*args, **kw)
    return e.value


def printable(s):
    return all(" " <= c <= "~" for c in s)


@pytest.fixture
def spawns(monkeypatch):
    """Counts the workers `client.spawn` starts during the test."""
    real, started = client.spawn, []

    def counting(*a):
        started.append(1)
        return real(*a)
    monkeypatch.setattr(client, "spawn", counting)
    return started


def forging(frame: str, then: str = "time.sleep(30)") -> str:
    """A prepare() body that writes `frame` (a bytes expression) to the protocol descriptor, fd 4,
    which a plugin can always find, then does `then`."""
    return f"""
        import json, struct
        os.write(4, {frame})
        {then}"""


def header(obj: str) -> str:
    """A whole frame with no sections, its header the Python expression `obj`."""
    return f'(lambda h: struct.pack(">I", len(h)) + h)(json.dumps({obj}).encode())'


@pytest.mark.sandbox
@pytest.mark.parametrize("name", REAL_PLUGINS)
def test_sandboxed_batch_equals_in_process(cfg, name):
    bars = with_moving_columns(make_bars(seed=1))
    s = load_class(name)()
    ref = run_backtest(s, bars, s.prepare(bars), symbol="X", tf="1h", ep=EngineParams())
    res = batch_run(cfg, real_source(name), bars)
    assert len(ref.trades) > 0
    assert res.trades == ref.trades and np.array_equal(res.equity, ref.equity)


@pytest.mark.sandbox
def test_in_process_twin_gives_the_same_output(cfg):
    text = (REPO / "strategies" / "bb_vwap_reversion.py").read_bytes() + b'\nprint("hello from the plugin")\n'
    src = PluginSource.from_text("bb_vwap_reversion", text)
    bars = with_moving_columns(make_bars(seed=2))
    ep = EngineParams()
    outs = []
    for make in (client.spawn, InProcessWorker):
        with make(cfg, src) as w:
            w.bars("X", "1h", bars)
            runs = [w.run(src.meta.params, ep, features=True),
                    w.run({**src.meta.params, "bb_n": 30}, ep, k=1500, random=(3, 0.05), features=True),
                    w.run(src.meta.params, ep, start_i=200)]
        outs.append((runs, w.stderr))                 # complete once the worker is closed
    (real, real_err), (twin, twin_err) = outs
    assert real[0].decisions.n_actions() > 0
    for a, b in zip(real, twin):
        assert a.decisions.same(b.decisions) and a.fingerprint == b.fingerprint
        if a.features is None:
            assert b.features is None
        else:
            pd.testing.assert_frame_equal(a.features, b.features)
    assert real[1].features.index.equals(bars.index[:1500])
    assert b"hello from the plugin" in real_err and b"hello from the plugin" in twin_err


@pytest.mark.sandbox
def test_a_plugin_exception_comes_back_with_its_type(cfg):
    e = plugin_error(batch_run, cfg, plugin("broken", 'return bars[["no such column"]]'))
    assert (e.kind, e.type) == ("exception", "KeyError")
    assert "no such column" in e.message and e.reason().startswith("plugin_error: exception: KeyError: ")


@pytest.mark.sandbox
def test_a_busy_loop_is_cpu_limit(cfg):
    t0 = time.monotonic()
    e = plugin_error(batch_run, sandboxed(cfg, task_cpu_s=2), plugin("spin", "while True:\n    pass"))
    assert e.kind == "cpu_limit"
    assert time.monotonic() - t0 < 2 * 4                  # caught on CPU, well before the wall limit


@pytest.mark.sandbox
def test_os_exit_is_retried_once_then_died(cfg, monkeypatch, tmp_path):
    real, spawned = client.spawn, []

    def counting(*a):
        spawned.append(1)
        return real(*a)
    monkeypatch.setattr(client, "spawn", counting)
    src = plugin("exits", 'print("about to exit \\x1b[31m", flush=True)\nos._exit(3)')
    with client.Batch(cfg, src, "BTCUSDT", "1h", make_bars(), stderr_dir=tmp_path / "err") as b:
        e = plugin_error(b.backtest, {}, EngineParams())
    assert e.kind == "died" and len(spawned) == 2
    assert b"about to exit" in e.stderr and b"\x1b" not in e.stderr
    files = sorted((tmp_path / "err").glob("BTCUSDT-*.txt"))
    assert len(files) == 2 and all(b"about to exit" in f.read_bytes() for f in files)


@pytest.mark.sandbox
@pytest.mark.parametrize("value", ["10 ** 400", "float('nan')", "float('inf')"])
def test_a_fingerprint_that_is_not_a_finite_number_is_a_protocol_error(cfg, value):
    # A well-formed run reply (no decisions) whose only fault is the final equity.
    empty = [{"name": k, "dtype": d, "n": 0} for k, d in (("i", "<i8"), ("action", "<i1"), ("stop", "<f8"),
                                                         ("target", "<f8"), ("has_stop", "<i1"), ("has_target", "<i1"))]
    src = plugin("fp", forging(header(f'{{"ok": True, "fingerprint": [0, {value}], "features": None, '
                                      f'"sections": {empty!r}}}')))
    e = plugin_error(batch_run, cfg, src)
    assert e.kind == "protocol" and printable(e.message)


@pytest.mark.sandbox
def test_a_protocol_error_from_the_reader_is_cut_to_printable_ascii(cfg):
    src = plugin("dtype", forging(header('{"ok": True, "sections": [{"name": "a", "dtype": "\\u00e9\\x1b" * 1500, '
                                        '"n": 1}]}')))
    e = plugin_error(batch_run, cfg, src)
    assert e.kind == "protocol" and "dtype" in e.message
    assert printable(e.message) and len(e.message) <= 200
    assert printable(str(e)) and len(str(e)) <= 300


@pytest.mark.sandbox
@pytest.mark.parametrize("kind", ["cpu_limit", "died", "timeout", "protocol", "mismatch", "metadata", "features",
                                  "bogus"])
def test_a_worker_cannot_claim_a_parent_only_error(cfg, spawns, kind):
    # Only "exception" (and "features" when features were asked for) may come from a worker;
    # claiming "timeout" or "died" must not earn a retry.
    src = plugin("claims", forging(header(f'{{"error": {{"kind": "{kind}", "type": "X", "message": "forged"}}, '
                                          f'"sections": []}}')))
    e = plugin_error(batch_run, cfg, src)
    assert e.kind == "protocol" and len(spawns) == 1


@pytest.mark.sandbox
def test_a_non_numeric_feature_is_a_features_error(cfg):
    src = plugin("objfeat", 'return bars[[]].assign(s="x")')
    e = plugin_error(client.fresh_run, cfg, src, "X", "1h", make_bars(n=50), {}, EngineParams(), features=True)
    assert e.kind == "features" and "numeric" in e.message


@pytest.mark.sandbox
def test_half_a_frame_then_exit_is_retried_then_died(cfg, spawns):
    src = plugin("half", forging('struct.pack(">I", 100) + b\'{"ok": tr\'', then="os._exit(3)"))
    e = plugin_error(batch_run, cfg, src)
    assert e.kind == "died" and len(spawns) == 2


@pytest.mark.sandbox
def test_half_a_frame_from_a_live_worker_is_a_protocol_error(cfg, spawns):
    src = plugin("halfalive", forging('struct.pack(">I", 100) + b\'{"ok": tr\'', then="os.close(4); time.sleep(30)"))
    e = plugin_error(batch_run, cfg, src)
    assert e.kind == "protocol" and len(spawns) == 1


@pytest.mark.sandbox
def test_sleeping_past_the_wall_limit_is_a_timeout(cfg):
    t0 = time.monotonic()
    e = plugin_error(batch_run, sandboxed(cfg, task_cpu_s=1, wall_factor=2), plugin("sleeps", "time.sleep(5)"))
    assert e.kind == "timeout"
    assert time.monotonic() - t0 < 2 * 5                  # two attempts, each cut at 2 s, not 5


@pytest.mark.sandbox
def test_runtime_metadata_must_match_the_source(cfg):
    src = plugin("rewrites", params='{"n": 1}', tail='Fixture.params["n"] = 9')
    assert src.meta.params == {"n": 1}
    e = plugin_error(client.spawn, cfg, src)
    assert e.kind == "metadata"
    assert plugin_error(batch_run, cfg, src).kind == "metadata"


@pytest.mark.sandbox
def test_string_hashes_agree_across_workers(cfg):
    src = plugin("hashes", """
        h = hash("abc")
        return bars[[]].assign(lo=float(h & 0xFFFFFFFF), hi=float((h >> 32) & 0xFFFFFFFF))""")
    a, b = (client.fresh_run(cfg, src, "X", "1h", make_bars(n=50), {}, EngineParams(), features=True)
            for _ in range(2))
    pd.testing.assert_frame_equal(a.features, b.features)


@pytest.mark.sandbox
def test_no_environment_reaches_the_worker(cfg, monkeypatch):
    monkeypatch.setenv("PGPASSWORD", "sentinel")
    e = plugin_error(batch_run, cfg, plugin("env", 'raise RuntimeError(" ".join(sorted(os.environ)))'))
    assert e.type == "RuntimeError"
    assert set(e.message.split()) == {"LANG", "PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                      "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"}


@pytest.mark.sandbox
def test_data_and_runs_are_unreadable(cfg):
    lab_db = cfg.runs_dir / "lab.db"
    lab_db.parent.mkdir(parents=True, exist_ok=True)
    lab_db.write_bytes(b"SQLite format 3\0")
    paths = [cfg.path, lab_db, cfg.data_dir / "BTCUSDT" / "4h" / "all.parquet",
             REPO / "config" / "lab.yaml", REPO / "strategies" / "donchian_atr.py"]
    assert all(p.is_file() for p in paths)
    src = plugin("reader", f"""
        out = []
        for p in {[str(p) for p in paths]!r}:
            try:
                open(p, "rb").read(1)
                out.append("READ")
            except OSError as e:
                out.append(type(e).__name__)
        raise RuntimeError(" ".join(out))""")
    e = plugin_error(batch_run, cfg, src)
    assert e.message.split() == ["PermissionError"] * len(paths)


@pytest.mark.sandbox
def test_a_forged_token_cannot_load_holdout_bars(cfg, tmp_path):
    token = issue_holdout_token("lin", "test", tmp_path / "log.jsonl")
    assert len(load_candles(cfg, "BTCUSDT", "4h", holdout=token)) > 0   # the bars are there to read
    src = plugin("forger", f"""
        from perp_lab import holdout
        from perp_lab.config import Config
        from perp_lab.data.store import load_bars
        tok = holdout.HoldoutToken("lin", "forged", "n", holdout._MINT_KEY)
        holdout.check_token(tok)                      # minting works inside the worker...
        load_bars(Config(raw={cfg.raw!r}), "BTCUSDT", "4h", holdout=tok)   # ...reading doesn't
        raise RuntimeError("holdout bars were read")""")
    e = plugin_error(batch_run, cfg, src)
    assert e.kind == "exception" and e.type in ("FileNotFoundError", "PermissionError"), e.reason()


@pytest.mark.sandbox
def test_one_worker_cannot_see_another_through_sysv(cfg):
    libc = ctypes.CDLL(None, use_errno=True)
    mine, left = 0x5A170000 | (os.getpid() & 0xFFFF), 0x5A180000 | (os.getpid() & 0xFFFF)
    sid = libc.shmget(mine, 4096, IPC_CREAT | 0o600)
    assert sid >= 0
    try:
        src = plugin("sysv", f"""
            import ctypes, errno
            libc = ctypes.CDLL(None, use_errno=True)
            out = []
            for key, size, flags in (({mine}, 0, 0), ({left}, 4096, {IPC_CREAT | 0o600}), ({left}, 0, 0)):
                r = libc.shmget(key, size, flags)
                out.append("OK" if r >= 0 else errno.errorcode[ctypes.get_errno()])
            raise RuntimeError(" ".join(out))""")
        for _ in range(2):                            # the second worker finds nothing left by the first
            e = plugin_error(client.fresh_run, cfg, src, "X", "1h", make_bars(n=50), {}, EngineParams())
            assert e.message == "EPERM EPERM EPERM"
        assert libc.shmget(left, 0, 0) == -1 and ctypes.get_errno() == errno.ENOENT
    finally:
        libc.shmctl(sid, IPC_RMID, None)


PARENT = """
import os, sys
from pathlib import Path
from perp_lab.config import load_config
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client
w = client.spawn(load_config(sys.argv[1]), PluginSource.from_text("donchian_atr", Path(sys.argv[2]).read_bytes()))
print(w.pid, flush=True)
os._exit(0)                                           # a crash: no close(), the pipes just go away
"""


def _running(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat[stat.rfind(")") + 2] not in "ZX"


@pytest.mark.sandbox
def test_worker_exits_when_its_parent_goes_away(cfg):
    r = subprocess.run([sys.executable, "-c", PARENT, str(cfg.path), str(REPO / "strategies" / "donchian_atr.py")],
                       cwd=REPO, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    pid = int(r.stdout.split()[0])
    try:
        deadline = time.monotonic() + 5
        while _running(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _running(pid)
    finally:
        if _running(pid):
            os.kill(pid, 9)


SLEEPER_PARENT = """
import sys, threading, time
from pathlib import Path
from perp_lab.config import load_config
from perp_lab.engine.backtest import EngineParams
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client
from perp_lab.validate.causality import synthetic_bars
w = client.spawn(load_config(sys.argv[1]), PluginSource.from_text("sleeper", Path(sys.argv[2]).read_bytes()))
w.bars("X", "1h", synthetic_bars(n=50))
threading.Thread(target=w.run, args=({}, EngineParams()), daemon=True).start()
while b"sleeping" not in w.stderr:                    # the worker is inside prepare(), not reading
    time.sleep(0.01)
print(w.pid, flush=True)
time.sleep(600)
"""


@pytest.mark.sandbox
def test_a_busy_worker_dies_with_its_parent(cfg, tmp_path):
    # EOF can't reach a worker that is busy in plugin code; the parent-death signal must.
    path = tmp_path / "sleeper.py"
    path.write_bytes(plugin("sleeper", 'print("sleeping", flush=True)\ntime.sleep(600)').text)
    helper = subprocess.Popen([sys.executable, "-c", SLEEPER_PARENT, str(cfg.path), str(path)], cwd=REPO,
                              stdout=subprocess.PIPE, text=True)
    pid = None
    try:
        assert select.select([helper.stdout], [], [], 60)[0], "the helper never reported its worker"
        pid = int(helper.stdout.readline())
        assert _running(pid)
        helper.kill()                                 # SIGKILL: no close(), no atexit, no finalizer
        helper.wait()
        deadline = time.monotonic() + 5
        while _running(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _running(pid)
    finally:
        helper.kill()
        helper.wait()
        helper.stdout.close()
        if pid is not None and _running(pid):
            os.kill(pid, 9)


def test_a_worker_whose_parent_is_already_gone_exits_at_once(cfg):
    # The parent died before the worker set its parent-death signal: getppid() is no longer it.
    r = subprocess.run([sys.executable, "-s", "-P", "-m", "perp_lab.sandbox.worker", "--deny-probe", str(cfg.path),
                        "--memory-mb", "3072", "--cpu-s", "60", "--parent-pid", str(os.getpid() + 1_000_000)],
                       env=client.ENV, cwd="/", capture_output=True, timeout=60)
    assert r.returncode == worker.ORPHANED_EXIT and r.stdout == b"", r.stderr


def test_check_raises_when_confinement_fails(cfg, monkeypatch):
    # The deny-probe is a file the worker is allowed to read, so its self-test must fail.
    readable = Config(raw=cfg.raw, path=Path(confine.__file__))
    with pytest.raises(SandboxUnavailable) as e:
        sandbox_fixtures.REAL_CHECK(readable)
    if confine.probe() is None:
        assert "deny-probe" in str(e.value)
    monkeypatch.setattr(confine, "probe", lambda min_abi=6: "Landlock ABI 5 < 6")
    with pytest.raises(SandboxUnavailable, match="Landlock ABI 5 < 6"):
        sandbox_fixtures.REAL_CHECK(cfg)


def test_max_workers_caps_by_memory(cfg, monkeypatch):
    monkeypatch.setattr(client, "_physical_mb", lambda: 30 * 1024)
    assert cfg["sandbox"]["memory_mb"] == 3072
    assert client.max_workers(cfg) == 7
    monkeypatch.setattr(client, "_physical_mb", lambda: 2048)
    assert client.max_workers(cfg) == 1


@pytest.mark.sandbox
def test_oversized_output_is_a_protocol_error(cfg):
    src = plugin("big", 'return pd.DataFrame({f"c{j}": bars.close for j in range(100)}, index=bars.index)')
    e = plugin_error(client.fresh_run, sandboxed(cfg, max_output_mb=1), src, "X", "1h", make_bars(), {},
                     EngineParams(), features=True)
    assert e.kind == "protocol"


@pytest.mark.sandbox
def test_a_forged_error_reply_is_sanitised(cfg):
    # The plugin writes its own error frame to the protocol descriptor (fd 4), full of text that
    # must never reach a log or a verdict as it is.
    src = plugin("forges", r"""
        import json, struct, sys
        sys.stderr.write("\x1b]0;title\x07 tail é\n")
        sys.stderr.flush()
        h = json.dumps({"error": {"kind": "exception", "type": "Evil\x1b[2J" + "é" * 300,
                                  "message": "\ud800\n" + "x" * 5000}, "sections": []}).encode()
        os.write(4, struct.pack(">I", len(h)) + h)
        time.sleep(30)""")
    e = plugin_error(batch_run, cfg, src)
    assert e.kind == "exception"
    assert printable(e.type) and len(e.type) <= 200 and e.type.startswith("Evil")
    assert printable(e.message) and len(e.message) == 200 and e.message.endswith("x" * 190)
    assert all(c in b"\t\n" or 0x20 <= c < 0x7F for c in e.stderr)


def test_error_reply_is_printable_and_capped():
    r = session.error_reply(ValueError("é\x1b\n" + "x" * 500))["error"]
    assert r == {"kind": "exception", "type": "ValueError", "message": "?? " + "x" * 197}
    r = session.error_reply(PluginError("features", "", "column 'a' has dtype object"))["error"]
    assert r == {"kind": "features", "type": "", "message": "column 'a' has dtype object"}
