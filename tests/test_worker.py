"""The confined worker and its trusted client: batch runs, limits, failures and confinement seen
through a real worker. Tests marked `sandbox` spawn real workers; the rest run on the twin."""

import ast
import ctypes
import errno
import json
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


SPIN = """
t = time.process_time()
while time.process_time() - t < self.p["spin"]:
    pass
return bars[[]]
"""


def check_prepare(cfg, make):
    """`prepare` on the held bars: the CPU it took, measured outside the plugin; plugin errors as usual."""
    with make(cfg, plugin("spinner", SPIN, params='{"spin": 0.0}')) as w:
        with pytest.raises(RuntimeError, match="send the bars"):
            w.prepare({})
        w.bars("X", "1h", make_bars())
        assert 0.4 <= w.prepare({"spin": 0.5}) < 5
        e = plugin_error(w.prepare, {"nope": 1})
        assert (e.kind, e.type) == ("exception", "ValueError") and "unknown params" in e.message
    with make(cfg, plugin("no_frame", "return None")) as w:      # the engine's arrays are built too
        w.bars("X", "1h", make_bars())
        e = plugin_error(w.prepare, {})
        assert (e.kind, e.type) == ("exception", "AttributeError")


def test_the_twin_times_prepare(cfg):
    check_prepare(cfg, InProcessWorker)


@pytest.mark.sandbox
def test_a_worker_times_prepare_from_outside(cfg):
    check_prepare(cfg, client.spawn)


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
def test_a_plugin_can_neither_mint_a_token_nor_read_holdout_bars(cfg, tmp_path):
    # The worker never loads perp_lab.holdout (where _MINT_KEY lives), and once sealed it can't
    # load anything, so no token can be minted inside it; the holdout bars can't be opened anyway.
    token = issue_holdout_token("lin", "test", tmp_path / "log.jsonl")
    assert len(load_candles(cfg, "BTCUSDT", "4h", holdout=token)) > 0   # the bars are there to read
    parquet = cfg.data_dir / "BTCUSDT" / "4h" / "all.parquet"
    src = plugin("forger", f"""
        import errno
        out = []
        try:
            import perp_lab.holdout
            out.append("imported")
        except ImportError as e:
            out.append(type(e).__name__)
        try:
            open({str(parquet)!r}, "rb").read(1)
            out.append("read")
        except OSError as e:
            out.append(type(e).__name__ + ":" + errno.errorcode[e.errno])
        raise RuntimeError(" ".join(out))""")
    e = plugin_error(batch_run, cfg, src)
    assert e.message == "ModuleNotFoundError PermissionError:EPERM", e.reason()


@pytest.mark.sandbox
def test_a_sealed_plugin_can_open_stat_or_readlink_nothing(cfg):
    # The allowlist is readable while the worker loads its modules; once sealed, no path resolves.
    stdlib, ours, link = json.__file__, REPO / "perp_lab" / "strategy.py", Path(sys.base_prefix, "bin", "python3")
    assert Path(stdlib).is_file() and ours.is_file() and link.is_symlink()
    src = plugin("sealed", f"""
        import errno
        out = []
        for fn in (lambda: open({stdlib!r}, "rb"), lambda: open({str(ours)!r}, "rb"),
                   lambda: os.stat({stdlib!r}), lambda: os.lstat({str(ours)!r}),
                   lambda: os.readlink({str(link)!r}), lambda: os.listdir({str(ours.parent)!r})):
            try:
                fn()
                out.append("OK")
            except OSError as e:
                out.append(errno.errorcode[e.errno])
        raise RuntimeError(" ".join(out))""")
    e = plugin_error(batch_run, cfg, src)
    assert e.message == "EPERM EPERM EPERM EPERM EPERM EPERM", e.reason()


OLD_NS = 10**18                                       # 2001-09-09: older than any relatime window


@pytest.mark.sandbox
def test_a_plugin_moves_no_access_time(cfg, tmp_path):
    # On a relatime mount, reading a file, and following or reading a symlink, each update its
    # access time when that is old: a bit per file per day a worker could leave for a later one,
    # which anyone reads back with stat(). Checked from outside, on a scratch file inside the
    # allowlist (the virtualenv: this user's, and no repo file) and on a symlink to it.
    target = Path(sys.prefix) / f"perp_lab_atime_test_{os.getpid()}_{time.monotonic_ns()}.txt"
    link = tmp_path / "link"
    link.symlink_to(target)

    def age():
        os.utime(target, ns=(OLD_NS, OLD_NS))
        os.utime(link, ns=(OLD_NS, OLD_NS), follow_symlinks=False)

    def atimes():
        return os.lstat(link).st_atime_ns, os.stat(target).st_atime_ns
    try:
        target.write_text("x")
        age()
        target.read_bytes()                               # a read from outside moves both: the
        os.readlink(link)                                 # mount does record access times here
        assert OLD_NS not in atimes()
        age()
        src = plugin("toucher", f"""
            import errno
            out = []
            for fn in (lambda: os.stat({str(link)!r}), lambda: os.readlink({str(link)!r}),
                       lambda: open({str(link)!r}, "rb").read(1), lambda: open({str(target)!r}, "rb").read(1)):
                try:
                    fn()
                    out.append("OK")
                except OSError as e:
                    out.append(errno.errorcode[e.errno])
            raise RuntimeError(" ".join(out))""")
        e = plugin_error(batch_run, cfg, src)
        assert atimes() == (OLD_NS, OLD_NS)
        assert e.message == "EPERM EPERM EPERM EPERM", e.reason()
    finally:
        target.unlink(missing_ok=True)


# The modules a plugin may import (perp_lab/strategy.py's contract), and the ones they load.
PLUGIN_IMPORTS = ["math", "numpy", "pandas", "perp_lab.strategy", "perp_lab.indicators", "perp_lab.engine.backtest"]


@pytest.mark.sandbox
def test_a_plugin_can_import_only_what_the_worker_preloaded(cfg):
    src = plugin("importer", f"""
        import sys
        for name in {PLUGIN_IMPORTS!r}:
            __import__(name)
        for sub in ("char", "ctypeslib", "fft", "polynomial", "rec", "strings"):
            getattr(np, sub)                          # numpy 2 loads these on first use
        print(bars.head())                            # pandas loads its frame formatter on first use
        assert "colorsys" not in sys.modules          # standard library, on the allowlist, never loaded
        import colorsys
        return bars[[]]""")
    e = plugin_error(batch_run, cfg, src)
    assert (e.kind, e.type) == ("exception", "ModuleNotFoundError") and "colorsys" in e.message, e.reason()
    assert e.reason().startswith("plugin_error: exception: ModuleNotFoundError: ")


@pytest.mark.sandbox
def test_a_sealed_plugin_can_neither_see_nor_steer_the_page_cache(cfg):
    # Which pages of a mapped library are resident is a channel to a later reader (mincore, or
    # fault timing): no residency query, no eviction or prefetch advice. MADV_HUGEPAGE is prefetch
    # on a file mapping (a fault reads a whole 2 MiB folio in, which a live mapping of any of its
    # pages then keeps resident past the eviction). Freeing memory and plain allocation still work,
    # large arrays included: numpy's MADV_HUGEPAGE hint on them is refused, and numpy ignores that.
    src = plugin("pagecache", """
        import ctypes, errno
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
        page = 4096
        lib = ctypes.cast(libc.mmap, ctypes.c_void_p).value & ~(page - 1)    # a page of libc: file-backed
        anon = libc.mmap(None, 4 * page, 3, 0x22, -1, 0)                     # read-write, private, anonymous

        def call(fn, *args):
            return "OK" if fn(*args) == 0 else errno.errorcode[ctypes.get_errno()]
        out = [call(libc.mincore, lib, page, ctypes.create_string_buffer(1))]
        out += [call(libc.madvise, lib, page, a) for a in (21, 20, 3, 14)]  # PAGEOUT, COLD, WILLNEED, HUGEPAGE
        out += [call(libc.madvise, anon, page, a) for a in (22, 23)]        # POPULATE_READ, POPULATE_WRITE
        out += [call(libc.madvise, anon, page, 4)]                          # DONTNEED
        out.append("OK" if np.ones(1 << 23).sum() == 1 << 23 else "BAD")    # 64 MiB, allocated and freed
        raise RuntimeError(" ".join(out))""")
    e = plugin_error(batch_run, cfg, src)
    assert e.message == "EPERM EPERM EPERM EPERM EPERM EPERM EPERM OK OK", e.reason()


@pytest.mark.sandbox
def test_pandas_methods_that_load_code_lazily_work_in_a_sealed_worker(cfg):
    src = plugin("lazy", """
        f = bars[["close", "open"]].head()
        out = []
        for fn in (lambda: f.to_dict(), lambda: f.stack().unstack(), lambda: f["close"].to_frame().unstack(),
                   lambda: pd.crosstab(bars.regime, bars.regime), lambda: f.to_csv()):
            try:
                fn()
                out.append("ok")
            except Exception as e:
                out.append(f"{type(e).__name__}:{e}")
        raise RuntimeError(" ".join(out))""")
    e = plugin_error(batch_run, cfg, src)
    assert e.message == "ok ok ok ok ok", e.reason()


PRELOADED = """
import json, sys
from perp_lab.sandbox import worker
worker._preload()
print(json.dumps(sorted(sys.modules)))
"""
# Modules numpy and pandas import inside functions that the worker does not preload, each with why
# a plugin doesn't need it. Anything else such a function imports must be preloaded: a sealed
# worker can't load it, so the method would fail with ModuleNotFoundError.
NOT_PRELOADED = {
    "numpy.core": "deprecated alias of numpy._core",
    "numpy.f2py": "Fortran wrapper generator",
    "numpy.matlib": "deprecated matrix library",
    "numpy.testing": "test helpers",
    "numpy.lib.recfunctions": "imported only by numpy.testing",
    "numpy._core._multiarray_tests": "numpy's own tests",
    "pandas._version": "pd.show_versions()",
    "pandas.core._numba": "the numba engine (numba is not installed)",
    "pandas.core.arrays.arrow": "Arrow-backed extension dtypes, from Arrow data a plugin never gets",
    "pandas.core.arrays.sparse.scipy_sparse": "scipy (not installed)",
    "pandas.core.interchange": "the dataframe interchange protocol, for other libraries",
    "pandas.io.clipboard": "clipboard I/O",
    "pandas.io.formats.excel": "Excel I/O (openpyxl)",
    "pandas.io.formats.html": "HTML output",
    "pandas.io.formats.style": "Styler (jinja2)",
    "pandas.io.formats.xml": "XML output (lxml)",
    "pandas.io.sas": "SAS file I/O",
    "pandas.plotting": "plotting (matplotlib)",
}


def _lazy_imports(package: str, loaded: set[str]) -> dict[str, list[str]]:
    """Modules of `package` imported inside its functions and not in `loaded`: {module: [where]}."""
    root = Path(__import__(package).__file__).parent

    def is_module(name):
        base = root.parent.joinpath(*name.split("."))
        return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file() \
            or any(base.parent.glob(base.name + ".*.so"))
    found: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        if "tests" in path.relative_to(root).parts:
            continue
        here = ".".join(path.relative_to(root.parent).with_suffix("").parts).removesuffix(".__init__")
        package_of_here = here if path.name == "__init__.py" else here.rpartition(".")[0]
        for fn in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Import):
                    targets = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    if node.level:
                        pkg = package_of_here
                        for _ in range(node.level - 1):
                            pkg = pkg.rpartition(".")[0]
                        base = f"{pkg}.{base}" if base else pkg
                    targets = [base] + [f"{base}.{a.name}" for a in node.names if is_module(f"{base}.{a.name}")]
                else:
                    continue
                for t in targets:
                    if t.split(".")[0] == package and t not in loaded:
                        found.setdefault(t, []).append(f"{path.relative_to(root)}:{node.lineno}")
    return found


@pytest.mark.parametrize("package", ["numpy", "pandas"])
def test_the_preload_leaves_no_lazy_import_a_plugin_could_hit(package):
    # A fresh interpreter launched as the worker is: what it holds after _preload() is what a
    # sealed worker has. (Confinement doesn't change which modules load, so none is needed here.)
    r = subprocess.run([sys.executable, "-s", "-P", "-c", PRELOADED], env=client.ENV, cwd="/", capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    lazy = _lazy_imports(package, set(json.loads(r.stdout)))
    unexplained = {m: w[:3] for m, w in lazy.items()
                   if not any(m == x or m.startswith(x + ".") for x in NOT_PRELOADED)}
    assert not unexplained, unexplained


NO_SEAL = """
import sys
from perp_lab.sandbox import confine, worker
confine.seal = lambda: None                           # a seal that silently did nothing
sys.exit(worker.main(sys.argv[1:]))
"""


@pytest.mark.sandbox
def _unavailable(r) -> str:
    """The message of the "unavailable" error a worker run by hand reported, or the test fails."""
    assert r.returncode == worker.UNAVAILABLE_EXIT, (r.returncode, r.stderr)
    n = int.from_bytes(r.stdout[:4], "big")
    error = json.loads(r.stdout[4:4 + n])["error"]
    assert error["kind"] == "unavailable", error
    return error["message"]


def _by_hand(cfg, *python_args, stdin=subprocess.DEVNULL):
    return subprocess.run([sys.executable, "-s", "-P", *python_args, "--deny-probe", str(cfg.path), "--memory-mb",
                           "3072", "--cpu-s", "60", "--parent-pid", str(os.getpid())],
                          env=client.ENV, cwd="/", stdin=stdin, capture_output=True, timeout=120)


@pytest.mark.sandbox
def test_a_worker_whose_seal_did_not_take_is_unavailable(cfg):
    assert "sealed open" in _unavailable(_by_hand(cfg, "-c", NO_SEAL))


@pytest.mark.sandbox
def test_a_sealed_worker_holding_anything_but_pipes_is_unavailable(cfg, tmp_path):
    # A descriptor on a file would let it read the file (moving its access time) without opening
    # anything. Here the protocol's input is a file instead of the parent's pipe.
    held = tmp_path / "held.txt"
    held.write_bytes(b"x")
    with open(held, "rb") as f:
        message = _unavailable(_by_hand(cfg, "-m", "perp_lab.sandbox.worker", stdin=f))
    assert "fd 3 is not a pipe" in message, message


@pytest.mark.sandbox
def test_the_seal_comes_before_any_plugin_code(cfg):
    # Code at a plugin's module level runs as the worker loads it: sealed already.
    src = plugin("early", "raise RuntimeError(AT_LOAD)", tail="""
        import errno
        try:
            open(np.__file__, "rb").close()
            AT_LOAD = "OK"
        except OSError as e:
            AT_LOAD = errno.errorcode[e.errno]
        """)
    assert plugin_error(batch_run, cfg, src).message == "EPERM"


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


@pytest.mark.sandbox
def test_a_plugin_cannot_change_file_metadata(cfg, tmp_path):
    # Mode bits, owner, timestamps and extended attributes would each hold bytes for a later worker.
    target = tmp_path / "scratch.txt"                 # outside the allowlist, owned by this user
    target.write_text("x")
    os.chmod(target, 0o644)
    os.utime(target, ns=(10**18, 10**18))
    os.setxattr(target, "user.kept", b"v")
    before = os.stat(target)
    src = plugin("meta", f"""
        import ctypes, errno
        T = {str(target)!r}
        libc = ctypes.CDLL(None, use_errno=True)
        L = ctypes.c_long

        def setxattrat():                             # 463 on every architecture; newer than libseccomp 2.5
            value = ctypes.create_string_buffer(b"y")
            xattr_args = (ctypes.c_uint64 * 2)(ctypes.addressof(value), 1)    # value, size 1, flags 0
            if libc.syscall(L(463), L(-100), T.encode(), L(0), b"user.at", xattr_args, L(16)) != 0:
                raise OSError(ctypes.get_errno(), "setxattrat")
        out = []
        for fn in (lambda: os.chmod(T, 0o600), lambda: os.chown(T, os.getuid(), os.getgid()),
                   lambda: os.utime(T, (1_500_000_000, 1_500_000_000)), lambda: os.setxattr(T, "user.new", b"x"),
                   lambda: os.removexattr(T, "user.kept"), lambda: os.getxattr(T, "user.kept"),
                   lambda: os.listxattr(T), setxattrat):
            try:
                fn()
                out.append("OK")
            except OSError as e:
                out.append(errno.errorcode[e.errno])
        raise RuntimeError(" ".join(out))""")
    e = plugin_error(batch_run, cfg, src)
    assert e.message == "EPERM EPERM EPERM EPERM EPERM EPERM EPERM ENOSYS", e.reason()
    after = os.stat(target)
    assert (after.st_mode, after.st_uid, after.st_gid) == (before.st_mode, before.st_uid, before.st_gid)
    assert (after.st_mtime_ns, after.st_ctime_ns) == (before.st_mtime_ns, before.st_ctime_ns)   # chown moves ctime
    assert os.listxattr(target) == ["user.kept"] and os.getxattr(target, "user.kept") == b"v"


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


def test_a_reply_that_fails_to_parse_is_logged_with_its_traceback(caplog):
    # A parent-side bug in a parser must not be charged to the plugin unseen: the plugin gets a
    # protocol error, and the original exception and traceback go to the parent's log.
    class Fake:
        pid = 4242

        def _fail(self, exc):
            return exc

    def parse():
        raise ValueError("a parser bug near \x1b[2J\u00e9")      # it may quote what the worker sent
    with caplog.at_level("WARNING", logger="perp_lab.sandbox.client"):
        with pytest.raises(PluginError) as e:
            client.Worker._parsed(Fake(), "run", parse)
    assert e.value.kind == "protocol" and e.value.message == "malformed run reply (ValueError)"
    [rec] = [r for r in caplog.records if r.name == "perp_lab.sandbox.client"]
    text = rec.getMessage()
    assert "4242" in text and "Traceback" in text and "in parse" in text
    assert "ValueError: a parser bug near ?[2J??" in text           # printable only


def test_mapped_files_that_cannot_be_read_are_logged(caplog):
    # A worker whose /proc/<pid>/maps can't be read at ready would have nothing evicted after its
    # exam: a warning. At close it is only a note: a sealed worker maps nothing new, so what ready
    # noted is still evicted, and a plugin can make its maps unreadable at will (PR_SET_DUMPABLE 0).
    class Fake:
        pid = -1                                  # /proc/-1 doesn't exist
        _mapped = {"/usr/lib/libfake.so"}
    with caplog.at_level("INFO", logger="perp_lab.sandbox.client"):
        client.Worker._note_mapped(Fake(), "ready")
        client.Worker._note_mapped(Fake(), "close")
    ready, close = [r for r in caplog.records if r.name == "perp_lab.sandbox.client"]
    text = ready.getMessage()
    assert ready.levelname == "WARNING" and "worker -1" in text and "ready" in text and "won't be evicted" in text, text
    text = close.getMessage()
    assert close.levelname == "INFO" and "worker -1" in text and "close" in text, text
    assert "the 1 file noted at ready is still evicted" in text, text


def test_error_reply_is_printable_and_capped():
    r = session.error_reply(ValueError("é\x1b\n" + "x" * 500))["error"]
    assert r == {"kind": "exception", "type": "ValueError", "message": "?? " + "x" * 197}
    r = session.error_reply(PluginError("features", "", "column 'a' has dtype object"))["error"]
    assert r == {"kind": "features", "type": "", "message": "column 'a' has dtype object"}
