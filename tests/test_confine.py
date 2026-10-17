"""Confinement: every escape a plugin could try fails, checked from outside where the test can.

Each test runs tests/confine_probe.py in a fresh interpreter launched the way the worker is
(`-s -P`, the spec's environment, cwd /, a new session), since confinement can't be undone.
"""

import resource
import signal
import site
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from perp_lab.sandbox import confine

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "tests" / "confine_probe.py"
ENV = {"LANG": "C.UTF-8", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}

_UNAVAILABLE = confine.probe()
pytestmark = [pytest.mark.sandbox,
              pytest.mark.skipif(_UNAVAILABLE is not None, reason=f"sandbox unavailable: {_UNAVAILABLE}")]


def run_probe(name, target=""):
    cmd = [sys.executable, "-s", "-P", str(PROBE)]
    for p in confine.allowlist():
        cmd += ["--allow", p]
    cmd += ["--target", str(target), name]
    r = subprocess.run(cmd, env=ENV, cwd="/", close_fds=True, start_new_session=True,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.splitlines()


EXPECTED = {
    "read_repo_file": "DENIED EACCES",            # Landlock: the repo root is outside the allowlist
    "write_tmp": "DENIED EACCES",                 # Landlock: nothing is writable
    "tcp_connect": "DENIED EPERM",                # seccomp: socket()
    "udp_send": "DENIED EPERM",
    "unix_connect": "DENIED EPERM",
    "fork": "DENIED EPERM",                       # seccomp: clone without CLONE_THREAD
    "subprocess": "DENIED EPERM",
    "shmget": "DENIED EPERM",                     # seccomp: System V IPC
    "prlimit_parent": "DENIED EPERM",             # seccomp: prlimit64 aimed at another pid
    "kill_parent": "DENIED EPERM",                # Landlock: signal scope
    "sched_getaffinity_parent": "DENIED EPERM",   # seccomp: sched_* aimed at another pid
    "read_pre_existing_thread": "REFUSED",        # confine() must be the only thread
}


@pytest.mark.parametrize("probe", list(EXPECTED))
def test_each_escape_is_denied(probe, tmp_path):
    target, listener, signals = "", None, []
    if probe in ("read_repo_file", "read_pre_existing_thread"):
        target = REPO / "README.md"
        assert target.is_file()
    elif probe == "write_tmp":
        target = tmp_path / "sentinel"
    elif probe in ("tcp_connect", "unix_connect"):
        listener = socket.socket(socket.AF_UNIX if probe == "unix_connect" else socket.AF_INET)
        listener.bind(str(tmp_path / "s.sock") if probe == "unix_connect" else ("127.0.0.1", 0))
        listener.listen()
        target = listener.getsockname() if probe == "unix_connect" else listener.getsockname()[1]
    elif probe == "udp_send":
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(("127.0.0.1", 0))
        target = listener.getsockname()[1]
    old_term = signal.signal(signal.SIGTERM, lambda *_: signals.append("SIGTERM"))
    nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        assert run_probe(probe, target)[-1] == EXPECTED[probe]
    finally:
        signal.signal(signal.SIGTERM, old_term)
        now = resource.getrlimit(resource.RLIMIT_NOFILE)
        if now != nofile:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(nofile[0], now[1]), now[1]))
    assert now == nofile                          # prlimit on the parent changed nothing
    assert signals == []                          # without the handler, SIGTERM would have ended this run
    assert not (tmp_path / "sentinel").exists()
    if listener is not None:
        listener.setblocking(False)
        with pytest.raises(BlockingIOError):      # no connection, no datagram
            listener.recv(64) if probe == "udp_send" else listener.accept()
        listener.close()


def test_reading_inside_the_allowlist_still_works():
    import numpy
    target = Path(numpy.__file__)
    assert any(target.is_relative_to(p) for p in site.getsitepackages())
    assert run_probe("read_allowlisted", target)[-1] == "ALLOWED"


def test_numpy_pandas_pyarrow_import_after_confinement():
    out = run_probe("import_numpy_pandas_pyarrow", REPO / "README.md")
    threads = [line for line in out if line.startswith("THREAD ")]
    count = int(next(line for line in out if line.startswith("THREADS ")).split()[1])
    # The main thread plus at least pyarrow's allocator thread, which blocks every signal it can.
    assert len(threads) == count >= 2, out
    assert all(line.endswith(" DENIED EACCES") for line in threads), out
    assert out[-1] == "DENIED EACCES"


def test_probe_reports_none_here():
    assert confine.probe() is None


def test_probe_reports_a_low_abi(monkeypatch):
    monkeypatch.setattr(confine, "_landlock_abi", lambda: 5)
    assert confine.probe() == "Landlock ABI 5 < 6"
