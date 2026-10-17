"""Confinement: every escape a plugin could try fails, checked from outside where the test can.

Each test runs tests/confine_probe.py in a fresh interpreter launched the way the worker is
(`-s -P`, the spec's environment, cwd /, a new session), since confinement can't be undone.
"""

import ctypes
import errno
import os
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


def _run(name, target=""):
    cmd = [sys.executable, "-s", "-P", str(PROBE)]
    for p in confine.allowlist():
        cmd += ["--allow", p]
    cmd += ["--target", str(target), name]
    return subprocess.run(cmd, env=ENV, cwd="/", close_fds=True, start_new_session=True,
                          capture_output=True, text=True, timeout=120)


def run_probe(name, target=""):
    r = _run(name, target)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.splitlines()


def _queue_left(name):
    """Whether the POSIX message queue `name` exists, seen from this (unconfined) process."""
    if Path("/dev/mqueue", name[1:]).exists():
        return True
    libc = ctypes.CDLL(None, use_errno=True)
    q = libc.mq_open(name.encode(), os.O_RDONLY, 0, None)
    if q >= 0:
        libc.mq_close(q)
    return q >= 0 or ctypes.get_errno() != errno.ENOENT


EXPECTED = {
    "read_repo_file": "DENIED EACCES",            # Landlock: the repo root is outside the allowlist
    "write_tmp": "DENIED EACCES",                 # Landlock: nothing is writable
    "tcp_connect": "DENIED EPERM",                # seccomp: socket()
    "udp_send": "DENIED EPERM",
    "unix_connect": "DENIED EPERM",
    "fork": "DENIED EPERM",                       # seccomp: clone without CLONE_THREAD
    "clone3": "DENIED ENOSYS",                    # seccomp: clone3, so libc falls back to clone
    "subprocess": "DENIED EPERM",
    "shmget": "DENIED EPERM",                     # seccomp: System V IPC
    "prlimit_parent": "DENIED EPERM",             # seccomp: prlimit64 aimed at another pid
    "kill_parent": "DENIED EPERM",                # Landlock: signal scope
    "sched_getaffinity_parent": "DENIED EPERM",   # seccomp: sched_* aimed at another pid
    "setpriority_pgrp": "DENIED EPERM",           # seccomp: which must be PRIO_PROCESS
    "ioprio_get_user": "DENIED EPERM",            # seccomp: IOPRIO_WHO_USER, likewise
    "mq_open": "DENIED EPERM",                    # seccomp: POSIX message queues
    "inotify_init1": "DENIED EPERM",              # seccomp: Landlock has no right for file watches
    "pdeathsig_clear": "DENIED EPERM",            # seccomp: a worker must die with its parent
    "pdeathsig_clear_wide": "DENIED EPERM",       # seccomp: the option's low 32 bits are what count
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
    elif probe == "mq_open":
        target = f"/perp_lab_confine_{os.getpid()}"
        assert not _queue_left(target)
    old_term = signal.signal(signal.SIGTERM, lambda *_: signals.append("SIGTERM"))
    nofile, nice = resource.getrlimit(resource.RLIMIT_NOFILE), os.getpriority(os.PRIO_PROCESS, 0)
    queue_left = False
    try:
        assert run_probe(probe, target)[-1] == EXPECTED[probe]
    finally:
        signal.signal(signal.SIGTERM, old_term)
        now = resource.getrlimit(resource.RLIMIT_NOFILE)
        if now != nofile:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(nofile[0], now[1]), now[1]))
        if probe == "mq_open" and (queue_left := _queue_left(target)):
            ctypes.CDLL(None).mq_unlink(target.encode())
    assert now == nofile                          # prlimit on the parent changed nothing
    assert os.getpriority(os.PRIO_PROCESS, 0) == nice
    assert not queue_left                         # a denied mq_open leaves no queue behind
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


def test_priority_of_this_process_still_works():
    assert run_probe("getpriority_self")[-1] == "ALLOWED"


def test_other_prctl_calls_still_work():
    assert run_probe("prctl_set_name")[-1] == "ALLOWED"


@pytest.mark.skipif(os.uname().machine != "x86_64", reason="x32 is an x86_64 ABI")
def test_a_syscall_through_another_abi_kills_the_process():
    # Made from a second thread: the kernel kills the whole process anyway when the last thread
    # makes a killed syscall, so only a second thread tells kill-process from kill-thread.
    r = _run("x32_syscall")
    assert r.returncode == -signal.SIGSYS, r.stdout + r.stderr
    assert r.stdout.splitlines() == ["CALLING"]


def test_probe_reports_none_here():
    assert confine.probe() is None


def test_probe_reports_a_low_abi(monkeypatch):
    monkeypatch.setattr(confine, "_landlock_abi", lambda: 5)
    assert confine.probe() == "Landlock ABI 5 < 6"
