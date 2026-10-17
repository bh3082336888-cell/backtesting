"""Run by tests/test_confine.py in a fresh interpreter: confine this process, then try one thing.

    python -s -P tests/confine_probe.py [--allow PATH]... [--target VALUE] PROBE

The last line printed is the result: `DENIED <errno name>`, `ALLOWED`, or `REFUSED` (confine()
raised). Before confine() runs, only the standard library and perp_lab.sandbox.confine are
imported; each probe imports what else it needs, so those imports happen under confinement too.
"""

import ctypes
import errno
import os
import resource
import sys
import threading
import time

from perp_lab.sandbox.confine import confine

libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long


def _ok(r):
    if r == -1:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e))
    return r


def _try_open(path):
    try:
        os.close(os.open(path, os.O_RDONLY))
        return "ALLOWED"
    except OSError as e:
        return "DENIED " + errno.errorcode[e.errno]


def read_repo_file(target, pre):
    with open(target, "rb") as f:
        f.read(1)


def write_tmp(target, pre):
    with open(target, "wb") as f:
        f.write(b"escaped")


def tcp_connect(target, pre):
    import socket
    socket.create_connection(("127.0.0.1", int(target)), timeout=5).close()


def udp_send(target, pre):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(b"escaped", ("127.0.0.1", int(target)))


def unix_connect(target, pre):
    import socket
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(target)


def fork(target, pre):
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)


def subprocess_(target, pre):
    import subprocess
    subprocess.run([sys.executable, "-c", "pass"], check=True)


def shmget(target, pre):
    sid = _ok(libc.shmget(0, 4096, 0o1000 | 0o600))   # IPC_PRIVATE, IPC_CREAT | 0600
    libc.shmctl(sid, 0, None)                          # IPC_RMID


def prlimit_parent(target, pre):
    soft, hard = pre["nofile"]                         # inherited from the parent, read before confine
    resource.prlimit(os.getppid(), resource.RLIMIT_NOFILE, (soft - 1, hard))


def kill_parent(target, pre):
    import signal
    os.kill(os.getppid(), signal.SIGTERM)


def sched_getaffinity_parent(target, pre):
    os.sched_getaffinity(os.getppid())


def read_pre_existing_thread(target, pre):
    pre["go"].set()                                    # only reached if confine() did not refuse
    pre["thread"].join(10)
    if pre["result"].startswith("DENIED "):
        raise OSError(getattr(errno, pre["result"].split()[1]), pre["result"])
    if pre["result"] != "ALLOWED":
        raise SystemExit(f"pre-existing thread: {pre['result']}")


def read_allowlisted(target, pre):
    if "json" in sys.modules:
        raise SystemExit("json was imported before confine(), so this probe proves nothing")
    import json
    json.loads("[1]")
    with open(target, "rb") as f:
        f.read(1)


def _in_each_thread(taskfd, path):
    """Open `path` from inside every thread of this process; {tid: (thread name, result)}.

    Native threads can block every signal (pyarrow's jemalloc background thread does), but glibc
    never lets a thread block its internal SIGSETXID (33). So: point 33 at a ctypes handler (its
    kernel sigaction copied from one glibc's signal() builds, for the restorer), tgkill each
    thread with it, and restore glibc's handler afterwards.
    """
    nr = {"x86_64": 13, "aarch64": 134}[os.uname().machine]   # rt_sigaction
    sigsetxid, sigusr2 = 33, 12
    results = {}

    @ctypes.CFUNCTYPE(None, ctypes.c_int)
    def handler(sig):
        name = ctypes.create_string_buffer(16)
        libc.prctl(16, name)                           # PR_GET_NAME, of the thread running this
        results[threading.get_native_id()] = (name.value.decode(), _try_open(path))

    def rt_sigaction(sig, new, old):
        _ok(libc.syscall(ctypes.c_long(nr), ctypes.c_int(sig), new, old, ctypes.c_size_t(8)))

    old_xid, old_usr2, ours = (ctypes.c_char * 64)(), (ctypes.c_char * 64)(), (ctypes.c_char * 64)()
    rt_sigaction(sigusr2, None, old_usr2)
    libc.signal.restype = ctypes.c_void_p
    libc.signal.argtypes = [ctypes.c_int, ctypes.c_void_p]
    libc.signal(sigusr2, ctypes.cast(handler, ctypes.c_void_p))
    rt_sigaction(sigusr2, None, ours)
    rt_sigaction(sigusr2, old_usr2, None)
    rt_sigaction(sigsetxid, None, old_xid)
    rt_sigaction(sigsetxid, ours, None)
    try:
        me = threading.get_native_id()
        for tid in sorted(int(t) for t in os.listdir(taskfd)):
            if tid == me:
                results[tid] = ("main", _try_open(path))
                continue
            _ok(libc.tgkill(os.getpid(), tid, sigsetxid))
            deadline = time.monotonic() + 10
            while tid not in results and time.monotonic() < deadline:
                time.sleep(0.01)
            results.setdefault(tid, ("?", "NO ANSWER"))
    finally:
        rt_sigaction(sigsetxid, old_xid, None)
    return results


def import_numpy_pandas_pyarrow(target, pre):
    import numpy as np
    import pandas as pd
    import pyarrow as pa

    import perp_lab.engine.backtest  # noqa: F401
    import perp_lab.indicators  # noqa: F401
    import perp_lab.strategy  # noqa: F401

    df = pd.DataFrame({"t": pd.date_range("2025-01-01", periods=500, freq="h", tz="UTC"),
                       "x": np.random.default_rng(0).normal(size=500)})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, pa.Schema.from_pandas(df)) as w:
        w.write_table(pa.Table.from_pandas(df))
    back = pa.ipc.open_stream(sink.getvalue()).read_all().to_pandas()
    assert back.equals(df) and float(np.ones((50, 50)) @ np.ones(50) @ np.ones(50)) == 2500.0
    threads = _in_each_thread(pre["taskfd"], target)
    for tid, (name, result) in sorted(threads.items()):
        print("THREAD", tid, name, result)
    print("THREADS", len(os.listdir(pre["taskfd"])))
    if {r for _, r in threads.values()} == {"DENIED EACCES"}:
        raise PermissionError(errno.EACCES, "every thread was denied")


PROBES = {f.__name__.rstrip("_"): f for f in (
    read_repo_file, write_tmp, tcp_connect, udp_send, unix_connect, fork, subprocess_, shmget,
    prlimit_parent, kill_parent, sched_getaffinity_parent, read_pre_existing_thread, read_allowlisted,
    import_numpy_pandas_pyarrow)}


def main(argv):
    allow, target = [], ""
    while len(argv) > 1:
        flag, value, argv = argv[0], argv[1], argv[2:]
        if flag == "--allow":
            allow.append(value)
        elif flag == "--target":
            target = value
        else:
            raise SystemExit(f"unknown flag {flag}")
    name, = argv
    probe = PROBES[name]

    pre = {"nofile": resource.getrlimit(resource.RLIMIT_NOFILE)}
    if name == "import_numpy_pandas_pyarrow":
        pre["taskfd"] = os.open("/proc/self/task", os.O_RDONLY | os.O_DIRECTORY)  # opened while allowed
    if name == "read_pre_existing_thread":
        pre["go"], pre["result"] = threading.Event(), "NO ANSWER"

        def reader():
            pre["go"].wait()
            pre["result"] = _try_open(target)
        pre["thread"] = threading.Thread(target=reader, daemon=True)
        pre["thread"].start()

    try:
        confine(memory_mb=1024, cpu_s=60, read_paths=allow)
    except RuntimeError as e:
        print("confine:", e, file=sys.stderr)
        print("REFUSED")
        return
    try:
        probe(target, pre)
    except OSError as e:
        print("DENIED", errno.errorcode.get(e.errno, e.errno))
        return
    print("ALLOWED")


if __name__ == "__main__":
    main(sys.argv[1:])
