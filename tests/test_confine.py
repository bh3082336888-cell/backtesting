"""Confinement: every escape a plugin could try fails, checked from outside where the test can.

Each test runs tests/confine_probe.py in a fresh interpreter launched the way the worker is
(`-s -P`, the spec's environment, cwd /, a new session), since confinement can't be undone. Most
probes apply stage 1 only (`confine()`); a worker also seals itself (`seal()`) before plugin code
runs, which tests/test_worker.py checks through real workers.
"""

import ctypes
import errno
import os
import resource
import signal
import site
import socket
import struct
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
    "flock": "DENIED EPERM",                      # seccomp: a lock on a shared readable file is a channel
    "fcntl_setlk": "DENIED EPERM",
    "fcntl_ofd_setlk": "DENIED EPERM",
    "fcntl_notify": "DENIED EPERM",               # seccomp: a directory watch, like inotify
    "set_inode_flags": "DENIED EPERM",            # seccomp: chattr on a file it may open
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
    elif probe in ("flock", "fcntl_setlk", "fcntl_ofd_setlk"):
        target = Path(socket.__file__)            # on the allowlist: the open works, the lock doesn't
    elif probe == "fcntl_notify":
        target = Path(socket.__file__).parent
    elif probe == "set_inode_flags":
        target = REPO / "perp_lab" / "__init__.py"   # this user's, on the allowlist
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


def test_the_seal_reaches_every_thread():
    # seal() runs once numpy and pyarrow have started their threads; seccomp's thread sync must
    # reach those too, or plugin code could borrow one to open files (through ctypes).
    import numpy
    out = run_probe("seal_every_thread", Path(numpy.__file__))
    threads = [line for line in out if line.startswith("THREAD ")]
    count = int(next(line for line in out if line.startswith("THREADS ")).split()[1])
    assert len(threads) == count >= 2, out
    assert all(line.endswith(" DENIED EPERM") for line in threads), out
    assert out[-1] == "DENIED EPERM"


def test_priority_of_this_process_still_works():
    assert run_probe("getpriority_self")[-1] == "ALLOWED"


def test_other_prctl_calls_still_work():
    assert run_probe("prctl_set_name")[-1] == "ALLOWED"


def test_other_fcntl_calls_still_work():
    assert run_probe("fcntl_other", Path(socket.__file__))[-1] == "ALLOWED"


@pytest.mark.skipif(os.uname().machine != "x86_64", reason="x32 is an x86_64 ABI")
def test_a_syscall_through_another_abi_kills_the_process():
    # Made from a second thread: the kernel kills the whole process anyway when the last thread
    # makes a killed syscall, so only a second thread tells kill-process from kill-thread.
    r = _run("x32_syscall")
    assert r.returncode == -signal.SIGSYS, r.stdout + r.stderr
    assert r.stdout.splitlines() == ["CALLING"]


# ---------------------------------------------------------------- the whole filter, rule by rule

ALLOW, KILL_PROCESS = 0x7FFF0000, 0x80000000
EPERM, ENOSYS = 0x00050000 | errno.EPERM, 0x00050000 | errno.ENOSYS
JUNK = 0xDEAD_BEEF_0000_0000              # set above an argument's low 32 bits
# The spec's always-denied calls (Confinement, step 5).
DENIED = """
    socket socketpair execve execveat fork vfork
    shmget shmat shmctl shmdt msgget msgsnd msgrcv msgctl semget semop semtimedop semctl
    mq_open mq_unlink mq_timedsend mq_timedreceive mq_notify mq_getsetattr
    inotify_init inotify_init1 fanotify_init
    ptrace process_vm_readv process_vm_writev process_madvise kcmp pidfd_open pidfd_getfd migrate_pages move_pages
    mount umount2 pivot_root chroot unshare setns bpf perf_event_open userfaultfd
    io_uring_setup io_uring_enter io_uring_register keyctl add_key request_key
    init_module finit_module delete_module kexec_load kexec_file_load
    chmod fchmod fchmodat chown fchown lchown fchownat utime utimes utimensat futimesat
    setxattr lsetxattr fsetxattr removexattr lremovexattr fremovexattr
    getxattr lgetxattr fgetxattr listxattr llistxattr flistxattr
    flock""".split()
# File-metadata calls newer than some libseccomp releases. Since 424, a new syscall has one number on
# every architecture.
NEWER = {"fchmodat2": 452, "setxattrat": 463, "getxattrat": 464, "listxattrat": 465, "removexattrat": 466,
         "file_getattr": 468, "file_setattr": 469}
PID_ARG0 = """prlimit64 sched_setaffinity sched_getaffinity sched_setscheduler sched_getscheduler sched_setparam
    sched_getparam sched_rr_get_interval sched_setattr sched_getattr""".split()
WHICH_WHO = {"getpriority": 0, "setpriority": 0, "ioprio_get": 1, "ioprio_set": 1}
# ioctl requests that change an inode's flags, attributes or generation (asm-generic encoding).
SET_INODE = {"FS_IOC_SETFLAGS": 0x40086602, "FS_IOC32_SETFLAGS": 0x40046602, "FS_IOC_FSSETXATTR": 0x401C5820,
             "FS_IOC_SETVERSION": 0x40087602, "FS_IOC32_SETVERSION": 0x40047602,
             "EXT4_IOC_SETVERSION": 0x40086604}
OTHER_IOCTLS = {"TCGETS": 0x5401, "FIONREAD": 0x541B, "FIONBIO": 0x5421, "FS_IOC_GETFLAGS": 0x80086601,
                "FS_IOC_FSGETXATTR": 0x801C581F}
FCNTL_DENIED = {"F_SET_RW_HINT": 1036, "F_SETLK": 6, "F_SETLKW": 7, "F_OFD_SETLK": 37, "F_OFD_SETLKW": 38,
                "F_SETLEASE": 1024, "F_NOTIFY": 1026}
FCNTL_ALLOWED = {"F_DUPFD": 0, "F_GETFD": 1, "F_SETFD": 2, "F_GETFL": 3, "F_SETFL": 4, "F_GETLK": 5,
                 "F_DUPFD_CLOEXEC": 1030, "F_GET_RW_HINT": 1035}
CLONE_THREAD = 0x10000
PTHREAD_FLAGS = 0x3D0F00                  # what glibc's pthread_create passes, CLONE_THREAD among them


def _scmp():
    """libseccomp through a handle of the test's own, so the types set here stay here."""
    lib = ctypes.CDLL("libseccomp.so.2")
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_num_arch.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_num_arch.argtypes = [ctypes.c_uint32, ctypes.c_int]
    lib.seccomp_arch_native.restype = ctypes.c_uint32
    lib.seccomp_arch_native.argtypes = []
    return lib


def _highest_known(lib, arch) -> int:
    libc = ctypes.CDLL(None)
    libc.free.argtypes = [ctypes.c_void_p]
    top = -1
    for nr in range(1024):
        name = lib.seccomp_syscall_resolve_num_arch(arch, nr)
        if name:
            top = nr
            libc.free(name)
    return top


def _filter_bpf(stage: int = 1) -> bytes:
    """confine's seccomp filter (stage 1) or seal's (stage 2) as the kernel would be given it,
    exported without loading it."""
    lib = confine._seccomp()
    ctx = confine._build_filter(lib, stage)
    fd = os.memfd_create("seccomp-bpf")
    try:
        assert lib.seccomp_export_bpf(ctx, fd) == 0
        os.lseek(fd, 0, os.SEEK_SET)
        return b"".join(iter(lambda: os.read(fd, 65536), b""))
    finally:
        os.close(fd)
        lib.seccomp_release(ctx)


def _run_bpf(prog: bytes, arch: int, nr: int, args=()) -> int:
    """Classic BPF over a struct seccomp_data, as the kernel runs a seccomp filter: the action."""
    args = [a & (2**64 - 1) for a in args] + [0] * (6 - len(args))
    data = struct.pack("<iIQ6Q", nr, arch, 0, *args)
    ins = [struct.unpack_from("<HBBI", prog, o) for o in range(0, len(prog), 8)]
    a, pc = 0, 0
    while True:
        code, jt, jf, k = ins[pc]
        pc += 1
        if code == 0x20:                      # ld [k]
            a = struct.unpack_from("<I", data, k)[0]
        elif code == 0x00:                    # ld #k
            a = k
        elif code == 0x54:                    # and #k
            a &= k
        elif code == 0x05:                    # ja k
            pc += k
        elif code in (0x15, 0x25, 0x35, 0x45):  # jeq, jgt, jge, jset #k
            taken = {0x15: a == k, 0x25: a > k, 0x35: a >= k, 0x45: bool(a & k)}[code]
            pc += jt if taken else jf
        elif code == 0x06:                    # ret #k
            return k
        else:
            raise AssertionError(f"BPF opcode {code:#x} is not one this interpreter knows")


def test_the_filter_pins_every_rule():
    lib, prog = _scmp(), _filter_bpf()
    arch = lib.seccomp_arch_native()

    def nr(name):
        n = lib.seccomp_syscall_resolve_name(name.encode())
        assert n != -1, f"libseccomp cannot name {name}"
        return n

    def run(n, *args):
        return _run_bpf(prog, arch, n, args)

    for name in DENIED:
        if nr(name) >= 0:                     # (a negative number: no such call on this architecture)
            assert run(nr(name)) == EPERM and run(nr(name), *[JUNK | 7] * 6) == EPERM, name
    for name in ("read", "write", "openat", "close", "mmap", "munmap", "futex", "getpid", "fstat", "newfstatat",
                 "statx", "lseek", "rt_sigaction", "exit_group", "landlock_create_ruleset"):
        if nr(name) >= 0:
            assert run(nr(name)) == ALLOW, name
    top = _highest_known(lib, arch)
    assert all(run(n) == ENOSYS for n in range(top + 1, 1024)), "every call newer than libseccomp: ENOSYS"
    for name, n in NEWER.items():
        assert run(n) == (EPERM if lib.seccomp_syscall_resolve_name(name.encode()) == n else ENOSYS), name
    # Processes, not threads.
    assert run(nr("clone"), PTHREAD_FLAGS) == ALLOW
    assert run(nr("clone"), signal.SIGCHLD) == EPERM and run(nr("clone"), JUNK | signal.SIGCHLD) == EPERM
    assert run(nr("clone3")) == ENOSYS
    # The parent-death signal stays; other prctl options are fine.
    assert run(nr("prctl"), 1) == EPERM and run(nr("prctl"), JUNK | 1) == EPERM
    assert run(nr("prctl"), 15) == ALLOW and run(nr("prctl"), 38, 1) == ALLOW
    # Calls aimed at another process.
    for name in PID_ARG0:
        if nr(name) >= 0:
            assert run(nr(name), 0) == ALLOW, name
            assert run(nr(name), 1234) == EPERM and run(nr(name), 1 << 32) == EPERM, name
    for name, which in WHICH_WHO.items():
        assert run(nr(name), which, 0) == ALLOW, name
        assert run(nr(name), which + 1, 0) == EPERM and run(nr(name), which, 99) == EPERM, name
        assert run(nr(name), which | 1 << 32, 0) == EPERM, name
    # Inode flags and attributes (chattr), on any file the worker can open; the request's low 32 bits.
    for request in SET_INODE.values():
        assert run(nr("ioctl"), 3, request) == EPERM and run(nr("ioctl"), 3, JUNK | request) == EPERM, hex(request)
    for request in OTHER_IOCTLS.values():
        assert run(nr("ioctl"), 3, request) == ALLOW, hex(request)
    for cmd in FCNTL_DENIED.values():
        assert run(nr("fcntl"), 3, cmd) == EPERM and run(nr("fcntl"), 3, JUNK | cmd) == EPERM, cmd
    for cmd in FCNTL_ALLOWED.values():
        assert run(nr("fcntl"), 3, cmd) == ALLOW, cmd
    # Another ABI kills the process.
    assert run(nr("getpid")) == ALLOW
    assert _run_bpf(prog, 0x40000003, 20) == KILL_PROCESS            # i386
    if os.uname().machine == "x86_64":
        assert run(0x40000000 | 39) == KILL_PROCESS                   # x32


# Stage 2 (Confinement, the seal): every call that resolves a pathname, so a sealed worker follows
# and reads no symlink and opens, stats or names no file.
PATHNAME = """
    open openat openat2 creat open_by_handle_at name_to_handle_at uselib execve execveat
    stat lstat newfstatat statx statfs access faccessat faccessat2 readlink readlinkat
    chdir chroot pivot_root mount umount2 open_tree move_mount fsopen fsconfig fspick mount_setattr
    mkdir mkdirat mknod mknodat rmdir unlink unlinkat rename renameat renameat2 link linkat symlink symlinkat
    truncate acct swapon swapoff quotactl inotify_add_watch fanotify_mark
    chmod fchmodat chown lchown fchownat utime utimes utimensat futimesat
    getxattr lgetxattr setxattr lsetxattr removexattr lremovexattr listxattr llistxattr""".split()
NEWER_PATHNAME = {**NEWER, "open_tree_attr": 467}     # every one of NEWER takes a path too
# What a sealed worker still does with what it holds: its pipes, memory, threads.
SEALED_ALLOWED = """read write close fstat fstatfs lseek mmap munmap mprotect futex fcntl ioctl getdents64 fchdir
    getcwd dup dup3 pipe2 clone memfd_create getrandom clock_gettime exit_group""".split()


def test_the_seal_pins_every_rule():
    lib, seal, stage1 = _scmp(), _filter_bpf(2), _filter_bpf(1)
    arch = lib.seccomp_arch_native()

    def nr(name):
        n = lib.seccomp_syscall_resolve_name(name.encode())
        assert n != -1, f"libseccomp cannot name {name}"
        return n

    for name in PATHNAME:
        if nr(name) >= 0:                     # (a negative number: no such call on this architecture)
            assert _run_bpf(seal, arch, nr(name)) == EPERM, name
            assert _run_bpf(seal, arch, nr(name), [JUNK | 7] * 6) == EPERM, name
    for name, n in NEWER_PATHNAME.items():    # unknown to this libseccomp: stage 1's ENOSYS stands in
        known = lib.seccomp_syscall_resolve_name(name.encode()) == n
        assert (_run_bpf(seal, arch, n) == EPERM) if known else (_run_bpf(stage1, arch, n) == ENOSYS), name
    for name in SEALED_ALLOWED:
        if nr(name) >= 0:
            assert _run_bpf(seal, arch, nr(name)) == ALLOW, name
    assert _run_bpf(seal, 0x40000003, 5) == KILL_PROCESS                 # i386 open
    if os.uname().machine == "x86_64":
        assert _run_bpf(seal, arch, 0x40000000 | 2) == KILL_PROCESS       # x32 open


def test_probe_reports_none_here():
    assert confine.probe() is None


def test_probe_reports_a_low_abi(monkeypatch):
    monkeypatch.setattr(confine, "_landlock_abi", lambda: 5)
    assert confine.probe() == "Landlock ABI 5 < 6"
