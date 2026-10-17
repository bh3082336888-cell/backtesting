"""Confines the calling process for good: the spec's Confinement steps 1-5, in order.

1. exactly one thread; 2. hard rlimits; 3. no_new_privs; 4. Landlock (read-only allowlist, no TCP,
abstract UNIX sockets and signals scoped); 5. seccomp (sockets, exec, fork, System V IPC, calls
aimed at another process, and kernel-facing calls return an error).

Landlock and seccomp confine the calling thread and the threads it starts later; a thread that
already exists stays unconfined, and plugin code could reach it through ctypes. So `confine()`
refuses unless the process has one thread, and this module uses only the standard library: the
worker calls it before importing numpy, which starts threads.

Linux on x86_64 or aarch64 (the Landlock syscall numbers below are theirs). `probe()` says, without
spawning anything, whether this machine can confine; if it can't, plugin code does not run.
"""

from __future__ import annotations

import ctypes
import errno
import os
import resource
import site
import sys

MIN_ABI = 6                                   # scoped signals and abstract UNIX sockets arrive in ABI 6
NOFILE = 64

_ARCHES = ("x86_64", "aarch64")

# Landlock: include/uapi/linux/landlock.h
_SYS_LANDLOCK_CREATE_RULESET, _SYS_LANDLOCK_ADD_RULE, _SYS_LANDLOCK_RESTRICT_SELF = 444, 445, 446
_CREATE_RULESET_VERSION = 1 << 0
_RULE_PATH_BENEATH = 1
_RESTRICT_SELF_TSYNC = 1 << 3                 # ABI 8: the domain covers every thread of the process
_FS_READ_FILE, _FS_READ_DIR = 1 << 2, 1 << 3
_FS_REFER, _FS_TRUNCATE, _FS_IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15
_NET_BIND_TCP, _NET_CONNECT_TCP = 1 << 0, 1 << 1
_SCOPE_ABSTRACT_UNIX_SOCKET, _SCOPE_SIGNAL = 1 << 0, 1 << 1
_PR_SET_NO_NEW_PRIVS = 38


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64), ("handled_access_net", ctypes.c_uint64),
                ("scoped", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1                                # __attribute__((packed)) in the kernel: 12 bytes
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


# seccomp: libseccomp's seccomp.h
_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_KILL_PROCESS = 0x80000000
_SCMP_FLTATR_ACT_BADARCH, _SCMP_FLTATR_CTL_TSYNC = 2, 4
_SCMP_CMP_NE, _SCMP_CMP_MASKED_EQ = 1, 7
_NR_SCMP_ERROR = -1                           # seccomp_syscall_resolve_name: a name it doesn't know
_CLONE_THREAD = 0x10000


def _errno_action(e: int) -> int:
    return 0x00050000 | e                     # SCMP_ACT_ERRNO(e)


class _ScmpArgCmp(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint32), ("op", ctypes.c_uint32), ("datum_a", ctypes.c_uint64),
                ("datum_b", ctypes.c_uint64)]


_EPERM, _ENOSYS = _errno_action(errno.EPERM), _errno_action(errno.ENOSYS)
_ALWAYS_EPERM = """
    socket socketpair execve execveat fork vfork
    shmget shmat shmctl shmdt msgget msgsnd msgrcv msgctl semget semop semtimedop semctl
    ptrace process_vm_readv process_vm_writev process_madvise kcmp pidfd_open pidfd_getfd
    migrate_pages move_pages mount umount2 pivot_root chroot unshare setns bpf perf_event_open
    userfaultfd io_uring_setup io_uring_enter io_uring_register keyctl add_key request_key
    init_module finit_module delete_module kexec_load kexec_file_load
    mq_open mq_unlink mq_timedsend mq_timedreceive mq_notify mq_getsetattr""".split()
_PID_ARG0 = """
    prlimit64 sched_setaffinity sched_getaffinity sched_setscheduler sched_getscheduler
    sched_setparam sched_getparam sched_rr_get_interval sched_setattr sched_getattr""".split()
# (which, who): who 0 is this process only when which is PRIO_PROCESS (0) / IOPRIO_WHO_PROCESS (1);
# with PRIO_USER or IOPRIO_WHO_USER it is every process of this user, the judge included.
_WHICH_WHO = {"getpriority": 0, "setpriority": 0, "ioprio_get": 1, "ioprio_set": 1}

# (action, syscall name, conditions (arg, op, datum_a, datum_b) that must all hold); several rules
# for one syscall deny if any of them matches.
_RULES = (
    [(_EPERM, name, ()) for name in _ALWAYS_EPERM]
    + [(_EPERM, "clone", ((0, _SCMP_CMP_MASKED_EQ, _CLONE_THREAD, 0),)),  # a thread is fine, a process isn't
       (_ENOSYS, "clone3", ())]                                          # so libc falls back to clone
    + [(_EPERM, name, ((0, _SCMP_CMP_NE, 0, 0),)) for name in _PID_ARG0]  # pid 0: this process only
    + [rule for name, which in _WHICH_WHO.items()                        # this process only
       for rule in ((_EPERM, name, ((0, _SCMP_CMP_NE, which, 0),)), (_EPERM, name, ((1, _SCMP_CMP_NE, 0, 0),)))]
)

_libc_dll = None
_seccomp_dll = None


def _libc():
    global _libc_dll
    if _libc_dll is None:
        lib = ctypes.CDLL(None, use_errno=True)
        lib.syscall.restype = ctypes.c_long
        lib.prctl.restype = ctypes.c_int
        _libc_dll = lib
    return _libc_dll


def _seccomp():
    """libseccomp.so.2 with every function this module calls typed; OSError if it's missing."""
    global _seccomp_dll
    if _seccomp_dll is None:
        lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
        ctx = ctypes.c_void_p
        for name, restype, argtypes in (
                ("seccomp_init", ctx, [ctypes.c_uint32]),
                ("seccomp_attr_set", ctypes.c_int, [ctx, ctypes.c_int, ctypes.c_uint32]),
                ("seccomp_syscall_resolve_name", ctypes.c_int, [ctypes.c_char_p]),
                ("seccomp_rule_add_array", ctypes.c_int,
                 [ctx, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(_ScmpArgCmp)]),
                ("seccomp_load", ctypes.c_int, [ctx]),
                ("seccomp_release", None, [ctx])):
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = restype, argtypes
        _seccomp_dll = lib
    return _seccomp_dll


def _syscall(nr: int, *args) -> int:
    r = _libc().syscall(ctypes.c_long(nr), *args)
    if r < 0:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e))
    return r


def _landlock_abi() -> int:
    """The running kernel's Landlock ABI version; OSError when Landlock is unavailable."""
    return _syscall(_SYS_LANDLOCK_CREATE_RULESET, None, ctypes.c_size_t(0),
                    ctypes.c_uint32(_CREATE_RULESET_VERSION))


def probe(min_abi: int = MIN_ABI) -> str | None:
    """None when this machine can confine a worker; otherwise the reason. Spawns nothing."""
    if sys.platform != "linux":
        return "not Linux"
    machine = os.uname().machine
    if machine not in _ARCHES:
        return f"unsupported architecture {machine}"
    try:
        abi = _landlock_abi()
    except OSError as e:
        return f"Landlock unavailable: {e.strerror}"
    if abi < min_abi:
        return f"Landlock ABI {abi} < {min_abi}"
    try:
        _seccomp()
    except OSError:
        return "libseccomp.so.2 not found"
    except AttributeError as e:
        return f"libseccomp.so.2 is too old: {e}"
    return None


def allowlist() -> list[str]:
    """The directories a confined worker may read (files and listings); nothing else is readable."""
    paths = [sys.base_prefix,                 # the standard library
             sys.prefix,                      # the virtualenv, if any
             *site.getsitepackages(),         # numpy, pandas, pyarrow
             os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]  # the perp_lab package
    # Extra read-only system paths numpy, pandas or pyarrow need, each with its reason: none so far
    # (tests/test_confine.py::test_numpy_pandas_pyarrow_import_after_confinement).
    # getsitepackages() names directories that may not exist; a rule needs an existing path.
    return [p for p in dict.fromkeys(paths) if os.path.isdir(p)]


def confine(*, memory_mb: int, cpu_s: int, read_paths: list[str]) -> int:
    """Confine this process for good; returns the Landlock ABI.

    Raises RuntimeError if any step fails; the process must then not run plugin code.
    """
    # 1. One thread, or an existing thread would stay unconfined.
    try:
        threads = os.listdir("/proc/self/task")
    except OSError as e:
        raise RuntimeError(f"cannot count this process's threads: {e}") from e
    if len(threads) != 1:
        raise RuntimeError(f"{len(threads)} threads; confinement needs this process to have exactly one")
    try:                                      # loaded now: step 4 may leave its file unreadable
        libc, seccomp = _libc(), _seccomp()
    except (OSError, AttributeError) as e:
        raise RuntimeError(f"libseccomp.so.2: {e}") from e

    # 2. Hard rlimits (soft = hard, so the process can't raise them again).
    for name, limit in (("RLIMIT_AS", memory_mb << 20), ("RLIMIT_FSIZE", 0), ("RLIMIT_CORE", 0),
                        ("RLIMIT_NOFILE", NOFILE), ("RLIMIT_CPU", cpu_s)):
        try:
            resource.setrlimit(getattr(resource, name), (limit, limit))
        except (OSError, ValueError) as e:
            raise RuntimeError(f"setrlimit({name}, {limit}): {e}") from e

    # 3. no_new_privs: landlock_restrict_self requires it of an unprivileged process.
    if libc.prctl(ctypes.c_int(_PR_SET_NO_NEW_PRIVS), ctypes.c_ulong(1), ctypes.c_ulong(0),
                  ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
        raise RuntimeError(f"prctl(PR_SET_NO_NEW_PRIVS): {os.strerror(ctypes.get_errno())}")

    # 4. Landlock, then 5. seccomp.
    abi = _landlock(read_paths)
    _seccomp_filter(seccomp)
    return abi


def _landlock(read_paths: list[str]) -> int:
    step = "landlock_create_ruleset(VERSION)"
    try:
        abi = _landlock_abi()
        if abi < MIN_ABI:
            raise RuntimeError(f"Landlock ABI {abi} < {MIN_ABI}")
        fs = (1 << 13) - 1                    # EXECUTE .. MAKE_SYM, every right of ABI 1
        if abi >= 2:
            fs |= _FS_REFER
        if abi >= 3:
            fs |= _FS_TRUNCATE
        if abi >= 5:
            fs |= _FS_IOCTL_DEV
        attr = _RulesetAttr(fs, _NET_BIND_TCP | _NET_CONNECT_TCP,
                            _SCOPE_ABSTRACT_UNIX_SOCKET | _SCOPE_SIGNAL)
        step = "landlock_create_ruleset"
        ruleset = _syscall(_SYS_LANDLOCK_CREATE_RULESET, ctypes.byref(attr),
                           ctypes.c_size_t(ctypes.sizeof(attr)), ctypes.c_uint32(0))
        try:
            for path in read_paths:
                step = f"open({path!r}, O_PATH)"
                fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
                try:
                    step = f"landlock_add_rule({path!r})"
                    rule = _PathBeneathAttr(_FS_READ_FILE | _FS_READ_DIR, fd)
                    _syscall(_SYS_LANDLOCK_ADD_RULE, ctypes.c_int(ruleset),
                             ctypes.c_int(_RULE_PATH_BENEATH), ctypes.byref(rule), ctypes.c_uint32(0))
                finally:
                    os.close(fd)
            step = "landlock_restrict_self"
            _syscall(_SYS_LANDLOCK_RESTRICT_SELF, ctypes.c_int(ruleset),
                     ctypes.c_uint32(_RESTRICT_SELF_TSYNC if abi >= 8 else 0))
        finally:
            os.close(ruleset)
    except OSError as e:
        raise RuntimeError(f"{step}: {e.strerror}") from e
    return abi


def _seccomp_filter(lib) -> None:
    ctx = lib.seccomp_init(_SCMP_ACT_ALLOW)
    if not ctx:
        raise RuntimeError("seccomp_init failed")
    try:
        # A syscall made through another ABI (i386, x32) never runs, and the whole process dies
        # (the default kills only the calling thread).
        _check(lib.seccomp_attr_set(ctx, _SCMP_FLTATR_ACT_BADARCH, _SCMP_ACT_KILL_PROCESS),
               "seccomp_attr_set(ACT_BADARCH)")
        _check(lib.seccomp_attr_set(ctx, _SCMP_FLTATR_CTL_TSYNC, 1), "seccomp_attr_set(CTL_TSYNC)")
        for action, name, conditions in _RULES:
            nr = lib.seccomp_syscall_resolve_name(name.encode())
            if nr == _NR_SCMP_ERROR:          # unknown to this libseccomp: fail closed, not a hole
                raise RuntimeError(f"libseccomp cannot name the syscall {name}")
            if nr < 0:
                continue                      # no such syscall on this architecture
            args = (_ScmpArgCmp * len(conditions))(*conditions) if conditions else None
            _check(lib.seccomp_rule_add_array(ctx, action, nr, len(conditions), args),
                   f"seccomp_rule_add_array({name})")
        _check(lib.seccomp_load(ctx), "seccomp_load")
    finally:
        lib.seccomp_release(ctx)


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what}: {os.strerror(-rc)}")
