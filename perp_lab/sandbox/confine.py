"""Confines the calling process for good, in two stages.

`confine()` is stage 1, the spec's Confinement steps 1-5, in order: 1. exactly one thread; 2. hard
rlimits; 3. no_new_privs; 4. Landlock (read-only allowlist, no TCP, abstract UNIX sockets and
signals scoped); 5. seccomp (sockets, exec, fork, System V IPC, calls aimed at another process,
clearing the parent-death signal, changing a file's metadata, file locks, leases and
notifications, and kernel-facing calls return an error; so does every call newer than
libseccomp, with ENOSYS).

`seal()` is stage 2, once the process has loaded everything it will need: a second seccomp filter
under which no call that resolves a pathname works, so the process opens, stats and reads no
file and follows no symlink. Reading a file, or following or reading a symlink, updates its access
time on a relatime mount: without the seal a worker could leave a bit per allowlisted file or
symlink per day for a later worker to read back with stat(). Nor can it query or steer the page
cache of the files it has mapped (mincore, cachestat, eviction and prefetch advice).

Landlock and seccomp confine the calling thread and the threads it starts later; a thread that
already exists stays unconfined, and plugin code could reach it through ctypes. So `confine()`
refuses unless the process has one thread, and this module uses only the standard library: the
worker calls it before importing numpy, which starts threads. `seal()` is seccomp only, with
filter thread synchronisation, so it reaches the threads that exist by then (pyarrow's) too.

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
MIN_GLIBC = (2, 39)                           # makes fstat with the fstat syscall, which the seal leaves
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
_PR_SET_PDEATHSIG = 1
_PR_GET_SECCOMP, _SECCOMP_MODE_FILTER = 21, 2


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
    mq_open mq_unlink mq_timedsend mq_timedreceive mq_notify mq_getsetattr
    inotify_init inotify_init1 fanotify_init""".split()     # Landlock has no right for file watches
# File metadata. Landlock has no right for a file's mode, owner, timestamps or extended attributes,
# and seccomp can't filter by path, so a worker could otherwise leave bytes in the metadata of any
# file this user owns, for a later worker to read. Reading xattrs goes too; stat() stays (R5).
_METADATA = """
    chmod fchmod fchmodat chown fchown lchown fchownat utime utimes utimensat futimesat
    setxattr lsetxattr fsetxattr removexattr lremovexattr fremovexattr
    getxattr lgetxattr fgetxattr listxattr llistxattr flistxattr""".split()
# Metadata calls newer than some libseccomp releases: denied by name where this libseccomp knows
# them. Where it doesn't, their numbers are above every number it knows, which return ENOSYS.
_NEWER_METADATA = "fchmodat2 setxattrat getxattrat listxattrat removexattrat file_getattr file_setattr".split()
SYSCALL_BOUND = 1024                          # every number from libseccomp's highest + 1 to here: ENOSYS
# ioctl requests that change an inode's flags (chattr), its fsxattr attributes or its generation
# number, on any file the worker can open (its allowlist: the venv and perp_lab/ are this user's).
# Same values on x86_64 and aarch64 (asm-generic ioctl encoding); the request is an unsigned int.
_SET_INODE_IOCTLS = {"FS_IOC_SETFLAGS": 0x40086602, "FS_IOC32_SETFLAGS": 0x40046602,
                     "FS_IOC_FSSETXATTR": 0x401C5820, "FS_IOC_SETVERSION": 0x40087602,
                     "FS_IOC32_SETVERSION": 0x40047602, "EXT4_IOC_SETVERSION": 0x40086604}
# Locks, leases and directory notifications on a file another process can open are a live channel
# Landlock doesn't restrict (a read-only fd is enough): a worker holding holdout bars could signal
# them to a worker of another exam or check. So flock(), and these fcntl commands, return EPERM.
_LOCKS = ["flock"]
# fcntl commands denied: locks (POSIX and open-file-description), leases, directory notifications,
# and the inode's write-life hint (owner only, read back by anyone with an fd). F_GETFL, F_SETFL,
# F_SETFD, F_DUPFD_CLOEXEC and the other commands stay.
_FCNTL_EPERM = {"F_SETLK": 6, "F_SETLKW": 7, "F_OFD_SETLK": 37, "F_OFD_SETLKW": 38, "F_SETLEASE": 1024,
                "F_NOTIFY": 1026, "F_SET_RW_HINT": 1036}
_PID_ARG0 = """
    prlimit64 sched_setaffinity sched_getaffinity sched_setscheduler sched_getscheduler
    sched_setparam sched_getparam sched_rr_get_interval sched_setattr sched_getattr""".split()
# (which, who): who 0 is this process only when which is PRIO_PROCESS (0) / IOPRIO_WHO_PROCESS (1);
# with PRIO_USER or IOPRIO_WHO_USER it is every process of this user, the judge included.
_WHICH_WHO = {"getpriority": 0, "setpriority": 0, "ioprio_get": 1, "ioprio_set": 1}

# (action, syscall name, conditions (arg, op, datum_a, datum_b) that must all hold); several rules
# for one syscall deny if any of them matches.
_RULES = (
    [(_EPERM, name, ()) for name in _ALWAYS_EPERM + _METADATA + _NEWER_METADATA + _LOCKS]
    + [(_EPERM, "clone", ((0, _SCMP_CMP_MASKED_EQ, _CLONE_THREAD, 0),)),  # a thread is fine, a process isn't
       (_ENOSYS, "clone3", ())]                                          # so libc falls back to clone
    # The worker sets its parent-death signal before confining itself; plugin code may not clear
    # or change it. The option is an int: compare its low 32 bits, or junk above them would pass.
    + [(_EPERM, "prctl", ((0, _SCMP_CMP_MASKED_EQ, 0xFFFFFFFF, _PR_SET_PDEATHSIG),))]
    + [(_EPERM, name, ((0, _SCMP_CMP_NE, 0, 0),)) for name in _PID_ARG0]  # pid 0: this process only
    + [rule for name, which in _WHICH_WHO.items()                        # this process only
       for rule in ((_EPERM, name, ((0, _SCMP_CMP_NE, which, 0),)), (_EPERM, name, ((1, _SCMP_CMP_NE, 0, 0),)))]
    # ioctl's request and fcntl's command are unsigned ints: their low 32 bits, as for prctl.
    + [(_EPERM, "ioctl", ((1, _SCMP_CMP_MASKED_EQ, 0xFFFFFFFF, req),)) for req in _SET_INODE_IOCTLS.values()]
    + [(_EPERM, "fcntl", ((1, _SCMP_CMP_MASKED_EQ, 0xFFFFFFFF, cmd),)) for cmd in _FCNTL_EPERM.values()]
)

# Stage 2 (`seal`): every syscall that resolves a pathname (from the x86_64 and aarch64 tables).
# Resolving a path follows its symlinks, which updates their access times, before Landlock or
# anything else can refuse the call; so even a refused open() through a symlink leaves a mark,
# and readlink() leaves one too. A sealed process names no path at all: it opens, stats and lists
# nothing and reads no link. What it already holds (its pipes, memory, threads) keeps working, so
# fstat on a descriptor does too, as long as libc makes it with fstat and not newfstatat (glibc
# 2.39 and later). Some names exist on one architecture only (open, stat, ...: x86_64), as in
# stage 1. Calls with a name libseccomp doesn't know are newer than it: stage 1 answers ENOSYS.
_PATHNAME = """
    open openat openat2 creat open_by_handle_at name_to_handle_at uselib execve execveat
    stat lstat newfstatat fstatat64 statx statfs access faccessat faccessat2 readlink readlinkat
    chdir chroot pivot_root mount umount2 open_tree move_mount fsopen fsconfig fspick mount_setattr
    mkdir mkdirat mknod mknodat rmdir unlink unlinkat rename renameat renameat2 link linkat
    symlink symlinkat truncate acct swapon swapoff quotactl inotify_add_watch fanotify_mark
    chmod fchmodat chown lchown fchownat utime utimes utimensat futimesat
    getxattr lgetxattr setxattr lsetxattr removexattr lremovexattr listxattr llistxattr""".split()
_NEWER_PATHNAME = _NEWER_METADATA + ["open_tree_attr"]    # every newer metadata call takes a path too
# Stage 2 also covers the page cache of the files a worker has mapped (its libraries): which of
# their pages are resident is visible to a later process (mincore, or how long a fault takes), so
# a sealed worker may not query residency (mincore, cachestat) or evict or prefetch pages:
# madvise with these advice values. HUGEPAGE is prefetch on a file mapping: a fault there reads a
# whole 2 MiB, PMD-aligned stretch in as one large folio, and a live process mapping any page of a
# folio keeps all of it resident past the eviction (numpy's hint for large arrays is refused too,
# and numpy ignores that). Other advice (DONTNEED, FREE, NOHUGEPAGE: what allocators use) stays.
# It can still fault pages in by touching them; after each batch of an exam the parent evicts
# every file its workers mapped (`client.evict_page_cache`). That narrows the channel but doesn't
# close it (spec R7: fault-around by a live process can keep a touched page resident).
_PAGE_CACHE = ["mincore", "cachestat"]
_MADV_DENIED = {"MADV_WILLNEED": 3, "MADV_HUGEPAGE": 14, "MADV_COLD": 20, "MADV_PAGEOUT": 21,
                "MADV_POPULATE_READ": 22, "MADV_POPULATE_WRITE": 23,
                "MADV_COLLAPSE": 25}                                 # COLLAPSE reads a whole huge page in
_SEAL_RULES = (
    [(_EPERM, name, ()) for name in _PATHNAME + _NEWER_PATHNAME + _PAGE_CACHE]
    # madvise's advice is an int: its low 32 bits, as for prctl, ioctl and fcntl.
    + [(_EPERM, "madvise", ((2, _SCMP_CMP_MASKED_EQ, 0xFFFFFFFF, advice),)) for advice in _MADV_DENIED.values()]
)
_NEWER = set(_NEWER_PATHNAME) | {"cachestat"}                # names some libseccomp releases lack

_libc_dll = None
_seccomp_dll = None


def _libc():
    global _libc_dll
    if _libc_dll is None:
        lib = ctypes.CDLL(None, use_errno=True)
        lib.syscall.restype = ctypes.c_long
        lib.prctl.restype = ctypes.c_int
        lib.free.restype, lib.free.argtypes = None, [ctypes.c_void_p]
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
                ("seccomp_syscall_resolve_num_arch", ctypes.c_void_p, [ctypes.c_uint32, ctypes.c_int]),
                ("seccomp_arch_native", ctypes.c_uint32, []),
                ("seccomp_rule_add_array", ctypes.c_int,
                 [ctx, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(_ScmpArgCmp)]),
                ("seccomp_load", ctypes.c_int, [ctx]),
                ("seccomp_export_bpf", ctypes.c_int, [ctx, ctypes.c_int]),
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
    glibc = _glibc()
    if glibc is None:
        return "glibc 2.39 or later needed, and this C library is not glibc"
    if glibc < MIN_GLIBC:
        return f"glibc {glibc[0]}.{glibc[1]} < {MIN_GLIBC[0]}.{MIN_GLIBC[1]}"
    return None


def _glibc() -> tuple[int, int] | None:
    """glibc's version, or None if the C library isn't glibc. Older glibc makes fstat with
    newfstatat, which the seal denies, so a sealed worker couldn't fstat its own pipes."""
    try:
        name, version = os.confstr("CS_GNU_LIBC_VERSION").split()
        major, minor = (int(x) for x in version.split(".")[:2])
    except (AttributeError, OSError, ValueError):     # None (not defined), or an unknown name
        return None
    return (major, minor) if name == "glibc" else None


def allowlist() -> list[str]:
    """The directories a confined worker may read (files and listings) until it seals itself;
    nothing else is readable, and once sealed nothing at all."""
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
    _seccomp_filter(seccomp, 1)
    return abi


def seal() -> None:
    """Stage 2, for good, in every thread: from now on no call that resolves a pathname works, and
    none that queries, evicts or prefetches page cache (EPERM). Call it once everything the
    process will need is loaded.

    Raises RuntimeError if it fails, or if `confine()` hasn't run (stage 2 alone would leave all
    that stage 1 denies allowed); the process must then not run plugin code.

    The guard checks only that some seccomp filter is in place (PR_GET_SECCOMP is 2). Under a
    filter inherited from outside (a Docker container's, for one) it is 2 before `confine()` too,
    so there it can't tell. That is acceptable: the guard catches a caller's mistake, not an
    attack (no plugin code has run yet), and the worker calls `seal()` only after `confine()` has
    returned and its stage-1 self-test has passed.
    """
    mode = _libc().prctl(ctypes.c_int(_PR_GET_SECCOMP), ctypes.c_ulong(0), ctypes.c_ulong(0), ctypes.c_ulong(0),
                         ctypes.c_ulong(0))
    if mode != _SECCOMP_MODE_FILTER:
        raise RuntimeError(f"seal() needs confine() first (seccomp mode {mode})")
    try:
        lib = _seccomp()
    except (OSError, AttributeError) as e:
        raise RuntimeError(f"libseccomp.so.2: {e}") from e
    _seccomp_filter(lib, 2)


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


def _seccomp_filter(lib, stage: int) -> None:
    ctx = _build_filter(lib, stage)
    try:
        _check(lib.seccomp_load(ctx), f"seccomp_load (stage {stage})")
    finally:
        lib.seccomp_release(ctx)


def _build_filter(lib, stage: int = 1):
    """Stage 1's filter (`confine`) or stage 2's (`seal`), built but not loaded: a libseccomp
    context the caller must release. (The tests export it with seccomp_export_bpf and check every
    rule without confining themselves.)"""
    ctx = lib.seccomp_init(_SCMP_ACT_ALLOW)
    if not ctx:
        raise RuntimeError("seccomp_init failed")
    try:
        # A syscall made through another ABI (i386, x32) never runs, and the whole process dies
        # (the default kills only the calling thread).
        _check(lib.seccomp_attr_set(ctx, _SCMP_FLTATR_ACT_BADARCH, _SCMP_ACT_KILL_PROCESS),
               "seccomp_attr_set(ACT_BADARCH)")
        _check(lib.seccomp_attr_set(ctx, _SCMP_FLTATR_CTL_TSYNC, 1), "seccomp_attr_set(CTL_TSYNC)")
        for action, name, conditions in (_RULES if stage == 1 else _SEAL_RULES):
            nr = lib.seccomp_syscall_resolve_name(name.encode())
            if nr == _NR_SCMP_ERROR and name in _NEWER:
                continue                      # newer than this libseccomp: stage 1's ENOSYS, below
            if nr == _NR_SCMP_ERROR:          # unknown to this libseccomp: fail closed, not a hole
                raise RuntimeError(f"libseccomp cannot name the syscall {name}")
            if nr < 0:
                continue                      # no such syscall on this architecture
            args = (_ScmpArgCmp * len(conditions))(*conditions) if conditions else None
            _check(lib.seccomp_rule_add_array(ctx, action, nr, len(conditions), args),
                   f"seccomp_rule_add_array({name})")
        # A call newer than this libseccomp (e.g. setxattrat, file_setattr on a kernel ahead of it)
        # can't be named here, so none of the rules above can reach it: ENOSYS, as from an older
        # kernel, which callers already handle. Stage 1 stays loaded under stage 2, so this covers both.
        if stage == 1:
            for nr in range(_highest_known(lib) + 1, SYSCALL_BOUND):
                _check(lib.seccomp_rule_add_array(ctx, _ENOSYS, nr, 0, None), f"seccomp_rule_add_array({nr})")
    except BaseException:
        lib.seccomp_release(ctx)
        raise
    return ctx


def _highest_known(lib) -> int:
    """The highest syscall number this libseccomp can name on the native architecture."""
    arch, top = lib.seccomp_arch_native(), -1
    for nr in range(SYSCALL_BOUND):
        name = lib.seccomp_syscall_resolve_num_arch(arch, nr)
        if name:
            top = nr
            _libc().free(name)              # a copy libseccomp made with malloc
    if top < 0:
        raise RuntimeError("libseccomp names no syscall on this architecture")
    return top


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what}: {os.strerror(-rc)}")
