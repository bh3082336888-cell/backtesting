# Phase 3b-1: the plugin sandbox

Status: draft, 2026-10-17, revised after an independent builder's review (whose confinement
claims were tested on this machine). The design conversation was skipped at the user's request,
so every decision below is the author's call and open to review. The ones most worth a second
look are marked ★.
Scope: running strategy plugin code. The LLM proposer is Phase 3b-2; requirements this design
places on it are listed at the end.

## Goal

Plugin code, hand-written or proposer-written, never runs in a process that can read the
holdout, the store, the holdout log or the verdicts, or reach the network. The harness computes
every score itself from the plugin's *decisions*, so a plugin cannot forge its results. In the
holdout exam and in paper, a plugin decides at bar t having been given only bars up to t, so a
passed exam is honest whatever the plugin's code does.

This retires the 3a caveat that its guarantees "hold for honest plugin code only".

### What this guarantees

- **G1 Isolation.** A plugin's process loads its modules from the Python installation (everything
  under the interpreter's prefixes: for a system Python, all of `/usr`), site-packages and the
  `perp_lab/` package source, then seals itself before any plugin byte arrives. Once sealed it
  can't resolve any path at all: it opens, stats, lists, reads and writes no file and follows or
  reads no symlink, and it gets everything it works on (the plugin's source, bars, parameters)
  through its pipe. So it can't change any file's metadata either (mode, owner, timestamps,
  access times of files and symlinks included, extended attributes, inode flags). It cannot open
  sockets, start processes, share memory with another process, or signal, trace, or change the
  limits or scheduling of any process outside itself. It inherits no environment variables, so no
  secrets.
- **G2 Scoring.** Trades, equity and every metric come from the trusted engine replaying the
  plugin's decisions. A worker that reports anything inconsistent with the replay is caught
  (`mismatch`).
- **G3 Honest exam.** In the exam and in paper the plugin is streamed one bar at a time and
  never holds a bar later than the one it is deciding on, so lookahead is impossible there,
  not merely tested for.
- **G4 One version per run.** A plugin's bytes are read once per command, hashed, and those
  bytes are what every worker executes. The recorded `strategy_hash` is the code that ran.

### What it does not guarantee

- **R1 Behaving differently under test.** A plugin that recognises the synthetic lookahead bars,
  or notices it is inside the random-entry benchmark, can pass those checks while peeking at
  research data or sabotaging its benchmark. The cost is a false *promotion*: it can spend its
  family's exam read, but it cannot pass the exam (G3), so it cannot reach paper. The 10-read
  cap bounds the damage. A follow-up can narrow this (see the end). Two further routes of the
  same kind, also bounded by G3: walk-forward workers for different symbols run at the same time
  and could pass a symbol's later bars to another through CPU or cache timing, and so could a
  plugin's concurrent walk-forward runs in general. Each worker is sent one symbol only (below),
  which closes the direct route.
- **R2 Side channels.** An exam's outcome (pass or fail and a fixed code) carries a bit or two of
  holdout information per read. With at most 10 reads, accepted. Its duration would carry far
  more, since a plugin controls how long it takes, so the proposer never gets to time an exam:
  `lab judge` no longer runs exams, and 3b-2 must hide exam timing (see the end).
- **R3 Waste.** A plugin can burn CPU and memory up to its limits, or crash. Both end in a
  recorded kill.
- **R4 Kernel bugs.** Confinement rests on Landlock and seccomp. A kernel exploit escapes it.
- **R5 File metadata.** Closed for plugins, on both sides. A sealed worker can't `stat()`
  anything (Confinement, step 8), so it reads no file's size or times, and it changes no file's
  metadata, access times included (steps 5 and 8). Landlock has no right for `stat()` and seccomp
  can't filter by path, which is why the seal denies every call that resolves a path rather than
  some paths. What remains is outside the sandbox: any other process of this user can `stat()`
  `data/`, which 3b-2 must deny its proposer (see the end).
- **R6 Isolation across processes.** The wave and lockstep isolation holds within one lookahead
  check or one exam. It does not hold across lookahead checks that run at the same time in
  separate processes, or between such a check and an exam (through CPU timing). Two exams never
  run at once on one machine: `attempt_exam` holds a machine-wide lock throughout. File locks,
  leases and directory notifications, a live channel Landlock doesn't restrict, are denied.
- **R7 Page-cache residency.** Which pages of a file are in the page cache is visible to any
  process, through `mincore` on a mapping of it or the time a page fault takes. A worker maps its
  libraries (libarrow alone is about 11,000 pages) and could fault chosen pages in or push them
  out, for a later worker or process to read the pattern back. The seal narrows this (Confinement,
  step 8): a worker can't query residency (`mincore`, `cachestat`) or evict or prefetch pages
  (`madvise` with `MADV_PAGEOUT`, `MADV_COLD`, `MADV_WILLNEED`, `MADV_POPULATE_READ` or
  `MADV_POPULATE_WRITE`), so all it can do is fault pages in by touching them. After every exam
  attempt, whatever its outcome and once its workers are gone, the parent evicts every file those
  workers mapped (`POSIX_FADV_DONTNEED`, files read from `/proc/<pid>/maps`), so nothing carries
  into the next exam or judgement. What remains: a process running during an exam (a concurrent
  `lab judge`'s workers) could read residency by fault timing while it runs, which is R6's live
  overlap; and pages another process keeps mapped (the parent's own numpy and pyarrow, for one)
  stay resident, which carries nothing from a worker, while other processes add noise.

## Threat model

The author to defend against is an LLM proposer (3b-2) writing plugins under pressure to pass
the judge, able to read the harness source, plus honest mistakes in hand-written plugins. What
such a plugin could want: a promotion, a passed exam, holdout data, or a changed budget, log or
verdict. A human with a shell on this machine is out of scope.

Before this phase a plugin ran inside the judge's process, so it could read `data/` (holdout
prices) directly, mint an unlogged token through `_MINT_KEY`, rewrite `runs/lab.db`, the holdout
log or the verdicts, or patch the engine in-process to report better numbers. On this machine it
could also reach `/var/run/docker.sock`, which is root on the host, because the user is in the
`docker` group.

Access times, found after the first build: on a `relatime` mount (this machine's `/`), reading a
file updates its access time when that is a day old, and so does following or reading a symlink.
Reading the allowlist was allowed, so a worker could mark about a bit per allowlisted file or
symlink per day (`/usr` alone has hundreds of thousands), and a later worker, such as a later
family's exam, or any process could read the marks back with `stat()`: holdout data carried from
one exam to the next. A Landlock layer that denies every read does not close it, because resolving
a path follows its symlinks, and updates their access times, before Landlock refuses the open.
Hence the seal (Confinement, step 8): once its modules are loaded, a worker resolves no path.

## Decisions

| Question | Decision |
|---|---|
| Confinement ★ | Landlock (ABI ≥ 6) plus seccomp plus rlimits plus `no_new_privs`, applied by each worker process to itself while it is still single-threaded; then, once its modules are loaded and before any plugin byte is read, a second seccomp filter seals it: no path resolves. Linux only. No new pip dependency (ctypes; libseccomp comes from the OS). |
| No sandbox available | Fail closed: plugin code does not run. No CLI flag bypasses it. Running a plugin in-process exists only as a test fixture under `tests/`. |
| What crosses the boundary | In: plugin bytes, bars, params, engine params. Out: decisions, plus features for the lookahead check. Nothing from a worker is ever unpickled. |
| Who computes results | The trusted engine, replaying the decisions. |
| Exam and paper ★ | Streamed: the decision at bar t is made with bars ≤ t only. |
| Who starts exams ★ | Only `lab exam`, run by a human or a scheduled job. `lab judge` leaves a promoted plugin `pending` and returns, so the proposer that runs it can't time an exam. |
| Exam window on a retry | Fixed when the read is minted, stored in the holdout log, and reused on every retry of that read; at most `exam_max_attempts` attempts per read. |
| Runs that are compared | Each in a fresh worker sent only its own bars, and never run alongside a worker holding later bars. |
| Symbols | A worker is only ever sent one symbol's bars. |
| Plugin metadata ★ | `PARENT`, the docstring, `params`, `grid` and `timeframes` are read from the source with `ast`, never by running it, and must be literals. |
| Plugin failures ★ | A failure caused by the plugin is a recorded kill (`plugin_error`); in the exam it is `failed`. Infrastructure failures stay errors with nothing recorded, as today. |
| Paper | Streamed through a sandbox worker; a worker failure stops the paper run. |

Rejected:

- **bubblewrap or other user-namespace sandboxes.** Ubuntu 24.04 restricts unprivileged user
  namespaces through AppArmor; on this machine `bwrap` fails with `setting up uid map:
  Permission denied`. Enabling it needs root and a system-wide change.
- **Docker or podman.** The Docker daemon runs as root and membership in the `docker` group is
  itself root-equivalent; containers are heavy for thousands of short runs per judgement and
  are unavailable inside many CI containers.
- **Python-level restrictions** (audit hooks, RestrictedPython, import blockers). Not a security
  boundary: `ctypes`, `gc` and frame introspection get around them.
- **The engine outside the sandbox, calling the plugin per bar over IPC, for walk-forward.** A
  judgement runs thousands of backtests; a round trip per bar per run is far too slow. Replay
  gives the same scoring guarantee at the cost of running the engine loop twice.
- **Trusting a worker's results with a checksum.** The worker can forge the checksum too.

## Components

New package `perp_lab/sandbox/`, one job per file:

- `confine.py`: applies confinement to the calling process (`confine()`, then `seal()`);
  `probe()` reports whether this machine can (Landlock ABI, libseccomp present) without spawning
  anything.
- `worker.py`: entry point `python -s -P -m perp_lab.sandbox.worker`. Confines itself, imports
  `session` and everything a plugin may import, seals itself, runs its self-tests and serves
  requests.
- `session.py`: the **only** module that executes plugin code (split from `worker.py`, which
  must confine itself before it imports numpy).
- `protocol.py`: framing and validation. The parent-side reader is the trust boundary.
- `client.py` (trusted): spawns workers, enforces limits, classifies failures; batch runs
  (`Batch`, `fresh_run`).
- `stream.py` (trusted): the streamed exam, its lockstep and its one rerun.
- `decisions.py`, `replay.py` (trusted): decision logs, `ScriptedStrategy`, random-entry
  verification, fingerprint comparison.
- `errors.py`: the error types both sides share.

`perp_lab/plugins.py` (trusted): `PluginSource(name)` reads `strategies/<name>.py` once and
holds its bytes, its hash and its AST metadata.

`load_strategy_class` is gone: plugin code is loaded only by `session.load_plugin`, inside a
worker (or the tests' in-process twin). `plugin_parent` and the hypothesis come from
`PluginSource`. `RandomEntry` stays in `strategy.py`, used only inside the worker.

## Confinement

Landlock and seccomp confine the calling thread, and a thread that already exists stays
unconfined: plugin code could reach it through ctypes. Importing numpy starts threads (24 on this
machine without the thread-count variables; pyarrow's allocator adds one even with them). So the
worker confines itself **first**, while it is the only thread, with nothing but the standard
library imported. Until step 9 ends, only trusted code has run in it.

1. Checks that `/proc/self/task` has exactly one entry; otherwise it exits with an
   infrastructure error.
2. Lowers hard rlimits: `RLIMIT_AS` = `sandbox.memory_mb`, `RLIMIT_FSIZE` 0, `RLIMIT_CORE` 0,
   `RLIMIT_NOFILE` small (e.g. 64), and `RLIMIT_CPU` at the session's total budget as a backstop
   to the parent's own accounting.
3. `prctl(PR_SET_NO_NEW_PRIVS, 1)`. Before this step, in the worker's startup, it also sets
   its parent-death signal to SIGKILL and exits at once if its parent is already gone. EOF on
   the pipe reaches only an idle worker; the signal reaches a busy one, which may hold holdout
   bars, when the judge crashes or is interrupted.
4. Landlock: a ruleset that handles every filesystem right the running ABI knows, TCP bind and
   connect, and scopes abstract UNIX sockets and signals. The only rules grant read-file and
   read-dir beneath the interpreter's prefixes (`sys.base_prefix` and `sys.prefix`),
   site-packages and the `perp_lab/` package directory. Nothing else is readable: not `data/`,
   `runs/`, `config/`, `strategies/`, `.git`, `$HOME`, `/tmp`, `/run` or `/var`. The builder may
   add read-only system paths that numpy, pandas or pyarrow need at import or in the warm-up (e.g.
   `/usr/share/zoneinfo`, parts of `/sys/devices/system/cpu`, the worker's own `/proc` entry)
   and must list each one with its reason in the module. Where the ABI offers thread
   synchronisation (`LANDLOCK_RESTRICT_SELF_TSYNC`, ABI 8), it is used as well.
5. seccomp, through libseccomp loaded with ctypes, with filter thread synchronisation on. The
   default is allow. At least these return `EPERM`:
   - `socket`, `socketpair`;
   - `execve`, `execveat`; `fork`, `vfork`, and `clone` without `CLONE_THREAD` (`clone3` returns
     `ENOSYS` so libc falls back to `clone`);
   - System V IPC: `shmget`, `shmat`, `shmctl`, `shmdt`, `msgget`, `msgsnd`, `msgrcv`, `msgctl`,
     `semget`, `semop`, `semtimedop`, `semctl`. Landlock doesn't cover these; without this
     rule, one worker can leave bars in a shared-memory segment for another (tested);
   - POSIX message queues: `mq_open`, `mq_unlink`, `mq_timedsend`, `mq_timedreceive`,
     `mq_notify`, `mq_getsetattr`. Like System V IPC, Landlock doesn't cover them;
   - file watches: `inotify_init`, `inotify_init1`, `fanotify_init`. Landlock has no right for
     them;
   - file metadata: `chmod`, `fchmod`, `fchmodat`, `fchmodat2`, `chown`, `fchown`, `lchown`,
     `fchownat`, `utime`, `utimes`, `utimensat`, `futimesat`, and setting, removing, reading or
     listing extended attributes (`setxattr`, `lsetxattr`, `fsetxattr`, `removexattr`,
     `lremovexattr`, `fremovexattr`, `getxattr`, `lgetxattr`, `fgetxattr`, `listxattr`,
     `llistxattr`, `flistxattr`, and their `*at` forms); `ioctl` requests that set an inode's
     flags, attributes or generation (`FS_IOC_SETFLAGS`, `FS_IOC_FSSETXATTR`, `FS_IOC_SETVERSION`
     and their aliases) and `fcntl(F_SET_RW_HINT)`. Landlock has no right for a file's metadata,
     so without these rules a worker could leave bytes in the mode bits, timestamps or extended
     attributes of any file this user owns, or the flags of any it can open, for a later worker
     to read (tested);
   - file locks, leases and notifications: `flock`, and `fcntl` with `F_SETLK`, `F_SETLKW`,
     `F_OFD_SETLK`, `F_OFD_SETLKW`, `F_SETLEASE` or `F_NOTIFY` (other commands, such as `F_GETFL`
     or `F_SETFD`, keep working). A read-only descriptor on an allowlisted file is enough to
     lock it, so two workers could signal each other through it while both run (tested);
   - calls aimed at another process: `ptrace`, `process_vm_readv`, `process_vm_writev`,
     `process_madvise`, `kcmp`, `pidfd_open`, `pidfd_getfd`, `migrate_pages`, `move_pages`.
     `prlimit64` and the `sched_*` calls that take a pid are allowed only on the calling
     process (pid 0). `getpriority`, `setpriority`, `ioprio_get` and `ioprio_set` are allowed
     only with `which` set to process and `who` 0, because the user-wide forms reach every
     process of this user. Without these rules a worker can lower the judge's limits through
     `getppid()` (tested);
   - `prctl(PR_SET_PDEATHSIG)`: the worker sets its parent-death signal before it confines
     itself, and plugin code may not clear or change it;
   - `mount`, `umount2`, `pivot_root`, `chroot`, `unshare`, `setns`; `bpf`, `perf_event_open`,
     `userfaultfd`, the `io_uring_*` calls; `keyctl`, `add_key`, `request_key`; module loading
     and `kexec` calls.

   Every syscall number above the highest this libseccomp can name, up to 1024, returns `ENOSYS`:
   on a kernel newer than the library, calls such as `setxattrat` or `file_setattr` have no name
   a rule could use, and `ENOSYS` is what an older kernel answers, which callers handle. A test
   exports the filter without loading it and runs it over every rule above.

   A syscall through a non-native ABI (i386, x32) kills the process (`SCMP_ACT_KILL_PROCESS`). Sockets are denied outright
   because Landlock's network rules cover TCP only: UDP and existing UNIX sockets such as the
   Docker socket are outside them.
6. Preload and warm-up. Imports numpy, pandas, pyarrow, `perp_lab.strategy`,
   `perp_lab.indicators` and `perp_lab.engine.backtest`; numpy's public submodules that numpy 2
   otherwise loads on first use (`numpy.rec`, which pandas reads in `isna`, and `char`,
   `ctypeslib`, `fft`, `polynomial`, `strings`); the pandas modules pandas imports inside
   methods a plugin may call (`to_dict`; `stack`, `unstack` and `crosstab`; `to_csv`); and
   `socket`, for the self-test. A test scans numpy and pandas for any other module their
   functions import that the preload leaves out, each exception listed with its reason (plotting,
   optional-dependency I/O, test helpers), so an upgrade can't add one silently. Then it warms
   up the lazy paths it takes later, one comment each in `worker.py`: an Arrow decode of a small
   frame shaped like bars (a UTC index, int, float and string columns), which loads pyarrow's
   conversion and timezone code, and one `repr` of a DataFrame, which loads pandas' formatter (a
   plugin's `print`). Threads these start inherit the confinement. These modules, and whatever
   they load, are all a plugin can import, besides the interpreter's built-in and frozen modules,
   which need no file: once sealed, nothing more can be loaded (`ModuleNotFoundError`).
7. Self-test, stage 1: opening the parent's config file must fail with `EACCES` (Landlock), and
   creating a socket, `os.fork()` and `shmget` must each fail.
8. Seal: a second seccomp filter, with filter thread synchronisation on (so it reaches the
   threads numpy and pyarrow started), the same kill-process on a non-native ABI, and default
   allow. Every syscall that resolves a pathname returns `EPERM`: `open`, `openat`, `openat2`,
   `creat`, `stat`, `lstat`, `newfstatat`, `statx`, `statfs`, `access`, `faccessat`,
   `faccessat2`, `readlink`, `readlinkat`, `chdir`, `truncate`; every call that makes, removes,
   renames or links a name (`mkdir`, `mknod`, `rmdir`, `unlink`, `rename`, `link`, `symlink` and
   their `*at` forms, `renameat2`); `execve`, `execveat`, `uselib`, `acct`, `swapon`, `swapoff`,
   `quotactl`, `chroot`, `pivot_root`, `mount`, `umount2`, `open_tree`, `move_mount`, `fsopen`,
   `fsconfig`, `fspick`, `mount_setattr`, `inotify_add_watch`, `fanotify_mark`,
   `name_to_handle_at`, `open_by_handle_at`; and the path forms of the metadata and
   extended-attribute calls (`chmod`, `fchmodat`, `fchmodat2`, `chown`, `lchown`, `fchownat`,
   `utime`, `utimes`, `utimensat`, `futimesat`, `getxattr`, `setxattr`, `listxattr`,
   `removexattr`, their `l*` forms, and the newer `setxattrat`, `getxattrat`, `listxattrat`,
   `removexattrat`, `file_getattr`, `file_setattr` and `open_tree_attr`). Names a libseccomp can't
   resolve on this architecture are skipped, as in step 5; those newer than it already get
   `ENOSYS` from step 5. A Landlock layer would not do: resolving a path follows its symlinks,
   which updates their access times, before Landlock refuses anything. Once sealed the worker
   opens, stats and reads no file, follows no symlink and names no path; it holds its pipes and
   what it has loaded, and gets the rest through the pipe. The same filter denies `mincore`,
   `cachestat`, and `madvise` with advice that evicts or prefetches pages (R7); other advice, which
   allocators use (`MADV_DONTNEED`, `MADV_FREE`, `MADV_HUGEPAGE`), keeps working. `fstat` on a
   descriptor keeps working where libc makes it with the `fstat` syscall (glibc 2.39 and later);
   older glibc makes it with `newfstatat`, so step 9 fails there and the sandbox is unavailable
   (`probe()` reports it up front). `seal()` refuses unless a seccomp filter (stage 1) is already
   in place: stage 2 alone would leave everything stage 1 denies allowed.
9. Self-test, stage 2: opening and stat-ing a file on the allowlist (numpy's `__init__.py`) and
   `readlink` on the interpreter's `python3` link (under `/usr` for a system Python) must each
   fail with `EPERM`, which only the seal answers; `fstat` on a protocol pipe must work; and every
   descriptor the worker holds (up to `RLIMIT_NOFILE`) must be a pipe, since one on a file would
   let it read the file without opening it. The worker then reports `ready` with its Landlock
   ABI. A failure in steps 1–9 is an infrastructure error.

The parent launches the worker as `python -s -P -m perp_lab.sandbox.worker`, not `-I`: `-I`
ignores every `PYTHON*` variable, so `PYTHONHASHSEED` would have no effect and fresh workers in
one lookahead check would hash strings differently (tested). The environment is exactly
`LANG=C.UTF-8`, `PYTHONHASHSEED=0` and `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` = 1; cwd `/`; `close_fds`; a new session; pipes for
stdin and stdout; stderr into a buffer capped at 64 KiB. After `ready`, the worker moves the protocol to private file descriptors and points fd 1 at
stderr, so a plugin's `print` cannot corrupt the framing. (A plugin can still write to the
protocol descriptors on purpose; the result is a protocol violation, nothing more.)

Requirements: Linux with Landlock ABI ≥ `sandbox.min_landlock_abi` (6, kernel 6.12 or later),
`libseccomp.so.2` and glibc 2.39 or later (step 8); `probe()` checks all three. This machine has
Ubuntu 24.04, kernel 7.0, Landlock ABI 8, libseccomp 2.5.5 and glibc 2.39.

The parent caps concurrent workers so that workers × `memory_mb` stays within about three
quarters of physical memory, whatever `--workers` asks for.

## Protocol

Length-prefixed frames on the pipes: a 4-byte length, a JSON header of at most 64 KiB, then the
binary sections the header declares.

- Parent to worker: may carry Arrow IPC tables (the worker is the less trusted side).
- Worker to parent: JSON plus raw little-endian arrays of a declared dtype (`int64`, `int8`,
  `float64`) and length, read with `numpy.frombuffer` after every size has been checked against
  `sandbox.max_output_mb`. No pickle, no Arrow, no other parser on this side.

Requests:

- `load {name, source}` → `loaded {params, grid, timeframes}`, the values at runtime. The parent
  requires them to equal the AST values; otherwise `plugin_error: metadata` (a plugin that
  rewrites its own parameters at import).
- `bars {symbol, tf, table}` → `ok`. Replaces the bars the worker holds.
- `run {params, engine_params, mode, k?, features?}` → `decisions` (arrays `i`, `action`, `stop`,
  `target`, `has_stop`, `has_target`; the masks are needed because the engine treats `None` and
  NaN differently: a held NaN target clears the target, `None` leaves it, and
  `bb_vwap_reversion` returns NaN targets), `fingerprint {trades, final_equity}`, and, if asked,
  `features` (column names plus
  one float64 array each; bool and int are cast, anything else is `plugin_error: features`).
  `k` runs on the first k bars only. `mode` is `plain` or `random {seed, prob}`.
- `stream_open {params, warmup_k}` → the decisions of a batch run over the first `warmup_k` bars.
  Then `step {new_rows, position}` → `signal {action, stop, target}`. A stop or target is a number,
  finite or NaN; an infinite one, in a decision log or a signal, is a protocol violation.
- Any failure inside plugin code → `error {kind, type, message}`, with `message` cut to 200
  printable ASCII characters.

## Batch runs and replay

Used by walk-forward, sensitivity, the random-entry benchmark, `lab backtest` and the
dashboard's paper divergence.

- **Decision log.** Every signal the plugin returns other than a bare `HOLD`: bar index, action,
  stop, target. Holds that move the stop or the target are included.
- **Replay.** `ScriptedStrategy(log)` returns the logged signal at bar i and `HOLD` elsewhere. The
  parent runs the unchanged `run_backtest` with it on the same bars and `EngineParams`; that run's
  trades and equity are the result.
- **Fingerprint.** The worker also reports its own trade count and final equity. Any difference
  from the replay is `plugin_error: mismatch`. For an honest plugin they are identical: same
  engine code, same inputs.
- **Random entry.** The parent makes the draw from the seed itself (the same draw as
  `RandomEntry.prepare`). A long or short logged while the replay is flat is valid only on a bar
  where the draw fires, with the drawn side; anything else is `mismatch`. A long or short logged
  while the replay holds a position comes from the plugin's own `manage()` and is a reversal,
  which the engine allows, so it is accepted as in any run. A firing bar with no logged entry is
  allowed, since the plugin's bracket may have no valid stop there. The plugin still supplies the
  brackets and the exits, so it can still sabotage its benchmark (R1).
- **Walk-forward shape.** The `ProcessPoolExecutor` stays and runs trusted code only. Each pool
  worker keeps one sandbox worker per symbol for the run's plugin version (jobs are ordered by
  symbol, so it rarely holds more than one), sends it that symbol's bars once and replays every
  decision log it gets back. A sandbox worker never sees a second symbol: one that held BTC's
  whole series could look at BTC's next bar while deciding ETH, which the single-symbol
  lookahead check cannot see. `run_one` returns the same compact result as today, so the
  walk-forward, sensitivity and benchmark logic is unchanged.
- **Cost.** `prepare` runs once per job, as today; the engine loop runs twice (in the worker and
  in the replay); memory per job roughly doubles.

## Lookahead check

`assert_causal`, `causality_failures` and `dated_lookahead_failures` keep their logic, variants
and cut points, and get their features and decisions from the sandbox. The full run and every
cut run each go to a **fresh worker sent only its own bars** (`bars[:k]` for a cut). The
comparison happens in the parent.

Why fresh workers: a worker that has run the full series could remember it and return the full
run's features on the cut runs, so the check would pass while the plugin peeks. A worker that
has only ever been sent `bars[:k]` cannot know what comes after k, provided no worker holding
later bars runs at the same time (they could signal through CPU timing). So the runs go in waves
of increasing length: every variant's run at the first cut together, then the next cut, and the
full runs last.

Cost: (1 + cuts) worker starts per parameter variant, about a second each before the run itself;
variants run in parallel within a wave.

## Streamed exam ★

This replaces `run_exam`'s single batch run per symbol. Each symbol gets a fresh worker:

1. **Warm-up.** The worker is sent bars from `history_start` to the last bar before
   `holdout_start`, and runs them in batch. The parent replays those decisions. The engine
   makes no decision on a run's last bar, so the warm-up covers every bar before the last
   research bar.
2. **Stream.** From the last research bar to the window's end, one bar at a time: the parent
   appends the next bar (none for the first step), sends the trusted engine's position, and the
   worker runs `prepare` on everything it holds and `on_bar` on its last bar. The parent's
   engine executes the signal.

The worker never holds a bar later than the one it is deciding on. For a causal plugin this
gives exactly the decisions of 3a's batch exam (a test pins this); a plugin that would peek has
nothing to peek at.

Symbols' workers that run at the same time advance in **lockstep**: every one decides bar t
before any is sent bar t+1, so none can pass another a later bar through CPU timing. All
warm-ups finish before streaming starts. During warm-ups research bars are not ordered across
symbols; no holdout bar is affected. When the memory cap allows fewer workers than symbols
(7 × 3 GiB on this machine, for 8 symbols), symbols go in batches, one batch after another, each
in lockstep.

**One exam at a time.** `attempt_exam` holds a machine-wide lock (`<holdout_log>.exams.lock`,
taken without waiting) besides its judgement's own, for the whole attempt. While another exam
runs, any judgement is reported `in_progress`, unchanged and with nothing spent, so exams of
different families, which would not be in lockstep with each other, never overlap.

Each stream step has its own limit, `sandbox.stream_step_s` of CPU and of wall-clock time,
tight compared to the batch limit because one step is one `prepare`. A plugin that spins fails
the step as `cpu_limit`, not retried: with equal limits the wall clock may trip first, so a step
that has used at least 90% of its CPU limit by then is a busy loop, not lost time. One that
sleeps fails it as `timeout`, and after the one retry (which restarts the whole batch in fresh
workers: restarting one symbol would put it behind the others and break lockstep) the exam fails
with `plugin_error`.

The retry is held to the lost run: it must repeat every decision that run made, symbol by symbol,
up to the bar where each symbol stopped (the warm-up's decisions, then each step's signal). Any
difference is a `mismatch`, and the exam fails. Otherwise a plugin that has seen holdout bars
could get a second draw: one that decides at random there, or that kills its own worker once it
dislikes how its exam is going.

Cost: one `prepare` over the whole history per holdout bar. Measured on synthetic bars,
`prepare` plus building the arrays takes about 6 ms at 40k bars (1h), 25 ms at 160k (15m) and
60–75 ms at 470k (5m). For a 180-day window per symbol: 4h about 3 s, 1h about 30 s, 15m about
7 min, 5m about 65 min. Symbols in a batch run in parallel. A one-year window doubles these. At
most 10 exams ever run.

**Only `lab exam` runs exams.** `lab judge` records a promotion as `pending` (`awaiting_exam`)
and returns; a human or a scheduled job runs `lab exam`. The proposer runs `lab judge`, so it
must not be able to time an exam: a plugin controls how long its exam takes and could spell out
holdout data in that duration. This changes 3a's step 10.

**The window is fixed at the mint.** `budget.mint` writes the window end into the read's
holdout-log entry, and every retry of that read uses the stored end. Otherwise a retry after a
crash would run on a longer window, and a plugin that sees its running results could prefer a
retry. A read gets at most `judge.exam_max_attempts` attempts (each retry appends a log entry,
as in 3a); after that the judgement is `failed` with `exam_error`.

`attempt_exam` runs, in order:

1. `plugin_changed` check (as in 3a). The plugin's bytes are read once here, and those bytes are
   what every later step sends.
2. If the judgement already holds a read (`budget.holds_read`): the stored window end, and on to
   step 6. The read is spent and the window fixed, so the checks in between can only strand it.
3. Window end.
4. The dated lookahead check (as in 3a, now sandboxed). It still matters: it stops a plugin whose
   promotion came from peeking from spending a read.
5. **Cost probe:** time `prepare` on the full research bars in a separate worker and multiply by
   the window's bar count. Above `sandbox.exam_cpu_hours` the judgement is `blocked` with reason
   `too_slow` and no read is spent.
6. **Sandbox check:** spawn a worker and wait for `ready`. If that fails, nothing is minted (or,
   on a retry, no attempt is used) and the judgement stays `pending` with `exam_error`.
7. Mint, as in 3a, storing the window end; on a retry, the existing read. Just before it the
   judgement is read again: one no longer `pending` (superseded meanwhile) stops there, with no
   read. A promotion supersedes its parent only under the parent's exam lock, taken without
   waiting; while `lab exam` holds it, the parent is left alone and its exam goes ahead.
8. The streamed exam, in the judge's process (the token never leaves it).

## Paper

`PaperBook` keeps the position and the fills, as today. On each closed bar the runner sends the
window it already holds (`warmup_bars` rows) and the position to a stream worker and gets the
signal back. Each symbol has one worker for the life of the paper run. A worker failure stops
the paper run with an error, and a human restarts it.

## Plugin metadata without running code

`PluginSource(name)`:

- **Hash:** the git blob SHA-1 of the bytes it read. This equals `git hash-object` for these
  files, so existing judgements keep their keys; a test pins the equality for every current
  plugin. The `git` subprocess goes away.
- **Hypothesis:** the first line of the module docstring.
- **PARENT:** a single module-level `PARENT = "<string literal>"`, or none. Anything else (not a
  literal, assigned more than once, assigned inside a branch) is refused (exit 2).
- **Class:** exactly one class whose bases include the name `Strategy`. Its `params`, `grid` and
  `timeframes` must be literals (`ast.literal_eval`); otherwise refused (exit 2). All four
  current plugins already comply.

Users: `lab strategies` (no plugin code runs at all), family resolution, the judge's pre-checks
and the hypothesis in run records.

Within one `lab judge`, one `PluginSource` per plugin flows through every step: the lookahead
check, the walk-forward, sensitivity and the benchmark. 3a's checks for a plugin edited mid-run
(`validate_strategy`'s second hash, `NAME changed while it was being judged`, `parent P changed
while it was being checked or run`) can no longer fire; the builder may remove them or keep them
as assertions.

## Where plugin code ran before this phase, and now

| Before | Now |
|---|---|
| `lab strategies` loads every plugin | AST only |
| `lab backtest` | batch sessions plus replay |
| `lab paper` / `PaperRunner` | stream sessions |
| `walkforward.run_one` (walk-forward, sensitivity, random entry) | batch plus replay in each pool worker |
| `validate_strategy`: hypothesis, hash at start and end | `PluginSource` |
| `judge`: `plugin_parent`, lookahead checks | AST, and the sandbox with fresh workers |
| `exam.run_exam` | streamed exam |
| `paper/tracking.replay_divergence` (the dashboard) | batch plus replay |
| tests | may still load honest plugins in-process, through a fixture in `tests/` |

## Failures

**Before the plugin is loaded** (spawn, confinement, self-test): infrastructure. `lab judge`
exits 1 and records nothing; an exam mints nothing. `lab judge` checks the sandbox before doing
anything else.

**After it is loaded**, the failure is the plugin's:

- an exception in plugin code, including `MemoryError` at `memory_mb`;
- CPU time over `sandbox.task_cpu_s` for one batch run or one stream step, measured by the parent
  from `/proc/<pid>/stat`;
- a protocol violation, oversized output, or a metadata, features or replay `mismatch`, including
  equity that is not finite after a replay, a streamed run or a paper fill (a day whose return is
  NaN would be dropped from the portfolio, not counted against the plugin);
- the worker dying without a report, or a run exceeding `task_cpu_s × wall_factor` of wall-clock
  time (or `stream_step_s` for a stream step): this is retried once in a fresh worker, and only
  a second failure on the same task counts (`died` or `timeout`). The retry absorbs transient
  machine trouble.

Where it is recorded:

- **Lookahead check:** as in 3a, a failing variant is a `lookahead:` reason, e.g.
  `lookahead: defaults: plugin_error: cpu_limit`.
- **Walk-forward, sensitivity, benchmark:** the judgement is `kill` with reason
  `plugin_error: <kind>`, plus `<Type>: <message>` for exceptions. These runs only ever see
  research data, so the message may reach the verdict. No run is recorded, so it is not a trial.
- **Parent plugin:** a plugin error in the parent's own runs refuses the child (exit 2), like
  3a's parent lookahead rule: `parent P fails in the sandbox; fix or judge it first`.
- **Exam:** `holdout_status` `failed`, reason `plugin_error`. The read is spent. Neither the kind
  nor the message reaches the verdict; the human sees both in the holdout run's report.
- **Paper:** the run stops with an error.

A plugin-error kill is final for that version: the judgement is keyed by the plugin's hash, so
judging it again returns the same kill. Re-judging needs an edited plugin.

A worker's stderr is kept, capped, in the run directory for humans. It never reaches a verdict.

New holdout reason codes: `plugin_error`, `too_slow`.

## Config (`lab.yaml`)

```yaml
sandbox:
  memory_mb: 3072        # per worker (RLIMIT_AS)
  task_cpu_s: 900        # one batch run (one params x symbol)
  wall_factor: 4         # its wall-clock limit = task_cpu_s x wall_factor
  stream_step_s: 30      # one stream step, CPU and wall clock (a 5m step takes ~0.1 s)
  exam_cpu_hours: 6      # per symbol; the cost probe blocks an exam estimated above it
  max_output_mb: 64      # largest message a worker may send
  min_landlock_abi: 6

judge:
  exam_max_attempts: 3   # attempts per holdout read, then failed / exam_error
```

## Testing

Each test is written first and seen failing. Tests that need the kernel features skip with
`sandbox unavailable: <probe reason>` where `probe()` fails; on this machine they all run.

- **Confinement,** with a hostile fixture plugin and checked from outside the worker where
  possible. Each of these fails: reading a file under `data/`, `runs/lab.db` or the holdout log;
  writing a sentinel file (the test checks it doesn't exist); a TCP connect to a listener the test
  runs (the listener sees no connection); a UDP send; connecting to a UNIX socket the test
  creates; `os.fork`, `subprocess.run`; `os.kill` on the parent (the parent survives);
  `chmod`, `chown`, `utime` and extended attributes on a scratch file the test owns (its mode,
  times and attributes are unchanged afterwards), an inode-flag `ioctl`, and a syscall newer than
  libseccomp (`ENOSYS`). A
  sentinel secret set in the parent's environment is absent in the worker. Minting a token
  through `_MINT_KEY` inside the worker succeeds but `load_bars` with it fails, because `data/` is
  unreadable. A System V shared-memory segment the test creates can't be attached, and one worker
  can't leave a segment for the next. `prlimit` on the parent fails and the parent's limits are
  unchanged. Every thread the worker has after its imports (`/proc/self/task`) is denied a read
  outside the allowlist. The inode-flag `ioctl` is checked in a process with stage 1 only, since a
  sealed worker can open no file to try it on.
- **The seal,** through real workers unless noted. A plugin's `open`, `stat`, `lstat`, `readlink`
  and `listdir` of files and symlinks on the allowlist each fail with `EPERM`, and so does an
  `open` at its module level: the seal comes before any plugin code. Checked from outside, the
  access times of a scratch file in the virtualenv and of a symlink to it stay where the test set
  them after a plugin tries `stat`, `readlink` and `open` through them (a read from outside moves
  both first, so the mount does record them). A worker holding a descriptor that isn't a pipe is
  unavailable; `seal()` without `confine()` refuses; `probe()` reports glibc before 2.39 or
  another C library. Every syscall name this libseccomp resolves is classified as resolving a
  path (the seal denies it), always denied by stage 1, page cache, or listed as path-free, so a
  newer libseccomp can't leave a path call allowed unseen. A plugin can import
  what the worker preloaded and print a frame, and importing a module it didn't preload is
  `plugin_error: exception: ModuleNotFoundError`; so is `perp_lab.holdout`, so no token can be
  minted inside a worker. A sealed plugin's `mincore` and its eviction and prefetch `madvise`
  calls get `EPERM`, while `MADV_DONTNEED` and allocation work. The exam evicts every file its
  workers mapped after each attempt, passed or failed, once they are closed (with the twin and a
  spy), and does so after a real streamed exam with no worker left running; eviction drops a
  file's pages from the cache. A worker whose seal silently did nothing reports unavailable. In a probe
  process, every thread, those started before the seal included, is refused an allowlisted file
  once sealed; a test exports the stage-2 filter and runs it over every rule.
- **Determinism:** two fresh workers give `hash()` of a string the same value.
- **Fail closed:** with `probe()` failing, `lab judge` exits 1 and records nothing, and
  `attempt_exam` doesn't mint.
- **Single executor:** no module in `perp_lab/` or `dashboard/` except `perp_lab/sandbox/session.py`
  calls `exec`, `compile`, `exec_module`, `spec_from_file_location`, `load_plugin` or
  `import_module` (AST scan anchored on the repo root, so it can't pass vacuously from another
  directory).
- **Metadata:** AST values equal runtime values for every real plugin; a computed `PARENT` and a
  computed `grid` are refused; a plugin that changes `params` at import gets
  `plugin_error: metadata`.
- **Hash:** `PluginSource` hash equals `git hash-object` for every real plugin.
- **Replay:** for every real plugin on synthetic bars, sandbox plus replay gives trades and
  equity identical to today's in-process run; a forged fingerprint is `mismatch`; a random-entry
  log with an entry while flat where the draw doesn't fire is `mismatch`, while a fixture plugin
  whose `manage()` reverses passes its benchmark; a plugin returning NaN and `None` targets on
  holds replays identically.
- **One symbol per worker:** a plugin that keeps the first symbol's series in a module global
  finds it empty when deciding the second.
- **Streaming:** for every real plugin, streamed decisions equal batch decisions. An oracle
  plugin that trades on the next bar's return has a large Sharpe in batch and none to speak of
  streamed, on a long synthetic random walk.
- **Fresh workers:** a memoising cheater (caches the full run's features in a module global and
  returns them on later runs) is caught by the lookahead check.
- **Failures:** a plugin that raises on holdout-dated bars gets `failed` / `plugin_error` with no
  message in the verdict; `os._exit` is retried once, then `plugin_error: died`; a busy loop is
  `cpu_limit`; a plugin that sleeps in a stream step fails the exam; the cost probe blocks a slow
  plugin as `too_slow` without a read.
- **Exams:** `lab judge` on a promotable plugin leaves it `pending` and runs no exam; a retry on a
  held read uses the stored window end even after more data has arrived, skips the cost probe,
  and the read fails with `exam_error` after `exam_max_attempts`.
- **End to end** on small synthetic data with a temp config and store, through the real sandbox:
  a kill, a promotion left `pending`, and an exam examined and passed.

## Changes made to existing documents

- The 3a spec: the caveat in its Goal and the last bullet of its carried requirements point to
  this spec. Its step 10 changed: `lab judge` no longer attempts the exam. Its holdout budget
  section gained the stored window end and the attempt cap.
- README: the rules table gained the sandbox rows; the "(plugin code is not sandboxed yet …)"
  qualifier went.
- The `strategy.py` contract docstring: metadata must be literals, features numeric, a plugin
  gets only the bars it is sent, and it can import only the modules the worker preloads.

## Out of scope / follow-ups

- **Narrowing R1:** while judging, re-run the selected params streamed over a secret, randomly
  chosen stretch of real OOS data and require decisions identical to the batch run. This defeats
  date- or shape-based evasion of the synthetic check by any plugin that peeks systematically.
  It costs about one exam-length streamed run per judgement (about an hour for one 5m symbol and
  180 days; one random symbol would do).
- Hosts without Landlock ≥ 6, including macOS.
- Storing the judged plugin bytes so an edited file no longer holds an exam as `plugin_changed`.

## Requirements carried into the Phase 3b-2 proposer spec

- The proposer can write only to `strategies/`; in particular not to `perp_lab/`, `config/`,
  `tests/` or `.venv/`, any of which would let it weaken the sandbox or its limits.
- Plugins it writes keep `PARENT`, `params`, `grid` and `timeframes` literal, or `lab judge`
  refuses them.
- It learns of exam failures only as reason codes.
- It never runs `lab exam`, and it gets verdicts as their contents only: no file timestamps, and
  nothing else that tells it when an exam ran or how long it took (R2).
- It can read the harness source, so assume it knows exactly how the lookahead check works (R1).
- R5 applies to the proposer's own process too: it must not be able to `stat()` `data/` (the sizes
  and modification times of holdout files).
