# Plugin Sandbox (Phase 3b-1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Plugin code runs only in Landlock + seccomp confined worker processes; every score comes from the trusted engine replaying the plugin's decisions; the holdout exam is streamed one bar at a time and runs only through `lab exam`.

**Architecture:** A new `perp_lab/sandbox/` package. `worker.py` confines itself first, then `session.py` (the only module that executes plugin code) serves requests over a framed pipe protocol (`protocol.py`). The trusted side (`client.py`) spawns workers, enforces limits, classifies failures and replays decision logs (`decisions.py`, `replay.py`); `stream.py` runs lockstep streams for the exam. `perp_lab/plugins.py` reads plugin metadata with `ast`. Walk-forward, the lookahead check, the judge, the exam, `lab backtest`, paper and the dashboard switch to these; the in-process loader disappears from production code.

**Tech Stack:** Python 3.12, numpy, pandas 3, pyarrow (bars to the worker), ctypes (Landlock syscalls, `libseccomp.so.2`), pytest. No new pip dependencies.

**Spec:** `docs/superpowers/specs/2026-10-17-plugin-sandbox-design.md` (read it with this plan; section names below refer to it). It builds on `docs/superpowers/specs/2026-10-16-judge-design.md`.

## Global Constraints

- No new pip dependencies. Landlock via `ctypes` raw syscalls (x86_64 and aarch64: `landlock_create_ruleset` 444, `landlock_add_rule` 445, `landlock_restrict_self` 446); seccomp via `ctypes.CDLL("libseccomp.so.2")` using `seccomp_rule_add_array` (never the variadic `seccomp_rule_add`).
- Linux only; Landlock ABI ≥ `sandbox.min_landlock_abi` (6) and `libseccomp.so.2`, or plugin code does not run. No CLI flag bypasses the sandbox.
- Config values, verbatim: `sandbox: {memory_mb: 3072, task_cpu_s: 900, wall_factor: 4, stream_step_s: 30, exam_cpu_hours: 6, max_output_mb: 64, min_landlock_abi: 6}`; `judge.exam_max_attempts: 3`.
- Worker launch: `[sys.executable, "-s", "-P", "-m", "perp_lab.sandbox.worker", "--deny-probe", <config file path>]`, never `-I`; env exactly `LANG=C.UTF-8`, `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`; `cwd="/"`, `close_fds=True`, `start_new_session=True`. (If jemalloc's reservations trip `RLIMIT_AS`, add `ARROW_DEFAULT_MEMORY_POOL=system` and say so in `client.py`.)
- Nothing a worker sends is unpickled or parsed as Arrow. Worker → parent is JSON plus raw arrays of dtype `<i8`, `<i1`, `<f8`.
- Only `perp_lab/sandbox/session.py` executes plugin code (`exec`, `compile`, module loading). The spec names this "the worker"; it is split from `worker.py` because `worker.py` must confine itself before importing numpy.
- Every caller reaches workers through the module attributes `client.spawn` and `client.check` (write `from perp_lab.sandbox import client` and call `client.spawn(...)`), so tests can swap in the in-process twin.
- Holdout reason codes are now: `n/a`, `awaiting_exam`, `too_short`, `budget_spent`, `family_examined`, `superseded`, `exam_error`, `plugin_changed`, `lookahead`, `too_slow`, `plugin_error`, `passed`, `failed`.
- A plugin error's message never reaches a verdict from an exam. From any other stage it may (`PluginError.reason()`).
- Worker stderr (capped at 64 KiB per worker) never reaches a verdict. A batch worker with any stderr writes it to `<stderr_dir>/<symbol>-<pid>.txt`. `validate_strategy` uses `<runs_dir>/<run_id>/sandbox_stderr/`, which is a run's directory, and `lab ingest` ignores it until `summary.json` exists. Exams use `<runs_dir>/_exam_stderr/judgement-<id>/`. The last 2 KiB also travel on `PluginError.stderr`, for the judge's log.
- Tests: `.venv/bin/pytest -q` from the repo root. Tests that need the real sandbox carry `@pytest.mark.sandbox` and skip with `sandbox unavailable: <reason>` when `confine.probe()` reports one; every other test runs on the in-process twin (Task 5). Each test is written first and seen failing.
- One task = one commit, on branch `spec/plugin-sandbox`. Every commit message ends with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_012ois2oyK8ceZgshdyc9Z57
  ```

## Review Focus

Inputs the spec implies but doesn't spell out, most likely to bite first. Each has a test in the task that owns the code:

1. `bb_vwap_reversion` returns a NaN target on holds when rolling volume is 0, and the engine treats NaN and `None` differently: the replay must still match. → Task 2, `test_nan_and_none_targets_replay_identically`.
2. A symbol with missing bars in the holdout (SOL/XRP had gaps in 2022): lockstep must not deadlock and must still never let one worker get ahead. → Task 9, `test_lockstep_with_a_symbol_missing_bars`.
3. `lab judge --workers 24` on a 30 GB machine: concurrency is capped by memory, not by the flag. → Task 5, `test_max_workers_caps_by_memory`.
4. A plugin whose features include a column named like a bar column (`close`, `funding_charge`) changes the worker's engine prices: that must be a `mismatch`, never silently different numbers. → Task 2, `test_a_feature_named_close_is_a_mismatch`.
5. Ctrl-C or a crash in the parent must not leave confined workers running (they are in their own session, so SIGINT doesn't reach them). → Task 5, `test_worker_exits_when_its_parent_goes_away`.

---

### Task 1: Plugin source and metadata without running code

**Files:**
- Create: `perp_lab/plugins.py`
- Modify: `perp_lab/strategy.py` (`plugin_hash` without `git`; delete `plugin_parent`)
- Modify: `perp_lab/cli.py` (`strategies` command), `perp_lab/validate/pipeline.py` (hypothesis), `perp_lab/judge/judge.py` (`resolve_family`, pre-checks)
- Modify: `tests/test_lookahead.py` (`test_plugin_has_one_sentence_hypothesis`)
- Test: `tests/test_plugins.py`

**Interfaces:**
- Produces: `blob_hash(data: bytes) -> str` (git blob SHA-1); `PluginMeta(parent: str | None, hypothesis: str, params: dict, grid: dict, timeframes: list[str])` (frozen dataclass); `PluginSource(name: str, text: bytes, hash: str, meta: PluginMeta)` (frozen dataclass) with `PluginSource.read(name) -> PluginSource` (one `read_bytes()` of `strategy.plugin_path(name)`) and `PluginSource.from_text(name, text: str | bytes) -> PluginSource`; `parse_meta(text: bytes, name: str) -> PluginMeta`; `PluginMetaError(ValueError)` with `.name`, `.hash`.

- [ ] **Step 1: Write the failing tests** in `tests/test_plugins.py`:

```python
@pytest.mark.parametrize("name", REAL_PLUGINS)
def test_metadata_from_the_source_equals_the_loaded_class(name, monkeypatch):
    monkeypatch.setattr("perp_lab.strategy.STRATEGY_DIR", REPO / "strategies")
    src, cls = PluginSource.read(name), load_strategy_class(name)
    assert (src.meta.params, src.meta.grid, src.meta.timeframes) == (cls.params, cls.grid, cls.timeframes)
    assert src.meta.hypothesis == cls.hypothesis() and len(src.meta.hypothesis) > 20

@pytest.mark.parametrize("name", REAL_PLUGINS)
def test_hash_equals_git_hash_object(name):
    p = REPO / "strategies" / f"{name}.py"
    assert blob_hash(p.read_bytes()) == subprocess.check_output(["git", "hash-object", str(p)], text=True).strip()

def test_parent_is_read_without_running_the_module():
    text = '"""h"""\nraise SystemExit("ran")\nPARENT = "donchian_atr"\nclass A(Strategy):\n    params = {}\n'
    assert PluginSource.from_text("x", text).meta.parent == "donchian_atr"

@pytest.mark.parametrize("snippet", [
    'PARENT = "a" + "b"', 'PARENT = "a"\nPARENT = "b"', 'if True:\n    PARENT = "a"', 'PARENT = 3'])
def test_a_parent_that_is_not_one_string_literal_is_refused(snippet): ...   # PluginMetaError

@pytest.mark.parametrize("body", ['grid = {"n": list(range(3))}', 'params = dict(n=1)', 'timeframes = TFS'])
def test_class_metadata_must_be_literal(body): ...                         # PluginMetaError

def test_exactly_one_strategy_class(): ...   # zero classes and two Strategy subclasses both raise PluginMetaError

def test_a_judge_of_a_plugin_with_computed_metadata_is_refused(cfg, plugins): ...
    # add_plugin with grid = {"n": list(range(3))}; judge raises JudgeRefused naming the plugin
```

Defaults when absent: `params {}`, `grid {}`, `timeframes ["1h"]` (the `Strategy` class defaults); `parent None`; `hypothesis ""`. The class is the single `ClassDef` whose bases include `Name("Strategy")` or an `Attribute` ending in `.Strategy`; each of `params`, `grid`, `timeframes` is assigned at most once in its body and must pass `ast.literal_eval`. `PARENT`: every `Assign`/`AnnAssign`/`AugAssign` to the name `PARENT` anywhere in the tree is counted; there may be at most one, it must be a top-level statement, and its value a `str` constant.

- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_plugins.py` — FAIL, `ModuleNotFoundError: perp_lab.plugins`.

- [ ] **Step 3: Implement `perp_lab/plugins.py`.** `blob_hash` = `sha1(b"blob %d\0" % len(data) + data).hexdigest()`.

- [ ] **Step 4: Switch the callers.**
  - `strategy.plugin_hash(name)` returns `blob_hash(plugin_path(name).read_bytes())` (no subprocess). Delete `plugin_parent`.
  - `lab strategies` prints from `PluginSource.read(n)` only (hash, `meta.timeframes`, `meta.hypothesis`); no plugin is loaded.
  - `validate_strategy` takes the hypothesis from `PluginSource.read(name).meta.hypothesis`.
  - `judge`: the pre-checks and `resolve_family` use `PluginSource.read(...).meta.parent`; a `PluginMetaError` becomes `JudgeRefused(f"{name}: {e}")`.
  - `test_plugin_has_one_sentence_hypothesis` reads `PluginSource.read(name).meta.hypothesis`.

- [ ] **Step 5: Run** `.venv/bin/pytest -q` — all pass.

- [ ] **Step 6: Commit** — `feat(plugins): read plugin metadata and hash from the source, never by running it`

---

### Task 2: Decision logs and the trusted replay

Pure code, no processes yet.

**Files:**
- Create: `perp_lab/sandbox/__init__.py` (empty), `perp_lab/sandbox/errors.py`, `perp_lab/sandbox/decisions.py`, `perp_lab/sandbox/replay.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Produces, `errors.py`: `SandboxUnavailable(RuntimeError)`; `PluginError(Exception)` with `__init__(self, kind: str, type_: str = "", message: str = "", stderr: bytes = b"")`, attributes `kind`, `type`, `message`, `stderr`, and `reason() -> str` = `"plugin_error: <kind>"`, plus `": <type>: <message>"` when `kind == "exception"`; `KINDS = ("exception", "cpu_limit", "protocol", "mismatch", "metadata", "features", "died", "timeout")`; `ProtocolError(PluginError)` (kind fixed to `"protocol"`); `WorkerLost(Exception)` with `kind` in `("died", "timeout")`.
- Produces, `decisions.py`: `ACTIONS = ("hold", "long", "short", "exit")` (codes 0–3); `Decisions` (frozen dataclass of numpy arrays `i` `<i8`, `action` `<i1`, `stop` `<f8`, `target` `<f8`, `has_stop` `<i1`, `has_target` `<i1`) with `from_log(log: list[tuple[int, Signal]])`, `empty()`, `by_bar() -> dict[int, Signal]`, `before(n: int) -> Decisions`, `same(other) -> bool` (exact, NaN equal to NaN), `n_actions() -> int` (entries whose action isn't hold), `validate(n_bars: int) -> None` (raises `ProtocolError`: dtypes, equal lengths, `i` strictly increasing within `[0, n_bars)`, codes in 0–3, masks 0/1); `Recorder(Strategy)` wrapping a strategy, whose `on_bar` calls the inner `on_bar`, raises `TypeError` unless the result is a `Signal` with an action in `ACTIONS` and `stop`/`target` each `None` or a real number, appends `(ctx.i, sig)` to `.log` unless it is a bare hold (`hold` with `stop` and `target` both `None`), and returns it; `decide_last(strategy, symbol: str, bars: pd.DataFrame, position: PositionView | None) -> Signal` (today's `PaperBook.on_closed_bar` decision: `prepare`, `build_arrays`, `ctx.i = len(bars) - 1`, `ctx.position`, `on_bar`, then the same `Signal` check).
- Produces, `replay.py`: `random_draw(n: int, seed: int, prob: float) -> tuple[np.ndarray, np.ndarray]` (`fire` bool, `side` ±1, the same draw as `RandomEntry.prepare`); `ScriptedStrategy(Strategy)` with `__init__(self, dec: Decisions, draw=None)`; `replay(bars, dec, *, symbol, tf, ep, start_i=0, end_i=None, draw=None) -> BacktestResult` (unchanged `run_backtest` with `ScriptedStrategy` and `bars[[]]` as features); `check_fingerprint(res: BacktestResult, fp: tuple[int, float], ep: EngineParams) -> None` (raises `PluginError("mismatch")` unless `len(res.trades) == fp[0]` and the final equity, `res.equity[-1]` or `ep.equity0` when empty, `== fp[1]` exactly).

`ScriptedStrategy.on_bar` returns the logged signal at `ctx.i` (`stop`/`target` set to `None` where the mask is 0), else `HOLD`. With a `draw`, a logged `long`/`short` while `ctx.position is None` must have `fire[ctx.i]` and the drawn side, otherwise it raises `PluginError("mismatch", "", "random entry off the draw at bar <i>")`. While in a position anything is accepted: a reversal from `manage()`.

- [ ] **Step 1: Write the failing tests** in `tests/test_replay.py` (these run real plugins in-process, as tests may):

```python
@pytest.mark.parametrize("name", REAL_PLUGINS)
def test_replaying_the_log_reproduces_the_run(name, monkeypatch):
    bars = with_moving_columns(make_bars(seed=1)); s = Recorder(load_class(name)())
    ref = run_backtest(s, bars, s.inner.prepare(bars), symbol="X", tf="1h", ep=EngineParams())
    res = replay(bars, Decisions.from_log(s.log), symbol="X", tf="1h", ep=EngineParams())
    assert res.trades == ref.trades and np.array_equal(res.equity, ref.equity)
    check_fingerprint(res, (len(ref.trades), float(ref.equity[-1])), EngineParams())

def test_nan_and_none_targets_replay_identically(): ...
    # a fixture strategy alternating Signal("hold", target=nan) and Signal("hold", target=None) while in a position

def test_a_forged_fingerprint_is_a_mismatch(): ...       # trades+1, then equity+1e-9: both raise PluginError kind "mismatch"

def test_a_feature_named_close_is_a_mismatch(): ...
    # a strategy whose prepare() returns a "close" column scaled by 1.01: the worker-side run (build_arrays
    # with those features) and the replay disagree, and check_fingerprint raises "mismatch"

def test_random_draw_matches_random_entry():
    f = RandomEntry(load_class("donchian_atr")(), 0.02, seed=7).prepare(make_bars(seed=2))
    fire, side = random_draw(len(f), 7, 0.02)
    assert np.array_equal(np.where(fire, side, 0), f["_rand"].to_numpy())

def test_random_entry_log_off_the_draw_is_a_mismatch(): ...   # insert a long while flat on a non-firing bar
def test_a_reversal_from_manage_passes_random_entry_replay(): ... # fixture whose manage() returns a short while long
def test_validate_rejects_bad_logs(): ...   # unsorted i, i >= n_bars, code 4, mask 2, wrong dtype, unequal lengths
def test_bare_holds_are_not_logged(): ...
def test_a_non_signal_return_is_a_type_error(): ...   # on_bar returning ("long", 1.0)
```

`load_class(name)` is a small helper in `tests/sandbox_fixtures.py` for now (`load_strategy_class` with `STRATEGY_DIR` pointed at the repo); Task 12 re-points it.

- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_replay.py` — FAIL (missing modules).
- [ ] **Step 3: Implement** `errors.py`, `decisions.py`, `replay.py`.
- [ ] **Step 4: Run** `.venv/bin/pytest -q` — all pass.
- [ ] **Step 5: Commit** — `feat(sandbox): decision logs and the trusted replay`

---

### Task 3: The wire protocol

**Files:**
- Create: `perp_lab/sandbox/protocol.py`
- Test: `tests/test_protocol.py`

**Interfaces:**
- Produces: `MAX_HEADER = 65536`; `WORKER_DTYPES = {"<i8", "<i1", "<f8"}`; `write_frame(f, header: dict, sections: dict[str, np.ndarray | bytes] | None = None) -> None`; `read_frame(f, *, max_bytes: int, from_worker: bool) -> tuple[dict, dict[str, np.ndarray | bytes]]`; `table_bytes(df: pd.DataFrame) -> bytes` and `table_frame(b: bytes) -> pd.DataFrame` (Arrow IPC, index preserved; worker side only); `decisions_sections(dec) -> dict` and `decisions_from(sections, n_bars) -> Decisions` (calls `Decisions.validate`).

Frame layout: `>I` header length H (1 ≤ H ≤ `MAX_HEADER`), H bytes of UTF-8 JSON object (NaN allowed), then for each entry of `header["sections"]` (a list of `{"name", "dtype", "n"}` in order) exactly `n × itemsize` bytes. Allowed dtypes: `WORKER_DTYPES` from a worker; from the parent also `"bytes"` (n = byte count, returned as `bytes`) and `"arrow"` (n = byte count). `read_frame` raises `ProtocolError` on: short read or EOF mid-frame, header too long or not a JSON object, a malformed or duplicate section entry, a dtype not allowed for the direction, a negative or non-int `n`, or a declared total over `max_bytes`; it checks every size before reading the bytes. EOF before the first byte raises `EOFError` (the client turns it into `WorkerLost("died")`). Arrays come back via `np.frombuffer(..., dtype=...)`.

- [ ] **Step 1: Write the failing tests:** a round trip of every worker dtype plus NaN values in the header; `"arrow"` and `"bytes"` are refused `from_worker=True`; a frame declaring 65 MiB is refused at `max_bytes = 64 MiB` before any section byte is read (the test's reader counts the bytes consumed); a truncated header, truncated section and a non-object header are each `ProtocolError`; empty input is `EOFError`; `table_frame(table_bytes(df))` equals `df` for `make_bars()` with its index and the `regime` column.
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_protocol.py` — FAIL.
- [ ] **Step 3: Implement** `protocol.py`.
- [ ] **Step 4: Run** `.venv/bin/pytest -q` — all pass.
- [ ] **Step 5: Commit** — `feat(sandbox): framed pipe protocol that never unpickles worker output`

---

### Task 4: Confinement

**Files:**
- Create: `perp_lab/sandbox/confine.py` (standard library only: `ctypes`, `os`, `resource`, `site`, `sys`)
- Modify: `pyproject.toml` (register the `sandbox` marker under `[tool.pytest.ini_options]`)
- Test: `tests/test_confine.py` (all `@pytest.mark.sandbox`), `tests/confine_probe.py` (a script the tests run in a subprocess)

**Interfaces:**
- Produces: `probe(min_abi: int = 6) -> str | None` (`None` when this machine can confine; otherwise the reason, e.g. `"not Linux"`, `"Landlock ABI 5 < 6"`, `"libseccomp.so.2 not found"`; it spawns nothing); `allowlist() -> list[str]` (`sys.base_prefix`, `sys.prefix`, each `site.getsitepackages()` entry, the `perp_lab` package directory, plus any extra read-only system path a later step proves necessary, each with a comment giving its reason); `confine(*, memory_mb: int, cpu_s: int, read_paths: list[str]) -> int` (returns the Landlock ABI; raises `RuntimeError` if any step fails).

`confine` does exactly the spec's Confinement steps 1–5 in order:

1. Exactly one entry in `/proc/self/task`.
2. Hard rlimits: `RLIMIT_AS` = `memory_mb` MiB, `RLIMIT_FSIZE` 0, `RLIMIT_CORE` 0, `RLIMIT_NOFILE` 64, `RLIMIT_CPU` = `cpu_s`.
3. `prctl(PR_SET_NO_NEW_PRIVS=38, 1, 0, 0, 0)`.
4. Landlock: query the ABI (`landlock_create_ruleset(NULL, 0, LANDLOCK_CREATE_RULESET_VERSION=1)`). Handle every fs right of that ABI (bits 0–12, plus REFER 1<<13 from ABI 2, TRUNCATE 1<<14 from ABI 3, IOCTL_DEV 1<<15 from ABI 5), net `BIND_TCP|CONNECT_TCP` (3), scoped `ABSTRACT_UNIX_SOCKET|SIGNAL` (3). Add one `LANDLOCK_RULE_PATH_BENEATH` (1) rule granting `READ_FILE|READ_DIR` (1<<2 | 1<<3) per read path (the `landlock_path_beneath_attr` struct is packed: `u64 allowed_access; s32 parent_fd`; open the path with `O_PATH`). Restrict self with `LANDLOCK_RESTRICT_SELF_TSYNC` (1<<3) when ABI ≥ 8, else 0.
5. seccomp: `seccomp_init(SCMP_ACT_ALLOW=0x7fff0000)`; `seccomp_attr_set(ctx, SCMP_FLTATR_CTL_TSYNC=4, 1)`; bad arch keeps the default kill. Rules, all through `seccomp_rule_add_array` with names resolved by `seccomp_syscall_resolve_name` (skip a name that resolves to a negative number on this arch). `ERRNO(e)` = `0x00050000 | e`. `scmp_arg_cmp` is `{u32 arg; u32 op; u64 datum_a; u64 datum_b}`, with ops `NE=1`, `MASKED_EQ=7`.

| Action | Syscalls | Condition |
|---|---|---|
| `ERRNO(EPERM)` | `socket socketpair execve execveat fork vfork shmget shmat shmctl shmdt msgget msgsnd msgrcv msgctl semget semop semtimedop semctl ptrace process_vm_readv process_vm_writev process_madvise kcmp pidfd_open pidfd_getfd migrate_pages move_pages mount umount2 pivot_root chroot unshare setns bpf perf_event_open userfaultfd io_uring_setup io_uring_enter io_uring_register keyctl add_key request_key init_module finit_module delete_module kexec_load kexec_file_load` | always |
| `ERRNO(EPERM)` | `clone` | `arg0 MASKED_EQ (mask CLONE_THREAD=0x10000, value 0)` |
| `ERRNO(ENOSYS)` | `clone3` | always |
| `ERRNO(EPERM)` | `prlimit64 sched_setaffinity sched_getaffinity sched_setscheduler sched_getscheduler sched_setparam sched_getparam sched_rr_get_interval sched_setattr sched_getattr` | `arg0 NE 0` |
| `ERRNO(EPERM)` | `getpriority setpriority ioprio_get ioprio_set` | `arg1 NE 0` |

- [ ] **Step 1: Write the failing tests.** `tests/confine_probe.py` takes `--allow` paths and one probe name, calls `confine(memory_mb=1024, cpu_s=60, read_paths=...)`, attempts the probe and prints `DENIED <errno name>` or `ALLOWED`. The tests run it with `[sys.executable, "-s", "-P", "tests/confine_probe.py", ...]` and the spec's environment, and assert:

```python
@pytest.mark.parametrize("probe", ["read_repo_file", "write_tmp", "tcp_connect", "udp_send", "unix_connect",
                                   "fork", "subprocess", "shmget", "prlimit_parent", "kill_parent",
                                   "sched_getaffinity_parent", "read_pre_existing_thread"])
def test_each_escape_is_denied(probe, tmp_path): ...
```
  - `write_tmp`: the sentinel file doesn't exist afterwards.
  - `tcp_connect`: a listener the test runs on 127.0.0.1 sees no connection.
  - `unix_connect`: aims at a UNIX socket the test creates.
  - `prlimit_parent`: the test's own `RLIMIT_NOFILE` is unchanged.
  - `kill_parent`: the test process survives.
  - `read_pre_existing_thread`: the probe starts a thread *before* calling `confine`; `confine` must then raise, and the script prints `REFUSED`.

  Also: `test_reading_inside_the_allowlist_still_works` (import `json` and read a site-packages file); `test_numpy_pandas_pyarrow_import_after_confinement`, which then counts `/proc/self/task` and checks that every listed thread is denied a read of `README.md` (open the file from inside each thread); `test_probe_reports_none_here` (this machine); `test_probe_reports_a_low_abi` (monkeypatch the ABI query to return 5).
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_confine.py` — FAIL.
- [ ] **Step 3: Implement `confine.py`.** Add allowlist paths only when a test proves numpy, pandas or pyarrow need them.
- [ ] **Step 4: Run** `.venv/bin/pytest -q` — all pass, none of `test_confine.py` skipped on this machine.
- [ ] **Step 5: Commit** — `feat(sandbox): Landlock and seccomp confinement`

---

### Task 5: The worker and the client (batch runs)

**Files:**
- Create: `perp_lab/sandbox/worker.py` (standard library plus `perp_lab.sandbox.confine` at module level), `perp_lab/sandbox/session.py`, `perp_lab/sandbox/client.py`
- Create: `tests/sandbox_fixtures.py` (`InProcessWorker`, `load_class`), modify `tests/conftest.py` (autouse fixture)
- Modify: `config/lab.yaml` (the `sandbox:` section, verbatim from Global Constraints, with the spec's comments)
- Test: `tests/test_worker.py`

**Interfaces:**
- `worker.main(argv)` order: (a) move the protocol to private fds: `proto_in = os.dup(0)`, `proto_out = os.dup(1)`, then `dup2(2, 1)` and `dup2(2, 0)`; (b) read the limits from `--memory-mb` (`sandbox.memory_mb`) and `--cpu-s` argv, which the client passes; `--cpu-s` is the lifetime backstop `ceil(exam_cpu_hours × 3600 × 2)`, since per-request limits are the parent's; (c) `confine(...)` with `allowlist()`; (d) import `session`, which imports numpy, pandas, pyarrow and the engine; (e) self-test: opening `--deny-probe`, `socket.socket()`, `os.fork()` and `shmget` must each fail; (f) send `{"ok": true, "ready": true, "abi": n}`; (g) serve requests until EOF on `proto_in`, then exit 0. A failure in (a)–(e) sends `{"error": {"kind": "unavailable", "type": ..., "message": ...}}` and exits 3.
- `session.load_plugin(name: str, text: bytes) -> type[Strategy]` (compile and exec into a fresh `types.ModuleType(f"lab_strategies.{name}")` registered in `sys.modules` under that name; exactly one `Strategy` subclass defined in it; `cls.name = name`). `session.Session` has one method per op: `load(name, text) -> dict` (runtime `params`, `grid`, `timeframes`), `bars(symbol, tf, df) -> int`, `run(params, ep: dict, k, start_i, random, features) -> (Decisions, (int, float), list[str], list[np.ndarray])`. `run` builds the strategy (`RandomEntry(cls(**params), prob, seed)` when `random`), takes `bars.iloc[:k]` when `k` is set, wraps the strategy in `Recorder`, calls `prepare` and `run_backtest(..., start_i=start_i)`, and casts the features to float64 (bool and int are cast; any other dtype raises `PluginError("features")`). `session.error_reply(exc) -> dict` maps `PluginError` to its kind and any other exception to `("exception", type(e).__name__, str(e))`, the message cut to 200 printable ASCII characters.
- `client.py` produces:
  - `check(cfg) -> int` (spawn with no plugin, wait for ready, close; `SandboxUnavailable` with the worker's reason).
  - `spawn(cfg, src: PluginSource) -> Worker` (raises `SandboxUnavailable` before the plugin loads; `PluginError("metadata")` if the runtime `params`, `grid` or `timeframes` differ from `src.meta`).
  - `max_workers(cfg) -> int` = `max(1, floor(0.75 × physical MB / memory_mb))`.
  - `RunOutput(decisions: Decisions, fingerprint: tuple[int, float], features: pd.DataFrame | None)`.
  - `Worker` with `pid`, `last_cpu_s: float`, `stderr: bytes` (first 64 KiB), `bars(symbol, tf, df) -> None`, `run(params, ep: EngineParams, *, k=None, start_i=0, random=None, features=False) -> RunOutput` (features indexed by the bars' index), `close()`, context manager.
  - `Batch(cfg, src, symbol, tf, bars, stderr_dir: Path | None = None)`, a context manager, with `backtest(params, ep, *, random=None, start_i=0) -> BacktestResult`: run, then `replay` (with `random_draw` when `random`), then `check_fingerprint`. On closing each of its workers (retried ones included) it writes that worker's stderr, if any, to `stderr_dir`.
  - `fresh_run(cfg, src, symbol, tf, bars, params, ep, *, features=False) -> RunOutput` (one new worker that is sent only `bars`).

  `Batch` and `fresh_run` retry a `WorkerLost` once in a fresh worker (re-sending the bars) and turn a second one into `PluginError(kind)`.

How `Worker` enforces limits (an algorithm the tests don't fix): a daemon thread per worker blocks on `read_frame` and puts each frame (or the exception) on a queue, and another drains stderr into the capped buffer. A request records the worker's CPU time, from `/proc/<pid>/stat` fields 14+15 over `SC_CLK_TCK`, then waits on the queue in 0.25 s slices. CPU used above the op's limit means kill and `PluginError("cpu_limit")`; wall time above the limit means kill and `WorkerLost("timeout")`; EOF or a dead process means `WorkerLost("died")`. Limits: every batch op gets `task_cpu_s` CPU and `task_cpu_s × wall_factor` wall. `close()` closes stdin, waits 2 s, then kills.

`InProcessWorker(cfg, src)` in `tests/sandbox_fixtures.py` has the same public interface as `Worker` (with `pid = os.getpid()`, `last_cpu_s` from `time.process_time()` deltas, and `stderr` = what the plugin printed, captured with `contextlib.redirect_stdout`/`redirect_stderr` around each `Session` call), and gains each op later tasks add to `Worker`. It runs a `Session` in-process (a fresh module per instance), applies the same metadata check, converts exceptions through `error_reply`, and builds the same `RunOutput`. The autouse fixture in `tests/conftest.py`: for tests without the `sandbox` marker, monkeypatch `perp_lab.sandbox.client.spawn` to `InProcessWorker` and `client.check` to `lambda cfg: 8`. For marked tests, skip with `sandbox unavailable: <reason>` when `confine.probe()` returns one.

- [ ] **Step 1: Write the failing tests** in `tests/test_worker.py` (`@pytest.mark.sandbox` unless noted):

```python
@pytest.mark.parametrize("name", REAL_PLUGINS)
def test_sandboxed_batch_equals_in_process(cfg, name): ...
    # Batch.backtest on with_moving_columns(make_bars(seed=1)) == run_backtest in-process (trades, equity)
def test_in_process_twin_gives_the_same_output(cfg): ...   # unmarked twin vs marked real worker: same RunOutput
def test_a_plugin_exception_comes_back_with_its_type(cfg): ...      # KeyError → kind "exception", type "KeyError"
def test_a_busy_loop_is_cpu_limit(cfg): ...                 # sandbox.task_cpu_s=2 in a market_config copy
def test_os_exit_is_retried_once_then_died(cfg): ...        # spawn counter == 2, PluginError kind "died"
def test_sleeping_past_the_wall_limit_is_a_timeout(cfg): ... # task_cpu_s=1, wall_factor=2, plugin sleeps 5 s
def test_runtime_metadata_must_match_the_source(cfg): ...   # plugin does `Cls.params["n"] = 9` at import → "metadata"
def test_string_hashes_agree_across_workers(cfg): ...       # plugin stores hash("abc") as a feature in two workers
def test_no_environment_reaches_the_worker(cfg, monkeypatch): ...  # PGPASSWORD=sentinel in parent; plugin raises with os.environ keys
def test_data_and_runs_are_unreadable(cfg): ...             # plugin opens cfg.path and runs/lab.db → PermissionError
def test_a_forged_token_cannot_load_holdout_bars(cfg): ...  # HoldoutToken(..., _MINT_KEY) + load_bars → error naming data/
def test_one_worker_cannot_see_another_through_sysv(cfg): ...
def test_worker_exits_when_its_parent_goes_away(cfg): ...   # close the pipes without close(); pid gone within 5 s
def test_check_raises_when_confinement_fails(cfg, monkeypatch): ...  # unmarked: probe fails → SandboxUnavailable
def test_max_workers_caps_by_memory(cfg, monkeypatch): ...  # unmarked: 30 GiB physical, memory_mb 3072 → 7
def test_oversized_output_is_a_protocol_error(cfg): ...     # max_output_mb=1, features of 2 MB
```
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_worker.py` — FAIL.
- [ ] **Step 3: Implement** `worker.py`, `session.py`, `client.py`, the fixtures and the config section.
- [ ] **Step 4: Run** `.venv/bin/pytest -q` — all pass; none of `tests/test_worker.py` skipped here.
- [ ] **Step 5: Commit** — `feat(sandbox): confined worker and client for batch runs`

---

### Task 6: Walk-forward, sensitivity and the benchmark through the sandbox

**Files:**
- Modify: `perp_lab/validate/walkforward.py`, `perp_lab/validate/sensitivity.py`, `perp_lab/validate/benchmarks.py`, `perp_lab/validate/pipeline.py`
- Test: `tests/test_walkforward_sandbox.py`; update `tests/judge_fixtures.py` (`StubValidate` accepts `src=None` and records `src.hash` when given)

**Interfaces:**
- Consumes: `client.Batch`, `client.max_workers`, `PluginSource`.
- Produces: `walkforward(cfg, src: PluginSource, tf, symbols=None, workers=None, stderr_dir=None) -> WFResult` (`WFResult` gains `src: PluginSource` and `stderr_dir: Path | None`, both `field(repr=False)`, which `sensitivity` and `random_entry` reuse); `run_many(cfg, src, jobs, workers=None, stderr_dir=None) -> list[dict]`, where a job is `(params, symbol, tf, overrides, seed, entry_prob)` and each result dict is as today; `chunks(jobs, workers) -> list[tuple[str, str, list]]`; `run_chunk(args) -> list[dict]` (pool task: one `Batch` for one symbol); `validate_strategy(cfg, name, tf, symbols=None, seeds=None, workers=None, notes="", parent_run_id=None, lineage_id=None, src: PluginSource | None = None)`, which reads `PluginSource.read(name)` when `src` is None and records `src.hash`.

`chunks` groups jobs by symbol in order and splits each group into pieces of `per = max(1, ceil(len(jobs) / max(2 × workers, n_symbols)))`. A chunk never mixes symbols. Effective pool size is `min(workers or os.cpu_count(), client.max_workers(cfg))`; `workers == 1` runs the chunks in-process, as today. `validate_strategy` takes `rid = new_run_id()` first and passes `stderr_dir = cfg.runs_dir / rid / "sandbox_stderr"`. It drops the end-of-run `plugin_hash` comparison: the bytes are read once.

- [ ] **Step 1: Write the failing tests:**

```python
def test_chunks_never_mix_symbols_and_spread_work(): ...    # 3 symbols × 10 jobs, workers 4 → ≥ 8 chunks, one symbol each
def test_run_many_matches_a_direct_backtest(tmp_path): ...
    # market_config; donchian_atr; for every (combo, symbol) job, the result's daily returns and trades equal an
    # in-process run_backtest on the same research bars and params
def test_plugin_output_lands_in_the_run_directory(tmp_path, plugins): ...
    # a plugin that prints "hello" in prepare(); validate_strategy; <run_dir>/sandbox_stderr/*.txt contains "hello"
    # (the twin captures print output into its `stderr` for this)
@pytest.mark.sandbox
def test_a_worker_never_sees_a_second_symbol(tmp_path): ...
    # plugin keeps a module-global list of the first close it saw per prepare() call; it raises if two differ
def test_an_edit_during_validation_does_not_change_what_is_recorded(tmp_path, plugins): ...
    # edit the file from inside run_many (monkeypatch); the recorded strategy_hash is the original
```
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_walkforward_sandbox.py` — FAIL.
- [ ] **Step 3: Implement.** Remove `load_strategy_class` from these four modules.
- [ ] **Step 4: Run** `.venv/bin/pytest -q` — all pass (`test_a_plugin_edited_mid_walk_forward_is_not_recorded` in `tests/test_judge_exam.py` is replaced by the edit test above; delete it).
- [ ] **Step 5: Commit** — `feat(validate): walk-forward, sensitivity and benchmark run plugins in the sandbox`

---

### Task 7: The lookahead check in fresh workers

**Files:**
- Modify: `perp_lab/validate/causality.py`, `perp_lab/judge/judge.py` (`lookahead_failures`, `dated_lookahead_failures`, `attempt_exam`'s call)
- Modify: `tests/test_causality.py`, `tests/test_lookahead.py`, `tests/test_judge.py` (fixture plugins become source text; `load_strategy_class(name)` becomes `PluginSource.read(name)`)
- Test: additions to `tests/test_causality.py`

**Interfaces:**
- Consumes: `client.fresh_run`, `client.max_workers`.
- Produces: `causality_failures(cfg, src: PluginSource, bars=None, *, cuts=None, require_signals=True, params=None) -> list[str]`; `compare(full: RunOutput, cut: RunOutput, k: int) -> None`, which raises `AssertionError("features at cut k ...")` (`assert_frame_equal` on the first k rows, `rtol=1e-9`, `atol=1e-12`, `check_dtype=False`) or `AssertionError("signals diverge before cut k")` (`full.decisions.before(k-1).same(cut.decisions.before(k-1))` is false); `lookahead_failures(cfg, src)`; `dated_lookahead_failures(cfg, src, until, params=None)`. `assert_causal` is deleted.

Runs go in waves: for each k in sorted cuts, then `len(bars)` (the full run), every still-passing variant runs at once on a `ThreadPoolExecutor(client.max_workers(cfg))`, each as `fresh_run(cfg, src, "X", "1h", bars.iloc[:k], params, EngineParams(), features=True)`. No longer run starts before every shorter one has finished. A `PluginError` removes the variant from later waves and is reported as `f"{label}: {e.type}: {e.message}"` for `kind == "exception"`, else `f"{label}: plugin_error: {e.kind}"`. Variants, labels and the no-signals rule (`full.decisions.n_actions() == 0` on defaults) are unchanged.

- [ ] **Step 1: Write the failing tests:**

```python
def test_waves_never_overlap_a_longer_run(cfg, monkeypatch): ...
    # spy on client.fresh_run recording (start, end, len(bars)); every run of length L ends before any longer one starts
@pytest.mark.sandbox
def test_a_memoising_cheater_is_caught(cfg): ...
    # module-global cache of the longest prepare() output (computed with shift(-1)), returned truncated on shorter inputs
def test_a_cut_run_that_dies_is_reported_for_its_variant(cfg): ...   # "defaults: plugin_error: died"
```
  Port the existing tests' intent: `Cheater`, `Broken`, `peeker(col)` and `Silent` become source strings built through `PluginSource.from_text`, and each assertion stays as it is (`"defaults: KeyError: 'no such column'"`, `startswith("defaults: AssertionError")`, the `NO_SIGNALS` line).
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_causality.py tests/test_lookahead.py tests/test_judge.py` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `.venv/bin/pytest -q` — all pass.
- [ ] **Step 5: Commit** — `feat(judge): the lookahead check runs every compared run in its own fresh worker`

---

### Task 8: The judge on the sandbox

**Files:**
- Modify: `perp_lab/judge/judge.py`, `perp_lab/cli.py` (`judge` command)
- Modify: `tests/test_judge.py` (delete `test_a_parent_edited_while_its_walk_forward_runs_gets_no_judgement` and `test_a_parent_edited_during_its_lookahead_check_gets_no_judgement`)
- Test: additions to `tests/test_judge.py`

**Interfaces:**
- Consumes: `client.check`, `PluginSource`, `PluginError`, `SandboxUnavailable`.
- Produces: `judge(cfg, name, tf, symbols=None, workers=None, *, validate=validate_strategy, attempt=None)` (`attempt` goes in Task 10). Flow changes:
  1. `client.check(cfg)` comes first; `SandboxUnavailable` propagates.
  2. One `PluginSource` per plugin is read up front and passed to every step (`validate(..., src=src)`, `lookahead_failures(cfg, src)`).
  3. A `PluginError` from the child's `validate` records a `kill` judgement with `reasons=[e.reason()]`, `child_run_id=None`, `holdout_status`/`holdout_reason` `"n/a"`, and no edge.
  4. A `PluginError` from the parent's `validate` raises `JudgeRefused(f"parent {parent} fails in the sandbox; fix or judge it first")`.
  5. Both `... changed while ...` `RuntimeError`s are gone.
  6. When a plugin error is recorded, the judge logs `e.stderr` at WARNING on its logger. That goes to the judge's own stderr, which the proposer never sees.

  `lab judge` catches `SandboxUnavailable` and prints `sandbox unavailable: <reason>` in red, exit 1.

- [ ] **Step 1: Write the failing tests:**

```python
def test_a_plugin_error_in_the_walk_forward_is_a_recorded_kill(cfg, plugins): ...
    # StubValidate raising PluginError("exception", "ValueError", "boom") for the child
    # → decision kill, reasons ["plugin_error: exception: ValueError: boom"], child_run_id None
def test_a_parent_that_fails_in_the_sandbox_refuses_the_child(cfg, plugins): ...
def test_judge_refuses_to_start_without_a_sandbox(cfg, plugins, monkeypatch): ...
    # client.check raises SandboxUnavailable → raised, nothing recorded; CLI exit code 1 and the message
def test_an_edit_during_judging_does_not_change_what_is_judged(cfg, plugins): ...
    # edit the parent inside StubValidate; the judgement and both runs carry the hashes read at the start
@pytest.mark.sandbox
def test_end_to_end_judge_through_the_real_sandbox(tmp_path, plugins, fresh_wf_caches): ...
    # the assertions of test_end_to_end_real_walk_forward_on_a_synthetic_market (tests/test_judge_exam.py),
    # with workers=2 and no exam: a kill has reasons, a promotion is pending / awaiting_exam
```
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_judge.py` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `.venv/bin/pytest -q` — all pass.
- [ ] **Step 5: Commit** — `feat(judge): judge through the sandbox; plugin failures are recorded kills`

---

### Task 9: The streamed exam

**Files:**
- Create: `perp_lab/sandbox/stream.py`
- Modify: `perp_lab/sandbox/session.py` (`stream_open`, `step`), `perp_lab/sandbox/client.py` (`Worker.stream_open`, `Worker.step`; `InProcessWorker` likewise), `perp_lab/judge/exam.py`, `perp_lab/judge/judge.py` (the `run(...)` call passes `src`)
- Modify: `tests/test_exam.py`, `tests/test_judge_exam.py` (`stub_run` takes `src`)
- Test: `tests/test_stream.py`

**Interfaces:**
- Produces, worker ops:
  - `stream_open {params, ep, warmup_k}` returns the decisions of a batch run over the first `warmup_k` held bars (none when `warmup_k == 0`) and keeps the strategy instance.
  - `step {position, mode: "none" | "append" | "window"}` (plus a `table` section for append and window) returns `{"signal": {action, stop, target, has_stop, has_target}}`, computed by `decide_last` on everything held. `append` adds rows; `window` replaces the held bars.
- Produces, client: `Worker.stream_open(params, ep, warmup_k) -> Decisions`; `Worker.step(position: PositionView | None, *, append: pd.DataFrame | None = None, window: pd.DataFrame | None = None) -> Signal`. A step's limit is `stream_step_s` for both CPU and wall clock.
- Produces, `stream.py`: `Lockstep(symbols)` with `turn(symbol, t: int) -> None`, `finish(symbol)`, `abort()` (afterwards `turn` raises `Aborted`); `StreamedStrategy(Strategy)` with `__init__(self, worker, warm: Decisions, k0: int, bars: pd.DataFrame, symbol: str, lockstep: Lockstep)`.
- Produces, `exam.py`: `run_exam(cfg, src: PluginSource, tf, symbols, params, token, end, stderr_dir: Path | None = None) -> ExamResult` (same result as today). Each worker's stderr is written to `stderr_dir` when it closes.

`Lockstep.turn(s, t)`: under a `threading.Condition`, set `at[s] = t`, `notify_all`, then wait until every other symbol still in `at` has a value that isn't `None` and is `≥ t`. `finish(s)` deletes `at[s]` and notifies. Every symbol starts as `None`, so no stream step happens until every warm-up is done.

`StreamedStrategy.on_bar(ctx)`:
- For `i < k0 - 1`, return the warm-up signal at `i` (or `HOLD`).
- Otherwise call `lockstep.turn(symbol, open_time[i])`, then `worker.step(ctx.position, append=bars.iloc[[i]] if i >= k0 else None)`.

`run_exam`:
- Symbols go in batches of at most `client.max_workers(cfg)`, one batch after another.
- Within a batch, one thread per symbol: load the bars (token), `k0` = first index with `open_time ≥ holdout_start`, spawn, send `bars.iloc[:k0]`, `stream_open(params, ep, k0)`, then `run_backtest(StreamedStrategy(...), bars, bars[[]], ...)`, then slice as today.
- A thread's `PluginError` calls `lockstep.abort()` and is re-raised after the threads join.
- A `WorkerLost` restarts the whole batch once in fresh workers; a second loss raises `PluginError(kind)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_stream.py`:

```python
@pytest.mark.parametrize("name", REAL_PLUGINS)
def test_streamed_exam_equals_the_batch_exam(tmp_path, name): ...
    # market_config; token from issue_holdout_token; run_exam vs the 3a batch computation
    # (load_bars + run_backtest in-process): identical daily returns and trades
def test_an_oracle_plugin_gains_nothing_streamed(tmp_path): ...
    # long iff next close > close: batch Sharpe > 5 on the synthetic market; streamed |Sharpe| < 1
def test_lockstep_orders_steps_across_symbols(): ...
    # 3 threads, a fake worker logging (symbol, t, phase); no step for t' > t starts before all steps at t end
def test_lockstep_with_a_symbol_missing_bars(): ...   # one symbol lacks 5 timestamps: no deadlock, ordering holds
def test_a_failing_symbol_aborts_the_others(tmp_path): ...
@pytest.mark.sandbox
def test_sleeping_in_a_stream_step_fails_the_exam(tmp_path): ...   # stream_step_s=1; plugin sleeps on holdout dates
@pytest.mark.sandbox
def test_streamed_exam_through_real_workers(tmp_path): ...         # donchian_atr, equality as above
```
  `tests/test_exam.py`'s window tests keep their assertions (92 days, the holdout window boundaries) with `PluginSource.read("donchian_atr")`.
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_stream.py tests/test_exam.py` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `.venv/bin/pytest -q` — all pass.
- [ ] **Step 5: Commit** — `feat(exam): stream the holdout one bar at a time, symbols in lockstep`

---

### Task 10: The exam flow: only `lab exam`, a fixed window, an attempt cap, the cost probe

**Files:**
- Modify: `perp_lab/holdout.py` (`issue_holdout_token(..., window_end: str | None = None)` writes `"window_end"` into the entry), `perp_lab/judge/budget.py`, `perp_lab/judge/judge.py`, `perp_lab/judge/exam.py`, `perp_lab/sandbox/session.py` and `client.py` (`prepare` op), `perp_lab/cli.py` (`judge` passes no `attempt`), `config/lab.yaml` (`judge.exam_max_attempts: 3` with comment `# attempts per holdout read, then failed / exam_error`)
- Modify: `tests/test_judge_exam.py` (`judged()` becomes `judge(...)` then `attempt_exam(...)`)
- Test: additions to `tests/test_judge_exam.py`, `tests/test_budget.py`

**Interfaces:**
- Produces, budget:
  - `mint(cfg, lineage_id, judgement_id, window_days, window_end: pd.Timestamp)` writes `window_end.isoformat()`.
  - `stored_window_end(cfg, judgement_id) -> pd.Timestamp | None` (from the first entry for that judgement that has one).
  - `attempts(cfg, judgement_id) -> int` (entries for that judgement).
- Produces, worker op `prepare {params}` runs `prepare` and `build_arrays` on the held bars; `Worker.prepare(params) -> float` returns the parent-measured CPU seconds (`InProcessWorker` likewise).
- Produces, exam: `cost_hours(cfg, src, tf, symbols, params, end) -> float` = median CPU seconds of 3 `prepare` requests on the first symbol's full research bars (in a worker of its own) × bars in `[holdout_start, end)` / 3600.
- Produces, judge: `judge(...)` loses `attempt`, and a promotion stays `pending` / `awaiting_exam`. `attempt_exam(cfg, jid, *, run=run_exam, end_fn=window_end, cost_fn=cost_hours)` follows the spec's order:
  1. `PluginSource.read` (a `PluginMetaError` or a hash ≠ the judged hash means `pending` / `plugin_changed`).
  2. If `budget.holds_read`: when `attempts ≥ exam_max_attempts`, the judgement is `failed` / `exam_error`; otherwise use `stored_window_end` (falling back to `end_fn` when `None`) and jump to step 6.
  3. `end_fn`.
  4. The dated lookahead check (fail means `blocked` / `lookahead`).
  5. `cost_fn > exam_cpu_hours` means `blocked` / `too_slow`.
  6. `client.check` (`SandboxUnavailable` means `pending` / `exam_error`, write the verdict, re-raise).
  7. `mint(..., window_end=end)`.
  8. `run(cfg, src, tf, syms, params, token, end, stderr_dir=cfg.runs_dir / "_exam_stderr" / f"judgement-{jid}")`. A `PluginError` means `failed` / `plugin_error` and a holdout run recorded by `_record_failed_holdout_run(cfg, j, s, end, err) -> str`: status `failed`, reasons `[err.reason()]`, empty metrics, and a `report.md` naming the window, the frozen params, the error and the stderr directory. Any other exception keeps today's `pending` / `exam_error` and re-raise.

- [ ] **Step 1: Write the failing tests:**

```python
def test_lab_judge_never_runs_an_exam(cfg, plugins, monkeypatch): ...
    # attempt_exam patched to fail if called; CLI `lab judge` of a promotable stub → pending awaiting_exam
def test_a_retry_uses_the_window_stored_at_the_mint(cfg, plugins): ...
    # first attempt crashes after mint with end_after(200); retry with end_after(260): run receives the day-200 end
def test_a_retry_skips_the_cost_probe_and_the_dated_check(cfg, plugins): ...   # cost_fn/lookahead patched to raise
def test_a_read_fails_after_its_last_attempt(cfg, plugins): ...                 # 3 crashes → failed / exam_error
def test_a_slow_plugin_is_blocked_before_any_read(cfg, plugins): ...            # cost_fn → 7.0 → blocked / too_slow, reads 0
def test_a_plugin_error_fails_the_exam_without_its_message_in_the_verdict(cfg, plugins): ...
    # run raises PluginError("exception", "ValueError", "close=123.45") → failed / plugin_error;
    # the verdict JSON contains neither "ValueError" nor "123.45"; the holdout run's report.md contains both
def test_no_mint_when_the_sandbox_is_unavailable(cfg, plugins, monkeypatch): ... # reads_used 0, pending / exam_error
@pytest.mark.sandbox
def test_end_to_end_exam_through_the_real_sandbox(tmp_path, plugins, fresh_wf_caches): ...
    # test_end_to_end_real_exam_on_a_synthetic_market's assertions (status passed or failed, window
    # 2026-07-01 → 2026-10-01, 92 holdout days), promoted by `judge` and then examined by `exam_pending`
```
  In `tests/test_budget.py`: `test_mint_stores_the_window_end` and `test_attempts_counts_entries_per_judgement`.
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_judge_exam.py tests/test_budget.py` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `.venv/bin/pytest -q` — all pass.
- [ ] **Step 5: Commit** — `feat(exam): only lab exam runs exams, on a window fixed at the mint`

---

### Task 11: `lab backtest`, paper and the dashboard's divergence

**Files:**
- Modify: `perp_lab/cli.py` (`backtest`, `paper`), `perp_lab/paper/runner.py`, `perp_lab/paper/tracking.py`, `dashboard/app.py`
- Modify: `tests/test_paper.py`, `tests/test_store.py::test_replay_divergence_on_identical_execution_is_zero`

**Interfaces:**
- Produces:
  - `PaperBook(decide: Callable[[pd.DataFrame, PositionView | None], Signal], symbol, tf, ep)`. `on_closed_bar` builds the position view as today, then calls `decide(bars, pv)`.
  - `PaperRunner(cfg, src: PluginSource, params, tf, symbols, out_dir, exchange=None)`. Per symbol: `client.spawn`, `bars(sym, tf, warm)`, `stream_open(params, ep, 0)`, and `decide = lambda bars, pv: worker.step(pv, window=bars)`. A worker failure propagates and stops the run. Workers are closed in a `finally`.
  - `replay_divergence(cfg, paper_dir, src: PluginSource, params, tf)` via `client.Batch(...).backtest(params, ep, start_i=live[0])`.
  - `lab backtest` uses one `Batch` per symbol with `{**src.meta.params, **p}`.
  - `lab paper` uses `PluginSource.read` and `meta.hypothesis`.
  - The dashboard passes `PluginSource.read(p.strategy)`.

- [ ] **Step 1: Write the failing tests.** In `tests/test_paper.py`, the `Scripted` book tests pass `decide=lambda bars, pv: decide_last(Scripted({...}), "X", bars, pv)`; the runner test builds an "always long, stop 90" plugin with `PluginSource.from_text` and keeps its assertions (filled at the live ask 100.1). Add `test_paper_stops_when_its_worker_fails` (the twin raises on the 2nd step; `run()` raises and `state.json` exists) and `test_cli_backtest_runs_through_the_sandbox` (spawn spy called once per symbol). Port the divergence test to `PluginSource.read("donchian_atr")`.
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_paper.py tests/test_store.py` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `.venv/bin/pytest -q` — all pass.
- [ ] **Step 5: Commit** — `feat(paper): backtest, paper and divergence run plugins in the sandbox`

---

### Task 12: One executor, and the documents

**Files:**
- Modify: `perp_lab/strategy.py` (delete `load_strategy_class` and the `importlib` import; the module docstring states the contract changes: metadata literal, features numeric or bool, a plugin gets only the bars it is sent, imports limited to the read allowlist)
- Modify: `tests/sandbox_fixtures.py` (`load_class(name)` = `session.load_plugin(name, PluginSource.read(name).text)` with `STRATEGY_DIR` at the repo), every test that imported `load_strategy_class`
- Modify: `README.md`, `docs/superpowers/specs/2026-10-16-judge-design.md`
- Test: `tests/test_single_executor.py`

**Interfaces:** none new.

- [ ] **Step 1: Write the failing test:**

```python
def test_only_the_session_executes_plugin_code():
    # AST scan of every .py under REPO/"perp_lab" and REPO/"dashboard" (anchored on tests/conftest.REPO; assert
    # the file count > 30 so it can't pass vacuously). Outside perp_lab/sandbox/session.py: no call to exec,
    # compile, eval, __import__, importlib.import_module, spec_from_file_location, exec_module,
    # load_strategy_class or load_plugin.
```
- [ ] **Step 2: Run** `.venv/bin/pytest -q tests/test_single_executor.py` — FAIL (`load_strategy_class` is still defined and imported).
- [ ] **Step 3: Remove the in-process loader** from `perp_lab/` and switch the remaining tests to `load_class`.
- [ ] **Step 4: Documents.**
  - README: add rows to "Rules the code enforces" for plugin confinement (`perp_lab/sandbox/confine.py`, `tests/test_confine.py`), scores from the trusted replay (`perp_lab/sandbox/replay.py`), the streamed exam (`perp_lab/sandbox/stream.py`), and "only `lab exam` runs exams"; drop "(plugin code is not sandboxed yet, so this assumes honest plugins)"; give `lab judge` / `lab exam` one line each on the new split; add a "Plugin sandbox (Phase 3b-1)" section of at most 10 lines pointing to the spec.
  - The 3a spec: after the Goal's caveat sentence and after the last carried requirement, add "Resolved by the 3b-1 sandbox: `docs/superpowers/specs/2026-10-17-plugin-sandbox-design.md`." Under step 10, add "Changed in 3b-1: `lab judge` no longer attempts the exam; only `lab exam` does." Under Holdout budget, add one line on the stored window end and `exam_max_attempts`.
- [ ] **Step 5: Run** `.venv/bin/pytest -q` — all pass, and `.venv/bin/pytest -q -m sandbox` shows no skips on this machine.
- [ ] **Step 6: Commit** — `refactor: plugin code executes only in the sandbox session; docs`
