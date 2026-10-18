# Perp Lab

Honest backtesting, walk-forward validation and paper-trading for Binance USDⓈ-M perps.
The harness is the judge; nothing a strategy (or later, an LLM proposer) does can influence scoring.

## Quickstart

```bash
pip install -e .[dev]
lab backfill                          # data.binance.vision → ./data (idempotent, resumable)
lab tail                              # ccxt: stitch parquet → now
lab strategies                        # plugins + hypotheses
lab walkforward donchian_atr --tf 4h  # OOS walk-forward + full validation → ./runs/<id>/report.md
lab backtest donchian_atr --tf 4h     # single fixed-param run (IN-SAMPLE, labelled as such)
lab dashboard                         # Streamlit on localhost: strategies, detail, pairs, lineage (--address 0.0.0.0 to expose)
lab ingest                            # record any run dir not yet in the store
lab link PARENT_RUN CHILD_RUN --decision kill --reason "..."   # hand-made lineage edge
lab paper RUN_ID                      # only runs approved for paper in the dashboard
lab paper-refresh PAPER_RUN_ID        # recount regime changes → paper_eligible
lab holdout-init                      # once: create the holdout read log (the judge refuses without it)
lab judge donchian_atr_wide --tf 4h   # promote or kill against PARENT; a promotion waits for `lab exam`
lab exam                              # the only command that runs holdout exams, and retries pending ones (budget rules apply)
pytest
```

No API keys are needed for any of the above (public market data only). If keys are ever
needed they come from env vars — never from config files.

## Layout

```
config/lab.yaml          universe, TFs, costs, account, holdout date, validation knobs
perp_lab/data/vision.py  backfill (klines, fundingRate, metrics/OI) with checksum verification
perp_lab/data/store.py   parquet readers, holdout guard, funding/OI/regime alignment
perp_lab/data/live.py    ccxt REST tail + ccxt.pro closed-candle stream
perp_lab/engine/         event-driven backtest, liquidation math, metrics
perp_lab/validate/       walk-forward, sensitivity, Monte Carlo, regime, benchmarks, pipeline
perp_lab/paper/          paper runner (same Strategy class, live book fills)
strategies/              plugins — one hypothesis per file
tests/                   lookahead (auto on every plugin), fees/funding, liquidation, holdout, paper
```

## Results store + dashboard (Phase 2)

- `perp_lab/store/db.py`: SQLAlchemy Core schema (`runs`, `run_folds`, `run_symbols`, `run_regimes`,
  `lineage_edges`, `status_log`). SQLite at `runs/lab.db`; point `store.url` at Postgres to switch
  (env vars are expanded, so the password stays in `$PGPASSWORD`). `write_run()` records every run.
- Status: `noise`/`candidate` come from the harness verdict. `paper` only via the dashboard's approval
  button, and only for a walk-forward candidate. Nothing in the system can set `live`.
- Paper eligibility = `min_paper_days` elapsed **and** ≥1 regime change. A regime change is a switch of
  the daily dominant 4h regime that persists ≥ `regime_change_min_days` (raw 4h labels flip ~24×/60d,
  which would make the gate meaningless), counted per pair, min across pairs.
- Backtest-vs-paper divergence replays the engine on exactly the bars the paper runner saw.
- Trade charts (run detail page, `dashboard/trade_chart.py`): one symbol's candles over a date range with
  every trade on them: the entry at the fill (▲ long, ▼ short), the signal bar's close, the exit coloured by
  `exit_reason`, the initial stop and initial target as dotted lines, and a hover with P&L, R, regime at signal,
  fees, funding, fold and params. Walk-forward runs shade every other test fold and label each (hover: its train
  window and params); paper runs shade the warmup. "Jump to trade" zooms to one trade; a window over 6000 bars
  is drawn at the next coarser timeframe. Walk-forward and paper runs only.
- Chart candles come from where the run got them, so nothing new is stored: a walk-forward reads the research
  parquet store through `load_candles` (no token, window cut at `holdout_start`), so any machine that can re-run
  it can chart it; a paper run reads its own `bars_<symbol>.parquet`. Holdout exams and IS runs aren't charted.
  Only the initial stop and target are logged, so a trailing stop or a moved target exits off its dotted line.

## The judge (Phase 3a)

`lab judge PLUGIN --tf TF` decides **promote** or **kill** with no human involved
(spec: `docs/superpowers/specs/2026-10-16-judge-design.md`).

- A plugin names its parent with a module-level `PARENT = "name"`; without one it starts a family.
- A family is `"family:<root plugin name>"`, found by walking PARENT links, so a whole PARENT chain
  (e.g. `donchian_atr` and `donchian_atr_wide`) shares one family and therefore one holdout read. A plugin
  that was already judged keeps its family, so deleting its PARENT line later doesn't start a new one. Parent
  runs are reused only within the family, so the first `lab judge donchian_atr_wide` re-runs the parent once
  under the family rather than reusing an older plain run.
- The lookahead check runs at judge time on the plugin and on its parent (a cheating parent is refused),
  on synthetic bars in which every column moves; a plugin that never signals there is killed as unverifiable.
  It runs again on synthetic bars dated across the holdout (to a month past today) so lookahead keyed to
  holdout dates is caught, and once more over the actual exam window just before a holdout read; a hit
  there blocks the exam with reason `lookahead` and spends no read.
- Gates: the walk-forward verdict, a **Deflated Sharpe ≥ 0.95** with N = distinct (plugin version,
  timeframe) pairs among walk-forward runs store-wide and the luck correction based on the spread no-edge
  strategies would show over the same track (≈ 1.5 annualised OOS Sharpe needed at N = 6 over 4 years),
  and, against a parent that passed, beating it in ≥ 95% of paired block-bootstrap resamples.
- A promoted plugin gets **one holdout exam per family**, under a global cap of 10 reads, and only
  once the holdout has 180 days of complete data (from about 2026-12-28). The exam runs frozen params and must
  make ≥ 10 holdout trades, be positive, and be no worse than the walk-forward record's p5 for that length.
- When a child is promoted, its parent's exam, if still pending in the same family, is blocked with reason
  `superseded`, so the family's one read goes to its best member, unless the audit log shows the parent
  already holds that read (a crashed or interrupted exam), in which case it keeps it and retries, or `lab exam`
  is examining the parent at that moment, in which case its exam goes ahead.
- If a plugin file is edited after it was judged, its pending exam waits with holdout status `pending` and
  reason `plugin_changed` (no new read is spent; one already spent by a crashed exam stays with it) until the
  judged version is restored; an edited plugin is a new version and is judged separately.
- `lab exam` reports each failing exam with ` ERROR: ...`, continues with the rest, and exits 1 if any failed.
- **Paper approval needs a passed exam**, for every strategy.
- Public verdicts in `runs/verdicts/` carry the holdout outcome as status and code only, never numbers.

## Plugin sandbox (Phase 3b-1)

Spec: `docs/superpowers/specs/2026-10-17-plugin-sandbox-design.md`.

- Plugin code runs only in a confined worker process (`perp_lab/sandbox/`); `session.py` is the one module that executes it.
- A worker loads its modules from the Python installation (everything under `sys.base_prefix`, which for a system Python is all of `/usr`), the virtualenv and `perp_lab/`, never from `data/`, `runs/`, `config/`, `strategies/` or anything else in `$HOME`. Then, before any plugin byte arrives, it seals itself: from there on it can't open, stat or readlink any path, so it reads no file at all (the bars come through its pipe), and it changes no file's metadata (mode, owner, times, access times of files and symlinks included, extended attributes, inode flags). It opens no socket and starts no process.
- A sealed worker can't query page-cache residency or evict or prefetch pages (no `mincore`, no such `madvise` advice, `MADV_HUGEPAGE` included: on a file mapping it reads 2 MiB in per fault), and after each batch of an exam's workers the files they mapped are evicted from the page cache. That narrows the page-cache channel but doesn't close it: a live process mapping the same libraries can keep a page a worker touched resident through fault-around, and a worker can still touch the pages it chooses (spec R7). A `lab paper` run's workers, which hold live holdout-window bars, are neither evicted nor kept apart from a running exam; that is accepted, since paper runs only exam-passed, hand-approved plugins (spec R6).
- A plugin can import only what the worker loaded before sealing: `math`, `numpy` (not `numpy.testing` or `numpy.f2py`), `pandas`, `perp_lab.strategy`, `perp_lab.indicators` and `perp_lab.engine.backtest`, and whatever these load, plus the interpreter's built-in and frozen modules, which need no file (here, for one, `itertools`, `time`, `zlib` and `unicodedata`). Anything else is `ModuleNotFoundError`, and no named time zone but UTC works (the time zone database is files).
- It reports decisions, not results: the trusted engine replays them, and any disagreement is a `mismatch`.
- `lab judge` runs lookahead checks, walk-forward and the benchmark through workers; an error caused by the plugin is a recorded kill (`plugin_error`).
- `lab exam` streams the holdout to each worker one bar at a time, symbols in lockstep, and is the only command that runs exams; one exam runs at a time on a machine.
- Needs Linux, Landlock ABI 6 or later, `libseccomp.so.2` and glibc 2.39 or later (older glibc makes `fstat` with a call the seal denies); without them plugin code does not run. No flag bypasses it.
- Tests that need the real sandbox carry `@pytest.mark.sandbox`; run them alone with `pytest -m sandbox`.

## Rules the code enforces

| Rule | Where |
|---|---|
| Signal at close of t, fill at open of t+1 | `engine/backtest.py`, `tests/test_engine_costs.py` |
| No lookahead in any plugin | `tests/test_lookahead.py` runs every file in `strategies/`, incl. a planted cheater that must fail |
| Holdout (`holdout_start`, a **fixed date**) unreadable without a minted, logged token | `holdout.py`, `data/store.py`, `tests/test_holdout.py` |
| Funding charged at real settlement timestamps (4h/8h both handled), reduces isolated margin | `data/store.py::align_funding`, engine step 1 |
| Stop beats target when both inside one bar; targets need trade-through | engine step 5 |
| Entry rejected if liquidation price sits inside the stop | engine step 4 |
| OOS is the headline; IS is shown greyed and labelled | `validate/pipeline.py` report |
| A strategy must beat random-entry (same exits, matched frequency) p95 OOS | `validate/benchmarks.py` |
| Within `perp_lab` code, only the budget can unlock the holdout; one read per family, 10 in total, after 180 days | `judge/budget.py`, `tests/test_budget.py` |
| Plugin code runs only in a confined, sealed worker: Landlock, seccomp and rlimits, then a seal under which no path resolves; it reads no file (no data, store, log or verdict, and once sealed not even its own modules' files), writes no file or file metadata (access times included), opens no socket | `sandbox/confine.py`, `sandbox/worker.py`, `tests/test_confine.py`, `tests/test_worker.py` |
| Scores come from the trusted engine replaying the plugin's decisions, never from numbers the plugin reports | `sandbox/replay.py`, `tests/test_replay.py` |
| The exam streams the holdout one bar at a time, symbols in lockstep, so a plugin never holds a later bar | `sandbox/stream.py`, `tests/test_stream.py` |
| Only `lab exam` runs exams; `lab judge` records `awaiting_exam` and returns | `judge/judge.py`, `tests/test_judge_exam.py` |
| Paper approval needs a passed holdout exam | `store/db.py::approve_paper`, `tests/test_store.py` |
| Harness lineage edges can't be overwritten by hand, nor a harness-judged run re-parented | `store/db.py::add_edge` |
| Verdicts never carry holdout numbers | `judge/judge.py::write_verdict`, `tests/test_judge_exam.py` |
| Trade charts read research candles only before `holdout_start`, without a token; holdout exams are never charted | `dashboard/trade_chart.py`, `tests/test_trade_chart.py` |

## Modelling choices (and their bias)

- **Liquidation**: isolated, one-way, tier-1 maintenance margin from config (approximate public values —
  exact brackets need an authenticated endpoint). Checked against last-price high/low, which wicks
  further than mark price → slightly pessimistic.
- **Portfolio**: each symbol is an independent book with equal capital; portfolio = mean of daily returns.
- **Walk-forward**: each grid combo is run continuously and windows are sliced (prepare() is proven causal),
  so a position opened just before a test window can carry into it. Its entry used only pre-test data.
- **Param selection**: one param set per fold for the whole universe (pooled mean Sharpe on train).
- **Regimes**: computed on native 4h bars for every TF so labels are consistent across TFs and pairs.
- **Paper**: fills at live bid/ask (spread replaces the slippage param); warmup comes from the exchange,
  not the research loader.
- **Data gaps**: data.binance.vision is missing a few days for SOL/XRP in Feb/Mar 2022; logged, not patched.
