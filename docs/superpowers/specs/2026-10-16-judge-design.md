# Phase 3a: the judge

Status: design approved in conversation 2026-10-16; this document awaits review.
Scope: the judge only. The LLM proposer gets its own spec (Phase 3b); requirements this
design places on it are listed at the end.

## Goal

Given a strategy plugin, decide **promote** or **kill** with no human involved, record the
decision and its reasons in the lineage, and give each family at most one honest exam on
the locked holdout. A strategy that passes everything becomes *eligible* for paper; a human
still approves paper in the dashboard. Nothing sets `live`.

The harness is the judge: nothing a strategy or a proposer does can influence scoring.
Safety-critical rules (holdout access, budget, gates) are enforced in code, not prompts.
Until plugins run in a sandbox (a Phase 3b requirement, at the end), that guarantee holds for
honest plugin code only: a plugin runs in the judge's process.
Resolved by the 3b-1 sandbox: `docs/superpowers/specs/2026-10-17-plugin-sandbox-design.md`.

## Decisions

| Question | Decision |
|---|---|
| Holdout rationing | Global cap on reads **and** at most one read per family. The proposer sees pass/fail only. |
| Promote rule | Child passes every absolute gate. If its parent also passed, the child must also beat the parent on a paired comparison of OOS daily returns. A failing parent never lowers the bar. |
| Multiple testing | Deflated Sharpe Ratio gate. N counts trials store-wide, so starting new families doesn't reset it. A trial is registered before plugin code first runs on real bars, so N counts registered trials as well as recorded walk-forward runs: a run that crashes still counts. |
| Holdout pass | Positive, and consistent with the child's own walk-forward record. |
| Paper gate | Every strategy (hand-written included) needs a passed holdout exam to be approved for paper. |
| Shape | A synchronous `lab judge` command plus a `perp_lab/judge/` package. No queue or daemon. |
| Short holdout | No exam until the holdout window is at least 180 days of complete data. History start moved to 2020-01-01 by `2026-10-18-wider-data-design.md`. |

Rejected: a single 70/30 split instead of walk-forward (less OOS data than the current
4 years of rolling OOS), and moving `holdout_start` earlier (it would put already-seen data,
and data inside an LLM proposer's training knowledge, into the holdout).

## Components

New package `perp_lab/judge/`, one job per file:

- `gates.py`: pure functions, no I/O. `deflated_sharpe`, `beats_parent`, `holdout_consistent`.
- `budget.py`: the **only** caller of `issue_holdout_token`. Reads the audit log, enforces the
  cap and the per-family rule, mints under a file lock.
- `exam.py`: runs the holdout exam in the judge's own process with frozen parameters.
- `judge.py`: orchestration (the flow below) and the public verdict.

Moved: `assert_causal` and its synthetic-bar generator move from `tests/` into
`perp_lab/validate/causality.py` so the judge can run the lookahead check at judge time;
`tests/test_lookahead.py` imports them from there.

Changed: `validate_strategy` accepts an explicit `lineage_id`; `approve_paper` gains the
holdout requirement; `add_edge` refuses to overwrite an edge with `decided_by="harness"`,
and refuses any human edge to a child run that already has a harness edge (no re-parenting);
the dashboard shows judgements and the holdout budget.

## Flow: `lab judge STRATEGY --tf TF [--symbols ...] [--workers N]`

1. **Load** the plugin. `PARENT` (module attribute) names the parent plugin; absent means a root.
2. **Pre-checks** (refuse, exit code 2, nothing recorded):
   - `PARENT` names a plugin that doesn't exist.
   - The child file is byte-identical to its parent (same `plugin_hash`).
   - `TF` isn't in the plugin's declared `timeframes`, or in its parent's (added 2026-10-18): the
     walk-forward would be a trial that can't be taken back, so a mistyped or defaulted `--tf`
     (the CLI defaults to 4h) must not spend one. `lab walkforward` refuses the same way.

   These read plugin source only (no plugin code runs), so they come before 3b's sandbox check: a
   wrong `--tf` is refused as such on any machine.
3. **Idempotency.** If a judgement exists for this `(strategy_hash, timeframe)`, rewrite its
   verdict file, print it and exit 0. No new trial is created. The verdict and the printed
   header name the strategy that was judged, so a renamed byte-identical copy reports the
   original's judgement.
4. **Lookahead check** with `causality.assert_causal` on the default params and every grid
   corner. Failure → judgement `kill`, reason `lookahead`, no walk-forward run, not a trial.
   The synthetic bars first get seeded, causal, non-constant values in every column the
   generator leaves NaN or constant (`oi`, `oi_value`, `quote_volume`, `trades`,
   `taker_buy_volume`, `taker_buy_quote_volume`, `regime_adx`, `funding_interval_h`), so a
   peek at any column is seen. A plugin whose default params emit no signal on those bars
   cannot be verified and fails the check the same way
   (`defaults: no signals on synthetic data; causality cannot be verified`).
   The check runs twice: on the synthetic bars at their default (2023) dates, and on synthetic
   hourly bars dated from 60 days before the holdout start to 30 days past today, cut every
   30 days from 45 days before the holdout start and always at the span's end (the end cut
   catches any peek that starts before it; the spaced cuts catch peeks confined to a stretch).
   A span that hasn't reached the holdout has nothing to check. Lookahead keyed to holdout dates is caught
   here (reasons prefixed `holdout-dated:`), before any walk-forward or holdout read, without
   touching holdout data. Because the exam window keeps growing after a judgement, the exam
   step repeats the dated check up to the window's actual end just before the read (see 10).
   The no-signals rule applies only to the default-dated run: a plugin may honestly be quiet
   in the dated span (e.g. a seasonal strategy), and that is not lookahead.
5. **Parent run.** First the parent must pass the same lookahead check; if it fails, refuse
   (exit code 2, nothing recorded): `parent P fails the lookahead check; fix or judge it
   first`. Then: the latest walk-forward run of the parent at the child's timeframe whose
   `strategy_hash` equals the parent's current `plugin_hash` and whose symbol universe equals
   the child's. If none, run the parent's walk-forward first (recorded like any run; it counts
   as a trial). The parent's hash is taken before its lookahead check; if the run's
   `strategy_hash` is not that hash (the parent was edited during its check or its run),
   stop with an error (exit 1) before any judgement is recorded:
   `parent P changed while it was being checked or run; re-run lab judge`.
6. **Child walk-forward** via `validate_strategy(..., parent_run_id=parent_run, lineage_id=family)`.
   If the run's `strategy_hash` is not the hash being judged (the plugin was edited during the
   run), stop with an error (exit 1) before any judgement is recorded:
   `NAME changed while it was being judged; re-run lab judge`. `validate_strategy` itself hashes
   the plugin at its start and again before recording; a run whose plugin changed in between
   (its worker pool and sensitivity pass load the plugin separately, so it would mix versions)
   is refused and not recorded, so a later `lab judge` can never reuse it as a parent baseline.
7. **Gates** (section below) on the child. A strategy's *absolute gates* are: walk-forward
   verdict `candidate` and Deflated Sharpe ≥ `dsr_min`. The parent's absolute gates are
   evaluated the same way, with the same N, at the time the child is judged.
8. **Decide.** Promote iff:
   - the walk-forward verdict is `candidate` (existing gates: OOS Sharpe > 0, beats
     random-entry p95, no narrow parameter island, ≥ 30 OOS trades), **and**
   - Deflated Sharpe ≥ `dsr_min`, **and**
   - if the parent passed its absolute gates: `beats_parent` ≥ `parent_beat_frac`.

   Otherwise kill. Every failed check is listed as a reason.
9. **Record** the judgement; when there is a parent, also the lineage edge
   (`decided_by="harness"`, reason = the joined reasons). When a child with a parent is
   promoted and the parent's current version has a judgement at this timeframe whose
   `holdout_status` is `pending`, that judgement becomes `blocked` with reason `superseded`
   and its verdict is rewritten: a promoted child either beat its passing parent or its
   parent did not pass, so the parent's queued exam must not spend the family's single read.
   Only a parent judgement in the child's own family is superseded (a re-pointed PARENT must
   not cancel another family's exam), and never one that already holds the family's read
   according to the audit log (`budget.holds_read`: its exam crashed, was interrupted, or later
   waited as `plugin_changed`); it retries on that read. If the log can't be read, it is left
   queued. A failure while superseding still writes the child's
   verdict before the error propagates.
10. **If promoted**, attempt the exam (see Holdout exam). Outcome sets `holdout_status`.
    Changed in 3b-1: `lab judge` no longer attempts the exam; only `lab exam` does.
    Just before the budget is asked for a read, the dated lookahead check is re-run over
    `[60 days before holdout_start, exam window end]`, also trying the exam's frozen params as a
    variant; a failure blocks the exam with reason `lookahead` and no read is spent.
11. **Write the public verdict** and print a summary. The verdict file is written even when
    the exam cannot start (e.g. the holdout log was never initialised) or crashes; the error is
    then re-raised (exit 1).

`lab exam` retries every judgement with `holdout_status = pending`, oldest first, through the
same budget checks.

## Family identity

A family is the root strategy and all its descendants (`lineage_id`), written
`family:<root>`. It is resolved by plugin name along the `PARENT` chain: walking from the
plugin towards its root, the first plugin name that already has a judgement (on any timeframe
or file version) gives the family of its earliest judgement; if nothing on the chain has been
judged, the family is `family:<root>`. An existing judgement's family therefore takes
precedence over the chain: re-running or editing a root cannot reset the per-family read, a
judged child whose `PARENT` line is removed stays in its family, and a chain whose ancestor is
later edited into a root stays together. Plain walk-forward runs do not define a family. A
renamed and edited copy of a root starts a new family (a byte-identical copy returns the
original's judgement); the global cap bounds that.

## Gates (`gates.py`)

All resampling is seeded from a hash of the child's run ID, so judging is reproducible.

**Deflated Sharpe** (Bailey & López de Prado, 2014), on the child's OOS portfolio daily
returns `r` (T days):

- `SR` = mean(r) / std(r) (daily, not annualised); skew and kurtosis (Pearson, not excess) of r.
- `N` = number of distinct `(strategy_hash, timeframe)` among all walk-forward runs in the store.
- `V` = the no-edge sampling variance of a daily Sharpe over the candidate's own track,
  `1/(T−1)` (changed 2026-10-16 from the variance measured across past trials: one trial whose
  true Sharpe is far from zero — e.g. `bb_vwap_reversion` 15m at −4.30, ruined by costs —
  inflated the measured spread to 1.92 annualised and put the bar at ≈ 3.3 annualised OOS
  Sharpe for every later strategy; with `1/(T−1)` it is ≈ 1.5 at N = 6 over 4 years and
  ≈ 2.0 at N = 50). The parent is judged with the same N and its own T.
- `SR0` = 0 if N = 1, else `√V · ((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))`, γ = 0.5772156649.
- `DSR = Φ((SR − SR0)·√(T−1) / √(1 − skew·SR + (kurt−1)/4·SR²))`.
- Gate: `DSR ≥ dsr_min` (0.95).
- Reference value: OOS annualised Sharpe 0.45 over T = 1461, N = 1, normal returns → DSR 0.82.

**Beats parent** (only when the parent passed): align child and parent OOS portfolio daily
returns on common dates; moving-block bootstrap of the paired days (block
`bootstrap_block_days`, 5,000 resamples); result = fraction of resamples in which the child's
Sharpe exceeds the parent's. Gate: `≥ parent_beat_frac` (0.95).

**Holdout consistent**, on the exam's L daily portfolio returns `h`. Pass iff all of:

1. holdout trades, across all symbols, ≥ `holdout_min_trades` (10);
2. Sharpe(h) > 0;
3. cumulative return of h ≥ the `holdout_floor_pct` (5th) percentile of L-day cumulative
   returns drawn from the child's walk-forward OOS daily returns (moving-block bootstrap,
   block `bootstrap_block_days`, 10,000 resamples).

Too few trades is a fail ("not enough evidence"); the read is spent either way.

## Holdout budget (`budget.py`)

- **Record of reads:** the existing append-only audit log (`judge.holdout_log`, default
  `runs/holdout_log.jsonl`). `issue_holdout_token` writes the entry before any data is read;
  the entry now also carries `judgement_id`.
- **Global cap:** distinct `judgement_id`s in the log ≤ `holdout_max_reads` (10).
- **Per family:** at most one distinct `judgement_id` per `lineage_id`.
- **Retry:** a judgement that already has a log entry but no recorded exam result (a crash
  mid-exam) may mint again; it is the same read, since no result was published.
- **Minimum length:** no mint until the exam window (below) is ≥ `min_holdout_days` (180).
  Measured on complete data, not the calendar; 180 calendar days is 2026-12-28.
- **Order of checks:** length → cap → family. Outcomes: `pending: holdout too short`,
  `blocked: budget spent`, `blocked: family already examined`, or mint.
- **Initialisation:** `lab holdout-init` creates the empty log once and refuses if it already
  exists, so the record of reads is never reset.
- **Fail closed:** missing, unreadable or malformed log → refuse with an error, never assume
  zero reads.
- **Concurrency:** count-and-mint happens under an exclusive file lock
  (`<holdout_log>.lock`), so two judges can't both take the last read.
- **Stored window (3b-1):** the mint writes the window end into the read's log entry, every
  retry uses it, and a read gets at most `judge.exam_max_attempts` (3) attempts.

## Holdout exam (`exam.py`)

- **Window:** `[holdout_start, end)`, where `end` = the minimum over the run's symbols of
  min(last closed candle close, last funding settlement + its interval), floored to 00:00 UTC.
  Today funding is complete only through 2026-09-30 (data.binance.vision publishes finished
  months; `lab tail` stitches candles only), so the window currently ends there.
- **Parameters:** frozen at the walk-forward run's modal parameters (its
  `sensitivity.base_params`). No selection on holdout data.
- **Run:** in the judge's process only (the token is identity-checked and is invalid in worker
  processes). Per symbol: `load_bars(cfg, sym, tf, end=end, holdout=token)` from
  `history_start`, one continuous `run_backtest`, daily returns and trades (by signal time)
  sliced to the window. Portfolio = mean of daily returns, as in walk-forward.
- **Record:** a run with `validation_mode="holdout"` (metrics, trades, and a `report.md` with
  the window, frozen params, pass/fail with reasons and key metrics), linked from the
  judgement by `holdout_run_id`. Not a lineage child, not a trial.

## Who sees what

- **Public verdict:** `runs/verdicts/<strategy>@<tf>@<hash10>.json`. Contains decision,
  reasons, walk-forward OOS metrics and report path, DSR, N, SR0, beats-parent fraction,
  and `holdout_status` ∈ {`n/a` (killed), `pending`, `passed`, `failed`, `blocked`} with a
  fixed reason code.
  It never contains holdout returns, Sharpe, trades or any number computed from holdout data.
  This is the proposer's only input in Phase 3b.
- **Holdout details:** in the `runs` table and run directory like any run; shown to the human
  in the dashboard.

## Store

New table `judgements` (created by `md.create_all`; no changes to existing columns):

| column | notes |
|---|---|
| `id` | primary key |
| `strategy`, `timeframe`, `strategy_hash` | unique on `(strategy_hash, timeframe)` |
| `child_run_id` | FK runs; null when killed before walk-forward (lookahead) |
| `parent_run_id` | FK runs; null for roots |
| `lineage_id` | family |
| `decision` | `promote` \| `kill` |
| `reasons` | JSON list |
| `dsr`, `n_trials`, `sr0` | floats / int |
| `parent_passed`, `beats_parent_frac` | null when no parent |
| `holdout_status`, `holdout_reason` | `n/a` for kills; the reason is a fixed code (`awaiting_exam`, `too_short`, `budget_spent`, `family_examined`, `superseded`, `exam_error`, `plugin_changed`, `lookahead`, `passed`, `failed`; 3b adds `too_slow`, `plugin_error` and `incomplete_data`, and status and code `withdrawn` for a promotion withdrawn before its read by `lab withdraw`), never a number |
| `holdout_run_id` | FK runs, null until examined |
| `created_at`, `updated_at` | |

`approve_paper(run_id)` additionally requires a judgement for `run_id` with
`holdout_status = passed`; otherwise `PermissionError` naming the status.

## Dashboard

- Sidebar: `holdout reads: used / cap` and the **earliest exam date** (`holdout_start` +
  `min_holdout_days`). Funding data arrives a month at a time, so the exam window can reach
  `min_holdout_days` a few days later than that.
- Strategy detail: judgement panel (decision, reasons, DSR / N / SR0, parent comparison,
  holdout status and, for the human, holdout metrics). The approve button is disabled with
  the reason when the paper gate isn't met.
- Lineage: harness edges appear as they do today.

## Config (`lab.yaml`)

```yaml
judge:
  dsr_min: 0.95
  parent_beat_frac: 0.95
  bootstrap_block_days: 10
  holdout_max_reads: 10
  min_holdout_days: 180
  holdout_min_trades: 10
  holdout_floor_pct: 5
  holdout_log: runs/holdout_log.jsonl
  verdicts_dir: runs/verdicts
```

## CLI

- `lab judge STRATEGY --tf TF [--symbols] [--workers]`: exit 0 judged (promote or kill,
  including the idempotent re-print), 2 refused by a pre-check, 1 error.
- `lab exam`: retry pending exams; prints each outcome.
- `lab holdout-init`: create the empty holdout log, once.
- `lab link`: unchanged for human edges, but refuses to overwrite a harness edge or to give
  a harness-judged child run another parent.

## Testing

Each test is written first and seen failing.

- **Gates:** literal expected values on hand-built series, including the DSR reference value
  above; SR0 = 0 at N = 1; beats-parent scores 0 on identical series (ties are not wins) and
  1 for a child with the same spread and a higher mean; holdout fails on < 10 trades, on
  negative Sharpe, and below the bootstrap floor.
- **Budget:** cap reached; second family read refused; same-judgement retry allowed; missing
  and corrupt log refuse; two threads racing for the last read → exactly one mint; minimum
  length → pending.
- **Single minter:** no module in `perp_lab/`, `dashboard/` or `strategies/` other than
  `judge/budget.py` references `issue_holdout_token`, and none outside `perp_lab/holdout.py`
  references `_MINT_KEY` or calls `HoldoutToken(` directly (AST scan anchored on the repo root,
  so it cannot pass vacuously from another directory).
- **Paper gate:** candidate with holdout `pending` / `failed` / `blocked` refused; `passed` accepted.
- **Verdict:** a verdict for an examined judgement contains no key or value derived from the
  holdout run (checked against the holdout run's metric values).
- **Family identity:** re-judging a root on another timeframe keeps its `lineage_id`; a judged
  child whose `PARENT` line is removed keeps its family, and so does its descendant.
- **Lookahead:** a plugin peeking at the next bar of any exchange column (e.g. `oi.shift(-1)`) is
  reported; a plugin that never signals is reported as unverifiable; every real plugin passes;
  a child whose parent fails the check is refused.
- **Superseded:** a promoted child blocks its pending parent (`superseded`) and the family's
  read goes to the child; a killed child leaves the parent queued.
- **Edges:** `lab link` refuses to overwrite a harness edge; a human edge cannot re-parent a
  child run the harness judged.
- **End to end** on small synthetic data (temp config and store): a kill with reasons; a
  promotion left `pending` because the window is short; a promotion examined and passed.

## Out of scope / follow-ups

- Extend `lab tail` to stitch funding so the exam window reaches today.
- Phase 3b: the proposer.

## Requirements carried into the Phase 3b proposer spec

- The proposer model's training cutoff must fall **before** `holdout_start` (2026-07-01);
  otherwise its knowledge of market history leaks into the exam.
- The proposer process has no read access to `runs/` (except `runs/verdicts/`), `lab.db` or
  the holdout log, and never imports `perp_lab.judge` or `perp_lab.holdout`.
- The proposer submits work only by writing a plugin to `strategies/` and invoking `lab judge`
  as a subprocess.
- Plugin code, proposer-written or not, is untrusted: it must run in a sandboxed subprocess
  with no filesystem or network access, receiving only the bars it is given.
- The proposer has no read access to `data/` (it holds the holdout prices).
- The proposer sees only `lab judge`'s exit code and the verdict file, never its stdout or
  stderr.
- The proposer can't run `lab withdraw` (added 2026-10-18), nor call `judge.withdraw` or write
  `withdrawals.jsonl`: withdrawing its own promotions before their reads would let it pick which
  ones reach the holdout. Withdrawal is a human's decision on the walk-forward record.
- Until that sandbox exists, the "nothing a strategy does can influence scoring" guarantee
  holds only for honest plugins: a plugin running in the judge's process could read `data/`
  directly or forge an unlogged token via `_MINT_KEY`.
Resolved by the 3b-1 sandbox: `docs/superpowers/specs/2026-10-17-plugin-sandbox-design.md`.
