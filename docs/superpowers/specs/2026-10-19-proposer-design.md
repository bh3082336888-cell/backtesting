# Phase 3b-2: the proposer

Status: draft, 2026-10-19. The goal and the record of ideas were agreed in conversation on
2026-10-18. On 2026-10-19 the user skipped the remaining questions ("write the design"), so every
other decision below is the author's call and open to review. The ones most worth a second look
are marked ★, and the last section lists them. An independent builder's review, checked against
the code, raised nine questions and three contradictions. All are settled in this text.

Scope: the loop that proposes strategies and gets them judged, the harness changes it needs, and
where the user reads the result. The judge's bar is unchanged.

## Intent

In the user's words: "automate building new strategies and testing them on crypto. I am obsessed
with finding an edge." After a week unattended they want "some edge OR in the very least some
progress on finding an edge". Progress means a record of "which ideas showed signal before costs
and which are dead".

Why now: the harness is finished (judge 3a, sandbox 3b-1, wider data). Seven hand-written trials
gave seven kills. Trial 7 took a day of research for one idea. The proposer produces ideas faster,
and the harness keeps the bar honest.

What was agreed on 2026-10-18:
- The record is kept per idea, not per file. A dead idea stays dead however it is rewritten, so
  the builder doesn't spend trials on near-copies.
- The builder reads the record before it proposes. It aims where nothing is ruled out yet,
  especially at the "signal before costs" pile.
- "Signal before costs" needs its own luck check: it must beat random entries before costs, not
  just show gross > 0.
- A third pile for close calls: paid its costs, missed the Sharpe bar.
- Rules unchanged: every submitted strategy counts as a trial. The record aims the builder; it
  doesn't lower the bar.

## Goals

1. A loop that runs unattended for days. Each round it proposes an idea, writes its plugin, gets
   it judged, records the outcome, and goes again.
2. The record of ideas as agreed above, readable by the user in ten minutes.
3. Feedback rich enough to learn from a kill. Trial 7's verdict said only "kill, Sharpe −1.26".
   The useful diagnosis came from a separate script: gross against net, per side, per symbol and
   fold, crash months.
4. Every requirement that 3a and 3b-1 placed on the proposer holds by construction, not by
   instruction.

## Non-goals

- Lowering the bar. That includes discounting near-duplicate trials and screening many ideas
  then confirming a few on fresh data. Both are deferred judge changes.
- Running exams, withdrawing promotions, approving paper. These stay human.
- Live trading.
- New data such as open interest or liquidations. The proposer works with the columns already in
  the store.
- Fine-tuning, or several models voting.

## Anti-goals: failures even if it works

- The loop spends N on near-copies and raises the bar for every later idea.
- The model sees holdout data, exam timing, or output from real-data runs outside a counted trial.
- A long bias riding the 2020–26 drift gets read as an edge. Buy and hold had a Sharpe of 0.72
  over the OOS span, and 1.20 over its first two years.
- A week of runs leaves a record the user can't read.

## Requirements this design must meet

From the 3a spec ("Requirements carried into the Phase 3b proposer spec") and the 3b-1 spec
("Requirements carried into the Phase 3b-2 proposer spec"):

| requirement | how it is met |
|---|---|
| model training cutoff before `holdout_start` (2026-07-01) | config guard; the loop refuses to start otherwise (Model) |
| no read access to `runs/` except `runs/verdicts/`, nor to `lab.db` or the holdout log | the model reads nothing; the orchestrator reads only verdict contents and its own state (import-boundary and path tests) |
| never imports `perp_lab.judge` or `perp_lab.holdout` | AST test on `perp_lab/proposer/` |
| submits only by writing a plugin to `strategies/` and running `lab judge` as a subprocess | the orchestrator does exactly that; the model returns text only |
| writes only to `strategies/`, never to `perp_lab/`, `config/`, `tests/`, `.venv/` | the orchestrator writes new files in `strategies/` and its own state dir, nothing else (test) |
| no read access to `data/`, and can't `stat()` it | the model has no process; the orchestrator never touches `data/` (path guard in tests) |
| sees only `lab judge`'s exit code and the verdict file, never stdout or stderr | judge output goes to a human-only log that no prompt reads |
| learns of exam failures only as reason codes; no file timestamps, nothing that dates an exam | prompts render whitelisted verdict fields, no times; exam outcomes enter only at session start |
| never runs `lab exam` or `lab withdraw` | no code path; the subprocess argv allowlist is `lab judge` and `lab preflight` |
| plugins keep `PARENT`, `params`, `grid` and `timeframes` literal | AST check before preflight |
| assume it knows exactly how the lookahead check works (R1) | accepted, bounded by G3 (it can't pass an exam it peeks in); see Residuals |

## Decisions

| question | decision |
|---|---|
| Shape ★ | A trusted orchestrator, `lab propose`, in `perp_lab/proposer/`. The model is called through the API **with no tools**: text in, one typed proposal out. It reads no file and runs nothing. |
| Model ★ | Set in config, together with its training-data cutoff as the provider publishes it. The loop refuses to start unless the cutoff is before `holdout_start`. Default `claude-opus-5-5`, cutoff to be checked by the user. |
| Autonomy | Unattended. Exams, withdrawals and paper approval stay human. |
| Memory | Each proposal starts from a fresh context holding the record. The record is the model's only memory, so it can be audited. |
| What a trial is ★ | As today, plus one rule: a kill for a plugin error after plugin code ran on real research bars counts as a trial. |
| Feedback | The verdict gains a `diagnostics` block computed from the walk-forward run only. |
| Luck check before costs | The random-entry benchmark also runs before costs. "Signal before costs" means a gross OOS Sharpe above the gross random-entry p95. |
| Random-entry sides ★ | Each random entry's side is drawn with the plugin's long share, not 50/50 (issue #17). This applies to both benchmarks, so it changes the existing gate too. |
| Ideas and timeframes ★ | An idea has one timeframe. A refinement's `PARENT` is the idea's champion. The same idea on another timeframe is a *port*: a new idea and a new family, allowed only from an idea whose best pile is signal before costs or close call. |
| Near-copies ★ | Measured by the harness, never declared by the model. Before any trial, decisions on synthetic bars are compared with every earlier plugin's. After the trial, the verdict reports overlap with earlier OOS runs, and the record acts on it. A mirror (the same entries, opposite side) of any judged plugin is refused. |
| Preflight | `lab preflight FILE --tf TF`: metadata checks, the lookahead check and the overlap check, all on synthetic bars. No real data, nothing recorded, no trial. Its full report may go to the model. |
| Pace ★ | 12 trials a day, 100 trials in total for the proposer, 4 judged plugins per idea, one judge at a time. |
| Exams | A judge and an exam never run at once: a harness lock, in both directions, that can't starve exams. Exam outcomes reach the model only as status and code, only at session start. |
| Where the user reads it | A new dashboard page, Ideas, plus a markdown digest per session. |
| State | Its own SQLite file and folders under `proposer/` (git-ignored), never `lab.db`. |

## Shape

```
lab propose  (trusted harness code, perp_lab/proposer/)
  1. builds the prompt from: static harness context + the record (verdict contents + idea cards)
  2. calls the model (API, no tools)  ->  one proposal: an idea card + plugin source (JSON)
  3. checks it in code: schema, idea rules, AST           -- fails -> back to the model (bounded)
  4. lab preflight <staged file> --tf TF   (sandbox, synthetic bars only)
                                                         -- fails -> back to the model (bounded)
  5. copies the plugin into strategies/
  6. lab judge <name> --tf TF   (subprocess; stdout and stderr to a human-only log)
  7. reads the verdict file's contents
  8. classifies the outcome in code, updates the record, writes the digest line; next round
```

Why this shape:
- The requirements forbid the proposer to read `data/`, `runs/` or `lab.db`, yet `lab judge` must
  read them. An agent with a shell would need a confinement that denies those paths. Landlock
  restrictions are inherited by child processes, so that confinement would also stop the
  `lab judge` it starts. Working around it means a broker process. And the model would gain
  nothing it needs, since it may not read data or run backtests anyway.
- With no tools, the model can't break any of these rules. The orchestrator is ordinary harness
  code, tested like the judge, and every rule is a check in it or in the harness. This follows
  "code over prompts for safety-critical behaviour" (ADR 0006). Nothing relies on telling the
  model a rule.
- Typed contracts generate the prompt's output description, the JSON schema and the validation
  (ADR 0007).

Rejected:
- An agent with file and shell tools in a confined process: see above.
- Showing the model `lab judge`'s output or the run's `report.md`: forbidden. `report.md` is under
  `runs/`, and judge output carries plugin stderr from real-data runs.
- One long conversation across proposals: its context grows without bound and the model's memory
  can't be audited. The record replaces it.

## One round

1. **Snapshot.** At session start only, re-read the verdict of every proposal in the record, which
   picks up exam outcomes. Within a session, read only the verdicts the session's own judges have
   just written.
2. **Prompt** (see What the model sees).
3. **Model call** → one proposal (see What the model returns). Validate it against the contract.
   Invalid → send the errors back. At most `fixups_per_proposal` (3) fix-ups per proposal, shared
   with step 5. After that the proposal is dropped: no trial, recorded as dropped.
4. **Idea rules**, in code:
   - A new idea needs a complete card and a title not already in the record.
   - A refinement names an existing idea that is open (see The record) and has judged plugins left
     (`proposals_per_idea`, 4). It keeps the idea's timeframe.
   - A port names an idea that is open or exhausted, whose best pile is signal before costs or
     close call, and a timeframe that idea has no port on yet. At most one port per idea and
     timeframe.
   - Exactly one timeframe per plugin, from `proposer.timeframes` (15m, 1h, 4h). The plugin's
     `timeframes` must be exactly that one. The judge refuses a child whose parent doesn't declare
     the child's timeframe, and a parent declaring two would get a walk-forward at the second: an
     extra trial the record wouldn't see. That is why refinements keep the timeframe and ports
     are roots.
5. **Assemble and check.** The orchestrator writes `proposer/staging/<name>.py`:
   - It chooses the name, `px<NNNN>_<slug>`, a valid module name of at most 40 characters. NNNN is
     the proposal's sequence number.
   - It writes the module docstring's first line from the proposal's `hypothesis`, which the judge
     records.
   - For a refinement it inserts `PARENT = "<the idea's champion>"` (see The record), so the idea is
     one family with one holdout read. When the idea has no champion yet (only broken attempts), the
     plugin is a root, and the first one with a walk-forward run sets the idea's family. A port is
     always a root. The model never writes `PARENT` (a `PARENT` in its source is refused).
   - AST checks: exactly one `Strategy` subclass; literal `params`, `grid`, `timeframes`; imports only
     from the sandbox's list (math, numpy, pandas, `perp_lab.strategy`, `perp_lab.indicators`,
     `perp_lab.engine.backtest`); at most `max_grid_combos` (16) grid combinations; at most 64 KB.
6. **Preflight**: `lab preflight proposer/staging/<name>.py --tf TF` (see Preflight). On failure
   its report goes back to the model, within the same fix-up budget.
7. **Copy** to `strategies/<name>.py`, refusing if the name exists. From here the loop never edits
   or deletes the file: its git blob hash is the plugin version.
8. **Judge**: `lab judge <name> --tf TF`, with the API key removed from the environment, stdout and
   stderr to `proposer/logs/<name>.log`. Exit codes:
   - 0: judged; go on to step 9.
   - 2: refused. Recorded as `broken`; preflight should have caught it.
   - 1: infrastructure error. Recorded as `broken`; it counts toward the day's trials (Pace).
9. **Read the verdict**, `runs/verdicts/<name>@<tf>@<hash10>.json`. The orchestrator computes the
   hash from the bytes it wrote. It stores the verdict's contents in the record.
10. **Classify** in code (see The record), update the idea's status, append a line to the digest.

## What the model sees

Every prompt is built from three sources only. A test plants sentinel values everywhere else and
checks that none reaches the model.

1. **Static harness context**, the same for every call (a cached prompt prefix):
   - what the lab is;
   - the plugin API: the source of `perp_lab/strategy.py` and `perp_lab/indicators.py`, the bar
     columns and their units, the allowed imports;
   - two example plugins, `flow_shock_fade` and `donchian_atr`;
   - the cost model (fees, the per-coin slippage formula, funding);
   - the universe and spans: 17 coins, research bars 2020-01-01 → 2026-07-01, OOS from 2020-07 in
     36 two-month folds;
   - the gates, what a trial is, and the Deflated Sharpe bar table (Pace);
   - the piles and the idea rules;
   - sections 1 and 2 of `docs/research/2026-10-18-strategy-research.md` (what the baselines taught,
     the cost hurdle by timeframe), labelled as the human's notes on 8 coins from 2022;
   - the output contract.
2. **The record**, rendered by code. Open ideas appear in full: card, each proposal's whitelisted
   verdict fields and diagnostics, and the champion's source, so a refinement can be a real change
   rather than a rewrite from the card. Closed ideas get one line each: title, signal, and why
   closed (for example "dead: gross Sharpe −0.9 against the gross random p95 0.1").
3. **The instruction for this round**: aim where nothing is ruled out; refinements of ideas in the
   signal-before-costs and close-call piles are welcome; return one proposal.

Whitelisted verdict fields: decision, reasons, OOS metrics, Deflated Sharpe, N, sr0, beats-parent
fraction, holdout status and code, diagnostics. Run ids and report paths are left out: run ids
encode a time. The record numbers proposals in order and dates nothing.

Today's date is not hidden. It says nothing about when an exam ran beyond what the session-start
snapshot already allows, and preflight reports carry synthetic-bar dates anyway. What must never
reach the model is the time of a judge run, an exam, or a file.

## What the model returns

One JSON object, from a dataclass in `perp_lab/proposer/contract.py`. The API call's structured
output schema, the prompt's description of it and the validation are all generated from that
dataclass.

- `kind`: `new`, `refinement` or `port`.
- `idea` (new): `title`; `mechanism` (what happens, and who pays); `why_it_persists`; `signal` (in
  words); `trades_per_coin_per_year` (an estimate); `likely_failure`; `columns` (the bar columns it
  reads). This is the research note's candidate template.
- `refines` (refinement): `idea_id`, plus `change` (what changes and why, citing the diagnostics
  behind it).
- `ports` (port): `idea_id`, plus `why` (why the other timeframe should keep the effect and lower
  its cost, citing the diagnostics behind it). The port inherits the idea's card.
- `timeframe`.
- `hypothesis`: one line.
- `source`: the plugin module, with no `PARENT`.

## Preflight: `lab preflight FILE --tf TF`

A new harness command. It runs plugin code only in the sandbox, and only on synthetic bars.
1. The judge's pre-checks on the file's source (metadata literals, timeframe declared).
2. The lookahead check exactly as the judge runs it: the default-dated synthetic bars, the
   holdout-dated synthetic span, the defaults and every grid corner. The judge records a lookahead
   failure as a kill; preflight records nothing. Harness change: the dated span's end is pinned to
   the current UTC day, not the hour (`_now()` floored to 00:00). Today it moves with the hour,
   and the synthetic bars change with the span's length. With the pin, a preflight pass means the
   judge's own check passes on the same UTC day.
3. **Overlap**: the plugin's entry decisions on the synthetic bars at its defaults, against those
   of every plugin in `strategies/` at any timeframe. Earlier plugins' decisions are cached by hash
   under `proposer/cache/`. The measure is the Jaccard index of (bar, side). A "mirror" is a match
   on (bar, −side). Near-copy means ≥ `overlap_max` (0.5).
   - A near-copy of a closed idea's plugin is refused.
   - A near-copy of an open idea's plugin, proposed as a new idea, is refused with "propose it as a
     refinement (or a port) of X".
   - A port is exempt from overlap with the idea it ports, and only that idea.
   - A mirror of any judged plugin is refused. A mirror's OOS result is close to minus the
     original's, which has already been seen, so its walk-forward would test nothing. This is the
     research note's exclusion of candidate E (sections 4 and 10), applied to every idea.
   Synthetic bars are a proxy: their taker, open-interest and volume columns are random walks, so
   two plugins can differ there and still coincide on real data. The after-trial overlap (The
   record) catches those.
4. Prints a JSON report and exits 0 (passed), 3 (failed a check) or 1 (infrastructure error).

Its tracebacks come only from runs on synthetic bars, so they carry no market data. A test
asserts that preflight never calls `load_bars` and never opens anything under `data/`. That is
why its full report may go to the model.

## The record

Store: `proposer/record.db`, SQLite through SQLAlchemy like `lab.db`.
- `sessions`: start, end, model and cutoff, caps, trials used, tokens in and out, stop reason.
- `ideas`: id, title, card (JSON), status, root plugin, mirror-of.
- `proposals`: sequence number, session, idea, kind, plugin name, timeframe, hash, source text,
  preflight report, judge exit code, verdict contents, pile, tokens.
- Per proposal, the full transcript (prompts and responses) in
  `proposer/transcripts/<NNNN>.jsonl`.

**Piles**, assigned in code from the verdict:

| pile | rule |
|---|---|
| promoted | decision `promote` |
| close call | killed, but net OOS Sharpe > 0 and above the net random-entry p95 (it failed on the Deflated Sharpe or a narrow island) |
| signal before costs | not the above, and the gross OOS Sharpe is above the gross random-entry p95 |
| no signal | none of the above, with at least the gate's minimum OOS trades |
| broken | killed for lookahead or a plugin error, refused, an infrastructure error, or too few OOS trades |

A promotion the human withdraws (`holdout_status` `withdrawn`) is filed as a close call. The human
judged the effect not established.

**The champion** of an idea is the plugin a refinement names as `PARENT`. It is chosen among the
idea's plugins that have a walk-forward run (never a broken one, since the judge refuses a child
whose parent can't be run):
- first, a promoted plugin whose exam is pending, so a refinement must beat it, and supersedes its
  queued exam if promoted;
- otherwise the best by pile, then by net OOS Sharpe.

**An idea's status.** Open ideas take proposals; every other status is closed for good.
- `open`: a new idea; one with only broken attempts (it gets at most 4 judged plugins like any
  other); or one whose best pile is signal before costs, close call or promoted, with plugins left.
- `dead`: its best pile is `no signal`. One faithful test that showed no signal before costs ends
  an idea, as agreed.
- `exhausted`: it used its 4 judged plugins. It keeps its best pile in the record and gets no more
  trials.
- `passed` or `failed`: its family's holdout read was spent on an exam with that outcome. After
  that, any promotion in the family is blocked as `family_examined`, so another trial could never
  reach paper. A passed idea is the edge the user is after; the human takes it from there.
- `duplicate`: the after-trial overlap folded it into a closed idea (below).

**After-trial overlap.** When a new idea's verdict shows OOS signal overlap ≥ `overlap_max` with a
closed idea's run, the new idea becomes a `duplicate` of it. Its trial is recorded as spent, and
the closed idea stays closed. The same goes for a mirror match (symbol, signal time, opposite
side) with any judged run. The verdict names overlapping runs by strategy and timeframe, which the
record maps to ideas.

**Global holdout budget.** When any verdict shows `blocked` / `budget_spent`, no promotion can
reach an exam any more. `lab propose` then refuses to start and says so.

**One idea is one family at one timeframe**, once it has a champion. Ports and the seeded trials
fit that rule.

**Seed** ★: the record starts with six ideas from the seven judged human trials, which cover five
plugin files in four families:
- `donchian_atr` at 4h, with `donchian_atr_wide` as its refinement;
- `donchian_atr` at 1h, as a port of the 4h idea;
- `bb_vwap_reversion` at 15m;
- `bb_vwap_reversion` at 1h, as a port of the 15m idea;
- `funding_fade` at 1h;
- `flow_shock_fade` at 1h.

The two seeded ports only record how the human trials relate. The port rule (signal before costs
or close call first) doesn't apply to them.

Their cards come from the plugins' docstrings and, for `flow_shock_fade`, from the research note.
Six of the seven were judged on the old data (8 coins from 2022, flat slippage, 24 folds). So
their piles come from fresh walk-forward runs on today's data, made by `lab verdict-refresh`
(below). These add no trial: N counts distinct (plugin version, timeframe) pairs, and the
wider-data spec already re-runs parents on today's data the same way.

The record also seeds:
- the research note's untested candidates B, C and D, as open ideas with cards taken from section
  3;
- candidate E, marked excluded: it is A's sign flip, and choosing it after A's result would pick
  the sign from an OOS outcome (sections 4 and 10).

## Verdict diagnostics (judge change)

The verdict file gains `"schema": 2` and a `diagnostics` block. It is computed from the
walk-forward run only (its `oos_trades`, `oos_daily` and summary), never from a holdout run. The
existing test that a verdict holds nothing derived from holdout data is extended to cover it.

- **Gates' inputs:** the net random-entry mean and p95 and the plugin's percentile, and the OOS
  trade minimum the gate applied. Today these are only in the run's `summary.json`, which the
  orchestrator may not read, and the piles need them.
- **Before costs:** the gross OOS Sharpe, the gross random-entry mean and p95, and the plugin's
  percentile among them.
- **Per trade, in bp of notional:** gross, fees, slippage, funding, net.
- **Sides:** long and short trades, the long share, gross and net bp per trade.
- **Exits:** the share by exit reason, and mean bars held.
- **Regimes:** per regime at signal, trades and gross bp per trade.
- **Breadth:** coins with net > 0 out of all coins; folds with net > 0 out of 36. When net > 0, the
  top coin's and top fold's share of it (the research note's criterion 6).
- **IS against OOS Sharpe.**
- **Selection:** how often each grid combination was picked across the folds.
- **Overlap:** the three earlier judged runs at this timeframe with the most OOS signal overlap
  (Jaccard of symbol, signal time and side), the three with the most mirror overlap (the same with
  the side flipped), and the three with the highest OOS daily-return correlation. Each entry names
  the run's strategy and timeframe and gives the value.

Left out ★: per-month or per-coin return series, trade lists, and the dates of the best and worst
stretches. The model already knows the OOS span's market history from its training. The finer the
feedback, the easier it is to fit that span rather than find a mechanism.

**Plugin-error messages.** For a plugin error after real research bars, the verdict's reason
carries only the error kind and, for exceptions, the type's name if it is a Python builtin
exception (`ValueError`, `KeyError`, and so on); any other type becomes `other`. Never the
message. A plugin chooses both its message and its exception class, and it can build a class
whose name encodes research data. The full reason stays in `lab.db` and the human's log. The kind
is set by the harness, not the plugin.

**`lab verdict-refresh STRATEGY --tf TF`** writes a judgement's diagnostics from the latest
walk-forward run of that plugin version and timeframe on today's data, the current `data_id`. If
there is none, it first makes one, with both random-entry benchmarks under today's rules. The
plugin version and timeframe are unchanged, so this is not a new trial. The verdict's decision
fields are never changed: they stay those of the original judgement. `diagnostics.data` says
whether the diagnostics come from the judged run or from a refresh on today's data.

## Trials: one change ★

A judgement killed for a plugin error after plugin code ran on real research bars now counts as a
trial. `trial_stats` counts these alongside walk-forward runs. Without this rule, a plugin could
crash on purpose depending on what it saw, which is a free, uncounted look at the OOS data, and
the proposer could submit only what it had already checked. Lookahead kills stay uncounted: that
check runs on synthetic bars only.

This needs issue #15 fixed first. A transient sandbox failure (`died`, `timeout`, `cpu_limit`
caused by the machine) must be retried and, if it recurs only intermittently, become an
infrastructure error with nothing recorded. Otherwise an honest plugin pays a trial for a machine
hiccup. The builder settles the retry rule within #15.

## Random entry: before costs, and side-matched ★

- **Side-matched (#17).** Each random entry's side is drawn with the plugin's overall OOS long
  share, not 50/50. Without this, a long-biased plugin beats random entries on drift alone, and an
  optimising proposer will find that. The change applies to the existing net gate too. It makes
  the gate stricter for long-biased plugins only.
- **Before costs.** The plugin's selected per-fold parameters are re-run with fees and slippage
  set to zero, and so is the random-entry benchmark. Funding is kept: it is part of holding the
  position, and a strategy may earn it. This is narrower than section 7's "gross", which was before
  fees and funding but after slippage.
- Cost: one more strategy pass and one more random-entry pass per judgement. Trial 7's whole
  judgement took about two minutes.

## Pace and budget

The Deflated Sharpe bar (annualised OOS Sharpe at DSR 0.95, normal returns, over the 2,191-day OOS
span), computed with `gates.dsr_from_moments`:

| N | 7 | 10 | 20 | 30 | 50 | 100 | 200 | 500 | 1,000 |
|---|---|---|---|---|---|---|---|---|---|
| bar | 1.24 | 1.32 | 1.45 | 1.52 | 1.60 | 1.71 | 1.80 | 1.92 | 2.00 |

Caps (config):
- **`trials_per_day`: 12.** When the day's quota is used, the loop sleeps until the next 00:00
  UTC. Infrastructure errors count toward it, so a plugin can't loop on them.
- **`trials_total`: 100**, for the proposer's own trials. N would then be about 107 and the bar
  about 1.72. When it is reached, `lab propose` refuses to start. Raising it is the human's
  decision.
- **`proposals_per_idea`: 4** judged plugins.
- **`fixups_per_proposal`: 3.**
- **Token caps per day**, input and output. An optional price per million tokens lets the digest
  estimate spend.
- One session at a time (a lock file in `proposer/`), and one judge at a time.

Why pace at all: the total cap is what protects the bar. Pace spreads the trials so the user can
read a day's record and stop the loop before the budget is gone. At 12 a day, a week is 84 trials.

Stop conditions:
- a cap;
- 3 infrastructure errors in a row;
- 5 proposals in a row dropped at the schema or preflight stage (the model is stuck);
- the API failing after retries (see Failures);
- SIGINT or SIGTERM. The first one lets a running judge finish and then stops. The second kills it
  and records the proposal as `broken` / `interrupted`; it counts toward the day's quota.

Rough size: a 1h judgement took about 2 minutes. The model call takes a few minutes. So 12 trials
take a few hours of a day. Per call, the static context is about 30–40k tokens (cached), and the
record adds about 5k tokens per open idea (card, verdicts, diagnostics, the champion's source) and
about 60 per closed idea. Output is 5–15k tokens with thinking. At about 1.5 calls per trial, a day of 12 trials is about 18 calls: roughly 1–2M input
tokens (mostly cached) and 0.2M output. The digest reports actual tokens.

## Exams and the human

- **Unchanged:** only a human or a scheduled job runs `lab exam`; only a human runs `lab withdraw`
  or approves paper. `perp_lab/proposer/` has no code path to any of them (import and argv tests).
- **A judge and an exam never run at once.** This is a harness change. `lab judge`, `lab
  walkforward`, `lab backtest` and `lab preflight` hold a shared lock for their whole run, and
  `attempt_exam` takes it exclusively. Exams get writer preference, so a stream of judges can't
  starve them: for example, a turnstile lock that `lab exam` holds while it waits. This closes R6's
  overlap between a lookahead check and an exam for every caller, not just the loop. It is the
  "pause the proposer during exams" follow-up.
- **What the model learns of exams:** `holdout_status` and its code, only from the session-start
  snapshot, and nothing with a time attached.
- **Review before an exam:** for each promoted plugin, the Ideas page shows the criterion-6 checks
  from its diagnostics (sides, top coin and top fold share). The human decides whether to withdraw
  it before running `lab exam`.
- **Holdout reads:** 10 in total, one per family. Each new idea is a new family, so the loop can
  make many. What reaches an exam is still the human's choice, through `lab withdraw`.

## Model

- **Config:** `proposer.model`, plus `proposer.model_training_cutoff` (YYYY-MM-DD). The cutoff
  comes from the provider's published model documentation; the harness can't verify it, so the
  human enters it. The loop refuses to start when it is missing or not before `holdout_start`.
  Each session and proposal records the model and the cutoff.
- **Default:** `claude-opus-5-5`, the strongest model available. Before the first session the user
  checks its published training-data cutoff against 2026-07-01. If it isn't earlier, use an earlier
  model.
- **What the rule doesn't fix:** any model that meets it still knows the market history of the
  whole OOS span, 2020–2026. So the walk-forward is not out of sample for the proposer, and a
  promotion is a hypothesis until its holdout exam. That is why the cutoff rule exists and why
  exams stay scarce.
- **API key:** read from `ANTHROPIC_API_KEY`. It is never written to the record, transcripts or
  logs. The `lab judge` and `lab preflight` subprocesses get an environment without it.
- **Dependency:** the `anthropic` SDK, as an optional extra `[proposer]`. Whether calls use
  extended thinking, and how prompt caching is set up, is left to the builder.

## Where the user reads the record

- **Dashboard page "Ideas"** (Streamlit, beside the existing pages):
  - header: proposer trials used against the total cap; N store-wide and the bar at that N; the
    session's state (running, sleeping until a time, stopped and why); tokens today;
  - the ideas by status and pile, with progress first: passed, promoted, close call, signal before
    costs. Then the other open ideas. Then the closed ones (dead, exhausted, failed, duplicate) and
    broken attempts, collapsed;
  - per idea: its card and its proposals, each with plugin source, verdict, diagnostics and a link
    to the walk-forward report; for promoted ones, the review packet.
- **Session digest:** `proposer/sessions/<id>.md`. One line per proposal is written as it happens.
  At the end it adds counts per pile, any new ideas in the progress piles, the stop reason and
  tokens. It can be read without the dashboard.

The dashboard is for the human, so it may show everything, holdout details included, as it does
today.

## Config

```yaml
proposer:
  model: claude-opus-5-5
  model_training_cutoff: null        # YYYY-MM-DD from the provider's docs; required, < holdout_start
  state_dir: proposer                # record.db, staging/, logs/, transcripts/, sessions/, cache/
  timeframes: [15m, 1h, 4h]
  trials_per_day: 12
  trials_total: 100
  proposals_per_idea: 4
  fixups_per_proposal: 3
  max_grid_combos: 16
  overlap_max: 0.5
  max_input_tokens_per_day: 4000000
  max_output_tokens_per_day: 500000
  price_per_mtok: {input: null, output: null}   # optional, for the digest's spend estimate
```

`proposer/` is added to `.gitignore`.

## CLI

- `lab propose [--days 7] [--trials N]`: run a session until D days, N trials or a stop condition.
- `lab ideas`: print the record by pile.
- `lab preflight FILE --tf TF`: also useful to humans.
- `lab verdict-refresh STRATEGY --tf TF`: recompute an existing judgement's diagnostics.

## Failures

- **API errors:** retry with backoff, 5 tries. Then the session stops (reason `api`).
- **Invalid model output:** fix-ups, then the proposal is dropped (no trial).
- **Preflight infrastructure error** (exit 1): the session stops after 3 in a row.
- **Record or state-dir errors:** fail closed; the session stops.
- **Orchestrator crash:** on the next start, a proposal whose plugin is in `strategies/` but whose
  verdict was never read gets `lab judge` again. If a judgement exists, this only rewrites the
  verdict. If not, the same bytes are judged now. The model never saw the lost run, so it gains
  nothing.

## Residuals

- **Exit-code channel.** A plugin that could make `lab judge` exit 1 on purpose would get a bit
  without a trial. This is bounded by the stop rule (3 in a row) and by counting these toward the
  day's quota.
- **R1 from 3b-1** (behaving differently under test) matters more with an optimising author. It is
  bounded by G3: such a plugin can't pass its exam. The 3b-1 follow-up "narrowing R1" (a streamed
  re-run over a secret stretch of real OOS data) is recommended before the first exam of a
  proposer-written plugin.
- **The model knows the OOS history** (Model). The holdout is the only clean test.

## Testing

Each test is written first and seen failing. No test calls the network; the orchestrator's tests
use a fake model client.

The orchestrator:
- **Prompt sources:** sentinels planted in `lab.db`, a run's `report.md`, the holdout log, `data/`
  and the judge's stdout never appear in anything sent to the model client, over a full round.
- **No times:** distinctive times planted in verdict files' modification times, run ids, the
  holdout log and the judge's output never appear in anything sent to the model.
- **Boundaries:** `perp_lab/proposer/` imports none of `perp_lab.judge`, `perp_lab.holdout`,
  `perp_lab.store`, `perp_lab.data` (AST scan anchored on the repo root). Subprocesses are only
  `lab judge` and `lab preflight`. Files are written only under `state_dir`, plus new files in
  `strategies/`; nothing is overwritten.
- **Idea rules:** a closed idea refused; a refinement gets its idea's champion as `PARENT`. The
  champion is a pending promotion first, never a broken plugin; with no champion the plugin is a
  root. A `PARENT` written by the model refused; a port only from a signal-before-costs or
  close-call idea, once per timeframe, always a root; the one-timeframe rule, the grid cap and the
  import allowlist; the fix-up bound.
- **Piles and statuses:** a table of verdict fixtures with their expected piles, including trial
  7's verdict, which is no signal, and a withdrawn promotion, which is a close call. An exam
  passed or failed closes the idea. An after-trial overlap or mirror match makes the new idea a
  duplicate. `budget_spent` refuses the next start.
- **Session:** the daily quota sleeps; the total cap refuses to start; the cutoff guard refuses;
  SIGINT once and twice; 3 infrastructure errors stop.
- **Secrets:** the API key never appears in the record, transcripts or logs, and the subprocess
  environment lacks it.

The harness:
- **Preflight:** never calls `load_bars` and never opens `data/`. A lookahead plugin fails
  preflight and the judge alike. The dated span is the same at any hour of one UTC day. Overlap
  catches a byte-different copy and a mirror, and exempts a port from its own idea only.
- **Verdict v2:** diagnostics present and computed from the walk-forward run, and still nothing
  derived from holdout data. A plugin-error message is absent from the verdict and kept in
  `lab.db`. A builtin exception type is kept; a plugin-defined one becomes `other`.
- **Verdict refresh:** a refresh on today's data changes no decision field, adds no trial, and
  marks `diagnostics.data`.
- **Trials:** a plugin-error kill after real bars raises N; a lookahead kill doesn't.
- **Random entry:** side-matched draws follow the long share; the gross variant has zero fees and
  slippage.
- **Lock:** no exam starts while a judge runs, and no judge while an exam runs. A stream of judges
  doesn't starve a waiting exam.

End to end, on small synthetic data with a temp config and store: a fake model proposes
1. a broken plugin, then its fix;
2. a near-copy of a dead plugin, refused at preflight;
3. a sound plugin, which is judged.

The record shows the right piles and the digest is written.

## Suggested build order

1. Harness changes, each its own commit with its tests:
   - #15 (transient failures);
   - the trial-count rule and the verdict message rule;
   - random entry side-matched and before costs;
   - verdict diagnostics and `lab verdict-refresh`;
   - the judge–exam lock;
   - the dated lookahead span pinned to the day;
   - `lab preflight`.
2. The orchestrator: contract, record store, prompt builder, model client, the loop, the CLI.
3. The Ideas page and the digest.
4. Seed the record and refresh the seven trials' diagnostics.

## Changes to existing documents and tests

- `2026-10-17-plugin-sandbox-design.md`, Failures, walk-forward plugin errors: the sentence "the
  message may reach the verdict ... No run is recorded, so it is not a trial" is replaced by this
  spec's two rules. The message stays out of the verdict, the exception type is kept only if
  builtin, and the kill counts as a trial.
- `2026-10-16-judge-design.md`, Who sees what: the public verdict also carries the `diagnostics`
  block, computed from walk-forward data only.
- `tests/test_judge.py`: `test_a_plugin_error_in_the_walk_forward_is_a_recorded_kill` asserts the
  message in the reasons. It changes to assert the verdict's reduced reason, the full reason in
  `lab.db`, and N raised by one. Its neighbour for a child's plugin error gets the same N
  assertion.

## Out of scope and follow-ups

- Cheaper trials, both deferred judge changes: counting near-duplicates as fewer effective trials
  (harness-measured), and screening many ideas then confirming a few.
- Narrowing R1, before the first exam of a proposer plugin (Residuals).
- A review mode, where the human approves each idea before its trial.
- The change-7 spec (the exam rule for coins that stop trading), still due before the first exam,
  about 2026-12-28.
- Other model providers.
- Git: the loop never commits. The record keeps every plugin's bytes, and the human commits what
  they want.

## Calls made by the author, for review

★ in the text. The ones that most change what gets built:
1. **No tools for the model**, with a trusted orchestrator doing everything (Shape).
2. **Plugin errors after real bars count as trials**, with #15 fixed first (Trials).
3. **Random entry side-matched (#17)**, which changes the existing gate for every later trial.
   The user had deferred #17 before the proposer existed.
4. **Pace: 12 trials a day, 100 in total**, and 4 judged plugins per idea (Pace).
5. **Feedback granularity:** aggregates only, no per-month or per-coin series (Verdict
   diagnostics).
6. **"Before costs"** means zero fees and slippage with funding kept (Random entry).
7. **The seed:** six ideas from the seven trials, piled from fresh walk-forward runs on today's
   data (no new trial), plus candidates B, C and D, with E excluded (The record).
8. **Default model `claude-opus-5-5`**, its cutoff to be checked by the user (Model).
9. **Ideas are (family, timeframe) pairs.** A refinement keeps the timeframe and names the
   champion as `PARENT`. The same idea on another timeframe is a port: a new family, allowed only
   from a signal-before-costs or close-call idea (Ideas and timeframes, The record).
10. **Mirrors of any judged plugin are refused**, extending the research note's exclusion of E to
    every idea (Preflight).
11. **Closed for good:** an idea whose family's holdout read was spent takes no more trials, and a
    withdrawn promotion counts as a close call (The record).
