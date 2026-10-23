# Phase 3b-2: the proposer

Status: approved, 2026-10-21. The goal and the record of ideas were agreed in conversation on
2026-10-18. On 2026-10-19 the user skipped the remaining questions ("write the design"), so every
other decision below was the author's call. The ones most worth a second look are marked ★, and
the last section lists them. An independent builder's review, checked against the code, raised
nine questions and three contradictions. All are settled in this text. Revised the same day after
the user said that their only model access is their Claude Code login: the model runs through
Claude Code headless (`claude -p`), not the API. Revised again on 2026-10-21 after a second review
against the code. The user approved folding in all of its findings, and with them the ★ calls:
- a trial is counted when plugin code first runs on real bars;
- the seeded ports share a family with the ideas they port;
- the static context is current;
- "signal before costs" needs a gross Sharpe above 0;
- near-copies are measured on positions held over time;
- the orchestrator doesn't import `perp_lab.judge`;
- the harness is pinned for a session;
- proposer promotions are examined only by an explicit `lab exam --id`;
- the CLI version and the prompt size are checked.

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
- A trial is judged by harness code the human didn't mean to run, such as a feature branch checked
  out mid-session.
- A plugin reaches a holdout exam that no human chose.
- A long bias riding the 2020–26 drift gets read as an edge. Buy and hold had a Sharpe of 0.72
  over the OOS span, and 1.20 over its first two years.
- A week of runs leaves a record the user can't read.

## Requirements this design must meet

From the 3a spec ("Requirements carried into the Phase 3b proposer spec") and the 3b-1 spec
("Requirements carried into the Phase 3b-2 proposer spec"):

| requirement | how it is met |
|---|---|
| model training cutoff before `holdout_start` (2026-07-01) | config guard on a pinned model ID, checked against the model each call reports; the loop refuses to start otherwise (Model) |
| no read access to `runs/` except `runs/verdicts/`, nor to `lab.db` or the holdout log | the model reads nothing; the orchestrator reads only verdict contents and its own state (import-boundary and path tests) |
| never imports `perp_lab.judge` or `perp_lab.holdout` | AST test on `perp_lab/proposer/`. The Deflated Sharpe maths it needs for the bar moves to `perp_lab/dsr.py`, which `judge.gates` imports |
| submits only by writing a plugin to `strategies/` and running `lab judge` as a subprocess | the orchestrator does exactly that; the model returns text only |
| writes only to `strategies/`, never to `perp_lab/`, `config/`, `tests/`, `.venv/` | the orchestrator writes new files in `strategies/` and its own state dir, nothing else (test) |
| no read access to `data/`, and can't `stat()` it | the model's process is `claude -p` with no tools, started in an empty directory outside the repo, and killed at once if its first message shows any tool but `StructuredOutput`; the orchestrator never touches `data/` (path guard in tests) |
| sees only `lab judge`'s exit code and the verdict file, never stdout or stderr | judge output goes to a human-only log that no prompt reads |
| learns of exam failures only as reason codes; no file timestamps, nothing that dates an exam | prompts render whitelisted verdict fields, no times; exam outcomes enter only at session start |
| never runs `lab exam` or `lab withdraw` | no code path; the subprocess argv allowlist is `lab judge` and `lab preflight`. Its promotions are examined only by a human's `lab exam --id` (Exams and the human) |
| plugins keep `PARENT`, `params`, `grid` and `timeframes` literal | AST check before preflight |
| assume it knows exactly how the lookahead check works (R1) | accepted, bounded by G3 (it can't pass an exam it peeks in); see Residuals |

## Decisions

| question | decision |
|---|---|
| Shape ★ | A trusted orchestrator, `lab propose`, in `perp_lab/proposer/`. The model is called **with no tools**: text in, one typed proposal out. It reads no file and runs nothing. |
| Model access | Claude Code headless (`claude -p`) on the user's own login, the only model access the user has. No API key, no SDK, no new pip dependency. |
| Model | `claude-opus-5-5`, pinned by full ID. Its published training-data cutoff is June 2026 (Anthropic's models page, read 2026-10-19), so config sets 2026-06-30. The loop refuses to start unless the cutoff is before `holdout_start`, and refuses any call that reports another model. |
| Autonomy | Unattended. Exams, withdrawals and paper approval stay human. |
| Memory | Each proposal starts from a fresh context holding the record. The record is the model's only memory, so it can be audited. |
| What a trial is ★ | Counted at the start: a (plugin version, timeframe) becomes a trial the moment its plugin code first runs on real research bars, however the run ends. |
| Feedback | The verdict gains a `diagnostics` block computed from the walk-forward run only. |
| Luck check before costs | The random-entry benchmark also runs before costs. "Signal before costs" means a gross OOS Sharpe above both 0 and the gross random-entry p95. |
| Random-entry sides ★ | Each random entry's side is drawn with the plugin's long share, not 50/50 (issue #17). This applies to both benchmarks, so it changes the existing gate too: stricter for a plugin that leans with the drift, easier for one that leans against it. |
| Ideas and timeframes ★ | An idea has one timeframe. A refinement's `PARENT` is the idea's champion. The same idea on another timeframe is a *port*: a new idea and a new family, allowed only from an idea whose best pile is signal before costs or close call. |
| Near-copies ★ | Measured by the harness, never declared by the model, on positions held over time rather than exact entry bars. Before any trial, positions on synthetic bars are compared with every earlier plugin's. After the trial, the verdict reports position overlap and daily-return correlation with earlier OOS runs, and the record acts on both. A mirror (the same positions, opposite side) of any judged plugin is refused. |
| Preflight | `lab preflight FILE --tf TF`: metadata checks, the lookahead check and the overlap check, all on synthetic bars. No real data, nothing recorded, no trial. Its full report may go to the model. |
| Pace ★ | 12 trials a day, 100 trials in total for the proposer, 4 judged plugins per idea, one judge at a time. |
| Exams ★ | A judge and an exam never run at once: a harness lock, in both directions, that can't starve exams. A proposer promotion is examined only when a human names it, `lab exam --id N`; a bare `lab exam`, for example from a scheduled job, skips it. Exam outcomes reach the model only as status and code, only at session start. |
| Harness pin ★ | A session runs on one harness version. It refuses to start while `perp_lab/` or `config/` has uncommitted changes, records the commit, and stops if either changes during the session. |
| Where the user reads it | A new dashboard page, Ideas, plus a markdown digest per session. |
| State | Its own SQLite file and folders under `proposer/` (git-ignored), never `lab.db`. |

## Shape

```
lab propose  (trusted harness code, perp_lab/proposer/)
  1. builds the prompt from: static harness context + the record (verdict contents + idea cards)
  2. calls the model (claude -p, no tools)  ->  one proposal: an idea card + plugin source (JSON)
  3. checks it in code: schema, idea rules, AST           -- fails -> back to the model (bounded)
  4. lab preflight <staged file> --tf TF   (sandbox, synthetic bars only)
                                                         -- fails -> back to the model (bounded)
  5. copies the plugin into strategies/
  6. lab judge <name> --tf TF --proposer   (subprocess; stdout and stderr to a human-only log)
  7. reads the verdict file's contents
  8. classifies the outcome in code, updates the record, writes the digest line; next round
```

Why this shape:
- The requirements forbid the proposer to read `data/`, `runs/` or `lab.db`, yet `lab judge` must
  read them. An agent with a shell would need a confinement that denies those paths. Landlock
  restrictions are inherited by child processes, so that confinement would also stop the
  `lab judge` it starts. Working around it means a broker process. And the model would gain
  nothing it needs, since it may not read data or run backtests anyway.
- With no tools, the model can't break any of these rules. Its process is a `claude -p` that the
  orchestrator starts, feeds and reads. That process doesn't start `lab judge`, so nothing stops
  it being confined later (Out of scope). The orchestrator is ordinary harness code, tested like
  the judge, and every rule is a check in it or in the harness. This follows
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
   Invalid → send the errors back. Every call is a fresh `claude -p`, so a fix-up resends the
   round's prompt plus the rejected proposal and the errors. At most `fixups_per_proposal` (3)
   fix-ups per proposal, shared with steps 5 and 6. After that the proposal is dropped: no trial,
   recorded as dropped.
4. **Idea rules**, in code:
   - A new idea needs a complete card and a title not already in the record.
   - A refinement names an existing idea that is open (see The record) and has judged plugins left
     (`proposals_per_idea`, 4). It keeps the idea's timeframe.
   - A port names an idea that is open or exhausted, whose best pile is signal before costs or
     close call, and a timeframe that idea has no port on yet. At most one port per idea and
     timeframe. A promoted idea gets no port: its family is headed for an exam, and a port would
     be a second family, with a second holdout read, for the same mechanism.
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
8. **Judge**: first the harness pin is checked (see Harness pin). If the UTC day has changed since
   preflight, preflight runs again, so its lookahead span is the judge's. Then
   `lab judge <name> --tf TF --proposer`, with stdout and stderr to `proposer/logs/<name>.log`.
   Every call counts toward the day's quota (Pace), whatever its exit code:
   - 0: judged; go on to step 9.
   - 2: refused. Recorded as `broken`; preflight should have caught it.
   - 1: infrastructure error. Recorded as `broken`. If plugin code had reached real bars, the
     judge registered the trial first (Trials), so N already counts it.
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
   - the gates, what a trial is, and the Deflated Sharpe bar table (Pace), computed with
     `perp_lab.dsr` when the prompt is built;
   - the piles and the idea rules;
   - the human's lessons from the baselines: a checked-in excerpt of section 1 of
     `docs/research/2026-10-18-strategy-research.md` at `perp_lab/proposer/context/lessons.md`,
     labelled as notes on 8 coins from 2022. It keeps section 1's table and lessons 1–5 and drops
     lesson 6 and any references to other sections. Lesson 6's bar (1.52 at N = 7) and
     buy-and-hold figure are stale; this spec's table replaces the bar.
   - the output contract.

   Section 2 (the cost hurdle) is left out. Its fill costs predate per-coin slippage, and the
   cost model above replaces them. Its tables of move sizes and funding are statistics of real
   bars computed outside any trial, which anti-goal 2 rules out.

   The static context is built in code and checked when it is built. A test pins its size below
   100 KB, because it goes on the command line (`--system-prompt`) and Linux refuses a single
   argument over 128 KiB. Its sources come to about 24 KB today.
2. **The record**, rendered by code. Open ideas appear in full: card, each proposal's whitelisted
   verdict fields and diagnostics, and the champion's source, so a refinement can be a real change
   rather than a rewrite from the card. Closed ideas get one line each: title, signal, and why
   closed (for example "dead: gross Sharpe −0.9 against the gross random p95 0.1").
3. **The instruction for this round**: aim where nothing is ruled out; refinements of ideas in the
   signal-before-costs and close-call piles are welcome; return one proposal.

Whitelisted verdict fields: decision, reasons, OOS metrics, Deflated Sharpe, N, sr0, beats-parent
fraction, holdout status and code, diagnostics. Run ids and report paths are left out: run ids
encode a time. The record numbers proposals in order and dates nothing. For a verdict whose
diagnostics come from a refresh on today's data (`diagnostics.data`), the prompt shows the refreshed
numbers and gives the old judgement's decision only, so the model never sees two different OOS
Sharpes for one plugin.

Today's date is not hidden. It says nothing about when an exam ran beyond what the session-start
snapshot already allows, and preflight reports carry synthetic-bar dates anyway. What must never
reach the model is the time of a judge run, an exam, or a file.

## What the model returns

One JSON object, from a dataclass in `perp_lab/proposer/contract.py`. The schema passed to
`claude -p --json-schema`, the prompt's description of it and the validation are all generated
from that dataclass. The CLI validates the reply against the schema, and the orchestrator validates
it again.

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
3. **Overlap**, measured on positions held, not on entry bars. A one-bar delay or a small
   parameter change shifts entries by a bar or two, which drops an exact-bar match close to 0 while
   the trades stay nearly the same.
   - **The bars.** A fixed synthetic market: one 15m path per synthetic symbol from
     `synthetic_bars` (fixed seeds and start date, never the day-pinned holdout span), resampled to
     1h and 4h. So plugins at every timeframe trade the same market.
   - **The run.** Each plugin runs at its defaults, in the sandbox (`client.fresh_run`), at the
     timeframe in question. Earlier plugins run at each timeframe they declare.
   - **The measure.** Its trades become positions on an hourly grid: at each hour's close, per
     symbol, +1, −1 or 0. Overlap is the Jaccard index of (symbol, hour, side) over the hours
     either plugin holds a position. Mirror overlap is the same with one plugin's side flipped.
     Near-copy means overlap ≥ `overlap_max` (0.5); a mirror means mirror overlap
     ≥ `overlap_max`.
   - **The cache.** Earlier plugins' positions are cached under `proposer/cache/`, keyed by
     (plugin hash, timeframe, synthetic-market version).

   The rules:
   - A near-copy of a closed idea's plugin is refused, whatever the proposal's kind.
   - A near-copy of an open idea's plugin, proposed as a new idea, is refused with "propose it as a
     refinement (or a port) of X".
   - A refinement is exempt from overlap with its own idea, and a port with the idea it ports.
     Neither is exempt from any other idea.
   - A mirror of any judged plugin is refused. A mirror's OOS result is close to minus the
     original's, which has already been seen, so its walk-forward would test nothing. This is the
     research note's exclusion of candidate E (sections 4 and 10), applied to every idea.
   Synthetic bars are a proxy: their taker, open-interest and volume columns are random walks, so
   two plugins can differ there and still coincide on real data. The after-trial overlap (The
   record) catches those, on real OOS positions and daily returns.
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
| signal before costs | not the above, and the gross OOS Sharpe is above both 0 and the gross random-entry p95 |
| no signal | none of the above, with at least the gate's minimum OOS trades |
| broken | killed for lookahead or a plugin error, refused, an infrastructure error, or too few OOS trades |

A promotion the human withdraws (`holdout_status` `withdrawn`) is filed as a close call. The human
judged the effect not established.

The floor of 0 matters because random entries are side-matched (Random entry). Over the
up-drifting 2020–26 span, a short-leaning plugin's random entries lose money, so their gross p95
can fall below 0, and a plugin that lost money before costs would otherwise land in a progress
pile. Close calls already require a net Sharpe above 0.

A judged proposal can also be filed as a `duplicate` (After-trial overlap). Its pile then doesn't
count toward its idea's best pile.

**The champion** of an idea is the plugin a refinement names as `PARENT`. It is chosen among the
idea's plugins that have a walk-forward run (never a broken one, since the judge refuses a child
whose parent can't be run):
- first, the idea's most recent promotion whose exam is pending, so a refinement must beat it, and
  supersedes its queued exam if promoted. "Most recent" lets the record follow a supersede within a
  session without re-reading the parent's verdict, which could reveal an exam's timing;
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
  reach paper. A passed idea is the edge the user is after; the human takes it from there. This
  status is kept per family (`lineage_id`), not per idea. When a family's read is spent, every
  idea with a judged plugin in that family closes with the outcome. The seeded ports share a
  family with the ideas they port (Seed).
- `duplicate`: a new idea whose only judged proposals were filed as duplicates (below).

**After-trial overlap.** Every judged proposal, whatever its kind, is checked against its verdict's
overlap block (Verdict diagnostics). The verdict names runs by strategy and timeframe, which the
record maps to ideas. The same exemptions apply as in preflight: a refinement against its own idea,
a port against the idea it ports.
- **A near-copy of a closed idea.** It is one when, against any run of that idea, either:
  - its OOS position overlap is ≥ `overlap_max` (0.5); or
  - its OOS daily-return correlation is ≥ `corr_max` (0.8).

  The proposal is filed as a `duplicate` of that idea, and its pile doesn't count toward its own
  idea's best pile. There is one exception: if the proposal's pile is better than the closed idea's
  best pile, the overlap wasn't the whole story. The proposal then keeps its pile, and the overlap
  is noted on its card.
- **A mirror of any judged run.** It is one when its mirror overlap is ≥ `overlap_max`, or its
  correlation is ≤ −`corr_max`. It is filed as a `duplicate` whatever its pile, since a sign
  flip chosen after seeing the original's OOS result tests nothing.

A new idea whose only judged proposals are duplicates gets the status `duplicate`. Its trials stay
spent, and the closed idea stays closed.

**Global holdout budget.** When any verdict shows `blocked` / `budget_spent`, no promotion can
reach an exam any more. `lab propose` then refuses to start and says so.

**One idea is one family at one timeframe**, once it has a champion, for every idea the proposer
makes: a refinement joins its champion's family, and a port is a new root. The seeded ports are
the exception. The judge keys a family by plugin name (`db.family_of`), so a plugin judged at two
timeframes is one family. That makes `family:donchian_atr` hold both Donchian ideas (and
`donchian_atr_wide`), and `family:bb_vwap_reversion` both BB-VWAP ideas. Hence the per-family
exam statuses above.

**Seed** ★: the record starts with six ideas from the seven judged human trials, which cover five
plugin files in four families:
- `donchian_atr` at 4h, with `donchian_atr_wide` as its refinement;
- `donchian_atr` at 1h, as a port of the 4h idea;
- `bb_vwap_reversion` at 15m;
- `bb_vwap_reversion` at 1h, as a port of the 15m idea;
- `funding_fade` at 1h;
- `flow_shock_fade` at 1h.

The two seeded ports only record how the human trials relate. The port rule (signal before costs
or close call first) doesn't apply to them. Each shares its family with the idea it ports, because
it is the same plugin file.

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
- **Overlap:** against the latest walk-forward run of every earlier judged (plugin version,
  timeframe), at any timeframe, three lists:
  - the three with the most OOS position overlap. This is preflight's measure, on real OOS
    positions: the Jaccard index of (symbol, hour, side) held, on an hourly grid built from
    `oos_trades`;
  - the three with the most mirror overlap;
  - the three with the largest absolute OOS daily-return correlation, given with its sign.

  Each entry names the run's strategy and timeframe and gives the value.

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

## Trials: counted at the start ★

Today a trial is a recorded walk-forward run: `trial_stats` counts the distinct (plugin version,
timeframe) pairs among them. A run that ends in a plugin error records a kill but no run, and one
that ends in a harness exception (`lab judge` exits 1) records nothing. Either way plugin code
saw real research bars and N didn't move. A plugin could then fail on purpose depending on what it
saw. That is a free, uncounted look at the OOS data, and the proposer could submit only what it
had already checked.

The rule: a (plugin version, timeframe) becomes a trial the moment its plugin code is about to run
on real research bars, before the walk-forward's first batch.
- `validate_strategy` writes a row to a new `trials` table: strategy, plugin hash, timeframe,
  `registered_at`. It does this before the walk-forward starts, unless that pair is already a
  trial.
- `trial_stats` counts the distinct pairs in `trials` together with those among walk-forward runs.
  Runs made before this change have no row.
- So a plugin error, a sandbox death, a timeout and a harness crash after that point all count,
  and so does a run interrupted by Ctrl-C.
- Re-judging the same bytes at the same timeframe adds nothing, because N counts distinct pairs.
- A lookahead kill stays uncounted: that check runs on synthetic bars only, before anything is
  registered.
- A parent re-run and a `lab verdict-refresh` re-run an existing pair, so they add nothing either.

Issue #15 is still built first, for a different reason. Without it, a transient sandbox failure
(`died`, `timeout`, `cpu_limit` caused by the machine) records a permanent plugin-error kill. N is
right either way, since the trial was registered. But the record would file an honest plugin as
broken, and its idea would lose one of its 4 plugins. The builder settles the retry rule within
#15.

## Random entry: before costs, and side-matched ★

- **Side-matched (#17).** Each random entry's side is drawn with the plugin's overall OOS long
  share, not 50/50 (today `RandomEntry` draws `u < 0.5`, `perp_lab/strategy.py`). Without this, a
  long-biased plugin beats random entries on drift alone, and an optimising proposer will find
  that. The change applies to the existing net gate too, in both directions:
  - over the up-drifting 2020–26 span it makes the gate stricter for a plugin that leans long;
  - it makes the gate easier for a plugin that leans short, since its random entries lean short
    too and lose to the drift.

  That is the like-for-like comparison intended. The absolute floors keep a losing short-leaning
  plugin out of the progress piles: a net Sharpe above 0 for the gate and for close calls, and a
  gross Sharpe above 0 for signal before costs.
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
  UTC. Every `lab judge` call counts toward it, whatever its exit code, so a plugin can't loop on
  errors.
- **`trials_total`: 100**, for the proposer's own trials. N would then be about 107 and the bar
  about 1.72. When it is reached, `lab propose` refuses to start. Raising it is the human's
  decision. The orchestrator can't read the `trials` table, so it counts conservatively: every
  judge call except a refusal (exit 2) and a lookahead kill. That is never less than the
  proposer's share of N.
- **`proposals_per_idea`: 4** judged plugins.
- **`fixups_per_proposal`: 3.**
- **Token caps per day**, input and output, from the CLI's reported `usage`. They keep the loop
  from eating the user's whole Claude Code allowance.
- One session at a time (a lock file in `proposer/`), and one judge at a time.

Why pace at all: the total cap is what protects the bar. Pace spreads the trials so the user can
read a day's record and stop the loop before the budget is gone. At 12 a day, a week is 84 trials.

Stop conditions:
- a cap;
- 3 infrastructure errors in a row;
- 5 proposals in a row dropped at the schema or preflight stage (the model is stuck);
- the model call failing after retries (see Failures);
- the harness pin changing (`harness_changed`, see Harness pin);
- SIGINT or SIGTERM. The first one lets a running judge finish and then stops. The second kills it
  and records the proposal as `broken` / `interrupted`; it counts toward the day's quota.

Rough size:
- **Time.** A 1h judgement took about 2 minutes, and a model call takes a few minutes. So 12
  trials take a few hours of a day.
- **Static context.** With `--system-prompt`, Claude Code's own prompt is replaced: the probe's
  whole cached prefix was about 2,500 tokens. The static context's sources come to about 24 KB,
  so it is about 10–15k tokens per call, cached.
- **The record** adds about 5k tokens per open idea (card, verdicts, diagnostics, the champion's
  source) and about 60 per closed idea.
- **Output** is 5–15k tokens with thinking.
- **A day.** At about 1.5 calls per trial, 12 trials are about 18 calls. With ten open ideas that
  is roughly 1M input tokens (mostly cached) and 0.2M output.

That share of a Claude Code plan's limits is a guess until measured. The first session's digest
gives the real figure, and `trials_per_day` is the knob.

## Exams and the human

- **Unchanged:** only a human or a scheduled job runs `lab exam`; only a human runs `lab withdraw`
  or approves paper. `perp_lab/proposer/` has no code path to any of them (import and argv tests).
- **A proposer promotion is examined only when a human names it.** This is a harness change, in
  code because a minted read can't be taken back (`lab withdraw` works only before the mint):
  - `lab judge --proposer` records `judged_by = "proposer"` on the judgement. The orchestrator
    always passes the flag; a human never needs it.
  - A bare `lab exam`, as a scheduled job would run it, skips those judgements. It prints each
    one as "awaiting review: `lab exam --id N`" and leaves it pending, spending nothing.
  - `lab exam --id N` examines that one judgement, under today's budget rules. That is the human's
    approval, given after reading the review packet.
  - A refinement that supersedes a reviewed promotion has its own id, so the human always examines
    exactly what they reviewed.
  - The status and code in the verdict don't change (`pending` / `awaiting_exam`), so the model
    learns nothing new.
- **A judge and an exam never run at once.** This is a harness change. `lab judge`, `lab
  walkforward`, `lab backtest` and `lab preflight` hold a shared lock for their whole run, and
  `attempt_exam` takes it exclusively. Exams get writer preference, so a stream of judges can't
  starve them: for example, a turnstile lock that `lab exam` holds while it waits. This closes R6's
  overlap between a lookahead check and an exam for every caller, not just the loop. It is the
  "pause the proposer during exams" follow-up.
- **What the model learns of exams:** `holdout_status` and its code, only from the session-start
  snapshot, and nothing with a time attached.
- **Review before an exam:** for each promoted plugin, the Ideas page shows the criterion-6 checks
  from its diagnostics (sides, top coin and top fold share), and the exact `lab exam --id N`
  command. The human either runs it or withdraws the promotion.
- **Holdout reads:** 10 in total, one per family. Each new idea is a new family, so the loop can
  make many. What reaches an exam is the human's choice, through `lab exam --id` and
  `lab withdraw`.

## Harness pin ★

The loop runs `lab judge` from the main checkout, where the editable install points. If the human
checks out a branch or edits `perp_lab/` or `config/` during a week-long session, later trials
would be judged by code or settings nobody chose for them, and the record wouldn't show it.
- **At session start** `lab propose` refuses to start if `git status --porcelain -- perp_lab config`
  shows any change. It records the commit (`git rev-parse HEAD`), the branch, and a hash of the
  tracked files under `perp_lab/` and `config/`. The digest shows them, with a warning when the
  branch isn't `main`.
- **Before every preflight and judge** it checks the commit and the hash again. Any difference
  stops the session with reason `harness_changed`. A judge already running finishes first.
- **New plugins in `strategies/`** are untracked files outside both paths, so they never trip the
  check.

## Model

**Which model.**
- `claude-opus-5-5`. Anthropic's models overview (read 2026-10-19) gives Opus 5.5 a reliable
  knowledge cutoff and a training-data cutoff of "Jun 2026". So `model_training_cutoff` is
  2026-06-30, one day before `holdout_start`. The rule is met with no margin: the model may have
  seen data up to the eve of the holdout, but none inside it.
- Every current Claude model has the same June 2026 cutoff. Any later model will fail the guard,
  so the proposer stays on Opus 5.5. Anthropic commits to keeping it until at least 2027-09-22.
- Config names the model by its full pinned ID, never an alias. The alias `opus` would move to a
  newer model with a later cutoff.
- The cutoff is entered by the human from the published documentation; the harness can't verify
  it. The loop refuses to start when it is missing or not before `holdout_start`. Each session and
  proposal records the model and the cutoff.

**What the rule doesn't fix.** Any model that meets it still knows the market history of the whole
OOS span, 2020–2026. So the walk-forward is not out of sample for the proposer, and a promotion is
a hypothesis until its holdout exam. That is why the cutoff rule exists and why exams stay scarce.

**How it is called.** One `claude -p` subprocess per model call, on the user's Claude Code login.
The user has no API key and no other model access. The command line:

```
claude -p --model claude-opus-5-5 --effort <proposer.effort>
       --tools "" --strict-mcp-config --safe-mode --disable-slash-commands
       --no-session-persistence --permission-prompts none
       --system-prompt <the static harness context>
       --json-schema <the contract's schema>
       --output-format stream-json --verbose --include-hook-events
```

- **Input:** the record and the round's instruction go in on stdin.
- **Working directory:** a fresh empty directory outside the repo, so no `CLAUDE.md`, project
  memory or git state is picked up.
- **Environment:** a minimal one with what the CLI needs for the login (`HOME`, `PATH`), and none
  of the user's other variables.
- **What each flag does:**
  - `--tools ""` removes every built-in tool;
  - `--strict-mcp-config` with no `--mcp-config` loads no MCP server;
  - `--safe-mode` drops CLAUDE.md, skills, plugins and hooks but keeps the login, unlike `--bare`,
    which accepts only an API key;
  - `--system-prompt` replaces Claude Code's own system prompt with the static context.
- **Probe, 2026-10-19,** on this machine with CLI 2.1.303: the first stream message (`system` /
  `init`) reported model `claude-opus-5-5`, tools `["StructuredOutput"]` (the `--json-schema`
  mechanism), no MCP servers, no skills and no slash commands. The reply arrived as validated JSON
  in `structured_output`, with `usage` and a `rate_limit_event` in the stream.
- **The CLI updates itself.** It was 2.1.306 two days after the probe. Each session records
  `claude --version` at start, and the digest shows it. The init check below is what holds
  across versions.

**Checked in code, not trusted to flags** (ADR 0006). The orchestrator reads the stream line by
line. It kills the process the moment the `init` message shows any of these:
- a model other than the configured one;
- any tool but `StructuredOutput`;
- any MCP server, skill or slash command;
- or any hook event later in the stream.

The init message comes before the model's first turn, so a misconfigured call is stopped before
the model can act. Its output is discarded, and the session stops with reason `model_config`.
Every session also starts with a one-line probe call that must pass the same check.

**Usage limits.** The calls run on the user's Claude Code plan and share its usage limits with the
user's own sessions.
- When the stream reports a limit, the loop sleeps until the reset time it gives, or backs off
  hourly if none is given. It doesn't stop. The digest notes the pause.
- The result's `usage` feeds the daily token caps.
- Its `total_cost_usd` is recorded as the API-equivalent cost; on a subscription it isn't billed.

**No credentials** pass through the orchestrator: the CLI holds the login. Stream output is stored
in the transcripts as is. A test checks it carries no token-like strings. `lab judge` and
`lab preflight` run with the user's normal environment, as they do today.

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
  model: claude-opus-5-5             # full pinned ID, never an alias
  model_training_cutoff: 2026-06-30  # Anthropic models page, "Jun 2026" (read 2026-10-19); < holdout_start
  claude_bin: claude                 # the Claude Code CLI, logged in as the user
  effort: high
  state_dir: proposer                # record.db, staging/, logs/, transcripts/, sessions/, cache/
  timeframes: [15m, 1h, 4h]
  trials_per_day: 12
  trials_total: 100
  proposals_per_idea: 4
  fixups_per_proposal: 3
  max_grid_combos: 16
  overlap_max: 0.5                   # position overlap (Jaccard of symbol, hour, side)
  corr_max: 0.8                      # OOS daily-return correlation, after the trial
  max_input_tokens_per_day: 4000000
  max_output_tokens_per_day: 500000
```

`proposer/` is added to `.gitignore`.

## CLI

- `lab propose [--days 7] [--trials N]`: run a session until D days, N trials or a stop condition.
- `lab ideas`: print the record by pile.
- `lab preflight FILE --tf TF`: also useful to humans.
- `lab verdict-refresh STRATEGY --tf TF`: recompute an existing judgement's diagnostics.
- `lab judge ... --proposer`: marks the judgement as the proposer's (Exams and the human).
- `lab exam --id N`: examines one judgement. A bare `lab exam` skips the proposer's.

## Failures

- **Model-call errors:**
  - A usage limit sleeps until it resets (Model).
  - Other failures (the CLI exits non-zero, times out, or returns `is_error`) are retried with
    backoff, 5 tries. Then the session stops (reason `model_call`).
  - A failed init check stops the session at once (reason `model_config`).
  - If the CLI is missing or logged out, the session refuses to start.
- **Invalid model output:** fix-ups, then the proposal is dropped (no trial).
- **Preflight infrastructure error** (exit 1): the session stops after 3 in a row.
- **Record or state-dir errors:** fail closed; the session stops.
- **Orchestrator crash:** on the next start, a proposal whose plugin is in `strategies/` but whose
  verdict was never read gets `lab judge` again. If a judgement exists, this only rewrites the
  verdict. If not, the same bytes are judged now, and N doesn't move again since the pair is
  already registered. The model never saw the lost run, so it gains nothing.

## Residuals

- **Exit-code channel.** Mostly closed by counting trials at the start. A plugin that makes
  `lab judge` exit 1 after reaching real bars still pays a trial. What is left is one bit (exit 1
  or a verdict) on a trial that is already counted. The day's quota and the stop rule (3 in a
  row) bound how often that can happen.
- **R1 from 3b-1** (behaving differently under test) matters more with an optimising author. It is
  bounded by G3: such a plugin can't pass its exam. The 3b-1 follow-up "narrowing R1" (a streamed
  re-run over a secret stretch of real OOS data) is recommended before the first exam of a
  proposer-written plugin.
- **The model knows the OOS history** (Model). The holdout is the only clean test.
- **Admin-managed policy settings still apply** to a `--safe-mode` run. If the user's organisation
  injects instructions, they reach the proposer's context. They come from the user's own account,
  not from the lab, so they leak nothing. They do add tokens and may add instructions that don't
  fit the task. The transcripts show exactly what was sent.
- **The CLI is trusted code.** It reads its own config and login under `~`, and with
  `--safe-mode` it skips the repo's files. The model can't direct it to read anything, because it
  has no tools.

## Testing

Each test is written first and seen failing. No test calls the network or uses the user's
allowance. The orchestrator's tests put a fake `claude` executable first on `PATH`
(`proposer.claude_bin`). It records its argv, stdin and environment, and prints canned stream-json.

The orchestrator:
- **Prompt sources:** sentinels planted in `lab.db`, a run's `report.md`, the holdout log, `data/`
  and the judge's stdout never appear in anything sent to the model (the fake CLI's stdin and
  argv), over a full round.
- **No times:** distinctive times planted in verdict files' modification times, run ids, the
  holdout log and the judge's output never appear in anything sent to the model.
- **Boundaries:**
  - `perp_lab/proposer/` imports none of `perp_lab.judge`, `perp_lab.holdout`, `perp_lab.store` or
    `perp_lab.data` (an AST scan anchored on the repo root). It may import `perp_lab.dsr`.
  - Subprocesses are only `lab judge` (always with `--proposer`), `lab preflight`, `claude` and
    read-only `git`.
  - Files are written only under `state_dir`, plus new files in `strategies/`; nothing is
    overwritten.
- **Static context:** under 100 KB. It holds no lesson 6 and nothing from section 2 of the
  research note.
- **Harness pin:** a dirty `perp_lab/` or `config/` refuses the start. A commit, or an edit to a
  tracked file under either path, between rounds stops the session as `harness_changed`. A new
  file in `strategies/` doesn't.
- **Idea rules:** a closed idea refused; a refinement gets its idea's champion as `PARENT`. The
  champion is a pending promotion first, never a broken plugin; with no champion the plugin is a
  root. A `PARENT` written by the model refused; a port only from a signal-before-costs or
  close-call idea, once per timeframe, always a root; the one-timeframe rule, the grid cap and the
  import allowlist; the fix-up bound.
- **Piles and statuses:** a table of verdict fixtures with their expected piles. Among them:
  - trial 7's verdict, which is no signal;
  - a withdrawn promotion, which is a close call;
  - a gross Sharpe of −0.2 above a gross random p95 of −0.3, which is no signal.

  Status rules:
  - an exam passed or failed closes every idea in that family, including a seeded port that
    shares it;
  - an after-trial overlap or correlation match files the proposal as a duplicate, unless its pile
    beats the closed idea's;
  - a mirror is filed as a duplicate whatever its pile;
  - a new idea with only duplicates is `duplicate`;
  - `budget_spent` refuses the next start.
- **Champion:** with two pending promotions in an idea, the most recent one.
- **Session:** the daily quota sleeps, and every exit code counts toward it. The total cap refuses
  to start, counting exit 0 and 1 but not refusals or lookahead kills. Also tested: the cutoff
  guard refuses; SIGINT once and twice; 3 infrastructure errors stop; a fix-up resends the prompt
  with the rejected proposal and errors; a UTC day change between preflight and judge re-runs
  preflight.
- **The model call:**
  - the argv carries every flag in Model, the pinned model ID and the contract's schema;
  - the working directory is empty and outside the repo;
  - the environment is the minimal one;
  - an init message with another model, any tool but `StructuredOutput`, any MCP server, skill or
    slash command, or a later hook event kills the process before any output is used, and stops
    the session;
  - a usage-limit message sleeps until its reset;
  - the alias `opus` in config refuses to start;
  - the session records `claude --version`.

The harness:
- **Preflight:** never calls `load_bars` and never opens `data/`. A lookahead plugin fails
  preflight and the judge alike. The dated span is the same at any hour of one UTC day. Overlap
  checks:
  - it catches a byte-different copy, a copy whose entries are one bar late, a small parameter
    change and a mirror;
  - a 1h plugin and its 4h twin overlap on the shared synthetic market;
  - a refinement is exempt from its own idea only, and a port from the idea it ports only;
  - the cache key changes with the timeframe and the synthetic-market version.
- **Verdict v2:** diagnostics present and computed from the walk-forward run, and still nothing
  derived from holdout data. A plugin-error message is absent from the verdict and kept in
  `lab.db`. A builtin exception type is kept; a plugin-defined one becomes `other`.
- **Verdict refresh:** a refresh on today's data changes no decision field, adds no trial, and
  marks `diagnostics.data`.
- **Trials:** each of these after real bars raises N by one:
  - a plugin-error kill;
  - a harness exception;
  - an interrupted run.

  Judging the same bytes again doesn't raise it, and a lookahead kill doesn't. Runs from before
  the change still count once each.
- **Random entry:** side-matched draws follow the long share; the gross variant has zero fees and
  slippage.
- **Verdict overlap:** position overlap and correlation are computed across timeframes from
  `oos_trades` and `oos_daily`.
- **Proposer exams:** a bare `lab exam` leaves a `--proposer` judgement pending and spends nothing.
  `lab exam --id N` examines it. A human judgement is examined by a bare `lab exam` as today.
- **DSR module:** `perp_lab.dsr` gives the same values as before the move. The bar table in this
  spec is a test fixture: 1.24 at N = 7, 1.72 at N = 107, 2.00 at N = 1,000.
- **Lock:** no exam starts while a judge runs, and no judge while an exam runs. A stream of judges
  doesn't starve a waiting exam.

End to end, on small synthetic data with a temp config and store: a fake model proposes
1. a broken plugin, then its fix;
2. a near-copy of a dead plugin, refused at preflight;
3. a sound plugin, which is judged.

The record shows the right piles and the digest is written.

## Suggested build order

1. Harness changes, each its own commit with its tests:
   - `perp_lab/dsr.py` (the Deflated Sharpe maths moved out of `judge.gates`);
   - #15 (transient failures);
   - trials registered at the start, and the verdict message rule;
   - random entry side-matched and before costs;
   - verdict diagnostics, with position overlap and correlation, and `lab verdict-refresh`;
   - `lab judge --proposer` and `lab exam --id`;
   - the judge–exam lock;
   - the dated lookahead span pinned to the day;
   - `lab preflight`, with the shared synthetic market.
2. The orchestrator:
   - the contract;
   - the record store;
   - the static context and prompt builder;
   - the harness pin;
   - the `claude -p` caller with its init check;
   - the loop;
   - the CLI.
3. The Ideas page and the digest.
4. Seed the record and refresh the seven trials' diagnostics.

## Changes to existing documents and tests

- `2026-10-17-plugin-sandbox-design.md`, Failures, walk-forward plugin errors: the sentence "the
  message may reach the verdict ... No run is recorded, so it is not a trial" is replaced by this
  spec's two rules. The message stays out of the verdict, the exception type is kept only if
  builtin, and the trial was registered before the run began.
- `2026-10-16-judge-design.md`:
  - Who sees what: the public verdict also carries the `diagnostics` block, computed from
    walk-forward data only.
  - The trial definition: N counts registered trials as well as walk-forward runs.
  - Exams: a bare `lab exam` skips proposer judgements, and `lab exam --id` examines one.
- `tests/test_judge.py`: `test_a_plugin_error_in_the_walk_forward_is_a_recorded_kill` asserts the
  message in the reasons. It changes to assert the verdict's reduced reason, the full reason in
  `lab.db`, and N raised by one. Its neighbour for a child's plugin error gets the same N
  assertion.
- `docs/research/2026-10-18-strategy-research.md` is unchanged. The proposer's excerpt of
  section 1 is a separate checked-in file.

## Out of scope and follow-ups

- Cheaper trials, both deferred judge changes: counting near-duplicates as fewer effective trials
  (harness-measured), and screening many ideas then confirming a few.
- Narrowing R1, before the first exam of a proposer plugin (Residuals).
- A review mode, where the human approves each idea before its trial.
- The change-7 spec (the exam rule for coins that stop trading), still due before the first exam,
  about 2026-12-28.
- Confining the `claude -p` process itself with Landlock: read access only to its install and
  `~/.claude`, nothing in the repo. It is defense in depth on top of the init check. Unlike an
  agent that starts `lab judge`, this process starts nothing that needs the repo.
- Other model providers, should the user get access to one.
- Git: the loop never commits. The record keeps every plugin's bytes, and the human commits what
  they want.

## Calls made by the author, for review

★ in the text. The user approved all of them on 2026-10-21, with the review's changes folded in
(items 2, 3, 9 and 10 changed; 12 and 13 are new). The ones that most change what gets built:
1. **No tools for the model**, with a trusted orchestrator doing everything (Shape).
2. **A trial is counted at the start**, when plugin code first runs on real bars, however the run
   ends (Trials). #15 is still built first so that a machine hiccup doesn't file an honest plugin
   as broken.
3. **Random entry side-matched (#17)**, which changes the existing gate for every later trial, in
   both directions. "Signal before costs" also needs a gross Sharpe above 0. The user had
   deferred #17 before the proposer existed.
4. **Pace: 12 trials a day, 100 in total**, and 4 judged plugins per idea (Pace).
5. **Feedback granularity:** aggregates only, no per-month or per-coin series (Verdict
   diagnostics).
6. **"Before costs"** means zero fees and slippage with funding kept (Random entry).
7. **The seed:** six ideas from the seven trials, piled from fresh walk-forward runs on today's
   data (no new trial), plus candidates B, C and D, with E excluded (The record).
8. **The model runs through Claude Code headless on the user's login**, as `claude-opus-5-5`
   pinned. Its training-data cutoff was checked on Anthropic's models page (June 2026). The tools
   are removed by flags, and that is verified in code from each call's first message (Model).
9. **Ideas are (family, timeframe) pairs.** A refinement keeps the timeframe and names the
   champion as `PARENT`. The same idea on another timeframe is a port: a new family, allowed only
   from a signal-before-costs or close-call idea (Ideas and timeframes, The record). The seeded
   ports are the exception: they share their family, so exam statuses are kept per family.
10. **Near-copies and mirrors are measured on positions held over time**, before the trial on a
    shared synthetic market and after it on real OOS positions. After the trial, return
    correlation counts too. Mirrors of any judged plugin are refused, extending the research
    note's exclusion of E to every idea (Preflight, The record).
11. **Closed for good:** an idea whose family's holdout read was spent takes no more trials, and a
    withdrawn promotion counts as a close call (The record).
12. **Proposer promotions are examined only by `lab exam --id N`**; a bare or scheduled
    `lab exam` skips them (Exams and the human).
13. **The harness is pinned for a session**: a clean `perp_lab/` and `config/` to start, and a stop
    on any change (Harness pin).
