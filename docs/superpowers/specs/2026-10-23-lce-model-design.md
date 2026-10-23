# LCE model: Tradewriter's ES day-trading system, tested on the crypto book

Status: draft 2026-10-23, for the user's approval. Branch `feat/lce-model` (worktree `.worktrees/lce`).
Harness: option A (user, 2026-10-23). Land `feat/proposer`'s general harness commits on `main`
first, then build LCE on `main`.
Scope: two strategy plugins (`lce_pct`, `lce_atr`) that code the LCE model as taught. To support
them: 5m candles for the 17 coins, and a fix that keeps the random-entry benchmark to the hours
a plugin can trade. Then the two trials and a research note. The ES/SPX run is the next step,
with its own spec once the user has found ES data (see "ES leg").

## Why

The user is mining strategies that day traders sell on YouTube, Instagram and TikTok for
hypotheses the lab can test. The result of this work decides **one thing: whether LCE goes into
the lab's crypto book** (user, 2026-10-23). So the crypto trials are the ones that count. They
clear the same bar as every other plugin: `lab judge` walk-forward on the 17 coins, net of fees,
funding and slippage, against the random-entry benchmark, with N counted store-wide. Then
`lab exam` on the holdout, if it is promoted.

The ES run comes later and doesn't block the verdict. If crypto kills LCE, it tells us which of
two things happened: no signal anywhere, or a signal that exists on ES but that crypto's costs or
behaviour remove. The gross-vs-net split in the research note answers the "costs" half on crypto
alone.

Our prior is against it. `donchian_atr` at 1h had gross −372 over 4,793 trades, so no intraday
trend edge before costs on these coins (`docs/research/2026-10-18-strategy-research.md`, section 1).
LCE is intraday trend-following, gated by volume levels. This spec doesn't try to make it pass.
It tries to make a kill mean "LCE as taught has no edge here", not "we coded a strawman".

## What LCE is (sources)

**Who.** Tradewriter: @Tradewrite on X, Instagram and Whop; YouTube @Tradewrite_, 24.2K
subscribers, channel started March 2024. Sells the "LCE Accelerator" and a Whop mentorship.
Trades ES micros, often copied across many prop-firm accounts.

**What it is.** "LCE" = Level → Confirmation → Execution. The rules below come from 16 video
transcripts (pulled 2026-10-23) and the user's `some_stats/LCE lowchart.pdf` ("LCE Method
Flowchart"). Key videos:
- `dA5xyAMcVhs`, the EMA ribbon
- `fLp-ub7zvjA`, the cloud
- `EKCMg2eecJw`, drawing levels
- `CA00047_F7I` and `g2EQzWwck9c`, week walk-throughs
- `ATWyVRbrDvs` and `2hskGksJuDk`, management and session rules

| Part | As taught |
|---|---|
| Market, session | ES futures. Regular trading hours 9:30–16:00 New York time. "Skip the first 5 to 10 minutes no matter what." Usually done before noon. |
| Cloud | EMA20 and EMA50, shaded between. Higher timeframes are drawn on the 5m chart as a ribbon: 60/150 (15m), 120/300 (30m), 240/600 (1h). Read by slope, thickness and price position. |
| Levels | TradingView visible-range volume profile (row size 200) for high-volume nodes, plus session volume profile (row size 100) for each day's point of control (POC). The **anchor** is where a node and POCs line up. Zones are **7–10 ES points wide**, centred on the node. Further levels sit **20–30 points** apart (the market's "breathing pace"), at prices with "multiple tests and rejections". "Draw them once, then they work forever." |
| Decision (flowchart) | The higher-timeframe cloud (30m, 1h, 2h) is bullish, ranging or bearish. If price isn't at a supply/demand level, no trade. If it is, the lower-timeframe cloud (5m, 15m) picks one of six actions (table below). |
| Execution | Stop beyond the cloud or the structure ("halfway between the two levels"). Target the next level. No partials, no trailing. Move the stop to breakeven once price has gone 50–60% of the way to target. At most 2 trades a day. Stop for the day after one win. Same risk on every trade. He claims 1.5–2R. |

The six branches (flowchart):

| # | Higher timeframe | Lower timeframe | Action |
|---|---|---|---|
| 1 | bullish | bullish | buy the dip into the cloud at demand; target the next supply level |
| 2 | bullish | bearish or choppy | wait for a cloud flip, then go long; target the next supply level |
| 3 | ranging | bullish | wait for a bearish cloud flip, then go short; target the next demand level |
| 4 | ranging | bearish | wait for a bullish cloud flip, then go long; target the next supply level |
| 5 | bearish | bullish | wait for a cloud flip, then go short; target the next demand level |
| 6 | bearish | bearish | sell the rip into the cloud at supply; target the next demand level |

What he publishes as evidence: single-week replays told with the outcome known. Levels are drawn
with the week's later price already on screen, and none of them gives a trade count, win rate or
drawdown. No audited record or independent review was found.

## Decisions

| Question | Decision |
|---|---|
| Which LCE | **All six flowchart branches** in one plugin (user). |
| Session on crypto | **Match ES**: Mon–Fri, New York time with daylight saving, entries 09:40–15:50, flat by 15:55 (user: "match the times with ES"; the minutes are this spec's call, see "Session"). |
| Translating ES points | **Both**, as two plugins (user: "try both"): `lce_pct` uses fixed percentages of price, `lce_atr` uses multiples of daily ATR. |
| Trials | Two: N goes 7 → 8 → 9 (user: "start backtesting", having chosen both translations). |
| Lineage | One family. `lce_pct` is the root; `lce_atr` declares `PARENT = "lce_pct"`. They are two translations of one hypothesis, so they share one holdout read. If the parent passes, the child must beat it to be promoted (lab rule). **This spec's call.** |
| Judge order | `lce_pct` first, then `lce_atr`. **This spec's call.** |
| Per-branch view | One plugin, one trial per translation. The research note breaks P&L down by branch; no trial is spent per branch (user agreed to the playback). |
| Gross vs net | Reported per trade and per branch, in bp (user agreed to the playback). |
| Skipping the open | The first 10 minutes ("5 to 10"; the safe end), which the user agreed to. |

## Harness this builds on

**Decision (user, 2026-10-23): A.** The general judge and validation commits below leave
`feat/proposer` for their own PR(s) to `main`. Proposer-only commits stay on `feat/proposer`: its
docs, settings and start guards, `lab judge --proposer`, and `lab exam --id`. LCE is then rebased
onto that `main`.

The plan decides the exact split (some harness commits may depend on proposer ones) and is
written so that `feat/proposer` can rebase cleanly afterwards. If a needed harness commit can't
be separated from proposer code without rewriting it, stop and ask the user.

`feat/lce-model` was cut from `main` (b38cca6). Several things this spec relies on exist only on
`feat/proposer`, which is 25 commits ahead of `main`, unmerged, and still being committed to:

| Piece | Commit | Why LCE needs it |
|---|---|---|
| before-costs pass and its random benchmark | a9c736c | decides which kind of kill it is |
| per-trade slippage cost, scalable to zero | 8e3812f | the gross / fees / slippage / funding split |
| random entries take the plugin's long share | 107b1c6, c818011 | the benchmark the trials face |
| a trial is registered before plugin code touches real bars | d75bb0f | a crash counts toward N |
| transient sandbox failures retried | 2a2908e | |
| verdict diagnostics block | 48a812b, debf6a4 | the per-trade bp breakdown |
| run lock (judge vs exam) | cef051a, 5b3561f | |
| `lab preflight` | 0e85e93, 09604bd | the pre-trial check |

The two trials can't be rerun, so the harness they face has to be settled first. Three options:
- **A.** Land the harness commits above on `main` first, then build LCE on `main`.
- **B.** Build LCE on top of `feat/proposer`.
- **C.** Build on `main` as it is now, porting only the before-costs pass and the slippage
  column.

On any harness, `lab preflight --tf 5m` fails today: its synthetic market is built from 15m bars
and can't make 5m ones. If preflight is used, that needs fixing first.

## The translation constants (fixed here, before any trial)

Tradewriter's numbers are ES points, quoted in videos from 2025-07 to 2026-10. Over that span the
ES front-month (Yahoo `ES=F` daily bars, fetched 2026-10-23) had **median close 6,923** and
**median Wilder ATR(14) of 82.4 points = 1.169% of price**.

Every distance below is a multiple of one unit **u**:
- **`lce_pct`:** u = 0.01169 × the signal bar's close. Fixed percentage, as if every coin moved
  like ES.
- **`lce_atr`:** u = Wilder ATR(14) of completed session days (defined under "Session"), in
  price. As if every coin's level spacing scaled with its own volatility the way ES's does.

Both use the same code, apart from how u is computed (`params["unit"]`, a string, which the
sensitivity check skips).

| Quantity | ES as taught | In u | `lce_pct` | e.g. `lce_atr` on BTC (ATR ≈ 3%) |
|---|---|---|---|---|
| level step (target spacing) | 25 pts (20–30) | 0.303 | 0.361% | ≈ 0.91% |
| step search window | 20–30 pts | step × [0.8, 1.2] | 0.289–0.433% | |
| zone width | 8.5 pts (7–10) | 0.103 | 0.123% | ≈ 0.31% |
| "too big a candle to chase" | 20–30 pts in one 5m candle | 0.303 | 0.361% | ≈ 0.91% |

Costs, for scale: a stopped-out trade pays taker fees in and out (10 bp) plus slippage. With the
stop about half a step away:
- **`lce_pct`:** about 0.55R per loser before slippage.
- **`lce_atr` on BTC:** about 0.2R.

These constants never change after a crypto result has been seen (anti-goal 1).

## The plugin (`strategies/lce_pct.py`; `lce_atr.py` is a copy with `unit = "atr"`)

Timeframe **5m** (`timeframes = ["5m"]`). One-sentence hypotheses:
- `lce_pct`: "Taken at Tradewriter's ES-sized levels in fixed percentages, trades that follow
  his six-branch EMA-cloud flowchart at volume-profile levels during US equity hours earn more
  than random entries with the same exits, net of costs."
- `lce_atr`: the same claim, with levels in multiples of daily ATR.

### Session

Session times are computed from `open_time` (UTC ms) inside the plugin. The sandbox knows no named
time zone, so the plugin applies the US daylight-saving rule itself. New York is UTC−4 from the
second Sunday of March at 07:00 UTC to the first Sunday of November at 06:00 UTC, and UTC−5
otherwise. That has been the rule since 2007, which covers all our data.

A 5m bar with open time t closes at t+5m. Signals happen at the close and fill at the next open.
- **Entry window:** signal-bar close times from 09:40 to 15:45 New York time, Mon–Fri. That makes
  the latest fill 15:45, keeping entries inside "09:40–15:50".
- **Flat:** `manage` returns EXIT at the close of the bar that closes at 15:55. The fill is at
  15:55, before 16:00.
- **Session day:** for levels, daily ATR and daily counters, the day runs from 18:00 New York time
  to 18:00 the next day, matching the ES trading day. All 7 days count for crypto's levels and ATR
  (crypto trades at weekends). Entries happen only Monday–Friday.
- **US holidays:** not excluded. This spec's call; it's about 9 days a year and they don't matter
  on crypto.

### Clouds

EMAs of close (`ewm(span=n, adjust=False)`, causal) on 5m bars, as Tradewriter's ribbon does it.
For `ema_fast = 20` and `ema_slow = 50`, each cloud k uses lengths × M_k, with
**M = {5m: 1, 15m: 3, 30m: 6, 1h: 12, 2h: 24}**. So the 2h cloud is EMA 480 against 1200.

**State of cloud k at bar t:**
- `mid = (fast + slow) / 2`
- `slope = (mid[t] − mid[t − M_k]) / u × 12 / M_k`, in u per hour
- state = +1 if fast > slow and slope ≥ θ; −1 if fast < slow and slope ≤ −θ; else 0 (flat or
  ranging)

**Higher timeframe (HTF)** = +1 if at least 2 of {30m, 1h, 2h} are +1 and none is −1; −1
mirrored; otherwise 0 (ranging).

**Lower timeframe (LTF)** = +1 if both 5m and 15m are +1; −1 if both are −1; otherwise 0
(choppy).

**Flip:**
- bullish 5m flip at t: fast[t] > slow[t] and fast[t−1] ≤ slow[t−1]
- bearish flip: mirrored

These are this spec's calls. Tradewriter gives no thresholds ("mature", "healthy", "flat"). θ is a
walk-forward grid parameter, so the data picks between two pre-registered values.

### Levels (point-in-time, recomputed at each session-day start D)

Only bars with `open_time < D` are used, and the levels stay fixed until the next D. "Draw once,
work forever" is approximated by daily redraws from recent data, because a backtest can't draw
once with hindsight.

1. **Visible-range profile (VRVP).** Over the last `vp_days` session days before D, split
   [min low, max high] into **200 rows**. Spread each bar's volume evenly over the rows its
   [low, high] covers, in proportion to the overlap.
2. **Session profiles (SVP).** Each of those days gets its own profile: **100 rows** over that
   day's range. Its POC is the centre of its highest row.
3. **High-volume nodes (HVN).** A row is an HVN if its volume is the maximum within ±2 rows and
   at least the 70th percentile of the 200 rows.
4. **Anchor.** Among the HVNs, take the one with the most POCs within ± zone/2 of its centre. Ties
   go to the higher volume. If no HVN has a POC that close, take the HVN with the highest volume
   and count that day in a diagnostic.
5. **Grid.** From the anchor's centre c, the next level up is the centre of the highest-volume
   VRVP row whose centre lies in [c + 0.8·step·u, c + 1.2·step·u]. That is his "look 20–30
   points above for a place with a lot of tests and rejections", with volume standing in for
   tests. Repeat from each new level. The same goes downward.
   - Outside the VRVP's range, use c ± step·u.
   - Extend until there are at least 10 levels beyond the VRVP range on each side.
6. **Zones.** Each level's zone is centre ± zone·u / 2, with u taken at D. Zones can't overlap
   (step ≥ 0.8 × 0.303 > 0.103).

Constants written as literals in the plugin, not params: 200 and 100 rows (his settings), ±2 rows
and the 70th percentile (this spec's HVN rule), the 0.8–1.2 search window (his 20–30 around 25),
and the cloud multipliers M (his timeframes). The sensitivity check perturbs params, not
literals.

### "At a level", the six branches, and the bracket

Definitions, all at the signal bar t and all computed in `prepare` as features:
- **at_zone(t):** bar t's [low, high] overlaps a zone.
- **touched(t):** at_zone held on any bar in [t − wait_bars + 1, t].
- **in_window(t):** bar t's close lies in the entry window.
- **big(t):** |close − open| > big_bar·u, which means skip.

Branch conditions (a bar meets at most one):

| # | Condition at t | Side |
|---|---|---|
| 1 | HTF +1, LTF +1, at_zone, low ≤ fast5m, close > slow5m (dipped into the cloud and held) | long |
| 2 | HTF +1, bullish 5m flip, touched | long |
| 3 | HTF 0, LTF(t−1) = +1, bearish 5m flip, touched | short |
| 4 | HTF 0, LTF(t−1) = −1, bullish 5m flip, touched | long |
| 5 | HTF −1, bearish 5m flip, touched | short |
| 6 | HTF −1, LTF −1, at_zone, high ≥ fast5m, close < slow5m (ripped into the cloud and failed) | short |

Branch 2 lumps "bearish or choppy" together, as the flowchart does; a bullish flip implies 5m
wasn't +1 at t−1. Branch 5 mirrors it. `prepare` writes `sig_side` (−1/0/+1) and `sig_branch`
(0–6) per bar, without the daily rules, so trades can be attributed to branches afterwards.

**`bracket(ctx, side)`** depends only on the close and the day's levels, so it is identical for
the strategy and for random entries. For a long:
- **Base level:** Z = the highest zone whose centre ≤ close.
- **Target:** the lower edge of the lowest zone whose lower edge > close.
- **Stop:** the midpoint between Z's centre and the centre of the zone below Z ("halfway between
  the two levels").
- **Valid only if:** stop < close < target and in_window(t). Otherwise it returns a NaN stop, so
  neither the strategy nor RandomEntry enters (`enter()` returns HOLD).

Shorts mirror this. With these geometries a stop is about half a step away and a target about one
step, which matches the 1.5–2R he states.

### Entry, management and the daily rules

**`entry(ctx)`** (stateful per run; a new instance per run, as the sandbox already creates):
- Enter `sig_side` with `bracket(...)` when all hold: in_window, not big, a valid bracket, fewer
  than `max_trades` filled entries this session day, and no win yet this session day.
- **Filled entries:** counted when a position appears at the next bar. Signals the engine rejects
  don't count.
- **Win:** a trade whose exit bar traded through the target while the stop in force wasn't hit,
  checked the way the engine checks it (stop first). Breakeven and end-of-day exits aren't wins.
  The plugin records the stop and target it last saw, to detect this.

**`manage(ctx)`** must not depend on anything `entry` stored, because RandomEntry calls the base's
`manage` without ever calling its `entry`:
- **Breakeven:** when (best_close − entry)/(target − entry) ≥ `be_frac` (mirrored for shorts),
  hold with stop = entry price. The engine only ratchets the stop forward.
- **End of day:** EXIT at the close of the 15:55 bar.
- Nothing else: no partials, no trailing.

### Params and grid

`params = {"ema_fast": 20, "ema_slow": 50, "vp_days": 5, "step": 0.303, "zone": 0.103,
"big_bar": 0.303, "theta": 0.01, "wait_bars": 6, "be_frac": 0.5, "max_trades": 2, "atr_n": 14,
"unit": "pct"}` (`"atr"` in `lce_atr`).

`grid = {"vp_days": [5, 10], "theta": [0.01, 0.02]}`: 4 combinations. θ is in u per hour;
0.01 is about a quarter of a u per day.

The other values come from his numbers and the ES calibration above. The sensitivity check
(±10/20% on each numeric param) shows whether the result depends on any of them.

## The random-entry benchmark has to stay in session (harness change)

Today `RandomEntry` fires on any flat bar with probability n_trades / OOS flat bars. LCE only
trades about a fifth of the week (6h05m of 24h, five days of seven), so two problems follow:

- **If random entries could fire at any hour,** LCE would be compared with night-time trades, and
  the session alone could make it look skilled.
- **With `bracket` returning NaN outside the window,** random entries stay in session. But they
  fire about a fifth as often as LCE trades, so the benchmark's spread is wrong.

**Change:**
- **A new literal.** A plugin may declare `SESSION`, which the parent reads from the source like
  `PARENT` and `timeframes`: New York time, weekdays, and the first and last signal-bar close
  times.
- **Calibration.** When `SESSION` is declared, `validate.benchmarks.random_entry` counts only OOS
  flat bars whose close falls in the window. That makes the expected number of random trades
  match the strategy's.
- **A check.** The walk-forward verdict gets a fail reason if any trade's `signal_time` falls
  outside the plugin's declared `SESSION`. `lab judge` always runs that check; preflight doesn't
  have to. The plugin's unit tests check the same thing before any trial.

Plugins without `SESSION` behave exactly as before, so earlier verdicts stand. This goes in a
separate commit with its own tests.

## Data

**Backfill:** `lab backfill --tfs 5m` (klines only) for the 17 coins. That's about 690k bars per
coin from 2020 to the holdout. Funding and metrics are already present. Add `5m` to
`config/lab.yaml` `timeframes` only if the harness requires it.

**No 1m data or aggTrades.** The profiles are built from 5m bars by spreading each bar's volume
over its range. That is a coarser approximation than TradingView's intrabar data, and it is noted
under fidelity below.

## Performance budget (a hard constraint)

A streamed exam runs `prepare` over every bar held at each step. `exam.cost_hours` measures that
when `lab exam` runs, multiplied by the holdout bars by then, which grow by 288 a day.

The earliest exam is day 180 of the holdout (`min_holdout_days`, about 2026-12-28). Sized for an
exam by **day 240** (about 2027-02-26, two months' slack) under the 6 CPU-hour budget, **`prepare`
must take ≤ 0.3 s**. That's the median in a worker, on BTC's full research 5m bars, at the slower
grid setting `vp_days = 10`. Measure it before the first trial. **This spec's call.**

If it can't be met, stop and report the measured cost. A plugin that can't be examined can't
enter the book, and raising `exam_cpu_hours` is the user's call.

## Looks at real data before the trials

This follows the trial-7 note (`docs/research/2026-10-18-strategy-research.md`, section 8).

**Allowed:** looks that show no P&L, using research bars only and run through the sandbox:
- per-coin counts of each branch's signals
- the share of signals with a valid bracket
- the anchor-fallback share and the level spacing
- whether every session day has its 15:55 bar, and what happens at listings and gaps

**Never:** forward returns, hit rates, R, or anything joined to later prices. `lab backtest`
isn't used, because it prints Sharpe.

**After the looks:**
- Only bugs (code that doesn't do what this spec says) may change.
- Constants, thresholds and rules don't change.
- If a look shows the spec itself can't work (say a branch fires on no coin), stop and bring it to
  the user.
- Every look is listed in the research note, with its reason.

**This spec's call, following precedent.**

Walk-forward cost, for planning: about 4 grid runs + 200 random seeds + ~40 sensitivity variants,
each × 17 coins, plus the before-costs runs. That's thousands of 5m runs, roughly 1–3 hours per
trial on 8 workers.

## Where this differs from Tradewriter (so a kill can be read)

1. **Levels** are redrawn daily from the last 5–10 days of 5m volume. His are drawn by eye, with
   "tests and rejections" judged visually, and he keeps them for weeks.
2. **Cloud states** use one numeric slope threshold. His "mature", "healthy" and "Goldilocks
   distance from the cloud" are judgement calls; the Goldilocks distance is left out.
3. **Entries** are market orders at the next 5m open (the engine has no resting entry orders).
   He enters on a TradingView alert, sometimes with limits.
4. **No A+/B grading,** and no "additional context" (the flowchart's footnote).
5. **Crypto has no 9:30 open auction.** The window is clock time only.
6. **Volume profile** comes from 5m bars, not intrabar data.
7. **Multiple coins.** He trades one ES contract, with 2 trades a day in total. We apply 2 a day
   *per coin*, across 17 coins.

The research note discusses which of these could plausibly hide an edge.

## Research note (after the two verdicts)

`docs/research/2026-10-2x-lce/`, following the trial-7 note's format:
- **Verdicts:** both, with every fail reason.
- **Per branch:** trades, win rate, mean R, gross / fees / slippage / funding in bp per trade.
  Attribution joins `sig_branch` at each trade's `signal_time`, from a features-on rerun of the
  modal params in the sandbox; this spends no trial.
- **Against his claims:** trades per day per coin, R per trade, and win rate, compared with his
  1.5–2R and "85%".
- **Levels:** the share of days on the anchor fallback, and the distribution of level spacing.
- **What a kill means**, by a rule fixed before the trials. It goes in
  `docs/research/2026-10-2x-lce/checks.py` and is committed before the first `lab judge`, as
  trial 7's checks were:
  - **No signal:** the before-costs pass's OOS Sharpe doesn't beat its own random-entry p95 (same
    exits, zero fees, zero slippage, zero funding).
  - **Signal eaten by costs:** the before-costs pass beats its random p95, but the net run doesn't
    beat its random p95.
  - **Other:** both beat random, but another gate kills it (island, Deflated Sharpe, trade
    count). The note names the gate.
  - **Gross, wherever per-trade numbers are shown:** before fees, slippage and funding
    (`pnl_gross + slippage` on the `feat/proposer` engine).

  **This spec's call.**

## ES leg (next spec, not built here)

Needs ES data with **real exchange volume at 1m or 5m**; CFD tick-count feeds won't do for volume
profiles. The user is looking for a source. That spec will cover:
- an ES bar loader
- an ES cost model (commission plus 1 tick)
- the CME calendar (holidays, early closes)
- a third, literal "points" translation (u = 82.4 pts fixed)

It will run the same plugin code, so the comparison is fair.

## Non-goals

- The ES leg (above).
- The videos' "break of the level at the alert" trigger as a separate variant. The user chose the
  flowchart.
- 1m data, aggTrades, limit-order entries.
- NQ or other markets.
- Tuning anything after a result.

## Anti-goals (failure even if it works)

1. **Changing a constant, threshold, branch rule or session time after a crypto result has been
   seen.** If a re-test is wanted, it is a new spec and a new trial, and the user decides.
2. **Levels that use any bar at or after their session-day start.**
3. **A random benchmark that trades at hours or frequencies the strategy can't.**
4. **Reading the holdout outside `lab exam`.**
5. **Reporting a promotion without the per-branch and gross/net view,** or a kill without saying
   which kind.
6. **Spending more than the two trials.**

## Order of work

0. **Harness base,** per the user's answer under "Harness this builds on".
1. **Data:** backfill 5m klines for the 17 coins (plus a config change if needed). Done
   2026-10-23: 1,328 months downloaded, 0 errors, BTC has 683,424 research bars.
2. **Harness:** the `SESSION` literal, benchmark calibration and the out-of-session verdict check,
   with tests. Also preflight's 5m synthetic market, if the base has preflight.
3. **Plugin `lce_pct`, with tests:**
   - session clock, including both DST switches
   - volume profiles and POC/HVN detection
   - anchor and grid
   - causality of the levels
   - cloud states and flips
   - each branch firing and not firing
   - bracket geometry and NaN out of session
   - breakeven and end-of-day exits, and `manage` being stateless
   - the daily cap, no entry after a win, breakeven exits not counting as wins
   - `lab preflight`
   - the 0.4 s performance budget
4. **Plugin `lce_atr`:** a copy with `unit = "atr"` and `PARENT = "lce_pct"`. Tests that only u
   differs.
5. **Trials:** `lab judge lce_pct --tf 5m`, then `lab judge lce_atr --tf 5m`.
6. **The research note.**

Each step is its own commit; one PR per step or per pair of steps, merged to `main`.

## Left to the builder

- How the parent gets `SESSION` into benchmark calibration, as long as it meets the change above.
- Vectorising the profiles to meet the performance budget, as long as the numbers match a simple
  reference implementation in tests.
- How the plugin reconstructs wins, as long as it matches the engine's fill order (tested).
- Edge cases: no levels yet (warm-up), a coin listed mid-history, gaps. Use NaN or no entry, never
  a guess.
