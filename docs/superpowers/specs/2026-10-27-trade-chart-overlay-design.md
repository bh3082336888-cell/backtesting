# Trade charts: TradingView-style trades for every run, and an LCE overlay

Status: approved by the user 2026-10-27 ("no need to wait for my approval; start implementing"). Branch `feat/trade-overlay` (worktree
`.worktrees/overlay`). Mockups: the brainstorm screens `overlay-options.html` and
`overlay-entry-v5.html` (option B1).

## Why

The user opens a walk-forward or paper run in the dashboard to see what trades a strategy took and
when, and to check them against how a discretionary trader would have drawn them.

Today's chart marks a trade with an entry triangle, an exit ✕ and dotted stop and target step
lines. The user wants two things:
1. **Every run:** trades drawn the way TradingView's position tool draws them, so the entry price and
   time, stop, target, R:R and result can be read at a glance. This becomes the standard look of
   trades on every trade chart: every strategy, and both walk-forward and paper runs.
2. **LCE runs only:** the LCE model's own context on the same chart:
   - the liquidity zones
   - the 5m EMA cloud
   - the volume profile the zones were built from

   This way the user can see whether the code trades the way Tradewriter does on his charts.

## What the user decided

| Question | Decision |
|---|---|
| Trade style | TradingView position box (option B) with **axis tags** (B1): price tags on the price edge, time tags on the time axis |
| Ratio | The box shows the **R:R** ratio, like TradingView |
| Standard | The trade style is the standard for **all** runs' trade charts, any strategy. The indicator layers are LCE-only. |
| LCE layers on by default | Liquidity zones, 5m EMA cloud, volume profile |
| LCE layers off by default, each with a toggle | Higher-timeframe cloud, branch label |
| Volume profile | By default, the profile the plugin **built that day's levels from**. A **visible-range** profile of the window is a toggle. |

## Part 1: how a trade looks (every run)

Each trade in the window is drawn from the engine's trade record (`oos_trades.parquet` or the
paper runner's `trades.parquet`), on the same candles the run read. Nothing is drawn from an
assumption: a target exit on a candle that never reached the target would be visible at once.

- **Reward box:** green, from the entry fill to the target, over [entry_time, exit_time]. If the
  target moved (the level path), the box follows it in steps.
- **Risk box:** red, from the entry to the stop, following the stop's level path. A move to
  breakeven makes it collapse at that time; a trailing stop makes it step. Stops never move
  against the trade, because the engine only ratchets.
- **Trades without a target** (plugins that exit on a signal or a trailing stop, e.g.
  `donchian_atr`): there is no reward box. The span from entry to the exit price is shaded green
  for a gain and red for a loss, and labelled "no fixed target".
- **Entry:** a bold line at the entry price across the trade, and a dot on the entry candle at the
  fill price.
- **Exit:** a dot on the exit candle at the exit price, coloured by exit reason with the existing
  colours (target, stop, signal, liq, eow).
- **Axis tags (B1):**
  - on the price edge: **Entry** (blue), **TP** (green, if any) and **SL** (red) at the initial
    prices;
  - on the time axis: **"HH:MM entry"** and **"HH:MM exit"**, with dotted vertical lines.

  To keep a busy window readable, tags are drawn for the **focused trade only** (the one picked in
  "Jump to trade"), or for every trade when at most 5 are in view. **This spec's call.**
- **Label inside the box:**
  - **"R:R 1 : x.xx"**, the planned |target − entry| / |entry − stop_init|; "no fixed target" when
    there is none;
  - **"±a.aaR gross · ±b.bbR net"**, where gross = `pnl_gross` / (qty·|entry − stop_init|) (after
    slippage, before fees and funding) and net = the engine's `r_multiple`.

  The label is shown under the same rule as the tags.
- **Breakeven marker:** where the stop reached the entry price, a small "SL → BE HH:MM" note.
- **Hover:** hovering the box shows today's full tooltip (P&L, fees, funding, fold and combo,
  regime), plus R:R and gross/net R.
- **Kept as they are:** fold shading, the paper warmup band, coarser candles for long windows
  (`MAX_CANDLES`), the date picker and "Jump to trade". On coarser candles, boxes and tags stay
  at exact times and prices.
- **Older runs** without level paths: the boxes use the initial stop and target, as today's chart
  does.

The signal-bar marker (an open circle at the signal bar's close, joined to the fill) stays, as a
legend entry the user can hide. **This spec's call.**

## Part 2: the LCE overlay (runs of plugins that produce LCE's features)

### Where the data comes from

Plugin code runs only in the sandbox, and the judged plugin files must not change: their hashes
are the trials' identity.

**The plugin's data (`lab overlay RUN_ID [--symbols ...]`).** A new trusted command that, for a
walk-forward run of a plugin whose `prepare` output has LCE's feature columns:
- **Reads the run:** its plugin hash and source (it refuses if the file in `strategies/` no longer
  hashes to the run's `strategy_hash`), its folds and each fold's selected `combo` and params.
- **Reruns the plugin in the sandbox:** for each coin and each combo used by some fold, it runs the
  plugin on the coin's research 5m bars with `features=True` (`client.spawn` → `w.bars` → `w.run`).
  It reads only the features, never the run's decisions, trades or equity, and research bars only,
  never the holdout. A whole-history feature run is about 150 MB per coin, over the 64 MB worker
  output limit, so the command raises `sandbox.max_output_mb` for its own workers in memory, as
  `docs/research/2026-10-23-lce/looks.py` does.
- **Stitches by fold:** each fold's test window takes the features of that fold's selected combo,
  so every drawn zone, cloud value and branch is exactly what the traded parameters produced.
- **Writes `runs/<run_id>/overlay/`:**
  - `meta.json`: run id, plugin name and hash, per-fold combos, created_at
  - `<SYMBOL>_bars.parquet`: per bar over the OOS span: `open_time`, `fast5`, `slow5` (float32)
  - `<SYMBOL>_zones.parquet`: per session day (`sday`): every distinct zone centre the plugin's
    `z_dn_c`, `z_dn2_c`, `z_up_c` and `z_up2_c` named that day, and the day's zone half-width.
    The half-width is the median of `z_up_c − z_up_lo` over the day's bars where both name the same
    zone.
  - `<SYMBOL>_signals.parquet`: per bar with `sig_branch > 0`: `open_time`, `sig_branch`, `sig_side`
- **Overwriting:** it overwrites its own previous output for the run. A run with no overlay just
  charts without LCE layers.

**The dashboard's own layers.** Trusted code, computed from the candles and labelled display-only
in the legend:
- **Level-building volume profile:** the plugin's method re-implemented: an exact even spread of
  each bar's volume over 200 rows, across the `vp_days` session days before the shown day, with
  the fold's `vp_days`. The point of control and the high-volume nodes (the plugin's rule: largest
  within ±2 rows and ≥ the 70th percentile) are marked. The shown day is the focused trade's
  session day, otherwise the last day in the window.
- **Visible-range profile** (toggle): the same spread over the window's own candles.
- **Higher-timeframe cloud** (toggle): EMA(ema_fast×12) and EMA(ema_slow×12) of close, the 1h
  cloud, computed with a lead-in of at least 3× the slow length before the window so it matches
  the plugin's values to display precision.

### How the layers look

- **Liquidity zones:** for each session day in view, every zone is a horizontal band over that
  day's 18:00→18:00 New York span. Zones below the day's first open are demand (green), above it
  are supply (red). A zone a trade used (the entry zone the bracket started from, and the target
  zone) gets a stronger outline and a small label. **The colour rule is this spec's call.**
- **5m EMA cloud:** the band between `fast5` and `slow5`, green where fast > slow and red
  otherwise, behind the candles.
- **Volume profile:** a horizontal histogram along the right edge of the price area. The point of
  control is orange with a faint line across the chart; high-volume nodes are blue.
- **Branch label** (toggle): "Branch k · <short name>" added to the trade label. The six short
  names come from the spec's branch table: dip into the cloud at demand, flip long in an
  uptrend, fade at range high, fade at range low, flip short in a downtrend, rip into the cloud
  at supply.
- **Toggles:** a row of toggles above the chart, shown only when the run has overlay data for the
  symbol. Zones, the 5m cloud and the level-building profile start on; the visible-range profile,
  the higher-timeframe cloud and the branch label start off. The choices persist per run in the
  session.

## Non-goals

- LCE layers for other plugins, or a general plugin-overlay API.
- Overlays for paper runs (the trade style applies to paper runs; the LCE layers don't, yet).
- Holdout and exam runs; in-sample backtests (they aren't chartable today either).
- Any change to `strategies/lce_pct.py` or `lce_atr.py`.
- Drawing tools, editing, or exporting images beyond Plotly's own.

## Anti-goals (failure even if it works)

1. **Overlay values the traded parameters didn't produce:** a zone or cloud from a combo other
   than the fold's, or zones drawn from data after their day began.
2. **Plugin code running outside the sandbox,** or anything reading the holdout.
3. **A trade drawn from anything but the engine's record** (entry, exit, stop path, target path).
4. **A slow dashboard:** overlay files are read only for the window shown, and cached per (run,
   symbol), as the candles are.

## Tests (the builder's to place)

- **Trade style:**
  - long and short trades with a target;
  - a trade with no target;
  - a breakeven move (the risk box collapses at its time);
  - a trailing stop in steps;
  - an older run without level paths;
  - tags only for the focused trade (or for ≤5 in view);
  - the R:R and gross/net R text against hand-computed values;
  - a paper run;
  - coarse candles keep exact box coordinates.
- **`lab overlay`:**
  - on a synthetic walk-forward run (the existing sandbox fixtures), it stitches per fold;
  - it refuses a run whose plugin file changed, and a non-walk-forward run;
  - it writes only under the run's `overlay/`;
  - zones and half-widths match the plugin's own `level_features` on the synthetic bars (through
    `load_class` in tests, which is allowed on synthetic data);
  - it never loads holdout bars.
- **Dashboard layers:**
  - the level-building profile matches the plugin's reference profile on synthetic days (rows,
    POC, HVNs);
  - the higher-timeframe cloud matches the plugin's EMA within 1e-6 relative after the lead-in;
  - toggles' defaults;
  - no overlay data → no toggles and an unchanged chart.

## Order of work

1. The trade style for every run (`dashboard/trade_chart.py`, `dashboard/app.py`), with tests. It
   stands alone and changes every run's chart.
2. `lab overlay` and its artifact, with tests.
3. The LCE layers and toggles in the dashboard, with tests.
4. Run `lab overlay` on the two LCE trial runs once both verdicts exist (the per-branch numbers in
   the research note use the same rerun).

Each step is its own commit; one PR for steps 1–3, and step 4 runs from `main` after it merges.
Work happens only in `.worktrees/overlay`. The LCE trials run from `.worktrees/lce`, whose sandbox
workers import `perp_lab` from that tree, so nothing there may change while they run.

## Left to the builder

- Plotly specifics: shapes vs traces for the boxes, annotation styling, legend grouping, how tags
  sit at the axis edges.
- The exact toggle widgets.
- How the overlay files are partitioned for fast window reads, as long as the layout above holds.
- The colour values, from the dashboard's existing palette (light and dark).
