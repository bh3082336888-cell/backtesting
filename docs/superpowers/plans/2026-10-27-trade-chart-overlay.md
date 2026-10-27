# Trade-Chart Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draw every run's trades as TradingView-style position boxes with axis tags, R:R and
gross/net R. On LCE runs, add the zones, the 5m cloud and the level-building volume profile (plus
the visible-range profile, the 1h cloud and the branch label as toggles), fed by a new
`lab overlay` command.

**Architecture:**
- **Trade style:** stays in `dashboard/trade_chart.py` as plain, testable geometry functions plus
  the Plotly drawing.
- **The overlay artifact:** built by trusted `perp_lab/overlay.py`. It reruns the judged plugin in
  the sandbox per fold combo, stitches the features by fold, checks them against the run's own
  trades, and writes `runs/<run_id>/overlay/`.
- **Display-only LCE layers:** a new `dashboard/lce_overlay.py` computes the profiles and the 1h
  cloud from candles and draws all the LCE layers. `dashboard/app.py` adds the toggles and passes
  the focused trade.

**Tech Stack:** Python 3.12, pandas/numpy, Plotly, Streamlit (AppTest in tests), the lab's sandbox client, typer CLI, pytest.

**Spec:** `docs/superpowers/specs/2026-10-27-trade-chart-overlay-design.md` (approved 2026-10-27). Read it before any task.

## Global Constraints

**Every trade chart:**
- Trade style applies to all chartable runs (walk-forward and paper), any strategy.
- Reward box green, entry → target over [entry_time, exit_time], following the target's level path.
  Risk box red, entry → stop, following the stop's level path, collapsing at the entry once the stop
  reaches or passes it.
- No target: no reward box; entry → exit price shaded green (gain) or red (loss), labelled
  "no fixed target".
- Entry: a bold line at the entry price and a dot on the entry candle. Exit: a dot on the exit
  candle, coloured by reason (existing `exit_colors`).
- Axis tags: price tags "Entry" (blue), "TP" (green), "SL" (red) at the initial prices; time tags
  "HH:MM entry" and "HH:MM exit" with dotted verticals. Shown for the focused trade only, or for
  every trade when ≤ 5 are in view.
- Box label: "R:R 1 : x.xx" = |target − entry| / |entry − stop_init|, else "no fixed target".
  Then "±a.aaR gross · ±b.bbR net", with gross = (pnl_gross + slippage)/(qty·|entry − stop_init|),
  net = `r_multiple`. With no `slippage` column, slippage = 0 and the label reads "after slippage".
- Breakeven note "SL → BE HH:MM" where the stop reaches the entry price.
- Times are shown in UTC, as the chart's axis is UTC.

**LCE layers:**
- Toggles: `zones`, `cloud5`, `profile` start ON; `vrvp`, `cloud1h`, `branch` start OFF. They
  appear only when the run has overlay data for the symbol.
- Overlay files under `runs/<run_id>/overlay/`:
  - `meta.json`
  - `<SYMBOL>_bars.parquet` (open_time, fast5, slow5 float32)
  - `<SYMBOL>_zones.parquet` (sday, centre, first_time, hz)
  - `<SYMBOL>_signals.parquet` (open_time, sig_branch, sig_side)
- Profiles (display-only): 200 rows with an exact even spread; session POCs on 100 rows; HVN = the
  largest within ±2 rows and ≥ 70th percentile, ties within 1e-9 × the largest row going to the
  lowest row; anchor = the HVN with the most POCs within ± hz, then the higher volume, then the
  lower price.
- 1h cloud: EMA(close, span = ema_fast×12) and EMA(close, span = ema_slow×12), `adjust=False`,
  with a lead-in of ≥ 12,000 5m bars.
- Branch names, by `sig_branch` 1–6:
  1. "dip into the cloud at demand"
  2. "flip long in an uptrend"
  3. "fade at range high"
  4. "fade at range low"
  5. "flip short in a downtrend"
  6. "rip into the cloud at supply"

**`lab overlay`:**
- Walk-forward runs only, and only when `strategies/<name>.py` hashes to the run's
  `strategy_hash`.
- Plugin code runs only in the sandbox.
- Research bars only.
- `sandbox.max_output_mb` is raised in memory to 512 for its own workers.
- From each rerun, only `.features` is read.
- It self-checks every OOS trade before writing, and writes nothing on any mismatch.

**Repo rules:**
- Never edit `strategies/lce_pct.py` or `strategies/lce_atr.py`.
- Work only in `/home/burhan/backtesting/.worktrees/overlay` (own `.venv`; `data` and `runs` link
  to the main checkout). Never touch `.worktrees/lce`: trials run from it.
- Commit trailers: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01WZgKtXA9TNg2AeUCiUM4ii`. One concern per
  commit. Tests ship with the change.

## Review Focus

1. **Short trades:** reward box below, risk above, the R:R and tags mirrored. Tested in Task 1.
2. **A trade that opens before the window or closes after it:** its box is drawn over the part in
   view, with no error and no tag for an off-screen time. Tested in Task 1.
3. **More than 5 trades in view:** boxes for all of them; tags and labels only for the focused
   trade; hover on every one. Tested in Task 1.
4. **A coin with no overlay file** (no trades, or built with `--symbols`): no toggles, chart
   unchanged. Tested in Task 3.
5. **A run that `lab overlay` must refuse:** not walk-forward, a changed plugin file, or a
   self-check mismatch. A clear error, nothing written. Tested in Task 2.

---

### Task 1: TradingView-style trades on every trade chart

**Files:**
- Modify: `dashboard/trade_chart.py` (replace `_draw_trades`; extend `TRADE_COLS` with `qty`,
  `pnl_gross`, `slippage` where present; add the functions below)
- Modify: `dashboard/app.py` (`trades_chart_section` passes `focus` = the trade picked in
  "Jump to trade", or None)
- Test: `tests/test_trade_chart.py` (update tests that asserted the old triangles and ✕; add the
  new ones)

**Interfaces:**
- Produces:
  - `rr_text(r: dict) -> str`: "R:R 1 : 1.03" or "no fixed target".
  - `result_text(r: dict) -> str`: "+1.03R gross · +0.73R net", or "… after slippage …" without a
    `slippage` column.
  - `trade_boxes(r: dict) -> dict`: plain geometry in epoch ms and prices, with keys `reward`
    (list of (t0, t1, y0, y1)), `risk` (same), `entry` (t, price), `exit` (t, price, reason),
    `be_time` (ms or None), `no_target` (bool).
  - `trade_figure(candles, trades, tf, *, palette=None, folds=None, focus: int | None = None, overlay=None, layers: frozenset = frozenset()) -> go.Figure`.
    `focus` is a row label of `trades`. `overlay` and `layers` are used by Task 3; with
    `overlay=None` they're ignored.

- [ ] **Step 1: Write the failing tests**
  - `test_rr_and_result_text_long_and_short`: long entry 99.06, stop 98.82, target 99.30,
    qty 10, pnl_gross 2.4, slippage 0.072, r_multiple 0.73 → "R:R 1 : 1.00" and
    "+1.03R gross · +0.73R net". The mirrored short gives the same text.
  - `test_result_text_without_slippage_column_says_after_slippage`
  - `test_no_target_trade_has_no_reward_box_and_says_so`: a gain is shaded green, a loss red,
    `rr_text` gives "no fixed target".
  - `test_breakeven_collapses_the_risk_box_at_its_time`: level path stops [98.82, 99.06] →
    `be_time` = the second level time; no risk rectangle after it.
  - `test_trailing_stop_past_entry_collapses_risk_and_keeps_the_step_line`
  - `test_short_trade_boxes_are_mirrored` (Review Focus 1)
  - `test_trade_starting_before_the_window_draws_from_the_window_start` (Review Focus 2)
  - `test_tags_only_for_the_focused_trade_when_more_than_five` and `test_tags_for_all_when_five_or_fewer` (Review Focus 3): count the price- and time-tag annotations.
  - `test_coarse_candles_keep_box_coordinates_exact` (update the existing coarse-candles test)
  - `test_paper_run_draws_boxes`
  - `test_old_trades_without_levels_use_the_initial_stop_and_target` (update the existing one)
  - `test_app_passes_the_jumped_to_trade_as_focus` (AppTest)
- [ ] **Step 2:** Run `.venv/bin/python -m pytest -q tests/test_trade_chart.py`. Expected: the new tests FAIL.
- [ ] **Step 3:** Implement. Boxes as Plotly rectangle shapes, or filled scatter traces, so hover
  works on them (builder's choice). Tags as annotations: price tags at the plot's right edge
  (`xref="paper"`), time tags at the bottom (`yref="paper"`). Keep the signal-bar circle as a
  hideable legend entry, plus fold shading, the paper warmup band and coarse candles. Colours
  come from `PALETTE` and the dashboard's palette.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest -q tests/test_trade_chart.py tests/test_dashboard.py`. Expected: PASS, no warnings.
- [ ] **Step 5:** Commit: "feat(dashboard): trades drawn as TradingView-style position boxes on every trade chart".

### Task 2: `lab overlay`

**Files:**
- Create: `perp_lab/overlay.py`
- Modify: `perp_lab/cli.py` (an `overlay` command: `lab overlay RUN_ID [--symbols A,B] [--workers N]`, which prints the written directory)
- Test: `tests/test_overlay_build.py`

**Interfaces:**
- Produces:
  - `fold_plan(summary: dict) -> list[dict]`: one dict per fold with `fold`, `test_start`,
    `test_end` (UTC Timestamps), `combo` (str) and `params` (dict).
  - `stitch(feats_by_combo: dict[str, pd.DataFrame], folds: list[dict]) -> pd.DataFrame`.
    A session day belongs to the fold that holds its NY trading date (the date of its 09:40–15:45
    window), even where its 18:00 start falls in the previous UTC month.
  - `zones_table(feats: pd.DataFrame) -> pd.DataFrame`, with columns `sday, centre, first_time, hz`:
    - centres: distinct values of `z_dn_c`, `z_dn2_c`, `z_up_c`, `z_up2_c` per sday (deduplicated
      within 1e-9 relative);
    - `first_time`: the first bar naming each one;
    - `hz`: the median of `z_up_c − z_up_lo` over the day's bars where 0 < difference < the day's
      median gap between consecutive centres.
  - `self_check(feats: pd.DataFrame, trades: pd.DataFrame) -> list[str]`: mismatch messages; empty
    when every trade's signal bar has `sig_side == side`, and `stop_init` and `target` equal the
    bracket rebuilt from the zone columns within 1e-9 relative. Long: ((z_dn_c+z_dn2_c)/2, z_up_lo).
    Short: ((z_up_c+z_up2_c)/2, z_dn_hi).
  - `build_overlay(cfg, run_id: str, symbols: list[str] | None = None, workers: int | None = None) -> Path`.
    It raises `OverlayRefused(msg)` for a run that isn't walk-forward, a plugin whose hash differs,
    a plugin without LCE's feature columns, or a self-check mismatch.

- [ ] **Step 1: Write the failing tests**
  - `test_fold_plan_reads_folds_and_combos` (from a hand-written summary.json)
  - `test_stitch_takes_each_fold_from_its_combo` and `test_session_day_straddling_a_month_goes_to_its_trading_date_fold`
  - `test_zones_table_matches_the_plugins_level_features`: synthetic 5m bars and `level_features`
    through `load_class("lce_pct")`, so the plugin runs in the test process on synthetic data. The
    zones equal the distinct level centres the plugin's lookup columns named, `first_time` is right,
    and `hz` equals `zone·u_d/2`.
  - `test_self_check_passes_on_matching_trades_and_names_a_mismatch`: trades from `run_backtest`
    of lce_pct on the same synthetic bars pass. Altering one `stop_init` is reported.
  - `test_build_overlay_refuses_non_walkforward_changed_hash_and_mismatch` (Review Focus 5)
  - `test_build_overlay_end_to_end_on_a_small_synthetic_run`: a tmp cfg and data dir with 2 synthetic
    5m coins, a hand-made run dir (summary.json with 2 folds and 2 combos, and an `oos_trades.parquet`
    from in-process `run_backtest` of those combos over each fold's window), the real sandbox (mark
    `sandbox`). The test checks four things:
    - the files are written under the run's `overlay/` only;
    - `meta.json` has the fields;
    - the bars, zones and signals parquets have the columns and dtypes;
    - no holdout read happens: `load_bars` is called without a token, as `tests/test_trade_chart.py` checks.
- [ ] **Step 2:** Run `.venv/bin/python -m pytest -q tests/test_overlay_build.py`. Expected: FAIL.
- [ ] **Step 3:** Implement. Reruns use `client.spawn(cfg2, src)` → `w.bars(sym, "5m", load_bars(cfg, sym, "5m"))` → `w.run(params, EngineParams.from_config(cfg, sym), features=True).features`, with `cfg2` = cfg with `sandbox.max_output_mb = 512`. One worker per (coin, combo), bounded by `--workers` (default 4). Keep only the needed columns, then stitch, then self-check against the run's `oos_trades.parquet`, then write.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest -q tests/test_overlay_build.py tests/test_lce.py`. Expected: PASS.
- [ ] **Step 5:** Commit: "feat: lab overlay rebuilds a walk-forward run's LCE context for the trade chart".

### Task 3: The LCE layers and toggles in the dashboard

**Files:**
- Create: `dashboard/lce_overlay.py`
- Modify: `dashboard/trade_chart.py` (`trade_figure` draws the layers when `overlay` is given)
- Modify: `dashboard/app.py` (the toggle row and the overlay load, cached per (run, symbol))
- Test: `tests/test_lce_overlay.py`

**Interfaces:**
- Consumes:
  - Task 1's `trade_figure(..., overlay=, layers=)`.
  - Task 2's file layout. Read `meta.json` for each fold's `params` (`vp_days`, `ema_fast`,
    `ema_slow`, `zone`, `unit`).
- Produces:
  - `load_overlay(run_dir, symbol, start, end) -> Overlay | None`. `Overlay` is a dataclass with
    bars, zones, signals and folds. None when the files are absent.
  - `spread_profile(lo, hi, vol, lo_edge, hi_edge, rows) -> np.ndarray`: an exact even spread.
  - `session_pocs(candles, sday_series, days) -> np.ndarray`
  - `hvn_rows(p) -> np.ndarray`
  - `anchor_row(p, edges, pocs, hz) -> int`
  - `level_profile(candles_hist, day, vp_days) -> dict`, with keys `edges, p, hvn, pocs, anchor`.
  - `visible_profile(candles) -> dict`
  - `htf_cloud(close, ema_fast, ema_slow) -> pd.DataFrame`, with columns `fast, slow`.
  - `draw_layers(fig, overlay, candles, layers, palette, focus_trade=None) -> None`
  - `branch_name(k: int) -> str`
  - `DEFAULT_LAYERS = frozenset({"zones", "cloud5", "profile"})`

- [ ] **Step 1: Write the failing tests**
  - `test_spread_profile_matches_the_plugin_profile`: synthetic days, against `profile` from
    `load_class("lce_pct")`'s module, within 1e-9 relative.
  - `test_session_pocs_and_hvn_and_anchor_match_the_plugin` (synthetic days; the plugin's `poc`,
    `hvn` and its anchor rule)
  - `test_htf_cloud_matches_within_1e_6_after_the_lead_in`
  - `test_zone_bands_start_at_first_time_and_end_at_the_day_end`
  - `test_demand_below_the_days_first_open_supply_above`
  - `test_anchor_not_matched_note_appears_when_off_the_zones`
  - `test_default_layers_and_toggles_appear_only_with_overlay_data` (AppTest; Review Focus 4)
  - `test_branch_label_added_to_the_box_when_toggled`: "Branch 1 · dip into the cloud at demand"
  - `test_no_overlay_leaves_the_chart_unchanged`
- [ ] **Step 2:** Run `.venv/bin/python -m pytest -q tests/test_lce_overlay.py`. Expected: FAIL.
- [ ] **Step 3:** Implement. The level-building profile is computed for the focused trade's
  session day, otherwise the window's last day, using that day's fold `vp_days` and loading the
  extra history from the research store (`load_candles`, no token). Draw it as a horizontal
  histogram in a narrow right-hand area: an x-axis domain split, or shapes in paper coordinates.
  Session POCs are orange ticks with a faint horizontal line, HVNs are blue, and the anchor is
  labelled.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest -q tests/test_lce_overlay.py tests/test_trade_chart.py tests/test_dashboard.py`. Expected: PASS.
- [ ] **Step 5:** Run the full suite: `.venv/bin/python -m pytest -q`. Expected: 0 failures.
- [ ] **Step 6:** Commit: "feat(dashboard): the LCE overlay — zones, clouds, volume profiles and branches".

### Task 4: Map the LCE trials onto the new view (controller)

- [ ] **Step 1:** When trial 8's verdict exists, and after Tasks 1–3 pass the final review, run
  `.venv/bin/lab overlay <trial-8 run_id>` from `.worktrees/overlay`. Expected: it prints the
  `overlay/` directory, and the self-check passes. A mismatch is a bug to investigate, not to work
  around.
- [ ] **Step 2:** Do the same for trial 9 when it finishes.
- [ ] **Step 3:** Launch the dashboard from `.worktrees/overlay` (`.venv/bin/lab dashboard`) and
  check one coin's trades render with every layer. Take a Playwright screenshot of a focused trade
  for the user.
- [ ] **Step 4:** Push `feat/trade-overlay` and open the PR. The user merges.
