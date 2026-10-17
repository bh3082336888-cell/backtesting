# Strategy research pass: implementation plan

Branch `feat/strategy-research` off `main`. Part B of the 2026-10-18 work: a research pass done by
us (there is no outside brief), ending in one new strategy plugin that is implemented and tested but
**not** run through walk-forward. Running it spends a trial, so that stays the user's call.

## Global Constraints

1. **Holdout integrity (hard).** The holdout starts 2026-07-01 and nothing in this work may see it.
   - No internet at all: no WebSearch, no WebFetch, no curl, no pip downloads. Anything published
     after 2026-07-01 could describe holdout-period markets.
   - Read market data only through `perp_lab.data.store` (`load_bars`, `load_candles`,
     `load_funding`) with the research config
     `/tmp/claude-1000/-home-burhan-backtesting/464fe316-f53a-4226-89ab-e02f2b7426f3/scratchpad/research/lab.yaml`
     (it points `data_dir` at the main checkout's data, read-only). Never pass a holdout token, never
     pass an `end` after 2026-07-01, and never open a data file directly.
   - Don't read or touch `runs/holdout_log.jsonl`, and don't call anything in `perp_lab/judge/budget.py`
     or `perp_lab/holdout.py`.
2. **No hidden trials.** Never measure the forward returns, P&L or hit rate of a candidate signal on
   the data. That is a backtest, and it belongs to the walk-forward, where it is counted in N. Looks at
   the data are limited to feasibility: how often a condition occurs, feature distributions, bar ranges
   against costs, funding regimes, data gaps. Every look goes in the research doc's "Data looks"
   appendix: what was computed and why it isn't a performance measurement. Don't run `lab backtest`,
   `lab walkforward`, `lab judge`, `lab exam` or `lab paper`.
3. **What a plugin can do (architecture facts).**
   - One symbol at a time. A plugin sees only its own symbol's bars, so cross-sectional ideas
     (ranking coins, BTC as a reference series, pairs) are impossible today.
   - Bar columns: `open, high, low, close, volume, quote_volume, trades, taker_buy_volume,
     taker_buy_quote_volume, open_time, close_time, funding_charge` (the rate settled inside the bar),
     `funding_last` (last settled rate known at the bar's close), `funding_interval_h`, and `regime`
     (`trending | ranging | high_vol | transition`, from 4h bars). `oi` and `oi_value` are all NaN
     locally because OI metrics aren't backfilled, so a strategy needing OI would first need
     `lab backfill` of metrics. Prefer ideas that don't.
   - Local timeframes: 15m, 1h and 4h for BTC, ETH, SOL, BNB, XRP, DOGE, AVAX and LINK (USDT perps)
     from 2022-01-01. No 5m. The research span ends 2026-07-01.
   - Engine: the signal is taken at bar t's close and filled at t+1's open. One position per symbol;
     isolated margin at 3x; each trade risks 1% of equity, sized off the stop. Every entry needs a
     stop. A stop only ratchets (`manage` may tighten it). A target is optional, filled as a maker
     limit only when price trades through it. If stop and target are both inside a bar, the stop wins.
     Each symbol is an independent book with equal capital.
   - Costs: taker 5 bp plus 2 bp slippage per side on market fills and stops; maker 2 bp on targets;
     funding charged at real settlement times.
   - Plugin contract: see `perp_lab/strategy.py` and the plugins in `strategies/`. `params`, `grid`,
     `timeframes` and an optional module-level `PARENT` must be literals. `prepare` must be causal
     (enforced by `tests/test_lookahead.py`). Imports are numpy, pandas and `perp_lab` only.
4. **What a candidate must survive** (`perp_lab/runs.py::verdict`, `perp_lab/judge/`, README):
   OOS Sharpe > 0; beat the p95 of random-entry OOS Sharpe, where the random entries use **the same
   exits** (bracket and manage) at matched frequency, so the edge must be in entry timing or
   direction, not exits; ≥ 30 OOS trades; no narrow parameter island (Sharpe falling more than 50% or
   flipping sign at ±20% moves); a Deflated Sharpe ≥ 0.95 with N = distinct (plugin version,
   timeframe) walk-forward runs store-wide (N = 6 today, about 1.5 annualised OOS Sharpe needed over
   4 years); then one holdout exam per family, needing ≥ 10 holdout trades, a positive result, and no
   worse than the walk-forward p5.
5. **What has been tried (all killed).** Walk-forward OOS, 2022-07 → 2026-07, 8 symbols:

   | plugin | tf | OOS Sharpe | IS Sharpe | trades | gross / fees / funding (USDT) | vs random | note |
   |---|---|---|---|---|---|---|---|
   | donchian_atr (breakout + ATR chandelier) | 4h | 0.45 | 1.21 | 1390 | +13,060 / 3,102 / 1,159 | p48 | high_vol regime avg R −0.135 |
   | donchian_atr_wide | 4h | 0.81 | 1.16 | 1160 | +17,449 / 1,814 / 1,314 | p77 | trending avg R +0.283 |
   | donchian_atr | 1h | −0.46 | 0.63 | 4793 | −372 / 15,362 / 1,669 | p49 | costs ate it |
   | bb_vwap_reversion (BB + RSI → VWAP) | 15m | −4.30 | −4.02 | 8435 | −21,466 / 26,366 / −165 | — | costs ×2 the gross loss |
   | bb_vwap_reversion | 1h | −2.18 | −1.83 | 3759 | −21,548 / 10,584 / −621 | p72 | |
   | funding_fade (fade extreme funding z) | 1h | −0.30 | 1.35 | 542 | −621 / 1,293 / −572 | p68 | narrow island; IS overfit |

   Buy and hold over the same span: Sharpe 0.59, max drawdown −67%. The trend baselines made money,
   but no better than random entries with the same trailing exit. Their profit came from the exit,
   not the timing.
6. **Trial budget.** The chosen plugin targets exactly one timeframe (`timeframes` has one entry). It
   has a small grid (at most 3 parameters with at most 3 values each, at most 12 combinations) and
   wide, round parameter values chosen from the mechanism, not the data. Each extra timeframe is
   another trial.
7. **Honest sources.** Cite only work you are confident exists, mark citations as "from memory", and
   never invent a paper, author, year or number. Prefer explaining the mechanism (who is on the other
   side, and why the edge persists) over appeals to authority.
8. Environment: work only in `/home/burhan/backtesting-research`, plus scratch scripts under
   `/tmp/claude-1000/-home-burhan-backtesting/464fe316-f53a-4226-89ab-e02f2b7426f3/scratchpad/research/`.
   Never run anything in `/home/burhan/backtesting`, where another session works. Python is
   `/home/burhan/backtesting/.venv/bin/python`. Run it from the worktree so `perp_lab` resolves to the
   worktree, and run tests with `-m pytest -p no:cacheprovider -m "not sandbox"`: real-sandbox tests
   can't confine a worker from a second worktree.
9. Commits: conventional subject, prose body, one concern per commit, ending with
   ```
   Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
   Claude-Session: https://claude.ai/code/session_01JATenKtwepruP3BXHdYHpL
   ```
   Never amend; never push.

## Task 1: Research and choose one hypothesis

Deliverable: `docs/research/2026-10-18-strategy-research.md`, committed as `docs(research): ...`. No
code is committed in this task; scratch analysis scripts stay in the scratch directory.

The document must contain, in this order:
1. **What the six baselines teach**, from constraint 5. Be specific about costs per timeframe,
   exit-driven versus entry-driven profit, regime behaviour, and in-sample overfitting.
2. **Cost hurdle by timeframe.** The round-trip cost in bp for a stop exit and for a target exit,
   against the typical bar range and the typical move over a 1–20 bar holding period at 15m, 1h and
   4h (feasibility looks; constraint 2). This says which timeframes can pay their costs at all.
3. **Candidates.** 4–6 hypotheses that fit constraint 3. Give each:
   - the mechanism and who pays for the edge;
   - why it should persist;
   - why the entry timing (not the exit) carries it, so it can beat random entry;
   - expected trade frequency, and whether ≥ 30 OOS trades is realistic;
   - the data it needs;
   - how it most likely fails.
   At least one must use something none of the six baselines used (for example taker buy/sell
   imbalance, volume or trade-count shocks, funding as a regime filter rather than a trigger, time of
   day or week, or volatility compression).
4. **Choice.** The one to implement and why it beats the others on expected survival of every gate
   in constraint 4. If none is likely to survive, still choose the best one and say so plainly. Name
   the runner-up.
5. **Plugin spec** for the choice, precise enough to implement without questions:
   - file name;
   - the one-sentence hypothesis (the module docstring's first line);
   - `PARENT` (or "none, new family"), with the reason;
   - `params` with defaults, `grid`, and `timeframes` (one entry);
   - every feature in `prepare`, with its exact causal formula, warmup behaviour and NaN handling;
   - `entry` rules for long and short;
   - `bracket` (the stop formula, and the target formula or none);
   - `manage` (the trailing rule, time exit or none);
   - expected trades per symbol over 2022-07 → 2026-07.
6. **What would kill it**: the walk-forward and judge outcomes that should end this line of
   research rather than lead to a tweaked child.
7. **Data looks** appendix (constraint 2).

## Task 2: Implement the chosen plugin

Implement the plugin exactly as section 5 of `docs/research/2026-10-18-strategy-research.md`
specifies. Files: `strategies/<name>.py`, plus any new indicator in `perp_lab/indicators.py` (only if
it is reusable; otherwise keep the helper in the plugin), and `tests/test_<name>.py`. One commit for
the plugin with its tests.

- Follow the existing plugins' shape (`strategies/donchian_atr.py`, `strategies/funding_fade.py`).
  The module docstring's first line is the hypothesis sentence from the spec. The other module
  docstring lines summarise the rules.
- Tests, written first:
  - On small synthetic bars (helpers in `tests/conftest.py`), each entry rule fires exactly where the
    spec says, and nowhere else.
  - `bracket` returns the specified stop and target for both sides.
  - `manage` behaves as specified (trail, time exit).
  - Warmup rows produce no signal.
  - `tests/test_lookahead.py` picks the plugin up automatically and must pass on it (check it does).
- `lab strategies` lists the plugin with its hypothesis: run
  `/home/burhan/backtesting/.venv/bin/python -m perp_lab.cli strategies` from the worktree.
- If any part of the spec is ambiguous or impossible in the engine, stop and report NEEDS_CONTEXT.
  Don't improvise a rule.
