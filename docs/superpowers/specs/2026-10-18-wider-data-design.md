# Wider data: history from 2020 and 17 coins

Status: chosen in conversation 2026-10-18 (option B of three); both open questions answered by
the user the same day (see "Answered questions"); the document as a whole awaits the user's
review. Nothing is built yet.
Scope: the research data only: `history_start`, the coin universe, per-coin costs, and the fixes
the code needs for coins that start (and may stop) on different dates. It comes **before** the
Phase 3b-2 proposer and is not part of it; the proposer will run on this data.

## Why

The user wants the 3b-2 proposer to search for an edge at scale. Every strategy judged raises the
Deflated Sharpe bar (N counts store-wide), so the search is limited by how hard a real edge is to
tell apart from luck, not by compute. Two levers lower that difficulty without hiding trials:

- **A longer track lowers the bar.** With the judge's own formula (`perp_lab.judge.gates`, normal
  returns, DSR ≥ 0.95), the annualised OOS Sharpe needed:

  | strategies judged (N) | 4 years OOS (now) | 6 years OOS (this change) |
  |---|---|---|
  | 7 | 1.52 | 1.24 |
  | 50 | 1.96 | 1.60 |
  | 1,000 | 2.45 | 2.00 |

- **More coins give a real edge more independent bets**, so its portfolio Sharpe rises, while a
  lucky one gains nothing. Crypto coins are correlated, so this helps coin-specific signals (taker
  flow shocks) more than market-wide ones.

A "dead" verdict also means more on 6 years of 17 coins than on 4 years of 8, which matters for
the proposer's record of dead ideas.

## Decisions

| Question | Decision |
|---|---|
| Option | **B**: history from 2020 plus the 17 coins that stayed liquid throughout (user, 2026-10-18). Rejected: A (history only, same 8 coins) as leaving breadth on the table; C (a universe that follows volume month by month, dead coins included) as the biggest job, kept for later. |
| `history_start` | `2020-01-01`. Coins listed later start at their first bar. First test window 2020-07; folds go from 24 to 36. This reverses the judge spec's "History start stays 2022-01-01" (see "Changes to existing documents"). |
| `holdout_start` | Unchanged: `2026-07-01`. |
| Universe | The 17 coins below, chosen by the rule below. Fixed list, not re-chosen over time. |
| Survivor bias | Accepted and disclosed (user, choosing B knowing it). Every report and the dashboard say the universe was chosen in 2026 from coins still liquid then. |
| Costs | Per-coin, point-in-time slippage (`costs.slippage_model`, see "Slippage model") and per-coin margin brackets, fixed **before** any trial runs on the new data and never tuned after seeing a strategy result. |
| Earlier trials | Still count. N stays store-wide over every (plugin version, timeframe) pair, old data included. |

## The universe

**Rule** (applied 2026-10-18, every input read before 2026-07-01): a USDT perpetual on Binance
USDⓈ-M that was trading in every sampled month from 2021-01 to 2026-06 and ranked in the top 50 by
average daily quote volume in at least 5 of those 7 samples (2021-01, 2021-07, 2022-01, 2023-01,
2024-01, 2025-01, 2026-06). Ranks come from data.binance.vision monthly `1d` klines
(`quote_volume`). Non-crypto contracts (gold, silver, oil, stocks: e.g. XAU, XAG, CL, INTC, MU)
reach today's top 30 but were listed between 2025-12 and 2026-06, so they fail the "trading
since 2021" part.

**Result: 17 coins**, the current 8 plus 9 new ones:

| coin | first month in archive (1h klines and funding) | volume rank 2020-07 / 2021-01 / 2022-01 / 2024-01 / 2026-06 | avg daily volume 2026-06 |
|---|---|---|---|
| BTC | 2020-01 | 1 / 1 / 1 / 1 / 1 | $12.8B |
| ETH | 2020-01 | 2 / 2 / 2 / 2 / 2 | $9.0B |
| SOL | 2020-09 | – / 25 / 8 / 3 / 3 | $1.9B |
| BNB | 2020-02 | 14 / 15 / 14 / 17 / 17 | $439M |
| XRP | 2020-01 | 7 / 3 / 20 / 11 / 11 | $613M |
| DOGE | 2020-07 | 26 / 6 / 12 / 16 / 18 | $394M |
| AVAX | 2020-09 | – / 28 / 16 / 13 / 39 | $117M |
| LINK | 2020-01 | 3 / 4 / 9 / 21 / 45 | $102M |
| ADA (new) | 2020-01 | 4 / 10 / 7 / 25 / 26 | $196M |
| DOT (new) | 2020-08 | – / 7 / 13 / 34 / 65 | $67M |
| LTC (new) | 2020-01 | 11 / 5 / 25 / 30 / 63 | $68M |
| BCH (new) | 2020-01 | 8 / 8 / 46 / 41 / 44 | $106M |
| FIL (new) | 2020-10 | – / 38 / 31 / 23 / 42 | $109M |
| AAVE (new) | 2020-10 | – / 19 / 32 / 94 / 47 | $101M |
| XLM (new) | 2020-01 | 10 / 9 / 49 / 125 / 31 | $171M |
| ATOM (new) | 2020-02 | 16 / 30 / 5 / 47 / 165 | $17M |
| ETC (new) | 2020-01 | 17 / 23 / 40 / 20 / 128 | $23M |

ATOM and ETC have thinned out recently ($17–23M a day); per-coin costs must reflect that. 11 coins
trade from the first train window (2020-01/02); DOGE joins in 2020-07, and SOL, AVAX, DOT, FIL and
AAVE by 2020-10.

**Known facts about the archive** (checked 2026-10-18):
- Klines from 2020 have the same 12 columns as today's, taker-flow columns filled (no header row
  in early files; `vision._strip_header` already handles both).
- Funding files exist from each coin's first month.
- Open interest (`metrics`) starts later: BTC 2020-09-01, most coins 2021-12-01. Before that the
  column must load as NaN, never crash.
- After a delisting, the archive keeps publishing flat candles with zero trades (`count == 0`,
  zero volume; seen for WAVES and FTT). None of the 17 is delisted today, but the loader must treat
  such candles as "not trading" if one is later.

## Changes

File references are from a read-only check on main at e2fc3d2. Each change ships with a regression
test in the same PR (org rule), one concern per commit.

1. **Coins with no research bars** (small). A coin with no bars before `holdout_start` raises
   `FileNotFoundError` (`data/store.py:47`) or yields an empty frame, `daily_returns` then returns
   a RangeIndex series (`engine/metrics.py:15-16`) and the walk-forward's `_window` raises
   TypeError (`validate/walkforward.py:172-173`). Skip such a coin with a logged warning, and make
   the empty `daily_returns` carry a UTC DatetimeIndex.
2. **Backfill knows listing dates** (small). A month before listing (or after delisting) 404s on
   the monthly file, then ~30 daily files per timeframe, returns `"missing"` before `man.mark`
   (`data/vision.py:229-254`), and is fetched again on every backfill. Record such months as absent
   in the manifest (listing date from the archive's first file) and don't refetch them. Treat
   zero-trade candles after a coin's last real trade as absent too.
3. **Per-coin costs** (medium).
   - Slippage: today a flat `costs.slippage_bps: 2.0` for every coin and every date
     (`engine/backtest.py:51`). Replace with a point-in-time slippage per coin and bar, a monotone
     function of the coin's trailing liquidity (e.g. 30-day average daily quote volume from the
     klines already stored, ending before the bar). Floor: today's 2 bp, which stays the value for
     BTC/ETH-level liquidity. Thinner coins and thinner years (2020 volumes were a fraction of
     today's) pay more. The builder picks the function and its calibration source and documents
     both in this spec before any trial runs; when unsure, be pessimistic.
   - Margin: only the 8 current coins have brackets in `lab.yaml`; others default to mmr 1%
     (`config.py:48-50`). Add tier-1 brackets for the 9 new coins from Binance's public leverage
     information, rounding mmr up when unsure.
   - Paper keeps using live bid/ask (spread replaces slippage), unchanged.
4. **Gates scale with the number of coins** (small). `walkforward.min_train_trades: 20` (pooled,
   `validate/walkforward.py:230-231`) and `MIN_OOS_TRADES = 30` (`runs.py:16`) were set for 8 coins;
   with 17 they get weaker. Scale both with the number of coins listed in the window, keeping
   today's per-coin rate. *My call:* `judge.holdout_min_trades` (10) scales the same way, for the
   same reason; it only makes the exam stricter, never looser.
5. **Monte Carlo and random entry** (small). Monte Carlo divides by the full coin count
   (`validate/montecarlo.py:18`, `validate/pipeline.py:89`), understating drawdown when fewer coins
   are listed; use the per-day count. Random entry drops coins with 0 OOS trades
   (`validate/benchmarks.py:37`) while the strategy portfolio keeps them; make them consistent.
6. **Runs remember their data** (small). `latest_wf_run` matches plugin version, timeframe and
   universe but not the date span or costs (`store/db.py:389-403`). The 8-to-17 switch alone can't
   pair a child with an old parent run, since the universe must match, but a run restricted to the
   8 coins with `--symbols`, or any later cost change, could be compared with a parent run on other
   data over the overlapping days only (inner join, `judge/gates.py:84`). Record on every run the
   data identity (history start, universe, cost model version) and match on all of it, so the judge
   re-runs the parent under the family on the new data (it already does when no matching run
   exists).
7. **The exam with coins that stop trading: deferred, not built here** (user, 2026-10-18; see
   "Answered questions", 1). Recorded so the follow-up spec starts from it. Today `window_end` is the earliest last candle across all coins
   (`judge/exam.py:57-73`): a coin delisted before holdout + 180 days leaves the exam
   `pending`/`too_short` forever (`judge/judge.py:448-450`), one delisted later freezes the window
   at its delisting, and the per-day candle check (commit 33969fa) means one whole-day halt on any
   coin blocks the exam for good. Zero-trade flat candles inside the window (a halt, not a
   delisting) would be filled at their flat price. With 17 coins all of this is more likely.
8. **Paper and tail per coin** (small). `lab paper` starts one stream worker per coin for the
   whole run, plus three loops per coin, with no cap (`paper/runner.py:228, 292-294`): 17 workers ×
   3,072 MB is more than this machine's 30 GB (`client.max_workers` allows about 7). *My call:* in
   this project `lab paper` refuses to start (exit 2, a clear message) when the coins exceed
   `client.max_workers`, instead of running out of memory. Which real fix to pick (paper on a
   subset, smaller paper workers, or more memory) is decided when a strategy first passes an exam,
   since paper needs one and no exam can run before about 2026-12-28. A delisted coin's ccxt
   `BadSymbol` is not retried and fails the run (`data/live.py:39-42`), and `lab tail` stops at the
   first failing coin (`cli.py:61-65`): handle errors per coin.
9. **5m** (my call). Drop `5m` from `lab.yaml` `timeframes`. No plugin declares it, the research
   note shows 15m already can't pay costs, no 5m data is stored locally, and a walk-forward pool
   process caches 16 symbols' bars outside the sandbox's memory budget
   (`validate/walkforward.py:85`): about 15 GB at 5m over 6.5 years.
10. **Disclosure** (small). Walk-forward reports and the dashboard sidebar state the universe rule
    and the survivor bias in one line.
11. **Config and data.** `history_start: "2020-01-01"`, the 17-coin `universe`, the new brackets,
    then `lab backfill` for the 9 new coins and the 2020–2021 months of the 8 (about 400 MB in all;
    today's store is 142 MB). The holdout months of the new coins are downloaded like the others'
    and stay behind the existing guard.

## Slippage model (calibrated 2026-10-18)

**Formula.** Per bar, for a taker fill:

`slip_bps = min(50, 2.0 + 2.05 × (adv / 1e8) ** -0.63)`

The values live in `config/lab.yaml` `costs.slippage_model` (`adv_days: 30`, `spread_bps_at_100m: 2.05`, `alpha: 0.63`, `max_bps: 50.0`). The code is `perp_lab/engine/slippage.py`; the calibration is `docs/research/slippage-calibration/` (`calibrate.py`, `calibration.csv`).

**`adv` and the cap.** `adv` is the mean daily quote volume (USDT) over the 30 full UTC days before the bar's day, skipping days with no volume data. The cap `max_bps = 50` applies in two cases: the formula exceeds 50 (`adv` below about 1.3 million USDT a day), and `adv` is unknown or not above zero (a coin's first bars), where the bar pays 50 outright.

**Why the floor is additive.** The 2 bp floor is today's latency-and-impact buffer. BTC's quoted half-spread is about 0.02 bp, so the 2 bp was never spread. Adding each coin's quoted half-spread on top of the floor keeps BTC and ETH near 2 bp, makes thinner coins and thinner years pay more, and means no coin or date ever pays less than today.

**Decision of the user, 2026-10-18.** The pooled p90 curve puts BTC and ETH at about 2.1 bp (2.10 and 2.12 on 2026-06-30) rather than exactly 2.0, while their measured half-spreads are about 0.02 bp. The extra ~0.1 bp (about 5% above today's 2 bp) is accepted as the price of one pessimistic pooled curve. It is never kinder than today.

**Source.** data.binance.vision `futures/um/daily/bookTicker`, 17 coins (the universe) on 2023-06-15, 2023-09-15, 2023-12-15 and 2024-03-15: 68 coin-days. That dataset starts 2023-05-16. Every file, and every `adv` window (monthly `1d` klines), is dated before the holdout (2026-07-01); the script asserts it.

**Half-spread.** `(ask - bid) / (ask + bid)` in bp, time-weighted over the UTC day from the first quote until midnight, each quote weighing the time until the next one. Quotes with a bid not above zero or an ask not above the bid are dropped.

**Fit.** OLS of log half-spread on log `adv / 1e8` over the 68 points. `alpha` is minus the slope, rounded to 0.01 (0.63). The intercept is then lifted so that the curve lies above 90% of the points: the 90th percentile of the residuals (computed with the rounded `alpha`), rounded up to 0.01 bp (2.05 at 100 million USDT a day). The fit is pessimistic on purpose.

**Result** (`slippage_bps`, floor 2.0, a bar on the date; adv from the 30 days before it; "–" = no volume yet, so the bar pays 50):

| coin | 2020-08-01 | 2022-02-01 | 2026-06-30 |
|---|---|---|---|
| BTC | 2.33 | 2.09 | 2.10 |
| ETH | 2.82 | 2.16 | 2.12 |
| SOL | – | 2.49 | 2.33 |
| BNB | 5.82 | 2.66 | 2.76 |
| XRP | 4.45 | 2.83 | 2.65 |
| DOGE | 9.55 | 2.62 | 2.87 |
| AVAX | – | 2.76 | 3.86 |
| LINK | 3.21 | 2.55 | 4.02 |
| ADA | 3.67 | 2.48 | 3.34 |
| DOT | – | 2.65 | 4.63 |
| LTC | 5.00 | 3.03 | 4.61 |
| BCH | 4.51 | 3.96 | 3.98 |
| FIL | – | 3.43 | 3.94 |
| AAVE | – | 3.46 | 4.07 |
| XLM | 4.82 | 4.00 | 3.36 |
| ATOM | 6.75 | 2.43 | 8.29 |
| ETC | 6.98 | 3.72 | 7.18 |

Median measured half-spread per coin (four days) against the model's spread term at the median `adv`, in bp. The curve sits above most coins, and below a few (ADA, XRP); that is what a pooled p90 fit does.

| coin | measured | model | coin | measured | model |
|---|---|---|---|---|---|
| BTC | 0.016 | 0.103 | LTC | 0.688 | 0.981 |
| ETH | 0.026 | 0.165 | BCH | 0.227 | 1.123 |
| SOL | 0.171 | 0.383 | FIL | 1.219 | 1.545 |
| BNB | 0.207 | 0.781 | AAVE | 0.698 | 3.225 |
| XRP | 0.902 | 0.585 | XLM | 0.414 | 2.902 |
| DOGE | 0.665 | 0.699 | ATOM | 0.517 | 1.679 |
| AVAX | 0.310 | 0.927 | ETC | 0.286 | 1.703 |
| LINK | 0.573 | 1.050 | ADA | 1.361 | 0.870 |
| DOT | 0.912 | 1.462 | | | |

**Limits.**
- 2020 is extrapolated by volume alone: the data starts in 2023, and a 2020 bar with a given `adv` pays what a 2023 coin of that volume paid.
- A spread in bp also depends on tick size relative to price, which the model ignores (ADA and XRP sit above the curve for that reason).
- The model applies to taker fills only (market entries and exits, stop-market). Targets stay maker and pay no slippage.
- The model is fixed before any trial on the new data. Changing any of its values changes the `costs` section and so every run's data identity (Task 8), and no run on the old model is paired with one on a new model.

## What happens to existing results

- The 6 judged baselines keep their judgements and verdicts (on 2022–2026, 8 coins). `lab judge`
  returns an existing judgement for a (plugin version, timeframe) pair rather than re-judging it.
- Their trials still count in N.
- Optional, after this lands: re-measure them with `lab walkforward` on the new data. Same
  (plugin version, timeframe) pairs, so N doesn't change. Useful for the proposer's record of dead
  ideas; the user's call.
- `flow_shock_fade` has never been run; see open question 2.

## Non-goals

- A universe that changes over time (option C), coins that died, non-crypto contracts.
- The 3b-2 proposer and its record of ideas (next project).
- Screening many strategies and confirming few on fresh data (changes the judge; later, if ever).
- Any change to `holdout_start`, the holdout budget, the Deflated Sharpe gate or the sandbox.

## Anti-goals (failure even if it works)

- Any holdout read (on or after 2026-07-01) during universe selection, cost calibration or tests.
- Costs kinder than today's for any coin or date, or costs adjusted after a trial's result is
  seen.
- Survivor bias that isn't stated where results are shown.
- An exam that can be unblocked by excluding a coin because of how it did in the holdout.

## Order of work

Agreed with the user 2026-10-18:

1. This project (changes 1–6 and 8–11; per-coin costs and run identity before any walk-forward on
   the new data).
2. A dated addendum to `docs/research/2026-10-18-strategy-research.md` re-deriving its
   data-dependent numbers on the wider data (answered question 2).
3. `lab judge flow_shock_fade --tf 1h` on the wider data: trial 7.
4. Phase 3b-2, the proposer.

In parallel, before the first exam (about 2026-12-28): the small spec for change 7.

## Answered questions

Both answered by the user 2026-10-18: "yes to both, go with your recommendations".

1. **Is change 7 in this project?** **No, deferred** to its own small spec, decided
   before the first exam (no exam can run before about 2026-12-28, and none of the 17 coins is
   delisted today). Until then today's exam guards stay exactly as merged. Any rule for it touches
   when a holdout read is spent, and a family gets one read, so a wrong rule can't be undone. The
   direction we'd propose then: treat a coin that stops trading as the walk-forward does (its open
   trade closes at its last real trade, it drops out of the portfolio average, the window ends at
   the earliest last candle among coins still trading), never drop a coin from the exam because of
   how it did in the holdout, and decide separately whether a whole-day gap in a month the archive
   has published complete (a real exchange halt) should still block. That last part would loosen
   the guard from commit 33969fa for all 17 coins, so it needs an explicit yes.
2. **When to run `flow_shock_fade` (trial 7).** **After this lands, on the wider data.**
   The bar drops (1.24 instead of 1.52 at N = 7), and its plugin and kill rules don't depend on the
   coin list. But pre-registration only counts before a result is seen, so before that run, a dated
   addendum to its research note re-derives the numbers that depend on the data (the "600–3,000 OOS
   trades" expectation, and a 2020–21 counterpart to the "profit only from the 2022 crash months"
   check), using the same kind of looks as the note (counts and distributions, no forward returns).
   That addendum is a small task after this project, not part of it. The alternative, running it
   before this lands, means waiting for that trial before switching the config (change 11).

## Left to the builder

- The slippage function and its calibration (change 3), within the floor and the
  fixed-before-trials rule.
- Where the data identity lives (change 6): new `runs` columns or a hash.
- How listing and delisting dates are stored (change 2).
- Exact scaling for the trade-count gates (change 4).
- Order of the work, provided per-coin costs (3) and run identity (6) are in place before any
  walk-forward runs on the new data.

## Changes to existing documents

- `docs/superpowers/specs/2026-10-16-judge-design.md`, Decisions table, "Short holdout": "History
  start stays 2022-01-01" becomes "History start moved to 2020-01-01 by
  `2026-10-18-wider-data-design.md`".
- `README.md`: universe, history start, per-coin costs, the survivor-bias note, the data gaps list.
- `docs/research/2026-10-18-strategy-research.md`: a dated note that its numbers are for 8 coins
  from 2022.
