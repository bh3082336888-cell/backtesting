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
lab dashboard                         # Streamlit: strategies, detail, pairs, lineage
lab ingest                            # record any run dir not yet in the store
lab link PARENT_RUN CHILD_RUN --decision kill --reason "..."   # hand-made lineage edge
lab paper RUN_ID                      # only runs approved for paper in the dashboard
lab paper-refresh PAPER_RUN_ID        # recount regime changes → paper_eligible
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
