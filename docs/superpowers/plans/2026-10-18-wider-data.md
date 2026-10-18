# Wider data (history from 2020, 17 coins): implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Widen the research data to history from 2020-01-01 and the 17 coins the spec chose. Do it without making any cost kinder, without reading the holdout, and without letting a run be compared with a run made on other data.

**Architecture:** The data and engine changes come first: coins that start late, listing dates in the backfill, and per-coin, point-in-time slippage carried on every bar as a `slippage` column. Then the validation fixes: gates that scale with the number of coins, Monte Carlo and random entry, and a data identity recorded on every run and matched by the judge. Then paper and tail errors per coin, and the survivor-bias disclosure. The config switches to 17 coins from 2020 only at the end, followed by the backfill. No plugin runs on real data in this project.

**Tech Stack:** Python ≥ 3.11, pandas, numpy, pyarrow, SQLAlchemy Core (SQLite), typer, Streamlit, ccxt, pytest. Data comes from data.binance.vision.

**Spec:** `docs/superpowers/specs/2026-10-18-wider-data-design.md` (approved by the user 2026-10-18; read it before any task).

## Global Constraints

Values are copied from the spec. Every task's requirements include these.

1. `history_start: "2020-01-01"`. `holdout_start: "2026-07-01"` does not change.
2. The universe is exactly `[BTC, ETH, SOL, BNB, XRP, DOGE, AVAX, LINK, ADA, DOT, LTC, BCH, FIL, AAVE, XLM, ATOM, ETC]`, in this order. It is a fixed list, never re-chosen.
3. **No holdout read** (any date on or after 2026-07-01) during cost calibration or tests. Tests use synthetic markets (`tests/conftest.py::market_config`). Nothing in this project drops a coin because of holdout data.
4. **Costs are never kinder than today's** for any coin or date. Slippage never falls below today's 2 bp (`costs.slippage_bps: 2.0`, the floor). No margin bracket goes below its current value or below the current default (mmr 0.010). Costs are fixed before any trial runs on the new data and are never tuned after a strategy result is seen.
5. **No plugin runs on real data in this project.** Never run `lab walkforward`, `lab judge`, `lab exam`, `lab backtest` or `lab paper` against `./data`. `flow_shock_fade` stays unrun: its trial 7 comes after the research-note addendum, which is outside this project.
6. **Change 7 (an exam with coins that stop trading) is not built.** `judge/exam.py::window_end`, `window_data_problems` and the `too_short` rule stay exactly as merged.
7. No change to `holdout_start`, the holdout budget, the Deflated Sharpe gate or the sandbox. N still counts every (plugin version, timeframe) pair store-wide, old data included.
8. The gates keep today's per-coin rates: `min_train_trades` 20 / 8 coins = 2.5, `MIN_OOS_TRADES` 30 / 8 = 3.75, `holdout_min_trades` 10 / 8 = 1.25. The exam minimum never drops below 10.
9. Survivor bias is stated wherever results are shown: walk-forward reports and the dashboard sidebar.
10. Org rules: every new behaviour ships with a regression test in the same commit; one concern per commit; never amend; no secrets in code, tests or docs. Push and open the PR only when the user asks.
11. Branch: `feat/wider-data`, created from `spec/wider-data` so the spec and this plan travel with the code. Work in the main checkout `/home/burhan/backtesting`: real-sandbox tests can't confine a worker started from a second worktree. Run tests with `.venv/bin/python -m pytest -p no:cacheprovider` (add `-m "not sandbox"` for a quick run; finish each task with the full suite).
12. Commits use a conventional subject (`feat(data): ...`) and a prose body, and end with the attribution lines this session's system reminder gives.
13. Leave `docs/research/2026-10-18-looks/lab.yaml` alone. It records the config those research looks ran on.

## Decisions this plan makes

The spec's "Left to the builder", plus three calls the spec implies:

- **Slippage function.** Per bar: `slip_bps = min(max_bps, slippage_bps + spread_bps_at_100m × (adv / 1e8) ** -alpha)`. `adv` is the mean daily quote volume (USDT) over the 30 full UTC days before the bar's day, skipping days with no volume data. When `adv` is unknown or ≤ 0, the bar pays `max_bps = 50`. The 2 bp floor stays as today's latency-and-impact buffer (BTC's spread is ~0.02 bp, so the 2 bp was never spread), and each coin's quoted half-spread is added on top. This keeps BTC/ETH at 2 bp to within a few hundredths. Thinner coins and thinner years pay more, and no coin or date pays less than today.
- **Calibration source.** data.binance.vision `futures/um/daily/bookTicker` (published 2023-05-16 to at least 2024-03-30, all before the holdout), for the 17 coins on 2023-06-15, 2023-09-15, 2023-12-15 and 2024-03-15. The half-spread is time-weighted over each UTC day. `adv` for those days comes from monthly `1d` klines. The fit is OLS of log half-spread on log `adv`; `alpha` is the slope, rounded to 0.01. The intercept is then lifted to the 90th percentile of the residuals (computed with the rounded `alpha`) and rounded up to 0.01 bp. The method and the fitted values go into the spec before any trial (Task 4).
- **Data identity.** `runs.data_identity(cfg, symbols)` = {history start, universe, cost model version, `costs` section, margin brackets of those coins}. Its hash is `data_id`, 16 hex characters of a sha256 over canonical JSON. It is stored in a new nullable `runs.data_id` column, with the full identity in `summary.json`. `write_run` stamps it on every run. The store adds the column to an existing database on first open, and old runs get NULL, so they are never paired with a new run.
- **Listing dates.** One listing request to the archive's S3 index per (coin, dataset, timeframe) finds the first file. Months before it are written to the manifest as `{"complete": true, "source": "absent", "rows": 0}` and never fetched again. Zero-trade candles after a coin's last real trade are dropped by the loader rather than marked in the manifest, because one month can hold both real and zero-trade candles.
- **Gate scaling.** `ceil(rate × coins listed in the window)`, where a coin is listed in a window if it has at least one daily return in it. The rates are 2.5 (train), 3.75 (OOS) and 1.25 (holdout, floored at 10).
- **Margin brackets for the 9 new coins.** One bracket per coin with cum 0. Its mmr is the highest maintenance rate Binance lists for any notional up to the initial equity × leverage (10,000 × 3 = 30,000 USDT), read 2026-10-18 from the public endpoint `https://www.binance.com/bapi/futures/v1/public/future/common/brackets`. Binance's own tier 1 caps at 10,000 USDT for most of them, so the spec's "tier 1" (rounded up when unsure) is read as the bracket our positions actually reach. Values: ADA 0.010, DOT 0.010, LTC 0.010, BCH 0.010, FIL 0.015, AAVE 0.010, XLM 0.015, ATOM 0.010, ETC 0.010. The existing 8 stay as they are, since each is already at or above the same rule's value.
- **Backfill scope.** `lab backfill --datasets klines,fundingRate`. Today's store has no `metrics` (OI), and the spec's ~400 MB estimate excludes it. OI keeps loading as NaN.

## Review Focus

The five inputs most likely to bite that the spec implies but doesn't spell out. Each has a test in the task named.

1. **A coin listed inside an early fold** (DOGE in 2020-07; SOL, AVAX, DOT, FIL and AAVE by 2020-10): no crash; the portfolio averages only the coins trading that day; the train gate counts only coins listed in the fold; the coin's first day pays `max_bps`. Tests are in Tasks 1, 5 and 6.
2. **Today's `runs/lab.db`**, which holds the 6 judged baselines, opened by the new code: the `data_id` column is added in place, old runs keep NULL and are never reused as a parent, and N is unchanged. Tests are in Task 8.
3. **Bars with unknown volume** (ccxt-stitched candles, synthetic bars, zero-volume days): slippage is `max_bps`, never NaN. A NaN slippage would make a fill NaN, and the replay would then kill an innocent plugin as a `mismatch`. Tests are in Tasks 3 and 5.
4. **The archive listing unreachable, throttled or garbled during a backfill**: no month is marked absent, and the backfill behaves exactly as today. Tests are in Task 2.
5. **The same config read twice**, with keys reordered or `2` written for `2.0`: the same `data_id`. A changed cost or history start: a different one. Tests are in Task 8.

## File map

| File | Change |
|---|---|
| `perp_lab/engine/metrics.py` | empty `daily_returns` carries a UTC DatetimeIndex (T1) |
| `perp_lab/data/store.py` | `research_symbols` (T1); drop trailing zero-trade candles (T2); `attach_slippage`, slippage in `load_bars` (T5) |
| `perp_lab/data/vision.py` | `first_archive_month`, absent months, `Job.first` (T2) |
| `perp_lab/engine/slippage.py` (new) | cost model version, trailing ADV, slippage curve, half-spread, fit (T3) |
| `docs/research/slippage-calibration/calibrate.py`, `calibration.csv` (new) | the calibration run and its data (T4) |
| `perp_lab/engine/backtest.py` | per-bar `slippage` column, `ep.slippage` as fallback (T5) |
| `perp_lab/runs.py` | per-coin gate minimums, `verdict(..., n_coins)` (T6); `data_identity`, `data_id`, stamped in `write_run` (T8) |
| `perp_lab/validate/walkforward.py` | skip coins without research bars (T1); `listed_coins`, scaled train gate (T6) |
| `perp_lab/validate/montecarlo.py`, `benchmarks.py`, `pipeline.py` | per-day coin count, zero-trade coins in random entry (T7); `n_coins` to `verdict` (T6); universe line in the report (T10) |
| `perp_lab/judge/judge.py` | `holdout_min_trades` (T6); match the parent run by `data_id` (T8) |
| `perp_lab/store/db.py` | `runs.data_id` column, migration, `latest_wf_run(..., data_id=)` (T8) |
| `perp_lab/paper/runner.py`, `perp_lab/cli.py` | drop a delisted coin per coin; paper refuses above `client.max_workers`; tail carries on per coin (T9); `backtest` skips coins (T1); paper stamps data identity (T8) |
| `dashboard/app.py` | universe rule in the sidebar (T10) |
| `config/lab.yaml` | slippage model (T4); gate rates (T6); `universe_rule` (T10); brackets, no 5m, history and universe (T11) |
| `tests/…` | new: `test_late_listing.py`, `test_backfill_listing.py`, `test_slippage.py`, `test_coin_gates.py`, `test_benchmarks.py`, `test_data_identity.py`, `test_config.py`; additions to `test_engine_costs.py`, `test_judgements_store.py`, `test_judge.py`, `test_paper.py`, `test_walkforward_sandbox.py`, `test_dashboard.py` |
| `README.md`, judge spec, research note, this spec | documents (T4, T12) |

---

### Task 1: Coins with no research bars (spec change 1)

**Files:**
- Modify: `perp_lab/engine/metrics.py:13-21` (`daily_returns`)
- Modify: `perp_lab/data/store.py` (add `research_symbols`)
- Modify: `perp_lab/validate/walkforward.py:206-213` (`walkforward`)
- Modify: `perp_lab/cli.py:96` (`backtest` loop)
- Test: `tests/test_late_listing.py` (new)

**Interfaces:**
- Produces: `research_symbols(cfg, symbols: list[str], tf: str) -> list[str]` in `perp_lab/data/store.py`. It keeps the given order and drops each coin that has no `tf` directory, no parquet file, or no candle with `open_time` in `[history_start, holdout_start)`, logging `log.warning("%s %s: no research bars before %s; skipped", symbol, tf, holdout_start.date())` for each. It reads the `open_time` column only.
- Produces: `daily_returns` of an empty curve is `pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))`.
- Produces: `walkforward` works on `research_symbols(cfg, symbols or cfg.symbols, tf)` and raises `ValueError(f"no coin has {tf} research bars; run `lab backfill`")` when none is left, so `WFResult.symbols` holds only coins that ran. `lab backtest` loops over the same filtered list.

- [ ] **Step 1: Write the failing tests** in `tests/test_late_listing.py`

```python
import logging
import shutil

import numpy as np
import pandas as pd

from perp_lab.data.store import load_bars, research_symbols
from perp_lab.engine.metrics import daily_returns
from perp_lab.plugins import PluginSource
from perp_lab.validate import walkforward as wfm
from tests.conftest import market_config
from tests.test_walkforward_sandbox import DONCHIAN

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def keep_from(cfg, symbol, tf, since):
    """Drop a coin's candles before `since`, as if it were listed then."""
    p = cfg.data_dir / symbol / tf / "all.parquet"
    df = pd.read_parquet(p)
    df[df.open_time >= T(since).value // 10**6].to_parquet(p)


def test_daily_returns_of_an_empty_curve_has_a_utc_index():
    r = daily_returns(np.array([]), np.array([], dtype="int64"))
    assert r.empty and isinstance(r.index, pd.DatetimeIndex) and str(r.index.tz) == "UTC"
    assert r[r.index >= T("2025-01-01")].empty            # what walkforward._window does


def test_research_symbols_skips_coins_without_research_bars(tmp_path, caplog):
    cfg = market_config(tmp_path, universe=["BTC", "ETH", "SOL"])
    keep_from(cfg, "ETHUSDT", "4h", "2026-07-01")         # holdout-dated bars only
    shutil.rmtree(cfg.data_dir / "SOLUSDT" / "4h")        # never backfilled
    with caplog.at_level(logging.WARNING):
        assert research_symbols(cfg, cfg.symbols, "4h") == ["BTCUSDT"]
    assert "ETHUSDT 4h" in caplog.text and "SOLUSDT 4h" in caplog.text


def test_walkforward_skips_a_coin_without_research_bars(tmp_path):
    cfg = market_config(tmp_path, universe=["BTC", "ETH", "SOL"])
    keep_from(cfg, "SOLUSDT", "4h", "2026-07-01")
    wf = wfm.walkforward(cfg, PluginSource.from_text("donchian_atr", DONCHIAN), "4h", workers=1)
    assert wf.symbols == ["BTCUSDT", "ETHUSDT"] and set(wf.oos_daily) == {"BTCUSDT", "ETHUSDT"}


def test_walkforward_runs_with_a_coin_listed_mid_span(tmp_path):
    cfg = market_config(tmp_path, universe=["BTC", "ETH", "SOL"])
    keep_from(cfg, "SOLUSDT", "4h", "2025-10-01")         # after fold 0's test window (2025-07..09)
    wf = wfm.walkforward(cfg, PluginSource.from_text("donchian_atr", DONCHIAN), "4h", workers=1)
    assert wf.oos_daily["SOLUSDT"].index.min() >= T("2025-10-01")
    f0 = wf.folds[0]
    p = wf.oos_portfolio[(wf.oos_portfolio.index >= f0["test_start"]) & (wf.oos_portfolio.index < f0["test_end"])]
    both = (wf.oos_daily["BTCUSDT"] + wf.oos_daily["ETHUSDT"]) / 2
    pd.testing.assert_series_equal(p, both.loc[p.index], check_names=False)


def test_oi_is_nan_before_the_first_metrics_snapshot(tmp_path):
    cfg = market_config(tmp_path)
    d = cfg.data_dir / "BTCUSDT" / "metrics"
    d.mkdir()
    ct = pd.date_range("2025-06-01", "2026-07-01", freq="1h", tz="UTC", inclusive="left").as_unit("ms").asi8
    pd.DataFrame({"create_time": ct, "oi": 1.0, "oi_value": 2.0}).to_parquet(d / "all.parquet")
    b = load_bars(cfg, "BTCUSDT", "4h")
    assert b.oi[b.index < T("2025-05-31")].isna().all() and (b.oi[b.index >= T("2025-06-01")] == 1.0).all()
```

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_late_listing.py -v`
Expected: FAIL. `research_symbols` can't be imported, and the empty `daily_returns` has a RangeIndex. `test_oi_is_nan_before_the_first_metrics_snapshot` may already pass. It pins a known fact from the spec, so keep it.

- [ ] **Step 3: Implement** `research_symbols` in `perp_lab/data/store.py`, the empty-index fix in `daily_returns`, the filter at the top of `walkforward` (before `jobs` is built), and the same filter in `cli.backtest`'s loop.

- [ ] **Step 4: Run the tests and the suite**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_late_listing.py -v` then the full suite
Expected: all PASS.

- [ ] **Step 5: Commit** (two commits: the empty-index fix with its test, then the coin skip with its tests)

```bash
git add perp_lab/engine/metrics.py tests/test_late_listing.py
git commit   # fix(engine): an empty equity curve's daily returns keep a UTC index
git add perp_lab/data/store.py perp_lab/validate/walkforward.py perp_lab/cli.py tests/test_late_listing.py
git commit   # fix(validate): skip a coin with no research bars instead of crashing
```

---

### Task 2: Backfill knows listing dates; zero-trade tails load as absent (spec change 2)

**Files:**
- Modify: `perp_lab/data/vision.py` (`Job`, `run_job`, `backfill`, new `first_archive_month`)
- Modify: `perp_lab/data/store.py:41-60` (`load_candles`)
- Test: `tests/test_backfill_listing.py` (new)

**Interfaces:**
- Produces: `LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"`. `first_archive_month(symbol: str, dataset: str, tf: str | None, max_retries: int) -> pd.Period | None` makes one `_get` of `f"{LIST_URL}?delimiter=/&prefix={prefix}"`. The prefix is `data/futures/um/monthly/klines/{SYMBOL}/{tf}/`, `data/futures/um/monthly/fundingRate/{SYMBOL}/` or `data/futures/um/daily/metrics/{SYMBOL}/`. It returns the earliest `YYYY-MM` among `<Key>`s ending in `.zip` (S3 lists in ascending order, so the first page holds it). It returns `None`, with a logged warning, when `_get` returns None or raises `FetchError`, when the body isn't parseable XML, or when it holds no such key.
- Produces: `Job` gains `first: pd.Period | None = None`. In `run_job`, right after the `is_complete` skip, a month before `job.first` is written as `man.mark(mkey, complete=True, source="absent", rows=0)` and returns `("absent", 0)` without any fetch. `backfill` looks up `first_archive_month` once per (symbol, dataset, tf) group that has at least one month not yet complete in its manifest, and not at all for a fully complete group. Its stats dict gains the key `"absent"`.
- Produces: `load_candles` drops the run of candles after the last candle with `trades > 0` in the loaded window when every one of them has `trades == 0` and `volume == 0`, logging once per (symbol, tf) at warning level. Zero-trade candles between real trades stay (a halt; change 7 decides them later). Candles with NaN `trades` (ccxt-stitched) are never dropped.

- [ ] **Step 1: Write the failing tests** in `tests/test_backfill_listing.py`

```python
import json

import numpy as np
import pandas as pd

from perp_lab.data import vision
from perp_lab.data.store import load_candles
from tests.conftest import market_config

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
XML = ("<ListBucketResult>"
       "<Contents><Key>data/futures/um/monthly/klines/SOLUSDT/1h/SOLUSDT-1h-2020-09.zip</Key></Contents>"
       "<Contents><Key>data/futures/um/monthly/klines/SOLUSDT/1h/SOLUSDT-1h-2020-09.zip.CHECKSUM</Key></Contents>"
       "<Contents><Key>data/futures/um/monthly/klines/SOLUSDT/1h/SOLUSDT-1h-2020-10.zip</Key></Contents>"
       "</ListBucketResult>")


def stub(monkeypatch, first):
    """The listing says `first`; every fetch 404s. Returns (lookups, fetched urls)."""
    looks, urls = [], []
    monkeypatch.setattr(vision, "first_archive_month", lambda *a, **kw: looks.append(a[:3]) or first)
    monkeypatch.setattr(vision, "_fetch_zip_csv", lambda url, *a, **kw: urls.append(url) or None)
    return looks, urls


def test_months_before_the_first_archive_file_are_absent_and_never_fetched(tmp_path, monkeypatch):
    cfg = market_config(tmp_path)
    looks, urls = stub(monkeypatch, pd.Period("2025-03", "M"))
    run = lambda: vision.backfill(cfg, ["SOLUSDT"], ["1h"], ["klines"], start=T("2025-01-01"),  # noqa: E731
                                  end=T("2025-03-31"), workers=1)
    assert run()["absent"] == 2 and len(looks) == 1
    assert not [u for u in urls if "2025-01" in u or "2025-02" in u]
    man = json.loads((cfg.data_dir / "SOLUSDT" / "1h" / "_manifest.json").read_text())
    assert man["2025-01"] == man["2025-02"] == {"complete": True, "source": "absent", "rows": 0}
    urls.clear()
    again = run()
    assert again["absent"] == 0 and again["skip"] == 2
    assert not [u for u in urls if "2025-01" in u or "2025-02" in u]


def test_a_complete_group_needs_no_listing(tmp_path, monkeypatch):
    cfg = market_config(tmp_path)
    looks, _ = stub(monkeypatch, pd.Period("2025-03", "M"))
    m = vision.Manifest(cfg.data_dir / "SOLUSDT" / "1h")
    for k in ("2025-01", "2025-02"):
        m.mark(k, complete=True, source="monthly", rows=1)
    vision.backfill(cfg, ["SOLUSDT"], ["1h"], ["klines"], start=T("2025-01-01"), end=T("2025-02-28"), workers=1)
    assert looks == []


def test_metrics_months_before_the_first_daily_file_are_absent(tmp_path, monkeypatch):
    cfg = market_config(tmp_path)
    _, urls = stub(monkeypatch, pd.Period("2025-02", "M"))
    s = vision.backfill(cfg, ["SOLUSDT"], [], ["metrics"], start=T("2025-01-01"), end=T("2025-02-28"), workers=1)
    assert s["absent"] == 1 and not [u for u in urls if "2025-01-" in u]


def test_an_unreadable_listing_marks_nothing_absent(tmp_path, monkeypatch):
    cfg = market_config(tmp_path)
    _, urls = stub(monkeypatch, None)
    s = vision.backfill(cfg, ["SOLUSDT"], ["1h"], ["klines"], start=T("2025-01-01"), end=T("2025-01-31"), workers=1)
    assert s["absent"] == 0 and s["missing"] == 1 and any("2025-01" in u for u in urls)   # today's behaviour


def test_first_archive_month_reads_the_earliest_zip(monkeypatch):
    seen = []
    monkeypatch.setattr(vision, "_get", lambda url, r=6: seen.append(url) or XML.encode())
    assert vision.first_archive_month("SOLUSDT", "klines", "1h", 1) == pd.Period("2020-09", "M")
    assert "prefix=data/futures/um/monthly/klines/SOLUSDT/1h/" in seen[0]
    vision.first_archive_month("SOLUSDT", "metrics", None, 1)
    assert "prefix=data/futures/um/daily/metrics/SOLUSDT/" in seen[1]


def test_first_archive_month_is_none_when_the_listing_cant_be_read(monkeypatch):
    for body in (None, b"<ListBucketResult></ListBucketResult>", b"not xml"):
        monkeypatch.setattr(vision, "_get", lambda url, r=6, b=body: b)
        assert vision.first_archive_month("SOLUSDT", "klines", "1h", 1) is None

    def down(url, r=6):
        raise vision.FetchError("HTTP 503")
    monkeypatch.setattr(vision, "_get", down)
    assert vision.first_archive_month("SOLUSDT", "klines", "1h", 1) is None


def write_1h(cfg, trades):
    t0 = T("2025-01-01").value // 10**6
    ot = t0 + np.arange(len(trades)) * 3_600_000
    tr = np.array(trades, dtype=float)
    d = cfg.data_dir / "BTCUSDT" / "1h"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"open_time": ot, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                  "volume": np.where(tr == 0, 0.0, 1.0), "quote_volume": np.where(tr == 0, 0.0, 100.0),
                  "trades": tr, "taker_buy_volume": 0.0, "taker_buy_quote_volume": 0.0,
                  "close_time": ot + 3_599_999}).to_parquet(d / "all.parquet")


def test_zero_trade_candles_after_the_last_trade_load_as_absent(tmp_path):
    cfg = market_config(tmp_path)
    write_1h(cfg, [5, 5, 0, 0, 5, 5, 0, 0, 0])
    assert load_candles(cfg, "BTCUSDT", "1h").trades.tolist() == [5, 5, 0, 0, 5, 5]   # the halt stays


def test_candles_without_a_trade_count_are_never_dropped(tmp_path):
    cfg = market_config(tmp_path)
    write_1h(cfg, [5, 5, np.nan, np.nan])                  # ccxt-stitched
    assert len(load_candles(cfg, "BTCUSDT", "1h")) == 4
```

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_backfill_listing.py -v`
Expected: FAIL (`first_archive_month` is missing; stats have no `absent` key; trailing zero-trade candles are still there).

- [ ] **Step 3: Implement** `first_archive_month`, `Job.first`, the absent branch in `run_job`, the per-group lookup in `backfill` (`stats` starts with `"absent": 0`), and the trailing-zero-trade drop in `load_candles`. Parse the listing with `xml.etree.ElementTree`, matching tags by local name, because S3 adds a namespace.

- [ ] **Step 4: Run the tests and the suite**

Expected: all PASS. `tests/test_exam.py` still passes: `window_data_problems` doesn't change, and no exam window holds an absent month.

- [ ] **Step 5: Commit** (two commits)

```bash
git commit   # feat(data): backfill marks months before a coin's first archive file absent
git commit   # feat(data): zero-trade candles after a coin's last trade load as absent
```

---

### Task 3: The slippage model, as pure functions (spec change 3, part 1)

**Files:**
- Create: `perp_lab/engine/slippage.py`
- Test: `tests/test_slippage.py` (new)

**Interfaces:**
- Produces, in `perp_lab/engine/slippage.py`:
  - `COST_MODEL_VERSION = 2`, with a comment: `# 1: flat costs.slippage_bps for every coin and date; 2: per coin and bar (costs.slippage_model)`.
  - `DAY_MS = 86_400_000`, `ADV_REF_USD = 1e8`.
  - `trailing_adv(open_time_ms: np.ndarray, quote_volume: np.ndarray, days: int) -> np.ndarray`, one value per bar: the mean of the daily quote-volume sums over the `days` calendar UTC days before the bar's day. A day whose bars have no finite `quote_volume` is skipped, not counted as 0; a day whose volume is 0 counts as 0. The value is NaN when none of those days has data. One way: daily sums with `min_count=1`, reindexed onto a continuous day range, `.rolling(days, min_periods=1).mean().shift(1)`, then mapped back by day.
  - `slippage_bps(adv_usd: np.ndarray, floor_bps: float, model: dict) -> np.ndarray`, equal to `min(model["max_bps"], floor_bps + model["spread_bps_at_100m"] * (adv / ADV_REF_USD) ** -model["alpha"])`. It returns `model["max_bps"]` where `adv` is NaN or ≤ 0, and raises `ValueError` if `model["max_bps"] < floor_bps`.
  - `half_spread_bps(bid: np.ndarray, ask: np.ndarray, t_ms: np.ndarray, end_ms: int) -> float`: the time-weighted mean over `[t_ms[0], end_ms)` of `(ask - bid) / (ask + bid) * 1e4`. Quotes are sorted by `t_ms`. Each quote is weighted by the time until the next one, and the last by the time until `end_ms`. Rows with `bid <= 0` or `ask <= bid` are dropped first.
  - `fit_spread_curve(adv_usd: np.ndarray, half_spread_bps: np.ndarray, pct: float = 90.0) -> dict`, returning `{"alpha": float, "spread_bps_at_100m": float, "n": int}`. With `x = log(adv / ADV_REF_USD)` and `y = log(hs)`: the OLS slope is `b`; `alpha = round(-b, 2)`; the residuals are `e = y + alpha * x`; `spread_bps_at_100m = ceil(exp(percentile(e, pct)) * 100) / 100`. It raises `ValueError` when fewer than 10 points are finite and positive, or when `b >= 0` (the spread doesn't fall as volume rises).

- [ ] **Step 1: Write the failing tests** in `tests/test_slippage.py`

```python
import numpy as np
import pytest

from perp_lab.engine.slippage import (DAY_MS, fit_spread_curve, half_spread_bps, slippage_bps,
                                      trailing_adv)

M = {"adv_days": 30, "spread_bps_at_100m": 1.5, "alpha": 0.5, "max_bps": 50.0}


def test_slippage_is_floor_plus_spread_capped_and_never_below_the_floor():
    got = slippage_bps(np.array([1e12, 1e8, 1e6, 1e4, 0.0, np.nan]), 2.0, M)
    assert got == pytest.approx([2.015, 3.5, 17.0, 50.0, 50.0, 50.0])
    grid = slippage_bps(np.logspace(3, 13, 200), 2.0, M)
    assert (grid >= 2.0).all() and (np.diff(grid) <= 0).all()


def test_a_cap_below_the_floor_is_refused():
    with pytest.raises(ValueError):
        slippage_bps(np.array([1e8]), 2.0, {**M, "max_bps": 1.0})


def hourly(days):
    t = np.arange(days * 24) * 3_600_000
    return t, t // DAY_MS


def test_trailing_adv_uses_only_full_days_before_the_bar():
    t, day = hourly(40)
    adv = trailing_adv(t, (day + 1).astype(float), 30)       # day k's sum is 24 * (k + 1)
    assert np.isnan(adv[day == 0]).all()
    assert np.allclose(adv[day == 1], 24) and np.allclose(adv[day == 35], 24 * 20.5)   # days 5..34


def test_trailing_adv_skips_days_without_volume_data():
    t, day = hourly(12)
    qv = (day + 1).astype(float)
    qv[day == 10] = np.nan
    assert np.allclose(trailing_adv(t, qv, 30)[day == 11], 24 * 5.5)   # days 0..9 only


def test_half_spread_is_time_weighted():
    h = 3_600_000
    got = half_spread_bps(np.array([99.99, 99.9]), np.array([100.01, 100.1]), np.array([0, h]), 24 * h)
    assert got == pytest.approx((1.0 * 1 + 10.0 * 23) / 24)


def test_fit_recovers_a_power_law_and_lifts_it_to_the_envelope():
    rng = np.random.default_rng(0)
    adv = np.logspace(6, 10, 40)
    hs = 3.0 * (adv / 1e8) ** -0.4 * np.exp(rng.normal(0, 0.1, 40))
    f = fit_spread_curve(adv, hs)
    assert f["alpha"] == pytest.approx(0.4, abs=0.03) and f["spread_bps_at_100m"] >= 3.0 and f["n"] == 40
    curve = f["spread_bps_at_100m"] * (adv / 1e8) ** -f["alpha"]
    assert (hs <= curve).mean() >= 0.9


def test_fit_refuses_a_spread_that_does_not_fall_with_volume():
    adv = np.logspace(6, 10, 20)
    with pytest.raises(ValueError):
        fit_spread_curve(adv, 0.5 * (adv / 1e8) ** 0.2)
    with pytest.raises(ValueError):
        fit_spread_curve(adv[:5], 3.0 * (adv[:5] / 1e8) ** -0.4)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_slippage.py -v`
Expected: FAIL with `ModuleNotFoundError: perp_lab.engine.slippage`.

- [ ] **Step 3: Implement** `perp_lab/engine/slippage.py` (numpy and pandas only; no config or I/O).

- [ ] **Step 4: Run the tests**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit   # feat(engine): per-coin slippage model and its spread-curve fit
```

---

### Task 4: Calibrate the slippage model and fix it before any trial (spec change 3, part 2)

**Files:**
- Create: `docs/research/slippage-calibration/calibrate.py`, `docs/research/slippage-calibration/calibration.csv`
- Modify: `docs/superpowers/specs/2026-10-18-wider-data-design.md` (new section "Slippage model")
- Modify: `config/lab.yaml` (`costs`)
- Test: `tests/test_config.py` (new)

**Interfaces:**
- Consumes: `half_spread_bps`, `fit_spread_curve`, `slippage_bps` and `ADV_REF_USD` from Task 3; `vision._get`, `vision.parse_klines` and `vision._strip_header`.
- Produces: `config/lab.yaml` `costs.slippage_model = {adv_days: 30, spread_bps_at_100m: <fit>, alpha: <fit>, max_bps: 50.0}`. `costs.slippage_bps: 2.0` stays, now commented as the floor and as the fallback for bars without a `slippage` column.

- [ ] **Step 1: Write `calibrate.py`.** It runs as `.venv/bin/python docs/research/slippage-calibration/calibrate.py` from the repo root:
  - Constants: the 17 coins (Global Constraint 2); `DAYS = ["2023-06-15", "2023-09-15", "2023-12-15", "2024-03-15"]`. It asserts that every day it reads, including each ADV window, is before `2026-07-01`.
  - For each coin and day: download `https://data.binance.vision/data/futures/um/daily/bookTicker/{S}/{S}-bookTicker-{day}.zip` with `vision._get` and verify its `.CHECKSUM`, as `vision._fetch_zip_csv` does. Read the CSV straight from the zip as numbers, only `best_bid_price`, `best_ask_price` and `transaction_time`. Never use `dtype=str`: a BTC day is about 140 MB zipped. Handle files with and without a header row. Compute `half_spread_bps(bid, ask, t, day_end_ms)`.
  - `adv`: the mean daily `quote_volume` over the 30 days before the day, from `https://data.binance.vision/data/futures/um/monthly/klines/{S}/1d/{S}-1d-{YYYY-MM}.zip` (parsed with `vision.parse_klines`).
  - Write `calibration.csv` next to the script, with columns `coin, day, adv_usd, half_spread_bps`. Then print `fit_spread_curve(...)`.
  - Print a table of `slippage_bps` per coin, using the fitted model and floor 2.0, for a bar on 2020-08-01, 2022-02-01 and 2026-06-30. Each uses `adv` from the 30 days before that date (monthly `1d` klines, all before 2026-07-01), with "–" where the coin has no volume yet.
  - Keep nothing on disk except the CSV.

- [ ] **Step 2: Run it**

Run: `.venv/bin/python docs/research/slippage-calibration/calibrate.py | tee <your scratchpad>/calibration.txt` (it downloads about 3 GB; expect 30–60 min).
Expected: 68 rows in the CSV, a fit with `alpha > 0`, BTC and ETH at 2026-06-30 within 0.1 bp of 2.0, and every printed value ≥ 2.0. **If the fit raises, or BTC/ETH come out more than 0.1 bp above 2.0, stop and report to the user. Do not adjust the method.**

- [ ] **Step 3: Write the failing config test** in `tests/test_config.py`

```python
import yaml

from perp_lab.config import load_config

CFG = load_config()
RAW = yaml.safe_load(open("config/lab.yaml"))


def test_the_slippage_floor_is_todays_cost_and_the_model_is_set():
    c = CFG["costs"]
    assert c["slippage_bps"] == 2.0
    m = c["slippage_model"]
    assert set(m) == {"adv_days", "spread_bps_at_100m", "alpha", "max_bps"} and m["adv_days"] == 30
    assert m["alpha"] > 0 and m["spread_bps_at_100m"] > 0 and m["max_bps"] == 50.0
```

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_config.py -v`. Expected: FAIL (`KeyError: 'slippage_model'`).

- [ ] **Step 4: Add `costs.slippage_model`** to `config/lab.yaml` with the printed values, plus a comment pointing to the spec section. Run the test. Expected: PASS.

- [ ] **Step 5: Document it in the spec.** Add a section `## Slippage model (calibrated <run date>)` after "Changes" in `docs/superpowers/specs/2026-10-18-wider-data-design.md` with:
  - the formula with the fitted values;
  - what `adv` is, and the cap with when it applies;
  - why the floor is additive (see "Decisions this plan makes" above);
  - the source (dataset, coins, days, and that all of it is before the holdout);
  - how the half-spread was measured;
  - the fit (OLS, p90 lift, rounding);
  - the printed table for 2020-08-01 / 2022-02-01 / 2026-06-30;
  - the limits: 2020 is extrapolated by volume alone; spreads in bps also depend on tick size; the model applies to taker fills only, and targets stay maker;
  - the rule that the model is fixed before any trial on the new data, and that changing it changes every run's data identity (Task 8).
  - Also set `costs.slippage_model` in "Decisions" to point at the section.

- [ ] **Step 6: Commit** (two commits)

```bash
git add docs/research/slippage-calibration docs/superpowers/specs/2026-10-18-wider-data-design.md
git commit   # docs(research): calibrate per-coin slippage from pre-holdout bookTicker spreads
git add config/lab.yaml tests/test_config.py
git commit   # feat(config): per-coin slippage model, fixed before any trial on the new data
```

---

### Task 5: Every bar carries its slippage; the engine fills with it (spec change 3, part 3)

**Files:**
- Modify: `perp_lab/data/store.py:130-141` (`load_bars`, new `attach_slippage`)
- Modify: `perp_lab/engine/backtest.py` (fills at steps 2–5 and end of window; `EngineParams` docstring)
- Test: `tests/test_engine_costs.py` (additions), `tests/test_slippage.py` (additions)

**Interfaces:**
- Consumes: `trailing_adv` and `slippage_bps` (Task 3); `costs.slippage_model` (Task 4).
- Produces: `attach_slippage(cfg, candles: pd.DataFrame) -> pd.DataFrame` adds the column `slippage` as a fraction: `slippage_bps(trailing_adv(open_time, quote_volume, model["adv_days"]), costs["slippage_bps"], model) / 1e4`. `load_bars` applies it to every bar it returns. When a `start` is given, it loads candles from `start - (adv_days + 1) days`, attaches, then trims to `[start, end)`, so a bar's slippage never depends on where the load starts.
- Produces: `run_backtest` reads `A.get("slippage")`. Every taker fill at bar `i` (gap stop, signal exit, entry, intrabar stop, end of window) uses `slippage[i]` when the column exists and `ep.slippage` otherwise. The fallback covers synthetic bars and paper replays. Maker targets and liquidations have no slippage, as today. The worker and the trusted replay both build their arrays from the same bars, so their fingerprints still agree.

- [ ] **Step 1: Write the failing tests.** Add to `tests/test_engine_costs.py`:

```python
def test_fills_use_each_bars_own_slippage():
    b = ramp_bars()
    b["slippage"] = 0.0002
    b.loc[b.index[6], "slippage"] = 0.001
    b.loc[b.index[11], "slippage"] = 0.003
    t = run(b, {5: Signal("long", stop=90.0), 10: EXIT}).trades[0]
    assert t["entry_price"] == pytest.approx(106.0 * 1.001)
    assert t["exit_price"] == pytest.approx(111.0 * (1 - 0.003))


def test_a_stop_fills_with_its_bars_slippage():
    b = ramp_bars()
    b["slippage"] = 0.0002
    b.loc[b.index[8], ["low", "slippage"]] = [103.0, 0.002]
    t = run(b, {5: Signal("long", stop=104.0)}).trades[0]
    assert t["exit_reason"] == "stop" and t["exit_price"] == pytest.approx(104.0 * (1 - 0.002))
```

Add to `tests/test_slippage.py`:

```python
import pandas as pd

from perp_lab.data.store import load_bars
from tests.conftest import market_config

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def with_volume(cfg, values):
    p = cfg.data_dir / "BTCUSDT" / "4h" / "all.parquet"
    df = pd.read_parquet(p)
    df["quote_volume"] = values(len(df))
    df.to_parquet(p)


def test_each_bar_pays_for_the_volume_before_its_day(tmp_path):
    cfg = market_config(tmp_path)
    with_volume(cfg, lambda n: np.full(n, 1e6))               # 6 bars a day: 6e6 USDT
    b = load_bars(cfg, "BTCUSDT", "4h")
    c = cfg["costs"]
    assert (b.slippage[b.index < T("2025-01-02")] == c["slippage_model"]["max_bps"] / 1e4).all()
    want = slippage_bps(np.array([6e6]), c["slippage_bps"], c["slippage_model"])[0] / 1e4
    assert np.allclose(b.slippage[b.index >= T("2025-02-01")], want) and b.slippage.notna().all()


def test_a_bars_slippage_does_not_depend_on_where_the_load_starts(tmp_path):
    cfg = market_config(tmp_path)
    with_volume(cfg, lambda n: np.random.default_rng(3).uniform(1e5, 1e7, n))
    full = load_bars(cfg, "BTCUSDT", "4h")
    part = load_bars(cfg, "BTCUSDT", "4h", start=T("2025-03-10"))
    pd.testing.assert_series_equal(part.slippage, full.slippage.loc[part.index])


def test_bars_without_volume_pay_the_cap_never_nan(tmp_path):
    cfg = market_config(tmp_path)                              # synthetic quote_volume is NaN
    b = load_bars(cfg, "ETHUSDT", "4h")
    assert (b.slippage == cfg["costs"]["slippage_model"]["max_bps"] / 1e4).all()
```

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_engine_costs.py tests/test_slippage.py -v`
Expected: the new tests FAIL (no `slippage` column; the engine ignores it).

- [ ] **Step 3: Implement** `attach_slippage` and the lookback in `load_bars`. In `run_backtest`, replace each use of the scalar `slip` in a fill with the bar's value. Keep the step order and every branch as it is. Update the module docstring's steps 3 and 4 and `EngineParams.slippage`'s comment (the fallback for bars without a `slippage` column).

- [ ] **Step 4: Run the tests and the full suite**

Expected: the new tests PASS. Existing tests that compare a sandbox run with a direct `run_backtest` on `load_bars` bars still pass, because both read the same column. If an existing test fails only because synthetic bars now pay `max_bps`, give that test a config with `costs={"slippage_model": {**model, "max_bps": 2.0}}`, which reproduces the flat 2 bp exactly, and add a one-line comment. Change nothing else in existing tests, and never change the model to make a test pass.

- [ ] **Step 5: Commit**

```bash
git commit   # feat(engine): fill each bar at its coin's point-in-time slippage
```

---

### Task 6: Gates scale with the number of coins (spec change 4)

**Files:**
- Modify: `perp_lab/runs.py:16, 48-59` (`MIN_OOS_TRADES`, `verdict`)
- Modify: `perp_lab/validate/walkforward.py:226-236` (train gate), new `listed_coins`
- Modify: `perp_lab/validate/pipeline.py:94` (`verdict` call)
- Modify: `perp_lab/judge/judge.py:494-497` (`_attempt`), new `holdout_min_trades`
- Modify: `config/lab.yaml` (`walkforward`, `judge`)
- Test: `tests/test_coin_gates.py` (new)

**Interfaces:**
- Produces, in `perp_lab/runs.py`: `MIN_OOS_TRADES_PER_COIN = 30 / 8` replaces `MIN_OOS_TRADES`. `per_coin_min(per_coin: float, n_coins: int) -> int` returns `math.ceil(per_coin * n_coins)`. The new signature is `verdict(oos: dict, rnd: dict, sens: dict, n_coins: int)`, and its trade reason reads `f"only {t} OOS trades (< {m}: {MIN_OOS_TRADES_PER_COIN:g} per coin × {n_coins} coins)"`.
- Produces, in `perp_lab/validate/walkforward.py`: `listed_coins(daily: dict[str, pd.Series], a, b) -> int`, the number of coins with at least one daily return in `[a, b)`. In fold `k`, a combo scores `-inf` when its pooled train trades are fewer than `per_coin_min(wf["min_train_trades_per_coin"], listed_coins({s: R[(keys[0], s)]["daily"] for s in symbols}, a, b))`. Listing doesn't depend on the combo, so compute this minimum once per fold, before the combo loop. `walkforward.py` imports `per_coin_min` from `perp_lab.runs` and calls it by that global name, which lets the test spy on `wfm.per_coin_min`.
- Produces, in `perp_lab/judge/judge.py`: `holdout_min_trades(cfg, n_coins: int) -> int` returns `max(jc["holdout_min_trades"], per_coin_min(jc["holdout_min_trades_per_coin"], n_coins))`. `_attempt` passes it with `n_coins = sum(len(d) > 0 for d in res.daily.values())`.
- Config: `walkforward.min_train_trades: 20` becomes `min_train_trades_per_coin: 2.5`, commented "pooled; × coins listed in the train window (20 at 8 coins)". `judge.holdout_min_trades: 10` stays as the floor. Add `judge.holdout_min_trades_per_coin: 1.25`, commented "× coins in the exam, never below holdout_min_trades".
- `pipeline.validate_strategy` calls `verdict(..., listed_coins(wf.oos_daily, oa, ob))`.

- [ ] **Step 1: Write the failing tests** in `tests/test_coin_gates.py`

```python
import pandas as pd
import pytest

from perp_lab.judge.judge import holdout_min_trades
from perp_lab.plugins import PluginSource
from perp_lab.runs import per_coin_min, verdict
from perp_lab.validate import walkforward as wfm
from tests.conftest import market_config
from tests.test_late_listing import keep_from
from tests.test_walkforward_sandbox import DONCHIAN

OK_RND, OK_SENS = {"beats_random": True}, {"narrow_island": False}


@pytest.mark.parametrize("n, train, oos, exam", [(8, 20, 30, 10), (17, 43, 64, 22), (11, 28, 42, 14)])
def test_per_coin_minimums_keep_todays_counts_at_8_coins(n, train, oos, exam):
    assert (per_coin_min(2.5, n), per_coin_min(30 / 8, n), per_coin_min(1.25, n)) == (train, oos, exam)


def test_the_oos_trade_minimum_scales_with_coins():
    status, fails = verdict({"sharpe": 1.0, "trades": 63}, OK_RND, OK_SENS, 17)
    assert status == "noise" and fails == ["only 63 OOS trades (< 64: 3.75 per coin × 17 coins)"]
    assert verdict({"sharpe": 1.0, "trades": 64}, OK_RND, OK_SENS, 17)[0] == "candidate"


def test_the_exam_minimum_scales_but_never_below_the_floor(cfg):
    assert [holdout_min_trades(cfg, n) for n in (2, 8, 17)] == [10, 10, 22]


def test_listed_coins_counts_coins_with_a_return_in_the_window():
    d = pd.date_range("2025-01-01", periods=60, freq="D", tz="UTC")
    daily = {"A": pd.Series(0.0, index=d), "B": pd.Series(0.0, index=d[40:])}
    assert wfm.listed_coins(daily, d[0], d[30]) == 1 and wfm.listed_coins(daily, d[30], d[59]) == 2


def test_the_train_gate_counts_only_coins_listed_in_each_fold(tmp_path, monkeypatch):
    cfg = market_config(tmp_path)
    keep_from(cfg, "ETHUSDT", "4h", "2025-09-01")          # after fold 0's train window (2025-01..07)
    seen = []
    real = wfm.per_coin_min
    monkeypatch.setattr(wfm, "per_coin_min", lambda r, n: seen.append(n) or real(r, n))
    wf = wfm.walkforward(cfg, PluginSource.from_text("donchian_atr", DONCHIAN), "4h", workers=1)
    assert seen[0] == 1 and seen[-1] == 2 and len(seen) == len(wf.folds)
```

Import `cfg` from `tests.test_store` as the other judge tests do (`from tests.test_store import cfg  # noqa: F401`).

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_coin_gates.py -v`
Expected: FAIL (the names are missing).

- [ ] **Step 3: Implement** the helpers, the scaled train gate, the `verdict` change and its caller, `holdout_min_trades` and its use in `_attempt`, and the config keys.

- [ ] **Step 4: Run the tests and the full suite**

Expected: all PASS. The judge-exam tests use 2 coins, so their exam minimum stays 10.

- [ ] **Step 5: Commit** (three commits: walk-forward train gate; OOS verdict; exam minimum)

```bash
git commit   # feat(validate): the train-trade gate scales with coins listed in the fold
git commit   # feat(runs): the OOS trade minimum scales with coins in the OOS span
git commit   # feat(judge): the exam's trade minimum scales with coins, never below 10
```

---

### Task 7: Monte Carlo per-day coin count; random entry keeps zero-trade coins (spec change 5)

**Files:**
- Modify: `perp_lab/validate/montecarlo.py` (`mc_drawdown`, new `coins_per_day`; module docstring)
- Modify: `perp_lab/validate/pipeline.py:89` (the call)
- Modify: `perp_lab/validate/benchmarks.py:29-62` (`random_entry`)
- Test: `tests/test_benchmarks.py` (new)

**Interfaces:**
- Produces: `coins_per_day(daily: dict[str, pd.Series]) -> pd.Series`, indexed by UTC day, giving the number of coins with a return that day.
- Produces: `mc_drawdown(trades: pd.DataFrame, coins_per_day: pd.Series, iterations: int = 10_000, seed: int = 7) -> dict`. Each trade's `pnl_pct` is divided by the count on its entry day, the UTC floor of `entry_time`. A day missing from the index takes the previous day's count (ffill), then the next day's (bfill). The output keys are unchanged.
- Produces: in `random_entry`, a coin with `probs[s] == 0` gets no job, but it enters every seed's portfolio as zeros on the days of `wf.oos_daily[s]` in `[oa, ob)`, so the random portfolio averages the same coins on the same days as the strategy's.

- [ ] **Step 1: Write the failing tests** in `tests/test_benchmarks.py`

```python
import numpy as np
import pandas as pd
import pytest

from perp_lab.engine.metrics import portfolio_returns, sharpe
from perp_lab.validate import benchmarks
from perp_lab.validate.montecarlo import coins_per_day, mc_drawdown
from tests.test_store import cfg  # noqa: F401

DAYS = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")


def test_monte_carlo_divides_each_trade_by_the_coins_listed_on_its_entry_day():
    cpd = coins_per_day({"A": pd.Series(0.0, index=DAYS), "B": pd.Series(0.0, index=DAYS[5:])})
    assert cpd.tolist() == [1] * 5 + [2] * 5
    ms = lambda d: int(DAYS[d].value // 10**6) + 3_600_000  # noqa: E731
    trades = pd.DataFrame({"entry_time": [ms(0), ms(1), ms(6), ms(7), ms(8)],
                           "pnl_pct": [0.0, -0.1, -0.1, 0.0, 0.0]})       # -0.1 alone, then -0.1 / 2
    assert mc_drawdown(trades, cpd, iterations=10, seed=1)["actual_dd"] == pytest.approx(0.9 * 0.95 - 1)


def test_random_entry_keeps_coins_the_strategy_never_traded(cfg, monkeypatch):
    rng = np.random.default_rng(0)
    ra = {seed: pd.Series(rng.normal(0, 0.01, 10), index=DAYS) for seed in range(3)}
    monkeypatch.setattr(benchmarks, "run_many", lambda cfg, src, jobs, workers, stderr_dir: [
        {"seed": j[4], "symbol": j[1], "daily": ra[j[4]], "trades": pd.DataFrame()} for j in jobs])
    out = benchmarks.random_entry(cfg, make_wf(), seeds=3)
    want = [sharpe(portfolio_returns({"A": ra[s], "B": pd.Series(0.0, index=DAYS)})) for s in range(3)]
    assert sorted(out["sharpes"]) == pytest.approx(sorted(want))
```

`make_wf()` is a small helper in the same file. It builds a `WFResult` with `symbols=["A", "B"]`, `tf="4h"`, `oos_daily={"A": zeros, "B": zeros}` on `DAYS`, `oos_trades` holding one trade for `A` (`symbol`, `signal_time`) and none for `B`, `oos_flat_bars={"A": 100.0, "B": 100.0}`, `selected=[{"combo": "default", "params": {}}]`, `oos_summary={"sharpe": 0.0}`, `raw={"oos_span": (DAYS[0], DAYS[-1] + pd.Timedelta(days=1))}`, and placeholders for the rest (`src=None`, `stderr_dir=None`).

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_benchmarks.py -v`
Expected: FAIL (`coins_per_day` can't be imported; the random portfolio leaves B out).

- [ ] **Step 3: Implement** both changes and update `pipeline.validate_strategy` to `mc_drawdown(wf.oos_trades, coins_per_day(wf.oos_daily), cfg["validation"]["mc_iterations"])`.

- [ ] **Step 4: Run the tests and the full suite.** Expected: PASS.

- [ ] **Step 5: Commit** (two commits)

```bash
git commit   # fix(validate): Monte Carlo weights each trade by the coins listed that day
git commit   # fix(validate): the random-entry portfolio keeps coins the strategy never traded
```

---

### Task 8: Runs remember their data (spec change 6)

**Files:**
- Modify: `perp_lab/runs.py` (`data_identity`, `data_id`, stamp in `write_run`)
- Modify: `perp_lab/store/db.py` (`runs.data_id` column, migration in `_engine`, `record_run`, `latest_wf_run`)
- Modify: `perp_lab/judge/judge.py:224-227` (match by `data_id`)
- Modify: `perp_lab/cli.py:172-177` (paper run stamps its identity)
- Test: `tests/test_data_identity.py` (new); `tests/test_judgements_store.py:71-84` (pass `data_id`); `tests/test_judge.py` and `tests/test_paper.py` (one addition each)

**Interfaces:**
- Consumes: `COST_MODEL_VERSION` (Task 3).
- Produces, in `perp_lab/runs.py`:
  - `data_identity(cfg, symbols: list[str]) -> dict` returns `{"history_start": "YYYY-MM-DD", "universe": list(symbols), "cost_model": COST_MODEL_VERSION, "costs": cfg["costs"], "brackets": {s: cfg.bracket(s) for s in symbols}}`.
  - `data_id(identity: dict) -> str` returns the first 16 hex characters of the sha256 of `json.dumps(canon, sort_keys=True, separators=(",", ":"))`. `canon` is the identity with every int (not bool) turned into a float, so `2` and `2.0` hash alike.
  - `write_run` sets `record["data"] = data_identity(cfg, record["universe"])` and `record["data_id"]` before it writes `summary.json`.
- Produces, in `perp_lab/store/db.py`: `runs` gains `sa.Column("data_id", sa.String(16))`, written by `record_run` from `rec.get("data_id")`. After `md.create_all`, `_engine` checks `sa.inspect(eng).get_columns("runs")` and runs `ALTER TABLE runs ADD COLUMN data_id VARCHAR(16)` when the column is missing. Old rows stay NULL. The new signature is `latest_wf_run(cfg, strategy, timeframe, strategy_hash, universe, *, data_id: str, lineage_id: str | None = None)`, and it also requires `r.data_id == data_id`.
- `judge()` passes `data_id=data_id(data_identity(cfg, universe))`, where `universe = symbols or cfg.symbols` as today. A parent run whose coins were skipped (Task 1) or whose data differs never matches, so the parent is re-run under the family, as the judge already does when nothing matches.
- `cli.paper` adds `data` and `data_id` (from `syms`) to the paper run it records with `record_run`.

- [ ] **Step 1: Write the failing tests** in `tests/test_data_identity.py`

```python
import json
import sqlite3

import yaml

from perp_lab.config import Config, load_config
from perp_lab.runs import data_id, data_identity
from perp_lab.store import db
from tests.test_store import cfg, fake_run  # noqa: F401

U = ["BTCUSDT", "ETHUSDT"]


def other(cfg, **raw):
    return Config(raw={**cfg.raw, **raw}, path=cfg.path)


def test_data_id_follows_history_universe_and_costs_only(cfg):
    base = data_id(data_identity(cfg, U))
    costs = {**cfg["costs"], "slippage_model": {**cfg["costs"]["slippage_model"], "alpha": 9.0}}
    br = cfg["margin_brackets"]
    assert data_id(data_identity(other(cfg, judge={**cfg["judge"], "dsr_min": 0.5}), U)) == base
    assert data_id(data_identity(other(cfg, history_start="2019-01-01"), U)) != base
    assert data_id(data_identity(cfg, ["BTCUSDT"])) != base
    assert data_id(data_identity(other(cfg, costs=costs), U)) != base
    assert data_id(data_identity(other(cfg, margin_brackets={**br, "ETHUSDT": {"mmr": 0.02, "cum": 0.0}}), U)) != base
    assert data_id(data_identity(other(cfg, margin_brackets={**br, "SOLUSDT": {"mmr": 0.02, "cum": 0.0}}), U)) == base


def test_the_same_config_written_differently_has_the_same_data_id(cfg, tmp_path):
    raw = yaml.safe_load(open(cfg.path))
    raw["costs"] = dict(reversed(list(raw["costs"].items())))
    raw["costs"]["slippage_bps"] = int(raw["costs"]["slippage_bps"])          # 2 instead of 2.0
    p = tmp_path / "again.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert data_id(data_identity(load_config(p), U)) == data_id(data_identity(cfg, U))


def test_every_written_run_records_its_data(cfg):
    fake_run(cfg, "r1")
    s = json.loads((cfg.runs_dir / "r1" / "summary.json").read_text())
    assert s["data"]["history_start"] == str(cfg.history_start.date()) and s["data"]["universe"] == U
    assert s["data_id"] == data_id(data_identity(cfg, U)) == db.run_row(cfg, "r1")["data_id"]


def test_an_existing_store_gains_the_data_id_column(cfg):
    fake_run(cfg, "old")
    path = cfg["store"]["url"][len("sqlite:///"):]
    with sqlite3.connect(path) as c:
        c.execute("ALTER TABLE runs DROP COLUMN data_id")
    db._engine.cache_clear()
    assert db.run_row(cfg, "old")["data_id"] is None                 # added back, empty for old runs
    fake_run(cfg, "new")
    assert db.run_row(cfg, "new")["data_id"] == data_id(data_identity(cfg, U))


def test_a_parent_run_on_other_data_is_not_matched(cfg):
    fake_run(other(cfg, history_start="2019-01-01"), "elsewhere", strategy="p", strategy_hash="hp")
    assert db.latest_wf_run(cfg, "p", "4h", "hp", U, data_id=data_id(data_identity(cfg, U))) is None
    fake_run(cfg, "here", strategy="p", strategy_hash="hp")
    assert db.latest_wf_run(cfg, "p", "4h", "hp", U, data_id=data_id(data_identity(cfg, U)))["run_id"] == "here"


def test_n_counts_a_plugin_version_once_across_data(cfg):
    fake_run(other(cfg, history_start="2019-01-01"), "a", strategy="p", strategy_hash="hp")
    fake_run(cfg, "b", strategy="p", strategy_hash="hp")
    assert db.trial_stats(cfg)[0] == 1
```

Add to `tests/test_judge.py`:

```python
def test_the_parent_is_re_run_when_its_run_used_other_data(cfg, plugins):
    v = StubValidate({"donchian_atr": ("candidate", STRONG), "donchian_atr_wide": ("candidate", STRONG + 0.001)})
    elsewhere = Config(raw={**cfg.raw, "history_start": "2019-01-01"}, path=cfg.path)
    old, _ = v(elsewhere, "donchian_atr", "4h", lineage_id="family:donchian_atr")
    j = judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert v.calls == ["donchian_atr", "donchian_atr", "donchian_atr_wide"] and j["parent_run_id"] != old["run_id"]
```

Add to `tests/test_paper.py`. First move the approval setup of `test_cli_paper_without_a_sandbox_says_so` (market, `fake_run` c1, passed judgement, `approve_paper`) into a helper `approved_paper_run(tmp_path) -> cfg`, which that test and Task 9's test also use:

```python
def test_a_paper_run_records_its_data(tmp_path, monkeypatch, plugins):
    from typer.testing import CliRunner

    from perp_lab.cli import app
    from perp_lab.runs import data_id, data_identity
    from perp_lab.store import db
    cfg = approved_paper_run(tmp_path)
    monkeypatch.setattr(client, "max_workers", lambda c: 8)      # Task 9's refusal must not apply here
    monkeypatch.setattr(PaperRunner, "run", lambda self, max_bars=None: asyncio.sleep(0))
    r = CliRunner().invoke(app, ["paper", "c1", "--config", str(cfg.path)])
    assert r.exit_code == 0, r.output
    p = db.read_table(cfg, db.runs, db.runs.c.validation_mode == "paper")
    assert p.data_id.tolist() == [data_id(data_identity(cfg, ["BTCUSDT", "ETHUSDT"]))]
```

In `tests/test_judgements_store.py::test_latest_wf_run_matches_hash_and_universe`, pass `data_id=did(u)` to every `latest_wf_run` call, with `did = lambda u: data_id(data_identity(cfg, u))`. Change nothing else in that test. `test_an_existing_parent_run_is_reused` must keep passing unchanged.

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_data_identity.py tests/test_judgements_store.py tests/test_judge.py tests/test_paper.py -v`
Expected: FAIL (`data_identity` can't be imported). `tests/test_judge.py` needs `from perp_lab.config import Config`.

- [ ] **Step 3: Implement** in this order: `runs.py` helpers and the `write_run` stamp, the `db.py` column, migration and `latest_wf_run`, the judge call, and the paper stamp.

- [ ] **Step 4: Run the tests and the full suite.** Expected: PASS.

- [ ] **Step 5: Commit** (two commits: record the identity on every run plus the store column and migration; then match it in the judge)

```bash
git commit   # feat(runs): record each run's data identity (history start, universe, costs)
git commit   # feat(judge): a parent run is reused only on the same data
```

---

### Task 9: Paper and tail per coin (spec change 8)

**Files:**
- Modify: `perp_lab/cli.py:58-67` (`tail`), `perp_lab/cli.py:146-181` (`paper`)
- Modify: `perp_lab/paper/runner.py` (`PaperRunner`: `dropped`, the three loops, `run`, `_persist`)
- Test: `tests/test_paper.py` (additions)

**Interfaces:**
- `lab paper` refuses before it records anything when `len(syms) > client.max_workers(cfg)`. It exits 2 with `refused: {n} coins need {n} paper workers of {memory_mb} MB; this machine holds {m}. Pass --symbols with at most {m} coins.` Which real fix to pick waits until a strategy first passes an exam (spec).
- `lab tail` handles each (coin, tf) separately. A failure prints `{s} {tf}: failed: {type(e).__name__}: {e}` in red and the loop carries on. At the end, any failure prints `tail failed for: ` followed by the failed `"{s} {tf}"` pairs joined by `", "`, and exits 1.
- `PaperRunner.dropped: dict[str, str]`, mapping each dropped coin to `f"{type(e).__name__}: {e}"`. A `ccxt.BadSymbol` from any of a coin's three loops, warm-up included, drops that coin: it is logged at error, recorded, and its other loops stop at their next turn. Other coins carry on. An open position of a dropped coin stays as it is in `state.json`, with no fill invented. `state.json` gains `"dropped": <reason>` under that coin. Any other error in any loop still stops the whole run at once, including after a coin was dropped: `run` keeps waiting on the remaining side loops until the bars are done. When every coin is dropped, `run` raises `RuntimeError("paper: every coin was dropped: ...")`.

- [ ] **Step 1: Write the failing tests** (add to `tests/test_paper.py`, with `import json` and `import ccxt` at the top)

```python

class GoneEx(FakeEx):
    """ETH is no longer listed: every call for it raises BadSymbol, as ccxt does after a delisting."""
    GONE = {"ETH/USDT:USDT"}

    def _check(self, sym):
        if sym in self.GONE or (isinstance(sym, list) and set(sym) & self.GONE):
            raise ccxt.BadSymbol(f"binance does not have market symbol {sym}")

    async def fetch_ohlcv(self, sym, tf, since=None, limit=None):
        self._check(sym)
        return await super().fetch_ohlcv(sym, tf, since, limit)

    async def watch_ohlcv(self, sym, tf):
        self._check(sym)
        return await super().watch_ohlcv(sym, tf)

    async def watch_bids_asks(self, symbols):
        self._check(symbols)
        return await super().watch_bids_asks(symbols)

    async def fetch_funding_rate(self, sym):
        self._check(sym)
        return await super().fetch_funding_rate(sym)


def test_paper_drops_a_coin_the_exchange_no_longer_lists(tmp_path):
    from perp_lab.config import load_config
    r = PaperRunner(load_config(), PluginSource.from_text("always", ALWAYS), {}, "1h", ["BTCUSDT", "ETHUSDT"],
                    tmp_path, exchange=GoneEx())
    asyncio.run(r.run(max_bars=3))
    assert set(r.dropped) == {"ETHUSDT"} and r.books["BTCUSDT"].pos is not None
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["ETHUSDT"]["dropped"].startswith("BadSymbol: ") and "dropped" not in state["BTCUSDT"]


def test_paper_still_stops_on_a_failure_after_a_coin_was_dropped(tmp_path):
    from perp_lab.config import load_config

    class SlowGone(GoneEx, SlowFakeEx):
        pass
    r = PaperRunner(load_config(), PluginSource.from_text("overflows", OVERFLOWS), {}, "1h",
                    ["BTCUSDT", "ETHUSDT"], tmp_path, exchange=SlowGone())
    with pytest.raises(PluginError) as e:
        asyncio.run(r.run(max_bars=8))
    assert e.value.kind == "mismatch" and set(r.dropped) == {"ETHUSDT"}


def test_paper_fails_when_every_coin_is_dropped(tmp_path):
    from perp_lab.config import load_config

    class AllGone(GoneEx):
        GONE = {"ETH/USDT:USDT", "BTC/USDT:USDT"}
    r = PaperRunner(load_config(), PluginSource.from_text("always", ALWAYS), {}, "1h", ["BTCUSDT", "ETHUSDT"],
                    tmp_path, exchange=AllGone())
    with pytest.raises(RuntimeError, match="every coin"):
        asyncio.run(r.run(max_bars=3))


def test_cli_paper_refuses_more_coins_than_workers(tmp_path, monkeypatch, plugins):
    from typer.testing import CliRunner

    from perp_lab.cli import app
    from perp_lab.store import db
    cfg = approved_paper_run(tmp_path)                  # helper from Task 8
    monkeypatch.setattr(client, "max_workers", lambda c: 1)
    monkeypatch.setattr(PaperRunner, "run", lambda self, max_bars=None: asyncio.sleep(0))
    r = CliRunner().invoke(app, ["paper", "c1", "--config", str(cfg.path)])
    assert r.exit_code == 2 and "refused: 2 coins need 2 paper workers" in r.output
    assert db.read_table(cfg, db.runs, db.runs.c.validation_mode == "paper").empty
    r = CliRunner().invoke(app, ["paper", "c1", "--symbols", "BTC", "--config", str(cfg.path)])
    assert r.exit_code == 0, r.output


def test_cli_tail_carries_on_past_a_failing_coin(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from perp_lab.cli import app
    from perp_lab.data import live
    cfg = market_config(tmp_path)

    def tail(c, s, tf, ex):
        if s == "ETHUSDT":
            raise ccxt.BadSymbol("binance does not have market symbol ETH/USDT:USDT")
        return 3
    monkeypatch.setattr(live, "make_exchange", lambda: object())
    monkeypatch.setattr(live, "tail", tail)
    r = CliRunner().invoke(app, ["tail", "--symbols", "BTC,ETH,SOL", "--tfs", "1h", "--config", str(cfg.path)])
    assert r.exit_code == 1
    assert "BTCUSDT 1h: +3" in r.output and "ETHUSDT 1h: failed: BadSymbol" in r.output
    assert "SOLUSDT 1h: +3" in r.output and "tail failed for: ETHUSDT 1h" in r.output
```

- [ ] **Step 2: Run them and watch them fail**

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_paper.py -v`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement** the CLI refusal (import `client` in `paper`), the per-coin tail, and in `PaperRunner`: `dropped`, a per-coin stop event, `BadSymbol` handling in `_bars_loop`, `_quotes_loop` and `_funding_loop`, the waiting loop in `run`, and the `dropped` entry in `_persist`.

- [ ] **Step 4: Run the tests and the full suite.** Expected: PASS, the existing paper tests included.

- [ ] **Step 5: Commit** (three commits)

```bash
git commit   # feat(cli): lab paper refuses more coins than the machine holds workers
git commit   # fix(cli): lab tail carries on past a failing coin and exits 1 at the end
git commit   # fix(paper): a coin the exchange no longer lists is dropped, not the run
```

---

### Task 10: Disclose the universe rule and the survivor bias (spec change 10)

**Files:**
- Modify: `config/lab.yaml` (new top-level `universe_rule`)
- Modify: `perp_lab/validate/pipeline.py` (`rec["universe_rule"]`, one line in `render_report`)
- Modify: `dashboard/app.py:613-614` (sidebar caption)
- Test: `tests/test_config.py`, `tests/test_walkforward_sandbox.py`, `tests/test_dashboard.py` (additions)

**Interfaces:**
- Config: `universe_rule: "Universe: 17 Binance USDⓈ-M USDT perpetuals, chosen 2026-10-18 as those that traded in every sampled month 2021-01 to 2026-06 and ranked top 50 by volume in at least 5 of 7 samples. Survivor bias: coins that died or faded before 2026 are not in it."`
- `validate_strategy` puts `"universe_rule": cfg["universe_rule"]` in the run record. `render_report` prints it as its own line right after the run line (`run ... folds`). The dashboard adds `st.sidebar.caption(CFG["universe_rule"])` after the holdout captions.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
def test_the_universe_rule_states_the_survivor_bias():
    assert "survivor bias" in CFG["universe_rule"].lower() and "2026-10-18" in CFG["universe_rule"]

# tests/test_walkforward_sandbox.py
def test_the_report_states_the_universe_rule(tmp_path, plugins):
    cfg = market_config(tmp_path, validation={"random_seeds": 2, "mc_iterations": 50})
    rec, report = pipeline.validate_strategy(cfg, "donchian_atr", "4h", workers=1)
    assert cfg["universe_rule"] in report and rec["universe_rule"] == cfg["universe_rule"]

# tests/test_dashboard.py
def test_sidebar_states_the_universe_rule(populated):
    at = AppTest.from_file("../dashboard/app.py", default_timeout=60)
    at.run()
    assert any(populated["universe_rule"] in c.value for c in at.sidebar.caption)
```

- [ ] **Step 2: Run them and watch them fail.** Expected: FAIL (`KeyError: 'universe_rule'`).

- [ ] **Step 3: Implement** the config key, the record field, the report line and the sidebar caption.

- [ ] **Step 4: Run the tests and the full suite.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git commit   # feat(report): state the universe rule and its survivor bias where results are shown
```

---

### Task 11: Switch the config to 17 coins from 2020, then backfill (spec changes 3-margin, 9, 11)

**Files:**
- Modify: `config/lab.yaml` (`margin_brackets`, `timeframes`, `history_start`, `universe`)
- Test: `tests/test_config.py` (additions)

**Interfaces:**
- Produces the final config. The universe and order are as in Global Constraint 2, with `history_start: "2020-01-01"` and `timeframes: [15m, 1h, 4h]`. `margin_brackets` keeps the 8 existing entries and adds the 9 from "Decisions this plan makes". The margin comment block is rewritten to state the rule, the endpoint and the read date, and to say that Binance's tier 1 caps at 10,000 USDT for most coins, so the bracket is the one our notionals reach. The old claim "our notionals stay in tier 1 at default equity" goes.

- [ ] **Step 1: Write the failing tests** (add to `tests/test_config.py`)

```python
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK",
         "ADA", "DOT", "LTC", "BCH", "FIL", "AAVE", "XLM", "ATOM", "ETC"]
BEFORE = {"BTCUSDT": 0.004, "ETHUSDT": 0.005, "SOLUSDT": 0.0065, "BNBUSDT": 0.0065, "XRPUSDT": 0.0065,
          "DOGEUSDT": 0.0065, "AVAXUSDT": 0.010, "LINKUSDT": 0.010}
NEW = {"ADAUSDT": 0.010, "DOTUSDT": 0.010, "LTCUSDT": 0.010, "BCHUSDT": 0.010, "FILUSDT": 0.015,
       "AAVEUSDT": 0.010, "XLMUSDT": 0.015, "ATOMUSDT": 0.010, "ETCUSDT": 0.010}


def test_the_universe_is_the_17_coins_from_2020():
    assert RAW["universe"] == COINS
    assert (RAW["history_start"], RAW["holdout_start"]) == ("2020-01-01", "2026-07-01")


def test_every_coin_has_its_own_bracket_and_none_is_kinder_than_before():
    br = RAW["margin_brackets"]
    assert {f"{c}USDT" for c in COINS} <= set(br)
    assert {s: br[s]["mmr"] for s in BEFORE} == BEFORE and {s: br[s]["mmr"] for s in NEW} == NEW
    assert all(br[s]["mmr"] >= br["default"]["mmr"] for s in NEW) and all(br[s]["cum"] == 0.0 for s in NEW)


def test_5m_is_not_a_research_timeframe():
    assert RAW["timeframes"] == ["15m", "1h", "4h"]
```

- [ ] **Step 2: Run them and watch them fail.** Expected: FAIL.

- [ ] **Step 3: Change the config in three commits, running the full suite after each.**
  1. Margin brackets: `feat(config): margin brackets for the 9 new coins, at the bracket our notionals reach`.
  2. Remove `5m` from `timeframes`: `chore(config): drop 5m from the research timeframes`. The commit body gives the spec's reasons: no plugin declares it, 15m already can't pay costs, no 5m data is stored, and a walk-forward pool caches about 15 GB at 5m over 6.5 years.
  3. `history_start` and `universe`: `feat(config): research history from 2020-01-01 over 17 coins`.
  If a test pinned the old universe or history start, fix it to read the config, never to pin the new list a second time, and say so in the commit body.

- [ ] **Step 4: Backfill.** Run: `.venv/bin/lab backfill --datasets klines,fundingRate` (all 17 coins, the 3 timeframes, from 2020-01-01).
  Expected: `"error": 0`. `absent` counts the months before each late coin's first file, including all of 2020-01 to 2020-08 for SOL. `missing` covers only the current month's funding, which has no daily files on the archive. Run it again. Expected: no `absent`, nearly everything `skip`, and no request for a month before a coin's first file. `du -sh data` comes to roughly 400–550 MB.

- [ ] **Step 5: Check the research data (reads research bars only; runs no plugin).** Write a throwaway script in the scratchpad. For each coin and each of 15m, 1h and 4h, call `load_bars(cfg, s, tf)` (default end = `holdout_start`) and print: the first bar's date; the bar count; every run of one or more whole UTC days without a candle (for README's data-gaps list); and, at 1h, the median and 95th percentile of `slippage` in bps per calendar year. Also check that `research_symbols(cfg, cfg.symbols, tf)` returns all 17 for each tf.
  Expected: first bars match the spec's "first month in archive" column; no slippage below 2.0 bp; BTC and ETH medians for 2024–2026 within 0.1 bp of 2.0; 2020 medians above later years for the thinner coins. Save the output to the scratchpad; Task 12 uses the gap list.

---

### Task 12: Documents (spec "Changes to existing documents")

**Files:**
- Modify: `docs/superpowers/specs/2026-10-16-judge-design.md:30`
- Modify: `README.md`
- Modify: `docs/research/2026-10-18-strategy-research.md` (top)
- Modify: `docs/superpowers/specs/2026-10-18-wider-data-design.md` (status line)

- [ ] **Step 1: Judge spec.** In the Decisions table, "Short holdout" row, replace "History start stays 2022-01-01." with "History start moved to 2020-01-01 by `2026-10-18-wider-data-design.md`."
- [ ] **Step 2: README.**
  - Add a "Universe" bullet under "Modelling choices": the `universe_rule` text, and that the coins started trading on different dates (each from its first bar).
  - Add a "Costs" bullet: per-coin point-in-time slippage, its formula, floor and cap, and a pointer to the spec section.
  - Rewrite the "Liquidation" bullet: one bracket per coin at the bracket our notionals reach, read 2026-10-18 from Binance's public brackets endpoint.
  - Replace "Data gaps" with the list from Task 11 step 5.
  - Add rows to "Rules the code enforces", each with its test file: no slippage below 2 bp (`tests/test_slippage.py`, `tests/test_config.py`); a parent run is reused only on the same data (`tests/test_data_identity.py`); gates scale with coins (`tests/test_coin_gates.py`); a coin without research bars is skipped (`tests/test_late_listing.py`).
  - Note in the Quickstart that `lab paper` refuses more coins than `client.max_workers`.
- [ ] **Step 3: Research note.** Under the title, add a dated note: "2026-10-<day>: the numbers below are for 8 coins from 2022-01-01. The research data now runs from 2020-01-01 over 17 coins (`docs/superpowers/specs/2026-10-18-wider-data-design.md`); a dated addendum re-derives the data-dependent numbers before `flow_shock_fade` runs."
- [ ] **Step 4: Spec status.** Change the first lines of the wider-data spec to "Status: approved 2026-10-18; built <date> on branch `feat/wider-data`. Change 7 is deferred to its own spec."
- [ ] **Step 5: Run the full suite one last time**, real sandbox included (main checkout): `.venv/bin/python -m pytest -p no:cacheprovider`. Expected: all pass.
- [ ] **Step 6: Commit** (one commit per document) and report to the user. Include the test count, the backfill stats and the slippage table. The next step is theirs: the research-note addendum, then `lab judge flow_shock_fade --tf 1h`.

---

## Spec coverage

| Spec item | Task |
|---|---|
| Decisions: history_start 2020-01-01, holdout unchanged, universe, costs fixed before trials, earlier trials count | 11, 4, 8 (N test) |
| Survivor bias disclosed | 10, 12 |
| Known facts: 2020 klines without header, funding from first month, OI NaN before metrics, zero-trade candles after delisting | existing `test_parse_klines_with_and_without_header`; 1 (OI); 2 (zero-trade) |
| Change 1 coins with no research bars | 1 |
| Change 2 listing dates; zero-trade candles absent | 2 |
| Change 3 slippage (function, calibration, documented before trials), margin brackets, paper unchanged | 3, 4, 5, 11 |
| Change 4 gates scale | 6 |
| Change 5 Monte Carlo and random entry | 7 |
| Change 6 runs remember their data | 8 |
| Change 7 deferred, guards unchanged | Global Constraint 6 |
| Change 8 paper refusal, BadSymbol per coin, tail per coin | 9 |
| Change 9 drop 5m | 11 |
| Change 10 disclosure | 10 |
| Change 11 config and backfill | 11 |
| What happens to existing results (verdicts kept, N, optional re-measure) | 8 (N test); re-measuring is the user's later call |
| Non-goals and anti-goals | Global Constraints 3–7 |
| Order of work (3 and 6 before any walk-forward on new data) | Tasks 3–5 and 8 come before 11; no walk-forward in this project |
| Changes to existing documents | 12 |
