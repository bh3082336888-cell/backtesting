"""Walk-forward: rolling train/test, OOS-only headline.

Mechanics (and why):
  * Every grid combo is run once, continuously, over the whole research span
    [history_start, holdout_start) per symbol. prepare() is causal (tested), so a
    continuous run is equivalent to re-running each window with warmup — and
    ~#folds times cheaper.
  * Fold k: score every combo on its TRAIN window (pooled: mean per-symbol daily
    Sharpe; combos with fewer pooled entries than min_train_trades_per_coin ×
    coins listed in the train window score -inf). Pick the
    best. One param set for the whole universe per fold — per-symbol selection
    triples the overfitting surface for little gain.
  * OOS for fold k = the selected combo's daily returns on the TEST window, and
    its trades whose *signal* falls in the test window. Selection never sees
    test data. Stitched across folds → the OOS track record.
  * Caveat: a position opened by the selected combo just before a test window
    carries into it (it is the same continuous run). No lookahead — its entry was
    decided on pre-test data — but it's not a cold start. Documented, accepted.
  * Plugin code runs only in sandbox workers. The process pool runs trusted code:
    each task is a one-symbol chunk of jobs in one `client.Batch`, whose decision
    logs the trusted engine replays. Every run measures the same `PluginSource`.
"""

from __future__ import annotations

import itertools
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from perp_lab.config import TF_MS, Config, load_config
from perp_lab.data.store import load_bars, research_symbols
from perp_lab.engine.backtest import BacktestResult, EngineParams
from perp_lab.engine.metrics import daily_returns, portfolio_returns, sharpe, summarize
from perp_lab.plugins import PluginSource
from perp_lab.runs import per_coin_min
from perp_lab.session_window import in_session
from perp_lab.sandbox import client

log = logging.getLogger(__name__)


def make_folds(start: pd.Timestamp, end: pd.Timestamp, train_m: int, test_m: int) -> list[dict]:
    folds = []
    k = 0
    ts = start + pd.DateOffset(months=train_m)
    while ts + pd.DateOffset(months=test_m) <= end:
        folds.append({"fold": k, "train_start": ts - pd.DateOffset(months=train_m), "train_end": ts,
                      "test_start": ts, "test_end": ts + pd.DateOffset(months=test_m)})
        ts = ts + pd.DateOffset(months=test_m)
        k += 1
    return folds


def grid_combos(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def combo_key(c: dict) -> str:
    return ",".join(f"{k}={c[k]}" for k in sorted(c)) or "default"


# ---------------------------------------------------------------- worker side

_CFG_PATH: str | None = None


def _init_worker(cfg_path: str):
    global _CFG_PATH
    _CFG_PATH = cfg_path


@lru_cache(maxsize=None)
def _cfg() -> Config:
    return load_config(_CFG_PATH)


@lru_cache(maxsize=16)
def _bars(symbol: str, tf: str) -> pd.DataFrame:
    return load_bars(_cfg(), symbol, tf)


LEVEL_COLS = ("level_time", "level_stop", "level_target")


def _compact(res: BacktestResult, symbol: str, params: dict, seed, tf: str, keep_levels: bool = False,
             session: dict | None = None) -> dict:
    """What a walk-forward keeps of one continuous run. The trades' stop/target path (`LEVEL_COLS`)
    only with `keep_levels`: only the walk-forward grid's trades are charted. `flat_bars` counts the
    bars flat at their close and, for a plugin that declares a `session`, closing inside it: the
    random-entry benchmark fires at the strategy's frequency over the bars it could trade."""
    dr = daily_returns(res.equity, res.time)
    idx = pd.to_datetime(res.time, unit="ms", utc=True)
    expo = pd.Series(res.in_pos.astype(float), index=idx).resample("1D").mean()
    flat = ~res.in_pos
    if session:
        flat = flat & in_session(np.asarray(res.time, dtype=np.int64) + TF_MS[tf], session)
    flat_daily = pd.Series(flat.astype(float), index=idx).resample("1D").sum()
    trades = pd.DataFrame(res.trades)
    if not keep_levels:
        trades = trades.drop(columns=[c for c in LEVEL_COLS if c in trades.columns])
    return {"symbol": symbol, "params": params, "seed": seed, "daily": dr, "exposure": expo,
            "flat_bars": flat_daily, "trades": trades, "rejected": res.rejected}


def run_chunk(args) -> list[dict]:
    """Pool task: one chunk of jobs, all on one symbol, run in one sandbox `Batch` and replayed by
    the trusted engine. The Batch is opened and closed here, by the thread that runs the chunk (its
    worker dies with that thread), and its worker is sent this one symbol's bars only."""
    src, symbol, tf, jobs, stderr_dir, keep_levels = args
    if any(j[1] != symbol or j[2] != tf for j in jobs):
        raise ValueError(f"a chunk holds jobs for one symbol and timeframe only ({symbol} {tf})")
    cfg = _cfg()
    out = []
    with client.Batch(cfg, src, symbol, tf, _bars(symbol, tf), stderr_dir=stderr_dir) as b:
        for params, _, _, overrides, seed, rand in jobs:
            ep = EngineParams.from_config(cfg, symbol, **(overrides or {}))
            res = b.backtest(params, ep, random=None if seed is None else (seed, *rand))
            out.append(_compact(res, symbol, params, seed, tf, keep_levels, src.meta.session))
    return out


def _pieces(jobs: list[tuple], workers: int) -> list[tuple[str, str, list[int]]]:
    """`chunks`, as indices into `jobs`."""
    groups: dict[tuple[str, str], list[int]] = {}
    for i, j in enumerate(jobs):
        groups.setdefault((j[1], j[2]), []).append(i)
    per = max(1, math.ceil(len(jobs) / max(2 * workers, len(groups))))
    return [(s, tf, idx[k:k + per]) for (s, tf), idx in groups.items() for k in range(0, len(idx), per)]


def chunks(jobs: list[tuple], workers: int) -> list[tuple[str, str, list]]:
    """Jobs grouped by symbol in order, each group cut into pieces of
    `per = ceil(len(jobs) / max(2 * workers, n_symbols))` jobs, enough to keep `workers` busy. A
    chunk never mixes symbols, so a sandbox worker never sees a second one."""
    return [(s, tf, [jobs[i] for i in idx]) for s, tf, idx in _pieces(jobs, workers)]


def run_many(cfg: Config, src: PluginSource, jobs: list[tuple], workers: int | None = None,
             stderr_dir: Path | None = None, keep_levels: bool = False) -> list[dict]:
    """Every job, a `(params, symbol, tf, overrides, seed, rand)` (rand None, or
    (entry_prob, long_share)), through the sandbox; the results in job order. The pool runs trusted code only: each task is one chunk in one Batch.
    The trades keep their stop/target path only with `keep_levels` (the walk-forward grid's, which
    become the charted OOS trades): sensitivity reads daily returns only and the random-entry
    benchmark signal_time only, and the lists would cross the pool for nothing."""
    if not jobs:
        return []
    n = max(1, min(workers or os.cpu_count() or 1, client.max_workers(cfg)))
    pieces = _pieces(jobs, n)
    tasks = [(src, s, tf, [jobs[i] for i in idx], stderr_dir, keep_levels) for s, tf, idx in pieces]
    if n == 1:
        _init_worker(str(cfg.path))
        done = [run_chunk(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(n, len(tasks)), initializer=_init_worker,
                                 initargs=(str(cfg.path),)) as ex:
            try:
                done = list(ex.map(run_chunk, tasks, chunksize=1))
            except BaseException:
                ex.shutdown(cancel_futures=True)    # a failed chunk fails the run: don't wait out the rest
                raise
    out: list = [None] * len(jobs)
    for (_, _, idx), results in zip(pieces, done):
        for i, r in zip(idx, results):
            out[i] = r
    return out


# ---------------------------------------------------------------- orchestration

def _window(s: pd.Series, a, b) -> pd.Series:
    return s[(s.index >= a) & (s.index < b)]


def _trades_in(tr: pd.DataFrame, a, b) -> pd.DataFrame:
    if tr is None or tr.empty:
        return tr if tr is not None else pd.DataFrame()
    st = pd.to_datetime(tr.signal_time, unit="ms", utc=True)
    return tr[(st >= a) & (st < b)]


@dataclass
class WFResult:
    strategy: str
    tf: str
    symbols: list[str]
    folds: list[dict]
    oos_daily: dict[str, pd.Series]
    oos_portfolio: pd.Series
    oos_trades: pd.DataFrame
    oos_exposure: dict[str, float]
    oos_summary: dict
    oos_by_symbol: dict[str, dict]
    is_summary: dict
    grid_table: pd.DataFrame          # per combo: fixed-param metrics over the OOS span (diagnostic)
    selected: list[dict]
    oos_flat_bars: dict[str, float]
    # The plugin bytes every run measured, and where its workers' output went; sensitivity and
    # the random-entry benchmark run the same bytes and write there too.
    src: PluginSource = field(repr=False)
    stderr_dir: Path | None = field(repr=False)
    raw: dict = field(repr=False, default_factory=dict)


def listed_coins(daily: dict[str, pd.Series], a, b) -> int:
    """Coins with at least one daily return in [a, b): the ones that could trade there."""
    return sum(len(_window(d, a, b)) > 0 for d in daily.values())


def stitch_oos(R: dict, selected: list[dict], symbols: list[str]) -> dict[str, pd.Series]:
    """Per-symbol OOS daily returns: each fold's selected combo on its test window, joined. `R` is
    keyed `(combo, symbol)`; the before-costs pass stitches its own re-run the same way."""
    parts = {s: [_window(R[(x["combo"], s)]["daily"], x["test_start"], x["test_end"]) for x in selected]
             for s in symbols}
    return {s: pd.concat(p).sort_index() for s, p in parts.items()}


def walkforward(cfg: Config, src: PluginSource, tf: str, symbols: list[str] | None = None,
                workers: int | None = None, stderr_dir: Path | None = None) -> WFResult:
    symbols = research_symbols(cfg, symbols or cfg.symbols, tf)
    if not symbols:
        raise ValueError(f"no coin has {tf} research bars; run `lab backfill`")
    wf = cfg["walkforward"]
    combos = grid_combos(src.meta.grid)
    log.info("walkforward %s %s: %d combos x %d symbols", src.name, tf, len(combos), len(symbols))
    jobs = [(c, s, tf, None, None, None) for c in combos for s in symbols]
    results = run_many(cfg, src, jobs, workers, stderr_dir, keep_levels=True)
    R: dict[tuple[str, str], dict] = {(combo_key(r["params"]), r["symbol"]): r for r in results}
    keys = [combo_key(c) for c in combos]
    cmap = dict(zip(keys, combos))

    folds = make_folds(cfg.history_start, cfg.holdout_start, wf["train_months"], wf["test_months"])
    if not folds:
        raise ValueError("not enough history for one walk-forward fold")
    expo_parts = {s: [] for s in symbols}
    flat_parts = {s: 0.0 for s in symbols}
    trade_parts, is_parts, selected = [], [], []

    for f in folds:
        a, b = f["train_start"], f["train_end"]
        best_k, best_score, best_n = None, -np.inf, 0
        min_train = per_coin_min(wf["min_train_trades_per_coin"],
                                 listed_coins({s: R[(keys[0], s)]["daily"] for s in symbols}, a, b))
        for k in keys:
            n_tr = sum(len(_trades_in(R[(k, s)]["trades"], a, b)) for s in symbols)
            if n_tr < min_train:
                score = -np.inf
            else:
                score = float(np.mean([sharpe(_window(R[(k, s)]["daily"], a, b)) for s in symbols]))
            if score > best_score or best_k is None:
                best_k, best_score, best_n = k, score, n_tr
        ta, tb = f["test_start"], f["test_end"]
        selected.append({**f, "params": cmap[best_k], "combo": best_k,
                         "is_score": best_score, "is_trades": best_n})
        is_parts.append(portfolio_returns({s: _window(R[(best_k, s)]["daily"], a, b) for s in symbols}))
        for s in symbols:
            r = R[(best_k, s)]
            expo_parts[s].append(_window(r["exposure"], ta, tb))
            flat_parts[s] += float(_window(r["flat_bars"], ta, tb).sum())
            t = _trades_in(r["trades"], ta, tb)
            if len(t):
                trade_parts.append(t.assign(fold=f["fold"], combo=best_k))

    oos_daily = stitch_oos(R, selected, symbols)
    oos_port = portfolio_returns(oos_daily)
    oos_trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    oos_expo = {s: float(pd.concat(p).mean()) for s, p in expo_parts.items()}

    by_sym = {}
    for s in symbols:
        ts = oos_trades[oos_trades.symbol == s] if len(oos_trades) else oos_trades
        by_sym[s] = summarize(oos_daily[s], ts, oos_expo[s])
    oos_summary = summarize(oos_port, oos_trades, float(np.mean(list(oos_expo.values()))))
    is_port = pd.concat(is_parts)
    is_port = is_port[~is_port.index.duplicated()]   # overlapping train windows
    is_summary = summarize(is_port)

    # Fixed-param diagnostic over the OOS span (heatmap source; NOT an OOS result).
    oa, ob = folds[0]["test_start"], folds[-1]["test_end"]
    rows = []
    for k in keys:
        pr = portfolio_returns({s: _window(R[(k, s)]["daily"], oa, ob) for s in symbols})
        n = sum(len(_trades_in(R[(k, s)]["trades"], oa, ob)) for s in symbols)
        rows.append({**cmap[k], "combo": k, "sharpe": sharpe(pr),
                     "total_return": float((1 + pr).prod() - 1), "trades": n,
                     "times_selected": sum(1 for x in selected if x["combo"] == k)})
    grid_table = pd.DataFrame(rows)

    return WFResult(src.name, tf, symbols, folds, oos_daily, oos_port, oos_trades, oos_expo,
                    oos_summary, by_sym, is_summary, grid_table, selected, flat_parts,
                    src=src, stderr_dir=stderr_dir, raw={"R": R, "oos_span": (oa, ob)})


def modal_params(wf: WFResult) -> dict:
    counts = pd.Series([x["combo"] for x in wf.selected]).value_counts()
    top = counts.index[0]
    return next(x["params"] for x in wf.selected if x["combo"] == top)
