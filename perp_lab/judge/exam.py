"""Holdout exam: one run with frozen parameters on the locked window, in the judge's process.

The token is checked by object identity and is invalid in any other process, so the exam never
uses the walk-forward worker pool, and the holdout bars are loaded here. The plugin is streamed
(`perp_lab.sandbox.stream`): per symbol, a fresh sandbox worker is sent the research bars, then one
holdout bar at a time, and the trusted engine runs on the plugin's decisions; symbols go in batches
of at most `client.max_workers`, each batch in lockstep. Indicators warm up on research data (each
run starts at history_start); only returns and trades inside [holdout_start, end) are kept.

`cost_hours` estimates an exam's CPU time per symbol before any read is spent, from research data
only, so a plugin too slow to stream is blocked without one.

After each batch, whatever its outcome, and before the next batch's workers start, the files the
batch's workers mapped (their libraries) are evicted from the page cache: a worker can fault in
chosen pages of them, which a later batch's workers, holding earlier bars, or a later exam could
read back (spec R7). `stream.run_batch` does the same before a rerun.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from perp_lab.config import TF_MS
from perp_lab.data.store import load_bars
from perp_lab.engine.backtest import EngineParams
from perp_lab.engine.metrics import daily_returns, portfolio_returns
from perp_lab.holdout import check_token
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client, stream

DEFAULT_FUNDING_H = 8.0
COST_PROBES = 3                         # timed `prepare` calls; the estimate uses their median


@dataclass
class ExamResult:
    start: pd.Timestamp
    end: pd.Timestamp
    params: dict
    daily: dict[str, pd.Series]
    portfolio: pd.Series
    trades: pd.DataFrame


def _files(d: Path) -> list[Path]:
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {d}; run `lab backfill`")
    return files


def window_end(cfg, symbols: list[str], tf: str) -> pd.Timestamp:
    """The last 00:00 UTC up to which every symbol has closed candles and funding.

    Reads timestamps only (open_time, funding_time, interval_hours), never prices."""
    ends = []
    for s in symbols:
        last_open = max(int(pd.read_parquet(f, columns=["open_time"]).open_time.max())
                        for f in _files(cfg.data_dir / s / tf))
        fund = (pd.concat([pd.read_parquet(f, columns=["funding_time", "interval_hours"])
                           for f in _files(cfg.data_dir / s / "funding")])
                .sort_values("funding_time").iloc[-1])
        ih = fund.interval_hours if pd.notna(fund.interval_hours) else DEFAULT_FUNDING_H
        ends.append(min(last_open + TF_MS[tf], int(fund.funding_time) + int(ih * 3_600_000)))
    return pd.Timestamp(min(ends), unit="ms", tz="UTC").floor("D")


def cost_hours(cfg, src: PluginSource, tf: str, symbols: list[str], params: dict, end: pd.Timestamp) -> float:
    """Estimated CPU hours of one symbol's streamed exam on [holdout_start, end): the median CPU time
    of COST_PROBES `prepare` calls on the first symbol's full research bars, in a worker of its own,
    times the window's bar count. Each stream step runs one `prepare` over everything held, so this
    is a slight underestimate. The bar count comes from the window's length: no holdout data is read."""
    sym = symbols[0]
    bars = load_bars(cfg, sym, tf)                         # research bars only: no token
    n = max(0, (end - cfg.holdout_start) // pd.Timedelta(milliseconds=TF_MS[tf]))

    def probe() -> list[float]:
        with client.spawn(cfg, src) as w:
            w.bars(sym, tf, bars)
            return [w.prepare(params) for _ in range(COST_PROBES)]
    return float(np.median(client._retry_once(probe))) * n / 3600


def run_exam(cfg, src: PluginSource, tf: str, symbols: list[str], params: dict, token, end: pd.Timestamp,
             stderr_dir: Path | None = None) -> ExamResult:
    """The streamed exam of `src` with frozen `params` on [holdout_start, end). A plugin failure is
    a PluginError; each worker's stderr goes to `stderr_dir` when it closes. After each batch,
    whatever the outcome, the files its workers mapped are evicted from the page cache."""
    check_token(token)
    start = cfg.holdout_start
    start_ms = start.value // 10**6

    def load(s: str):
        bars = load_bars(cfg, s, tf, end=end, holdout=token)
        k0 = int(np.searchsorted(bars["open_time"].to_numpy(), start_ms))    # the first holdout bar
        return bars, k0, EngineParams.from_config(cfg, s)

    runs = {}
    per = client.max_workers(cfg)
    for b in range(0, len(symbols), per):
        mapped: list = []
        try:
            runs.update(stream.run_batch(cfg, src, tf, params, symbols[b:b + per], load, stderr_dir, mapped))
        finally:
            stream.evict(mapped)            # before the next batch's workers start
    daily, parts = {}, []
    for s in symbols:
        res = runs[s]
        dr = daily_returns(res.equity, res.time)
        daily[s] = dr[(dr.index >= start) & (dr.index < end)] if len(dr) else dr
        t = pd.DataFrame(res.trades)
        if len(t):
            st = pd.to_datetime(t.signal_time, unit="ms", utc=True)
            parts.append(t[(st >= start) & (st < end)])
    trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return ExamResult(start, end, dict(params), daily, portfolio_returns(daily), trades)
