"""Holdout exam: one run with frozen parameters on the locked window, in the judge's process.

The token is checked by object identity and is invalid in any other process, so the exam never
uses the walk-forward worker pool. Indicators warm up on research data (each run starts at
history_start); only returns and trades inside [holdout_start, end) are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from perp_lab.config import TF_MS
from perp_lab.data.store import load_bars
from perp_lab.engine.backtest import EngineParams, run_backtest
from perp_lab.engine.metrics import daily_returns, portfolio_returns
from perp_lab.holdout import check_token
from perp_lab.strategy import load_strategy_class

DEFAULT_FUNDING_H = 8.0


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


def run_exam(cfg, name: str, tf: str, symbols: list[str], params: dict, token, end: pd.Timestamp) -> ExamResult:
    check_token(token)
    cls = load_strategy_class(name)
    start = cfg.holdout_start
    daily, parts = {}, []
    for s in symbols:
        bars = load_bars(cfg, s, tf, end=end, holdout=token)
        strat = cls(**params)
        res = run_backtest(strat, bars, strat.prepare(bars), symbol=s, tf=tf, ep=EngineParams.from_config(cfg, s))
        dr = daily_returns(res.equity, res.time)
        daily[s] = dr[(dr.index >= start) & (dr.index < end)] if len(dr) else dr
        t = pd.DataFrame(res.trades)
        if len(t):
            st = pd.to_datetime(t.signal_time, unit="ms", utc=True)
            parts.append(t[(st >= start) & (st < end)])
    trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return ExamResult(start, end, dict(params), daily, portfolio_returns(daily), trades)
