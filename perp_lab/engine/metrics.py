"""Performance metrics. Sharpe/Sortino from daily returns, annualised with sqrt(365)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

ANN = 365.0


def daily_returns(equity: np.ndarray, time_ms: np.ndarray) -> pd.Series:
    """Daily close-to-close returns of an equity curve sampled at bar closes."""
    if len(equity) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(equity, index=pd.to_datetime(time_ms, unit="ms", utc=True))
    d = s.resample("1D").last().dropna()
    # first day's return measured from the starting equity (first bar's pre-trade value)
    prev = pd.concat([pd.Series([equity[0]], index=[d.index[0] - pd.Timedelta(days=1)]), d])
    return (prev.pct_change().dropna())


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * math.sqrt(ANN))


def sortino(r: pd.Series) -> float:
    r = r.dropna()
    dd = np.sqrt((np.minimum(r, 0) ** 2).mean()) if len(r) else 0
    if len(r) < 2 or dd == 0:
        return 0.0
    return float(r.mean() / dd * math.sqrt(ANN))


def max_drawdown_from_returns(r: pd.Series) -> float:
    if len(r) == 0:
        return 0.0
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1).min())


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or len(trades) == 0:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "avg_r": 0.0,
                "avg_bars_held": 0.0, "liquidations": 0, "fees": 0.0, "funding": 0.0}
    w = trades.pnl[trades.pnl > 0].sum()
    l = -trades.pnl[trades.pnl < 0].sum()
    return {
        "trades": int(len(trades)),
        "win_rate": float((trades.pnl > 0).mean()),
        "profit_factor": float(w / l) if l > 0 else float("inf") if w > 0 else 0.0,
        "avg_r": float(trades.r_multiple.mean()),
        "avg_bars_held": float(trades.bars_held.mean()),
        "liquidations": int((trades.exit_reason == "liq").sum()),
        "fees": float(trades.fees.sum()),
        "funding": float(trades.funding.sum()),
    }


def summarize(r: pd.Series, trades: pd.DataFrame | None = None, exposure: float | None = None) -> dict:
    tot = float((1 + r).prod() - 1) if len(r) else 0.0
    yrs = len(r) / ANN if len(r) else 0
    out = {
        "sharpe": sharpe(r), "sortino": sortino(r), "max_dd": max_drawdown_from_returns(r),
        "total_return": tot,
        "cagr": float((1 + tot) ** (1 / yrs) - 1) if yrs > 0 and tot > -1 else 0.0,
        "days": int(len(r)),
    }
    if exposure is not None:
        out["exposure"] = float(exposure)
    if trades is not None:
        out.update(trade_stats(trades))
    return out


def monthly_table(r: pd.Series) -> pd.DataFrame:
    if len(r) == 0:
        return pd.DataFrame()
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    t = m.unstack()
    t.index.name, t.columns.name = "year", "month"
    return t


def portfolio_returns(per_symbol: dict[str, pd.Series]) -> pd.Series:
    """Equal capital per symbol, rebalanced daily: mean of available symbol returns."""
    if not per_symbol:
        return pd.Series(dtype=float)
    df = pd.concat(per_symbol, axis=1).sort_index()
    return df.mean(axis=1, skipna=True)
