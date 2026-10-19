"""Monte Carlo on trade order → max-drawdown distribution.

Trades are pooled across symbols in entry order; each trade contributes
pnl_pct / (coins listed on its entry day) to the portfolio (equal capital per
listed coin, as the daily portfolio averages). Shuffling the order keeps the
edge identical and shows how much of the realised DD was luck.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def coins_per_day(daily: dict[str, pd.Series]) -> pd.Series:
    """Number of coins with a return on each UTC day."""
    return pd.concat([pd.Series(1, index=d.index) for d in daily.values()], axis=1).sum(axis=1).sort_index()


def mc_drawdown(trades: pd.DataFrame, coins_per_day: pd.Series, iterations: int = 10_000, seed: int = 7) -> dict:
    if trades is None or len(trades) < 5:
        return {"n_trades": 0 if trades is None else int(len(trades)), "note": "too few trades"}
    t = trades.sort_values("entry_time")
    days = pd.DatetimeIndex(pd.to_datetime(t.entry_time.to_numpy(), unit="ms", utc=True)).floor("D")
    cpd = coins_per_day[~coins_per_day.index.duplicated()]
    full = cpd.reindex(cpd.index.union(days.unique())).ffill().bfill()
    n = full.to_numpy()[full.index.get_indexer(days)]
    r = t.pnl_pct.to_numpy() / n
    rng = np.random.default_rng(seed)

    def mdd(x):
        eq = np.cumprod(1 + x, axis=-1)
        return (eq / np.maximum.accumulate(eq, axis=-1) - 1).min(axis=-1)

    actual = float(mdd(r))
    dds = np.empty(iterations)
    chunk = 1000
    for a in range(0, iterations, chunk):
        m = min(chunk, iterations - a)
        perm = rng.permuted(np.tile(r, (m, 1)), axis=1)
        dds[a:a + m] = mdd(perm)
    return {
        "n_trades": int(len(r)), "actual_dd": actual,
        "p50_dd": float(np.percentile(dds, 50)), "p5_dd": float(np.percentile(dds, 5)),
        "p1_dd": float(np.percentile(dds, 1)),
        "actual_dd_pctile": float((dds <= actual).mean()),
    }
