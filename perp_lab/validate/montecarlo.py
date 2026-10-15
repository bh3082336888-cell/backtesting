"""Monte Carlo on trade order → max-drawdown distribution.

Trades are pooled across symbols in entry order; each trade contributes
pnl_pct / n_symbols to the portfolio (equal capital per symbol). Shuffling the
order keeps the edge identical and shows how much of the realised DD was luck.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mc_drawdown(trades: pd.DataFrame, n_symbols: int, iterations: int = 10_000, seed: int = 7) -> dict:
    if trades is None or len(trades) < 5:
        return {"n_trades": 0 if trades is None else int(len(trades)), "note": "too few trades"}
    t = trades.sort_values("entry_time")
    r = (t.pnl_pct.to_numpy() / n_symbols)
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
