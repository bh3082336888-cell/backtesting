"""Look 6 (trade-count bounds for the chosen spec's grid; no forward data).

For each grid combo of the flow-shock fade at 1h (move_k = 1, flow_x in {0.5, 1.0}, hold_bars in
{6, 12, 24}), over the walk-forward OOS span 2022-07-01 -> 2026-07-01:
  upper bound = every bar whose trigger is true (as if each could be taken);
  lower bound = greedy: take a trigger at t, skip triggers at t+1..t+H (the position is held H
                bars from the open of t+1), take the next one after that. A stop exit only
                shortens a hold, so it can only add trades.
No price after a trigger bar is read: neither bound needs to know how a trade ended.
Also: the pooled count per 6-month window (min_train_trades = 20) and per 2-month OOS fold,
from the lower bound."""

import numpy as np
import pandas as pd

from common import SYMBOLS, bars
from perp_lab.indicators import atr

pd.set_option("display.width", 220)
A, B = pd.Timestamp("2022-07-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")


def signals(b, x, k=1.0, n=168):
    vmed = b.volume.shift(1).rolling(n, min_periods=n).median()
    flow = (2 * b.taker_buy_volume - b.volume) / vmed.replace(0, np.nan)
    mv = (b.close - b.open) / atr(b, 14).shift(1)
    s = np.where((flow <= -x) & (mv <= -k), 1, np.where((flow >= x) & (mv >= k), -1, 0))
    return pd.Series(s, index=b.index)


def greedy(sig: pd.Series, h: int) -> pd.Series:
    idx = np.flatnonzero(sig.values != 0)
    taken, nxt = [], -1
    for i in idx:
        if i >= nxt:
            taken.append(i)
            nxt = i + h + 1
    return sig.iloc[taken]


rows, pooled = [], {}
for x in [0.5, 1.0]:
    for h in [6, 12, 24]:
        ub, lb, longs = [], [], 0
        allt = []
        for s in SYMBOLS:
            sig = signals(bars(s, "1h"), x)
            g = greedy(sig, h)
            sig_w, g_w = sig[(sig.index >= A) & (sig.index < B)], g[(g.index >= A) & (g.index < B)]
            ub.append(int((sig_w != 0).sum()))
            lb.append(len(g_w))
            longs += int((g_w > 0).sum())
            allt.append(g_w.index.to_series())
        ts = pd.concat(allt)
        p6 = ts.groupby([ts.index.year, (ts.index.month - 1) // 6]).size()
        p2 = ts.groupby([ts.index.year, (ts.index.month - 1) // 2]).size()
        rows.append({"flow_x": x, "hold": h,
                     "per_symbol_lower": f"{int(np.median(lb))} [{min(lb)},{max(lb)}]",
                     "per_symbol_upper": f"{int(np.median(ub))} [{min(ub)},{max(ub)}]",
                     "total_lower": sum(lb), "long_share": round(longs / max(sum(lb), 1), 2),
                     "pooled_6m_min": int(p6.min()), "pooled_2m_min": int(p2.min())})
print("Trades per symbol over 2022-07 -> 2026-07, median [min, max] over symbols; pooled minima per window:")
print(pd.DataFrame(rows).to_string(index=False))
