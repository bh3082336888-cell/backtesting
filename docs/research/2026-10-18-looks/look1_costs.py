"""Look 1 (feasibility, unconditional): bar range, ATR and the typical absolute move over h bars,
against the engine's round-trip costs, per timeframe. Every statistic is over ALL bars of the
research span (2022-01-01 -> 2026-07-01); nothing is conditioned on any candidate signal, so this
measures how big moves are, not whether any rule predicts them."""

import numpy as np
import pandas as pd

from common import SYMBOLS, TFS, bars
from perp_lab.indicators import atr

H = [1, 2, 4, 8, 12, 20]
COST_STOP = 2 * (5 + 2)        # taker + slippage, both sides
COST_TARGET = 5 + 2 + 2        # taker + slippage in, maker out

rows = []
for tf in TFS:
    for s in SYMBOLS:
        b = bars(s, tf)
        c = b.close
        rng = (b.high - b.low) / c * 1e4
        a = atr(b, 14) / c * 1e4
        r = {"tf": tf, "sym": s.replace("USDT", ""), "bars": len(b),
             "range_med": rng.median(), "atr_med": a.median()}
        lc = np.log(c)
        for h in H:
            mv = (lc.shift(-h) - lc).abs() * 1e4      # unconditional |move| over h bars
            r[f"mv{h}"] = mv.median()
        rows.append(r)
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("Per symbol medians (bp):")
print(df.round(0).to_string(index=False))
print()
agg = df.groupby("tf")[[c for c in df.columns if c not in ("tf", "sym", "bars")]].median().reindex(TFS)
print("Median across the 8 symbols (bp):")
print(agg.round(0).to_string())
print()
print(f"round trip: stop exit {COST_STOP} bp, target exit {COST_TARGET} bp")
for tf in TFS:
    g = agg.loc[tf]
    print(f"{tf}: cost/range {COST_STOP / g.range_med:.2f}; cost as R for stops of 1/2/3 ATR: "
          f"{COST_STOP / g.atr_med:.3f} / {COST_STOP / (2 * g.atr_med):.3f} / {COST_STOP / (3 * g.atr_med):.3f}; "
          + "; ".join(f"cost/mv{h} {COST_STOP / g[f'mv{h}']:.2f}" for h in H))
