"""Look 5 (frequency and distribution of the net-taker-flow shock; no forward data).

  F_t   = (2 * taker_buy_volume_t - volume_t) / median(volume_{t-W..t-1})   (W = 7 days of bars)
  mv_t  = (close_t - open_t) / ATR14_{t-1}

A long (fade) trigger is F <= -x and mv <= -k; a short trigger is F >= x and mv >= k. This look
counts how often those triggers are true per symbol per year, pooled per 6-month window, and how
many fire in the same bar across symbols, on the research span only. Only bar t and earlier bars
are read; nothing after the trigger bar.

It also counts how often the planned default (x=0.5, k=1) fires on the harness's synthetic bars
(perp_lab.validate.causality.synthetic_bars(seed=1), with and without with_moving_columns): the
judge cannot verify a plugin whose defaults never signal there. Synthetic data, not market data.
"""

import numpy as np
import pandas as pd

from common import SYMBOLS, bars
from perp_lab.indicators import atr
from perp_lab.validate.causality import synthetic_bars, with_moving_columns

pd.set_option("display.width", 220)
PER_DAY = {"15m": 96, "1h": 24}
YRS = 4.5


def feats(b: pd.DataFrame, w: int) -> pd.DataFrame:
    vmed = b.volume.shift(1).rolling(w, min_periods=w).median()
    return pd.DataFrame({"F": (2 * b.taker_buy_volume - b.volume) / vmed.replace(0, np.nan),
                         "mv": (b.close - b.open) / atr(b, 14).shift(1)}, index=b.index)


def fires(f, x, k):
    return ((f.F <= -x) & (f.mv <= -k)) | ((f.F >= x) & (f.mv >= k))


for tf in ["1h", "15m"]:
    Fs = {s: feats(bars(s, tf), 7 * PER_DAY[tf]) for s in SYMBOLS}
    P = pd.concat(Fs.values())
    print(f"\n=== {tf}: F quantiles pooled over symbols: "
          + ", ".join(f"p{q:g} {P.F.quantile(q / 100):.3f}" for q in [0.1, 1, 5, 50, 95, 99, 99.9]))
    print(f"{tf}: share of bars with mv agreeing in sign with F: {float((np.sign(P.F) == np.sign(P.mv)).mean()):.3f}")
    tab = []
    for x in [0.25, 0.5, 1.0, 2.0]:
        r = {"x": x}
        for k in [0.0, 1.0, 2.0, 3.0]:
            per = [float(fires(f, x, k).sum()) / YRS for f in Fs.values()]
            r[f"k={k:g}"] = f"{np.median(per):.0f} [{min(per):.0f},{max(per):.0f}]"
        tab.append(r)
    print(f"{tf}: fires per symbol per year, median [min, max] over symbols (rows x, columns k):")
    print(pd.DataFrame(tab).to_string(index=False))
    if tf == "1h":
        for x, k in [(0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (2.0, 2.0), (1.0, 3.0)]:
            ts = pd.concat([f.index[fires(f, x, k)].to_series() for f in Fs.values()])
            per6 = ts.groupby([ts.index.year, (ts.index.month - 1) // 6]).size()
            longs = sum(int(((f.F <= -x) & (f.mv <= -k)).sum()) for f in Fs.values())
            print(f"  x={x} k={k}: pooled per 6-month window min {per6.min()} median {per6.median():.0f} "
                  f"max {per6.max()}; long (fade-down) share {longs / max(len(ts), 1):.2f}")
        M = pd.DataFrame({s: fires(f, 1.0, 2.0) for s, f in Fs.items()}).fillna(False)
        cnt = M.sum(axis=1)
        tot = M.values.sum()
        print("  breadth at x=1,k=2: share of fires sharing their bar with >=1/>=3/>=5 other symbols: "
              + " / ".join(f"{float((M.values & (cnt.values[:, None] >= m + 1)).sum() / tot):.2f}" for m in [1, 3, 5])
              + f"; distinct bars {int((cnt > 0).sum())} for {int(tot)} fires")
        A = pd.concat([pd.DataFrame({"a": atr(bars(s, tf), 14) / bars(s, tf).close * 1e4, "fire": fires(f, 1.0, 2.0)})
                       for s, f in Fs.items()])
        print(f"  ATR14 at t (bp of price) on fire bars x=1,k=2: p10/p50/p90 = "
              f"{A.a[A.fire].quantile(.1):.0f}/{A.a[A.fire].median():.0f}/{A.a[A.fire].quantile(.9):.0f} "
              f"vs all bars {A.a.quantile(.1):.0f}/{A.a.median():.0f}/{A.a.quantile(.9):.0f}")

print("\nSynthetic bars (harness's synthetic_bars(seed=1), 3000 hourly bars), W = 168:")
sb = synthetic_bars(seed=1)
for label, b in [("raw (taker columns NaN)", sb), ("with_moving_columns", with_moving_columns(sb))]:
    f = feats(b, 168)
    print(f"  {label}: fires at x=0.5,k=1: {int(fires(f, 0.5, 1.0).sum())}; x=0.5,k=0: {int(fires(f, 0.5, 0.0).sum())}; "
          f"x=1,k=2: {int(fires(f, 1.0, 2.0).sum())}; max |F| {np.nanmax(np.abs(f.F.values)) if f.F.notna().any() else float('nan'):.2f}")

# Addendum: where the grid thresholds sit in the 1h net-flow distribution, and how rare flow_x=2 is
# per pooled 6-month window at move_k=1 (frequencies only).
Fs = {s: feats(bars(s, "1h"), 168) for s in SYMBOLS}
P = pd.concat(Fs.values())
for x in [0.5, 1.0, 2.0]:
    print(f"1h: share of bars with flow <= -{x}: {float((P.F <= -x).mean()):.4f}; with flow >= {x}: {float((P.F >= x).mean()):.4f}")
ts = pd.concat([f.index[fires(f, 2.0, 1.0)].to_series() for f in Fs.values()])
p6 = ts.groupby([ts.index.year, (ts.index.month - 1) // 6]).size()
print(f"1h x=2 k=1: pooled per 6-month window min {p6.min()} median {p6.median():.0f} max {p6.max()} "
      f"({dict((f'{a}H{b + 1}', int(n)) for (a, b), n in p6.items())})")
for x in [0.5, 1.0]:
    ts = pd.concat([f.index[fires(f, x, 1.0)].to_series() for f in Fs.values()])
    p6 = ts.groupby([ts.index.year, (ts.index.month - 1) // 6]).size()
    print(f"1h x={x} k=1: pooled fires per 6-month window incl. 2022H1: min {p6.min()} ({dict((f'{a}H{b + 1}', int(n)) for (a, b), n in p6.items())})")
# Addendum 2: breadth at the spec's defaults (flow_x=0.5, move_k=1) and at (1.0, 1); co-occurrence only.
for x in [0.5, 1.0]:
    M = pd.DataFrame({s: fires(f, x, 1.0) for s, f in Fs.items()}).fillna(False)
    cnt = M.sum(axis=1)
    tot = M.values.sum()
    print(f"1h breadth x={x} k=1: share of fires sharing their bar with >=1/>=3/>=5 other symbols: "
          + " / ".join(f"{float((M.values & (cnt.values[:, None] >= m + 1)).sum() / tot):.2f}" for m in [1, 3, 5])
          + f"; distinct bars {int((cnt > 0).sum())} for {int(tot)} fires")
