"""Look 3 (frequency and contemporaneous feature distributions, no forward data): how often a
volume-and-price shock bar occurs at 15m and 1h, for a coarse set of threshold pairs, so the
candidate's trade count can be checked against >= 30 OOS trades, >= 20 trades per 6-month train
window (pooled) and >= 10 trades in a 6-month holdout. Also the taker buy share ON the shock bar
(same bar, not after it) and how many symbols shock in the same bar (breadth). Every quantity is
computed from bars <= t; no close after the shock bar is read."""

import numpy as np
import pandas as pd

from common import SYMBOLS, bars
from perp_lab.indicators import atr

pd.set_option("display.width", 220)
DAYS = 7
PER_DAY = {"15m": 96, "1h": 24}


def shock_frame(b: pd.DataFrame, tf: str) -> pd.DataFrame:
    w = DAYS * PER_DAY[tf]
    a_prev = atr(b, 14).shift(1)
    vmed = b.volume.shift(1).rolling(w, min_periods=w).median()
    return pd.DataFrame({
        "vr": b.volume / vmed,
        "mv": (b.close - b.open) / a_prev,
        "tb": b.taker_buy_volume / b.volume.replace(0, np.nan),
    }, index=b.index)


for tf in ["15m", "1h"]:
    F = {s: shock_frame(bars(s, tf), tf) for s in SYMBOLS}
    span_yrs = 4.5
    print(f"\n=== {tf}: shock bars per symbol per year (vr >= v AND |mv| >= k), median over symbols "
          f"[min, max]; both directions")
    tab = []
    for v in [3, 4, 5, 6, 8]:
        r = {"v": v}
        for k in [1.5, 2, 3, 4]:
            per = [float(((f.vr >= v) & (f.mv.abs() >= k)).sum()) / span_yrs for f in F.values()]
            r[f"k={k}"] = f"{np.median(per):.0f} [{min(per):.0f},{max(per):.0f}]"
        tab.append(r)
    print(pd.DataFrame(tab).to_string(index=False))

    print(f"\n{tf}: share of shocks that are down moves, and taker buy share on the shock bar itself "
          f"(pooled over symbols), for v=4/k=2 and v=6/k=3:")
    for v, k in [(4, 2), (6, 3)]:
        P = pd.concat([f[(f.vr >= v) & (f.mv.abs() >= k)] for f in F.values()])
        dn, up = P[P.mv < 0], P[P.mv > 0]
        print(f"  v={v} k={k}: n={len(P)}, down share {len(dn) / max(len(P), 1):.2f}; "
              f"tb on down shocks p10/p50/p90 = {dn.tb.quantile(.1):.3f}/{dn.tb.median():.3f}/{dn.tb.quantile(.9):.3f}; "
              f"on up shocks = {up.tb.quantile(.1):.3f}/{up.tb.median():.3f}/{up.tb.quantile(.9):.3f}; "
              f"down shocks with tb>0.5: {float((dn.tb > 0.5).mean()):.3f}, up shocks with tb<0.5: {float((up.tb < 0.5).mean()):.3f}")

    print(f"\n{tf}: pooled shock count per 6-month window (v=4/k=2, v=5/k=3, v=6/k=3), min over windows:")
    for v, k in [(4, 2), (5, 3), (6, 3)]:
        ts = pd.concat([f.index[(f.vr >= v) & (f.mv.abs() >= k)].to_series() for f in F.values()])
        per6 = ts.groupby([ts.index.year, (ts.index.month - 1) // 6]).size()
        print(f"  v={v} k={k}: min {per6.min()}, median {per6.median():.0f}, max {per6.max()}  "
              f"({dict((f'{a}H{b + 1}', int(n)) for (a, b), n in per6.items())})")

    print(f"\n{tf}: breadth: of all shock bars (v=4/k=2), share whose bar has shocks in >=1 / >=3 / >=5 other symbols:")
    M = pd.DataFrame({s: ((f.vr >= 4) & (f.mv.abs() >= 2)) for s, f in F.items()}).fillna(False)
    cnt = M.sum(axis=1)
    tot = M.values.sum()
    for m in [1, 3, 5]:
        print(f"  >= {m} others: {float((M.values & (cnt.values[:, None] >= m + 1)).sum() / tot):.2f}")
    print(f"  distinct shock timestamps: {int((cnt > 0).sum())} for {int(tot)} symbol-shocks")
