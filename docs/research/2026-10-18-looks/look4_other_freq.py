"""Look 4 (frequency of the other candidates' trigger conditions; no forward data): how often each
condition is true or fires per symbol per year, to judge trade counts. Every condition uses bars
<= t only and nothing after the trigger bar is read.

  C2 compression breakout (4h): 20-bar Bollinger width in the bottom 10% of its trailing 180-bar
     values on bar t-1, and close_t beyond the prior 20-bar Donchian range.
  C3 weekend move (4h): Friday 20:00 -> Sunday 20:00 UTC close-to-close move (known at the close of
     the Sunday 20:00 bar), in units of the 4h ATR(14) x sqrt(12) (ATR scaled to the 12-bar, two-day span).
  C4 funding regime (1h): share of bars whose last settled rate is above 2 bp (crowded long) or
     below 0 (crowded short).
  C5 taker-flow persistence (4h): 6-bar (24h) net taker imbalance sum(2*tbv - vol) / sum(vol).
"""

import numpy as np
import pandas as pd

from common import SYMBOLS, bars
from perp_lab.indicators import atr, bollinger, donchian_prior

pd.set_option("display.width", 220)
YRS = 4.5
rows = []
for s in SYMBOLS:
    b4 = bars(s, "4h")
    _, up, lo = bollinger(b4.close, 20, 2.0)
    bw = (up - lo) / b4.close
    pct = bw.rolling(180, min_periods=180).rank(pct=True)
    hi20, lo20 = donchian_prior(b4, 20)
    comp = (pct.shift(1) <= 0.10)
    brk = (b4.close > hi20) | (b4.close < lo20)
    a4 = atr(b4, 14)
    c = b4.close
    sun = b4.index[(b4.index.dayofweek == 6) & (b4.index.hour == 20)]
    wk = []
    for t in sun:
        f = t - pd.Timedelta(days=2)
        if f in b4.index:
            wk.append(abs(c[t] - c[f]) / (a4[t] * np.sqrt(12)))
    wk = pd.Series(wk)
    b1 = bars(s, "1h")
    fl = b1.funding_last
    vol6 = b4.volume.rolling(6).sum()
    imb = (2 * b4.taker_buy_volume - b4.volume).rolling(6).sum() / vol6
    rows.append({
        "sym": s.replace("USDT", ""),
        "C2_compressed_share": float(comp.mean()),
        "C2_fires_per_yr": float((comp & brk).sum()) / YRS,
        "C3_weekends": len(wk), "C3_move_med_2dATR": wk.median(),
        "C3_share_gt_1x2dATR": float((wk > 1).mean()),
        "C4_share_f_gt_2bp": float((fl > 0.0002).mean()), "C4_share_f_lt_0": float((fl < 0).mean()),
        "C5_imb24_p05": imb.quantile(0.05), "C5_imb24_p95": imb.quantile(0.95),
        "C5_share_abs_gt_5pct": float((imb.abs() > 0.05).mean()),
    })
print(pd.DataFrame(rows).round(3).to_string(index=False))
