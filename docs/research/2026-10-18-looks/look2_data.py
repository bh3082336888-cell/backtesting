"""Look 2 (data quality and funding regimes, no prices forward): gaps, NaN / zero fractions of the
columns a candidate might read (taker_buy_volume, trades, quote_volume, oi), the taker buy share's
distribution over all bars, and the distribution of settled funding rates (carry cost per day and
how often funding sits away from the 0.01% baseline). Nothing here looks at returns after any
condition."""

import numpy as np
import pandas as pd

from common import SYMBOLS, TFS, bars
from perp_lab.config import TF_MS

pd.set_option("display.width", 200)
rows = []
for tf in TFS:
    for s in SYMBOLS:
        b = bars(s, tf)
        gaps = int((np.diff(b.open_time.values) != TF_MS[tf]).sum())
        tb = b.taker_buy_volume / b.volume.replace(0, np.nan)
        rows.append({
            "tf": tf, "sym": s.replace("USDT", ""), "start": b.index[0].date(), "end": b.index[-1],
            "gaps": gaps,
            "vol0": int((b.volume == 0).sum()),
            "tbv_nan": float(b.taker_buy_volume.isna().mean()),
            "trades_nan": float(b.trades.isna().mean()),
            "oi_nan": float(b.oi.isna().mean()),
            "tb_p01": tb.quantile(0.01), "tb_p10": tb.quantile(0.10), "tb_med": tb.median(),
            "tb_p90": tb.quantile(0.90), "tb_p99": tb.quantile(0.99),
        })
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\nFunding (settled rates, from 1h bars' funding_charge, research span only):")
frows = []
for s in SYMBOLS:
    b = bars(s, "1h")
    fc = b.funding_charge[b.funding_charge != 0.0]
    days = (b.index[-1] - b.index[0]).days
    frows.append({
        "sym": s.replace("USDT", ""), "settlements": len(fc),
        "mean_bp": fc.mean() * 1e4, "med_bp": fc.median() * 1e4,
        "long_carry_bp_per_day": fc.sum() * 1e4 / days,
        "frac_eq_1bp": float((fc.round(6) == 0.0001).mean()),
        "frac_gt_1bp": float((fc > 0.00010001).mean()),
        "frac_lt_0": float((fc < 0).mean()),
        "p01_bp": fc.quantile(0.01) * 1e4, "p99_bp": fc.quantile(0.99) * 1e4,
        "interval_h": ",".join(str(int(x)) for x in sorted(b.funding_interval_h.dropna().unique())),
    })
print(pd.DataFrame(frows).round(3).to_string(index=False))

print("\nRegime label shares (1h bars, research span):")
reg = {s.replace("USDT", ""): bars(s, "1h").regime.value_counts(normalize=True) for s in SYMBOLS}
print(pd.DataFrame(reg).round(3).to_string())
