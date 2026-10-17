"""Look 2b (data gap check): BNB has half the funding settlements of the others. Count nonzero
settled rates per half-year per symbol, and BNB's zero-rate settlement hours. Data coverage only."""
import pandas as pd
from common import SYMBOLS, bars
out = {}
for s in SYMBOLS:
    b = bars(s, "1h")
    nz = b.funding_charge != 0.0
    out[s.replace("USDT", "")] = nz.groupby([b.index.year, (b.index.month - 1) // 6]).sum()
print(pd.DataFrame(out).to_string())
b = bars("BNBUSDT", "1h")
h = b[b.index.hour.isin([0, 8, 16])]
print("BNB bars at 00/08/16 UTC:", len(h), " with nonzero charge:", int((h.funding_charge != 0).sum()),
      " funding_last NaN share:", float(b.funding_last.isna().mean()))
z = h[h.funding_charge == 0.0]
print("BNB zero-charge settlement-hour bars by year:", z.groupby(z.index.year).size().to_dict())
print("BNB settlement-hour bars whose funding_last is exactly 0 (a recorded zero rate, not a missing row):",
      int((h.funding_last == 0.0).sum()))
