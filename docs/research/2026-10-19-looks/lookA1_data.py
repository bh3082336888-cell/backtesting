"""Look A1 (data quality on the wider data; no forward data): per symbol at 1h over the research
span, the first bar, the bar count, the NaN shares of taker_buy_volume and trades, and the bars
with zero volume or zero trades; then the NaN share of taker_buy_volume per calendar year. The
counterpart of look 2's taker check. Data only: no price after any bar is read."""

import pandas as pd

from common import SYMBOLS, bars

pd.set_option("display.width", 220)
rows, yearly = [], {}
for s in SYMBOLS:
    b = bars(s, "1h")
    sym = s.replace("USDT", "")
    rows.append({"sym": sym, "first": str(b.index.min().date()), "last": str(b.index.max().date()),
                 "bars": len(b), "nan_taker": round(float(b.taker_buy_volume.isna().mean()), 5),
                 "nan_trades": round(float(b.trades.isna().mean()), 5),
                 "zero_volume": int((b.volume == 0).sum()), "zero_trades": int((b.trades == 0).sum())})
    yearly[sym] = b.taker_buy_volume.isna().groupby(b.index.year).mean()
print("1h research bars per symbol:")
print(pd.DataFrame(rows).to_string(index=False))
print("\nNaN share of taker_buy_volume per calendar year (blank: not listed yet):")
print(pd.DataFrame(yearly).T.round(5).to_string())
