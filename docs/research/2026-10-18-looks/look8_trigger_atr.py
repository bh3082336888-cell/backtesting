"""Look 8 (same-bar feature distribution; no forward data): the ATR14 at bar t, in bp of bar t's
close, on the bars where the spec's trigger fires, against all bars. This is the ATR the plugin's
stop is set from (stop = close_t -/+ stop_atr x ATR_t), so it converts an R-denominated hurdle into
bp. Look 5 printed this only for its (flow_x=1, move_k=2) trigger; here it is for the spec's
default (0.5, 1) and strict (1.0, 1) triggers, with (1.0, 2) repeated as a cross-check against
look 5's 57/111/255.

Two summaries, labelled because they differ:
  pooled           = quantiles over every (symbol, bar) pair together;
  per-symbol median = the median over the 8 symbols of each symbol's own median (how look 1 and
                      section 2's ATR table summarise, which gave 109 bp for all 1h bars).
Only bar t and earlier bars are read: the trigger and the ATR both use bars <= t."""

import numpy as np
import pandas as pd

from common import SYMBOLS, bars
from perp_lab.indicators import atr

pd.set_option("display.width", 220)
OOS_START = pd.Timestamp("2022-07-01", tz="UTC")


def frame(b: pd.DataFrame) -> pd.DataFrame:
    a = atr(b, 14)
    vmed = b.volume.shift(1).rolling(168, min_periods=168).median()
    return pd.DataFrame({
        "F": (2 * b.taker_buy_volume - b.volume) / vmed.replace(0, np.nan),
        "mv": (b.close - b.open) / a.shift(1),
        "atr_bp": a / b.close * 1e4,             # ATR at t, the stop's ATR, in bp of close_t
    }, index=b.index)


def fires(f: pd.DataFrame, x: float, k: float) -> pd.Series:
    return ((f.F <= -x) & (f.mv <= -k)) | ((f.F >= x) & (f.mv >= k))


Fs = {s.replace("USDT", ""): frame(bars(s, "1h")) for s in SYMBOLS}
P = pd.concat(Fs.values())


def q(s: pd.Series) -> str:
    return f"{s.quantile(.1):.0f}/{s.median():.0f}/{s.quantile(.9):.0f}"


def per_sym_median(mask_fn) -> float:
    return float(np.median([f.atr_bp[mask_fn(f)].median() for f in Fs.values()]))


print("1h ATR14 at t in bp of close_t, research span 2022-01 -> 2026-07")
print(f"all bars:  n={int(P.atr_bp.notna().sum())}  pooled p10/p50/p90 = {q(P.atr_bp)};  "
      f"median of per-symbol medians = {per_sym_median(lambda f: f.atr_bp.notna()):.0f}")
rows = []
for x, k in [(0.5, 1.0), (1.0, 1.0), (1.0, 2.0)]:
    m = fires(P, x, k)
    oos = m & (P.index >= OOS_START)
    print(f"trigger flow_x={x} move_k={k}:  n={int(m.sum())}  pooled p10/p50/p90 = {q(P.atr_bp[m])};  "
          f"median of per-symbol medians = {per_sym_median(lambda f: fires(f, x, k)):.0f};  "
          f"OOS span 2022-07 -> 2026-07 only: n={int(oos.sum())} pooled median {P.atr_bp[oos].median():.0f}")
    rows.append({"trigger": f"({x}, {k})",
                 **{s: round(float(f.atr_bp[fires(f, x, k)].median())) for s, f in Fs.items()}})
rows.append({"trigger": "all bars", **{s: round(float(f.atr_bp.median())) for s, f in Fs.items()}})
print("\nPer-symbol median ATR14 at t (bp of close):")
print(pd.DataFrame(rows).to_string(index=False))
print("\nPer symbol, median ATR on trigger bars / that symbol's median ATR on all bars:")
ratios = [{"trigger": f"({x}, {k})",
           **{s: round(float(f.atr_bp[fires(f, x, k)].median() / f.atr_bp.median()), 2) for s, f in Fs.items()}}
          for x, k in [(0.5, 1.0), (1.0, 1.0)]]
print(pd.DataFrame(ratios).to_string(index=False))
