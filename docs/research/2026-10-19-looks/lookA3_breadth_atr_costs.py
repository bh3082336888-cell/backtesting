"""Look A3 (same-bar features and costs on the wider data; no forward data): the counterpart of
look 5's breadth, look 8's trigger-bar ATR and look 1's cost ratios at 1h, for the 17 coins from
2020 and the per-coin slippage now in the engine.

  breadth: the share of triggers whose bar also has a trigger in >= 1/3/5 other symbols, and the
    distinct trigger hours, over the whole research span (as look 5);
  ATR at bar t in bp of close_t on trigger bars against all bars, from the plugin's own `atr`
    feature: pooled p10/p50/p90, the median of the per-symbol medians, and each symbol's
    trigger-to-all-bar ratio (as look 8), over the whole span and over the OOS span;
  round-trip cost of a trigger's trade: entry at the open of t+1 pays taker + that bar's
    `slippage` (the engine's per-coin column, from volume before t+1's day); the exit is costed at
    the same bar's slippage, so 2 x (5 bp + slippage[t+1]). In bp and in R (1 R = 3 ATR_t);
  the unconditional ratio look 1 used for the 12-bar spread: the median absolute 12-bar log move
    over ALL 1h bars, against the median ATR (both per-symbol medians, then the median across
    symbols). Like look 1 it is over every bar, conditioned on nothing.
Triggers come from the plugin's prepare() and entry rule (move_k = 1); nothing after bar t+1 is
read for a trigger, and t+1's slippage is a cost known before t+1's day."""

import numpy as np
import pandas as pd

from common import CFG, SYMBOLS, bars
from perp_lab.validate.walkforward import make_folds
from tests.sandbox_fixtures import load_class

pd.set_option("display.width", 240)
FLOW = load_class("flow_shock_fade")
FOLDS = make_folds(CFG.history_start, CFG.holdout_start, 6, 2)
A, B = FOLDS[0]["test_start"], FOLDS[-1]["test_end"]
TAKER_BP = CFG["costs"]["taker"] * 1e4


def frame(s: str, x: float) -> pd.DataFrame:
    b = bars(s, "1h")
    f = FLOW(flow_x=x).prepare(b)
    ok = np.isfinite(f.flow) & np.isfinite(f.mv) & np.isfinite(f.atr)
    fire = ok & (((f.flow <= -x) & (f.mv <= -1.0)) | ((f.flow >= x) & (f.mv >= 1.0)))
    slip_next = (b.slippage.shift(-1) * 1e4)
    atr_bp = f.atr / b.close * 1e4
    return pd.DataFrame({"sym": s, "fire": fire, "atr_bp": atr_bp,
                         "cost_bp": 2 * (TAKER_BP + slip_next), "oos": (b.index >= A) & (b.index < B)},
                        index=b.index)


for x in [0.5, 1.0]:
    F = {s: frame(s, x) for s in SYMBOLS}
    M = pd.DataFrame({s: f.fire for s, f in F.items()}).fillna(False)
    cnt = M.sum(axis=1)
    tot = int(M.values.sum())
    print(f"=== flow_x={x}, move_k=1: {tot} triggers, {int((cnt > 0).sum())} distinct hours "
          f"({(cnt > 0).sum() / tot:.0%}); sharing their hour with >=1/>=3/>=5 other symbols: "
          + " / ".join(f"{float((M.values & (cnt.values[:, None] >= m + 1)).sum() / tot):.0%}" for m in [1, 3, 5]))
    P = pd.concat(F.values())
    for label, sel in [("whole span", P.index.notna()), ("OOS span", P.oos.values)]:
        q = P[sel]
        t, a = q[q.fire], q
        med_sym_t = t.groupby("sym").atr_bp.median()
        med_sym_a = a.groupby("sym").atr_bp.median()
        ratio = (med_sym_t / med_sym_a).dropna()
        print(f"  ATR bp, {label}: triggers p10/p50/p90 {t.atr_bp.quantile(.1):.0f}/{t.atr_bp.median():.0f}/"
              f"{t.atr_bp.quantile(.9):.0f} vs all bars {a.atr_bp.quantile(.1):.0f}/{a.atr_bp.median():.0f}/"
              f"{a.atr_bp.quantile(.9):.0f}; median of per-symbol medians {med_sym_t.median():.0f} vs "
              f"{med_sym_a.median():.0f}; per-symbol trigger/all ratio {ratio.min():.2f}-{ratio.max():.2f}")
    o = P[P.oos & P.fire]
    print(f"  round trip on OOS triggers: median {o.cost_bp.median():.1f} bp, mean {o.cost_bp.mean():.1f} bp, "
          f"p90 {o.cost_bp.quantile(.9):.1f} bp (flat 2 bp would be {2 * (TAKER_BP + 2):.0f}); "
          f"in R (3 ATR stop): median {(o.cost_bp / (3 * o.atr_bp)).median():.3f}, "
          f"mean {(o.cost_bp / (3 * o.atr_bp)).mean():.3f}")
    print("  per symbol, OOS triggers: " + ", ".join(
        f"{s.replace('USDT', '')} {g.cost_bp.median():.1f}" for s, g in o.groupby("sym")) + " (median round trip, bp)")

rows = []
for s in SYMBOLS:
    b = bars(s, "1h")
    lc = np.log(b.close)
    a = FLOW().prepare(b).atr / b.close * 1e4
    rows.append({"sym": s, "mv12": float(((lc.shift(-12) - lc).abs() * 1e4).median()), "atr": float(a.median())})
d = pd.DataFrame(rows)
print(f"\nAll 1h bars, whole span: median |12-bar move| {d.mv12.median():.0f} bp, median ATR {d.atr.median():.0f} bp "
      f"(medians across symbols of per-symbol medians); ratio {d.mv12.median() / d.atr.median():.2f} ATR")
