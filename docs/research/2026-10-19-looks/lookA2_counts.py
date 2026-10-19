"""Look A2 (trade-count bounds on the wider data; no forward data): the counterpart of looks 5 and
6 for the 17 coins from 2020, against the gates as they now scale with the coins listed.

Triggers come from the committed plugin's own prepare() (loaded the way the tests load them) and
its entry rule on bar t at move_k = 1: long when flow <= -flow_x and mv <= -1, short when
flow >= flow_x and mv >= 1, with flow, mv and atr all finite. prepare reads bars <= t only
(tests/test_lookahead.py), and nothing after bar t is read here.

  fires per symbol per year: every trigger, over each symbol's own research span;
  lower bound per grid combo: greedy, as in look 6 (take a trigger at t, skip t+1..t+H; a stop
    exit only shortens a hold, so it can only add trades);
  per walk-forward fold (6-month train, 2-month test, from history_start): the pooled lower bound
    in the train window against the gate, ceil(2.5 x coins listed in that window);
  the OOS span (first test window to holdout_start): the pooled lower bound against
    ceil(3.75 x 17) = 64, and the pooled lower bound per 2-month test window;
  per calendar half-year: the pooled lower bound, against the exam's minimum at 17 coins,
    max(10, ceil(1.25 x 17)) = 22, for a holdout of about six months.
These are counts, not outcomes."""

import math

import numpy as np
import pandas as pd

from common import CFG, SYMBOLS, bars
from perp_lab.validate.walkforward import make_folds
from tests.sandbox_fixtures import load_class

pd.set_option("display.width", 240)
FLOW = load_class("flow_shock_fade")
FOLDS = make_folds(CFG.history_start, CFG.holdout_start, 6, 2)
A, B = FOLDS[0]["test_start"], FOLDS[-1]["test_end"]
FIRST = {s: bars(s, "1h").index.min() for s in SYMBOLS}


def side(b: pd.DataFrame, x: float) -> pd.Series:
    f = FLOW(flow_x=x).prepare(b)
    ok = np.isfinite(f.flow) & np.isfinite(f.mv) & np.isfinite(f.atr)
    return pd.Series(np.where(ok & (f.flow <= -x) & (f.mv <= -1.0), 1,
                              np.where(ok & (f.flow >= x) & (f.mv >= 1.0), -1, 0)), index=b.index)


def greedy(sig: pd.Series, h: int) -> pd.Series:
    taken, nxt = [], -1
    for i in np.flatnonzero(sig.values != 0):
        if i >= nxt:
            taken.append(i)
            nxt = i + h + 1
    return sig.iloc[taken]


def within(ts: pd.Series, a, b) -> pd.Series:
    return ts[(ts.index >= a) & (ts.index < b)]


SIG = {x: {s: side(bars(s, "1h"), x) for s in SYMBOLS} for x in [0.5, 1.0]}
print(f"OOS span {A.date()} -> {B.date()}, {len(FOLDS)} folds\n")

for x, sig in SIG.items():
    per = [float((v != 0).sum()) / ((CFG.holdout_start - FIRST[s]).days / 365.25) for s, v in sig.items()]
    print(f"flow_x={x}: fires per symbol per year, median {np.median(per):.0f} [{min(per):.0f}, {max(per):.0f}]")
print()

rows = []
for x, sig in SIG.items():
    for h in [6, 12, 24]:
        g = {s: greedy(v, h) for s, v in sig.items()}
        oos = {s: within(v, A, B) for s, v in g.items()}
        allg = pd.concat(list(g.values()))
        lb = [len(v) for v in oos.values()]
        ub = [int((within(v, A, B) != 0).sum()) for v in sig.values()]
        longs = sum(int((v > 0).sum()) for v in oos.values())
        margins = []
        for f in FOLDS:
            a, b = f["train_start"], f["train_end"]
            listed = sum(FIRST[s] < b for s in SYMBOLS)
            n = len(within(allg, a, b))
            margins.append((n - math.ceil(2.5 * listed), n, math.ceil(2.5 * listed), str(a.date())))
        worst = min(margins)
        test2m = [len(within(allg, f["test_start"], f["test_end"])) for f in FOLDS]
        hy = allg.groupby([allg.index.year, (allg.index.month - 1) // 6]).size()
        rows.append({"flow_x": x, "hold": h,
                     "per_sym_lower": f"{int(np.median(lb))} [{min(lb)},{max(lb)}]",
                     "per_sym_upper": f"{int(np.median(ub))} [{min(ub)},{max(ub)}]",
                     "oos_lower": sum(lb), "oos_gate": math.ceil(3.75 * len(SYMBOLS)),
                     "long_share": round(longs / max(sum(lb), 1), 2),
                     "worst_train": f"{worst[1]} vs {worst[2]} ({worst[3]})",
                     "test2m_min": min(test2m),
                     "halfyear_min": f"{int(hy.min())} ({hy.idxmin()[0]}H{hy.idxmin()[1] + 1})",
                     "2025H2/2026H1": f"{int(hy.get((2025, 1), 0))}/{int(hy.get((2026, 0), 0))} vs 22"})
print("Greedy lower bound (OOS totals and windows pooled over symbols); upper bound = every trigger:")
print(pd.DataFrame(rows).to_string(index=False))
print("\nworst_train: the fold whose train window has the smallest margin over ceil(2.5 x coins listed),"
      " as 'trades vs minimum (train start)'.")
