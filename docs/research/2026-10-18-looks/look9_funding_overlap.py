"""Look 9 (co-occurrence of two trigger conditions; no forward data): how often flow_shock_fade's
trigger fires on the same 1h bar as funding_fade's trigger, or while a funding_fade entry from the
previous 24 bars (its default hold) could still be open, against the share of ALL bars in the same
position. If the two triggers were the same trades, the flow trigger would sit on or near funding
triggers far more often than an arbitrary bar does.

Both triggers come from the committed plugins' own prepare() (loaded the way the tests load them),
evaluated with the plugins' entry rules on bar t. prepare reads bars <= t only (checked by
tests/test_lookahead.py), and nothing after bar t is read here: these are counts of conditions,
not outcomes."""

import numpy as np
import pandas as pd

from common import SYMBOLS, bars
from tests.sandbox_fixtures import load_class

FLOW = load_class("flow_shock_fade")
FUND = load_class("funding_fade")
WIN = 24        # funding_fade's default hold_bars


def flow_side(b, x):
    f = FLOW(flow_x=x).prepare(b)
    ok = np.isfinite(f.flow) & np.isfinite(f.mv) & np.isfinite(f.atr)
    return pd.Series(np.where(ok & (f.flow <= -x) & (f.mv <= -1.0), 1,
                              np.where(ok & (f.flow >= x) & (f.mv >= 1.0), -1, 0)), index=b.index)


def fund_side(b, thr):
    f = FUND(z_thr=thr).prepare(b)
    s = f.settle.astype(bool)
    return pd.Series(np.where(s & (f.fz > thr) & (f.frate > 0), -1,
                              np.where(s & (f.fz < -thr) & (f.frate < 0), 1, 0)), index=b.index), s


rows = []
for x in [0.5, 1.0]:
    for thr in [1.5, 2.0]:
        n_flow = same_any = same_side = on_settle = in_win = 0
        n_bars = bars_in_win = n_fund = 0
        for sym in SYMBOLS:
            b = bars(sym, "1h")
            fl = flow_side(b, x)
            fu, settle = fund_side(b, thr)
            fire = fl != 0
            ffire = fu != 0
            # a funding_fade trigger at t' covers bars t'..t'+WIN (signal bar plus its hold)
            win = ffire.astype(int).rolling(WIN + 1, min_periods=1).max().astype(bool)
            n_flow += int(fire.sum())
            same_any += int((fire & ffire).sum())
            same_side += int((fire & ffire & (fl == fu)).sum())
            on_settle += int((fire & settle).sum())
            in_win += int((fire & win).sum())
            n_bars += len(b)
            n_fund += int(ffire.sum())
            bars_in_win += int(win.sum())
        rows.append({"flow_x": x, "funding z_thr": thr, "flow triggers": n_flow,
                     "funding_fade triggers": n_fund,
                     "flow on a settlement bar": round(on_settle / n_flow, 3),
                     "flow on a funding trigger bar": round(same_any / n_flow, 4),
                     "  same side": round(same_side / n_flow, 4),
                     "all bars that are funding triggers": round(n_fund / n_bars, 4),
                     f"flow within {WIN} bars after a funding trigger": round(in_win / n_flow, 3),
                     f"all bars within {WIN} bars after one": round(bars_in_win / n_bars, 3)})
pd.set_option("display.width", 250)
print("1h, 8 symbols pooled, research span 2022-01 -> 2026-07")
print(pd.DataFrame(rows).astype(object).T.to_string(header=False))
