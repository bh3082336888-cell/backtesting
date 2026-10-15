"""An extreme funding settlement marks crowded positioning, so fading the crowded side right after it earns the unwind move plus the funding carry.

Perps-specific baseline. On each settlement bar, z-score the settled rate
against the previous `z_k` settlements (std floored at 1bp to avoid dividing by
the flat 0.01% baseline). z > z_thr and rate > 0 → short; z < -z_thr and rate < 0
→ long. Exit after `hold_bars` or at the ATR stop.
"""

import numpy as np
import pandas as pd

from perp_lab.indicators import atr
from perp_lab.strategy import EXIT, HOLD, Strategy, enter


class FundingFade(Strategy):
    params = {"z_k": 90, "z_thr": 2.0, "hold_bars": 24, "stop_atr": 3.0, "atr_n": 14}
    grid = {"z_thr": [1.5, 2.0, 2.5, 3.0], "hold_bars": [8, 24, 48]}
    timeframes = ["1h"]

    def prepare(self, bars):
        fc = bars.funding_charge
        is_settle = fc != 0.0
        s = fc[is_settle]
        k = self.p["z_k"]
        mu = s.shift(1).rolling(k, min_periods=k // 3).mean()
        sd = s.shift(1).rolling(k, min_periods=k // 3).std().clip(lower=1e-4)
        z = ((s - mu) / sd).reindex(bars.index)
        return bars[[]].assign(
            fz=z.fillna(0.0).values,
            frate=fc.values,
            settle=is_settle.values.astype(np.int8),
            atr=atr(bars, self.p["atr_n"]),
        )

    def bracket(self, ctx, side):
        return ctx["close"] - side * self.p["stop_atr"] * ctx["atr"], None

    def entry(self, ctx):
        if not ctx["settle"]:
            return HOLD
        z, f = ctx["fz"], ctx["frate"]
        if z > self.p["z_thr"] and f > 0:
            return enter(-1, *self.bracket(ctx, -1))
        if z < -self.p["z_thr"] and f < 0:
            return enter(1, *self.bracket(ctx, 1))
        return HOLD

    def manage(self, ctx):
        if ctx.position.bars_held >= self.p["hold_bars"]:
            return EXIT
        return HOLD
