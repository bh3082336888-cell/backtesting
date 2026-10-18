"""An hour of extreme one-sided taker flow that moves price marks forced or impatient trading (liquidation and stop cascades), so fading it at the next open earns the partial reversion over the following hours, net of costs.

Root of a new family (order-flow shock). flow = (taker buys - taker sells) on the bar, divided by
the median bar volume of the previous vol_n bars; mv = (close - open) / ATR at t-1. Long when
flow <= -flow_x and mv <= -move_k; short when flow >= flow_x and mv >= move_k. Stop stop_atr x ATR
from the close, no target, exit after hold_bars bars. The exits ignore the trigger on purpose:
random entries with the same exits earn nothing, so only the entry can beat them.
"""

import numpy as np

from perp_lab.indicators import atr
from perp_lab.strategy import EXIT, HOLD, Strategy, enter


class FlowShockFade(Strategy):
    params = {"flow_x": 0.5, "move_k": 1.0, "hold_bars": 12, "stop_atr": 3.0, "atr_n": 14, "vol_n": 168}
    grid = {"flow_x": [0.5, 1.0], "hold_bars": [6, 12, 24]}
    timeframes = ["1h"]

    def prepare(self, bars):
        p = self.p
        a = atr(bars, p["atr_n"])                                         # Wilder ATR of true range, includes bar t
        vmed = bars.volume.shift(1).rolling(p["vol_n"], min_periods=p["vol_n"]).median()  # bars t-vol_n .. t-1
        flow = (2.0 * bars.taker_buy_volume - bars.volume) / vmed.replace(0.0, np.nan)
        mv = (bars.close - bars.open) / a.shift(1).replace(0.0, np.nan)   # ATR at t-1: excludes bar t
        return bars[[]].assign(flow=flow, mv=mv, atr=a)

    def bracket(self, ctx, side):
        return ctx["close"] - side * self.p["stop_atr"] * ctx["atr"], None

    def entry(self, ctx):
        f, m, a = ctx["flow"], ctx["mv"], ctx["atr"]
        if not (np.isfinite(f) and np.isfinite(m) and np.isfinite(a)):
            return HOLD
        if f <= -self.p["flow_x"] and m <= -self.p["move_k"]:
            return enter(1, *self.bracket(ctx, 1))
        if f >= self.p["flow_x"] and m >= self.p["move_k"]:
            return enter(-1, *self.bracket(ctx, -1))
        return HOLD

    def manage(self, ctx):
        if ctx.position.bars_held >= self.p["hold_bars"]:
            return EXIT
        return HOLD
