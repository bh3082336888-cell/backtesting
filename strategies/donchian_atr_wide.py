"""If donchian_atr's 4h result comes from the wide 4-ATR trail rather than from parameter search, fixing atr_mult at 4.0 and selecting only n will hold OOS Sharpe with one fewer degree of freedom.

Hand-made child of donchian_atr (Phase 2 lineage demo). One change: atr_mult is
fixed at 4.0 and removed from the walk-forward grid.
"""

from perp_lab.indicators import atr, donchian_prior
from perp_lab.strategy import HOLD, Signal, Strategy, enter

PARENT = "donchian_atr"


class DonchianATRWide(Strategy):
    params = {"n": 40, "atr_n": 14, "atr_mult": 4.0}
    grid = {"n": [20, 40, 80]}
    timeframes = ["4h"]

    def prepare(self, bars):
        hi, lo = donchian_prior(bars, self.p["n"])
        return bars[[]].assign(dc_hi=hi, dc_lo=lo, atr=atr(bars, self.p["atr_n"]))

    def bracket(self, ctx, side):
        return ctx["close"] - side * self.p["atr_mult"] * ctx["atr"], None

    def entry(self, ctx):
        c = ctx["close"]
        if c > ctx["dc_hi"]:
            return enter(1, *self.bracket(ctx, 1))
        if c < ctx["dc_lo"]:
            return enter(-1, *self.bracket(ctx, -1))
        return HOLD

    def manage(self, ctx):
        pos = ctx.position
        a = ctx["atr"]
        if a != a:
            return HOLD
        return Signal("hold", stop=pos.best_close - pos.side * self.p["atr_mult"] * a)
