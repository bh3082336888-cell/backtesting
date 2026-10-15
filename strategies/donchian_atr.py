"""Crypto perps trend persistently enough that a close beyond the prior N-bar Donchian range, held with an ATR chandelier trail, earns more than it loses after fees, slippage and funding.

Trend baseline. Long on close above prior N-bar high, short on close below prior
N-bar low. Initial stop and trail: best close since entry -/+ atr_mult * ATR.
No target, no regime filter — that's on purpose; it's a baseline.
"""

from perp_lab.indicators import atr, donchian_prior
from perp_lab.strategy import HOLD, Signal, Strategy, enter


class DonchianATR(Strategy):
    params = {"n": 40, "atr_n": 14, "atr_mult": 3.0}
    grid = {"n": [20, 40, 80], "atr_mult": [2.0, 3.0, 4.0]}
    timeframes = ["1h", "4h"]

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
