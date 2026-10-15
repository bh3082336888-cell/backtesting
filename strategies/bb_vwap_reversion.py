"""When price closes outside a Bollinger band with RSI at an extreme, it reverts toward the rolling VWAP within a bounded number of bars, enough to pay costs.

Range baseline. Long when close < lower band and RSI < rsi_lo (short mirrored).
Target: rolling VWAP (resting limit, re-pegged each bar while it is still on the
profitable side; market exit once price has crossed it). ATR stop. Time stop.
"""

from perp_lab.indicators import atr, bollinger, rolling_vwap, rsi
from perp_lab.strategy import EXIT, HOLD, Signal, Strategy, enter


class BBVWAPReversion(Strategy):
    params = {"bb_n": 20, "bb_k": 2.0, "rsi_n": 14, "rsi_lo": 30.0, "rsi_hi": 70.0,
              "vwap_n": 48, "stop_atr": 2.0, "max_bars": 24, "atr_n": 14}
    grid = {"bb_k": [2.0, 2.5, 3.0], "stop_atr": [1.5, 2.5]}
    timeframes = ["15m", "1h"]

    def prepare(self, bars):
        mid, up, lo = bollinger(bars.close, self.p["bb_n"], self.p["bb_k"])
        return bars[[]].assign(bb_up=up, bb_lo=lo, rsi=rsi(bars.close, self.p["rsi_n"]),
                               vwap=rolling_vwap(bars, self.p["vwap_n"]),
                               atr=atr(bars, self.p["atr_n"]))

    def bracket(self, ctx, side):
        c, v = ctx["close"], ctx["vwap"]
        tgt = v if (v - c) * side > 0 else None
        return c - side * self.p["stop_atr"] * ctx["atr"], tgt

    def entry(self, ctx):
        c, r = ctx["close"], ctx["rsi"]
        if c < ctx["bb_lo"] and r < self.p["rsi_lo"] and ctx["vwap"] > c:
            return enter(1, *self.bracket(ctx, 1))
        if c > ctx["bb_up"] and r > self.p["rsi_hi"] and ctx["vwap"] < c:
            return enter(-1, *self.bracket(ctx, -1))
        return HOLD

    def manage(self, ctx):
        pos = ctx.position
        if pos.bars_held >= self.p["max_bars"]:
            return EXIT
        v = ctx["vwap"]
        if (v - ctx["close"]) * pos.side <= 0:
            return EXIT
        return Signal("hold", target=v)
