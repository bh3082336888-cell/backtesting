"""The trusted replay: a decision log run through the unchanged engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from perp_lab.engine.backtest import BacktestResult, EngineParams, run_backtest
from perp_lab.sandbox.decisions import Decisions
from perp_lab.sandbox.errors import PluginError
from perp_lab.strategy import HOLD, Ctx, Signal, Strategy


def random_draw(n: int, seed: int, prob: float, long_share: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """The draw RandomEntry.prepare makes: (fire bool, side +-1)."""
    u = np.random.default_rng(seed).random((n, 2))
    return u[:, 0] < prob, np.where(u[:, 1] < long_share, 1, -1)


class ScriptedStrategy(Strategy):
    name = "scripted_replay"

    def __init__(self, dec: Decisions, draw=None):
        self.p = {}
        self._sig = dec.by_bar()
        self._draw = draw

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        return bars[[]]

    def on_bar(self, ctx: Ctx) -> Signal:
        sig = self._sig.get(ctx.i)
        if sig is None:
            return HOLD
        if self._draw is not None and sig.action in ("long", "short") and ctx.position is None:
            fire, side = self._draw
            if not fire[ctx.i] or int(side[ctx.i]) != (1 if sig.action == "long" else -1):
                raise PluginError("mismatch", "", f"random entry off the draw at bar {ctx.i}")
        return sig


def replay(bars: pd.DataFrame, dec: Decisions, *, symbol: str, tf: str, ep: EngineParams,
           start_i: int = 0, end_i: int | None = None, draw=None) -> BacktestResult:
    return run_backtest(ScriptedStrategy(dec, draw), bars, bars[[]], symbol=symbol, tf=tf, ep=ep,
                        start_i=start_i, end_i=end_i)


def check_finite(res: BacktestResult) -> None:
    """The trusted engine's equity must stay finite. A fill at an infinite or overflowing price makes
    it -inf or NaN, and a day whose return is NaN is dropped from the portfolio rather than counted
    against the plugin, so that is the plugin's `mismatch`."""
    if not np.isfinite(res.equity).all():
        raise PluginError("mismatch", "", "equity is not finite")


def check_fingerprint(res: BacktestResult, fp: tuple[int, float], ep: EngineParams) -> None:
    final = float(res.equity[-1]) if len(res.equity) else ep.equity0
    if len(res.trades) != fp[0] or final != fp[1]:
        raise PluginError("mismatch", "", "replay disagrees with the worker's run")
