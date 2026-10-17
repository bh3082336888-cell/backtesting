"""Decision logs: the signals a strategy returned, as plain typed arrays.

A confined worker sends back only these; the trusted parent replays them through the engine.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

import numpy as np
import pandas as pd

from perp_lab.engine.backtest import build_arrays
from perp_lab.sandbox.errors import ProtocolError
from perp_lab.strategy import Ctx, PositionView, Signal, Strategy

ACTIONS = ("hold", "long", "short", "exit")
_CODE = {a: k for k, a in enumerate(ACTIONS)}
_DTYPES = {"i": "<i8", "action": "<i1", "stop": "<f8", "target": "<f8", "has_stop": "<i1", "has_target": "<i1"}


def _real(x) -> bool:
    return x is None or (isinstance(x, numbers.Real) and not isinstance(x, bool))


def check_signal(sig) -> Signal:
    if not isinstance(sig, Signal) or sig.action not in ACTIONS or not _real(sig.stop) or not _real(sig.target):
        raise TypeError(f"on_bar must return a Signal with a valid action and numeric stop/target, got {sig!r}")
    return sig


@dataclass(frozen=True)
class Decisions:
    i: np.ndarray
    action: np.ndarray
    stop: np.ndarray
    target: np.ndarray
    has_stop: np.ndarray
    has_target: np.ndarray

    @classmethod
    def from_log(cls, log: list[tuple[int, Signal]]) -> "Decisions":
        n = len(log)
        i = np.fromiter((e[0] for e in log), "<i8", n)
        action = np.fromiter((_CODE[e[1].action] for e in log), "<i1", n)
        stop = np.fromiter((math.nan if e[1].stop is None else float(e[1].stop) for e in log), "<f8", n)
        target = np.fromiter((math.nan if e[1].target is None else float(e[1].target) for e in log), "<f8", n)
        has_stop = np.fromiter((e[1].stop is not None for e in log), "<i1", n)
        has_target = np.fromiter((e[1].target is not None for e in log), "<i1", n)
        return cls(i, action, stop, target, has_stop, has_target)

    @classmethod
    def empty(cls) -> "Decisions":
        return cls.from_log([])

    def _arrays(self):
        return [getattr(self, k) for k in _DTYPES]

    def by_bar(self) -> dict[int, Signal]:
        out = {}
        for k in range(len(self.i)):
            out[int(self.i[k])] = Signal(
                ACTIONS[int(self.action[k])],
                float(self.stop[k]) if self.has_stop[k] else None,
                float(self.target[k]) if self.has_target[k] else None)
        return out

    def before(self, n: int) -> "Decisions":
        m = self.i < n
        return Decisions(*(a[m] for a in self._arrays()))

    def same(self, other: "Decisions") -> bool:
        return all(np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
                   for a, b in zip(self._arrays(), other._arrays()))

    def n_actions(self) -> int:
        return int(np.count_nonzero(self.action != 0))

    def validate(self, n_bars: int) -> None:
        for k, dt in _DTYPES.items():
            a = getattr(self, k)
            if not isinstance(a, np.ndarray) or a.ndim != 1 or a.dtype != np.dtype(dt):
                raise ProtocolError(f"decision column {k} must be a 1-d {dt} array")
        if len({len(a) for a in self._arrays()}) != 1:
            raise ProtocolError("decision columns differ in length")
        if len(self.i):
            if np.any((self.i < 0) | (self.i >= n_bars)) or np.any(self.i[1:] <= self.i[:-1]):
                raise ProtocolError("decision bars must be strictly increasing within [0, n_bars)")
        if np.any((self.action < 0) | (self.action >= len(ACTIONS))):
            raise ProtocolError("decision action code out of range")
        for k in ("has_stop", "has_target"):
            a = getattr(self, k)
            if np.any((a != 0) & (a != 1)):
                raise ProtocolError(f"{k} must be 0 or 1")


class Recorder(Strategy):
    """Wraps a strategy and logs every signal that is not a bare hold."""

    def __init__(self, inner: Strategy):
        self.inner = inner
        self.name = getattr(inner, "name", "recorder")
        self.p = inner.p
        self.log: list[tuple[int, Signal]] = []

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        return self.inner.prepare(bars)

    def on_bar(self, ctx: Ctx) -> Signal:
        sig = check_signal(self.inner.on_bar(ctx))
        if not (sig.action == "hold" and sig.stop is None and sig.target is None):
            self.log.append((ctx.i, sig))
        return sig


def decide_last(strategy: Strategy, symbol: str, bars: pd.DataFrame, position: PositionView | None) -> Signal:
    feats = strategy.prepare(bars)
    ctx = Ctx(build_arrays(bars, feats), symbol)
    ctx.i = len(bars) - 1
    ctx.position = position
    return check_signal(strategy.on_bar(ctx))
