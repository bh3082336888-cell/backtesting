"""Strategy plugin interface.

A plugin lives in ./strategies/<name>.py, has a module docstring whose first
line is its hypothesis (one sentence), and defines exactly one Strategy subclass.

Contract:
    prepare(bars) -> DataFrame of features, same index as bars. Vectorised, but
        MUST be causal: row t may only depend on bars[: t + 1]. Enforced by
        tests/test_lookahead.py on every plugin automatically.
    bracket(ctx, side) -> (stop, target|None) for an entry on this bar's close.
    entry(ctx) -> Signal   called when flat
    manage(ctx) -> Signal  called when in a position (exit / new trailing stop)
    on_bar(ctx) -> Signal  dispatches to entry/manage. Signal is computed at the
        close of bar t and filled at the open of bar t+1 by the engine.

`ctx` exposes only the current bar (ctx["col"]) and the past (ctx.hist / ctx.prev).

Plugin code is never loaded here. It runs only in the sandbox session
(perp_lab/sandbox/session.py), inside the confined worker. The contract that
holds there:
    - NAME, PARAMS, GRID and TIMEFRAMES are literals the parent reads from the
      source without executing it.
    - Features are numeric or bool columns.
    - A plugin gets only the bars it is sent; it can't see later ones.
    - Imports are limited to the read allowlist.
"""

from __future__ import annotations

import inspect
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STRATEGY_DIR = Path("strategies")


@dataclass(frozen=True, slots=True)
class Signal:
    action: str = "hold"          # long | short | exit | hold
    stop: float | None = None     # entry: initial stop (required). hold: new trailing stop (ratchets only)
    target: float | None = None   # entry: take-profit limit (optional). hold: new target


HOLD = Signal()
EXIT = Signal("exit")


class PositionView:
    __slots__ = ("side", "entry_price", "entry_i", "stop", "target", "qty", "best_close", "bars_held")

    def __init__(self):
        self.side = 0


class Ctx:
    """Read-only view of bar t and its past."""

    __slots__ = ("i", "_a", "position", "symbol")

    def __init__(self, arrays: dict[str, np.ndarray], symbol: str):
        self._a = arrays
        self.i = 0
        self.position: PositionView | None = None
        self.symbol = symbol

    def __getitem__(self, k: str):
        return self._a[k][self.i]

    def prev(self, k: str, n: int = 1):
        j = self.i - n
        return self._a[k][j] if j >= 0 else math.nan

    def hist(self, k: str, n: int) -> np.ndarray:
        return self._a[k][max(0, self.i - n + 1): self.i + 1]


class Strategy:
    name: str = "base"
    params: dict = {}
    grid: dict = {}               # walk-forward selection grid: {param: [values]}
    timeframes: list[str] = ["1h"]

    def __init__(self, **kw):
        unknown = set(kw) - set(self.params)
        if unknown:
            raise ValueError(f"{self.name}: unknown params {unknown}")
        self.p = {**self.params, **kw}

    @classmethod
    def hypothesis(cls) -> str:
        doc = inspect.getmodule(cls).__doc__ or ""
        return doc.strip().splitlines()[0].strip() if doc.strip() else ""

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def bracket(self, ctx: Ctx, side: int) -> tuple[float, float | None]:
        raise NotImplementedError

    def entry(self, ctx: Ctx) -> Signal:
        raise NotImplementedError

    def manage(self, ctx: Ctx) -> Signal:
        return HOLD

    def on_bar(self, ctx: Ctx) -> Signal:
        return self.entry(ctx) if ctx.position is None else self.manage(ctx)


def _valid(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def enter(side: int, stop, target=None) -> Signal:
    if not _valid(stop):
        return HOLD
    return Signal("long" if side > 0 else "short", float(stop), float(target) if _valid(target) else None)


# ---------------------------------------------------------------- plugin loading

def plugin_path(name: str) -> Path:
    p = STRATEGY_DIR / f"{name}.py"
    if not p.exists():
        raise FileNotFoundError(f"strategy plugin not found: {p}")
    return p


def list_plugins() -> list[str]:
    return sorted(p.stem for p in STRATEGY_DIR.glob("*.py") if not p.stem.startswith("_"))


def plugin_hash(name: str) -> str:
    """Git blob hash of the plugin file content (stable whether or not committed)."""
    from perp_lab.plugins import blob_hash
    return blob_hash(plugin_path(name).read_bytes())


# ---------------------------------------------------------------- random-entry benchmark

class RandomEntry(Strategy):
    """Benchmark: random entries, identical stop/target/exit rules to the wrapped strategy."""

    name = "random_entry"

    def __init__(self, base: Strategy, entry_prob: float, seed: int):
        self.base = base
        self.p = dict(base.p)
        self.entry_prob = entry_prob
        self.seed = seed

    def prepare(self, bars: pd.DataFrame) -> pd.DataFrame:
        f = self.base.prepare(bars).copy()
        rng = np.random.default_rng(self.seed)
        n = len(bars)
        u = rng.random((n, 2))          # row-major: bar t's draws don't depend on series length
        fire = u[:, 0] < self.entry_prob
        side = np.where(u[:, 1] < 0.5, 1, -1)
        f["_rand"] = np.where(fire, side, 0)
        return f

    def bracket(self, ctx, side):
        return self.base.bracket(ctx, side)

    def entry(self, ctx):
        r = ctx["_rand"]
        if r == 0:
            return HOLD
        stop, tgt = self.base.bracket(ctx, int(r))
        return enter(int(r), stop, tgt)

    def manage(self, ctx):
        return self.base.manage(ctx)
