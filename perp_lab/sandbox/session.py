"""Plugin code runs here, and only here: inside a confined worker, or in the tests' in-process twin.

`worker.py` imports this module only after `confine()`, so the threads numpy and pyarrow start
inherit the confinement. One `Session` per worker: it holds the plugin class (`load`), one
symbol's bars (`bars`) and runs the plugin over them (`run`). Its results are plain values the
worker encodes for the parent; an exception becomes `error_reply`.
"""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401  (imported now, so its threads start under confinement)

import perp_lab.indicators  # noqa: F401  (plugins import it)
from perp_lab.engine.backtest import EngineParams, run_backtest
from perp_lab.sandbox.decisions import Decisions, Recorder
from perp_lab.sandbox.errors import PluginError, ProtocolError, ascii_text
from perp_lab.sandbox.protocol import decisions_sections
from perp_lab.strategy import RandomEntry, Strategy

_META = ("params", "grid", "timeframes")


def load_plugin(name: str, text: bytes) -> type[Strategy]:
    """Execute the plugin's source in a fresh module `lab_strategies.<name>`; its one Strategy class."""
    modname = f"lab_strategies.{name}"
    mod = types.ModuleType(modname)
    sys.modules[modname] = mod
    code = compile(text, f"strategies/{name}.py", "exec", dont_inherit=True)
    exec(code, mod.__dict__)
    found = [v for v in vars(mod).values()
             if isinstance(v, type) and issubclass(v, Strategy) and v is not Strategy and v.__module__ == modname]
    if len(found) != 1:
        raise ValueError(f"{name}: expected exactly one Strategy subclass, found {len(found)}")
    cls = found[0]
    cls.name = name
    return cls


def _json(value, what: str):
    """The value as JSON gives it back, which is what the parent sees; PluginError if it isn't JSON."""
    try:
        return json.loads(json.dumps(value, allow_nan=True))
    except (TypeError, ValueError, RecursionError):
        raise PluginError("metadata", "", f"{what} is not a JSON value") from None


def _features(feats: pd.DataFrame, index: pd.Index) -> tuple[list[str], list[np.ndarray]]:
    """prepare()'s columns as float64 arrays (bool and int are cast; anything else is refused)."""
    if not isinstance(feats, pd.DataFrame) or not feats.index.equals(index):
        raise PluginError("features", "", "prepare() must return a DataFrame on the bars index")
    names = [str(c) for c in feats.columns]
    if len(set(names)) != len(names):
        raise PluginError("features", "", "feature names must be unique")
    arrays = []
    for name, (_, col) in zip(names, feats.items()):
        if col.dtype.kind not in "biuf":
            raise PluginError("features", "", f"feature {name!r} has dtype {col.dtype}; features must be numeric")
        arrays.append(np.ascontiguousarray(col.to_numpy(dtype="float64", na_value=np.nan), dtype="<f8"))
    return names, arrays


class Session:
    def __init__(self):
        self.cls: type[Strategy] | None = None
        self.symbol = self.tf = None
        self._bars: pd.DataFrame | None = None

    def load(self, name: str, text: bytes) -> dict:
        """Load the plugin; its params, grid and timeframes at runtime, for the parent to compare."""
        self.cls = load_plugin(name, text)
        return {k: _json(getattr(self.cls, k), k) for k in _META}

    def bars(self, symbol: str, tf: str, df: pd.DataFrame) -> int:
        """Replace the bars this worker holds."""
        self.symbol, self.tf, self._bars = symbol, tf, df
        return len(df)

    def run(self, params: dict, ep: dict, k: int | None, start_i: int, random: dict | None,
            features: bool) -> tuple[Decisions, tuple[int, float], list[str], list[np.ndarray]]:
        """One backtest of the plugin on the held bars (the first k when k is set), recorded."""
        if self.cls is None or self._bars is None:
            raise ProtocolError("run before load and bars")
        strat = self.cls(**params)
        if random is not None:
            strat = RandomEntry(strat, random["prob"], random["seed"])
        bars = self._bars if k is None else self._bars.iloc[:k]
        rec = Recorder(strat)
        feats = rec.prepare(bars)
        eng = EngineParams(**ep)
        res = run_backtest(rec, bars, feats, symbol=self.symbol, tf=self.tf, ep=eng, start_i=start_i)
        fp = (len(res.trades), float(res.equity[-1]) if len(res.equity) else float(eng.equity0))
        names, arrays = _features(feats, bars.index) if features else ([], [])
        return Decisions.from_log(rec.log), fp, names, arrays


def run_reply(result, features: bool) -> tuple[dict, dict]:
    """`Session.run`'s result as a reply frame: header and raw sections."""
    dec, fp, names, arrays = result
    header = {"ok": True, "fingerprint": [fp[0], fp[1]], "features": names if features else None}
    return header, {**decisions_sections(dec), **{f"f{j}": a for j, a in enumerate(arrays)}}


def error_reply(exc: BaseException) -> dict:
    """An exception as an error reply: a PluginError keeps its kind; anything else is "exception"."""
    if isinstance(exc, PluginError):
        kind, type_, message = exc.kind, exc.type, exc.message
    else:
        kind, type_ = "exception", type(exc).__name__
        try:
            message = str(exc)
        except BaseException:                 # plugin code: its __str__ may raise anything
            message = "<unprintable>"
    return {"error": {"kind": kind, "type": ascii_text(type_), "message": ascii_text(message)}}
