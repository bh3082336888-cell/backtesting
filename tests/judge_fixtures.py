"""Shared judge test fixtures: scripted OOS series, a stand-in for validate_strategy, and a
temporary strategies/ directory so tests can add plugins without touching the real one."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from perp_lab.engine.metrics import sharpe
from perp_lab.plugins import PluginSource
from perp_lab.runs import new_run_id, write_run
from perp_lab.strategy import plugin_hash

DAYS = pd.date_range("2022-07-01", periods=1461, freq="D", tz="UTC")


def series(mu, sd=0.01, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(mu, sd, len(DAYS)), index=DAYS)


STRONG = series(0.003, seed=1)    # DSR 1.0
NOISE = series(0.0, seed=2)       # DSR 0.02
WEAK = series(0.0002, seed=4)     # Sharpe +0.39 annualised, DSR 0.78: a "candidate" the DSR kills

WIDE_TEXT = Path("strategies/donchian_atr_wide.py").read_text()   # declares PARENT = "donchian_atr"

CHEATER = '''"""Peeks at the next close, to prove the judge's lookahead check works."""
from perp_lab.indicators import atr
from perp_lab.strategy import HOLD, Strategy, enter

PARENT = "donchian_atr"


class Cheater(Strategy):
    params = {}

    def prepare(self, bars):
        return bars[[]].assign(nxt=bars.close.shift(-1), atr=atr(bars, 14))

    def bracket(self, ctx, side):
        return ctx["close"] - side * 2 * ctx["atr"], None

    def entry(self, ctx):
        n = ctx["nxt"]
        if n != n:
            return HOLD
        side = 1 if n > ctx["close"] else -1
        return enter(side, *self.bracket(ctx, side))
'''


def add_plugin(plugin_dir: Path, name: str, text: str) -> None:
    (plugin_dir / f"{name}.py").write_text(text)


@pytest.fixture
def plugins(tmp_path, monkeypatch):
    d = tmp_path / "strategies"
    d.mkdir()
    for name in ("donchian_atr", "donchian_atr_wide"):
        shutil.copy(f"strategies/{name}.py", d / f"{name}.py")
    monkeypatch.setattr("perp_lab.strategy.STRATEGY_DIR", d)
    return d


class StubValidate:
    """Stands in for validate_strategy: records a walk-forward run with a scripted verdict and OOS series."""

    def __init__(self, outcomes: dict):
        self.outcomes, self.calls = outcomes, []

    def __call__(self, cfg, name, tf, symbols=None, seeds=None, workers=None, notes="",
                 parent_run_id=None, lineage_id=None, src=None):
        self.calls.append(name)
        verdict, oos = self.outcomes[name]
        rid = new_run_id()
        h = src.hash if src is not None else plugin_hash(name)
        rec = {"run_id": rid, "strategy": name, "strategy_hash": h, "hypothesis": "h",
               "params": {}, "universe": symbols or cfg.symbols, "timeframes": [tf],
               "validation_mode": "walkforward", "status": verdict,
               "fail_reasons": [] if verdict == "candidate" else ["does not beat random-entry p95 OOS"],
               "metrics": {"sharpe": sharpe(oos), "trades": 100},
               "sensitivity": {"base_params": dict((src or PluginSource.read(name)).meta.params),
                               "narrow_island": False},
               "parent_run_id": parent_run_id, "lineage_id": lineage_id,
               "created_at": pd.Timestamp.now(tz="UTC").isoformat()}
        d = write_run(cfg, rec, {"oos_daily": pd.DataFrame({"PORTFOLIO": oos})})
        return {**rec, "run_dir": str(d)}, "report"
