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
from perp_lab.store import db
from perp_lab.strategy import plugin_hash

DAYS = pd.date_range("2022-07-01", periods=1461, freq="D", tz="UTC")


def series(mu, sd=0.01, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(mu, sd, len(DAYS)), index=DAYS)


STRONG = series(0.003, seed=1)    # DSR 1.0
NOISE = series(0.0, seed=2)       # DSR 0.02
WEAK = series(0.0002, seed=4)     # Sharpe +0.39 annualised, DSR 0.78: a "candidate" the DSR kills

WIDE_TEXT = Path("strategies/donchian_atr_wide.py").read_text()   # declares PARENT = "donchian_atr"

H_MS = 3_600_000


def stub_trades() -> pd.DataFrame:
    """Four OOS trades on two coins and two folds, the engine's columns that diagnostics read."""
    t0 = DAYS[800].value // 1_000_000
    t = pd.DataFrame({"symbol": ["BTCUSDT", "BTCUSDT", "ETHUSDT", "ETHUSDT"], "side": [1, -1, 1, 1],
                      "entry_time": t0 + np.arange(4) * 24 * H_MS, "notional": 1000.0,
                      "pnl_gross": [12.0, -4.0, 6.0, -2.0], "slippage": 1.0, "fees": 1.1, "funding": 0.2,
                      "exit_reason": ["stop", "signal", "stop", "target"], "bars_held": 3,
                      "regime": "trending", "fold": [1, 1, 2, 2]})
    return t.assign(exit_time=t.entry_time + 12 * H_MS, pnl=t.pnl_gross - t.fees - t.funding)


def stub_folds() -> list[dict]:
    return [{"fold": k, "train_start": str(DAYS[0].date()), "train_end": str(DAYS[365 * k].date()),
             "test_start": str(DAYS[365 * k].date()), "test_end": str(DAYS[365 * k + 365].date()),
             "combo": "atr_mult=3.0", "params": {"atr_mult": 3.0}, "is_score": 1.0} for k in (1, 2)]


CHEATER = '''"""Peeks at the next close, to prove the judge's lookahead check works."""
from perp_lab.indicators import atr
from perp_lab.strategy import HOLD, Strategy, enter

PARENT = "donchian_atr"


class Cheater(Strategy):
    params = {}
    timeframes = ["4h"]

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
    """Stands in for validate_strategy: records a walk-forward run with a scripted verdict and OOS series, and
    the same few OOS trades and summary figures every time."""

    def __init__(self, outcomes: dict):
        self.outcomes, self.calls = outcomes, []

    def __call__(self, cfg, name, tf, symbols=None, seeds=None, workers=None, notes="",
                 parent_run_id=None, lineage_id=None, src=None):
        self.calls.append(name)
        verdict, oos = self.outcomes[name]
        rid = new_run_id()
        h = src.hash if src is not None else plugin_hash(name)
        db.register_trial(cfg, name, h, tf)   # as the real validate_strategy does
        rec = {"run_id": rid, "strategy": name, "strategy_hash": h, "hypothesis": "h",
               "params": {}, "universe": symbols or cfg.symbols, "timeframes": [tf],
               "validation_mode": "walkforward", "status": verdict,
               "fail_reasons": [] if verdict == "candidate" else ["does not beat random-entry p95 OOS"],
               "metrics": {"sharpe": sharpe(oos), "trades": 100}, "is_metrics": {"sharpe": 1.5},
               "random_entry": {"seeds": 10, "percentile": 95, "random_sharpe_mean": 0.05, "random_sharpe_p95": 0.7,
                                "strategy_pctile_vs_random": 0.99 if verdict == "candidate" else 0.4,
                                "beats_random": verdict == "candidate", "long_share": 0.75},
               "before_costs": {"sharpe": 0.9, "random_mean": 0.15, "random_p95": 0.8, "pctile": 0.97,
                                "long_share": 0.75},
               "min_oos_trades": 64, "folds": stub_folds(),
               "sensitivity": {"base_params": dict((src or PluginSource.read(name)).meta.params),
                               "narrow_island": False},
               "parent_run_id": parent_run_id, "lineage_id": lineage_id,
               "created_at": pd.Timestamp.now(tz="UTC").isoformat()}
        d = write_run(cfg, rec, {"oos_daily": pd.DataFrame({"PORTFOLIO": oos}), "oos_trades": stub_trades()})
        return {**rec, "run_dir": str(d)}, "report"
