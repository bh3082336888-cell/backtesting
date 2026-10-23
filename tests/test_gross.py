"""The before-costs pass re-runs the selected combos with fees and slippage off, and benchmarks
random entries on the same terms."""

import pandas as pd
import pytest

from perp_lab.validate import gross
from perp_lab.validate import walkforward as wfm
from tests.test_benchmarks import make_wf
from tests.test_store import cfg  # noqa: F401

DAYS = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")


def test_stitch_oos_joins_each_folds_selected_combo_on_its_test_window():
    lo = pd.Series(1.0, index=DAYS)
    hi = pd.Series(2.0, index=DAYS)
    R = {("a", "A"): {"daily": lo}, ("b", "A"): {"daily": hi}}
    sel = [{"combo": "a", "test_start": DAYS[0], "test_end": DAYS[4]},
           {"combo": "b", "test_start": DAYS[4], "test_end": DAYS[8]}]
    out = wfm.stitch_oos(R, sel, ["A"])["A"]
    assert out.tolist() == [1.0] * 4 + [2.0] * 4 and list(out.index) == list(DAYS[:8])


def test_before_costs_runs_each_selected_combo_once_with_zero_costs(cfg, monkeypatch):  # noqa: F811
    wf = make_wf()
    wf.selected = [{"combo": "n=1", "params": {"n": 1}, "test_start": DAYS[0], "test_end": DAYS[5]},
                   {"combo": "n=2", "params": {"n": 2}, "test_start": DAYS[5], "test_end": DAYS[9]},
                   {"combo": "n=1", "params": {"n": 1}, "test_start": DAYS[9], "test_end": DAYS[9]}]
    seen = []

    def run_many(cfg, src, jobs, workers=None, stderr_dir=None):
        seen.extend(jobs)
        return [{"params": j[0], "symbol": j[1], "daily": pd.Series(0.01, index=DAYS)} for j in jobs]

    monkeypatch.setattr(gross, "run_many", run_many)
    monkeypatch.setattr(gross, "random_entry", lambda *a, **kw: {})
    gross.before_costs(cfg, wf)
    assert sorted((j[0]["n"], j[1]) for j in seen) == [(1, "A"), (1, "B"), (2, "A"), (2, "B")]
    assert all(j[3] == gross.GROSS and j[4] is None and j[5] is None for j in seen)


def test_the_gross_random_benchmark_compares_the_gross_sharpe(cfg, monkeypatch):  # noqa: F811
    wf = make_wf()
    wf.selected = [{"combo": "default", "params": {}, "test_start": DAYS[0], "test_end": DAYS[9]}]
    rets = pd.Series([0.01, -0.005] * 5, index=DAYS)
    monkeypatch.setattr(gross, "run_many", lambda cfg, src, jobs, workers=None, stderr_dir=None: [
        {"params": j[0], "symbol": j[1], "daily": rets} for j in jobs])
    got = {}

    def fake(cfg, wf, seeds, workers, *, overrides, strat_sharpe):
        got.update(overrides=overrides, strat_sharpe=strat_sharpe)
        return {"random_sharpe_mean": 0.1, "random_sharpe_p95": 0.5, "strategy_pctile_vs_random": 0.9,
                "long_share": 0.7}

    monkeypatch.setattr(gross, "random_entry", fake)
    out = gross.before_costs(cfg, wf)
    assert got["overrides"] == gross.GROSS and got["strat_sharpe"] == out["sharpe"] != 0
    assert out == {"sharpe": out["sharpe"], "random_mean": 0.1, "random_p95": 0.5, "pctile": 0.9,
                   "long_share": 0.7}
