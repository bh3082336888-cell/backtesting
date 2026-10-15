import json

import numpy as np
import pandas as pd
import pytest
import yaml

from perp_lab.config import load_config
from perp_lab.paper.tracking import count_regime_changes, paper_regime_status, replay_divergence
from perp_lab.runs import write_run
from perp_lab.store import db


@pytest.fixture
def cfg(tmp_path):
    raw = yaml.safe_load(open("config/lab.yaml"))
    raw["runs_dir"] = str(tmp_path / "runs")
    raw["store"] = {"url": f"sqlite:///{tmp_path}/lab.db"}
    p = tmp_path / "lab.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(p)


def fake_run(cfg, run_id, strategy="s1", tf="4h", verdict="noise", sharpe=0.3, parent=None, mode="walkforward"):
    idx = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
    rng = np.random.default_rng(abs(hash(run_id)) % 2**32)
    daily = pd.DataFrame({"BTCUSDT": rng.normal(0, 0.01, 120), "ETHUSDT": rng.normal(0, 0.01, 120)}, index=idx)
    daily["PORTFOLIO"] = daily.mean(axis=1)
    folds = [{"fold": i, "train_start": str(idx[0].date()), "train_end": str(idx[30 * i].date()),
              "test_start": str(idx[30 * i].date()), "test_end": str(idx[min(30 * i + 30, 119)].date()),
              "combo": "n=20", "params": {"n": 20}, "is_score": 1.0} for i in range(1, 4)]
    rec = {"run_id": run_id, "strategy": strategy, "strategy_hash": "abc", "hypothesis": "h",
           "params": {"grid": {"n": [10, 20], "k": [1, 2]}, "defaults": {"n": 20, "k": 1}},
           "universe": ["BTCUSDT", "ETHUSDT"], "timeframes": [tf], "validation_mode": mode,
           "date_start": "2024-01-01", "date_end": "2024-04-30", "status": verdict,
           "fail_reasons": [] if verdict == "candidate" else ["x"],
           "metrics": {"sharpe": sharpe, "max_dd": -0.1, "trades": 50, "total_return": 0.05},
           "is_metrics": {"sharpe": 1.2},
           "random_entry": {"strategy_pctile_vs_random": 0.6, "beats_random": False, "seeds": 10,
                            "percentile": 95, "random_sharpe_p95": 0.9},
           "sensitivity": {"narrow_island": False, "base_sharpe": sharpe, "base_params": {"n": 20}},
           "by_symbol": {"BTCUSDT": {"sharpe": 0.5, "max_dd": -0.1, "total_return": 0.1, "trades": 25},
                         "ETHUSDT": {"sharpe": -0.2, "max_dd": -0.2, "total_return": -0.05, "trades": 25}},
           "regime_split": [{"regime": "trending", "trades": 30, "win_rate": 0.4, "avg_r": 0.1}],
           "folds": folds, "parent_run_id": parent, "created_at": "2026-10-01T00:00:00+00:00"}
    grid = pd.DataFrame([{"n": n, "k": k, "sharpe": n / 10 - k} for n in (10, 20) for k in (1, 2)])
    sens = pd.DataFrame([{"param": "n", "step": s, "value": 20 * (1 + s), "sharpe": sharpe + s} for s in (-0.2, 0.2)])
    write_run(cfg, rec, {"oos_daily": daily, "grid": grid, "sensitivity": sens,
                         "random_sharpes": pd.DataFrame({"sharpe": rng.normal(0, 0.5, 50)})})
    return rec


def test_write_run_records_automatically_with_child_rows(cfg):
    fake_run(cfg, "r1")
    r = db.read_table(cfg, db.runs)
    assert r.run_id.tolist() == ["r1"] and r.lineage_id.iloc[0] == "r1"
    f = db.read_table(cfg, db.run_folds)
    assert len(f) == 3 and f.oos_sharpe.notna().all()
    assert len(db.read_table(cfg, db.run_symbols)) == 2


def test_record_is_idempotent_and_ingest_catches_unrecorded(cfg):
    fake_run(cfg, "r1")
    rec = json.loads((cfg.runs_dir / "r1" / "summary.json").read_text())
    db.record_run(cfg, rec, cfg.runs_dir / "r1")
    assert len(db.read_table(cfg, db.runs)) == 1
    # a run dir that never made it into the store (crash) is picked up by ingest
    d = cfg.runs_dir / "r2"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({**rec, "run_id": "r2"}))
    assert db.ingest_runs(cfg) == ["r2"]
    assert db.ingest_runs(cfg) == []


def test_child_inherits_lineage_and_edge_requires_reason(cfg):
    fake_run(cfg, "p")
    fake_run(cfg, "c", parent="p")
    assert db.lineage_of(cfg, "c") == "p"
    with pytest.raises(ValueError):
        db.add_edge(cfg, "p", "c", "kill", "  ")
    with pytest.raises(ValueError):
        db.add_edge(cfg, "p", "nope", "kill", "x")
    db.add_edge(cfg, "p", "c", "kill", "fails random-entry")
    e = db.read_table(cfg, db.lineage_edges)
    assert e.decision.tolist() == ["kill"] and e.reason.iloc[0] == "fails random-entry"


def test_paper_approval_only_for_candidates_and_never_live(cfg):
    fake_run(cfg, "n1", verdict="noise")
    fake_run(cfg, "c1", strategy="s2", verdict="candidate")
    with pytest.raises(PermissionError):
        db.approve_paper(cfg, "n1", "me")
    db.approve_paper(cfg, "c1", "me")
    assert db.current_status(cfg, "s2", "4h", "candidate") == "paper"
    with pytest.raises(PermissionError):
        db.set_status(cfg, "s2", "4h", "live", "me")
    with pytest.raises(PermissionError):
        db.set_status(cfg, "s2", "4h", "paper", "me")


def _labels(seq):
    """seq: list of (label, days) → 4h label series."""
    out, t = [], pd.Timestamp("2026-01-01", tz="UTC")
    for lab, days in seq:
        for _ in range(days * 6):
            out.append((t, lab))
            t += pd.Timedelta(hours=4)
    return pd.Series([l for _, l in out], index=[x for x, _ in out])


def test_regime_change_needs_persistence():
    start = pd.Timestamp("2026-01-11", tz="UTC")
    # flickers of < 7 days don't count
    flick = _labels([("trending", 10)] + [("ranging", 2), ("trending", 2)] * 10)
    assert count_regime_changes(flick, start, 7)["changes"] == 0
    # a persisted switch after the start counts
    real = _labels([("trending", 10), ("ranging", 3), ("high_vol", 9), ("trending", 8)])
    r = count_regime_changes(real, start, 7)
    assert r["changes"] == 2 and r["sequence"][0]["to"] == "high_vol"
    # a switch before the paper start does not count
    assert count_regime_changes(real, pd.Timestamp("2026-02-01", tz="UTC"), 7)["changes"] == 0


def test_paper_eligible_needs_days_and_regime_change(cfg):
    fake_run(cfg, "pp", mode="paper")
    started = pd.Timestamp("2026-01-01", tz="UTC")
    with db.engine(cfg).begin() as c:
        c.execute(db.runs.update().where(db.runs.c.run_id == "pp").values(paper_started_at=started.to_pydatetime()))
    day = lambda d: (started + pd.Timedelta(days=d)).to_pydatetime()
    assert db.update_paper_tracking(cfg, "pp", 3, {}, now=day(30))["eligible"] is False   # too early
    assert db.update_paper_tracking(cfg, "pp", 0, {}, now=day(90))["eligible"] is False   # no regime change
    assert db.update_paper_tracking(cfg, "pp", 1, {}, now=day(60))["eligible"] is True


def test_regime_count_is_min_across_pairs(cfg):
    start = pd.Timestamp("2026-01-11", tz="UTC")
    moved = _labels([("trending", 10), ("ranging", 10)])
    stuck = _labels([("trending", 20)])
    n, detail = paper_regime_status(cfg, {"A": moved, "B": stuck}, start)
    assert n == 0 and detail["A"]["changes"] == 1


def test_replay_divergence_on_identical_execution_is_zero(cfg, tmp_path):
    from perp_lab.engine.backtest import EngineParams, run_backtest
    from perp_lab.strategy import load_strategy_class
    from tests.conftest import make_bars
    cls = load_strategy_class("donchian_atr")
    bars = make_bars(n=1500, seed=5)
    bars["live"] = np.arange(len(bars)) >= 500
    d = tmp_path / "paper"
    d.mkdir()
    bars.reset_index(drop=True).to_parquet(d / "bars_BTCUSDT.parquet")
    s = cls()
    res = run_backtest(s, bars, s.prepare(bars), symbol="BTCUSDT", tf="1h",
                       ep=EngineParams.from_config(cfg, "BTCUSDT"), start_i=500)
    pd.DataFrame(res.trades).assign(source="paper").to_parquet(d / "trades.parquet")
    dv = replay_divergence(cfg, d, cls, {}, "1h")
    assert len(dv) and np.allclose(dv.divergence, 0)
