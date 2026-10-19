import json
import sqlite3

import pytest
import sqlalchemy as sa
import yaml

from perp_lab.config import Config, load_config
from perp_lab.runs import data_id, data_identity
from perp_lab.store import db
from tests.test_store import cfg, fake_run  # noqa: F401

U = ["BTCUSDT", "ETHUSDT"]


def other(cfg, **raw):
    return Config(raw={**cfg.raw, **raw}, path=cfg.path)


def test_data_id_follows_history_universe_and_costs_only(cfg):
    base = data_id(data_identity(cfg, U))
    costs = {**cfg["costs"], "slippage_model": {**cfg["costs"]["slippage_model"], "alpha": 9.0}}
    br = cfg["margin_brackets"]
    assert data_id(data_identity(other(cfg, judge={**cfg["judge"], "dsr_min": 0.5}), U)) == base
    assert data_id(data_identity(other(cfg, history_start="2019-01-01"), U)) != base
    assert data_id(data_identity(cfg, ["BTCUSDT"])) != base
    assert data_id(data_identity(other(cfg, costs=costs), U)) != base
    assert data_id(data_identity(other(cfg, margin_brackets={**br, "ETHUSDT": {"mmr": 0.02, "cum": 0.0}}), U)) != base
    assert data_id(data_identity(other(cfg, margin_brackets={**br, "SOLUSDT": {"mmr": 0.02, "cum": 0.0}}), U)) == base


def test_the_same_config_written_differently_has_the_same_data_id(cfg, tmp_path):
    raw = yaml.safe_load(open(cfg.path))
    raw["costs"] = dict(reversed(list(raw["costs"].items())))
    raw["costs"]["slippage_bps"] = int(raw["costs"]["slippage_bps"])          # 2 instead of 2.0
    p = tmp_path / "again.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    assert data_id(data_identity(load_config(p), U)) == data_id(data_identity(cfg, U))


def test_every_written_run_records_its_data(cfg):
    fake_run(cfg, "r1")
    s = json.loads((cfg.runs_dir / "r1" / "summary.json").read_text())
    assert s["data"]["history_start"] == str(cfg.history_start.date()) and s["data"]["universe"] == U
    assert s["data_id"] == data_id(data_identity(cfg, U)) == db.run_row(cfg, "r1")["data_id"]


def test_an_existing_store_gains_the_data_id_column(cfg):
    fake_run(cfg, "old")
    path = cfg["store"]["url"][len("sqlite:///"):]
    with sqlite3.connect(path) as c:
        c.execute("ALTER TABLE runs DROP COLUMN data_id")
    db._engine.cache_clear()
    assert db.run_row(cfg, "old")["data_id"] is None                 # added back, empty for old runs
    fake_run(cfg, "new")
    assert db.run_row(cfg, "new")["data_id"] == data_id(data_identity(cfg, U))


def stale_inspect(monkeypatch, times):
    """sa.inspect that hides runs.data_id from its first `times` callers, as a process that looked
    before another one added the column would see it. Returns the list of calls made."""
    real, calls = sa.inspect, []

    class Stale:
        def __init__(self, inspector):
            self.inspector = inspector

        def get_columns(self, table):
            return [c for c in self.inspector.get_columns(table) if c["name"] != "data_id"]

    def inspect(eng):
        calls.append(eng)
        return Stale(real(eng)) if len(calls) <= times else real(eng)
    monkeypatch.setattr(sa, "inspect", inspect)
    return calls


def test_a_store_migrated_by_another_process_meanwhile_opens(cfg, monkeypatch):
    # Two processes open an old store at once: both see no data_id, one adds it, and the other's
    # ALTER then fails on a column that now exists. That is not an error.
    fake_run(cfg, "r1")
    db._engine.cache_clear()
    calls = stale_inspect(monkeypatch, times=1)
    assert db.run_row(cfg, "r1")["data_id"] == data_id(data_identity(cfg, U))
    assert len(calls) == 2                                  # the columns were looked at again


def test_a_failed_migration_still_fails_when_the_column_is_missing(cfg, monkeypatch):
    fake_run(cfg, "r1")
    db._engine.cache_clear()
    stale_inspect(monkeypatch, times=2)
    with pytest.raises(sa.exc.OperationalError, match="data_id"):
        db.run_row(cfg, "r1")


def test_a_parent_run_on_other_data_is_not_matched(cfg):
    fake_run(other(cfg, history_start="2019-01-01"), "elsewhere", strategy="p", strategy_hash="hp")
    assert db.latest_wf_run(cfg, "p", "4h", "hp", U, data_id=data_id(data_identity(cfg, U))) is None
    fake_run(cfg, "here", strategy="p", strategy_hash="hp")
    assert db.latest_wf_run(cfg, "p", "4h", "hp", U, data_id=data_id(data_identity(cfg, U)))["run_id"] == "here"


def test_n_counts_a_plugin_version_once_across_data(cfg):
    fake_run(other(cfg, history_start="2019-01-01"), "a", strategy="p", strategy_hash="hp")
    fake_run(cfg, "b", strategy="p", strategy_hash="hp")
    assert db.trial_stats(cfg)[0] == 1
