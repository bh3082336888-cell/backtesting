"""A trial is registered before plugin code first sees real bars, so a crash still counts toward N."""
from __future__ import annotations

import pytest

from perp_lab.store import db
from perp_lab.strategy import plugin_hash
from perp_lab.validate.pipeline import validate_strategy
from tests.test_store import cfg, fake_run  # noqa: F401


def test_register_is_idempotent_and_counts_attempts(cfg):
    assert db.trial_attempts(cfg, "abc", "4h") == 0
    assert db.register_trial(cfg, "s1", "abc", "4h") == 0
    assert db.register_trial(cfg, "s1", "abc", "4h", attempt=True) == 1
    assert db.register_trial(cfg, "s1", "abc", "4h", attempt=True) == 2
    assert db.register_trial(cfg, "s1", "abc", "4h") == 2
    assert db.trial_attempts(cfg, "abc", "4h") == 2 and len(db.read_table(cfg, db.trials)) == 1


def test_runs_from_before_the_trials_table_still_count_once(cfg):
    fake_run(cfg, "r1", strategy_hash="abc", tf="4h")
    db.register_trial(cfg, "s1", "abc", "4h")
    assert db.trial_stats(cfg)[0] == 1
    fake_run(cfg, "r2", strategy_hash="def", tf="4h")
    assert db.trial_stats(cfg)[0] == 2


def test_a_registered_trial_without_a_run_counts(cfg):
    db.register_trial(cfg, "s1", "abc", "4h")
    assert db.trial_stats(cfg) == (1, 0.0)


def test_validate_strategy_registers_before_the_walk_forward(cfg, monkeypatch):
    monkeypatch.setattr("perp_lab.validate.pipeline.walkforward",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        validate_strategy(cfg, "donchian_atr", "4h")
    assert db.trial_attempts(cfg, plugin_hash("donchian_atr"), "4h") == 0
    assert len(db.read_table(cfg, db.trials)) == 1
