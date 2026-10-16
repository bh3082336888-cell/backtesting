import pytest

from perp_lab.store import db
from tests.test_store import cfg, fake_run  # noqa: F401


def jrow(**over):
    row = {"strategy": "s1", "timeframe": "4h", "strategy_hash": "h1", "child_run_id": None,
           "parent_run_id": None, "lineage_id": "fam", "decision": "kill", "reasons": ["x"],
           "holdout_status": "n/a", "holdout_reason": "n/a"}
    return {**row, **over}


def test_record_and_fetch_judgement(cfg):
    fake_run(cfg, "r1")
    jid = db.record_judgement(cfg, jrow(child_run_id="r1"))
    j = db.get_judgement(cfg, "h1", "4h")
    assert j["id"] == jid and j["reasons"] == ["x"] and j["strategy"] == "s1"
    assert db.judgement_for_run(cfg, "r1")["id"] == jid
    assert db.get_judgement_by_id(cfg, jid)["decision"] == "kill"
    assert db.get_judgement(cfg, "h1", "1h") is None


def test_recording_the_same_strategy_version_twice_returns_the_first(cfg):
    a = db.record_judgement(cfg, jrow())
    b = db.record_judgement(cfg, jrow(decision="promote", holdout_status="pending", holdout_reason="awaiting_exam"))
    assert a == b and db.get_judgement_by_id(cfg, a)["decision"] == "kill"


def test_invalid_decision_or_status_is_rejected(cfg):
    with pytest.raises(ValueError):
        db.record_judgement(cfg, jrow(decision="maybe"))
    with pytest.raises(ValueError):
        db.record_judgement(cfg, jrow(holdout_status="live"))


def test_update_holdout_and_pending_queue(cfg):
    pend = dict(decision="promote", holdout_status="pending", holdout_reason="awaiting_exam")
    a = db.record_judgement(cfg, jrow(strategy_hash="ha", **pend))
    b = db.record_judgement(cfg, jrow(strategy_hash="hb", **pend))
    assert [j["id"] for j in db.pending_judgements(cfg)] == [a, b]
    db.update_holdout(cfg, a, "blocked", "budget_spent")
    assert [j["id"] for j in db.pending_judgements(cfg)] == [b]
    assert db.get_judgement_by_id(cfg, a)["holdout_reason"] == "budget_spent"
    with pytest.raises(ValueError):
        db.update_holdout(cfg, b, "maybe", "x")


def test_trial_stats_count_distinct_versions_and_use_the_latest_sharpe(cfg):
    assert db.trial_stats(cfg) == (0, 0.0)
    fake_run(cfg, "a1", strategy="a", strategy_hash="ha", sharpe=0.9, created_at="2026-10-01T00:00:00+00:00")
    fake_run(cfg, "a2", strategy="a", strategy_hash="ha", sharpe=0.45, created_at="2026-10-02T00:00:00+00:00")
    fake_run(cfg, "b1", strategy="b", strategy_hash="hb", sharpe=-0.2)
    fake_run(cfg, "c1", strategy="c", strategy_hash="hc", sharpe=0.1)
    fake_run(cfg, "x1", strategy="x", strategy_hash="hx", sharpe=5.0, mode="IS")   # not a walk-forward trial
    n, v = db.trial_stats(cfg)
    # variance of the daily Sharpes 0.45, -0.2, 0.1 (annualised / sqrt(365)), computed by hand
    assert n == 3 and v == pytest.approx(0.0002899543378995434, rel=1e-12)


def test_family_of_is_the_family_of_the_first_judgement_only(cfg):
    assert db.family_of(cfg, "s1") is None
    fake_run(cfg, "r1", strategy="s1", created_at="2026-10-01T00:00:00+00:00")
    assert db.family_of(cfg, "s1") is None                 # a walk-forward run alone is not a family
    db.record_judgement(cfg, jrow(strategy="s1", strategy_hash="h0", lineage_id=None))
    db.record_judgement(cfg, jrow(strategy="s1", strategy_hash="h1", lineage_id="fam1"))
    db.record_judgement(cfg, jrow(strategy="s1", strategy_hash="h2", lineage_id="fam2"))
    assert db.family_of(cfg, "s1") == "fam1"


def test_latest_wf_run_matches_hash_and_universe(cfg):
    fake_run(cfg, "old", strategy="p", strategy_hash="hp", created_at="2026-10-01T00:00:00+00:00")
    fake_run(cfg, "new", strategy="p", strategy_hash="hp", created_at="2026-10-02T00:00:00+00:00")
    fake_run(cfg, "edited", strategy="p", strategy_hash="hp2", created_at="2026-10-03T00:00:00+00:00")
    fake_run(cfg, "btc_only", strategy="p", strategy_hash="hp", universe=["BTCUSDT"],
             created_at="2026-10-04T00:00:00+00:00")
    assert db.latest_wf_run(cfg, "p", "4h", "hp", ["BTCUSDT", "ETHUSDT"])["run_id"] == "new"
    assert db.latest_wf_run(cfg, "p", "4h", "hp", ["BTCUSDT"])["run_id"] == "btc_only"
    assert db.latest_wf_run(cfg, "p", "1h", "hp", ["BTCUSDT", "ETHUSDT"]) is None
    fake_run(cfg, "fam", strategy="q", strategy_hash="hq", lineage_id="family:q")
    assert db.latest_wf_run(cfg, "q", "4h", "hq", ["BTCUSDT", "ETHUSDT"], lineage_id="family:q")["run_id"] == "fam"
    assert db.latest_wf_run(cfg, "q", "4h", "hq", ["BTCUSDT", "ETHUSDT"], lineage_id="family:z") is None
    assert db.latest_wf_run(cfg, "q", "4h", "hq", ["BTCUSDT", "ETHUSDT"])["run_id"] == "fam"
    assert db.run_row(cfg, "new")["strategy_hash"] == "hp" and db.run_row(cfg, "nope") is None
