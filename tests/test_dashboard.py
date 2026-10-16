"""Every dashboard page renders against a populated store without exceptions."""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from perp_lab.judge import budget
from perp_lab.store import db
from tests.test_store import cfg, fake_run  # noqa: F401


@pytest.fixture
def populated(cfg, monkeypatch):
    st.cache_data.clear()  # the app caches by function, not by store: drop other tests' data
    fake_run(cfg, "p1", strategy="alpha", sharpe=0.4)
    fake_run(cfg, "c1", strategy="alpha_child", sharpe=0.6, parent="p1", verdict="candidate")
    fake_run(cfg, "b1", strategy="beta", sharpe=-0.5)
    db.add_edge(cfg, "p1", "c1", "kill", "fails random-entry")
    monkeypatch.setenv("PERP_LAB_CONFIG", str(cfg.path))
    return cfg


@pytest.mark.parametrize("page", ["Strategies", "Strategy detail", "Pairs", "Lineage"])
def test_page_renders(populated, page):
    at = AppTest.from_file("../dashboard/app.py", default_timeout=60)
    at.run()
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, [e.message for e in at.exception]


def _detail(pick):
    at = AppTest.from_file("../dashboard/app.py", default_timeout=60)
    at.run()
    at.sidebar.radio(key="page").set_value("Strategy detail").run()
    at.selectbox(key="detail_pick").set_value(pick).run()
    return at


def _judge_c1(cfg, status, reason, holdout_run_id=None):
    jid = db.record_judgement(cfg, {
        "strategy": "alpha_child", "timeframe": "4h", "strategy_hash": "abc", "child_run_id": "c1",
        "parent_run_id": "p1", "lineage_id": "p1", "decision": "promote", "reasons": [],
        "dsr": 0.97, "n_trials": 3, "sr0": 0.01, "parent_passed": False, "beats_parent_frac": None,
        "holdout_status": status, "holdout_reason": reason})
    if holdout_run_id:
        db.update_holdout(cfg, jid, status, reason, holdout_run_id)


def test_approval_needs_a_candidate_with_a_passed_exam(populated):
    labels = lambda at: [b.label for b in at.get("form_submit_button")]  # noqa: E731
    assert "Approve for paper trading" not in labels(_detail("alpha @ 4h"))
    at = _detail("alpha_child @ 4h")
    assert "Approve for paper trading" not in labels(at)
    assert any("passed holdout exam" in c.value for c in at.caption)
    _judge_c1(populated, "passed", "passed")
    assert "Approve for paper trading" in labels(_detail("alpha_child @ 4h"))


def test_judgement_panel_shows_holdout_details_to_the_human(populated):
    fake_run(populated, "h1", strategy="alpha_child", mode="holdout", verdict="passed", sharpe=1.1)
    _judge_c1(populated, "passed", "passed", holdout_run_id="h1")
    at = _detail("alpha_child @ 4h")
    assert not at.exception, [e.message for e in at.exception]
    assert {"Decision", "Deflated Sharpe", "Holdout"} <= {m.label for m in at.metric}
    assert any("Holdout 2024-01-01" in c.value for c in at.caption)


def test_judgement_panel_survives_a_missing_holdout_run(populated):
    _judge_c1(populated, "passed", "passed", holdout_run_id="ghost")
    at = _detail("alpha_child @ 4h")
    assert not at.exception, [e.message for e in at.exception]
    assert any("not found" in c.value for c in at.caption)


def test_judgement_panel_survives_a_holdout_run_without_trades(populated):
    fake_run(populated, "h2", strategy="alpha_child", mode="holdout", verdict="failed")
    with db.engine(populated).begin() as c:
        c.execute(db.runs.update().where(db.runs.c.run_id == "h2").values(trades=None))
    _judge_c1(populated, "failed", "failed", holdout_run_id="h2")
    at = _detail("alpha_child @ 4h")
    assert not at.exception, [e.message for e in at.exception]
    assert any("0 trades" in c.value for c in at.caption)


def test_sidebar_shows_the_holdout_budget(populated):
    at = AppTest.from_file("../dashboard/app.py", default_timeout=60)
    at.run()
    assert any("not initialised" in c.value for c in at.sidebar.caption)
    budget.init_log(populated)
    at = AppTest.from_file("../dashboard/app.py", default_timeout=60)
    at.run()
    assert any("holdout reads: 0/10" in c.value for c in at.sidebar.caption)
