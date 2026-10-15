"""Every dashboard page renders against a populated store without exceptions."""

import pytest
from streamlit.testing.v1 import AppTest

from perp_lab.store import db
from tests.test_store import cfg, fake_run  # noqa: F401


@pytest.fixture
def populated(cfg, monkeypatch):
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


def test_approval_button_only_for_candidate(populated):
    at = AppTest.from_file("../dashboard/app.py", default_timeout=60)
    at.run()
    at.sidebar.radio(key="page").set_value("Strategy detail").run()
    at.selectbox(key="detail_pick").set_value("alpha @ 4h").run()
    labels = [b.label for b in at.get("form_submit_button")]
    assert "Approve for paper trading" not in labels
    at.selectbox(key="detail_pick").set_value("alpha_child @ 4h").run()
    labels = [b.label for b in at.get("form_submit_button")]
    assert "Approve for paper trading" in labels
