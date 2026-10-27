"""`lab verdict-refresh`: diagnostics recomputed on today's data and rules, with no decision field
changed and no trial added."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from perp_lab.cli import app
from perp_lab.judge.judge import JudgeRefused, judge, refresh_verdict
from perp_lab.store import db
from tests.judge_fixtures import CHEATER, NOISE, STRONG, StubValidate, add_plugin, plugins  # noqa: F401
from tests.test_store import cfg  # noqa: F401

DECISION = ("decision", "reasons", "verdict_reasons", "dsr", "n_trials", "sr0", "parent_passed",
            "beats_parent_frac", "holdout_status", "holdout_reason", "child_run_id", "parent_run_id", "lineage_id")


def _predate_todays_rules(run_dir):
    p = Path(run_dir) / "summary.json"
    s = json.loads(p.read_text())
    s.pop("before_costs")
    s["random_entry"].pop("long_share")
    p.write_text(json.dumps(s))


def _judged(cfg):
    v = StubValidate({"donchian_atr": ("noise", NOISE)})
    j = judge(cfg, "donchian_atr", "4h", validate=v)
    return v, j


def test_refresh_changes_no_decision_field_and_adds_no_trial(cfg, plugins):
    v, j = _judged(cfg)
    _predate_todays_rules(db.run_row(cfg, j["child_run_id"])["run_dir"])
    before = db.get_judgement_by_id(cfg, j["id"])
    trials = db.trial_stats(cfg)[0]
    out = refresh_verdict(cfg, "donchian_atr", "4h", validate=v)
    after = db.get_judgement_by_id(cfg, j["id"])
    assert all(after[k] == before[k] for k in DECISION)
    assert db.trial_stats(cfg)[0] == trials == 1
    assert out["diagnostics"]["data"] == "refreshed" and after["diagnostics_run_id"] != j["child_run_id"]
    assert json.loads(Path(out["verdict_path"]).read_text())["diagnostics"]["data"] == "refreshed"


def test_refresh_reuses_a_current_run_made_under_todays_rules(cfg, plugins):
    v, j = _judged(cfg)
    out = refresh_verdict(cfg, "donchian_atr", "4h", validate=lambda *a, **k: pytest.fail("made a run"))
    assert out["diagnostics"]["data"] == "judged" and v.calls == ["donchian_atr"]


def test_refresh_makes_a_run_when_the_latest_predates_todays_rules(cfg, plugins):
    v, j = _judged(cfg)
    _predate_todays_rules(db.run_row(cfg, j["child_run_id"])["run_dir"])
    refresh_verdict(cfg, "donchian_atr", "4h", validate=v)
    assert v.calls == ["donchian_atr", "donchian_atr"]
    # the new run is now the current one: a second refresh makes nothing
    refresh_verdict(cfg, "donchian_atr", "4h", validate=lambda *a, **k: pytest.fail("made a run"))


def test_refresh_refuses_an_unjudged_or_edited_plugin(cfg, plugins):
    with pytest.raises(JudgeRefused, match="no judgement of this plugin version at tf"):
        refresh_verdict(cfg, "donchian_atr", "4h", validate=StubValidate({}))
    v, j = _judged(cfg)
    f = plugins / "donchian_atr.py"
    f.write_text(f.read_text() + "# edited\n")
    with pytest.raises(JudgeRefused, match="no judgement of this plugin version at tf"):
        refresh_verdict(cfg, "donchian_atr", "4h", validate=v)
    assert v.calls == ["donchian_atr"]


def test_cli_verdict_refresh_refusal_exits_2(cfg, plugins):
    r = CliRunner().invoke(app, ["verdict-refresh", "donchian_atr", "--tf", "4h", "--config", str(cfg.path)])
    assert r.exit_code == 2 and "refused: no judgement of this plugin version at tf" in r.output


def test_refresh_refuses_a_kill_that_has_no_run(cfg, plugins):
    add_plugin(plugins, "cheater", CHEATER)
    v = StubValidate({})
    judge(cfg, "cheater", "4h", validate=v)
    with pytest.raises(JudgeRefused, match="no walk-forward run"):
        refresh_verdict(cfg, "cheater", "4h", validate=v)
    assert v.calls == []
