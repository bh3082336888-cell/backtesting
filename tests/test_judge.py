import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from perp_lab.cli import app
from perp_lab.judge.judge import VERDICT_KEYS, JudgeRefused, judge
from perp_lab.store import db
from tests.judge_fixtures import (CHEATER, NOISE, STRONG, WEAK, WIDE_TEXT, StubValidate, add_plugin,
                                  plugins)  # noqa: F401
from tests.test_store import cfg  # noqa: F401


def test_root_noise_is_killed_with_its_reasons(cfg, plugins):
    j = judge(cfg, "donchian_atr", "4h", validate=StubValidate({"donchian_atr": ("noise", NOISE)}))
    assert j["decision"] == "kill" and "does not beat random-entry p95 OOS" in j["reasons"]
    assert (j["holdout_status"], j["holdout_reason"]) == ("n/a", "n/a") and j["parent_run_id"] is None
    assert db.read_table(cfg, db.lineage_edges).empty
    out = json.loads(Path(j["verdict_path"]).read_text())
    assert set(out) == VERDICT_KEYS and out["decision"] == "kill" and out["walkforward"]["run_id"] == j["child_run_id"]


def test_strong_root_is_promoted_and_awaits_its_exam(cfg, plugins):
    j = judge(cfg, "donchian_atr", "4h", validate=StubValidate({"donchian_atr": ("candidate", STRONG)}))
    assert j["decision"] == "promote" and j["reasons"] == []
    assert (j["holdout_status"], j["holdout_reason"]) == ("pending", "awaiting_exam")
    assert j["dsr"] > 0.99 and j["n_trials"] == 1 and j["sr0"] == 0.0


def test_candidate_with_a_weak_deflated_sharpe_is_killed(cfg, plugins):
    j = judge(cfg, "donchian_atr", "4h", validate=StubValidate({"donchian_atr": ("candidate", WEAK)}))
    assert j["decision"] == "kill" and len(j["reasons"]) == 1
    assert j["reasons"][0].startswith("Deflated Sharpe 0.78 < 0.95")


def test_child_must_beat_a_passing_parent(cfg, plugins):
    v = StubValidate({"donchian_atr": ("candidate", STRONG), "donchian_atr_wide": ("candidate", STRONG - 0.001)})
    j = judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert v.calls == ["donchian_atr", "donchian_atr_wide"]          # the parent's walk-forward runs first
    assert j["decision"] == "kill" and j["parent_passed"] is True and j["beats_parent_frac"] == 0.0
    assert j["reasons"] == ["beats its passing parent in 0% of resamples < 95%"]
    e = db.read_table(cfg, db.lineage_edges)
    assert e.decision.tolist() == ["kill"] and e.decided_by.tolist() == ["harness"]


def test_child_that_beats_a_passing_parent_is_promoted_into_its_family(cfg, plugins):
    v = StubValidate({"donchian_atr": ("candidate", STRONG), "donchian_atr_wide": ("candidate", STRONG + 0.001)})
    j = judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert j["decision"] == "promote" and j["beats_parent_frac"] == 1.0
    assert j["lineage_id"] == db.run_row(cfg, j["parent_run_id"])["lineage_id"] == "family:donchian_atr"


def test_a_noise_parent_neither_lowers_nor_raises_the_bar(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE), "donchian_atr_wide": ("candidate", STRONG)})
    j = judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert j["decision"] == "promote" and j["parent_passed"] is False and j["beats_parent_frac"] is None


def test_an_existing_parent_run_is_reused(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE), "donchian_atr_wide": ("candidate", STRONG)})
    v(cfg, "donchian_atr", "4h", lineage_id="family:donchian_atr")
    judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert v.calls == ["donchian_atr", "donchian_atr_wide"]


def test_rejudging_an_unchanged_strategy_returns_the_existing_judgement(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE)})
    a = judge(cfg, "donchian_atr", "4h", validate=v)
    b = judge(cfg, "donchian_atr", "4h", validate=v)
    assert a["id"] == b["id"] and v.calls == ["donchian_atr"] and b["existing"] is True


def test_a_lookahead_cheater_is_killed_without_a_walk_forward_run(cfg, plugins):
    add_plugin(plugins, "cheater", CHEATER)
    v = StubValidate({})
    j = judge(cfg, "cheater", "4h", validate=v)
    assert j["decision"] == "kill" and j["child_run_id"] is None and v.calls == []
    assert j["reasons"] and all(r.startswith("lookahead: ") for r in j["reasons"])
    assert db.trial_stats(cfg) == (0, 0.0)
    assert json.loads(Path(j["verdict_path"]).read_text())["walkforward"] is None


@pytest.mark.parametrize("name,parent,msg", [("orphan", "nope", "not a plugin"), ("selfish", "selfish", "itself")])
def test_unknown_or_self_parent_is_refused(cfg, plugins, name, parent, msg):
    add_plugin(plugins, name, WIDE_TEXT.replace('PARENT = "donchian_atr"', f'PARENT = "{parent}"'))
    with pytest.raises(JudgeRefused, match=msg):
        judge(cfg, name, "4h", validate=StubValidate({}))


def test_a_byte_identical_child_is_refused(cfg, plugins):
    text = WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "twin_b"')
    add_plugin(plugins, "twin_a", text)
    add_plugin(plugins, "twin_b", text)
    with pytest.raises(JudgeRefused, match="byte-identical"):
        judge(cfg, "twin_a", "4h", validate=StubValidate({}))


def test_a_root_keeps_its_family_across_timeframes(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE)})
    a = judge(cfg, "donchian_atr", "4h", validate=v)
    b = judge(cfg, "donchian_atr", "1h", validate=v)
    assert a["lineage_id"] is not None and a["lineage_id"] == b["lineage_id"]


def test_cli_refusal_exits_2(cfg, plugins):
    add_plugin(plugins, "orphan", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "nope"'))
    r = CliRunner().invoke(app, ["judge", "orphan", "--tf", "4h", "--config", str(cfg.path)])
    assert r.exit_code == 2 and "refused" in r.output


def test_a_plain_walk_forward_run_from_another_family_is_not_reused(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE), "donchian_atr_wide": ("candidate", STRONG)})
    v(cfg, "donchian_atr", "1h")
    v(cfg, "donchian_atr", "4h")
    j = judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert v.calls == ["donchian_atr", "donchian_atr", "donchian_atr", "donchian_atr_wide"]
    assert j["lineage_id"] == "family:donchian_atr"
    assert db.run_row(cfg, j["parent_run_id"])["lineage_id"] == "family:donchian_atr"
    assert judge(cfg, "donchian_atr", "4h", validate=v)["lineage_id"] == "family:donchian_atr"


def test_a_grandchild_judged_first_shares_the_root_family(cfg, plugins):
    text = WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "donchian_atr_wide"') + "# grandchild variant\n"
    add_plugin(plugins, "grandchild", text)
    v = StubValidate({"donchian_atr_wide": ("noise", NOISE), "grandchild": ("candidate", STRONG),
                      "donchian_atr": ("noise", NOISE)})
    a = judge(cfg, "grandchild", "4h", validate=v)
    b = judge(cfg, "donchian_atr", "4h", validate=v)
    assert a["lineage_id"] == b["lineage_id"] == "family:donchian_atr"


def test_a_parent_chain_loop_is_refused(cfg, plugins):
    add_plugin(plugins, "loop_a", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "loop_b"') + "# a\n")
    add_plugin(plugins, "loop_b", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "loop_a"') + "# b\n")
    with pytest.raises(JudgeRefused, match="loops"):
        judge(cfg, "loop_a", "4h", validate=StubValidate({}))


def test_a_lookahead_kill_belongs_to_its_family(cfg, plugins):
    add_plugin(plugins, "cheater", CHEATER)
    j = judge(cfg, "cheater", "4h", validate=StubValidate({}))
    assert j["decision"] == "kill" and j["lineage_id"] == "family:donchian_atr"


def test_rejudging_rewrites_a_missing_verdict(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE)})
    a = judge(cfg, "donchian_atr", "4h", validate=v)
    Path(a["verdict_path"]).unlink()
    b = judge(cfg, "donchian_atr", "4h", validate=v)
    assert b["existing"] is True and b["id"] == a["id"] and Path(b["verdict_path"]).exists()


def test_a_renamed_identical_copy_returns_the_original_judgement(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE)})
    a = judge(cfg, "donchian_atr", "4h", validate=v)
    add_plugin(plugins, "copy_root", (plugins / "donchian_atr.py").read_text())
    b = judge(cfg, "copy_root", "4h", validate=v)
    assert b["existing"] is True and b["id"] == a["id"] and b["strategy"] == "donchian_atr"
    assert b["verdict_path"] == a["verdict_path"] and Path(b["verdict_path"]).exists()


def test_cli_names_the_judgement_it_returns(cfg, plugins):
    judge(cfg, "donchian_atr", "4h", validate=StubValidate({"donchian_atr": ("noise", NOISE)}))
    add_plugin(plugins, "copy_root", (plugins / "donchian_atr.py").read_text())
    r = CliRunner().invoke(app, ["judge", "copy_root", "--tf", "4h", "--config", str(cfg.path)])
    assert r.exit_code == 0 and "donchian_atr @ 4h" in r.output and "already judged" in r.output


def test_a_judged_child_keeps_its_family_when_its_parent_line_is_removed(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE), "donchian_atr_wide": ("noise", NOISE),
                      "grandchild": ("noise", NOISE)})
    a = judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert a["lineage_id"] == "family:donchian_atr"
    f = plugins / "donchian_atr_wide.py"
    f.write_text(f.read_text().replace('PARENT = "donchian_atr"\n', "") + "# now a root\n")
    b = judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert b["id"] != a["id"] and b["parent_run_id"] is None and b["lineage_id"] == "family:donchian_atr"
    add_plugin(plugins, "grandchild", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "donchian_atr_wide"')
               + "# grandchild variant\n")
    assert judge(cfg, "grandchild", "4h", validate=v)["lineage_id"] == "family:donchian_atr"


def test_a_parent_that_fails_the_lookahead_check_is_refused(cfg, plugins):
    add_plugin(plugins, "cheater", CHEATER)
    add_plugin(plugins, "honest_child", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "cheater"'))
    v = StubValidate({"cheater": ("candidate", STRONG), "honest_child": ("candidate", STRONG)})
    with pytest.raises(JudgeRefused, match="parent cheater fails the lookahead check; fix or judge it first"):
        judge(cfg, "honest_child", "4h", validate=v)
    assert v.calls == [] and db.read_table(cfg, db.judgements).empty and db.read_table(cfg, db.runs).empty
