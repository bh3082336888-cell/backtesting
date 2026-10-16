import json
import math
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from perp_lab.cli import app
from perp_lab.judge import budget
from perp_lab.judge.exam import ExamResult
from perp_lab.judge.judge import VERDICT_KEYS, attempt_exam, exam_pending, judge, verdict_path
from perp_lab.store import db
from perp_lab.validate import walkforward as wfm
from tests.conftest import market_config
from tests.judge_fixtures import STRONG, WIDE_TEXT, StubValidate, add_plugin, plugins  # noqa: F401
from tests.test_store import cfg  # noqa: F401

HOLDOUT_DAYS = pd.date_range("2026-07-01", periods=200, freq="D", tz="UTC")


def stub_run(mu, n_trades=30, seed=11, fail=False):
    """Stand-in for run_exam: a scripted holdout portfolio and n_trades trades."""
    port = pd.Series(np.random.default_rng(seed).normal(mu, 0.01, len(HOLDOUT_DAYS)), index=HOLDOUT_DAYS)
    trades = pd.DataFrame({"symbol": "BTCUSDT", "signal_time": HOLDOUT_DAYS[:n_trades].as_unit("ms").asi8,
                           "pnl": 1.0, "r_multiple": 0.5, "bars_held": 3, "exit_reason": "stop",
                           "fees": 0.1, "funding": 0.0})

    def run(cfg, name, tf, symbols, params, token, end):
        if fail:
            raise RuntimeError("exam crashed")
        return ExamResult(HOLDOUT_DAYS[0], end, dict(params), {"BTCUSDT": port}, port, trades)
    return run


def end_after(days):
    return lambda cfg, symbols, tf: cfg.holdout_start + pd.Timedelta(days=days)


def judged(cfg, name, outcomes, run, days=200):
    return judge(cfg, name, "4h", validate=StubValidate(outcomes),
                 attempt=partial(attempt_exam, run=run, end_fn=end_after(days)))


ROOT_STRONG = {"donchian_atr": ("candidate", STRONG)}


def test_too_short_stays_pending_and_reads_nothing(cfg, plugins):
    budget.init_log(cfg)
    j = judged(cfg, "donchian_atr", ROOT_STRONG, stub_run(0.003), days=100)
    assert (j["holdout_status"], j["holdout_reason"]) == ("pending", "too_short")
    assert budget.reads_used(cfg) == 0


def test_a_passing_exam_is_recorded(cfg, plugins):
    budget.init_log(cfg)
    j = judged(cfg, "donchian_atr", ROOT_STRONG, stub_run(0.003))
    assert (j["holdout_status"], j["holdout_reason"]) == ("passed", "passed")
    h = db.run_row(cfg, j["holdout_run_id"])
    assert h["validation_mode"] == "holdout" and h["verdict"] == "passed" and h["lineage_id"] == j["lineage_id"]
    entries = budget.read_log(budget.log_path(cfg))
    assert [e["judgement_id"] for e in entries] == [j["id"]]


def test_the_holdout_run_has_a_report(cfg, plugins):
    budget.init_log(cfg)
    j = judged(cfg, "donchian_atr", ROOT_STRONG, stub_run(-0.003))
    text = (Path(db.run_row(cfg, j["holdout_run_id"])["run_dir"]) / "report.md").read_text()
    end = str((cfg.holdout_start + pd.Timedelta(days=200)).date())
    assert "donchian_atr" in text and "2026-07-01" in text and end in text
    assert "FAILED" in text and "holdout Sharpe" in text and '"atr_mult": 3.0' in text


def test_a_failing_exam_is_recorded_with_its_reasons(cfg, plugins):
    budget.init_log(cfg)
    j = judged(cfg, "donchian_atr", ROOT_STRONG, stub_run(-0.003))
    assert (j["holdout_status"], j["holdout_reason"]) == ("failed", "failed")
    h = db.run_row(cfg, j["holdout_run_id"])
    assert h["verdict"] == "failed" and h["fail_reasons"][0].startswith("holdout Sharpe")


def test_a_family_gets_one_exam(cfg, plugins):
    budget.init_log(cfg)
    outcomes = {"donchian_atr": ("candidate", STRONG), "donchian_atr_wide": ("candidate", STRONG + 0.001)}
    judged(cfg, "donchian_atr", outcomes, stub_run(0.003))
    j = judged(cfg, "donchian_atr_wide", outcomes, stub_run(0.003))
    assert j["decision"] == "promote" and (j["holdout_status"], j["holdout_reason"]) == ("blocked", "family_examined")


def test_the_global_cap_blocks_a_new_family(cfg, plugins):
    cfg.raw["judge"]["holdout_max_reads"] = 1
    budget.init_log(cfg)
    add_plugin(plugins, "other_root", WIDE_TEXT.replace('PARENT = "donchian_atr"\n', ""))
    judged(cfg, "donchian_atr", ROOT_STRONG, stub_run(0.003))
    j = judged(cfg, "other_root", {"other_root": ("candidate", STRONG + 0.0005)}, stub_run(0.003))
    assert (j["holdout_status"], j["holdout_reason"]) == ("blocked", "budget_spent")


def test_exam_crash_keeps_pending_and_retry_uses_no_new_read(cfg, plugins):
    budget.init_log(cfg)
    with pytest.raises(RuntimeError, match="exam crashed"):
        judged(cfg, "donchian_atr", ROOT_STRONG, stub_run(0.003, fail=True))
    j = db.pending_judgements(cfg)[0]
    assert j["holdout_reason"] == "exam_error" and budget.reads_used(cfg) == 1
    [done] = exam_pending(cfg, run=stub_run(0.003), end_fn=end_after(200))
    assert done["holdout_status"] == "passed" and budget.reads_used(cfg) == 1


def test_an_edited_plugin_waits_and_spends_no_read(cfg, plugins):
    budget.init_log(cfg)
    judged(cfg, "donchian_atr", ROOT_STRONG, stub_run(0.003), days=100)
    f = plugins / "donchian_atr.py"
    original = f.read_text()
    f.write_text(original + "\n# edited\n")
    [j] = exam_pending(cfg, run=stub_run(0.003), end_fn=end_after(200))
    assert (j["holdout_status"], j["holdout_reason"]) == ("pending", "plugin_changed")
    assert budget.reads_used(cfg) == 0
    f.write_text(original)
    [j] = exam_pending(cfg, run=stub_run(0.003), end_fn=end_after(200))
    assert j["holdout_status"] == "passed" and budget.reads_used(cfg) == 1


def test_a_crashed_exam_still_writes_its_verdict(cfg, plugins):
    budget.init_log(cfg)
    with pytest.raises(RuntimeError, match="exam crashed"):
        judged(cfg, "donchian_atr", ROOT_STRONG, stub_run(0.003, fail=True))
    j = db.pending_judgements(cfg)[0]
    p = verdict_path(cfg, j["strategy"], j["timeframe"], j["strategy_hash"])
    out = json.loads(p.read_text())
    assert (out["holdout_status"], out["holdout_reason"]) == ("pending", "exam_error")


def test_an_exam_that_cannot_start_still_writes_its_verdict(cfg, plugins):
    with pytest.raises(budget.BudgetLogError):          # the holdout log was never initialised
        judged(cfg, "donchian_atr", ROOT_STRONG, stub_run(0.003))
    [j] = db.pending_judgements(cfg)
    p = verdict_path(cfg, j["strategy"], j["timeframe"], j["strategy_hash"])
    out = json.loads(p.read_text())
    assert (out["holdout_status"], out["holdout_reason"]) == ("pending", "awaiting_exam")


def test_one_crashing_exam_does_not_block_the_others(cfg, plugins):
    budget.init_log(cfg)
    add_plugin(plugins, "other_root", WIDE_TEXT.replace('PARENT = "donchian_atr"\n', ""))
    outcomes = {"donchian_atr": ("candidate", STRONG), "other_root": ("candidate", STRONG + 0.0005)}
    for name in ("donchian_atr", "other_root"):
        judged(cfg, name, outcomes, stub_run(0.003), days=100)
    good = stub_run(0.003)

    def run(cfg, name, tf, symbols, params, token, end):
        if name == "donchian_atr":
            raise RuntimeError("exam crashed")
        return good(cfg, name, tf, symbols, params, token, end)
    a, b = exam_pending(cfg, run=run, end_fn=end_after(200))
    assert a["strategy"] == "donchian_atr" and "error" in a
    assert (a["holdout_status"], a["holdout_reason"]) == ("pending", "exam_error")
    assert b["strategy"] == "other_root" and b["holdout_status"] == "passed" and "error" not in b


def test_a_judgement_that_already_has_a_result_is_never_re_examined(cfg, plugins):
    budget.init_log(cfg)

    def boom(*a, **k):
        raise AssertionError("must not run")
    outcomes = {"donchian_atr": ("candidate", STRONG), "donchian_atr_wide": ("candidate", STRONG + 0.001)}
    done = judged(cfg, "donchian_atr", outcomes, stub_run(0.003))
    blocked = judged(cfg, "donchian_atr_wide", outcomes, stub_run(0.003))
    assert blocked["holdout_status"] == "blocked" and budget.reads_used(cfg) == 1
    again = attempt_exam(cfg, done["id"], run=boom, end_fn=end_after(200))
    assert again["holdout_status"] == "passed" and again["holdout_run_id"] == done["holdout_run_id"]
    again = attempt_exam(cfg, blocked["id"], run=boom, end_fn=end_after(200))
    assert (again["holdout_status"], again["holdout_reason"]) == ("blocked", "family_examined")
    assert budget.reads_used(cfg) == 1


FAMILY = {"donchian_atr": ("candidate", STRONG), "donchian_atr_wide": ("candidate", STRONG + 0.001)}


def test_a_promoted_child_supersedes_its_pending_parent(cfg, plugins):
    budget.init_log(cfg)
    root = judged(cfg, "donchian_atr", FAMILY, stub_run(0.003), days=100)
    child = judged(cfg, "donchian_atr_wide", FAMILY, stub_run(0.003), days=100)
    assert child["decision"] == "promote" and child["holdout_status"] == "pending"
    r = db.get_judgement_by_id(cfg, root["id"])
    assert (r["holdout_status"], r["holdout_reason"]) == ("blocked", "superseded")
    out = json.loads(Path(root["verdict_path"]).read_text())
    assert (out["holdout_status"], out["holdout_reason"]) == ("blocked", "superseded")
    [done] = exam_pending(cfg, run=stub_run(0.003), end_fn=end_after(200))
    assert done["id"] == child["id"] and done["holdout_status"] == "passed" and budget.reads_used(cfg) == 1


def test_a_killed_child_leaves_its_pending_parent_queued(cfg, plugins):
    budget.init_log(cfg)
    outcomes = {**FAMILY, "donchian_atr_wide": ("candidate", STRONG - 0.001)}
    root = judged(cfg, "donchian_atr", outcomes, stub_run(0.003), days=100)
    child = judged(cfg, "donchian_atr_wide", outcomes, stub_run(0.003), days=100)
    assert child["decision"] == "kill"
    r = db.get_judgement_by_id(cfg, root["id"])
    assert (r["holdout_status"], r["holdout_reason"]) == ("pending", "too_short")


def _numbers(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _numbers(v)
    elif isinstance(o, list):
        for v in o:
            yield from _numbers(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield float(o)


def test_the_verdict_never_carries_holdout_numbers(cfg, plugins):
    budget.init_log(cfg)
    j = judged(cfg, "donchian_atr", ROOT_STRONG, stub_run(0.003))
    text = Path(j["verdict_path"]).read_text()
    out = json.loads(text)
    h = db.run_row(cfg, j["holdout_run_id"])
    secret = [x for x in (h["sharpe"], h["sortino"], h["total_return"], h["max_dd"], h["cagr"]) if x is not None]
    assert set(out) == VERDICT_KEYS and out["holdout_status"] == "passed"
    assert not any(math.isclose(x, s, rel_tol=1e-9) for x in _numbers(out) for s in secret)
    assert j["holdout_run_id"] not in text


def test_cli_exam_with_nothing_pending(cfg):
    budget.init_log(cfg)
    r = CliRunner().invoke(app, ["exam", "--config", str(cfg.path)])
    assert r.exit_code == 0 and "no pending exams" in r.output


@pytest.fixture
def fresh_wf_caches():
    wfm._cfg.cache_clear()
    wfm._bars.cache_clear()
    yield
    wfm._cfg.cache_clear()
    wfm._bars.cache_clear()


def test_end_to_end_real_exam_on_a_synthetic_market(tmp_path, plugins, fresh_wf_caches):
    cfg = market_config(tmp_path, judge={"min_holdout_days": 30})
    budget.init_log(cfg)
    j = judge(cfg, "donchian_atr", "4h", validate=StubValidate(ROOT_STRONG), attempt=attempt_exam)
    assert j["holdout_status"] in {"passed", "failed"}          # a real exam ran on 92 days
    h = db.run_row(cfg, j["holdout_run_id"])
    assert (h["date_start"], h["date_end"]) == ("2026-07-01", "2026-10-01")
    assert pd.read_parquet(Path(h["run_dir"]) / "holdout_daily.parquet").shape[0] == 92


def test_end_to_end_real_walk_forward_on_a_synthetic_market(tmp_path, plugins, fresh_wf_caches):
    cfg = market_config(tmp_path, judge={"min_holdout_days": 30},
                        validation={"random_seeds": 5, "mc_iterations": 200})
    budget.init_log(cfg)
    j = judge(cfg, "donchian_atr", "4h", workers=1, attempt=attempt_exam)
    child = db.run_row(cfg, j["child_run_id"])
    assert child["validation_mode"] == "walkforward" and len(db.read_table(cfg, db.run_folds)) == 6
    assert j["decision"] in {"promote", "kill"} and j["n_trials"] == 1
    assert j["holdout_status"] == "n/a" if j["decision"] == "kill" else j["holdout_status"] in {"passed", "failed"}
    assert set(json.loads(Path(j["verdict_path"]).read_text())) == VERDICT_KEYS


def test_a_parent_whose_exam_crashed_is_not_superseded(cfg, plugins):
    # the crashed parent already holds the family's read; superseding it would spend that read
    # on nothing, so it stays pending (exam_error) and retries on the same read
    budget.init_log(cfg)
    with pytest.raises(RuntimeError, match="exam crashed"):
        judged(cfg, "donchian_atr", FAMILY, stub_run(0.003, fail=True))
    root = db.pending_judgements(cfg)[0]
    child = judged(cfg, "donchian_atr_wide", FAMILY, stub_run(0.003))
    assert child["decision"] == "promote"
    r = db.get_judgement_by_id(cfg, root["id"])
    assert (r["holdout_status"], r["holdout_reason"]) == ("pending", "exam_error")
    [done] = exam_pending(cfg, run=stub_run(0.003), end_fn=end_after(200))
    assert done["id"] == root["id"] and done["holdout_status"] == "passed" and budget.reads_used(cfg) == 1


def test_a_parent_judged_in_another_family_is_not_superseded(cfg, plugins):
    # "kid" was first judged as a root (family:kid); a later version names rootb as its PARENT
    # but keeps family:kid, so promoting it must not cancel family:rootb's queued exam
    budget.init_log(cfg)
    root_text = WIDE_TEXT.replace('PARENT = "donchian_atr"\n', "")
    add_plugin(plugins, "rootb", root_text + "# rootb\n")
    add_plugin(plugins, "kid", root_text + "# kid v1\n")
    outcomes = {"rootb": ("candidate", STRONG), "kid": ("candidate", STRONG + 0.0005)}
    rb = judged(cfg, "rootb", outcomes, stub_run(0.003), days=100)
    judged(cfg, "kid", outcomes, stub_run(0.003), days=100)
    add_plugin(plugins, "kid", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "rootb"') + "# kid v2\n")
    kid2 = judged(cfg, "kid", outcomes, stub_run(0.003), days=100)
    assert kid2["decision"] == "promote" and kid2["lineage_id"] == "family:kid"
    r = db.get_judgement_by_id(cfg, rb["id"])
    assert r["lineage_id"] == "family:rootb"
    assert (r["holdout_status"], r["holdout_reason"]) == ("pending", "too_short")


def test_a_failed_supersede_still_writes_the_childs_verdict(cfg, plugins, monkeypatch):
    import perp_lab.judge.judge as jm
    budget.init_log(cfg)
    judged(cfg, "donchian_atr", FAMILY, stub_run(0.003), days=100)

    def boom(*a, **kw):
        raise RuntimeError("supersede failed")

    monkeypatch.setattr(jm, "supersede", boom)
    with pytest.raises(RuntimeError, match="supersede failed"):
        judged(cfg, "donchian_atr_wide", FAMILY, stub_run(0.003), days=100)
    j = [x for x in db.pending_judgements(cfg) if x["strategy"] == "donchian_atr_wide"][0]
    assert verdict_path(cfg, j["strategy"], j["timeframe"], j["strategy_hash"]).exists()


def test_a_plugin_edited_mid_walk_forward_is_not_recorded(tmp_path, plugins, fresh_wf_caches, monkeypatch):
    # validate_strategy hashes the plugin at the start and the end; a run that mixed versions is
    # refused before anything is recorded, so a later `lab judge` can't reuse it as a baseline
    from perp_lab.validate import pipeline
    cfg = market_config(tmp_path, validation={"random_seeds": 5, "mc_iterations": 200})
    real, f = pipeline.walkforward, plugins / "donchian_atr.py"

    def editing(*a, **kw):
        out = real(*a, **kw)
        f.write_text(f.read_text() + "\n# edited mid-run\n")
        return out

    monkeypatch.setattr(pipeline, "walkforward", editing)
    with pytest.raises(RuntimeError, match="donchian_atr changed during its walk-forward"):
        pipeline.validate_strategy(cfg, "donchian_atr", "4h", workers=1)
    assert db.read_table(cfg, db.runs).empty


class Interrupted(BaseException):
    """Stands in for Ctrl-C / a killed process: not an Exception, so no handler records exam_error."""


def test_a_parent_holding_a_read_is_never_superseded_even_after_an_interrupt(cfg, plugins):
    # the audit log, not the reason code, says whether the parent holds the family's read
    budget.init_log(cfg)

    def interrupted(*a, **kw):
        raise Interrupted

    with pytest.raises(Interrupted):
        judged(cfg, "donchian_atr", FAMILY, interrupted)
    root = db.pending_judgements(cfg)[0]
    assert root["holdout_reason"] == "awaiting_exam" and budget.reads_used(cfg) == 1
    child = judged(cfg, "donchian_atr_wide", FAMILY, stub_run(0.003))
    assert child["decision"] == "promote"
    r = db.get_judgement_by_id(cfg, root["id"])
    assert r["holdout_status"] == "pending"
    done = {j["id"]: j for j in exam_pending(cfg, run=stub_run(0.003), end_fn=end_after(200))}
    assert done[root["id"]]["holdout_status"] == "passed" and budget.reads_used(cfg) == 1


def test_supersede_does_nothing_when_the_read_record_cannot_be_read(cfg, plugins):
    # no holdout log: whether the parent holds a read is unknown, so leave it queued
    root = judge(cfg, "donchian_atr", "4h", validate=StubValidate(FAMILY))
    child = judge(cfg, "donchian_atr_wide", "4h", validate=StubValidate(FAMILY))
    assert child["decision"] == "promote"
    r = db.get_judgement_by_id(cfg, root["id"])
    assert (r["holdout_status"], r["holdout_reason"]) == ("pending", "awaiting_exam")


def test_lookahead_after_the_judged_span_blocks_the_exam_before_any_read(cfg, plugins, monkeypatch):
    # judged on 2026-10-16 the dated check covers to mid-November; the exam window later runs to
    # 2027-03-08, so the check is repeated over the actual window just before the read is spent
    import perp_lab.judge.judge as jm
    from tests.test_judge import date_peek
    monkeypatch.setattr(jm, "_now", lambda: pd.Timestamp("2026-10-16", tz="UTC"), raising=False)
    budget.init_log(cfg)
    add_plugin(plugins, "latepeek", date_peek("2027-02-01", parent=False))
    j = judged(cfg, "latepeek", {"latepeek": ("candidate", STRONG)}, stub_run(0.003), days=100)
    assert j["decision"] == "promote" and j["holdout_reason"] == "too_short"
    [done] = exam_pending(cfg, run=stub_run(0.003), end_fn=end_after(250))
    assert (done["holdout_status"], done["holdout_reason"]) == ("blocked", "lookahead")
    assert budget.reads_used(cfg) == 0


def test_lookahead_in_the_last_days_of_the_exam_window_blocks_the_exam(cfg, plugins, monkeypatch):
    import perp_lab.judge.judge as jm
    from tests.test_judge import date_peek
    monkeypatch.setattr(jm, "_now", lambda: pd.Timestamp("2026-10-16", tz="UTC"))
    budget.init_log(cfg)
    add_plugin(plugins, "tailpeek", date_peek("2026-12-20", parent=False))
    judged(cfg, "tailpeek", {"tailpeek": ("candidate", STRONG)}, stub_run(0.003), days=100)
    [done] = exam_pending(cfg, run=stub_run(0.003), end_fn=end_after(180))   # window ends 2026-12-28
    assert (done["holdout_status"], done["holdout_reason"]) == ("blocked", "lookahead")
    assert budget.reads_used(cfg) == 0


def test_an_honest_seasonal_plugin_is_not_blocked_at_exam_time(cfg, plugins, monkeypatch):
    import perp_lab.judge.judge as jm
    from tests.test_judge import SEASONAL
    monkeypatch.setattr(jm, "_now", lambda: pd.Timestamp("2026-10-16", tz="UTC"))
    budget.init_log(cfg)
    add_plugin(plugins, "seasonal", SEASONAL)
    judged(cfg, "seasonal", {"seasonal": ("candidate", STRONG)}, stub_run(0.003), days=100)
    [done] = exam_pending(cfg, run=stub_run(0.003), end_fn=end_after(100))   # window May-Oct only
    assert (done["holdout_status"], done["holdout_reason"]) == ("pending", "too_short")
