import json
import math
import threading
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from perp_lab.cli import app
from perp_lab.judge import budget
from perp_lab.judge.exam import ExamResult
from perp_lab.judge.budget import WithdrawRefused
from perp_lab.judge.judge import VERDICT_KEYS, attempt_exam, exam_pending, judge, verdict_path, withdraw
from perp_lab.sandbox import client
from perp_lab.sandbox.errors import PluginError, SandboxUnavailable
from perp_lab.store import db
from perp_lab.validate import walkforward as wfm
from tests.conftest import market_config
from tests.judge_fixtures import NOISE, STRONG, WIDE_TEXT, StubValidate, add_plugin, plugins  # noqa: F401
from tests.test_store import cfg  # noqa: F401

HOLDOUT_DAYS = pd.date_range("2026-07-01", periods=200, freq="D", tz="UTC")


def stub_run(mu, n_trades=30, seed=11, fail=False):
    """Stand-in for run_exam: a scripted holdout portfolio and n_trades trades."""
    port = pd.Series(np.random.default_rng(seed).normal(mu, 0.01, len(HOLDOUT_DAYS)), index=HOLDOUT_DAYS)
    trades = pd.DataFrame({"symbol": "BTCUSDT", "signal_time": HOLDOUT_DAYS[:n_trades].as_unit("ms").asi8,
                           "pnl": 1.0, "r_multiple": 0.5, "bars_held": 3, "exit_reason": "stop",
                           "fees": 0.1, "funding": 0.0})

    def run(cfg, src, tf, symbols, params, token, end, stderr_dir=None):
        if fail:
            raise RuntimeError("exam crashed")
        return ExamResult(HOLDOUT_DAYS[0], end, dict(params), {"BTCUSDT": port}, port, trades)
    return run


def end_after(days):
    return lambda cfg, symbols, tf: cfg.holdout_start + pd.Timedelta(days=days)


def no_cost(cfg, src, tf, symbols, params, end):
    """Stand-in for the cost probe: an exam that costs nothing."""
    return 0.0


def no_gaps(cfg, symbols, tf, end):
    """Stand-in for the exam window's data check: exchange files throughout."""
    return []


def boom(*a, **kw):
    raise AssertionError("must not run")


def judged(cfg, name, outcomes, run, days=200):
    """`lab judge`, then (as `lab exam` would) the exam of a new promotion."""
    j = judge(cfg, name, "4h", validate=StubValidate(outcomes))
    if j["decision"] != "promote" or j.get("existing"):
        return j
    return attempt_exam(cfg, j["id"], run=run, data_fn=no_gaps, end_fn=end_after(days), cost_fn=no_cost)


def exams(cfg, run, days=200):
    """`lab exam`, with a stand-in exam and window end."""
    return exam_pending(cfg, run=run, data_fn=no_gaps, end_fn=end_after(days), cost_fn=no_cost)


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
    [done] = exams(cfg, stub_run(0.003))
    assert done["holdout_status"] == "passed" and budget.reads_used(cfg) == 1


class Interrupted(BaseException):
    """Stands in for Ctrl-C: not an Exception."""


def test_an_interrupted_exam_after_the_mint_is_left_as_exam_error(cfg, plugins):
    # Ctrl-C after the read was minted: the judgement must not keep its stale awaiting_exam reason
    # (its verdict would say no exam had been tried, though its read is spent).
    budget.init_log(cfg)
    j = promoted(cfg)

    def interrupted(*a, **kw):
        raise Interrupted
    with pytest.raises(Interrupted):
        attempt_exam(cfg, j["id"], run=interrupted, data_fn=no_gaps, end_fn=end_after(200), cost_fn=no_cost)
    r = db.get_judgement_by_id(cfg, j["id"])
    assert (r["holdout_status"], r["holdout_reason"]) == ("pending", "exam_error") and budget.reads_used(cfg) == 1
    out = json.loads(verdict_path(cfg, r["strategy"], r["timeframe"], r["strategy_hash"]).read_text())
    assert (out["holdout_status"], out["holdout_reason"]) == ("pending", "exam_error")


def test_an_edited_plugin_waits_and_spends_no_read(cfg, plugins):
    budget.init_log(cfg)
    judged(cfg, "donchian_atr", ROOT_STRONG, stub_run(0.003), days=100)
    f = plugins / "donchian_atr.py"
    original = f.read_text()
    f.write_text(original + "\n# edited\n")
    [j] = exams(cfg, stub_run(0.003))
    assert (j["holdout_status"], j["holdout_reason"]) == ("pending", "plugin_changed")
    assert budget.reads_used(cfg) == 0
    f.write_text(original)
    [j] = exams(cfg, stub_run(0.003))
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

    def run(cfg, src, tf, symbols, params, token, end, stderr_dir=None):
        if src.name == "donchian_atr":
            raise RuntimeError("exam crashed")
        return good(cfg, src, tf, symbols, params, token, end, stderr_dir=stderr_dir)
    a, b = exams(cfg, run)
    assert a["strategy"] == "donchian_atr" and "error" in a
    assert (a["holdout_status"], a["holdout_reason"]) == ("pending", "exam_error")
    assert b["strategy"] == "other_root" and b["holdout_status"] == "passed" and "error" not in b


def test_a_judgement_that_already_has_a_result_is_never_re_examined(cfg, plugins):
    budget.init_log(cfg)
    outcomes = {"donchian_atr": ("candidate", STRONG), "donchian_atr_wide": ("candidate", STRONG + 0.001)}
    done = judged(cfg, "donchian_atr", outcomes, stub_run(0.003))
    blocked = judged(cfg, "donchian_atr_wide", outcomes, stub_run(0.003))
    assert blocked["holdout_status"] == "blocked" and budget.reads_used(cfg) == 1
    again = attempt_exam(cfg, done["id"], run=boom, data_fn=no_gaps, end_fn=end_after(200))
    assert again["holdout_status"] == "passed" and again["holdout_run_id"] == done["holdout_run_id"]
    again = attempt_exam(cfg, blocked["id"], run=boom, data_fn=no_gaps, end_fn=end_after(200))
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
    [done] = exams(cfg, stub_run(0.003))
    assert done["id"] == child["id"] and done["holdout_status"] == "passed" and budget.reads_used(cfg) == 1


def test_supersede_leaves_a_parent_that_is_being_examined(cfg, plugins):
    # Its exam lock is held: that exam may be about to mint the family's read, and superseding the
    # parent under it would spend the read on a judgement whose verdict then flips.
    budget.init_log(cfg)
    root = judged(cfg, "donchian_atr", FAMILY, stub_run(0.003), days=100)
    with budget.exam_lock(cfg, root["id"]) as mine:
        assert mine
        child = judged(cfg, "donchian_atr_wide", FAMILY, stub_run(0.003), days=100)
    assert child["decision"] == "promote"
    r = db.get_judgement_by_id(cfg, root["id"])
    assert (r["holdout_status"], r["holdout_reason"]) == ("pending", "too_short")


def test_a_judgement_superseded_before_the_mint_is_not_minted(cfg, plugins):
    # Between step 1 and the mint (the dated check and the cost probe take minutes) the judgement
    # stops being pending: the attempt stops there and spends no read.
    budget.init_log(cfg)
    j = judge(cfg, "donchian_atr", "4h", validate=StubValidate(ROOT_STRONG))

    def superseded_meanwhile(cfg, src, tf, symbols, params, end):
        db.update_holdout(cfg, j["id"], "blocked", "superseded")
        return 0.0
    r = attempt_exam(cfg, j["id"], run=boom, data_fn=no_gaps, end_fn=end_after(200), cost_fn=superseded_meanwhile)
    assert (r["holdout_status"], r["holdout_reason"]) == ("blocked", "superseded")
    assert budget.reads_used(cfg) == 0


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
    j = judge(cfg, "donchian_atr", "4h", validate=StubValidate(ROOT_STRONG))
    j = attempt_exam(cfg, j["id"])
    assert j["holdout_reason"] in {"passed", "failed"}          # a real exam ran on 92 days
    h = db.run_row(cfg, j["holdout_run_id"])
    assert (h["date_start"], h["date_end"]) == ("2026-07-01", "2026-10-01")
    assert pd.read_parquet(Path(h["run_dir"]) / "holdout_daily.parquet").shape[0] == 92


def test_end_to_end_real_walk_forward_on_a_synthetic_market(tmp_path, plugins, fresh_wf_caches):
    cfg = market_config(tmp_path, judge={"min_holdout_days": 30},
                        validation={"random_seeds": 5, "mc_iterations": 200})
    budget.init_log(cfg)
    j = judge(cfg, "donchian_atr", "4h", workers=1)
    if j["decision"] == "promote":
        j = attempt_exam(cfg, j["id"])
    child = db.run_row(cfg, j["child_run_id"])
    assert child["validation_mode"] == "walkforward" and len(db.read_table(cfg, db.run_folds)) == 6
    assert j["decision"] in {"promote", "kill"} and j["n_trials"] == 1
    assert j["holdout_status"] == "n/a" if j["decision"] == "kill" else j["holdout_reason"] in {"passed", "failed"}
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
    [done] = exams(cfg, stub_run(0.003))
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


def test_a_parent_holding_a_read_is_never_superseded_even_after_a_kill(cfg, plugins):
    # the audit log, not the reason code, says whether the parent holds the family's read: an exam
    # killed (SIGKILL) just after its mint records nothing, so its reason is still awaiting_exam
    budget.init_log(cfg)
    root = judge(cfg, "donchian_atr", "4h", validate=StubValidate(FAMILY))
    budget.mint(cfg, root["lineage_id"], root["id"], 200, DAY_200)
    root = db.get_judgement_by_id(cfg, root["id"])
    assert root["holdout_reason"] == "awaiting_exam" and budget.reads_used(cfg) == 1
    child = judged(cfg, "donchian_atr_wide", FAMILY, stub_run(0.003))
    assert child["decision"] == "promote"
    r = db.get_judgement_by_id(cfg, root["id"])
    assert r["holdout_status"] == "pending"
    done = {j["id"]: j for j in exams(cfg, stub_run(0.003))}
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
    [done] = exams(cfg, stub_run(0.003), 250)
    assert (done["holdout_status"], done["holdout_reason"]) == ("blocked", "lookahead")
    assert budget.reads_used(cfg) == 0


def test_lookahead_in_the_last_days_of_the_exam_window_blocks_the_exam(cfg, plugins, monkeypatch):
    import perp_lab.judge.judge as jm
    from tests.test_judge import date_peek
    monkeypatch.setattr(jm, "_now", lambda: pd.Timestamp("2026-10-16", tz="UTC"))
    budget.init_log(cfg)
    add_plugin(plugins, "tailpeek", date_peek("2026-12-20", parent=False))
    judged(cfg, "tailpeek", {"tailpeek": ("candidate", STRONG)}, stub_run(0.003), days=100)
    [done] = exams(cfg, stub_run(0.003), 180)   # window ends 2026-12-28
    assert (done["holdout_status"], done["holdout_reason"]) == ("blocked", "lookahead")
    assert budget.reads_used(cfg) == 0


def test_an_honest_seasonal_plugin_is_not_blocked_at_exam_time(cfg, plugins, monkeypatch):
    import perp_lab.judge.judge as jm
    from tests.test_judge import SEASONAL
    monkeypatch.setattr(jm, "_now", lambda: pd.Timestamp("2026-10-16", tz="UTC"))
    budget.init_log(cfg)
    add_plugin(plugins, "seasonal", SEASONAL)
    judged(cfg, "seasonal", {"seasonal": ("candidate", STRONG)}, stub_run(0.003), days=100)
    cfg.raw["judge"]["min_holdout_days"] = 100   # long enough, so the dated check runs over it
    [done] = exams(cfg, stub_run(0.003), 100)   # window May-Oct only
    assert (done["holdout_status"], done["holdout_reason"]) == ("passed", "passed")


# ---------------------------------------------------------------- only `lab exam` runs exams (3b-1)

DAY_200 = pd.Timestamp("2026-07-01", tz="UTC") + pd.Timedelta(days=200)


def promoted(cfg, name="donchian_atr"):
    """`lab judge` of a plugin that promotes: a pending judgement awaiting its exam."""
    j = judge(cfg, name, "4h", validate=StubValidate({name: ("candidate", STRONG)}))
    assert j["decision"] == "promote" and (j["holdout_status"], j["holdout_reason"]) == ("pending", "awaiting_exam")
    return j


def crash(cfg, jid, days=200):
    """One attempt that mints its read and then crashes."""
    with pytest.raises(RuntimeError, match="exam crashed"):
        attempt_exam(cfg, jid, run=stub_run(0.003, fail=True), data_fn=no_gaps, end_fn=end_after(days),
                     cost_fn=no_cost)


def test_lab_judge_never_runs_an_exam(cfg, plugins, monkeypatch):
    import perp_lab.judge.judge as jm
    budget.init_log(cfg)
    monkeypatch.setattr(jm, "attempt_exam", boom)
    monkeypatch.setattr(jm, "exam_pending", boom)
    monkeypatch.setattr(jm, "judge", partial(jm.judge, validate=StubValidate(ROOT_STRONG)))
    r = CliRunner().invoke(app, ["judge", "donchian_atr", "--config", str(cfg.path)])
    assert r.exit_code == 0, r.output
    assert "holdout: pending (awaiting_exam)" in r.output
    [j] = db.pending_judgements(cfg)
    assert (j["decision"], j["holdout_status"], j["holdout_reason"]) == ("promote", "pending", "awaiting_exam")
    assert budget.reads_used(cfg) == 0


def test_a_retry_uses_the_window_stored_at_the_mint(cfg, plugins):
    budget.init_log(cfg)
    j = promoted(cfg)
    crash(cfg, j["id"], days=200)
    ends, run = [], stub_run(0.003)

    def spy(cfg, src, tf, symbols, params, token, end, stderr_dir=None):
        ends.append(end)
        return run(cfg, src, tf, symbols, params, token, end, stderr_dir=stderr_dir)
    done = attempt_exam(cfg, j["id"], run=spy, data_fn=no_gaps, end_fn=end_after(260),   # more data since
                        cost_fn=no_cost)
    assert ends == [DAY_200] and done["holdout_status"] == "passed"
    assert db.run_row(cfg, done["holdout_run_id"])["date_end"] == str(DAY_200.date())
    entries = budget.read_log(budget.log_path(cfg))
    assert [e["window_end"] for e in entries] == [DAY_200.isoformat()] * 2 and budget.reads_used(cfg) == 1


def test_a_retry_skips_the_cost_probe_and_the_dated_check(cfg, plugins, monkeypatch):
    import perp_lab.judge.judge as jm
    budget.init_log(cfg)
    j = promoted(cfg)
    crash(cfg, j["id"])
    monkeypatch.setattr(jm, "dated_lookahead_failures", boom)
    done = attempt_exam(cfg, j["id"], run=stub_run(0.003), data_fn=no_gaps, end_fn=boom, cost_fn=boom)
    assert done["holdout_status"] == "passed" and budget.reads_used(cfg) == 1


def test_a_read_fails_after_its_last_attempt(cfg, plugins):
    budget.init_log(cfg)
    j = promoted(cfg)
    assert cfg["judge"]["exam_max_attempts"] == 3
    for _ in range(3):
        crash(cfg, j["id"])
    r = db.get_judgement_by_id(cfg, j["id"])
    assert (r["holdout_status"], r["holdout_reason"]) == ("pending", "exam_error")
    done = attempt_exam(cfg, j["id"], run=boom, data_fn=no_gaps, end_fn=end_after(200), cost_fn=no_cost)
    assert (done["holdout_status"], done["holdout_reason"]) == ("failed", "exam_error")
    assert done["holdout_run_id"] is None and budget.attempts(cfg, j["id"]) == 3 and budget.reads_used(cfg) == 1
    out = json.loads(Path(done["verdict_path"]).read_text())
    assert (out["holdout_status"], out["holdout_reason"]) == ("failed", "exam_error")
    assert exams(cfg, boom) == []


def test_a_slow_plugin_is_blocked_before_any_read(cfg, plugins):
    budget.init_log(cfg)
    j = promoted(cfg)
    probed = []

    def slow(cfg, src, tf, symbols, params, end):
        probed.append((src.name, tf, params, end))
        return 7.0                              # CPU hours per symbol; exam_cpu_hours is 6
    done = attempt_exam(cfg, j["id"], run=boom, data_fn=no_gaps, end_fn=end_after(200), cost_fn=slow)
    assert (done["holdout_status"], done["holdout_reason"]) == ("blocked", "too_slow")
    assert budget.reads_used(cfg) == 0
    assert probed == [("donchian_atr", "4h", {"n": 40, "atr_n": 14, "atr_mult": 3.0}, DAY_200)]


CCXT_MONTH = "ETHUSDT 4h 2026-12: stitched from ccxt (no taker, trade-count or quote-volume columns)"


def test_candles_not_from_exchange_files_keep_the_exam_pending_without_a_read(cfg, plugins, monkeypatch):
    # A month `lab tail` stitched has no taker columns: a plugin reading them would trade nothing in
    # the exam and spend its family's one read on a data gap. Checked before any worker starts.
    import perp_lab.judge.judge as jm
    budget.init_log(cfg)
    j = promoted(cfg)
    monkeypatch.setattr(jm, "dated_lookahead_failures", boom)
    seen = []

    def ccxt_month(cfg, symbols, tf, end):
        seen.append((symbols, tf, end))
        return [CCXT_MONTH]
    done = attempt_exam(cfg, j["id"], run=boom, data_fn=ccxt_month, end_fn=end_after(200), cost_fn=boom)
    assert (done["holdout_status"], done["holdout_reason"]) == ("pending", "incomplete_data")
    assert seen == [(cfg.symbols, "4h", DAY_200)]
    assert budget.read_log(budget.log_path(cfg)) == [] and budget.reads_used(cfg) == 0
    assert done["detail"][0] == CCXT_MONTH and "lab backfill" in done["detail"][-1]
    text = Path(done["verdict_path"]).read_text()
    assert (json.loads(text)["holdout_status"], json.loads(text)["holdout_reason"]) == ("pending", "incomplete_data")
    assert "ccxt" not in text                               # the verdict carries the code only
    monkeypatch.setattr(jm, "dated_lookahead_failures", lambda *a, **k: [])
    done = attempt_exam(cfg, j["id"], run=stub_run(0.003), data_fn=no_gaps, end_fn=end_after(200), cost_fn=no_cost)
    assert done["holdout_status"] == "passed" and budget.reads_used(cfg) == 1      # once the data is fixed


def test_a_retry_waits_for_exchange_data_without_using_an_attempt(cfg, plugins):
    # The read is spent and its window fixed; a retry over ccxt candles would still fail for want of
    # data, so it waits too, and waiting uses none of the read's attempts.
    budget.init_log(cfg)
    j = promoted(cfg)
    crash(cfg, j["id"], days=200)
    seen = []

    def ccxt_month(cfg, symbols, tf, end):
        seen.append(end)
        return [CCXT_MONTH]
    done = attempt_exam(cfg, j["id"], run=boom, data_fn=ccxt_month, end_fn=boom, cost_fn=boom)
    assert (done["holdout_status"], done["holdout_reason"]) == ("pending", "incomplete_data")
    assert seen == [DAY_200] and budget.attempts(cfg, j["id"]) == 1 and budget.reads_used(cfg) == 1
    done = attempt_exam(cfg, j["id"], run=stub_run(0.003), data_fn=no_gaps, end_fn=boom, cost_fn=boom)
    assert done["holdout_status"] == "passed" and budget.attempts(cfg, j["id"]) == 2 and budget.reads_used(cfg) == 1


def test_lab_exam_prints_why_the_data_is_refused(cfg, monkeypatch):
    import perp_lab.judge.judge as jm
    waiting = {"strategy": "s", "timeframe": "1h", "holdout_status": "pending", "holdout_reason": "incomplete_data",
               "detail": [CCXT_MONTH, "run `lab backfill` ..."]}
    monkeypatch.setattr(jm, "exam_pending", lambda cfg: [waiting])
    r = CliRunner().invoke(app, ["exam", "--config", str(cfg.path)])
    assert r.exit_code == 0 and "s @ 1h: pending (incomplete_data)" in r.output
    assert f"  {CCXT_MONTH}" in r.output and "lab backfill" in r.output


def test_a_real_market_with_a_ccxt_month_in_the_window_mints_nothing(tmp_path, plugins, monkeypatch):
    import perp_lab.judge.judge as jm
    cfg = market_config(tmp_path, judge={"min_holdout_days": 30})
    budget.init_log(cfg)
    p = cfg.data_dir / "ETHUSDT" / "4h" / "_manifest.json"
    man = json.loads(p.read_text())
    man["2026-08"] = {"complete": False, "source": "ccxt", "rows": 0}
    p.write_text(json.dumps(man))
    j = judge(cfg, "donchian_atr", "4h", validate=StubValidate(ROOT_STRONG))
    monkeypatch.setattr(jm, "dated_lookahead_failures", boom)       # the data check comes first
    done = attempt_exam(cfg, j["id"], run=boom, cost_fn=boom)        # the real window end and data check
    assert (done["holdout_status"], done["holdout_reason"]) == ("pending", "incomplete_data")
    assert done["detail"][0].startswith("ETHUSDT 4h 2026-08: stitched from ccxt") and budget.reads_used(cfg) == 0


def test_a_plugin_error_fails_the_exam_without_its_message_in_the_verdict(cfg, plugins):
    budget.init_log(cfg)
    j = promoted(cfg)
    dirs = []

    def raises(cfg, src, tf, symbols, params, token, end, stderr_dir=None):
        dirs.append(stderr_dir)
        raise PluginError("exception", "ValueError", "close=123.45")
    done = attempt_exam(cfg, j["id"], run=raises, data_fn=no_gaps, end_fn=end_after(200), cost_fn=no_cost)
    assert (done["holdout_status"], done["holdout_reason"]) == ("failed", "plugin_error")
    assert budget.reads_used(cfg) == 1
    text = Path(done["verdict_path"]).read_text()
    assert "ValueError" not in text and "123.45" not in text and done["holdout_run_id"] not in text
    h = db.run_row(cfg, done["holdout_run_id"])
    assert h["validation_mode"] == "holdout" and h["verdict"] == "failed" and h["lineage_id"] == j["lineage_id"]
    assert h["fail_reasons"] == ["plugin_error: exception: ValueError: close=123.45"]
    stderr_dir = cfg.runs_dir / "_exam_stderr" / f"judgement-{j['id']}"
    assert dirs == [stderr_dir]
    report = (Path(h["run_dir"]) / "report.md").read_text()
    assert "ValueError" in report and "123.45" in report and str(stderr_dir) in report
    assert "2026-07-01" in report and str(DAY_200.date()) in report and '"atr_mult": 3.0' in report
    assert exams(cfg, boom) == []               # the read is spent: never examined again


def test_no_mint_when_the_sandbox_is_unavailable(cfg, plugins, monkeypatch):
    budget.init_log(cfg)
    j = promoted(cfg)

    def unavailable(cfg):
        raise SandboxUnavailable("Landlock ABI 3 < 6")
    monkeypatch.setattr(client, "check", unavailable)
    with pytest.raises(SandboxUnavailable):
        attempt_exam(cfg, j["id"], run=boom, data_fn=no_gaps, end_fn=end_after(200), cost_fn=no_cost)
    assert budget.reads_used(cfg) == 0
    r = db.get_judgement_by_id(cfg, j["id"])
    assert (r["holdout_status"], r["holdout_reason"]) == ("pending", "exam_error")
    out = json.loads(verdict_path(cfg, r["strategy"], r["timeframe"], r["strategy_hash"]).read_text())
    assert (out["holdout_status"], out["holdout_reason"]) == ("pending", "exam_error")
    # on a retry, no attempt is used
    monkeypatch.setattr(client, "check", lambda cfg: 8)
    crash(cfg, j["id"])
    monkeypatch.setattr(client, "check", unavailable)
    with pytest.raises(SandboxUnavailable):
        attempt_exam(cfg, j["id"], run=boom, data_fn=no_gaps, end_fn=end_after(200), cost_fn=no_cost)
    assert budget.attempts(cfg, j["id"]) == 1


def test_the_exam_runs_on_the_bytes_whose_hash_was_checked(cfg, plugins, monkeypatch):
    # the plugin file is read once: an edit after that read can't put unjudged bytes in the exam
    import perp_lab.strategy as strategy_mod
    budget.init_log(cfg)
    j = promoted(cfg)
    judged_bytes = (plugins / "donchian_atr.py").read_bytes()
    reads, real = [], strategy_mod.plugin_path

    class EditedAfterTheFirstRead:
        def read_bytes(self):
            reads.append(1)
            return judged_bytes if len(reads) == 1 else judged_bytes + b"\n# edited\n"
    monkeypatch.setattr(strategy_mod, "plugin_path",
                        lambda name: EditedAfterTheFirstRead() if name == "donchian_atr" else real(name))
    sent = []

    def run(cfg, src, tf, symbols, params, token, end, stderr_dir=None):
        sent.append(src.text)
        return stub_run(0.003)(cfg, src, tf, symbols, params, token, end)
    done = attempt_exam(cfg, j["id"], run=run, data_fn=no_gaps, end_fn=end_after(200), cost_fn=no_cost)
    assert sent == [judged_bytes] and len(reads) == 1 and done["holdout_status"] == "passed"


@pytest.mark.sandbox
def test_end_to_end_exam_through_the_real_sandbox(tmp_path, plugins, fresh_wf_caches):
    cfg = market_config(tmp_path, judge={"min_holdout_days": 30})
    budget.init_log(cfg)
    j = judge(cfg, "donchian_atr", "4h", validate=StubValidate(ROOT_STRONG))
    assert (j["holdout_status"], j["holdout_reason"]) == ("pending", "awaiting_exam") and budget.reads_used(cfg) == 0
    [done] = exam_pending(cfg)
    assert done["id"] == j["id"] and "error" not in done
    assert done["holdout_reason"] in {"passed", "failed"}       # a real exam ran on 92 days
    h = db.run_row(cfg, done["holdout_run_id"])
    assert (h["date_start"], h["date_end"]) == ("2026-07-01", "2026-10-01")
    assert pd.read_parquet(Path(h["run_dir"]) / "holdout_daily.parquet").shape[0] == 92
    assert budget.stored_window_end(cfg, j["id"]) == pd.Timestamp("2026-10-01", tz="UTC")


def test_the_attempt_cap_is_enforced_at_the_mint(cfg, plugins, monkeypatch):
    # another `lab exam` used the last attempt after this one counted them: the mint, under the
    # budget lock, refuses and the read fails
    budget.init_log(cfg)
    j = promoted(cfg)
    for _ in range(3):
        crash(cfg, j["id"])
    monkeypatch.setattr(budget, "attempts", lambda *a, **kw: 0)    # a stale count before the lock
    done = attempt_exam(cfg, j["id"], run=boom, data_fn=no_gaps, end_fn=end_after(200), cost_fn=no_cost)
    assert (done["holdout_status"], done["holdout_reason"]) == ("failed", "exam_error")
    assert len(budget.read_log(budget.log_path(cfg))) == 3


@pytest.mark.parametrize("err", [PluginError("cpu_limit", "", "over 900 s of CPU"),
                                 PluginError("exception", "ValueError", "close=123.45"),
                                 PluginError("died", "", "worker exited with code -9")])
def test_a_plugin_error_in_the_cost_probe_blocks_without_a_read(cfg, plugins, err):
    budget.init_log(cfg)
    j = promoted(cfg)

    def failing(*a, **kw):
        raise err
    done = attempt_exam(cfg, j["id"], run=boom, data_fn=no_gaps, end_fn=end_after(200), cost_fn=failing)
    assert (done["holdout_status"], done["holdout_reason"]) == ("blocked", "plugin_error")
    assert budget.reads_used(cfg) == 0
    text = Path(done["verdict_path"]).read_text()
    assert err.message not in text and (not err.type or err.type not in text)
    assert exams(cfg, boom) == []               # blocked: never probed again


@pytest.mark.parametrize("step", ["dated check", "cost probe"])
def test_a_sandbox_down_before_the_mint_mints_nothing(cfg, plugins, monkeypatch, step):
    import perp_lab.judge.judge as jm
    budget.init_log(cfg)
    j = promoted(cfg)

    def unavailable(*a, **kw):
        raise SandboxUnavailable("cannot start a worker")
    if step == "dated check":
        monkeypatch.setattr(jm, "dated_lookahead_failures", unavailable)
    with pytest.raises(SandboxUnavailable):
        attempt_exam(cfg, j["id"], run=boom, data_fn=no_gaps, end_fn=end_after(200),
                     cost_fn=unavailable if step == "cost probe" else no_cost)
    assert budget.reads_used(cfg) == 0
    r = db.get_judgement_by_id(cfg, j["id"])
    assert (r["holdout_status"], r["holdout_reason"]) == ("pending", "exam_error")
    out = json.loads(verdict_path(cfg, r["strategy"], r["timeframe"], r["strategy_hash"]).read_text())
    assert (out["holdout_status"], out["holdout_reason"]) == ("pending", "exam_error")


def test_an_exam_in_progress_is_left_alone(cfg, plugins):
    # a scheduled `lab exam` starts while another is still examining the same judgement: it skips
    # it (no mint, no attempt, no status change) and says so, without an error
    budget.init_log(cfg)
    j = promoted(cfg)
    inside, release, out = threading.Event(), threading.Event(), {}
    held = stub_run(0.003)

    def slow(cfg, src, tf, symbols, params, token, end, stderr_dir=None):
        inside.set()
        assert release.wait(60)
        return held(cfg, src, tf, symbols, params, token, end, stderr_dir=stderr_dir)
    a = threading.Thread(target=lambda: out.setdefault("a", attempt_exam(
        cfg, j["id"], run=slow, data_fn=no_gaps, end_fn=end_after(200), cost_fn=no_cost)))
    a.start()
    try:
        assert inside.wait(60)
        b = attempt_exam(cfg, j["id"], run=boom, data_fn=no_gaps, end_fn=boom, cost_fn=boom)
        assert b["in_progress"] is True and (b["holdout_status"], b["holdout_reason"]) == ("pending", "awaiting_exam")
        r = CliRunner().invoke(app, ["exam", "--config", str(cfg.path)])
        assert r.exit_code == 0 and "in progress elsewhere" in r.output and "ERROR" not in r.output
        assert budget.attempts(cfg, j["id"]) == 1
        r = db.get_judgement_by_id(cfg, j["id"])
        assert (r["holdout_status"], r["holdout_reason"]) == ("pending", "awaiting_exam")
    finally:
        release.set()
        a.join(60)
    assert out["a"]["holdout_status"] == "passed" and budget.attempts(cfg, j["id"]) == 1


def test_one_exam_at_a_time_on_this_machine(cfg, plugins):
    # Two families' exams must not overlap: their workers could pass holdout bars to each other
    # outside lockstep (R6). One that would start while another runs waits, with nothing spent.
    budget.init_log(cfg)
    add_plugin(plugins, "rootb", WIDE_TEXT.replace('PARENT = "donchian_atr"\n', "") + "# rootb\n")
    a, b = promoted(cfg), promoted(cfg, "rootb")
    assert a["lineage_id"] != b["lineage_id"]
    inner, held = [], stub_run(0.003)

    def run(cfg_, src, tf, symbols, params, token, end, stderr_dir=None):
        inner.append(attempt_exam(cfg, b["id"], run=boom, data_fn=no_gaps, end_fn=boom, cost_fn=boom))
        return held(cfg_, src, tf, symbols, params, token, end, stderr_dir=stderr_dir)
    done = attempt_exam(cfg, a["id"], run=run, data_fn=no_gaps, end_fn=end_after(200), cost_fn=no_cost)
    assert done["holdout_status"] == "passed"
    [other] = inner
    assert other["in_progress"] is True and (other["holdout_status"], other["holdout_reason"]) == \
        ("pending", "awaiting_exam")
    assert budget.reads_used(cfg) == 1 and not budget.holds_read(cfg, b["id"], b["lineage_id"])
    done = attempt_exam(cfg, b["id"], run=held, data_fn=no_gaps, end_fn=end_after(200),  # later, its turn
                        cost_fn=no_cost)
    assert done["holdout_status"] == "passed" and budget.reads_used(cfg) == 2


def test_a_retry_minted_after_a_stale_count_runs_on_the_stored_end(cfg, plugins, monkeypatch):
    # the reviewer's probe: the read was minted on day 200 after this exam counted none, so its
    # mint is a retry, and the exam must run on the stored end, not on its own later one
    budget.init_log(cfg)
    j = promoted(cfg)
    budget.mint(cfg, j["lineage_id"], j["id"], 200, DAY_200)          # the other exam's mint
    monkeypatch.setattr(budget, "holds_read", lambda *a, **kw: False)  # this exam's stale view
    ends, run = [], stub_run(0.003)

    def spy(cfg, src, tf, symbols, params, token, end, stderr_dir=None):
        ends.append(end)
        return run(cfg, src, tf, symbols, params, token, end, stderr_dir=stderr_dir)
    done = attempt_exam(cfg, j["id"], run=spy, data_fn=no_gaps, end_fn=end_after(260), cost_fn=no_cost)
    assert ends == [DAY_200] and done["holdout_status"] == "passed"
    assert db.run_row(cfg, done["holdout_run_id"])["date_end"] == str(DAY_200.date())
    assert [e["window_end"] for e in budget.read_log(budget.log_path(cfg))] == [DAY_200.isoformat()] * 2


def test_a_short_window_waits_without_starting_a_worker(cfg, plugins, monkeypatch):
    import perp_lab.judge.judge as jm
    budget.init_log(cfg)
    j = promoted(cfg)
    monkeypatch.setattr(jm, "dated_lookahead_failures", boom)
    monkeypatch.setattr(client, "check", boom)
    monkeypatch.setattr(client, "spawn", boom)
    done = attempt_exam(cfg, j["id"], run=boom, data_fn=no_gaps, end_fn=end_after(100), cost_fn=boom)
    assert (done["holdout_status"], done["holdout_reason"]) == ("pending", "too_short")
    assert budget.reads_used(cfg) == 0


# ---------------------------------------------------------------- withdrawing a pending promotion

WHY = "longs carry all the gross OOS P&L (pre-registered per-side check)"


def withdrawals(cfg) -> list[dict]:
    p = budget.withdrawals_path(cfg)
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def test_a_pending_promotion_is_withdrawn_before_its_read_and_its_family_keeps_the_read(cfg, plugins):
    budget.init_log(cfg)
    j = promoted(cfg)
    w = withdraw(cfg, j["id"], WHY)
    assert (w["holdout_status"], w["holdout_reason"]) == ("withdrawn", "withdrawn")
    [e] = withdrawals(cfg)
    assert {k: e[k] for k in ("judgement_id", "strategy", "timeframe", "strategy_hash", "lineage_id", "reason")} == {
        "judgement_id": j["id"], "strategy": "donchian_atr", "timeframe": "4h", "strategy_hash": j["strategy_hash"],
        "lineage_id": "family:donchian_atr", "reason": WHY}
    assert pd.Timestamp(e["time"]).tzinfo is not None
    out = json.loads(Path(w["verdict_path"]).read_text())
    assert (out["holdout_status"], out["holdout_reason"]) == ("withdrawn", "withdrawn")
    assert db.pending_judgements(cfg) == [] and exams(cfg, boom) == [] and budget.read_log(budget.log_path(cfg)) == []
    # the family's one read is still there: a later promotion in it is examined
    v = StubValidate({"donchian_atr": ("candidate", STRONG), "donchian_atr_wide": ("candidate", STRONG + 0.001)})
    child = judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert child["decision"] == "promote" and child["lineage_id"] == "family:donchian_atr"
    assert db.get_judgement_by_id(cfg, j["id"])["holdout_status"] == "withdrawn"     # not superseded over it
    [done] = exams(cfg, stub_run(0.003))
    assert (done["id"], done["holdout_status"]) == (child["id"], "passed") and budget.reads_used(cfg) == 1


def test_a_promotion_holding_its_read_cannot_be_withdrawn(cfg, plugins):
    # Its read is minted (a crashed exam): the holdout was unlocked, so the exam must finish or fail.
    budget.init_log(cfg)
    j = promoted(cfg)
    crash(cfg, j["id"])
    with pytest.raises(WithdrawRefused, match="holds its family's read"):
        withdraw(cfg, j["id"], WHY)
    r = db.get_judgement_by_id(cfg, j["id"])
    assert (r["holdout_status"], r["holdout_reason"]) == ("pending", "exam_error") and withdrawals(cfg) == []


def test_only_a_pending_promotion_can_be_withdrawn(cfg, plugins):
    budget.init_log(cfg)
    k = judge(cfg, "donchian_atr", "4h", validate=StubValidate({"donchian_atr": ("noise", NOISE)}))
    with pytest.raises(WithdrawRefused, match=r"is n/a \(n/a\), not pending"):
        withdraw(cfg, k["id"], WHY)
    with pytest.raises(WithdrawRefused, match="no judgement 999"):
        withdraw(cfg, 999, WHY)
    assert withdrawals(cfg) == []


@pytest.mark.parametrize("why", ["", "   ", None])
def test_a_withdrawal_needs_a_reason(cfg, plugins, why):
    budget.init_log(cfg)
    j = promoted(cfg)
    with pytest.raises(WithdrawRefused, match="a reason is required"):
        withdraw(cfg, j["id"], why)
    assert db.get_judgement_by_id(cfg, j["id"])["holdout_status"] == "pending" and withdrawals(cfg) == []


def test_a_promotion_being_examined_cannot_be_withdrawn(cfg, plugins):
    budget.init_log(cfg)
    j = promoted(cfg)
    with budget.exam_lock(cfg, j["id"]) as mine:             # as `lab exam` holds it throughout an attempt
        assert mine
        with pytest.raises(WithdrawRefused, match="being examined"):
            withdraw(cfg, j["id"], WHY)
    assert db.get_judgement_by_id(cfg, j["id"])["holdout_status"] == "pending" and withdrawals(cfg) == []


def test_no_withdrawal_while_the_read_record_cannot_be_read(cfg, plugins):
    j = promoted(cfg)                                        # no holdout log: whether it holds a read is unknown
    with pytest.raises(WithdrawRefused, match="whether it holds a read is unknown"):
        withdraw(cfg, j["id"], WHY)
    assert db.get_judgement_by_id(cfg, j["id"])["holdout_status"] == "pending" and withdrawals(cfg) == []


def test_cli_withdraw(cfg, plugins):
    budget.init_log(cfg)
    j = promoted(cfg)
    base = ["withdraw", "donchian_atr", "--tf", "4h", "--config", str(cfg.path)]
    r = CliRunner().invoke(app, base)                          # no --reason
    assert r.exit_code == 2 and db.get_judgement_by_id(cfg, j["id"])["holdout_status"] == "pending"
    r = CliRunner().invoke(app, ["withdraw", "donchian_atr", "--tf", "1h", "--reason", WHY, "--config", str(cfg.path)])
    assert r.exit_code == 2 and "no pending promotion of donchian_atr @ 1h" in r.output
    r = CliRunner().invoke(app, [*base, "--reason", WHY])
    assert r.exit_code == 0, r.output
    assert f"donchian_atr @ 4h: withdrawn (judgement {j['id']})" in r.output and "keeps its holdout read" in r.output
    r = CliRunner().invoke(app, [*base, "--reason", WHY])        # nothing pending any more
    assert r.exit_code == 2 and "no pending promotion" in r.output and len(withdrawals(cfg)) == 1


def test_cli_withdraw_refuses_a_read_already_minted(cfg, plugins):
    budget.init_log(cfg)
    j = promoted(cfg)
    crash(cfg, j["id"])
    r = CliRunner().invoke(app, ["withdraw", "donchian_atr", "--tf", "4h", "--reason", WHY, "--config", str(cfg.path)])
    assert r.exit_code == 2 and "refused:" in r.output and "holds its family's read" in r.output
