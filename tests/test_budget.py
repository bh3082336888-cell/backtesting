import ast
import json
import threading
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from perp_lab.cli import app
from perp_lab.holdout import check_token
from perp_lab.judge import budget
from perp_lab.judge.budget import BudgetCheck, BudgetLogError
from tests.test_store import cfg  # noqa: F401


def test_missing_log_refuses(cfg):
    with pytest.raises(BudgetLogError, match="holdout-init"):
        budget.mint(cfg, "fam", 1, 200)


def test_init_creates_an_empty_log_once(cfg):
    p = budget.init_log(cfg)
    assert p.exists() and p.read_text() == ""
    with pytest.raises(FileExistsError, match="never reset"):
        budget.init_log(cfg)


def test_corrupt_line_refuses(cfg):
    budget.init_log(cfg).write_text('{"lineage_id": "a", "nonce": "x"}\nnot json\n')
    with pytest.raises(BudgetLogError, match="line 2"):
        budget.mint(cfg, "fam", 1, 200)


def test_blank_lines_are_fine(cfg):
    budget.init_log(cfg).write_text('\n{"lineage_id": "a", "nonce": "x", "judgement_id": 9}\n\n')
    chk, tok = budget.mint(cfg, "fam", 1, 200)
    assert chk == BudgetCheck("mint", "ok") and tok is not None


def test_too_short_is_pending_and_writes_nothing(cfg):
    p = budget.init_log(cfg)
    assert budget.mint(cfg, "fam", 1, 179) == (BudgetCheck("pending", "too_short"), None)
    assert p.read_text() == ""


def test_mint_logs_the_judgement_and_a_family_gets_one_read(cfg):
    p = budget.init_log(cfg)
    chk, tok = budget.mint(cfg, "fam", 1, 200)
    check_token(tok)
    assert chk == BudgetCheck("mint", "ok") and json.loads(p.read_text())["judgement_id"] == 1
    assert budget.mint(cfg, "fam", 2, 200) == (BudgetCheck("blocked", "family_examined"), None)


def test_the_same_judgement_may_retry_without_a_new_read(cfg):
    budget.init_log(cfg)
    budget.mint(cfg, "fam", 1, 200)
    chk, tok = budget.mint(cfg, "fam", 1, 200)
    assert chk == BudgetCheck("mint", "retry") and tok is not None
    assert budget.reads_used(cfg) == 1


def test_a_judgement_id_must_be_an_int(cfg):
    p = budget.init_log(cfg)
    for bad in (None, True):
        with pytest.raises(TypeError, match="judgement_id must be an int"):
            budget.mint(cfg, "fam", bad, 200)
    assert p.read_text() == ""


def test_a_reused_judgement_id_in_another_family_is_not_a_retry(cfg):
    cfg.raw["judge"]["holdout_max_reads"] = 1
    budget.init_log(cfg)
    assert budget.mint(cfg, "a", 1, 200)[0] == BudgetCheck("mint", "ok")
    assert budget.mint(cfg, "b", 1, 200) == (BudgetCheck("blocked", "budget_spent"), None)
    assert budget.reads_used(cfg) == 1


def test_a_reused_judgement_id_still_obeys_the_length_rule(cfg):
    budget.init_log(cfg)
    budget.mint(cfg, "a", 1, 200)
    assert budget.mint(cfg, "b", 1, 0) == (BudgetCheck("pending", "too_short"), None)


def test_global_cap(cfg):
    cfg.raw["judge"]["holdout_max_reads"] = 2
    budget.init_log(cfg)
    budget.mint(cfg, "a", 1, 200)
    budget.mint(cfg, "b", 2, 200)
    assert budget.mint(cfg, "c", 3, 200) == (BudgetCheck("blocked", "budget_spent"), None)


def test_two_judges_racing_for_the_last_read(cfg, monkeypatch):
    cfg.raw["judge"]["holdout_max_reads"] = 1
    budget.init_log(cfg)
    real = budget.issue_holdout_token

    def slow(*a, **kw):                 # widen the window between reading the log and writing it
        time.sleep(0.2)
        return real(*a, **kw)

    monkeypatch.setattr(budget, "issue_holdout_token", slow)
    got = []
    threads = [threading.Thread(target=lambda f=f, j=j: got.append(budget.mint(cfg, f, j, 200)[1] is not None))
               for f, j in (("a", 1), ("b", 2))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(got) == [False, True] and budget.reads_used(cfg) == 1


def test_only_budget_mints_holdout_tokens():
    allowed = {Path("perp_lab/holdout.py"), Path("perp_lab/judge/budget.py")}
    offenders = []
    for p in [*Path("perp_lab").rglob("*.py"), *Path("dashboard").rglob("*.py")]:
        if p in allowed:
            continue
        for node in ast.walk(ast.parse(p.read_text())):
            name = (node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute)
                    else node.name if isinstance(node, ast.alias) else None)
            if name == "issue_holdout_token":
                offenders.append(str(p))
    assert offenders == []


def test_cli_holdout_init_runs_once(cfg):
    r = CliRunner().invoke(app, ["holdout-init", "--config", str(cfg.path)])
    assert r.exit_code == 0 and "created" in r.output
    r = CliRunner().invoke(app, ["holdout-init", "--config", str(cfg.path)])
    assert r.exit_code == 1 and "never reset" in r.output
