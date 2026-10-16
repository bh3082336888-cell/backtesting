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


ROOT = Path(__file__).resolve().parents[1]


MINTERS = {"perp_lab/judge/budget.py"}     # may call issue_holdout_token
TOKEN_HOME = "perp_lab/holdout.py"          # the only file that may touch _MINT_KEY or build a HoldoutToken


def _name(node) -> str | None:
    return (node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute)
            else node.name if isinstance(node, ast.alias)
            else node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None)


def token_offenders(root: Path) -> tuple[list[str], list[str]]:
    """(files scanned, offenders) under root's perp_lab/, dashboard/ and strategies/."""
    scanned, offenders = [], []
    for d in ("perp_lab", "dashboard", "strategies"):
        for p in sorted((root / d).rglob("*.py")):
            rel = p.relative_to(root).as_posix()
            scanned.append(rel)
            if rel == TOKEN_HOME:
                continue
            for node in ast.walk(ast.parse(p.read_text())):
                name = _name(node)
                if name == "issue_holdout_token" and rel not in MINTERS:
                    offenders.append(f"{rel}: issue_holdout_token")
                if name == "_MINT_KEY":
                    offenders.append(f"{rel}: _MINT_KEY")
                if isinstance(node, ast.Call) and _name(node.func) == "HoldoutToken":
                    offenders.append(f"{rel}: HoldoutToken(")
    return scanned, offenders


def test_only_budget_mints_holdout_tokens():
    scanned, offenders = token_offenders(ROOT)
    assert {"perp_lab/judge/budget.py", "dashboard/app.py", "strategies/donchian_atr.py"} <= set(scanned)
    assert offenders == []


def test_the_token_scan_flags_every_way_around_the_budget(tmp_path):
    for d in ("perp_lab/judge", "dashboard", "strategies"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "perp_lab/holdout.py").write_text("_MINT_KEY = object()\nt = HoldoutToken('f', 'r', 'n', _MINT_KEY)\n")
    (tmp_path / "perp_lab/judge/budget.py").write_text("from perp_lab.holdout import issue_holdout_token\n")
    (tmp_path / "perp_lab/sneaky.py").write_text("from perp_lab.holdout import _MINT_KEY\n")
    (tmp_path / "dashboard/forge.py").write_text("import perp_lab.holdout as h\nt = h.HoldoutToken('f', 'r', 'n', k)\n")
    (tmp_path / "strategies/evil.py").write_text("from perp_lab.holdout import issue_holdout_token\n")
    (tmp_path / "strategies/getter.py").write_text("import perp_lab.holdout as h\nk = getattr(h, '_MINT_KEY')\n")
    assert sorted(token_offenders(tmp_path)[1]) == [
        "dashboard/forge.py: HoldoutToken(", "perp_lab/sneaky.py: _MINT_KEY",
        "strategies/evil.py: issue_holdout_token", "strategies/getter.py: _MINT_KEY"]


def test_cli_holdout_init_runs_once(cfg):
    r = CliRunner().invoke(app, ["holdout-init", "--config", str(cfg.path)])
    assert r.exit_code == 0 and "created" in r.output
    r = CliRunner().invoke(app, ["holdout-init", "--config", str(cfg.path)])
    assert r.exit_code == 1 and "never reset" in r.output


def test_holds_read_matches_judgement_and_family(cfg):
    with pytest.raises(BudgetLogError):
        budget.holds_read(cfg, 1, "fam")
    budget.init_log(cfg)
    assert budget.holds_read(cfg, 1, "fam") is False
    budget.mint(cfg, "fam", 1, 200)
    assert budget.holds_read(cfg, 1, "fam") is True
    assert budget.holds_read(cfg, 2, "fam") is False and budget.holds_read(cfg, 1, "other") is False
