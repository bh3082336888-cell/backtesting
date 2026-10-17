"""Only perp_lab/sandbox/session.py executes plugin code."""

import ast

from tests.conftest import REPO

SESSION = REPO / "perp_lab" / "sandbox" / "session.py"
BARE = {"exec", "compile", "eval", "__import__", "load_strategy_class", "load_plugin"}
ATTR = {"import_module", "spec_from_file_location", "exec_module", "load_strategy_class", "load_plugin"}


def _calls(path):
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in BARE:
                yield node.lineno, f.id
            elif isinstance(f, ast.Attribute) and f.attr in ATTR:
                yield node.lineno, f.attr


def test_only_the_session_executes_plugin_code():
    files = [p for d in ("perp_lab", "dashboard") for p in (REPO / d).rglob("*.py")]
    assert len(files) > 30
    found = [f"{p.relative_to(REPO)}:{line} {name}"
             for p in files if p != SESSION for line, name in _calls(p)]
    assert not found, found
    assert any(True for _ in _calls(SESSION))          # the scan does see the executor


def test_the_in_process_loader_is_gone():
    from perp_lab import strategy
    assert not hasattr(strategy, "load_strategy_class")
