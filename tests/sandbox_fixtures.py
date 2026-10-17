"""Shared helpers for sandbox tests."""

from perp_lab import strategy as _strategy
from tests.conftest import REPO


def load_class(name):
    """Load a real plugin in-process, from the repo's strategies/ (Task 12 re-points this)."""
    old = _strategy.STRATEGY_DIR
    _strategy.STRATEGY_DIR = REPO / "strategies"
    try:
        return _strategy.load_strategy_class(name)
    finally:
        _strategy.STRATEGY_DIR = old
