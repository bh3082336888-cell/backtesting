import pytest

from perp_lab import dsr
from perp_lab.judge import gates


@pytest.mark.parametrize("n,bar", [(7, 1.24), (10, 1.32), (20, 1.45), (30, 1.52), (50, 1.60),
                                   (100, 1.71), (107, 1.72), (200, 1.80), (500, 1.92), (1000, 2.00)])
def test_sharpe_bar_matches_the_spec_table(n, bar):
    assert dsr.sharpe_bar(n) == pytest.approx(bar, abs=0.005)


def test_gates_uses_the_moved_functions():
    assert gates.dsr_from_moments is dsr.dsr_from_moments
    assert gates.expected_max_sharpe is dsr.expected_max_sharpe
