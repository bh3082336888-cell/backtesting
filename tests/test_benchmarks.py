import pandas as pd
import pytest

from perp_lab.validate.montecarlo import coins_per_day, mc_drawdown

DAYS = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")


def test_monte_carlo_divides_each_trade_by_the_coins_listed_on_its_entry_day():
    cpd = coins_per_day({"A": pd.Series(0.0, index=DAYS), "B": pd.Series(0.0, index=DAYS[5:])})
    assert cpd.tolist() == [1] * 5 + [2] * 5
    ms = lambda d: int(DAYS[d].value // 10**6) + 3_600_000  # noqa: E731
    trades = pd.DataFrame({"entry_time": [ms(0), ms(1), ms(6), ms(7), ms(8)],
                           "pnl_pct": [0.0, -0.1, -0.1, 0.0, 0.0]})       # -0.1 alone, then -0.1 / 2
    assert mc_drawdown(trades, cpd, iterations=10, seed=1)["actual_dd"] == pytest.approx(0.9 * 0.95 - 1)
