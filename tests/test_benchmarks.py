import numpy as np
import pandas as pd
import pytest

from perp_lab.strategy import RandomEntry, Strategy
from perp_lab.engine.metrics import portfolio_returns, sharpe
from perp_lab.validate import benchmarks
from perp_lab.validate.montecarlo import coins_per_day, mc_drawdown
from perp_lab.validate.walkforward import WFResult
from tests.conftest import make_bars
from tests.test_store import cfg  # noqa: F401

DAYS = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")


class Quiet(Strategy):
    name = "quiet"
    p: dict = {}

    def prepare(self, bars):
        return bars[[]]


def make_wf(b_from=0, sides=None):
    zeros = lambda: pd.Series(0.0, index=DAYS)  # noqa: E731
    n = len(sides or [1])
    trades = pd.DataFrame({"symbol": ["A"] * n, "signal_time": [int(DAYS[2].value // 10**6)] * n,
                           "side": sides or [1]})
    return WFResult(
        strategy="x", tf="4h", symbols=["A", "B"], folds=[],
        oos_daily={"A": zeros(), "B": zeros()[b_from:]}, oos_portfolio=zeros(), oos_trades=trades,
        oos_exposure={}, oos_summary={"sharpe": 0.0}, oos_by_symbol={}, is_summary={},
        grid_table=pd.DataFrame(), selected=[{"combo": "default", "params": {}}],
        oos_flat_bars={"A": 100.0, "B": 100.0}, src=None, stderr_dir=None,
        raw={"oos_span": (DAYS[0], DAYS[-1] + pd.Timedelta(days=1))})


def test_monte_carlo_divides_each_trade_by_the_coins_listed_on_its_entry_day():
    cpd = coins_per_day({"A": pd.Series(0.0, index=DAYS), "B": pd.Series(0.0, index=DAYS[5:])})
    assert cpd.tolist() == [1] * 5 + [2] * 5
    ms = lambda d: int(DAYS[d].value // 10**6) + 3_600_000  # noqa: E731
    trades = pd.DataFrame({"entry_time": [ms(0), ms(1), ms(6), ms(7), ms(8)],
                           "pnl_pct": [0.0, -0.1, -0.1, 0.0, 0.0]})       # -0.1 alone, then -0.1 / 2
    assert mc_drawdown(trades, cpd, iterations=10, seed=1)["actual_dd"] == pytest.approx(0.9 * 0.95 - 1)


def test_random_entry_keeps_coins_the_strategy_never_traded(cfg, monkeypatch):
    rng = np.random.default_rng(0)
    ra = {seed: pd.Series(rng.normal(0, 0.01, 10), index=DAYS) for seed in range(3)}
    monkeypatch.setattr(benchmarks, "run_many", lambda cfg, src, jobs, workers, stderr_dir: [
        {"seed": j[4], "symbol": j[1], "daily": ra[j[4]], "trades": pd.DataFrame()} for j in jobs])
    out = benchmarks.random_entry(cfg, make_wf(), seeds=3)
    want = [sharpe(portfolio_returns({"A": ra[s], "B": pd.Series(0.0, index=DAYS)})) for s in range(3)]
    assert sorted(out["sharpes"]) == pytest.approx(sorted(want))


def test_random_entry_zero_fills_the_never_traded_coin_only_on_its_listed_days(cfg, monkeypatch):
    # Sharpe ignores a constant scale, so B must be listed on part of the span to show in it.
    rng = np.random.default_rng(1)
    ra = {seed: pd.Series(rng.normal(0, 0.01, 10), index=DAYS) for seed in range(3)}
    monkeypatch.setattr(benchmarks, "run_many", lambda cfg, src, jobs, workers, stderr_dir: [
        {"seed": j[4], "symbol": j[1], "daily": ra[j[4]], "trades": pd.DataFrame()} for j in jobs])
    out = benchmarks.random_entry(cfg, make_wf(b_from=5), seeds=3)
    want = [sharpe(portfolio_returns({"A": ra[s], "B": pd.Series(0.0, index=DAYS[5:])})) for s in range(3)]
    assert sorted(out["sharpes"]) == pytest.approx(sorted(want))


def test_monte_carlo_takes_several_trades_entering_on_one_day():
    cpd = coins_per_day({"A": pd.Series(0.0, index=DAYS)})
    day = int(DAYS[3].value // 10**6)
    trades = pd.DataFrame({"entry_time": [day + h * 3_600_000 for h in range(6)], "pnl_pct": [0.01] * 6})
    assert mc_drawdown(trades, cpd, iterations=10, seed=1)["n_trades"] == 6


def test_random_sides_follow_the_long_share():
    f = RandomEntry(Quiet(), 1.0, seed=3, long_share=0.8).prepare(make_bars(n=3000))
    nz = f["_rand"][f["_rand"] != 0]
    assert (nz > 0).mean() == pytest.approx(0.8, abs=0.03)


def test_a_long_share_of_one_half_draws_todays_entries():
    bars = make_bars(n=3000)
    u = np.random.default_rng(3).random((len(bars), 2))
    old = np.where(u[:, 0] < 0.3, np.where(u[:, 1] < 0.5, 1, -1), 0)
    assert (RandomEntry(Quiet(), 0.3, seed=3).prepare(bars)["_rand"].to_numpy() == old).all()


def test_the_benchmark_matches_the_plugins_overall_long_share(cfg, monkeypatch):
    captured = []

    def run_many(cfg, src, jobs, workers, stderr_dir):
        captured.extend(jobs)
        return [{"seed": j[4], "symbol": j[1], "daily": pd.Series(0.0, index=DAYS), "trades": pd.DataFrame()}
                for j in jobs]

    monkeypatch.setattr(benchmarks, "run_many", run_many)
    result = benchmarks.random_entry(cfg, make_wf(sides=[1, 1, 1, -1]), seeds=2)
    assert captured and all(j[5][1] == 0.75 for j in captured) and result["long_share"] == 0.75


def test_the_benchmark_takes_overrides_and_a_strategy_sharpe(cfg, monkeypatch):
    captured = []

    def run_many(cfg, src, jobs, workers, stderr_dir):
        captured.extend(jobs)
        return [{"seed": j[4], "symbol": j[1], "daily": pd.Series(0.0, index=DAYS), "trades": pd.DataFrame()}
                for j in jobs]

    monkeypatch.setattr(benchmarks, "run_many", run_many)
    out = benchmarks.random_entry(cfg, make_wf(), seeds=2, overrides={"slippage_scale": 0.0}, strat_sharpe=99.0)
    assert all(j[3] == {"slippage_scale": 0.0} for j in captured) and out["strategy_oos_sharpe"] == 99.0
