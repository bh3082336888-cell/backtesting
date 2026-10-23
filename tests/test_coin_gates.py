import pandas as pd
import pytest

from perp_lab.judge.judge import holdout_min_trades
from perp_lab.plugins import PluginSource
from perp_lab.runs import per_coin_min, verdict
from perp_lab.validate import walkforward as wfm
from tests.conftest import market_config
from tests.test_late_listing import keep_from
from tests.test_store import cfg  # noqa: F401
from tests.test_walkforward_sandbox import DONCHIAN


@pytest.fixture(autouse=True)
def fresh_wf_caches():
    wfm._cfg.cache_clear()
    wfm._bars.cache_clear()
    yield
    wfm._cfg.cache_clear()
    wfm._bars.cache_clear()


@pytest.mark.parametrize("n, train, oos, exam", [(8, 20, 30, 10), (17, 43, 64, 22), (11, 28, 42, 14)])
def test_per_coin_minimums_keep_todays_counts_at_8_coins(n, train, oos, exam):
    assert (per_coin_min(2.5, n), per_coin_min(30 / 8, n), per_coin_min(1.25, n)) == (train, oos, exam)


def test_listed_coins_counts_coins_with_a_return_in_the_window():
    d = pd.date_range("2025-01-01", periods=60, freq="D", tz="UTC")
    daily = {"A": pd.Series(0.0, index=d), "B": pd.Series(0.0, index=d[40:])}
    assert wfm.listed_coins(daily, d[0], d[30]) == 1 and wfm.listed_coins(daily, d[30], d[59]) == 2


def test_the_train_gate_counts_only_coins_listed_in_each_fold(tmp_path, monkeypatch):
    cfg = market_config(tmp_path)
    keep_from(cfg, "ETHUSDT", "4h", "2025-09-01")          # after fold 0's train window (2025-01..07)
    seen = []
    real = wfm.per_coin_min
    monkeypatch.setattr(wfm, "per_coin_min", lambda r, n: seen.append(n) or real(r, n))
    wf = wfm.walkforward(cfg, PluginSource.from_text("donchian_atr", DONCHIAN), "4h", workers=1)
    assert seen[0] == 1 and seen[-1] == 2 and len(seen) == len(wf.folds)


OK_RND, OK_SENS = {"beats_random": True}, {"narrow_island": False}


def test_the_oos_trade_minimum_scales_with_coins():
    status, fails = verdict({"sharpe": 1.0, "trades": 63}, OK_RND, OK_SENS, 17)
    assert status == "noise" and fails == ["only 63 OOS trades (< 64: 3.75 per coin × 17 coins)"]
    assert verdict({"sharpe": 1.0, "trades": 64}, OK_RND, OK_SENS, 17)[0] == "candidate"


def test_the_exam_minimum_scales_but_never_below_the_floor(cfg):
    assert [holdout_min_trades(cfg, n) for n in (2, 8, 17)] == [10, 10, 22]


def test_verdict_outside_session_fails():
    status, fails = verdict({"sharpe": 1.0, "trades": 64}, OK_RND, OK_SENS, 17, 3)
    assert status == "noise" and fails == ["3 OOS entries outside its declared session"]


def test_verdict_outside_session_zero_passes():
    assert verdict({"sharpe": 1.0, "trades": 64}, OK_RND, OK_SENS, 17, 0)[0] == "candidate"
    assert verdict({"sharpe": 1.0, "trades": 64}, OK_RND, OK_SENS, 17)[0] == "candidate"
