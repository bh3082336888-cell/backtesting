import logging
import shutil

import numpy as np
import pandas as pd
import pytest

from perp_lab.data.store import load_bars, research_symbols
from perp_lab.engine.metrics import daily_returns
from perp_lab.plugins import PluginSource
from perp_lab.validate import walkforward as wfm
from tests.conftest import market_config
from tests.test_walkforward_sandbox import DONCHIAN

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


@pytest.fixture(autouse=True)
def fresh_wf_caches():
    wfm._cfg.cache_clear()
    wfm._bars.cache_clear()
    yield
    wfm._cfg.cache_clear()
    wfm._bars.cache_clear()


def keep_from(cfg, symbol, tf, since):
    """Drop a coin's candles before `since`, as if it were listed then."""
    p = cfg.data_dir / symbol / tf / "all.parquet"
    df = pd.read_parquet(p)
    df[df.open_time >= T(since).value // 10**6].to_parquet(p)


def test_daily_returns_of_an_empty_curve_has_a_utc_index():
    r = daily_returns(np.array([]), np.array([], dtype="int64"))
    assert r.empty and isinstance(r.index, pd.DatetimeIndex) and str(r.index.tz) == "UTC"
    assert r[r.index >= T("2025-01-01")].empty            # what walkforward._window does


def test_research_symbols_skips_coins_without_research_bars(tmp_path, caplog):
    cfg = market_config(tmp_path, universe=["BTC", "ETH", "SOL"])
    keep_from(cfg, "ETHUSDT", "4h", "2026-07-01")         # holdout-dated bars only
    shutil.rmtree(cfg.data_dir / "SOLUSDT" / "4h")        # never backfilled
    with caplog.at_level(logging.WARNING):
        assert research_symbols(cfg, cfg.symbols, "4h") == ["BTCUSDT"]
    assert "ETHUSDT 4h" in caplog.text and "SOLUSDT 4h" in caplog.text


def test_walkforward_skips_a_coin_without_research_bars(tmp_path):
    cfg = market_config(tmp_path, universe=["BTC", "ETH", "SOL"])
    keep_from(cfg, "SOLUSDT", "4h", "2026-07-01")
    wf = wfm.walkforward(cfg, PluginSource.from_text("donchian_atr", DONCHIAN), "4h", workers=1)
    assert wf.symbols == ["BTCUSDT", "ETHUSDT"] and set(wf.oos_daily) == {"BTCUSDT", "ETHUSDT"}


def test_walkforward_runs_with_a_coin_listed_mid_span(tmp_path):
    cfg = market_config(tmp_path, universe=["BTC", "ETH", "SOL"])
    keep_from(cfg, "SOLUSDT", "4h", "2025-10-01")         # after fold 0's test window (2025-07..09)
    wf = wfm.walkforward(cfg, PluginSource.from_text("donchian_atr", DONCHIAN), "4h", workers=1)
    assert wf.oos_daily["SOLUSDT"].index.min() >= T("2025-10-01")
    f0 = wf.folds[0]
    p = wf.oos_portfolio[(wf.oos_portfolio.index >= f0["test_start"]) & (wf.oos_portfolio.index < f0["test_end"])]
    both = (wf.oos_daily["BTCUSDT"] + wf.oos_daily["ETHUSDT"]) / 2
    pd.testing.assert_series_equal(p, both.loc[p.index], check_names=False, check_freq=False)


def test_oi_is_nan_before_the_first_metrics_snapshot(tmp_path):
    cfg = market_config(tmp_path)
    d = cfg.data_dir / "BTCUSDT" / "metrics"
    d.mkdir()
    ct = pd.date_range("2025-06-01", "2026-07-01", freq="1h", tz="UTC", inclusive="left").as_unit("ms").asi8
    pd.DataFrame({"create_time": ct, "oi": 1.0, "oi_value": 2.0}).to_parquet(d / "all.parquet")
    b = load_bars(cfg, "BTCUSDT", "4h")
    assert b.oi[b.index < T("2025-05-31")].isna().all() and (b.oi[b.index >= T("2025-06-01")] == 1.0).all()
