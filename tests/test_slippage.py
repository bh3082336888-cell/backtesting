import numpy as np
import pandas as pd
import pytest

from perp_lab.data.store import load_bars
from perp_lab.engine.slippage import (DAY_MS, fit_spread_curve, half_spread_bps, slippage_bps,
                                      trailing_adv)
from tests.conftest import market_config

M = {"adv_days": 30, "spread_bps_at_100m": 1.5, "alpha": 0.5, "max_bps": 50.0}


def test_slippage_is_floor_plus_spread_capped_and_never_below_the_floor():
    got = slippage_bps(np.array([1e12, 1e8, 1e6, 1e4, 0.0, np.nan]), 2.0, M)
    assert got == pytest.approx([2.015, 3.5, 17.0, 50.0, 50.0, 50.0])
    grid = slippage_bps(np.logspace(3, 13, 200), 2.0, M)
    assert (grid >= 2.0).all() and (np.diff(grid) <= 0).all()


def test_a_cap_below_the_floor_is_refused():
    with pytest.raises(ValueError):
        slippage_bps(np.array([1e8]), 2.0, {**M, "max_bps": 1.0})


def hourly(days):
    t = np.arange(days * 24) * 3_600_000
    return t, t // DAY_MS


def test_trailing_adv_uses_only_full_days_before_the_bar():
    t, day = hourly(40)
    adv = trailing_adv(t, (day + 1).astype(float), 30)       # day k's sum is 24 * (k + 1)
    assert np.isnan(adv[day == 0]).all()
    assert np.allclose(adv[day == 1], 24) and np.allclose(adv[day == 35], 24 * 20.5)   # days 5..34


def test_trailing_adv_skips_days_without_volume_data():
    t, day = hourly(12)
    qv = (day + 1).astype(float)
    qv[day == 10] = np.nan
    assert np.allclose(trailing_adv(t, qv, 30)[day == 11], 24 * 5.5)   # days 0..9 only


def test_half_spread_is_time_weighted():
    h = 3_600_000
    got = half_spread_bps(np.array([99.99, 99.9]), np.array([100.01, 100.1]), np.array([0, h]), 24 * h)
    assert got == pytest.approx((1.0 * 1 + 10.0 * 23) / 24)


def test_fit_recovers_a_power_law_and_lifts_it_to_the_envelope():
    rng = np.random.default_rng(0)
    adv = np.logspace(6, 10, 40)
    hs = 3.0 * (adv / 1e8) ** -0.4 * np.exp(rng.normal(0, 0.1, 40))
    f = fit_spread_curve(adv, hs)
    assert f["alpha"] == pytest.approx(0.4, abs=0.03) and f["spread_bps_at_100m"] >= 3.0 and f["n"] == 40
    curve = f["spread_bps_at_100m"] * (adv / 1e8) ** -f["alpha"]
    assert (hs <= curve).mean() >= 0.9


def test_fit_refuses_a_spread_that_does_not_fall_with_volume():
    adv = np.logspace(6, 10, 20)
    with pytest.raises(ValueError):
        fit_spread_curve(adv, 0.5 * (adv / 1e8) ** 0.2)
    with pytest.raises(ValueError):
        fit_spread_curve(adv[:5], 3.0 * (adv[:5] / 1e8) ** -0.4)


T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731


def with_volume(cfg, values):
    p = cfg.data_dir / "BTCUSDT" / "4h" / "all.parquet"
    df = pd.read_parquet(p)
    df["quote_volume"] = values(len(df))
    df.to_parquet(p)


def test_each_bar_pays_for_the_volume_before_its_day(tmp_path):
    cfg = market_config(tmp_path)
    with_volume(cfg, lambda n: np.full(n, 1e6))               # 6 bars a day: 6e6 USDT
    b = load_bars(cfg, "BTCUSDT", "4h")
    c = cfg["costs"]
    assert (b.slippage[b.index < T("2025-01-02")] == c["slippage_model"]["max_bps"] / 1e4).all()
    want = slippage_bps(np.array([6e6]), c["slippage_bps"], c["slippage_model"])[0] / 1e4
    assert np.allclose(b.slippage[b.index >= T("2025-02-01")], want) and b.slippage.notna().all()


def test_a_bars_slippage_does_not_depend_on_where_the_load_starts(tmp_path):
    cfg = market_config(tmp_path)
    with_volume(cfg, lambda n: np.random.default_rng(3).uniform(1e5, 1e7, n))
    full = load_bars(cfg, "BTCUSDT", "4h")
    part = load_bars(cfg, "BTCUSDT", "4h", start=T("2025-03-10"))
    pd.testing.assert_series_equal(part.slippage, full.slippage.loc[part.index])


def test_bars_without_volume_pay_the_cap_never_nan(tmp_path):
    cfg = market_config(tmp_path)                              # synthetic quote_volume is NaN
    b = load_bars(cfg, "ETHUSDT", "4h")
    assert (b.slippage == cfg["costs"]["slippage_model"]["max_bps"] / 1e4).all()
