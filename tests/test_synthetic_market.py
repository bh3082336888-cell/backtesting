import numpy as np
import pandas as pd
import pytest

from perp_lab.config import TF_MS
from perp_lab.validate.causality import synthetic_bars, with_moving_columns
from perp_lab.validate.synthetic_market import DAYS, MARKET_VERSION, SYMBOLS, market


def test_the_market_is_one_path_at_every_timeframe():
    m5, m4h = market("5m")["SYN1"], market("4h")["SYN1"]
    assert m4h.close.iloc[0] == m5.close.iloc[47] and len(m4h) * 48 == len(m5)


def test_market_5m_exists():
    m = market("5m")
    assert tuple(m) == SYMBOLS
    assert all(len(df) == 180 * 288 for df in m.values())


def test_15m_is_resampled_from_5m():
    m5, m15 = market("5m")["SYN1"], market("15m")["SYN1"]
    assert len(m15) * 3 == len(m5)
    assert m15.close.iloc[0] == m5.close.iloc[2] and m15.open.iloc[1] == m5.open.iloc[3]
    assert m15.high.iloc[1] == m5.high.iloc[3:6].max() and m15.volume.iloc[1] == pytest.approx(m5.volume.iloc[3:6].sum())


def test_the_market_is_fixed():
    assert (MARKET_VERSION, SYMBOLS, DAYS) == (2, ("SYN1", "SYN2", "SYN3", "SYN4"), 180)
    m = market("5m")
    assert tuple(m) == SYMBOLS and market("5m") is m                 # built once per timeframe
    base = with_moving_columns(synthetic_bars(n=DAYS * 288, seed=102, start="2023-01-01", funding_every=96,
                                              tf_ms=300_000), seed=202)
    pd.testing.assert_frame_equal(m["SYN3"], base)
    assert m["SYN1"].index[0] == pd.Timestamp("2023-01-01", tz="UTC")
    assert not m["SYN1"].close.equals(m["SYN2"].close)                # one path per symbol


def test_resampling_aggregates_each_column_by_its_meaning():
    b, h = market("5m")["SYN2"].iloc[48:96], market("4h")["SYN2"].iloc[1]
    assert (h.open, h.high, h.low, h.close) == (b.open.iloc[0], b.high.max(), b.low.min(), b.close.iloc[-1])
    for c in ("volume", "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "funding_charge"):
        assert h[c] == pytest.approx(b[c].sum(), rel=1e-12), c
    for c in ("funding_last", "funding_interval_h", "oi", "oi_value", "regime", "regime_adx", "close_time"):
        assert h[c] == b[c].iloc[-1], c
    assert h.open_time == b.open_time.iloc[0] == market("4h")["SYN2"].index[1].value // 10**6
    assert h.close_time - h.open_time == TF_MS["4h"] - 1


def test_every_timeframe_has_every_column_with_no_gaps():
    cols = list(market("5m")["SYN1"].columns)
    for tf in ("15m", "1h", "4h", "1d"):
        for s, df in market(tf).items():
            assert list(df.columns) == cols and len(df) == DAYS * 86_400_000 // TF_MS[tf]
            assert df.drop(columns="regime").notna().all().all()
            assert (np.diff(df.open_time.to_numpy()) == TF_MS[tf]).all()


def test_funding_is_charged_once_whatever_the_timeframe():
    for s in SYMBOLS:
        total = market("5m")[s].funding_charge.sum()
        assert market("1h")[s].funding_charge.sum() == pytest.approx(total)
        assert market("4h")[s].funding_charge.sum() == pytest.approx(total)


@pytest.mark.parametrize("tf", ["1m", "7m", "2w"])
def test_no_market_below_5m(tf):      # finer than 5m, or unknown
    with pytest.raises(ValueError, match="no synthetic market"):
        market(tf)
