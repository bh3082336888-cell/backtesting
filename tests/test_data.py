import numpy as np
import pandas as pd

from perp_lab.data.store import align_funding, align_metrics
from perp_lab.data.vision import parse_funding, parse_klines

ROW = ["1640995200000", "1", "2", "0.5", "1.5", "10", "1640998799999", "15", "7", "5", "7.5", "0"]
HDR = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
       "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def test_parse_klines_with_and_without_header():
    a = parse_klines(pd.DataFrame([ROW], dtype=str))
    b = parse_klines(pd.DataFrame([HDR, ROW], dtype=str))
    pd.testing.assert_frame_equal(a, b)
    assert a.open_time.iloc[0] == 1640995200000 and a.close.iloc[0] == 1.5


def test_microsecond_timestamps_normalised():
    r = ROW.copy()
    r[0], r[6] = "1640995200000000", "1640998799999999"
    a = parse_klines(pd.DataFrame([r], dtype=str))
    assert a.open_time.iloc[0] == 1640995200000


def test_funding_time_snapped_and_aligned_to_bar():
    f = parse_funding(pd.DataFrame([["calc_time", "funding_interval_hours", "last_funding_rate"],
                                    ["1640995200006", "8", "0.0003"]], dtype=str))
    assert f.funding_time.iloc[0] == 1640995200000
    ot = 1640995200000 + np.arange(-2, 3) * 3_600_000
    c = pd.DataFrame({"open_time": ot, "close_time": ot + 3_599_999})
    out = align_funding(c, f, "1h")
    assert out.funding_charge.tolist() == [0, 0, 0.0003, 0, 0]
    # known at the close of the settlement bar, not before
    assert np.isnan(out.funding_last.iloc[1]) and out.funding_last.iloc[2] == 0.0003


def test_oi_asof_respects_publish_lag():
    ot = np.arange(0, 6) * 300_000
    c = pd.DataFrame({"open_time": ot, "close_time": ot + 299_999})
    m = pd.DataFrame({"create_time": [300_000], "oi": [5.0], "oi_value": [50.0]})
    out = align_metrics(c, m, publish_lag_ms=300_000)
    # snapshot at t=300000 known at 600000 → first bar whose close >= 600000 is bar 1 (close 599999)? no: bar 2
    assert np.isnan(out.oi.iloc[1]) and out.oi.iloc[2] == 5.0


def test_metrics_create_time_is_milliseconds():
    from perp_lab.data.vision import parse_metrics
    raw = pd.DataFrame([["create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
                         "a", "b", "c", "d"],
                        ["2022-01-01 00:05:00", "BTCUSDT", "1", "2", "", "", "", ""]], dtype=str)
    m = parse_metrics(raw)
    assert m.create_time.iloc[0] == 1640995500000
