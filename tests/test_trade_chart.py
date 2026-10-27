"""Trade charts: candles from the run's own source (never the holdout), trades drawn where they happened."""

import json

import numpy as np
import pandas as pd
import pytest
import streamlit as st
import yaml
from streamlit.testing.v1 import AppTest

from dashboard import trade_chart as tc
from perp_lab.config import load_config
from perp_lab.holdout import HoldoutViolation
from tests.test_store import fake_run

H4 = 4 * 3_600_000


def _ms(s) -> int:
    return int(pd.Timestamp(s, tz="UTC").value // 10**6)


def candles(start="2024-01-01", n=200, tf_ms=H4, price=100.0):
    ot = _ms(start) + np.arange(n, dtype="int64") * tf_ms
    close = price + np.arange(n, dtype=float)
    df = pd.DataFrame({"open_time": ot, "open": close - 0.5, "high": close + 2, "low": close - 2, "close": close,
                       "volume": 1.0, "quote_volume": 1.0, "trades": 1.0, "taker_buy_volume": 1.0,
                       "taker_buy_quote_volume": 1.0, "close_time": ot + tf_ms - 1})
    df.index = pd.to_datetime(df.open_time, unit="ms", utc=True)
    return df


def trade(symbol="BTCUSDT", side=1, sig=10, bars=5, reason="stop", stop=95.0, target=np.nan, start="2024-01-01",
          tf_ms=H4, **kw):
    t0 = _ms(start)
    row = {"symbol": symbol, "tf": "4h", "side": side, "signal_time": t0 + sig * tf_ms,
           "entry_time": t0 + (sig + 1) * tf_ms, "entry_price": 100.0 + sig + 0.5,
           "exit_time": t0 + (sig + 1 + bars) * tf_ms, "exit_price": 100.0 + sig + bars, "exit_reason": reason,
           "stop_init": stop, "target": target, "pnl": -12.5, "pnl_pct": -0.00125, "r_multiple": -0.75,
           "regime": "trending", "fees": 1.25, "funding": 0.4, "fold": 1, "combo": "n=20"}
    return {**row, **kw}


def write_candles(data_dir, symbol, tf, df):
    d = data_dir / symbol / tf
    d.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(d / "all.parquet")


@pytest.fixture
def cfg(tmp_path):
    raw = yaml.safe_load(open("config/lab.yaml"))
    raw["data_dir"] = str(tmp_path / "data")
    raw["runs_dir"] = str(tmp_path / "runs")
    raw["store"] = {"url": f"sqlite:///{tmp_path}/lab.db"}
    raw["judge"] = {**raw["judge"], "holdout_log": str(tmp_path / "runs" / "holdout_log.jsonl"),
                    "verdicts_dir": str(tmp_path / "runs" / "verdicts")}
    p = tmp_path / "lab.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(p)


# ---------------------------------------------------------------- where candles come from

@pytest.mark.parametrize("mode", ["holdout", "IS"])
def test_holdout_and_in_sample_runs_are_refused_before_any_read(cfg, tmp_path, monkeypatch, mode):
    monkeypatch.setattr(tc, "load_candles", lambda *a, **k: pytest.fail("read candles for a refused run"))
    t0 = pd.Timestamp("2026-07-01", tz="UTC")
    with pytest.raises(tc.NotChartable):
        tc.run_candles(cfg, mode, tmp_path, "BTCUSDT", "4h", t0, t0 + pd.Timedelta(days=30))
    with pytest.raises(tc.NotChartable):
        tc.load_trades(mode, tmp_path)
    with pytest.raises(tc.NotChartable):
        tc.chart_span(cfg, mode, tmp_path, "BTCUSDT", "4h", "2026-07-01", "2026-12-31")


def test_walkforward_candles_stop_at_holdout_start_and_never_carry_a_token(cfg, monkeypatch):
    write_candles(cfg.data_dir, "BTCUSDT", "4h", candles("2026-06-01", n=6 * 60))   # into the holdout
    seen = []
    real = tc.load_candles
    monkeypatch.setattr(tc, "load_candles", lambda *a, **k: seen.append((a, k)) or real(*a, **k))
    c = tc.run_candles(cfg, "walkforward", None, "BTCUSDT", "4h", pd.Timestamp("2026-06-01", tz="UTC"),
                       pd.Timestamp("2026-08-01", tz="UTC"))
    assert c.index.max() < cfg.holdout_start and len(c) == 30 * 6
    assert seen and all(len(a) <= 5 and k.get("holdout") is None for a, k in seen)   # no token, ever
    # the guard is still there underneath: an unclipped read of the same window is refused
    with pytest.raises(HoldoutViolation):
        real(cfg, "BTCUSDT", "4h", pd.Timestamp("2026-06-01", tz="UTC"), pd.Timestamp("2026-08-01", tz="UTC"))


def test_walkforward_span_is_the_run_window_cut_at_holdout_start(cfg):
    s, e = tc.chart_span(cfg, "walkforward", None, "BTCUSDT", "4h", "2022-01-01", "2026-09-01")
    assert s == pd.Timestamp("2022-01-01", tz="UTC") and e == cfg.holdout_start


def test_paper_candles_are_the_bars_the_run_saved(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "load_candles", lambda *a, **k: pytest.fail("paper read the research store"))
    b = candles("2026-08-01", n=30).assign(live=[False] * 20 + [True] * 10)
    b.reset_index(drop=True).to_parquet(tmp_path / "bars_ETHUSDT.parquet", index=False)
    assert tc.chart_symbols("paper", tmp_path, ["BTCUSDT"]) == ["ETHUSDT"]
    s, e = tc.chart_span(cfg, "paper", tmp_path, "ETHUSDT", "4h", None, None)
    assert s == b.index[0] and e == b.index[-1] + pd.Timedelta(hours=4)
    c = tc.run_candles(cfg, "paper", tmp_path, "ETHUSDT", "4h", s, e)
    assert c.open_time.tolist() == b.open_time.tolist()
    with pytest.raises(FileNotFoundError):
        tc.run_candles(cfg, "paper", tmp_path, "BTCUSDT", "4h", s, e)


def test_trades_come_from_the_run_dir(tmp_path):
    pd.DataFrame([trade()]).to_parquet(tmp_path / "oos_trades.parquet")
    assert len(tc.load_trades("walkforward", tmp_path)) == 1
    assert tc.load_trades("paper", tmp_path).empty      # paper keeps trades.parquet, not saved here


# ---------------------------------------------------------------- windows

def test_default_window_is_the_last_test_fold_for_walkforward_and_the_whole_paper_run():
    span = (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC"))
    folds = pd.DataFrame({"fold": [0, 1], "test_start": ["2026-03-01", "2026-05-01"],
                          "test_end": ["2026-05-01", "2026-07-01"]})
    assert tc.default_window("walkforward", span, folds) == (pd.Timestamp("2026-05-01").date(),
                                                              pd.Timestamp("2026-06-30").date())
    assert tc.default_window("paper", span) == (span[0].date(), pd.Timestamp("2026-06-30").date())


def test_trade_window_pads_the_trade_and_stays_inside_the_span():
    span = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-03-01", tz="UTC"))
    t = trade(sig=3, bars=5)
    s, e = tc.trade_window(t, "4h", span)
    assert s == span[0].date()                                   # 40 bars before the signal is before the span
    assert e == (pd.to_datetime(t["exit_time"], unit="ms") + pd.Timedelta(hours=4 * 40)).date()


def test_picked_window_handles_a_half_picked_range():
    span = (pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-03-01", tz="UTC"))
    d = pd.Timestamp("2024-02-10").date()
    assert tc.picked_window((d,), span) == (pd.Timestamp(d, tz="UTC"), pd.Timestamp(d, tz="UTC") + tc.DAY)
    assert tc.picked_window((), span) == span
    assert tc.picked_window((d, pd.Timestamp("2024-05-01").date()), span)[1] == span[1]


# ---------------------------------------------------------------- density

def test_long_windows_are_drawn_coarser():
    assert tc.display_tf("15m", 5000) == "15m"
    assert tc.display_tf("15m", 20_000) == "1h"
    assert tc.display_tf("4h", 9000) == "1d"


def test_resample_keeps_ohlc():
    c = candles(n=12, tf_ms=3_600_000)                        # 12 hourly bars → 3 × 4h
    r = tc.resample(c, "4h")
    assert len(r) == 3
    first = c.iloc[:4]
    assert r.iloc[0].tolist() == [first.open.iloc[0], first.high.max(), first.low.min(), first.close.iloc[-1]]


# ---------------------------------------------------------------- figure

def _traces(fig, name):
    return [t for t in fig.data if t.name == name]


def _rects(fig, name):
    """Every box in the named bar trace as (x0, x1, y0, y1): y0 the entry side (the bar's base)."""
    out = []
    for tr in fig.data:
        if tr.type == "bar" and tr.name == name:
            for x, w, b, y in zip(tr.x, tr.width, tr.base, tr.y):
                x0 = pd.Timestamp(x)
                out.append((x0, x0 + pd.Timedelta(milliseconds=int(w)), b, pytest.approx(b + y)))
    return out


def _notes(fig, name):
    """Annotations of one kind ("price tag", "time tag", "trade label", "be note")."""
    return [a for a in fig.layout.annotations if a.name == name]


def _xs(fig):
    """Every x a trade trace puts on the chart (not the candles)."""
    return [pd.Timestamp(x) for tr in fig.data if tr.type != "candlestick" for x in tr.x if x is not None]


def test_figure_draws_boxes_entries_signal_bar_exits_stops_targets_and_folds():
    c = candles(n=200)
    t = pd.DataFrame([
        trade(sig=10, reason="stop", stop=95.0),
        trade(sig=30, side=-1, reason="target", stop=140.0, target=120.0),
        trade(sig=50, reason="signal"), trade(sig=70, reason="liq"), trade(sig=90, reason="eow"),
        trade(sig=500, reason="stop"),                                     # outside the window
    ])
    folds = pd.DataFrame({"fold": [0, 1], "train_start": ["2023-07-01", "2023-09-01"],
                          "train_end": ["2024-01-10", "2024-01-20"], "test_start": ["2024-01-10", "2024-01-20"],
                          "test_end": ["2024-01-20", "2024-02-20"], "combo": ["n=10", "n=20"]})
    fig = tc.trade_figure(c, t, "4h", folds=folds)

    assert sum(x.type == "candlestick" for x in fig.data) == 1
    (ent,) = _traces(fig, "entry")                                  # one dot per trade, long or short
    assert ent.marker.color == tc.PALETTE["s1"] and len(ent.x) == 5
    assert pd.Timestamp(ent.x[1]) == pd.to_datetime(t.entry_time[1], unit="ms")      # UTC, zone dropped
    assert list(ent.y) == t.entry_price[:5].tolist()
    colors = tc.exit_colors()
    for reason in tc.EXIT_REASONS:
        (ex,) = _traces(fig, f"exit: {reason}")
        assert ex.marker.color == colors[reason] and ex.marker.symbol == "circle" and len(ex.x) == 1
    assert len(set(colors.values())) == len(colors)
    # the short's target is below its entry and its stop above; every trade has a risk box
    assert _rects(fig, tc.BOX_REWARD) == [(_at(31), _at(36), 130.5, 120.0)]
    assert len(_rects(fig, tc.BOX_RISK)) == 5 and _rects(fig, tc.BOX_RISK)[1] == (_at(31), _at(36), 130.5, 140.0)
    assert len(_rects(fig, tc.BOX_RESULT)) == 4                    # the four without a target
    (sig,) = _traces(fig, "signal bar (close)")
    assert list(sig.y) == [c.close.iloc[i] for i in (10, 30, 50, 70, 90)]     # the signal bar's own close
    (stop,) = _traces(fig, "initial stop")
    assert [y for y in stop.y if y is not None] == [95.0, 95.0, 140.0, 140.0] + [95.0] * 6
    (tgt,) = _traces(fig, "initial target")
    assert [y for y in tgt.y if y is not None] == [120.0, 120.0]               # NaN targets draw nothing
    hov = ent.hovertext[0]
    for s in ("P&L -12.50 USDT", "R -0.75", "regime at signal: trending", "fees 1.25", "funding paid 0.40",
              "fold 1 · n=20", "initial stop 95", "initial target —", "no fixed target"):
        assert s in hov, s
    (box,) = [tr for tr in fig.data if tr.name == tc.BOX_RISK]
    assert box.hovertext[0] == hov                                    # hovering a box gives the same tooltip
    shading = [a for a in fig.layout.annotations if not a.name]          # not the trades' tags
    assert [a.text for a in shading] == ["train only", "fold 0", "fold 1"]
    assert "trained on 2023-09-01 → 2024-01-20" in shading[2].hovertext
    assert len([s for s in fig.layout.shapes if s.type == "rect"]) == 2   # train-only region + fold 0 (folds alternate)


def test_coarse_candles_keep_box_coordinates_exact():
    c = candles(n=7000)
    t = pd.DataFrame([trade(sig=6500, bars=3, stop=6590.0, target=6650.0)])          # entry 6600.5
    fig = tc.trade_figure(c, t, "4h")
    (cs,) = [x for x in fig.data if x.type == "candlestick"]
    assert len(cs.x) <= tc.MAX_CANDLES and cs.name == "1d candles"
    assert _rects(fig, tc.BOX_RISK) == [(_at(6501), _at(6504), 6600.5, 6590.0)]
    assert _rects(fig, tc.BOX_REWARD) == [(_at(6501), _at(6504), 6600.5, 6650.0)]
    (ent,) = _traces(fig, "entry")
    assert pd.Timestamp(ent.x[0]) == _at(6501) and list(ent.y) == [6600.5]
    (ex,) = _traces(fig, "exit: stop")
    assert pd.Timestamp(ex.x[0]) == _at(6504) and list(ex.y) == [6603.0]
    assert [pd.Timestamp(a.x) for a in _notes(fig, "time tag")] == [_at(6501), _at(6504)]
    assert list(_traces(fig, "signal bar (close)")[0].y) == [c.close.iloc[6500]]


def test_paper_warmup_is_shaded():
    c = candles(n=30).assign(live=[False] * 20 + [True] * 10)
    fig = tc.trade_figure(c, pd.DataFrame(columns=tc.TRADE_COLS), "4h")
    assert [a.text for a in fig.layout.annotations] == ["warmup"]
    assert pd.Timestamp(fig.layout.shapes[0].x1) == c.index[20].tz_localize(None)


def test_received_funding_reads_as_received():
    (h,) = tc.hover_text(pd.DataFrame([trade(funding=-0.3)]))
    assert "funding received 0.30" in h
    (h,) = tc.hover_text(pd.DataFrame([trade(funding=0.0)]))
    assert "funding 0.00" in h


# ---------------------------------------------------------------- level path

def _lv(t0, offs, stops, targets):
    """Level columns for a trade whose entry is at t0 (epoch ms): one level per offset, in bars."""
    return dict(level_time=[t0 + o * H4 for o in offs], level_stop=stops, level_target=targets)


def _xy(tr):
    return [(pd.Timestamp(x), y) for x, y in zip(tr.x, tr.y)]


def _at(i):
    return pd.Timestamp("2024-01-01") + pd.Timedelta(milliseconds=i * H4)


def test_trailing_stop_is_drawn_as_a_step_line():
    c = candles(n=200)
    base = trade(sig=10, bars=6, stop=95.0)       # entry bar 11, exit bar 17
    t = pd.DataFrame([{**base, **_lv(base["entry_time"], [0, 2, 4], [95.0, 97.0, 99.0], [np.nan] * 3)}])
    fig = tc.trade_figure(c, t, "4h")
    (stop,) = _traces(fig, "stop")
    assert _xy(stop)[:-1] == [(_at(11), 95.0), (_at(13), 95.0), (_at(13), 97.0), (_at(15), 97.0),
                              (_at(15), 99.0), (_at(17), 99.0)]
    assert stop.x[-1] is None and stop.y[-1] is None
    assert not _traces(fig, "target") and not _traces(fig, "initial stop")
    assert stop.line.dash == "dot" and stop.line.color == tc.PALETTE["neg"] and stop.hoverinfo == "skip"


def test_a_target_that_goes_nan_breaks_the_line_and_resumes():
    c = candles(n=200)
    base = trade(sig=10, bars=6, stop=95.0)
    t = pd.DataFrame([{**base, **_lv(base["entry_time"], [0, 2, 4], [95.0] * 3, [120.0, np.nan, 130.0])}])
    (tgt,) = _traces(tc.trade_figure(c, t, "4h"), "target")
    assert tgt.x[-1] is None and tgt.line.color == tc.PALETTE["pos"]          # the target's green, as its box
    assert [(pd.Timestamp(x), y) if x is not None else None for x, y in zip(tgt.x, tgt.y)] == [
        (_at(11), 120.0), (_at(13), 120.0), None, (_at(15), 130.0), (_at(17), 130.0), None]


def test_levels_read_back_from_parquet_are_arrays(tmp_path):
    base = trade(sig=10, bars=6, stop=95.0)
    t = pd.DataFrame([{**base, **_lv(base["entry_time"], [0, 3], [95.0, 98.0], [np.nan, 120.0])},
                      {**trade(sig=40, bars=4, stop=90.0), **_lv(_ms("2024-01-01") + 41 * H4, [0], [90.0], [np.nan])}])
    t.to_parquet(tmp_path / "t.parquet")
    back = pd.read_parquet(tmp_path / "t.parquet")
    assert isinstance(back.level_stop.iloc[0], np.ndarray) and tc.has_levels(back)
    fig = tc.trade_figure(candles(n=200), back, "4h")
    (stop,) = _traces(fig, "stop")
    assert [y for y in stop.y if y is not None] == [95.0, 95.0, 98.0, 98.0, 90.0, 90.0]
    (tgt,) = _traces(fig, "target")
    assert [y for y in tgt.y if y is not None] == [120.0, 120.0]


@pytest.mark.parametrize("empty", [np.nan, None])
def test_an_all_nan_target_reads_back_from_parquet_and_draws_no_target_line(tmp_path, empty):
    # a strategy that never sets a target: pyarrow stores the column as list<null>, which reads back as
    # object arrays of None
    base = trade(sig=10, bars=6, stop=95.0)
    t = pd.DataFrame([{**base, **_lv(base["entry_time"], [0, 3], [95.0, 98.0], [empty, empty])},
                      {**trade(sig=40, bars=4, stop=90.0), **_lv(_ms("2024-01-01") + 41 * H4, [0], [90.0], [empty])}])
    t.to_parquet(tmp_path / "t.parquet")
    back = pd.read_parquet(tmp_path / "t.parquet")
    assert tc.has_levels(back)
    if empty is None:
        assert pd.isna(back.level_target.iloc[0][0])
    fig = tc.trade_figure(candles(n=200), back, "4h")
    assert not _traces(fig, "target")
    (stop,) = _traces(fig, "stop")
    assert [y for y in stop.y if y is not None] == [95.0, 95.0, 98.0, 98.0, 90.0, 90.0]
    assert "target at exit —" in tc.hover_text(back)[0]


def test_old_trades_without_levels_use_the_initial_stop_and_target():
    t = pd.DataFrame([trade(sig=10, stop=95.0, target=120.0)])          # entry bar 11 at 110.5, exit bar 16
    assert not tc.has_levels(t)
    fig = tc.trade_figure(candles(n=200), t, "4h")
    assert _rects(fig, tc.BOX_RISK) == [(_at(11), _at(16), 110.5, 95.0)]
    assert _rects(fig, tc.BOX_REWARD) == [(_at(11), _at(16), 110.5, 120.0)]
    (stop,) = _traces(fig, "initial stop")
    (tgt,) = _traces(fig, "initial target")
    assert [y for y in stop.y if y is not None] == [95.0, 95.0] and [y for y in tgt.y if y is not None] == [120.0] * 2
    assert not _traces(fig, "stop") and not _traces(fig, "target")
    # a row whose levels are None or empty (a frame mixing old and new) is drawn the same way
    for empty in (None, []):
        m = pd.DataFrame([{**trade(sig=10, stop=95.0), "level_time": empty, "level_stop": empty,
                           "level_target": empty}])
        assert not tc.has_levels(m)
        assert _traces(tc.trade_figure(candles(n=200), m, "4h"), "initial stop")


def test_hover_gives_the_levels_at_exit():
    base = trade(sig=10, bars=6, stop=95.0)
    t = pd.DataFrame([{**base, **_lv(base["entry_time"], [0, 2], [95.0, 99.5], [np.nan, 130.0])},
                      {**base, **_lv(base["entry_time"], [0], [95.0], [np.nan])}])
    h = tc.hover_text(t)
    assert "stop at exit 99.5 · target at exit 130" in h[0]
    assert "stop at exit 95 · target at exit —" in h[1]
    assert "initial stop 95" in h[0]
    assert "at exit" not in tc.hover_text(pd.DataFrame([trade()]))[0]


def test_a_level_stamped_before_the_entry_never_steps_backwards():
    base = trade(sig=10, bars=6, stop=95.0)       # entry bar 11, exit bar 17
    lv = _lv(base["entry_time"], [0, -2, 4], [95.0, 97.0, 99.0], [np.nan] * 3)
    t = pd.DataFrame([{**base, **lv}])
    (stop,) = _traces(tc.trade_figure(candles(n=200), t, "4h"), "stop")
    xs = [pd.Timestamp(x) for x in stop.x if x is not None]
    assert xs == sorted(xs) and xs[0] == _at(11)
    assert [y for y in stop.y if y is not None] == [95.0, 95.0, 97.0, 97.0, 99.0, 99.0]


# ---------------------------------------------------------------- position boxes, tags and labels

M5 = 300_000
T0 = _ms("2024-03-04 13:40")          # a Monday


def _pos(side=1, **kw):
    """A 5m trade like the LCE note's: long 99.06, stop 98.82, target 99.30, out at the target 30 minutes
    later, +0.73R net. The short mirrors it about the entry."""
    stop, target = (98.82, 99.30) if side > 0 else (99.30, 98.82)
    row = {"symbol": "BTCUSDT", "tf": "5m", "side": side, "signal_time": T0 - M5, "entry_time": T0,
           "entry_price": 99.06, "exit_time": T0 + 6 * M5, "exit_price": target, "exit_reason": "target",
           "stop_init": stop, "target": target, "qty": 10.0, "pnl_gross": 2.4, "slippage": 0.072, "fees": 0.6,
           "funding": 0.0, "pnl": 1.752, "pnl_pct": 0.000175, "r_multiple": 0.73, "regime": "trending"}
    return {**row, **kw}


def _day(price=99.0):
    """A day of 5m candles, the day of `_pos`."""
    return candles("2024-03-04", n=288, tf_ms=M5, price=price)


def test_rr_and_result_text_long_and_short():
    # R:R = |99.30 - 99.06| / |99.06 - 98.82| = 1.00; gross = (2.4 + 0.072) / (10 * 0.24) = 1.03R
    for side in (1, -1):
        assert tc.rr_text(_pos(side)) == "R:R 1 : 1.00"
        assert tc.result_text(_pos(side)) == "+1.03R gross · +0.73R net"
    assert tc.rr_text(_pos(target=99.42)) == "R:R 1 : 1.50"
    assert tc.result_text(_pos(pnl_gross=-2.472, r_multiple=-1.31)) == "-1.00R gross · -1.31R net"


def test_result_text_without_slippage_column_says_after_slippage():
    r = _pos()
    del r["slippage"]                                       # a run from before the engine kept it
    assert tc.result_text(r) == "+1.00R gross after slippage · +0.73R net"
    assert tc.result_text(_pos(slippage=np.nan)) == "+1.00R gross after slippage · +0.73R net"   # mixed frame
    assert tc.result_text(_pos(slippage=0.0)) == "+1.00R gross · +0.73R net"                     # paper: kept, 0


def test_no_target_trade_has_no_reward_box_and_says_so():
    win = trade(sig=10, bars=6, stop=95.0, exit_price=115.0, reason="signal")          # long 110.5 → 115
    loss = trade(sig=40, side=-1, bars=6, stop=160.0, exit_price=145.0, reason="signal")   # short 140.5 → 145
    b = tc.trade_boxes(win)
    assert b["no_target"] and b["reward"] == [] and b["risk"] == [(_ms("2024-01-01") + 11 * H4,
                                                                    _ms("2024-01-01") + 17 * H4, 110.5, 95.0)]
    assert tc.rr_text(win) == "no fixed target"
    fig = tc.trade_figure(candles(n=200), pd.DataFrame([win, loss]), "4h")
    assert not _rects(fig, tc.BOX_REWARD)
    assert _rects(fig, tc.BOX_RESULT) == [(_at(11), _at(17), 110.5, 115.0), (_at(41), _at(47), 140.5, 145.0)]
    (res,) = [tr for tr in fig.data if tr.name == tc.BOX_RESULT]
    assert list(res.marker.color) == [tc.tint(tc.PALETTE["pos"]), tc.tint(tc.PALETTE["neg"])]   # gain, loss
    assert len(_rects(fig, tc.BOX_RISK)) == 2                            # the stop is still the risk
    labels = _notes(fig, "trade label")
    assert len(labels) == 2 and all(a.text.startswith("<b>no fixed target</b>") for a in labels)
    assert [(a.y, a.yanchor) for a in labels] == [(115.0, "bottom"), (140.5, "top")]   # the reward side
    assert not [a for a in _notes(fig, "price tag") if a.text.startswith("TP")]


def test_breakeven_collapses_the_risk_box_at_its_time():
    r = _pos(level_time=[T0, T0 + 3 * M5], level_stop=[98.82, 99.06], level_target=[99.30, 99.30])
    b = tc.trade_boxes(r)
    assert b["be_time"] == T0 + 3 * M5
    assert b["risk"] == [(T0, T0 + 3 * M5, 99.06, 98.82)]               # nothing after the move
    assert b["reward"] == [(T0, T0 + 6 * M5, 99.06, 99.30)]             # the target never moved: one box
    assert b["entry"] == (T0, 99.06) and b["exit"] == (T0 + 6 * M5, 99.30, "target") and not b["no_target"]
    fig = tc.trade_figure(_day(), pd.DataFrame([r]), "5m")
    t = lambda ms: pd.Timestamp(ms, unit="ms")                           # noqa: E731
    assert _rects(fig, tc.BOX_RISK) == [(t(T0), t(T0 + 3 * M5), 99.06, 98.82)]
    (be,) = _notes(fig, "be note")
    assert be.text == "SL → BE 13:55" and pd.Timestamp(be.x) == t(T0 + 3 * M5) and be.y == 99.06
    assert sorted(a.text for a in _notes(fig, "time tag")) == ["13:40 entry", "14:10 exit"]   # one day: no date
    (lab,) = _notes(fig, "trade label")
    assert "R:R 1 : 1.00" in lab.text and "+1.03R gross · +0.73R net" in lab.text
    assert tc.trade_boxes(_pos())["be_time"] is None


def test_trailing_stop_past_entry_collapses_risk_and_keeps_the_step_line():
    c = candles(n=200)
    base = trade(sig=10, bars=6, stop=105.0, exit_price=113.0)          # long 110.5, entry bar 11, exit bar 17
    r = {**base, **_lv(base["entry_time"], [0, 2, 4], [105.0, 108.0, 112.0], [np.nan] * 3)}
    b = tc.trade_boxes(r)
    ms = lambda i: _ms("2024-01-01") + i * H4                             # noqa: E731
    assert b["risk"] == [(ms(11), ms(13), 110.5, 105.0), (ms(13), ms(15), 110.5, 108.0)]   # steps, then gone
    assert b["be_time"] == ms(15)                                        # trailed past the entry
    fig = tc.trade_figure(c, pd.DataFrame([r]), "4h")
    assert _rects(fig, tc.BOX_RISK) == [(_at(11), _at(13), 110.5, 105.0), (_at(13), _at(15), 110.5, 108.0)]
    (stop,) = _traces(fig, "stop")
    assert [y for y in stop.y if y is not None] == [105.0, 105.0, 108.0, 108.0, 112.0, 112.0]   # carries on


def test_short_trade_boxes_are_mirrored():
    long_, short = tc.trade_boxes(_pos()), tc.trade_boxes(_pos(-1))
    end = T0 + 6 * M5
    assert long_["reward"] == [(T0, end, 99.06, 99.30)] and long_["risk"] == [(T0, end, 99.06, 98.82)]
    assert short["reward"] == [(T0, end, 99.06, 98.82)] and short["risk"] == [(T0, end, 99.06, 99.30)]
    figs = {s: tc.trade_figure(_day(), pd.DataFrame([_pos(s)]), "5m") for s in (1, -1)}
    t = lambda ms: pd.Timestamp(ms, unit="ms")                           # noqa: E731
    assert _rects(figs[-1], tc.BOX_REWARD) == [(t(T0), t(end), 99.06, 98.82)]   # reward below the entry
    assert _rects(figs[-1], tc.BOX_RISK) == [(t(T0), t(end), 99.06, 99.30)]     # risk above
    tags = {a.text: a.y for a in _notes(figs[-1], "price tag")}
    assert tags == {"Entry 99.06": 99.06, "TP 98.82": 98.82, "SL 99.3": 99.30}
    assert {a.text: a.y for a in _notes(figs[1], "price tag")} == {"Entry 99.06": 99.06, "TP 99.3": 99.30,
                                                                    "SL 98.82": 98.82}
    (ls,) = _notes(figs[-1], "trade label")
    (ll,) = _notes(figs[1], "trade label")
    assert (ll.y, ll.yanchor) == (99.30, "top") and (ls.y, ls.yanchor) == (98.82, "bottom")   # inside the reward
    assert ls.text == ll.text


def test_trade_starting_before_the_window_draws_from_the_window_start():
    c = candles(n=200)
    early = trade(sig=-6, bars=10, stop=90.0, target=130.0)               # entry bar -5, exit bar 5
    late = trade(sig=195, bars=10, stop=280.0, target=320.0)              # entry bar 196, exit bar 206
    fig = tc.trade_figure(c, pd.DataFrame([early, late]), "4h")
    assert _rects(fig, tc.BOX_RISK) == [(_at(0), _at(5), 94.5, 90.0), (_at(196), _at(200), 295.5, 280.0)]
    assert _rects(fig, tc.BOX_REWARD) == [(_at(0), _at(5), 94.5, 130.0), (_at(196), _at(200), 295.5, 320.0)]
    assert [pd.Timestamp(x) for x in _traces(fig, "entry")[0].x] == [_at(196)]    # no dot off the window
    assert [pd.Timestamp(x) for x in _traces(fig, "exit: stop")[0].x] == [_at(5)]
    times = sorted((pd.Timestamp(a.x), a.text.split()[-1]) for a in _notes(fig, "time tag"))
    assert times == [(_at(5), "exit"), (_at(196), "entry")]               # no tag for an off-screen time
    assert len([s for s in fig.layout.shapes if s.name == "time tag"]) == 2
    xs = _xs(fig)
    assert min(xs) >= _at(0) and max(xs) <= _at(200)                      # nothing widens the window


def _row(k, **kw):
    return trade(sig=10 + 20 * k, bars=5, stop=95.0 + 20 * k, target=126.0 + 20 * k, **kw)   # R:R 1 : 1


def test_tags_only_for_the_focused_trade_when_more_than_five():
    t = pd.DataFrame([_row(k) for k in range(6)])
    t.index = t.index + 100                                               # focus is a row label, not a position
    fig = tc.trade_figure(candles(n=200), t, "4h")
    assert not _notes(fig, "price tag") and not _notes(fig, "time tag") and not _notes(fig, "trade label")
    assert len(_rects(fig, tc.BOX_RISK)) == 6 and len(_rects(fig, tc.BOX_REWARD)) == 6    # boxes for all
    for name in (tc.BOX_RISK, tc.BOX_REWARD):
        (box,) = [tr for tr in fig.data if tr.name == name]
        assert len(box.hovertext) == 6 and all("R:R 1 : 1.00" in h for h in box.hovertext)   # hover on every one
    fig = tc.trade_figure(candles(n=200), t, "4h", focus=103)            # sig 70: entry bar 71, exit bar 76
    assert sorted(a.text for a in _notes(fig, "price tag")) == ["Entry 170.5", "SL 155", "TP 186"]
    assert [(pd.Timestamp(a.x), a.text) for a in _notes(fig, "time tag")] == [
        (_at(71), "Fri 12 20:00 entry"), (_at(76), "Sat 13 16:00 exit")]          # a month in view: dated
    assert len([s for s in fig.layout.shapes if s.name == "time tag"]) == 2
    assert len(_notes(fig, "trade label")) == 1
    assert not _notes(tc.trade_figure(candles(n=200), t, "4h", focus=999), "price tag")   # not in view


def test_tags_for_all_when_five_or_fewer():
    t = pd.DataFrame([_row(k) for k in range(5)] + [_row(20)])            # the sixth is out of the window
    fig = tc.trade_figure(candles(n=200), t, "4h")
    assert len(_notes(fig, "price tag")) == 15 and len(_notes(fig, "time tag")) == 10
    assert len(_notes(fig, "trade label")) == 5
    assert len([s for s in fig.layout.shapes if s.name == "time tag"]) == 10


def test_paper_run_draws_boxes():
    c = candles("2026-08-01", n=60).assign(live=[False] * 40 + [True] * 20)
    base = trade(sig=45, bars=3, start="2026-08-01", stop=140.0, target=160.0, exit_price=148.0)   # 145.5
    # the paper runner's record: slippage kept as 0, levels from the fill (one may be stamped before it)
    r = {**base, "qty": 2.0, "pnl_gross": 5.0, "slippage": 0.0, "r_multiple": 0.4, "source": "paper",
         "level_time": [base["entry_time"] - H4, base["entry_time"] + H4], "level_stop": [140.0, 142.0],
         "level_target": [160.0, 160.0]}
    fig = tc.trade_figure(c, pd.DataFrame([r]), "4h")
    e = pd.Timestamp(base["entry_time"], unit="ms")
    x = pd.Timestamp(base["exit_time"], unit="ms")
    h = pd.Timedelta(hours=4)
    assert _rects(fig, tc.BOX_RISK) == [(e, e + h, 145.5, 140.0), (e + h, x, 145.5, 142.0)]
    assert _rects(fig, tc.BOX_REWARD) == [(e, x, 145.5, 160.0)]
    (lab,) = _notes(fig, "trade label")
    assert "+0.45R gross · +0.40R net" in lab.text                    # 5 / (2 × 5.5)
    assert [a.text for a in fig.layout.annotations if not a.name] == ["warmup"]


# ---------------------------------------------------------------- dashboard

@pytest.fixture
def wf_run(cfg, monkeypatch):
    st.cache_data.clear()
    fake_run(cfg, "wf1", strategy="alpha", verdict="candidate")
    t = pd.DataFrame([trade(sig=s, start="2024-01-01", reason=r)
                      for s, r in ((300, "stop"), (420, "target"), (640, "signal"))])
    t.to_parquet(cfg.runs_dir / "wf1" / "oos_trades.parquet")
    monkeypatch.setenv("PERP_LAB_CONFIG", str(cfg.path))
    return cfg


def _detail(pick="alpha @ 4h"):
    at = AppTest.from_file("../dashboard/app.py", default_timeout=60)
    at.run()
    at.sidebar.radio(key="page").set_value("Strategy detail").run()
    at.selectbox(key="detail_pick").set_value(pick).run()
    return at


def _trade_chart(at):
    """The figure JSON of the trade chart (the only one with a candlestick)."""
    for el in at.get("plotly_chart"):
        spec = json.loads(el.proto.spec)
        if any(d.get("type") == "candlestick" for d in spec["data"]):
            return spec
    return None


def test_detail_page_charts_walkforward_trades(wf_run):
    write_candles(wf_run.data_dir, "BTCUSDT", "4h", candles("2024-01-01", n=6 * 130))
    at = _detail()
    assert not at.exception, [e.message for e in at.exception]
    spec = _trade_chart(at)
    assert spec is not None
    # default window = the last test fold (2024-03-31 → 2024-04-30): the trade at bar 640 is in it
    names = {d.get("name") for d in spec["data"]}
    assert {"entry", "exit: signal", tc.BOX_RISK} <= names and "exit: stop" not in names
    assert any("1 trade in range" in c.value for c in at.caption)


def test_jump_to_trade_moves_the_window(wf_run):
    write_candles(wf_run.data_dir, "BTCUSDT", "4h", candles("2024-01-01", n=6 * 130))
    at = _detail()
    at.selectbox(key="tc_jump_wf1_BTCUSDT").set_value(0).run()
    assert not at.exception, [e.message for e in at.exception]
    names = {d.get("name") for d in _trade_chart(at)["data"]}
    assert "exit: stop" in names and "exit: signal" not in names


def test_jump_to_the_same_trade_again_after_picking_dates_by_hand(wf_run):
    write_candles(wf_run.data_dir, "BTCUSDT", "4h", candles("2024-01-01", n=6 * 130))
    at = _detail()
    jump = "tc_jump_wf1_BTCUSDT"
    at.selectbox(key=jump).set_value(0).run()                       # trade 0 (bar 300) exits on its stop
    assert "exit: stop" in {d.get("name") for d in _trade_chart(at)["data"]}
    dates = f"tc_dates_wf1_{at.session_state['tc_ver_wf1']}"
    at.date_input(key=dates).set_value((pd.Timestamp("2024-04-01").date(), pd.Timestamp("2024-04-20").date())).run()
    assert not at.exception, [e.message for e in at.exception]
    assert "exit: stop" not in {d.get("name") for d in _trade_chart(at)["data"]}   # picked by hand: elsewhere
    assert at.selectbox(key=jump).value is None                     # no longer showing a jumped-to trade
    at.selectbox(key=jump).set_value(0).run()                       # the same trade again
    assert not at.exception, [e.message for e in at.exception]
    assert "exit: stop" in {d.get("name") for d in _trade_chart(at)["data"]}
    assert at.selectbox(key=jump).value == 0
    assert not [w.value for w in at.warning]                       # resetting the widget's state warns of nothing


def test_app_passes_the_jumped_to_trade_as_focus(wf_run):
    # eight trades close together: more than five in view, so only the jumped-to one gets its tags
    t = pd.DataFrame([trade(sig=300 + 7 * k, start="2024-01-01", target=500.0 + 7 * k) for k in range(8)])
    t.to_parquet(wf_run.runs_dir / "wf1" / "oos_trades.parquet")
    write_candles(wf_run.data_dir, "BTCUSDT", "4h", candles("2024-01-01", n=6 * 130))
    at = _detail()
    at.selectbox(key="tc_jump_wf1_BTCUSDT").set_value(3).run()
    assert not at.exception, [e.message for e in at.exception]
    notes = _trade_chart(at)["layout"]["annotations"]
    assert any("8 trades in range" in c.value for c in at.caption)
    assert [pd.Timestamp(a["x"]) for a in notes if a.get("name") == "time tag"] == [
        pd.Timestamp(t.entry_time[3], unit="ms"), pd.Timestamp(t.exit_time[3], unit="ms")]
    assert sorted(a["text"] for a in notes if a.get("name") == "price tag") == ["Entry 421.5", "SL 95", "TP 521"]
    # dates picked by hand: no trade is jumped to, so none is tagged
    dates = f"tc_dates_wf1_{at.session_state['tc_ver_wf1']}"
    at.date_input(key=dates).set_value((pd.Timestamp("2024-02-15").date(), pd.Timestamp("2024-03-05").date())).run()
    assert not at.exception, [e.message for e in at.exception]
    assert not [a for a in _trade_chart(at)["layout"].get("annotations", []) if a.get("name") == "time tag"]


def test_missing_candles_say_how_to_get_them(wf_run):
    at = _detail()
    assert not at.exception, [e.message for e in at.exception]
    assert _trade_chart(at) is None
    assert any("lab backfill" in w.value for w in at.warning)


def test_paper_runs_chart_their_own_bars(wf_run):
    d = wf_run.runs_dir / "pp1"
    d.mkdir(parents=True)
    candles("2026-08-01", n=60).assign(live=[False] * 40 + [True] * 20).reset_index(drop=True).to_parquet(
        d / "bars_BTCUSDT.parquet", index=False)
    pd.DataFrame([trade(sig=45, bars=3, start="2026-08-01")]).drop(columns=["fold", "combo"]).to_parquet(
        d / "trades.parquet")
    (d / "summary.json").write_text(json.dumps({
        "run_id": "pp1", "strategy": "alpha", "timeframes": ["4h"], "validation_mode": "paper",
        "parent_run_id": "wf1", "universe": ["BTCUSDT"], "paper_started_at": "2026-08-07T16:00:00+00:00",
        "created_at": "2026-08-07T16:00:00+00:00"}))
    at = _detail()
    assert not at.exception, [e.message for e in at.exception]
    charts = [json.loads(el.proto.spec) for el in at.get("plotly_chart")]
    paper = [s for s in charts if any(a.get("text") == "warmup" for a in s["layout"].get("annotations", []))]
    assert len(paper) == 1
    assert {"entry", tc.BOX_RISK} <= {d.get("name") for d in paper[0]["data"]}


def test_in_sample_runs_say_they_are_not_charted(wf_run):
    fake_run(wf_run, "is1", strategy="beta", mode="IS")
    at = _detail("beta @ 4h")
    assert not at.exception, [e.message for e in at.exception]
    assert any("Trade charts cover walk-forward and paper runs" in c.value for c in at.caption)


def test_detail_page_says_the_levels_moved_when_trades_have_them(wf_run):
    t = pd.DataFrame([{**trade(sig=s, start="2024-01-01"), **_lv(_ms("2024-01-01") + (s + 1) * H4, [0, 2],
                                                                  [95.0, 97.0], [np.nan, np.nan])}
                      for s in (300, 420, 640)])
    t.to_parquet(wf_run.runs_dir / "wf1" / "oos_trades.parquet")
    write_candles(wf_run.data_dir, "BTCUSDT", "4h", candles("2024-01-01", n=6 * 130))
    at = _detail()
    assert not at.exception, [e.message for e in at.exception]
    assert any("stop (red) and target (green), as they moved" in c.value for c in at.caption)
    spec = _trade_chart(at)
    assert {"stop"} <= {d.get("name") for d in spec["data"]}
