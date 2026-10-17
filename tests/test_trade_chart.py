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


def test_figure_draws_entries_signal_bar_exits_stops_targets_and_folds():
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
    long_, short = _traces(fig, "long entry")[0], _traces(fig, "short entry")[0]
    assert long_.marker.symbol == "triangle-up" and short.marker.symbol == "triangle-down"
    assert len(long_.x) == 4 and len(short.x) == 1
    assert pd.Timestamp(short.x[0]) == pd.to_datetime(t.entry_time[1], unit="ms")    # UTC, zone dropped
    colors = tc.exit_colors()
    for reason in tc.EXIT_REASONS:
        (ex,) = _traces(fig, f"exit: {reason}")
        assert ex.marker.color == colors[reason] and len(ex.x) == 1
    assert len(set(colors.values())) == len(colors)
    (sig,) = _traces(fig, "signal bar (close)")
    assert list(sig.y) == [c.close.iloc[i] for i in (10, 30, 50, 70, 90)]     # the signal bar's own close
    (stop,) = _traces(fig, "initial stop")
    assert [y for y in stop.y if y is not None] == [95.0, 95.0, 140.0, 140.0] + [95.0] * 6
    (tgt,) = _traces(fig, "initial target")
    assert [y for y in tgt.y if y is not None] == [120.0, 120.0]               # NaN targets draw nothing
    hov = long_.hovertext[0]
    for s in ("P&L -12.50 USDT", "R -0.75", "regime at signal: trending", "fees 1.25", "funding paid 0.40",
              "fold 1 · n=20", "initial stop 95", "initial target —"):
        assert s in hov, s
    labels = [a.text for a in fig.layout.annotations]
    assert labels == ["train only", "fold 0", "fold 1"]
    assert "trained on 2023-09-01 → 2024-01-20" in fig.layout.annotations[2].hovertext
    assert len(fig.layout.shapes) == 2                       # train-only region + fold 0 (folds alternate)


def test_coarse_candles_keep_markers_at_the_exact_fill():
    c = candles(n=7000)
    t = pd.DataFrame([trade(sig=6500, bars=3)])
    fig = tc.trade_figure(c, t, "4h")
    (cs,) = [x for x in fig.data if x.type == "candlestick"]
    assert len(cs.x) <= tc.MAX_CANDLES and cs.name == "1d candles"
    assert pd.Timestamp(_traces(fig, "long entry")[0].x[0]) == pd.to_datetime(t.entry_time[0], unit="ms")
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
    assert tgt.x[-1] is None and tgt.line.color == tc.PALETTE["s1"]
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


def test_old_trades_keep_the_initial_lines():
    t = pd.DataFrame([trade(sig=10, stop=95.0, target=120.0)])
    assert not tc.has_levels(t)
    fig = tc.trade_figure(candles(n=200), t, "4h")
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
    assert {"long entry", "exit: signal"} <= names and "exit: stop" not in names
    assert any("1 trade in range" in c.value for c in at.caption)


def test_jump_to_trade_moves_the_window(wf_run):
    write_candles(wf_run.data_dir, "BTCUSDT", "4h", candles("2024-01-01", n=6 * 130))
    at = _detail()
    at.selectbox(key="tc_jump_wf1_BTCUSDT").set_value(0).run()
    assert not at.exception, [e.message for e in at.exception]
    names = {d.get("name") for d in _trade_chart(at)["data"]}
    assert "exit: stop" in names and "exit: signal" not in names


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
    assert "long entry" in {d.get("name") for d in paper[0]["data"]}


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
    assert any("stop (red) and target (blue), as they moved" in c.value for c in at.caption)
    spec = _trade_chart(at)
    assert {"stop"} <= {d.get("name") for d in spec["data"]}
