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
    assert tc.display_tf("15m", 5000, 6000) == "15m"
    assert tc.display_tf("15m", 20_000, 6000) == "1h"
    assert tc.display_tf("4h", 9000, 6000) == "1d"
    assert tc.display_tf("1m", 100, 6000) == "1m"            # a timeframe the coarser steps don't list: kept


def test_resample_keeps_ohlc():
    c = candles(n=12, tf_ms=3_600_000)                        # 12 hourly bars → 3 × 4h
    r = tc.resample(c, "4h")
    assert len(r) == 3
    first = c.iloc[:4]
    assert r.iloc[0].tolist() == [first.open.iloc[0], first.high.max(), first.low.min(), first.close.iloc[-1]]


# ---------------------------------------------------------------- hover

def test_received_funding_reads_as_received():
    (h,) = tc.hover_text(pd.DataFrame([trade(funding=-0.3)]))
    assert "funding received 0.30" in h
    (h,) = tc.hover_text(pd.DataFrame([trade(funding=0.0)]))
    assert "funding 0.00" in h


# ---------------------------------------------------------------- level path

def _lv(t0, offs, stops, targets):
    """Level columns for a trade whose entry is at t0 (epoch ms): one level per offset, in bars."""
    return dict(level_time=[t0 + o * H4 for o in offs], level_stop=stops, level_target=targets)


def _at(i):
    """Epoch ms of the i-th 4h bar from 2024-01-01."""
    return _ms("2024-01-01") + i * H4


def test_level_steps_follow_the_stop_and_target_as_they_moved():
    base = trade(sig=10, bars=6, stop=95.0)       # entry bar 11, exit bar 17
    r = {**base, **_lv(base["entry_time"], [0, 2, 4], [95.0, 97.0, 99.0], [120.0, np.nan, 130.0])}
    assert tc.level_steps(r, 1) == [(_at(11), _at(13), 95.0), (_at(13), _at(15), 97.0), (_at(15), _at(17), 99.0)]
    assert tc.level_steps(r, 2) == [(_at(11), _at(13), 120.0), (_at(15), _at(17), 130.0)]   # broken at the NaN
    # a level stamped before the entry (paper) is clamped: the line never steps backwards
    early = {**base, **_lv(base["entry_time"], [0, -2, 4], [95.0, 97.0, 99.0], [np.nan] * 3)}
    assert tc.level_steps(early, 1) == [(_at(11), _at(11), 95.0), (_at(11), _at(15), 97.0),
                                        (_at(15), _at(17), 99.0)]
    assert tc.level_steps(early, 2) == []


@pytest.mark.parametrize("empty", [np.nan, None])
def test_levels_read_back_from_parquet_are_arrays(tmp_path, empty):
    # a strategy that never sets a target: pyarrow stores the column as list<null>, which reads back as
    # object arrays of None
    base = trade(sig=10, bars=6, stop=95.0)
    t = pd.DataFrame([{**base, **_lv(base["entry_time"], [0, 3], [95.0, 98.0], [empty, empty])},
                      {**trade(sig=40, bars=4, stop=90.0), **_lv(_at(41), [0], [90.0], [empty])}])
    t.to_parquet(tmp_path / "t.parquet")
    back = pd.read_parquet(tmp_path / "t.parquet")
    assert isinstance(back.level_stop.iloc[0], np.ndarray) and tc.has_levels(back)
    if empty is None:
        assert pd.isna(back.level_target.iloc[0][0])
    a, b = back.to_dict("records")
    assert tc.level_steps(a, 1) == [(_at(11), _at(14), 95.0), (_at(14), _at(17), 98.0)]
    assert tc.level_steps(a, 2) == [] and tc.level_steps(b, 1) == [(_at(41), _at(45), 90.0)]
    assert tc.trade_boxes(a)["no_target"]
    assert "target at exit —" in tc.hover_text(back)[0]


def test_old_trades_without_levels_use_the_initial_stop_and_target():
    r = trade(sig=10, stop=95.0, target=120.0)                          # entry bar 11 at 110.5, exit bar 16
    assert not tc.has_levels(pd.DataFrame([r]))
    b = tc.trade_boxes(r)
    assert b["risk"] == [(_at(11), _at(16), 110.5, 95.0)] and b["reward"] == [(_at(11), _at(16), 110.5, 120.0)]
    assert tc.level_steps(r, 1) == [(_at(11), _at(16), 95.0)] and tc.level_steps(r, 2) == [(_at(11), _at(16), 120.0)]
    # a row whose levels are None or empty (a frame mixing old and new) is read the same way
    for empty in (None, []):
        m = {**trade(sig=10, stop=95.0), "level_time": empty, "level_stop": empty, "level_target": empty}
        assert not tc.has_levels(pd.DataFrame([m]))
        assert tc.level_steps(m, 1) == [(_at(11), _at(16), 95.0)] and tc.level_steps(m, 2) == []


def test_hover_gives_the_levels_at_exit():
    base = trade(sig=10, bars=6, stop=95.0)
    t = pd.DataFrame([{**base, **_lv(base["entry_time"], [0, 2], [95.0, 99.5], [np.nan, 130.0])},
                      {**base, **_lv(base["entry_time"], [0], [95.0], [np.nan])}])
    h = tc.hover_text(t)
    assert "stop at exit 99.5 · target at exit 130" in h[0]
    assert "stop at exit 95 · target at exit —" in h[1]
    assert "initial stop 95" in h[0]
    assert "at exit" not in tc.hover_text(pd.DataFrame([trade()]))[0]


# ---------------------------------------------------------------- position boxes (the drawing is tests/test_tv_chart.py)

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
    assert tc.result_text(_pos(slippage=0.0)) == "+1.00R gross · +0.73R net"                     # walk-forward, 0 kept
    # the paper runner fills at the ask/bid and writes slippage 0.0, so its gross is after slippage too
    assert tc.result_text(_pos(slippage=0.0, source="paper")) == "+1.00R gross after slippage · +0.73R net"


def test_no_target_trade_has_no_reward_box_and_says_so():
    win = trade(sig=10, bars=6, stop=95.0, exit_price=115.0, reason="signal")          # long 110.5 → 115
    b = tc.trade_boxes(win)
    assert b["no_target"] and b["reward"] == [] and b["risk"] == [(_at(11), _at(17), 110.5, 95.0)]
    assert b["entry"] == (_at(11), 110.5) and b["exit"] == (_at(17), 115.0, "signal")
    assert tc.rr_text(win) == "no fixed target"


def test_breakeven_collapses_the_risk_box_at_its_time():
    r = _pos(level_time=[T0, T0 + 3 * M5], level_stop=[98.82, 99.06], level_target=[99.30, 99.30])
    b = tc.trade_boxes(r)
    assert b["be_time"] == T0 + 3 * M5
    assert b["risk"] == [(T0, T0 + 3 * M5, 99.06, 98.82)]               # nothing after the move
    assert b["reward"] == [(T0, T0 + 6 * M5, 99.06, 99.30)]             # the target never moved: one box
    assert b["entry"] == (T0, 99.06) and b["exit"] == (T0 + 6 * M5, 99.30, "target") and not b["no_target"]
    assert tc.trade_boxes(_pos())["be_time"] is None


def test_trailing_stop_past_entry_collapses_risk_and_keeps_the_step_line():
    base = trade(sig=10, bars=6, stop=105.0, exit_price=113.0)          # long 110.5, entry bar 11, exit bar 17
    r = {**base, **_lv(base["entry_time"], [0, 2, 4], [105.0, 108.0, 112.0], [np.nan] * 3)}
    b = tc.trade_boxes(r)
    assert b["risk"] == [(_at(11), _at(13), 110.5, 105.0), (_at(13), _at(15), 110.5, 108.0)]   # steps, then gone
    assert b["be_time"] == _at(15)                                       # trailed past the entry
    assert [v for _, _, v in tc.level_steps(r, 1)] == [105.0, 108.0, 112.0]                     # the line carries on


def test_short_trade_boxes_are_mirrored():
    long_, short = tc.trade_boxes(_pos()), tc.trade_boxes(_pos(-1))
    end = T0 + 6 * M5
    assert long_["reward"] == [(T0, end, 99.06, 99.30)] and long_["risk"] == [(T0, end, 99.06, 98.82)]
    assert short["reward"] == [(T0, end, 99.06, 98.82)] and short["risk"] == [(T0, end, 99.06, 99.30)]
    assert tc.rr_text(_pos(-1)) == tc.rr_text(_pos()) and tc.result_text(_pos(-1)) == tc.result_text(_pos())


def test_target_exit_is_the_reward_green_not_the_entry_blue():
    assert tc.exit_colors()["target"] == tc.PALETTE["pos"] != tc.PALETTE["s1"]
    assert tc.exit_colors({"pos": "#00ff00"})["target"] == "#00ff00"


def test_trade_tags_for_one_trade():
    r = _pos(level_time=[T0, T0 + 3 * M5], level_stop=[98.82, 99.06], level_target=[99.30, 99.30])
    b = tc.trade_boxes(r)
    day = (T0 - 164 * M5, T0 + 124 * M5)                                 # 2024-03-04 00:00 → 03-05 00:00
    g = tc.trade_tags(r, b, tc.PALETTE, *day, "%H:%M")
    assert g["price"] == [("Entry", 99.06, tc.PALETTE["s1"]), ("TP", 99.30, tc.PALETTE["pos"]),
                          ("SL", 98.82, tc.PALETTE["neg"])]
    assert g["time"] == [(T0, "13:40 entry", tc.PALETTE["s1"], "entry"),
                         (T0 + 6 * M5, "14:10 exit", tc.exit_colors()["target"], "exit")]
    assert g["be"] == (T0 + 3 * M5, 99.06, "SL → BE 13:55")
    g = tc.trade_tags(r, b, tc.PALETTE, T0 + M5, T0 + 6 * M5, "%H:%M")   # entry before, exit at the window's end
    assert g["time"] == [] and g["be"] == (T0 + 3 * M5, 99.06, "SL → BE 13:55")


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


def _charts(at) -> list[dict]:
    """The payloads the trade chart component was given, in page order."""
    return [json.loads(el.proto.json_args)["payload"] for el in at.get("component_instance")
            if el.proto.component_name.endswith("tv_chart")]


def _trade_chart(at):
    """The payload of the page's first trade chart, or None when it drew none."""
    charts = _charts(at)
    return charts[0] if charts else None


def _reasons(p) -> set:
    return {tr["exit"]["reason"] for tr in p["trades"]}


def _tagged(p) -> list:
    return [tr for tr in p["trades"] if tr["price_lines"] or tr["time_tags"]]


def _click(at, trade_id, chart=0):
    """A click on a trade's box, as the browser reports it: the component's value, sent with the page's other
    widget values. AppTest has no setter for a custom component, so this sends the value the way the browser
    does (a widget state with the component's id)."""
    from streamlit.proto.WidgetStates_pb2 import WidgetStates
    el = [e for e in at.get("component_instance") if e.proto.component_name.endswith("tv_chart")][chart]
    ws = WidgetStates()
    ws.CopyFrom(at._tree.get_widget_states())
    w = ws.widgets.add()
    w.id = el.proto.id
    w.json_value = json.dumps({"clicked": trade_id, "at": 1})
    return at._run(ws)


def test_detail_page_charts_walkforward_trades(wf_run):
    write_candles(wf_run.data_dir, "BTCUSDT", "4h", candles("2024-01-01", n=6 * 130))
    at = _detail()
    assert not at.exception, [e.message for e in at.exception]
    p = _trade_chart(at)
    assert p is not None and p["candles"] and p["title"].startswith("BTCUSDT 4h · ")
    # default window = the last test fold (2024-03-31 → 2024-04-30): the trade at bar 640 is in it
    assert _reasons(p) == {"signal"} and p["trades"][0]["boxes"]["risk"]
    assert [f["label"] for f in p["folds"]] == ["fold 3"]          # the run's last test fold, shaded and labelled
    assert any("1 trade in range" in c.value for c in at.caption)
    assert any("drag to pan" in c.value and "click a box" in c.value for c in at.caption)


def test_jump_to_trade_moves_the_window(wf_run):
    write_candles(wf_run.data_dir, "BTCUSDT", "4h", candles("2024-01-01", n=6 * 130))
    at = _detail()
    at.selectbox(key="tc_jump_wf1_BTCUSDT").set_value(0).run()
    assert not at.exception, [e.message for e in at.exception]
    p = _trade_chart(at)
    assert "stop" in _reasons(p) and "signal" not in _reasons(p) and p["focus"] == 0


def test_jump_to_the_same_trade_again_after_picking_dates_by_hand(wf_run):
    write_candles(wf_run.data_dir, "BTCUSDT", "4h", candles("2024-01-01", n=6 * 130))
    at = _detail()
    jump = "tc_jump_wf1_BTCUSDT"
    at.selectbox(key=jump).set_value(0).run()                       # trade 0 (bar 300) exits on its stop
    assert "stop" in _reasons(_trade_chart(at))
    dates = f"tc_dates_wf1_{at.session_state['tc_ver_wf1']}"
    at.date_input(key=dates).set_value((pd.Timestamp("2024-04-01").date(), pd.Timestamp("2024-04-20").date())).run()
    assert not at.exception, [e.message for e in at.exception]
    assert "stop" not in _reasons(_trade_chart(at))                 # picked by hand: elsewhere
    assert at.selectbox(key=jump).value is None                     # no longer showing a jumped-to trade
    at.selectbox(key=jump).set_value(0).run()                       # the same trade again
    assert not at.exception, [e.message for e in at.exception]
    assert "stop" in _reasons(_trade_chart(at))
    assert at.selectbox(key=jump).value == 0
    assert not [w.value for w in at.warning]                       # resetting the widget's state warns of nothing


def _eight(wf_run):
    """Eight trades close together: more than five in view, so only the focused one gets its tags."""
    t = pd.DataFrame([trade(sig=300 + 7 * k, start="2024-01-01", target=500.0 + 7 * k) for k in range(8)])
    t.to_parquet(wf_run.runs_dir / "wf1" / "oos_trades.parquet")
    write_candles(wf_run.data_dir, "BTCUSDT", "4h", candles("2024-01-01", n=6 * 130))
    return t


def test_app_passes_the_jumped_to_trade_as_focus(wf_run):
    t = _eight(wf_run)
    at = _detail()
    at.selectbox(key="tc_jump_wf1_BTCUSDT").set_value(3).run()
    assert not at.exception, [e.message for e in at.exception]
    p = _trade_chart(at)
    assert any("8 trades in range" in c.value for c in at.caption) and len(p["trades"]) == 8
    assert p["focus"] == 3
    (tr,) = _tagged(p)
    assert tr["id"] == 3 and [x["time"] for x in tr["time_tags"]] == [t.entry_time[3], t.exit_time[3]]
    assert sorted(f"{x['title']} {x['price']:g}" for x in tr["price_lines"]) == ["Entry 421.5", "SL 95", "TP 521"]
    # dates picked by hand: no trade is jumped to, so none is tagged
    dates = f"tc_dates_wf1_{at.session_state['tc_ver_wf1']}"
    at.date_input(key=dates).set_value((pd.Timestamp("2024-02-15").date(), pd.Timestamp("2024-03-05").date())).run()
    assert not at.exception, [e.message for e in at.exception]
    p = _trade_chart(at)
    assert p["focus"] is None and not _tagged(p)


def test_a_click_on_a_box_focuses_the_trade_as_jump_to_trade_does(wf_run):
    _eight(wf_run)
    at = _detail()
    jump = "tc_jump_wf1_BTCUSDT"
    at.selectbox(key=jump).set_value(3).run()
    window = at.session_state["tc_win_wf1"]
    _click(at, 5)
    assert not at.exception, [e.message for e in at.exception]
    assert at.selectbox(key=jump).value == 5                        # the same state as Jump to trade
    p = _trade_chart(at)
    assert p["focus"] == 5 and [tr["id"] for tr in _tagged(p)] == [5]
    assert at.session_state["tc_win_wf1"] == window                 # the dates stay: the trade is in view
    assert not [w.value for w in at.warning]
    _click(at, 5)                                                   # the same box again: still focused, no error
    assert not at.exception and at.selectbox(key=jump).value == 5


def test_the_chart_payload_is_built_once_per_window_and_focus(wf_run, monkeypatch):
    from dashboard import tv_chart as tv
    built = []
    real = tv.chart_payload
    monkeypatch.setattr(tv, "chart_payload", lambda *a, **k: built.append(k.get("focus")) or real(*a, **k))
    write_candles(wf_run.data_dir, "BTCUSDT", "4h", candles("2024-01-01", n=6 * 130))
    at = _detail()
    assert built == [None] and _trade_chart(at)["trades"]
    at.run()                                                        # a rerun for anything else: not rebuilt
    assert built == [None] and _trade_chart(at)["trades"]
    at.selectbox(key="tc_jump_wf1_BTCUSDT").set_value(0).run()      # a new window and focus: built again
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert built == [None, 0] and _trade_chart(at)["focus"] == 0


def test_every_dashboard_chart_hangs_its_legend_below_the_title(wf_run):
    write_candles(wf_run.data_dir, "BTCUSDT", "4h", candles("2024-01-01", n=6 * 130))
    at = _detail()
    assert not at.exception, [e.message for e in at.exception]
    charts = [json.loads(el.proto.spec) for el in at.get("plotly_chart")]
    titled = [c["layout"] for c in charts if c["layout"].get("title", {}).get("text")]
    assert len(titled) == len(charts) > 1
    for lay in titled:
        title, legend, h = lay["title"], lay["legend"], lay["height"]
        assert tuple(title.get(k) for k in ("yref", "y", "yanchor", "x", "xanchor")) == (
            "container", 1, "top", 0, "left")
        band = title["pad"]["t"] + 1.5 * title["font"]["size"]          # the title's line, in px from the top
        top = (1 - legend["y"]) * h                                      # where the legend's first row starts
        assert tuple(legend.get(k) for k in ("yref", "yanchor", "orientation")) == ("container", "top", "h")
        assert band <= top < band + 8, title["text"]                    # just below the title
        # one row fits the set margin; plotly reserves the margin down to the last row of a wrapped legend
        assert lay["margin"]["t"] >= top + 24 and lay["margin"].get("autoexpand", True)


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
    paper = [p for p in _charts(at) if any(f["label"] == "warmup" for f in p["folds"])]
    assert len(paper) == 1
    (tr,) = paper[0]["trades"]
    assert tr["boxes"]["risk"] and [m["kind"] for m in paper[0]["markers"]] == ["entry", "exit"]


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
    (tr,) = _trade_chart(at)["trades"]
    assert [v for _, _, v in tr["stops"]] == [95.0, 97.0]


def test_trades_in_is_half_open_a_trade_closing_at_the_window_start_is_out():
    t = pd.DataFrame([trade(sig=10, bars=5), trade(sig=30, bars=5), trade(sig=50, bars=5)])
    start, end = pd.Timestamp(int(t.exit_time[0]), unit="ms", tz="UTC"), pd.Timestamp(int(t.entry_time[2]), unit="ms", tz="UTC")
    got = tc.trades_in(t, start, end)
    assert list(got.index) == [1]                       # 0 closes at the start; 2 opens at the end
    assert list(tc.trades_in(t, start - pd.Timedelta(1, "ms"), end).index) == [0, 1]
