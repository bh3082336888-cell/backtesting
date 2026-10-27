"""The trade chart on TradingView Lightweight Charts: the payload the component draws, and the vendored library.

Every number the chart draws is in the payload, and every trade number in it comes from the trade record through
`trade_chart` (`trade_boxes`, `rr_text`, `result_text`, `hover_text`). The browser only maps times and prices to
pixels (geometry.test.mjs covers that)."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from dashboard import trade_chart as tc
from dashboard import tv_chart as tv
from tests.test_trade_chart import H4, M5, T0, _day, _lv, _ms, _pos, candles, trade

P = tc.PALETTE


def payload(c, t, tf="4h", **kw) -> dict:
    """A payload as the dashboard builds it, checked to be strict JSON as Streamlit sends it (no NaN, tuples or
    numpy scalars)."""
    p = tv.chart_payload(c, pd.DataFrame(t) if isinstance(t, list) else t, tf, palette=P, **kw)
    assert json.loads(json.dumps(p, allow_nan=False)) == p
    return p


def ms(i: int) -> int:
    """Epoch ms of the i-th 4h bar of `candles()`."""
    return _ms("2024-01-01") + i * H4


def by_id(p: dict) -> dict:
    return {tr["id"]: tr for tr in p["trades"]}


def tagged(p: dict) -> set:
    return {tr["id"] for tr in p["trades"] if tr["tagged"] or tr["price_lines"] or tr["time_tags"]}


# ---------------------------------------------------------------- the window

def test_payload_is_json_and_holds_the_window():
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
    p = payload(c, t, folds=folds, title="BTCUSDT 4h")

    assert (p["tf"], p["shown"], p["bar_ms"], p["title"]) == ("4h", "4h", H4, "BTCUSDT 4h")
    assert p["window"] == [ms(0), ms(200)]                                 # [first bar, last bar + tf)
    assert len(p["candles"]) == 200
    assert p["candles"][0] == {"time": ms(0) // 1000, "open": 99.5, "high": 102.0, "low": 98.0, "close": 100.0}
    assert p["candles"][-1]["time"] == ms(199) // 1000                     # UTC seconds
    assert [tr["id"] for tr in p["trades"]] == [0, 1, 2, 3, 4]             # the one outside is left out
    assert p["layers"] == {} and p["focus"] is None and p["palette"] == P

    short = p["trades"][1]
    assert short["side"] == -1
    assert short["entry"] == {"time": ms(31), "price": 130.5}
    assert short["exit"] == {"time": ms(36), "price": 135.0, "reason": "target", "color": P["s1"]}
    colors = tc.exit_colors(P)
    assert [tr["exit"]["color"] for tr in p["trades"]] == [colors[r] for r in tc.EXIT_REASONS]
    assert p["exits"] == colors                                            # the key above the chart

    # dots on the entry and exit candles at the fill and exit prices, in time order, as series markers
    dots = p["markers"]
    assert [m["time"] for m in dots] == sorted(m["time"] for m in dots) and len(dots) == 10
    assert [(m["time"], m["price"], m["color"], m["id"]) for m in dots if m["kind"] == "entry"] == [
        (ms(s + 1) // 1000, 100.0 + s + 0.5, P["s1"], k) for k, s in enumerate((10, 30, 50, 70, 90))]
    assert [(m["price"], m["color"]) for m in dots if m["kind"] == "exit"] == [
        (100.0 + s + 5, colors[r]) for s, r in zip((10, 30, 50, 70, 90), tc.EXIT_REASONS)]

    # the signal bar's own close; the stop and target as they stood (no target: no line)
    assert [tr["signal"] for tr in p["trades"]] == [{"time": ms(s), "price": 100.0 + s} for s in (10, 30, 50, 70, 90)]
    assert short["stops"] == [[ms(31), ms(36), 140.0]] and short["targets"] == [[ms(31), ms(36), 120.0]]
    assert p["trades"][0]["stops"] == [[ms(11), ms(16), 95.0]] and p["trades"][0]["targets"] == []

    # the hover: today's tooltip as plain lines, the title first
    tip = p["trades"][0]["tooltip"]
    assert tip[0] == "LONG BTCUSDT · exit stop" and not any("<" in line for line in tip)
    for s in ("P&L -12.50 USDT", "R -0.75", "regime at signal: trending", "fees 1.25", "funding paid 0.40",
              "fold 1 · n=20", "initial stop 95", "initial target —", "no fixed target"):
        assert any(s in line for line in tip), s

    # fold shading: the train-only stretch, then every other test fold, each labelled with its hover
    assert [(f["label"], f["fill"]) for f in p["folds"]] == [
        ("train only", P["train"]), ("fold 0", P["band"]), ("fold 1", None)]
    assert [(f["from"], f["to"]) for f in p["folds"]] == [
        (ms(0), _ms("2024-01-10")), (_ms("2024-01-10"), _ms("2024-01-20")), (_ms("2024-01-20"), ms(200))]
    assert "trained on 2023-09-01 → 2024-01-20" in p["folds"][2]["hover"]
    assert "params n=20" in p["folds"][2]["hover"]


def test_payload_boxes_come_from_trade_boxes():
    rows = [
        _pos(level_time=[T0, T0 + 3 * M5], level_stop=[98.82, 99.06], level_target=[99.30, 99.30]),   # to BE
        _pos(-1, entry_time=T0 + 12 * M5, exit_time=T0 + 18 * M5),                                     # short
        _pos(target=np.nan, exit_price=99.2, exit_reason="signal", entry_time=T0 + 24 * M5,
             exit_time=T0 + 30 * M5),                                                                   # no target
        _pos(entry_time=T0 + 40 * M5, exit_time=T0 + 40 * M5),                                          # one bar
    ]
    p = payload(_day(), rows, "5m")
    assert len(p["trades"]) == 4
    for r, tr in zip(rows, p["trades"]):
        assert tr["boxes"] == json.loads(json.dumps(tc.trade_boxes(r, M5)))
        assert (tr["rr"], tr["result"]) == (tc.rr_text(r), tc.result_text(r))
    be, short, free, one = p["trades"]
    assert be["boxes"]["risk"] == [[T0, T0 + 3 * M5, 99.06, 98.82]] and be["boxes"]["be_time"] == T0 + 3 * M5
    assert be["boxes"]["reward"] == [[T0, T0 + 6 * M5, 99.06, 99.30]]
    assert short["boxes"]["reward"] == [[T0 + 12 * M5, T0 + 18 * M5, 99.06, 98.82]]          # below the entry
    assert free["boxes"]["no_target"] and free["boxes"]["reward"] == [] and free["rr"] == "no fixed target"
    assert one["boxes"]["span"] == [T0 + 40 * M5, T0 + 41 * M5]          # one bar wide, else it draws nothing
    assert (be["rr"], be["result"]) == ("R:R 1 : 1.00", "+1.03R gross · +0.73R net")
    assert be["tone"] == P["pos"]                                         # the result line's colour: net R > 0
    assert payload(_day(), [_pos(r_multiple=-1.31)], "5m")["trades"][0]["tone"] == P["neg"]


# ---------------------------------------------------------------- tags: the focused trade, or all when ≤ 5

def _row(k, **kw):
    return trade(sig=10 + 20 * k, bars=5, stop=95.0 + 20 * k, target=126.0 + 20 * k, **kw)   # R:R 1 : 1


def test_price_lines_only_for_focus_or_five_or_fewer():
    t = pd.DataFrame([_row(k) for k in range(6)])
    t.index = t.index + 100                                               # focus is a row label, not a position
    p = payload(candles(n=200), t)
    assert len(p["trades"]) == 6 and not tagged(p) and p["focus"] is None
    assert all(tr["be"] is None for tr in p["trades"])
    assert all(any("R:R 1 : 1.00" in line for line in tr["tooltip"]) for tr in p["trades"])   # hover on every one

    p = payload(candles(n=200), t, focus=103)                            # sig 70: entry bar 71, exit bar 76
    assert p["focus"] == 103 and tagged(p) == {103}
    tr = by_id(p)[103]
    assert tr["tagged"]
    assert tr["price_lines"] == [{"price": 170.5, "title": "Entry", "color": P["s1"]},
                                 {"price": 186.0, "title": "TP", "color": P["pos"]},
                                 {"price": 155.0, "title": "SL", "color": P["neg"]}]
    assert tr["time_tags"] == [                                           # a month in view: dated, in UTC
        {"time": ms(71), "text": "Fri 12 20:00 entry", "color": P["s1"], "side": "entry"},
        {"time": ms(76), "text": "Sat 13 16:00 exit", "color": tc.exit_colors(P)["stop"], "side": "exit"}]

    p = payload(candles(n=200), t, focus=999)                            # not in view: nothing tagged
    assert p["focus"] is None and not tagged(p)

    p = payload(candles(n=200), pd.DataFrame([_row(k) for k in range(5)] + [_row(20)]))   # the sixth is outside
    assert len(p["trades"]) == 5 and tagged(p) == {0, 1, 2, 3, 4}
    assert sum(len(tr["price_lines"]) for tr in p["trades"]) == 15
    assert sum(len(tr["time_tags"]) for tr in p["trades"]) == 10


def test_tags_mirror_for_a_short_and_have_no_tp_without_a_target():
    (s,) = payload(_day(), [_pos(-1)], "5m")["trades"]
    assert [(x["title"], x["price"]) for x in s["price_lines"]] == [("Entry", 99.06), ("TP", 98.82), ("SL", 99.30)]
    free = trade(sig=10, bars=6, stop=95.0, exit_price=115.0, reason="signal")
    (f,) = payload(candles(n=200), [free])["trades"]
    assert [x["title"] for x in f["price_lines"]] == ["Entry", "SL"]


def test_breakeven_note_and_one_day_times():
    r = _pos(level_time=[T0, T0 + 3 * M5], level_stop=[98.82, 99.06], level_target=[99.30, 99.30])
    (tr,) = payload(_day(), [r], "5m")["trades"]
    assert tr["be"] == {"time": T0 + 3 * M5, "price": 99.06, "text": "SL → BE 13:55"}
    assert [x["text"] for x in tr["time_tags"]] == ["13:40 entry", "14:10 exit"]       # one day: no date


def test_a_trade_across_the_window_edge_keeps_its_record_and_drops_what_is_off_screen():
    early = trade(sig=-6, bars=10, stop=90.0, target=130.0)               # entry bar -5, exit bar 5
    late = trade(sig=195, bars=10, stop=280.0, target=320.0)              # entry bar 196, exit bar 206
    a, b = payload(candles(n=200), [early, late])["trades"]
    assert a["boxes"]["risk"] == [[ms(-5), ms(5), 94.5, 90.0]]            # whole: the chart clips it to the window
    p = payload(candles(n=200), [early, late])
    assert [(m["kind"], m["time"]) for m in p["markers"]] == [("exit", ms(5) // 1000), ("entry", ms(196) // 1000)]
    assert [(x["side"], x["time"]) for tr in p["trades"] for x in tr["time_tags"]] == [
        ("exit", ms(5)), ("entry", ms(196))]                              # no tag for an off-screen time
    assert a["signal"] is None and b["signal"] == {"time": ms(195), "price": 295.0}   # its bar is in view or not


# ---------------------------------------------------------------- level paths

def test_stop_and_target_lines_follow_the_level_path():
    base = trade(sig=10, bars=6, stop=95.0)                               # entry bar 11, exit bar 17
    trail = {**base, **_lv(base["entry_time"], [0, 2, 4], [95.0, 97.0, 99.0], [120.0, np.nan, 130.0])}
    early = {**base, **_lv(base["entry_time"], [0, -2, 4], [95.0, 97.0, 99.0], [np.nan] * 3)}
    old = trade(sig=40, stop=90.0, target=150.0)                          # a run from before level paths
    a, b, o = payload(candles(n=200), [trail, early, old])["trades"]
    assert a["stops"] == [[ms(11), ms(13), 95.0], [ms(13), ms(15), 97.0], [ms(15), ms(17), 99.0]]
    assert a["targets"] == [[ms(11), ms(13), 120.0], [ms(15), ms(17), 130.0]]          # broken across the NaN
    assert b["stops"] == [[ms(11), ms(11), 95.0], [ms(11), ms(15), 97.0], [ms(15), ms(17), 99.0]]   # never back
    assert o["stops"] == [[ms(41), ms(46), 90.0]] and o["targets"] == [[ms(41), ms(46), 150.0]]
    assert o["boxes"]["risk"] == [[ms(41), ms(46), 140.5, 90.0]] and o["boxes"]["reward"] == [[ms(41), ms(46), 140.5,
                                                                                              150.0]]


@pytest.mark.parametrize("empty", [np.nan, None])
def test_level_paths_read_back_from_parquet(tmp_path, empty):
    # parquet hands list columns back as arrays; a strategy that never sets a target stores list<null>
    base = trade(sig=10, bars=6, stop=95.0)
    t = pd.DataFrame([{**base, **_lv(base["entry_time"], [0, 3], [95.0, 98.0], [empty, empty])},
                      {**trade(sig=40, bars=4, stop=90.0), **_lv(ms(41), [0], [90.0], [empty])}])
    t.to_parquet(tmp_path / "t.parquet")
    back = pd.read_parquet(tmp_path / "t.parquet")
    assert isinstance(back.level_stop.iloc[0], np.ndarray)
    a, b = payload(candles(n=200), back)["trades"]
    assert a["stops"] == [[ms(11), ms(14), 95.0], [ms(14), ms(17), 98.0]] and a["targets"] == []
    assert b["stops"] == [[ms(41), ms(45), 90.0]] and b["boxes"]["no_target"]
    assert any("target at exit —" in line for line in a["tooltip"])


# ---------------------------------------------------------------- paper runs, density, layers

def test_paper_warmup_is_shaded_and_paper_trades_are_boxed():
    c = candles("2026-08-01", n=60).assign(live=[False] * 40 + [True] * 20)
    base = trade(sig=45, bars=3, start="2026-08-01", stop=140.0, target=160.0, exit_price=148.0)   # 145.5
    # the paper runner's record: slippage kept as 0, levels from the fill (one may be stamped before it)
    r = {**base, "qty": 2.0, "pnl_gross": 5.0, "slippage": 0.0, "r_multiple": 0.4, "source": "paper",
         "level_time": [base["entry_time"] - H4, base["entry_time"] + H4], "level_stop": [140.0, 142.0],
         "level_target": [160.0, 160.0]}
    p = payload(c, [r])
    assert p["folds"] == [{"from": _ms("2026-08-01"), "to": _ms("2026-08-01") + 40 * H4, "fill": P["train"],
                           "label": "warmup", "hover": "bars fetched from the exchange before the paper run started"}]
    (tr,) = p["trades"]
    e, x = base["entry_time"], base["exit_time"]
    assert tr["boxes"]["risk"] == [[e, e + H4, 145.5, 140.0], [e + H4, x, 145.5, 142.0]]
    assert tr["boxes"]["reward"] == [[e, x, 145.5, 160.0]]
    assert tr["result"] == "+0.45R gross · +0.40R net"                    # 5 / (2 × 5.5)
    assert payload(candles(n=30), [])["folds"] == []                      # walk-forward bars carry no warmup


def test_payload_coarsens_only_past_50000_bars():
    assert tv.MAX_TV_BARS == 50_000
    p = payload(candles("2024-01-01", n=50_000, tf_ms=M5), [], "5m")
    assert p["shown"] == "5m" and len(p["candles"]) == 50_000 and p["bar_ms"] == M5

    c = candles("2024-01-01", n=50_001, tf_ms=M5)
    m5 = lambda i: _ms("2024-01-01") + i * M5                             # noqa: E731
    r = trade(sig=40_000, bars=3, stop=40_090.0, target=40_120.0, tf_ms=M5)   # entry bar 40,001 at 40,100.5
    p = payload(c, [r], "5m")
    assert p["shown"] == "15m" and len(p["candles"]) == 16_667 and p["bar_ms"] == 3 * M5
    (tr,) = p["trades"]
    assert tr["boxes"]["risk"] == [[m5(40_001), m5(40_004), 40_100.5, 40_090.0]]      # exact, whatever the candles
    assert tr["entry"] == {"time": m5(40_001), "price": 40_100.5}
    assert [(m["kind"], m["time"], m["price"]) for m in p["markers"]] == [
        ("entry", m5(39_999) // 1000, 40_100.5), ("exit", m5(40_002) // 1000, 40_103.0)]   # on the 15m candles
    assert tr["signal"] == {"time": m5(40_000), "price": 40_100.0}       # the run's own 5m bar, not the 15m one
    assert [x["time"] for x in tr["time_tags"]] == [m5(40_001), m5(40_004)]


def test_no_overlay_means_empty_layers():
    assert payload(candles(n=50), [], overlay=None, layers=frozenset({"zones", "cloud5", "profile"}))["layers"] == {}
    assert payload(candles(n=50), [])["layers"] == {}


def test_price_format_follows_the_price():
    assert payload(candles(n=10, price=60_000.0), [])["price_format"] == {"type": "price", "precision": 2,
                                                                         "minMove": 0.01}
    small = candles(n=10)
    small[["open", "high", "low", "close"]] *= 0.001                       # about 0.1, a DOGE-sized price
    assert payload(small, [])["price_format"] == {"type": "price", "precision": 6, "minMove": 1e-6}


def test_empty_window_is_an_empty_chart():
    p = payload(candles(n=0), [trade()])
    assert p["candles"] == [] and p["trades"] == [] and p["markers"] == [] and p["window"] is None


# ---------------------------------------------------------------- the component

def test_component_files_and_license_are_vendored():
    d = tv.COMPONENT
    lib = (d / "vendor" / "lightweight-charts.standalone.production.js").read_text(encoding="utf-8")
    assert "TradingView Lightweight Charts™ v5.2.1" in lib[:300] and "Apache License 2.0" in lib[:300]
    assert 'version:function(){return"5.2.1"}' in lib
    assert "Apache License" in (d / "vendor" / "LICENSE").read_text(encoding="utf-8")
    assert "TradingView, Inc." in (d / "vendor" / "NOTICE").read_text(encoding="utf-8")

    html = (d / "index.html").read_text(encoding="utf-8")
    srcs = re.findall(r"<script\b[^>]*\bsrc=\"([^\"]+)\"", html)
    assert srcs == ["vendor/lightweight-charts.standalone.production.js", "main.js"]     # the vendored build only
    assert not re.search(r"https?:|//[\w.-]+\.\w", html) and "<link" not in html        # nothing remote
    for name in ("main.js", "primitives.js", "geometry.mjs"):
        js = (d / name).read_text(encoding="utf-8")
        assert not re.search(r"https?://", js), name
        assert all(m.startswith("./") for m in re.findall(r"\bfrom\s+\"([^\"]+)\"", js)), name   # local modules
        assert not re.search(r"\bimport\s*\(", js) and "fetch(" not in js, name
    assert "attributionLogo: false" not in (d / "main.js").read_text(encoding="utf-8")   # TradingView's logo stays
