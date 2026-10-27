"""The trade chart on TradingView Lightweight Charts: a custom Streamlit component (`components/tv_chart/`, static
HTML and plain JavaScript over the vendored library, no build step).

`chart_payload` turns a window's candles and the run's trades into the JSON the component draws. Every number on
the chart is in it, and every trade number comes from the trade record through `trade_chart` (`trade_boxes`,
`trade_tags`, `level_steps`, `rr_text`, `result_text`, `hover_text`): the browser only maps times and prices to
pixels. `tv_chart` renders a payload and hands back the trade the user clicked.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components import v1 as components

from dashboard import trade_chart as tc
from perp_lab.config import TF_MS

MAX_TV_BARS = 50_000          # more candles than this in the window → the next coarser timeframe is drawn
COMPONENT = Path(__file__).resolve().parent / "components" / "tv_chart"
WARMUP_HOVER = "bars fetched from the exchange before the paper run started"


def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _ms(x) -> int:
    return int(_utc(x).value // 10**6)


def _index_ms(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.as_unit("ms").asi8


def _id(label):
    """A trade's id on the chart: its row label in the trades frame (app.py's "Jump to trade" value)."""
    return int(label) if isinstance(label, (int, np.integer)) else str(label)


def _number(v) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


# ---------------------------------------------------------------- pieces of the payload

def _candles(bars: pd.DataFrame) -> list[dict]:
    """OHLC with the open time in UTC seconds (Lightweight Charts' UTCTimestamp)."""
    b = bars[["open", "high", "low", "close"]].astype(float).dropna()
    secs = (_index_ms(b.index) // 1000).tolist()
    return [{"time": t, "open": o, "high": h, "low": lo, "close": c}
            for t, o, h, lo, c in zip(secs, *(b[k].tolist() for k in ("open", "high", "low", "close")))]


def _price_format(bars: pd.DataFrame) -> dict:
    """Six significant figures at the window's typical price (as `trade_chart._px` writes prices), at least two
    decimals: 2 for BTC, 6 for a coin near 0.1."""
    med = float(np.nanmedian(bars.close)) if len(bars) else 0.0
    dec = int(np.clip(5 - np.floor(np.log10(abs(med))), 2, 8)) if med else 2
    return {"type": "price", "precision": dec, "minMove": float(f"1e-{dec}")}


def _shading(lo: pd.Timestamp, hi: pd.Timestamp, p: dict, folds: pd.DataFrame | None, warmup_end) -> list[dict]:
    """Shaded spans with their labels and hover: walk-forward, the stretch before the first test window and every
    other test fold; paper, the warmup bars. `fill` is None for a fold that is labelled but not shaded."""
    out = []

    def add(a, b, fill, label, hover):
        out.append({"from": _ms(a), "to": _ms(b), "fill": fill, "label": label, "hover": hover})

    if folds is not None and len(folds):
        f = folds.sort_values("fold").reset_index(drop=True)
        first = _utc(f.test_start.iloc[0])
        if lo < first:
            add(lo, min(first, hi), p["train"], "train only",
                f"before the first test window ({first:%Y-%m-%d}): no OOS trades")
        for i, r in f.iterrows():
            a, b = _utc(r.test_start), _utc(r.test_end)
            if b <= lo or a >= hi:
                continue
            add(max(a, lo), min(b, hi), p["band"] if i % 2 == 0 else None, f"fold {r.fold}",
                f"fold {r.fold} test {a:%Y-%m-%d} → {b:%Y-%m-%d}\n"
                f"trained on {str(r.train_start)[:10]} → {str(r.train_end)[:10]}\nparams {r.get('combo') or '—'}")
    if warmup_end is not None and lo < warmup_end:
        add(lo, min(warmup_end, hi), p["train"], "warmup", WARMUP_HOVER)
    return out


def _plain(b: dict) -> dict:
    """`trade_boxes` as JSON: its tuples as lists."""
    return {k: [list(x) for x in v] if k in ("reward", "risk") else list(v) if isinstance(v, tuple) else v
            for k, v in b.items()}


def _trade(label, r: dict, b: dict, tip: str, p: dict, lo: int, hi: int, fmt: str, tagged: bool,
           close_at: pd.Series) -> dict:
    (e_t, e_px), (x_t, x_px, reason) = b["entry"], b["exit"]
    net = _number(r.get("r_multiple"))
    sig = int(r["signal_time"])
    close = _number(close_at.get(sig))
    tags = tc.trade_tags(r, b, p, lo, hi, fmt) if tagged else {"price": [], "time": [], "be": None}
    return {
        "id": _id(label),
        "side": 1 if r["side"] > 0 else -1,
        "boxes": _plain(b),
        "rr": tc.rr_text(r),
        "result": tc.result_text(r),
        "tone": p["muted"] if net is None or net == 0 else p["pos"] if net > 0 else p["neg"],
        "tooltip": [re.sub(r"</?b>", "", line) for line in tip.split("<br>")],
        "entry": {"time": e_t, "price": e_px},
        "exit": {"time": x_t, "price": x_px, "reason": reason, "color": tc.exit_colors(p).get(reason, p["muted"])},
        "signal": None if close is None else {"time": sig, "price": close},   # the signal bar's close, if in view
        "stops": [list(s) for s in tc.level_steps(r, 1)],
        "targets": [list(s) for s in tc.level_steps(r, 2)],
        "tagged": tagged,
        "price_lines": [{"price": y, "title": word, "color": c} for word, y, c in tags["price"]],
        "time_tags": [{"time": ms, "text": text, "color": c, "side": side} for ms, text, c, side in tags["time"]],
        "be": None if tags["be"] is None else dict(zip(("time", "price", "text"), tags["be"])),
    }


# ---------------------------------------------------------------- the payload

def chart_payload(candles: pd.DataFrame, trades: pd.DataFrame, tf: str, *, focus=None, palette: dict,
                  overlay=None, layers: frozenset = frozenset(), folds: pd.DataFrame | None = None,
                  title: str = "") -> dict:
    """What the trade chart draws, as JSON (strict: no NaN), for the run's candles at timeframe `tf` in the window
    [first bar, last bar + tf):
      * `candles`: {time (UTC s), open, high, low, close}, at a coarser timeframe (`shown`, `bar_ms`) only past
        MAX_TV_BARS bars; `window`: the window in epoch ms; `price_format` for the price axis;
      * `trades`: every trade open in the window, with its `trade_boxes` geometry (unclipped: the chart clips to
        the window), R:R and result text, tooltip lines, entry, exit, the signal bar's close (from the run's own
        bars), its stop and target steps, and its tags: `price_lines` (Entry, TP, SL), `time_tags` and the
        breakeven note, only for the trade in `focus` (a row label of `trades`), or for every trade when at most
        TAG_ALL are in view;
      * `markers`: the entry and exit dots, each on the (possibly coarser) candle holding its time, at its price;
      * `folds`: the shaded spans and their labels (test folds, or a paper run's warmup); `exits`: the exit
        reasons in view and their colours; `focus`: the focused trade's id, None when it isn't in view;
      * `layers`: the LCE overlay's layers; empty without an `overlay` (Task 4 fills it)."""
    p = {**tc.PALETTE, **palette}
    shown = tc.display_tf(tf, len(candles), MAX_TV_BARS)
    bars = candles if shown == tf else tc.resample(candles, shown)
    out = {"title": title, "tf": tf, "shown": shown, "bar_ms": TF_MS[shown], "window": None,
           "context": tc.CONTEXT_BARS, "price_format": _price_format(bars), "candles": _candles(bars), "trades": [],
           "markers": [], "focus": None, "folds": [], "exits": {}, "palette": p, "layers": {}}
    if candles.empty:
        return out
    lo, hi = candles.index[0], candles.index[-1] + pd.Timedelta(milliseconds=TF_MS[tf])
    lo_ms, hi_ms = _ms(lo), _ms(hi)
    out["window"] = [lo_ms, hi_ms]
    warmup_end = None
    if "live" in candles.columns and not candles.live.all():
        live = candles.index[candles.live.astype(bool).to_numpy()]
        warmup_end = live[0] if len(live) else hi
    out["folds"] = _shading(lo, hi, p, folds, warmup_end)

    t = tc.trades_in(trades, lo, hi)
    if not len(t):
        return out
    tagged = (set(t.index) if len(t) <= tc.TAG_ALL
              else {focus} if focus is not None and focus in t.index else set())
    out["focus"] = _id(focus) if focus is not None and focus in t.index else None
    fmt = "%a %d %H:%M" if hi - lo > tc.DAY else "%H:%M"             # UTC, as the axis; dated past one day
    close_at = pd.Series(candles.close.to_numpy(dtype=float), index=_index_ms(candles.index))
    bar_ms = _index_ms(bars.index)
    colors = tc.exit_colors(p)

    def on_bar(ms: int) -> int:
        """The open time (UTC s) of the drawn candle holding `ms`."""
        return int(bar_ms[max(int(np.searchsorted(bar_ms, ms, side="right")) - 1, 0)] // 1000)

    recs = t.to_dict("records")
    for label, r, tip in zip(t.index, recs, tc.hover_text(t)):
        b = tc.trade_boxes(r, TF_MS[tf])          # a trade in and out on one bar is drawn one bar wide
        tr = _trade(label, r, b, tip, p, lo_ms, hi_ms, fmt, label in tagged, close_at)
        out["trades"].append(tr)
        for kind, at, color in (("entry", tr["entry"], p["s1"]), ("exit", tr["exit"], tr["exit"]["color"])):
            if lo_ms <= at["time"] < hi_ms:       # no dot for a time off the window
                out["markers"].append({"time": on_bar(at["time"]), "price": at["price"], "color": color,
                                       "kind": kind, "id": tr["id"]})
    out["markers"].sort(key=lambda m: m["time"])
    reasons = set(t.exit_reason.astype(str))
    out["exits"] = {x: colors.get(x, p["muted"])
                    for x in [*(x for x in tc.EXIT_REASONS if x in reasons), *sorted(reasons - set(tc.EXIT_REASONS))]}
    return out


# ---------------------------------------------------------------- the component

def tv_chart(payload: dict, key: str, height: int = 560,
             on_click: Callable[[int | str], None] | None = None) -> dict | None:
    """Draw a payload. Returns {"clicked": trade_id} once the user has clicked inside a trade's box (the last one
    clicked), else None. `on_click(trade_id)` runs as the click's callback, before the page reruns, so it may set
    other widgets' state (app.py sets "Jump to trade" with it)."""
    chart = components.declare_component("tv_chart", path=str(COMPONENT))   # registered with this session's runtime

    def clicked():
        v = st.session_state.get(key)
        if on_click is not None and isinstance(v, dict) and v.get("clicked") is not None:
            on_click(v["clicked"])

    v = chart(payload=payload, height=height, key=key, default=None, on_change=clicked)
    return {"clicked": v["clicked"]} if isinstance(v, dict) and v.get("clicked") is not None else None
