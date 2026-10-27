"""Trade charts: a run's trades drawn on the candles they happened on.

Walk-forward and paper runs only. Candles come from where the run itself got them:
  * walk-forward: the research parquet store, through `load_candles` with no holdout token, so
    the holdout guard stays on; the range is cut at `holdout_start` before the read;
  * paper: the `bars_<symbol>.parquet` the paper runner saved, exactly the bars it saw.
Any other run (holdout exams, in-sample backtests) is refused with `NotChartable`.

No Streamlit in here: the dashboard caches and lays out; this loads the data and builds the figure.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from perp_lab.config import TF_MS
from perp_lab.data.store import load_candles

CHARTABLE = ("walkforward", "paper")
TRADE_FILES = {"walkforward": "oos_trades.parquet", "paper": "trades.parquet"}
TRADE_COLS = ["symbol", "side", "signal_time", "entry_time", "entry_price", "exit_time", "exit_price",
              "exit_reason", "stop_init", "target", "qty", "pnl_gross", "slippage", "pnl", "pnl_pct", "r_multiple",
              "regime", "fees", "funding"]
MAX_CANDLES = 6000            # more than this in the window → the next coarser timeframe is drawn
DISPLAY_TFS = ("5m", "15m", "1h", "4h", "1d")
EXIT_REASONS = ("stop", "target", "signal", "liq", "eow")
CONTEXT_BARS = 40             # bars either side of a trade when jumping to it
DAY = pd.Timedelta(days=1)
TAG_ALL = 5                   # at most this many trades in view → every one gets its tags and label

# Trace names of the position boxes (legend entries the user can click to hide).
BOX_REWARD = "reward (entry → target)"
BOX_RISK = "risk (entry → stop)"
BOX_RESULT = "result (no fixed target)"
BOX_ALPHA = 0.18
ON_FILL = "#ffffff"           # text on a coloured tag

# Light-theme defaults; the dashboard passes its own palette (dark or light) over these. "pos" is the
# reference palette's status green (mode-invariant); "bg" is the chart's background, for label plates.
PALETTE = {
    "s1": "#2a78d6", "s2": "#eb6834", "neg": "#e34948", "pos": "#0ca30c", "muted": "#898781",
    "band": "rgba(11,11,11,0.035)", "train": "rgba(11,11,11,0.07)", "bg": "#ffffff",
    "ink": "#1f1f1d", "candle": "#a9a7a0", "liq": "#8e2dff", "eow": "#1f9e89",
}


class NotChartable(ValueError):
    """A run v1 doesn't chart: anything but walk-forward and paper."""


def check_chartable(mode: str) -> None:
    if mode not in CHARTABLE:
        raise NotChartable(f"trade charts cover walk-forward and paper runs; this run is '{mode}'")


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _ms(t: pd.Timestamp) -> int:
    return int(_ts(t).value // 10**6)


def _utc(ms):
    return pd.to_datetime(ms, unit="ms", utc=True)


def _tf(tf: str) -> pd.Timedelta:
    return pd.Timedelta(milliseconds=TF_MS[tf])


def _x(t):
    """UTC wall time without a zone, the one form Plotly keeps as is: every x on the chart goes through here."""
    return t.tz_convert("UTC").tz_localize(None)


# ---------------------------------------------------------------- data

def load_trades(mode: str, run_dir) -> pd.DataFrame:
    """The run's trade log (walk-forward: its OOS trades), empty if it saved none."""
    check_chartable(mode)
    p = Path(run_dir) / TRADE_FILES[mode]
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=TRADE_COLS)


def chart_symbols(mode: str, run_dir, universe) -> list[str]:
    """Walk-forward: the run's universe. Paper: the symbols it saved bars for."""
    check_chartable(mode)
    if mode == "paper":
        return sorted(p.stem[len("bars_"):] for p in Path(run_dir).glob("bars_*.parquet"))
    return list(universe or [])


def _paper_bars(run_dir, symbol: str) -> pd.DataFrame:
    p = Path(run_dir) / f"bars_{symbol}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"this paper run saved no bars for {symbol}")
    b = pd.read_parquet(p).drop_duplicates("open_time", keep="last").sort_values("open_time")
    b.index = _utc(b.open_time.astype("int64"))
    b.index.name = "time"
    return b


def chart_span(cfg, mode: str, run_dir, symbol: str, tf: str, date_start,
               date_end) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) the run's candles may be drawn over. Walk-forward: the run's dates, never past
    `holdout_start`. Paper: the saved bars, warmup included."""
    check_chartable(mode)
    if mode == "walkforward":
        return _ts(date_start), min(_ts(date_end), cfg.holdout_start)
    b = _paper_bars(run_dir, symbol)
    if b.empty:
        raise FileNotFoundError(f"this paper run saved no bars for {symbol}")
    return b.index[0], b.index[-1] + _tf(tf)


def run_candles(cfg, mode: str, run_dir, symbol: str, tf: str, start, end) -> pd.DataFrame:
    """Candles at the run's timeframe with open time in [start, end), from the run's own source."""
    check_chartable(mode)
    start, end = _ts(start), _ts(end)
    if mode == "walkforward":
        end = min(end, cfg.holdout_start)       # research bars only; no token is ever asked for
        return load_candles(cfg, symbol, tf, start, max(start, end))
    b = _paper_bars(run_dir, symbol)
    return b[(b.index >= start) & (b.index < end)]


def trades_in(trades: pd.DataFrame, start, end) -> pd.DataFrame:
    """Trades open at some point in [start, end)."""
    if trades.empty:
        return trades
    return trades[(trades.entry_time < _ms(end)) & (trades.exit_time >= _ms(start))]


# ---------------------------------------------------------------- windows (dates for the picker, end inclusive)

def as_dates(s: pd.Timestamp, e: pd.Timestamp) -> tuple[date, date]:
    return s.date(), (e - pd.Timedelta(1, "ns")).date()


def default_window(mode: str, span, folds: pd.DataFrame | None = None) -> tuple[date, date]:
    """Walk-forward: the last test fold. Paper: the whole run."""
    s, e = span
    if mode == "walkforward" and folds is not None and len(folds):
        f = folds.sort_values("fold").iloc[-1]
        s, e = max(s, _ts(f.test_start)), min(e, _ts(f.test_end))
    return as_dates(s, e)


def trade_window(trade, tf: str, span) -> tuple[date, date]:
    """A trade with CONTEXT_BARS bars (at least a day) either side, inside the span."""
    pad = max(DAY, _tf(tf) * CONTEXT_BARS)
    s = max(span[0], _utc(int(trade["signal_time"])) - pad)
    e = min(span[1], _utc(int(trade["exit_time"])) + pad)
    return as_dates(s, e)


def picked_window(picked, span) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Date-picker value (one date while a range is half picked) → [start, end) inside the span."""
    ds = list(picked) if isinstance(picked, (list, tuple)) else [picked]
    if not ds:
        return span
    s = max(span[0], _ts(ds[0]))
    e = min(span[1], _ts(ds[-1]) + DAY)
    return s, max(s, e)


# ---------------------------------------------------------------- candles at a readable density

def display_tf(tf: str, n_bars: int) -> str:
    """The run's timeframe, or the first coarser one that fits the window in MAX_CANDLES."""
    for t in DISPLAY_TFS:
        if TF_MS[t] >= TF_MS[tf] and n_bars * TF_MS[tf] / TF_MS[t] <= MAX_CANDLES:
            return t
    return DISPLAY_TFS[-1]


def resample(candles: pd.DataFrame, tf: str) -> pd.DataFrame:
    """OHLC at a coarser timeframe, bins aligned to the epoch (so 4h and 1d start at 00:00 UTC)."""
    r = candles[["open", "high", "low", "close"]].resample(_tf(tf), origin="epoch", label="left", closed="left")
    return r.agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna(subset=["open"])


# ---------------------------------------------------------------- figure

def _px(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:,.6g}"


def _num(r, key) -> float:
    """A record's number, NaN when the column is missing or null (older runs, frames mixing old and new)."""
    v = r.get(key)
    return np.nan if v is None or pd.isna(v) else float(v)


def _levels(r):
    """One trade's (times, stops, targets) as plain lists, or None when it recorded no levels (a run made
    before they were logged: column missing, None or empty). Parquet hands list columns back as arrays."""
    lt = r.get("level_time")
    if lt is None or not hasattr(lt, "__len__") or len(lt) == 0:
        return None
    nan = lambda v: np.nan if v is None else float(v)       # noqa: E731  (parquet may store a NaN as null)
    return ([int(v) for v in lt], [nan(v) for v in r["level_stop"]], [nan(v) for v in r["level_target"]])


def has_levels(t: pd.DataFrame) -> bool:
    """Whether any trade in the frame recorded the stop and target as they moved."""
    return any(_levels(r) is not None for r in t.to_dict("records"))


def rr_text(r: dict) -> str:
    """The planned risk:reward as TradingView writes it: |target − entry| / |entry − stop_init|."""
    entry, stop, target = _num(r, "entry_price"), _num(r, "stop_init"), _num(r, "target")
    if pd.isna(target):
        return "no fixed target"
    risk = abs(entry - stop)
    return f"R:R 1 : {abs(target - entry) / risk:.2f}" if risk > 0 else "R:R —"


def _r(x: float) -> str:
    return "—R" if pd.isna(x) else f"{x:+.2f}R"


def result_text(r: dict) -> str:
    """Gross and net R. Gross = (pnl_gross + slippage) / (qty·|entry − stop_init|), before fees, slippage and
    funding (the engine's pnl_gross is after slippage); net = the engine's r_multiple. A run that kept no
    slippage counts it as 0, so its gross is after slippage, and the text says so."""
    slip = _num(r, "slippage")
    kept = not pd.isna(slip)
    risk = _num(r, "qty") * abs(_num(r, "entry_price") - _num(r, "stop_init"))
    gross = (_num(r, "pnl_gross") + (slip if kept else 0.0)) / risk if risk > 0 else np.nan
    return f"{_r(gross)} gross{'' if kept else ' after slippage'} · {_r(_num(r, 'r_multiple'))} net"


def hover_text(t: pd.DataFrame) -> list[str]:
    out = []
    for r in t.to_dict("records"):
        f = r.get("funding", np.nan)
        funding = ("—" if pd.isna(f) else f"{abs(f):,.2f}" if abs(f) < 0.005
                   else f"{'paid' if f > 0 else 'received'} {abs(f):,.2f}")
        pct = r.get("pnl_pct", np.nan)
        lines = [
            f"<b>{'LONG' if r['side'] > 0 else 'SHORT'} {r['symbol']}</b> · exit {r['exit_reason']}",
            f"P&L {r['pnl']:+,.2f} USDT{'' if pd.isna(pct) else f' ({pct:+.2%})'} · R {r['r_multiple']:+.2f}",
            f"regime at signal: {r.get('regime', 'unknown')}",
            f"fees {r['fees']:,.2f} · funding {funding}",
            f"signal bar {_utc(r['signal_time']):%Y-%m-%d %H:%M} · entry {_utc(r['entry_time']):%Y-%m-%d %H:%M} "
            f"@ {_px(r['entry_price'])}",
            f"exit {_utc(r['exit_time']):%Y-%m-%d %H:%M} @ {_px(r['exit_price'])}",
            f"initial stop {_px(r['stop_init'])} · initial target {_px(r['target'])}",
            f"{rr_text(r)} · {result_text(r)}",
        ]
        lv = _levels(r)
        if lv:
            lines.append(f"stop at exit {_px(lv[1][-1])} · target at exit {_px(lv[2][-1])}")
        if "fold" in r and pd.notna(r["fold"]):
            lines.append(f"fold {int(r['fold'])} · {r.get('combo') or ''}")
        out.append("<br>".join(lines))
    return out


def _times(ms: list) -> list:
    """Epoch-ms x values, None marking a break, as chart times (converted in one call)."""
    keep = np.array([m is not None for m in ms], dtype=bool)
    xs = np.empty(len(ms), dtype=object)
    xs[~keep] = None
    xs[keep] = list(_x(_utc(np.array([m for m in ms if m is not None], dtype="int64"))))
    return xs.tolist()


def _steps(rows, which: int, lo: int | None = None, hi: int | None = None) -> tuple[list, list]:
    """Stop (which=1) or target (2) as step lines in one trace, None between trades and across a NaN.
    `rows` holds (entry_ms, exit_ms, levels-or-None, initial value). A level runs from its time to the
    next one's, the last to the exit. Times never go backwards: a paper level can be stamped before the
    entry fill, so each is clamped to the previous vertex. With a window [lo, hi] (epoch ms), each level
    is cut to it, and one wholly outside it is left out."""
    ms, ys = [], []        # x in epoch ms, converted to times in one call below; None marks a break

    def cut():
        if ms and ms[-1] is not None:
            ms.append(None)
            ys.append(None)

    for entry, exit_, lv, init in rows:
        if lv is None:
            times, vals = [entry], [init]
        else:
            times, vals = lv[0], lv[which]
        at = np.maximum.accumulate(np.array([entry, *times], dtype="int64"))[1:]
        at = np.append(at, max(at[-1], exit_)).tolist()
        for k, v in enumerate(vals):
            a, b = at[k], at[k + 1]
            if lo is not None:
                a2, b2 = max(a, lo), min(b, hi)
                if b2 < a2 or (b2 == a2 and b > a):        # outside the window (a level at one instant stays)
                    continue
                a, b = a2, b2
            if pd.isna(v):
                cut()
                continue
            ms += [a, b]
            ys += [v, v]
        cut()
    return _times(ms), ys


# ---------------------------------------------------------------- position boxes (TradingView's long/short tool)

def _extend(boxes: list, box: tuple) -> None:
    """Append a box, or widen the last one when the level held (same price, touching in time)."""
    if boxes and boxes[-1][1] == box[0] and boxes[-1][3] == box[3]:
        boxes[-1] = (boxes[-1][0], box[1], box[2], box[3])
    else:
        boxes.append(box)


def trade_boxes(r: dict, min_ms: int = 0) -> dict:
    """One trade's position-tool geometry, in epoch ms and prices, from its record alone.

    `reward` and `risk` are boxes (t0, t1, entry, level) along the target's and the stop's level paths
    (the initial ones for a run without paths), one box while a level holds. The risk box is there only
    while the stop is on the losing side of the entry: once the stop reaches or passes it (breakeven, or a
    trail into profit) the box collapses, and `be_time` is when. `entry` is (time, price), `exit` (time,
    price, reason). `no_target`: no target at any point, so no reward box. `span` is the time the trade
    covers, widened to `min_ms` for a trade that entered and exited on one bar (else it draws nothing)."""
    side = 1 if r["side"] > 0 else -1
    e_t, x_t, e_px = int(r["entry_time"]), int(r["exit_time"]), float(r["entry_price"])
    lv = _levels(r)
    times, stops, targets = lv if lv else ([e_t], [_num(r, "stop_init")], [_num(r, "target")])
    at = np.maximum.accumulate(np.array([e_t, *times], dtype="int64"))[1:].tolist()   # as `_steps` does
    end = max(at[-1], x_t, e_t + min_ms)
    edges = [*at, end]
    reward, risk, be = [], [], None
    for k, (stop, target) in enumerate(zip(stops, targets)):
        a, b = edges[k], edges[k + 1]
        safe = not pd.isna(stop) and side * (stop - e_px) >= 0     # the stop at or past the entry
        if safe and be is None:
            be = a
        if b <= a:
            continue
        if not pd.isna(target):
            _extend(reward, (a, b, e_px, float(target)))
        if not pd.isna(stop) and not safe:
            _extend(risk, (a, b, e_px, float(stop)))
    return {"reward": reward, "risk": risk, "entry": (e_t, e_px),
            "exit": (x_t, float(r["exit_price"]), str(r["exit_reason"])), "be_time": be,
            "no_target": all(pd.isna(g) for g in targets), "span": (e_t, end)}


def tint(color: str, alpha: float = BOX_ALPHA) -> str:
    """A '#rrggbb' colour as rgba at the given opacity (box fills, the label's plate)."""
    h = color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _boxes(fig: go.Figure, name: str, rows: list) -> None:
    """One bar trace per kind of box, one bar per box: it starts at t0 (offset 0) and is as wide as the box
    lasts, based at the entry. Bars, unlike shapes, show their hover anywhere inside, and one trace holds
    every trade's boxes however many are in view."""
    if not rows:
        return
    t0, t1, y0, y1, color, hov = zip(*rows)
    fig.add_bar(x=_times(list(t0)), width=[b - a for a, b in zip(t0, t1)], offset=0, base=list(y0),
                y=[b - a for a, b in zip(y0, y1)], name=name, hovertext=list(hov),
                hovertemplate="%{hovertext}<extra></extra>",
                marker=dict(color=[tint(c) for c in color], line=dict(color=[tint(c, 0.6) for c in color], width=1)))


def _tags(fig: go.Figure, r: dict, b: dict, p: dict, lo: int, hi: int, fmt: str) -> None:
    """One trade's axis tags (price edge: Entry, TP, SL at the initial prices; time axis: entry and exit with
    dotted verticals, when in view), its R:R and result label, and its breakeven note."""
    side = 1 if r["side"] > 0 else -1
    (e_t, e_px), (x_t, x_px, reason) = b["entry"], b["exit"]
    for word, y, color in (("Entry", e_px, p["s1"]), ("TP", _num(r, "target"), p["pos"]),
                           ("SL", _num(r, "stop_init"), p["neg"])):
        if not pd.isna(y):
            fig.add_annotation(name="price tag", xref="paper", x=1, xanchor="right", yref="y", y=y,
                               text=f"{word} {_px(y)}", showarrow=False, bgcolor=color, borderpad=2,
                               font=dict(color=ON_FILL, size=10))
    for ms, word, color, anchor in ((e_t, "entry", p["s1"], "right"),
                                    (x_t, "exit", exit_colors(p).get(reason, p["muted"]), "left")):
        if lo <= ms < hi:
            x = _x(_utc(ms))
            fig.add_shape(name="time tag", type="line", xref="x", yref="paper", x0=x, x1=x, y0=0, y1=1,
                          line=dict(color=color, width=1, dash="dot"), opacity=0.6, layer="below")
            fig.add_annotation(name="time tag", xref="x", x=x, xanchor=anchor, yref="paper", y=0, yanchor="top",
                               text=f"{_utc(ms):{fmt}} {word}", showarrow=False, bgcolor=color, borderpad=2,
                               font=dict(color=ON_FILL, size=10))            # on the time axis, over its ticks
    # The label hangs inside the reward box from its far edge; with no target, it sits on the entry line's
    # reward side. Either way it stays within the prices the trade spans, so clear of the axes' tags.
    a, z = max(b["span"][0], lo), min(b["span"][1], hi)
    far = max if side > 0 else min
    if b["reward"]:
        y, inward = far(box[3] for box in b["reward"]), True
    else:
        y, inward = far(e_px, x_px), False
    net = _num(r, "r_multiple")
    tone = p["pos"] if net > 0 else p["neg"] if net < 0 else p["muted"]
    fig.add_annotation(name="trade label", xref="x", x=_x(_utc((a + z) // 2)), yref="y", y=y, xanchor="center",
                       yanchor=("top" if side > 0 else "bottom") if inward else ("bottom" if side > 0 else "top"),
                       text=f"<b>{rr_text(r)}</b><br><span style='color:{tone}'>{result_text(r)}</span>",
                       showarrow=False, align="left", bgcolor=tint(p["bg"], 0.85), bordercolor=p["muted"],
                       borderwidth=1, borderpad=3, font=dict(color=p["ink"], size=10))
    be = b["be_time"]
    if be is not None and lo <= be < hi:
        fig.add_annotation(name="be note", xref="x", x=_x(_utc(be)), yref="y", y=e_px, xanchor="left",
                           yanchor="top" if side > 0 else "bottom", text=f"SL → BE {_utc(be):{fmt}}",
                           showarrow=False, font=dict(color=p["s2"], size=9))


def _draw_trades(fig: go.Figure, t: pd.DataFrame, candles: pd.DataFrame, p: dict, lo: pd.Timestamp,
                 hi: pd.Timestamp, tf: str, tagged: set) -> None:
    """Every trade in view as a position box, clipped to [lo, hi); tags and labels for the `tagged` rows."""
    lo_ms, hi_ms = _ms(lo), _ms(hi)
    recs = t.to_dict("records")
    hov = hover_text(t)
    geo = [trade_boxes(r, TF_MS[tf]) for r in recs]
    skip = dict(hoverinfo="skip")

    rows = {BOX_REWARD: [], BOX_RISK: [], BOX_RESULT: []}
    for r, b, h in zip(recs, geo, hov):
        for key, name, color in (("reward", BOX_REWARD, p["pos"]), ("risk", BOX_RISK, p["neg"])):
            for t0, t1, y0, y1 in b[key]:
                a, z = max(t0, lo_ms), min(t1, hi_ms)
                if z > a:
                    rows[name].append((a, z, y0, y1, color, h))
        if b["no_target"]:            # no reward box: the move to the exit, green for a gain, red for a loss
            a, z = max(b["span"][0], lo_ms), min(b["span"][1], hi_ms)
            (_, e_px), (_, x_px, _) = b["entry"], b["exit"]
            gain = (1 if r["side"] > 0 else -1) * (x_px - e_px) > 0
            if z > a:
                rows[BOX_RESULT].append((a, z, e_px, x_px, p["pos"] if gain else p["neg"], h))
    for name, box_rows in rows.items():
        _boxes(fig, name, box_rows)

    lvs = [_levels(r) for r in recs]
    moved = any(lv is not None for lv in lvs)
    for which, col, name, color in ((1, "stop_init", "stop", p["neg"]), (2, "target", "target", p["pos"])):
        steps = [(int(r["entry_time"]), int(r["exit_time"]), lv, r[col]) for r, lv in zip(recs, lvs)]
        xs, ys = _steps(steps, which, lo_ms, hi_ms)
        if xs:
            fig.add_scatter(x=xs, y=ys, mode="lines", name=name if moved else f"initial {name}",
                            line=dict(color=color, width=1, dash="dot"), **skip)

    # the entry: a bold line at its price across the trade
    lx, ly = [], []
    for b in geo:
        lx += [max(b["span"][0], lo_ms), min(b["span"][1], hi_ms), None]
        ly += [b["entry"][1], b["entry"][1], None]
    fig.add_scatter(x=_times(lx), y=ly, mode="lines", legendgroup="entry", showlegend=False,
                    line=dict(color=p["s1"], width=2.5), **skip)

    # The signal bar's close, taken from the run's own bars, so it stays exact when coarser candles are drawn.
    sig_x, ent_x = (_x(pd.DatetimeIndex(_utc(t[c].astype("int64")))) for c in ("signal_time", "entry_time"))
    close_at = pd.Series(candles.close.to_numpy(), index=candles.open_time.astype("int64").to_numpy())
    sig_y = t.signal_time.astype("int64").map(close_at).to_numpy()
    ok = ~pd.isna(sig_y)
    hov_a = np.array(hov, dtype=object)
    if ok.any():
        cx, cy = [], []
        for a, ya, b, yb in zip(sig_x[ok], sig_y[ok], ent_x[ok], t.entry_price.to_numpy()[ok]):
            cx += [a, b, None]
            cy += [ya, yb, None]
        fig.add_scatter(x=cx, y=cy, mode="lines", legendgroup="signal", showlegend=False,
                        line=dict(color=p["ink"], width=1, dash="dot"), **skip)
        fig.add_scatter(x=sig_x[ok], y=sig_y[ok], mode="markers", name="signal bar (close)", legendgroup="signal",
                        marker=dict(symbol="circle-open", size=8, color=p["ink"], line=dict(width=1.5)),
                        hovertext=hov_a[ok], hovertemplate="%{hovertext}<extra></extra>")

    # dots on the entry and exit candles, at the fill and exit prices (only those in view)
    def dots(at, name, color, keep):
        i = [k for k, b in enumerate(geo) if keep(b) and lo_ms <= b[at][0] < hi_ms]
        if i:
            fig.add_scatter(x=_times([geo[k][at][0] for k in i]), y=[geo[k][at][1] for k in i], mode="markers",
                            name=name, legendgroup=name,
                            marker=dict(symbol="circle", size=9, color=color, line=dict(width=1.5, color=p["bg"])),
                            hovertext=[hov[k] for k in i], hovertemplate="%{hovertext}<extra></extra>")

    dots("entry", "entry", p["s1"], lambda b: True)
    colors = exit_colors(p)
    for reason in list(EXIT_REASONS) + sorted(set(t.exit_reason) - set(EXIT_REASONS)):
        dots("exit", f"exit: {reason}", colors.get(reason, p["muted"]), lambda b, x=reason: b["exit"][2] == x)

    fmt = "%a %d %H:%M" if hi - lo > DAY else "%H:%M"           # UTC, as the axis; dated past one day
    for label, r, b in zip(t.index, recs, geo):
        if label in tagged:
            _tags(fig, r, b, p, lo_ms, hi_ms, fmt)


def exit_colors(palette: dict | None = None) -> dict:
    p = {**PALETTE, **(palette or {})}
    return {"stop": p["neg"], "target": p["s1"], "signal": p["s2"], "liq": p["liq"], "eow": p["eow"]}


def _band(fig, x0, x1, color):
    fig.add_vrect(x0=_x(x0), x1=_x(x1), fillcolor=color, line_width=0, layer="below")


def _label(fig, x, text, hover, p):
    fig.add_annotation(x=_x(x), y=1, xref="x", yref="paper", xanchor="left", yanchor="top", text=text,
                       showarrow=False, font=dict(size=10, color=p["muted"]), hovertext=hover)


def _shade(fig: go.Figure, lo, hi, p: dict, folds: pd.DataFrame | None, warmup_end) -> None:
    if folds is not None and len(folds):
        f = folds.sort_values("fold").reset_index(drop=True)
        first = _ts(f.test_start.iloc[0])
        if lo < first:
            _band(fig, lo, min(first, hi), p["train"])
            _label(fig, lo, "train only", f"before the first test window ({first:%Y-%m-%d}): no OOS trades", p)
        for i, r in f.iterrows():
            a, b = _ts(r.test_start), _ts(r.test_end)
            if b <= lo or a >= hi:
                continue
            if i % 2 == 0:
                _band(fig, max(a, lo), min(b, hi), p["band"])
            _label(fig, max(a, lo), f"fold {r.fold}",
                   f"fold {r.fold} test {a:%Y-%m-%d} → {b:%Y-%m-%d}<br>trained on {str(r.train_start)[:10]} → "
                   f"{str(r.train_end)[:10]}<br>params {r.get('combo') or '—'}", p)
    if warmup_end is not None and lo < warmup_end:
        _band(fig, lo, min(warmup_end, hi), p["train"])
        _label(fig, lo, "warmup", "bars fetched from the exchange before the paper run started", p)


def trade_figure(candles: pd.DataFrame, trades: pd.DataFrame, tf: str, *, palette: dict | None = None,
                 folds: pd.DataFrame | None = None, focus: int | None = None, overlay=None,
                 layers: frozenset = frozenset()) -> go.Figure:
    """Candles (coarser if the window is long) with each trade in [first bar, last bar + tf) drawn as
    TradingView's position tool, at the exact times and prices of its record whatever the candles: a green
    reward box (entry → target) and a red risk box (entry → stop, collapsing once the stop reaches the entry),
    each along its level path (the initial levels for older runs), a bold entry line, dots on the entry and
    exit (coloured by reason), the signal bar, and the stop and target as dotted step lines.
    `focus` (a row label of `trades`) is the trade that gets its axis tags and R:R label; every trade in view
    gets them when at most TAG_ALL are. `overlay` and `layers` are for the LCE layers; with no overlay
    they're ignored. Walk-forward: test folds shaded in turn and labelled. Paper: the warmup shaded."""
    p = {**PALETTE, **(palette or {})}
    shown = display_tf(tf, len(candles))
    bars = candles if shown == tf else resample(candles, shown)
    fig = go.Figure(go.Candlestick(
        x=_x(bars.index), open=bars.open, high=bars.high, low=bars.low, close=bars.close, name=f"{shown} candles",
        increasing=dict(line=dict(color=p["candle"], width=1), fillcolor="rgba(0,0,0,0)"),
        decreasing=dict(line=dict(color=p["candle"], width=1), fillcolor=p["candle"]), showlegend=False))
    if len(candles):
        lo, hi = candles.index[0], candles.index[-1] + _tf(tf)
        t = trades_in(trades, lo, hi)
        if len(t):
            tagged = set(t.index) if len(t) <= TAG_ALL else {focus} & set(t.index)
            _draw_trades(fig, t, candles, p, lo, hi, tf, tagged)
        warmup_end = None
        if "live" in candles.columns and not candles.live.all():
            live = candles.index[candles.live.astype(bool).to_numpy()]
            warmup_end = live[0] if len(live) else hi
        _shade(fig, lo, hi, p, folds, warmup_end)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="price")
    return fig
