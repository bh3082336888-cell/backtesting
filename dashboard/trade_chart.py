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
              "exit_reason", "stop_init", "target", "pnl", "pnl_pct", "r_multiple", "regime", "fees", "funding"]
MAX_CANDLES = 6000            # more than this in the window → the next coarser timeframe is drawn
DISPLAY_TFS = ("5m", "15m", "1h", "4h", "1d")
EXIT_REASONS = ("stop", "target", "signal", "liq", "eow")
CONTEXT_BARS = 40             # bars either side of a trade when jumping to it
DAY = pd.Timedelta(days=1)

# Light-theme defaults; the dashboard passes its own palette (dark or light) over these.
PALETTE = {
    "s1": "#2a78d6", "s2": "#eb6834", "neg": "#e34948", "muted": "#898781",
    "band": "rgba(11,11,11,0.035)", "train": "rgba(11,11,11,0.07)",
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
        ]
        lv = _levels(r)
        if lv:
            lines.append(f"stop at exit {_px(lv[1][-1])} · target at exit {_px(lv[2][-1])}")
        if "fold" in r and pd.notna(r["fold"]):
            lines.append(f"fold {int(r['fold'])} · {r.get('combo') or ''}")
        out.append("<br>".join(lines))
    return out


def _steps(rows, which: int) -> tuple[list, list]:
    """Stop (which=1) or target (2) as step lines in one trace, None between trades and across a NaN.
    `rows` holds (entry_ms, exit_ms, levels-or-None, initial value). A level runs from its time to the
    next one's, the last to the exit. Times never go backwards: a paper level can be stamped before the
    entry fill, so each is clamped to the previous vertex."""
    xs, ys = [], []

    def cut():
        if xs and xs[-1] is not None:
            xs.append(None)
            ys.append(None)

    for entry, exit_, lv, init in rows:
        if lv is None:
            times, vals = [entry], [init]
        else:
            times, vals = lv[0], lv[which]
        at, prev = [], entry
        for tm in times:
            prev = max(prev, tm)
            at.append(prev)
        at.append(max(prev, exit_))
        for k, v in enumerate(vals):
            if pd.isna(v):
                cut()
                continue
            xs += [_x(_utc(at[k])), _x(_utc(at[k + 1]))]
            ys += [v, v]
        cut()
    return xs, ys


def _draw_trades(fig: go.Figure, t: pd.DataFrame, candles: pd.DataFrame, p: dict) -> None:
    sig_x, ent_x, ext_x = (_x(pd.DatetimeIndex(_utc(t[c].astype("int64"))))
                           for c in ("signal_time", "entry_time", "exit_time"))
    hov = np.array(hover_text(t), dtype=object)
    skip = dict(hoverinfo="skip")

    recs = t.to_dict("records")
    lvs = [_levels(r) for r in recs]
    moved = any(lv is not None for lv in lvs)
    for which, col, name, color in ((1, "stop_init", "stop", p["neg"]), (2, "target", "target", p["s1"])):
        rows = [(int(r["entry_time"]), int(r["exit_time"]), lv, r[col]) for r, lv in zip(recs, lvs)]
        xs, ys = _steps(rows, which)
        if xs:
            fig.add_scatter(x=xs, y=ys, mode="lines", name=name if moved else f"initial {name}",
                            line=dict(color=color, width=1, dash="dot"), **skip)

    # The signal bar's close, taken from the run's own bars, so it stays exact when coarser candles are drawn.
    close_at = pd.Series(candles.close.to_numpy(), index=candles.open_time.astype("int64").to_numpy())
    sig_y = t.signal_time.astype("int64").map(close_at).to_numpy()
    ok = ~pd.isna(sig_y)
    if ok.any():
        cx, cy = [], []
        for a, ya, b, yb in zip(sig_x[ok], sig_y[ok], ent_x[ok], t.entry_price.to_numpy()[ok]):
            cx += [a, b, None]
            cy += [ya, yb, None]
        fig.add_scatter(x=cx, y=cy, mode="lines", legendgroup="signal", showlegend=False,
                        line=dict(color=p["ink"], width=1, dash="dot"), **skip)
        fig.add_scatter(x=sig_x[ok], y=sig_y[ok], mode="markers", name="signal bar (close)", legendgroup="signal",
                        marker=dict(symbol="circle-open", size=8, color=p["ink"], line=dict(width=1.5)),
                        hovertext=hov[ok], hovertemplate="%{hovertext}<extra></extra>")

    for side, name, sym in ((1, "long entry", "triangle-up"), (-1, "short entry", "triangle-down")):
        m = (t.side > 0).to_numpy() if side > 0 else (t.side < 0).to_numpy()
        if m.any():
            fig.add_scatter(x=ent_x[m], y=t.entry_price.to_numpy()[m], mode="markers", name=name,
                            marker=dict(symbol=sym, size=12, color=p["ink"], line=dict(width=1, color=p["ink"])),
                            hovertext=hov[m], hovertemplate="%{hovertext}<extra></extra>")

    colors = exit_colors(p)
    reasons = list(EXIT_REASONS) + sorted(set(t.exit_reason) - set(EXIT_REASONS))
    for reason in reasons:
        m = (t.exit_reason == reason).to_numpy()
        if m.any():
            fig.add_scatter(x=ext_x[m], y=t.exit_price.to_numpy()[m], mode="markers", name=f"exit: {reason}",
                            marker=dict(symbol="x", size=10, color=colors.get(reason, p["muted"]),
                                        line=dict(width=1, color=colors.get(reason, p["muted"]))),
                            hovertext=hov[m], hovertemplate="%{hovertext}<extra></extra>")


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
                 folds: pd.DataFrame | None = None) -> go.Figure:
    """Candles (coarser if the window is long) with each trade in [first bar, last bar + tf):
    entry at the fill, the signal bar, exit coloured by reason, the stop and target (as they moved, or the initial ones for older runs).
    Walk-forward: test folds shaded in turn and labelled. Paper: the warmup shaded."""
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
            _draw_trades(fig, t, candles, p)
        warmup_end = None
        if "live" in candles.columns and not candles.live.all():
            live = candles.index[candles.live.astype(bool).to_numpy()]
            warmup_end = live[0] if len(live) else hi
        _shade(fig, lo, hi, p, folds, warmup_end)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="price")
    return fig
