"""Trade charts: a run's trades drawn on the candles they happened on.

Walk-forward and paper runs only. Candles come from where the run itself got them:
  * walk-forward: the research parquet store, through `load_candles` with no holdout token, so
    the holdout guard stays on; the range is cut at `holdout_start` before the read;
  * paper: the `bars_<symbol>.parquet` the paper runner saved, exactly the bars it saw.
Any other run (holdout exams, in-sample backtests) is refused with `NotChartable`.

No Streamlit and no drawing in here: this loads the data and works out each trade's geometry (boxes, level steps,
tags, label text) from its record alone. `tv_chart` turns that into what the chart draws; the dashboard caches and
lays out.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from perp_lab.config import TF_MS
from perp_lab.data.store import load_candles

CHARTABLE = ("walkforward", "paper")
TRADE_FILES = {"walkforward": "oos_trades.parquet", "paper": "trades.parquet"}
TRADE_COLS = ["symbol", "side", "signal_time", "entry_time", "entry_price", "exit_time", "exit_price",
              "exit_reason", "stop_init", "target", "qty", "pnl_gross", "slippage", "pnl", "pnl_pct", "r_multiple",
              "regime", "fees", "funding"]
DISPLAY_TFS = ("5m", "15m", "1h", "4h", "1d")
EXIT_REASONS = ("stop", "target", "signal", "liq", "eow")
CONTEXT_BARS = 40             # bars either side of a trade when jumping to it
DAY = pd.Timedelta(days=1)
TAG_ALL = 5                   # at most this many trades in view → every one gets its tags and label

# Light-theme defaults; the dashboard passes its own palette (dark or light) over these. "pos" is the
# reference palette's status green (mode-invariant); "bg" is the chart's background, for label plates.
PALETTE = {
    "s1": "#2a78d6", "s2": "#eb6834", "neg": "#e34948", "pos": "#0ca30c", "muted": "#898781",
    "band": "rgba(11,11,11,0.035)", "train": "rgba(11,11,11,0.07)", "bg": "#ffffff", "grid": "#e1e0d9",
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

def display_tf(tf: str, n_bars: int, cap: int) -> str:
    """The run's timeframe while the window has at most `cap` bars, else the first coarser one that fits it."""
    if n_bars <= cap:
        return tf
    for t in DISPLAY_TFS:
        if TF_MS[t] >= TF_MS[tf] and n_bars * TF_MS[tf] / TF_MS[t] <= cap:
            return t
    return DISPLAY_TFS[-1]


def resample(candles: pd.DataFrame, tf: str) -> pd.DataFrame:
    """OHLC at a coarser timeframe, bins aligned to the epoch (so 4h and 1d start at 00:00 UTC)."""
    r = candles[["open", "high", "low", "close"]].resample(_tf(tf), origin="epoch", label="left", closed="left")
    return r.agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna(subset=["open"])


# ---------------------------------------------------------------- one trade, from its record

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


def level_steps(r: dict, which: int) -> list[tuple[int, int, float]]:
    """The stop (which=1) or the target (2) as it stood over the trade: (t0, t1, price) steps in epoch ms, from the
    level path, or the initial level over the whole trade for a run without paths. A level runs from its time to
    the next one's, the last to the exit. A NaN level (none set) is left out, so the line breaks there. Times never
    go backwards: a paper level can be stamped before the entry fill, so each is clamped to the previous vertex."""
    e_t, x_t = int(r["entry_time"]), int(r["exit_time"])
    lv = _levels(r)
    times, vals = (lv[0], lv[which]) if lv else ([e_t], [_num(r, "stop_init" if which == 1 else "target")])
    at = np.maximum.accumulate(np.array([e_t, *times], dtype="int64"))[1:]
    at = np.append(at, max(at[-1], x_t)).tolist()
    return [(at[k], at[k + 1], float(v)) for k, v in enumerate(vals) if not pd.isna(v)]


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
    at = np.maximum.accumulate(np.array([e_t, *times], dtype="int64"))[1:].tolist()   # as `level_steps` does
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


def trade_tags(r: dict, b: dict, palette: dict | None, lo: int, hi: int, fmt: str) -> dict:
    """One trade's axis tags and breakeven note, from its record and its `trade_boxes` geometry `b`:
      * `price`: (word, price, colour) for Entry (blue), TP (green, none without a target) and SL (red), at the
        initial prices;
      * `time`: (ms, "HH:MM entry" / "HH:MM exit", colour, side) for the entry and exit times inside the window
        [lo, hi) (epoch ms), in UTC as `fmt` writes them; the exit takes its reason's colour;
      * `be`: (ms, price, "SL → BE HH:MM") where the stop reached the entry, if inside the window, else None."""
    p = {**PALETTE, **(palette or {})}
    (e_t, e_px), (x_t, _, reason) = b["entry"], b["exit"]
    price = [(word, float(y), color) for word, y, color in (
        ("Entry", e_px, p["s1"]), ("TP", _num(r, "target"), p["pos"]), ("SL", _num(r, "stop_init"), p["neg"]))
        if not pd.isna(y)]
    time = [(ms, f"{_utc(ms):{fmt}} {word}", color, word) for ms, word, color in (
        (e_t, "entry", p["s1"]), (x_t, "exit", exit_colors(p).get(reason, p["muted"]))) if lo <= ms < hi]
    be = b["be_time"]
    note = (be, e_px, f"SL → BE {_utc(be):{fmt}}") if be is not None and lo <= be < hi else None
    return {"price": price, "time": time, "be": note}


def exit_colors(palette: dict | None = None) -> dict:
    p = {**PALETTE, **(palette or {})}
    return {"stop": p["neg"], "target": p["s1"], "signal": p["s2"], "liq": p["liq"], "eow": p["eow"]}
