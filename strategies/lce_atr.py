"""Taken at Tradewriter's ES-sized levels in multiples of daily ATR, trades that follow his six-branch EMA-cloud flowchart at volume-profile levels during US equity hours earn more than random entries with the same exits, net of costs.

LCE (docs/superpowers/specs/2026-10-23-lce-model-design.md). Every distance is a multiple of one
unit u; in lce_pct u = 0.01169 x price. Levels are point-in-time: at each New York session-day start
D (18:00) they are drawn from the last vp_days complete session days before D, and stay fixed until
the next D.

On a bar whose close is in the entry window (09:40-15:45 New York, Mon-Fri) a flowchart branch enters
its side with the level bracket, unless the bar is big, the session day (18:00 -> 18:00 New York, of
the signal bar) already has max_trades filled entries or a win. A position moves its stop to the
entry price once its best close is be_frac of the way to the target, and exits at the close of the
15:55 bar (or of the first bar after a gap over it).
"""

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from perp_lab.indicators import ny_offset_ms, ny_session_day, ny_weekday_minute, wilder
from perp_lab.strategy import EXIT, HOLD, Signal, Strategy, enter

PARENT = "lce_pct"

# Literals of the model, not params: the sensitivity check perturbs params only.
PCT_U = 0.01169                  # lce_pct: u = 0.01169 x price (ES's median ATR(14) / close)
VP_ROWS = 200                    # visible-range profile (VRVP) rows
SVP_ROWS = 100                   # session profile rows
HVN_REACH = 2                    # an HVN is the largest row within +-2 rows ...
HVN_QUANTILE = 0.7               # ... and at least the 70th percentile of the profile
WIN_LO, WIN_HI = 0.8, 1.2        # the next level is searched for in [0.8, 1.2] x step
BEYOND = 10                      # levels past each edge of the visible range
EPS = 1e-9                       # tolerance on the search window's edges, in rows
TIE = 1e-9                       # volumes within 1e-9 x the profile's largest row are equal (ties: lowest row)
R_MIN = 0.05                     # a step under 1/20 row (thousands of levels) means a broken profile: no levels
FEATURES = ["z_dn_c", "z_dn2_c", "z_up_lo", "z_up_c", "z_up2_c", "z_dn_hi", "at_zone", "touched", "anchor_fb",
            "spacing_u"]
CLOUD_M = (1, 3, 6, 12, 24)      # the 5m, 15m, 30m, 1h and 2h clouds: EMA lengths x M on 5m bars
M_2H = 24                        # the 2h cloud: HTF and LTF wait for its slow EMA
CLOUDS = ["htf", "ltf", "ltf_prev", "flip", "fast5", "slow5"]
BRANCHES = ["sig_side", "sig_branch", "in_window", "big"]
SESSION_COLS = ["eod", "sday"]
SIDES = np.array([0.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0])      # by branch: 0 (none), 1 .. 6
ENTRY_FROM, ENTRY_TO = 9 * 60 + 40, 15 * 60 + 45                # signal-bar closes 09:40-15:45 New York
HELD_FROM, HELD_TO = 9 * 60 + 40, 15 * 60 + 55                  # positions held at closes in [09:40, 15:55)

_DAY, _HOUR = 86_400_000, 3_600_000
_BAR = 300_000                   # a 5m bar closes 5 minutes after it opens
_CHUNK = 64                      # source days binned together (keeps the work arrays in cache)
_BLOCK = 1 << 15                 # bars mapped together
_STRAGGLERS = 8                  # the grid walk finishes the last few long walks one at a time


class LceAtr(Strategy):
    params = {"ema_fast": 20, "ema_slow": 50, "vp_days": 5, "step": 0.303, "zone": 0.103,
              "big_bar": 0.303, "theta": 0.01, "wait_bars": 6, "be_frac": 0.5, "max_trades": 2, "atr_n": 14,
              "unit": "atr"}
    grid = {"vp_days": [5, 10], "theta": [0.01, 0.02]}
    timeframes = ["5m"]
    session = {"tz": "America/New_York", "days": [0, 1, 2, 3, 4], "start": "09:40", "end": "15:45"}

    def __init__(self, **kw):
        super().__init__(**kw)
        # The daily rules' memory, for entry only (manage never reads it). Keyed by the session day of
        # the signal bar. A new instance per run, as the sandbox makes.
        self._fills: dict[int, int] = {}
        self._wins: dict[int, int] = {}
        self._sent = None       # the entry sent at the last close: (side, stop, target, day)
        self._held = None       # the position held at the last close: (side, stop, target, day, exiting)

    @property
    def days(self) -> dict:
        """Per session day (of the signal bar): (filled entries, wins) so far. A copy, read-only."""
        return {d: (n, self._wins.get(d, 0)) for d, n in self._fills.items()}

    def prepare(self, bars):
        p = self.p
        u_d = u_day(bars, p["unit"], p["atr_n"])
        lv = level_features(bars, u_d, p)
        u = u_bar(bars["close"], p["unit"], u_d)
        cl = cloud_features(bars["close"], u, p)
        clock = close_clock(bars["open_time"])
        return pd.concat([lv, cl, branch_features(bars, lv, cl, u, p, clock),
                          pd.DataFrame({"u": u}, index=bars.index), session_features(bars, clock)], axis=1)

    def on_bar(self, ctx):
        """First what the engine did with the last close's signal (a fill, a win), then entry when
        flat and manage when in a position."""
        pos = ctx.position
        if self._sent is not None:
            side, stop, tgt, day = self._sent
            if pos is not None:
                self._count(day, False)
            elif (ctx["open"] - stop) * side > 0:
                # Filled at the open and closed inside this bar. The engine refuses an entry only when
                # its fill is at or beyond the stop: the open gapped through it (taken here without the
                # slippage the plugin can't see), or the liquidation price is inside the stop. That never
                # happens: levels are at most 1.2 steps apart, so a stop is under 1.8 steps (0.55 u) from
                # the close (under 0.7% of price in lce_pct, about half a daily ATR in lce_atr), and
                # liquidation at leverage 3 is about 33% away.
                self._count(day, _won(ctx, side, stop, tgt))
        elif self._held is not None and pos is None:
            side, stop, tgt, day, exiting = self._held
            if day is not None and not exiting and _won(ctx, side, stop, tgt):
                self._wins[day] = self._wins.get(day, 0) + 1
        if pos is None:
            sig = self.entry(ctx)
            self._held = None
            self._sent = None
            if sig.action in ("long", "short"):
                self._sent = (1 if sig.action == "long" else -1, sig.stop, sig.target, int(ctx["sday"]))
            return sig
        day = self._sent[3] if self._sent is not None else self._held[3] if self._held is not None else None
        sig = self.manage(ctx)
        stop, tgt = pos.stop, pos.target
        if sig.action == "hold":                        # what the engine keeps in force for the next bar
            if sig.stop is not None and (sig.stop - stop) * pos.side > 0:
                stop = sig.stop
            if sig.target is not None:
                tgt = sig.target
        self._held = (pos.side, stop, tgt, day, sig.action == "exit")
        self._sent = None
        return sig

    def _count(self, day, won):
        self._fills[day] = self._fills.get(day, 0) + 1
        if won:
            self._wins[day] = self._wins.get(day, 0) + 1

    def entry(self, ctx):
        """The bar's branch side with the bracket, in the window, on a bar that isn't big, while the
        session day has fewer than max_trades filled entries and no win."""
        side = ctx["sig_side"]
        if (side != 1 and side != -1) or ctx["in_window"] != 1 or ctx["big"] != 0:
            return HOLD
        day = int(ctx["sday"])
        if self._fills.get(day, 0) >= self.p["max_trades"] or self._wins.get(day, 0):
            return HOLD
        side = int(side)
        return enter(side, *self.bracket(ctx, side))

    def manage(self, ctx):
        """EXIT at eod. Otherwise the stop to the entry price once best_close is be_frac of the way to
        the target, unless the stop is there or past it already. Reads ctx only: RandomEntry calls it
        without ever calling entry."""
        if ctx["eod"] == 1:
            return EXIT
        pos = ctx.position
        s, e = pos.side, pos.entry_price
        if s * (pos.stop - e) < 0 and s * (pos.best_close - e) >= self.p["be_frac"] * s * (pos.target - e):
            return Signal("hold", float(e))
        return HOLD

    def bracket(self, ctx, side):
        """Long: stop halfway between the base level Z (the highest centre <= close) and the level
        below it, target the lower edge of the lowest zone above the close; short mirrored. A NaN stop
        (no entry) unless the bar is in the entry window and stop < close < target (target < close <
        stop for a short), and for any side other than +1 or -1."""
        c = ctx["close"]
        if side == 1:
            stop, tgt = (ctx["z_dn_c"] + ctx["z_dn2_c"]) / 2, ctx["z_up_lo"]
            ok = stop < c < tgt
        elif side == -1:
            stop, tgt = (ctx["z_up_c"] + ctx["z_up2_c"]) / 2, ctx["z_dn_hi"]
            ok = tgt < c < stop
        else:
            return math.nan, None
        if ok and ctx["in_window"] == 1 and math.isfinite(stop) and math.isfinite(tgt) and math.isfinite(c):
            return float(stop), float(tgt)
        return math.nan, None


def _won(ctx, side, stop, target) -> bool:
    """Whether a position closed in this bar closed at its target, as the engine decides: the stop
    first (at the open, then the bar's range), and a target needs a trade through it. (Liquidation is
    left out: see on_bar.)"""
    if side > 0:
        return ctx["low"] > stop and ctx["high"] > target
    return ctx["high"] < stop and ctx["low"] < target


# ---------------------------------------------------------------- session days

def day_starts(open_time) -> np.ndarray:
    """Start index of every New York session day (18:00 -> 18:00) from the first bar's day to the last
    bar's, then len(bars). Dense over day numbers, so a day with no bars has an empty range.

    Equal to grouping by `ny_session_day`, but it evaluates the clock once per day instead of once
    per bar. open_time (UTC ms) must be ascending."""
    t = np.asarray(open_time, dtype=np.int64)
    if len(t) == 0:
        return np.zeros(1, dtype=np.int64)
    first, last = ny_session_day(t[[0, -1]])
    eve = (np.arange(first + 1, last + 1, dtype=np.int64) - 1) * _DAY       # midnight UTC before each day
    # day N starts at 18:00 New York on date N-1; its offset is read at 22:30 UTC, between the EDT
    # (22:00) and EST (23:00) starts and hours away from any daylight-saving switch
    start = eve + 18 * _HOUR - ny_offset_ms(eve + 22 * _HOUR + _HOUR // 2)
    return np.concatenate(([0], np.searchsorted(t, start), [len(t)])).astype(np.int64)


def _whole(complete: np.ndarray, m: int) -> np.ndarray:
    """Per day d: days d-m .. d-1 are all complete."""
    k = len(complete)
    c = np.concatenate(([0], np.cumsum(complete)))
    out = np.zeros(k, dtype=bool)
    if k > m:
        d = np.arange(m, k)
        out[m:] = c[d] - c[d - m] == m
    return out


def _complete(cnt: np.ndarray) -> np.ndarray:
    """Days with bars, except the first day, which may be partial."""
    c = cnt > 0
    if len(c):
        c[0] = False
    return c


# ---------------------------------------------------------------- u

def u_day(bars: pd.DataFrame, unit: str, atr_n: int) -> np.ndarray:
    """Per bar, the u of its session day, from bars before that day only.

    "pct": 0.01169 x the close of the last bar before the day starts.
    "atr": Wilder ATR(atr_n) of the session-day OHLC of the complete days before it; NaN unless the
    atr_n days just before are all complete (present, and not the first day)."""
    t = bars["open_time"].to_numpy(np.int64)
    ds = day_starts(t)
    cnt = np.diff(ds)
    k = len(cnt)
    close = bars["close"].to_numpy(float)
    ud = np.full(k, np.nan)
    if unit == "pct":
        if k > 1:
            ud[1:] = PCT_U * close[ds[1:-1] - 1]
    elif unit == "atr":
        idx = np.flatnonzero(cnt > 0)
        if len(idx) > 1:
            st = ds[idx]
            dh = np.maximum.reduceat(bars["high"].to_numpy(float), st)[1:]
            dl = np.minimum.reduceat(bars["low"].to_numpy(float), st)[1:]
            pc = close[ds[idx + 1] - 1][:-1]                                 # the previous day's close
            tr = np.maximum(dh - dl, np.maximum(np.abs(dh - pc), np.abs(dl - pc)))
            after = np.full(k, np.nan)
            after[idx[1:]] = wilder(pd.Series(tr), atr_n).to_numpy()
            ud[1:] = after[:-1]
        ud[~_whole(_complete(cnt), atr_n)] = np.nan
    else:
        raise ValueError(f"unknown unit {unit!r}: 'pct' or 'atr'")
    return np.repeat(ud, cnt)


# ---------------------------------------------------------------- profiles

def _density(lo, hi, vol):
    """Per bar: volume per unit price (0 for a flat bar), and a flat bar's volume (else 0)."""
    w = hi - lo
    return np.divide(vol, w, out=np.zeros(len(w)), where=w > 0), np.where(w > 0, 0.0, vol)


def _fixed(total):
    """Fixed-point scale (a power of two) under which total volume stays below 2^50: interior row
    volumes are summed as integers in float64, so they are exact (an empty row is exactly 0, rows
    covered by the same bars are exactly equal)."""
    return np.ldexp(1.0, 50 - np.frexp(total)[1])


class _Work:
    """Work arrays for _spread, reused across passes."""

    __slots__ = ("x", "fl", "c", "m", "i", "w", "d")

    def __init__(self, shape):
        self.x, self.fl = np.empty((2, *shape)), np.empty((2, *shape), dtype=np.intp)
        self.c, self.m = np.empty(shape, dtype=np.intp), np.empty(shape, dtype=np.intp)
        self.i, self.w, self.d = np.empty((2, *shape), dtype=np.intp), np.empty((2, *shape)), np.empty((2, *shape))


def _spread(lh, vb, vf, edge, scale, hq, base, b):
    """bincount entries that spread each bar's volume evenly over the rows its [lo, hi] covers.
    lh = [lo, hi]; rows are unit steps of x = (price - edge) x scale (x >= 0; row floor(x)); hq = the
    row height x the fixed-point scale. Every row the bar covers fully, from row ceil(x_lo) up to its
    last row, gets one row's worth in fixed point; its first row gets what lies below ceil(x_lo) (nothing
    when x_lo is a row edge) and its last row what lies in it (a bar inside one row: all of it, in the
    first row's entry plus a negative last one), in units of the bar's volume per unit price (the
    caller multiplies the binned rows by the row height). Fills b.i = [first, last] + base with
    b.w = those amounts, b.c with ceil(x_lo), and b.d = [+1 row, -1 row] in fixed point: a difference
    array the caller bins at [ceil(x_lo), last]. vf (flat bars' volume) may be None when there are no
    flat bars."""
    x, fl, c, m = b.x, b.fl, b.c, b.m
    np.subtract(lh, edge, out=x)
    np.multiply(x, scale, out=x)
    np.copyto(fl, x, casting="unsafe")                                      # floor, as x >= 0
    (xl, xh), (first, last), (w0, w1) = x, fl, b.w
    np.add(first, xl > first, out=c)                                        # ceil(x_lo): the first full row
    np.subtract(c, xl, out=w0)
    w0 *= vb
    if vf is not None:
        w0 += vf * scale
    np.maximum(last, c, out=m)                                              # c = last + 1 for a bar inside one
    np.subtract(xh, m, out=w1)                                              # row: w1 < 0 takes back w0's excess
    w1 *= vb
    np.subtract(last, c, out=m)                                             # full rows c .. last - 1
    d0, d1 = b.d
    np.multiply(vb, hq, out=d0)
    np.rint(d0, out=d0)
    d0 *= m >= 1
    np.negative(d0, out=d1)
    np.add(fl, base, out=b.i)


def _pass(lh, vb, vf, grid, r, b=None):
    """Each row of bars (a day) spread on its own grid (edge, scale, h, hq; columns): row volumes in
    two parts, direct and fixed-point interior entries, (days, r) each. b: work arrays, if at hand."""
    n = lh.shape[1]
    b = b or _Work(lh.shape[1:])
    edge, scale, h, hq = grid
    base = (np.arange(n) * r)[:, None]
    _spread(lh, vb, vf, edge, scale, hq, base, b)
    ii = b.i.reshape(-1)
    direct = np.bincount(ii, weights=b.w.reshape(-1), minlength=n * r).reshape(n, r)
    direct *= h
    np.add(b.c, base, out=b.i[0])
    return direct, np.bincount(ii, weights=b.d.reshape(-1), minlength=n * r).reshape(n, r)


def _rows(direct, inner, q, rows):
    """Row volumes from _pass: the direct part plus the cumulated interior part; the top edge's row
    (x == rows) folds into the last row."""
    p = direct + np.cumsum(inner, axis=-1) / q
    p[..., rows - 1] += p[..., rows]
    return p[..., :rows]


def profile(lo, hi, vol, lo_edge, hi_edge, rows) -> np.ndarray:
    """Volume per row of [lo_edge, hi_edge] (rows equal rows), each bar's volume spread evenly over the
    rows its [lo, hi] covers, in proportion to the overlap; a flat bar's goes to its price's row,
    floor((price - lo_edge) / (hi_edge - lo_edge) x rows). Prices outside the edges are clipped to
    them. Zeros when hi_edge <= lo_edge or there is no volume."""
    vol = np.asarray(vol, float)
    total = vol.sum()
    if not (hi_edge > lo_edge and total > 0):
        return np.zeros(rows)
    lh = np.clip(np.array([lo, hi], dtype=float), lo_edge, hi_edge)[:, None, :]
    vb, vf = _density(lh[0, 0], lh[1, 0], vol)
    h, q = (hi_edge - lo_edge) / rows, _fixed(total)
    grid = [np.array([[v]]) for v in (lo_edge, rows / (hi_edge - lo_edge), h, h * q)]
    direct, inner = _pass(lh, vb[None], vf[None] if vf.any() else None, grid, rows + 2)
    return _rows(direct[0], inner[0], q, rows)


def poc(lo, hi, vol) -> float:
    """Centre of the highest row of a day's 100-row profile over its own range (NaN without range or
    volume)."""
    lo_e, hi_e = np.min(lo), np.max(hi)
    if not (hi_e > lo_e and np.sum(vol) > 0):
        return math.nan
    p = profile(lo, hi, vol, lo_e, hi_e, SVP_ROWS)
    return lo_e + (_top(p[None])[0] + 0.5) * ((hi_e - lo_e) / SVP_ROWS)


def _top(p):
    """Per row of p, the lowest index whose volume ties the largest (within TIE)."""
    mx = p.max(1, keepdims=True)
    return (p >= mx - TIE * mx).argmax(1)


def hvn(p) -> np.ndarray:
    """High-volume nodes of a profile (or of each row of a 2-D stack): the largest row within +-2 rows
    (ties count), at least the 70th percentile of the profile, and not empty; within TIE x the
    profile's largest row counts as equal."""
    p = np.asarray(p, float)
    p2 = np.atleast_2d(p)
    n = p2.shape[1]
    pad = np.pad(p2, ((0, 0), (HVN_REACH, HVN_REACH)), constant_values=-1.0)
    mx = pad[:, :n]
    for k in range(1, 2 * HVN_REACH + 1):
        mx = np.maximum(mx, pad[:, k:k + n])
    tol = TIE * p2.max(1, keepdims=True)
    out = (p2 >= mx - tol) & (p2 >= np.quantile(p2, HVN_QUANTILE, axis=1, keepdims=True) - tol) & (p2 > 0)
    return out[0] if p.ndim == 1 else out


def _bin(lo, hi, vb, vf, ds, e0, e1, vp_days, wg, dg):
    """profile()'s spreading for many windows at once. Source day e in [e0, e1) adds its bars to the
    VRVP of days e+1 .. e+vp_days (window grids wg = (edge, scale, h, hq) per day) and to its own SVP
    (grids dg per day). A grid of zeros bins everything into row 0 (callers ignore it). Returns the
    direct and interior entries: (vd, vi) of shape (days + vp_days + 1, 202), (sd, si) (days, 102).

    A window whose grid equals the previous window's gets each shared day's entries unchanged from
    it, so the day isn't spread again (consecutive windows often share their low, high and scale)."""
    k = len(ds) - 1
    r1, r2 = VP_ROWS + 2, SVP_ROWS + 2
    vd, vi = np.zeros((k + vp_days + 1, r1)), np.zeros((k + vp_days + 1, r1))
    sd, si = np.zeros((k, r2)), np.zeros((k, r2))
    same = np.zeros(k + vp_days + 1, dtype=bool)
    same[1:] = (wg[:, 1:] == wg[:, :-1]).all(0) & (wg[1, 1:] > 0)
    cnt = np.diff(ds)
    for c0 in range(e0, e1, _CHUNK):
        c1 = min(e1, c0 + _CHUNK)
        width = int(cnt[c0:c1].max())
        if width == 0:
            continue
        ar = np.arange(width)
        live = ar < cnt[c0:c1, None]
        idx = np.where(live, ds[c0:c1, None] + ar, np.minimum(ds[c0:c1, None], len(lo) - 1))   # pads: no volume
        bars = [np.stack([lo[idx], hi[idx]]), np.where(live, vb[idx], 0.0), np.where(live, vf[idx], 0.0)]
        if not bars[2].any():
            bars[2] = None
        work = _Work(idx.shape)
        d, n = _pass(*bars, [x[c0:c1, None] for x in dg], r2, work)
        sd[c0:c1] += d
        si[c0:c1] += n
        for j in range(1, vp_days + 1):
            g = slice(c0 + j, c1 + j)
            new = ~same[g] if j > 1 else np.ones(c1 - c0, dtype=bool)
            if new.all():
                d, n = _pass(*bars, [x[g, None] for x in wg], r1, work)
            elif new.any():
                sel = np.flatnonzero(new)
                d, n = d.copy(), n.copy()
                d[sel], n[sel] = _pass(bars[0][:, sel], bars[1][sel], None if bars[2] is None else bars[2][sel],
                                       [x[g][sel, None] for x in wg], r1)
            vd[g] += d
            vi[g] += n
    return vd, vi, sd, si


def _profiles(lo, hi, vol, ds, vp_days, want):
    """VRVP of each wanted day's window (days d-vp_days .. d-1, all complete, with range and volume)
    and every source day's POC. Returns (vp (days, 200), poc, L, H, good): vp rows and L/H are
    meaningful only where `good`; poc is NaN for days without range or volume or outside the windows."""
    k = len(ds) - 1
    cnt = np.diff(ds)
    pres = cnt > 0
    st = ds[:-1][pres]
    dlo, dhi, dvol = np.full(k, np.inf), np.full(k, -np.inf), np.zeros(k)
    if len(st):
        dlo[pres] = np.minimum.reduceat(lo, st)
        dhi[pres] = np.maximum.reduceat(hi, st)
        dvol[pres] = np.add.reduceat(vol, st)
    whole = _whole(_complete(cnt), vp_days)
    lw, hw, vt = np.full(k, np.nan), np.full(k, np.nan), np.zeros(k)
    if k > vp_days:
        lw[vp_days:] = sliding_window_view(dlo, vp_days).min(1)[:k - vp_days]
        hw[vp_days:] = sliding_window_view(dhi, vp_days).max(1)[:k - vp_days]
        vt[vp_days:] = sliding_window_view(dvol, vp_days).sum(1)[:k - vp_days]
    lw[~whole] = np.nan
    hw[~whole] = np.nan
    good = whole & (hw > lw) & (vt > 0)
    if want is not None:
        good &= want
    vp = np.zeros((k, VP_ROWS))
    pc = np.full(k, np.nan)
    gi = np.flatnonzero(good)
    if not len(gi):
        return vp, pc, lw, hw, good
    wg = np.zeros((4, k + vp_days + 1))                                     # edge, scale, h, h x q
    wg[0, gi] = lw[gi]
    wg[1, gi] = VP_ROWS / (hw[gi] - lw[gi])
    wg[2, gi] = (hw[gi] - lw[gi]) / VP_ROWS
    wq = _fixed(vt[gi])
    wg[3, gi] = wg[2, gi] * wq
    sday = pres & (dhi > dlo) & (dvol > 0)
    sday[:max(int(gi[0]) - vp_days, 0)] = False
    sday[int(gi[-1]):] = False
    si_ = np.flatnonzero(sday)
    dg = np.zeros((4, k))
    dg[0, si_] = dlo[si_]
    dg[1, si_] = SVP_ROWS / (dhi[si_] - dlo[si_])
    dg[2, si_] = (dhi[si_] - dlo[si_]) / SVP_ROWS
    dq = _fixed(dvol[si_])
    dg[3, si_] = dg[2, si_] * dq
    vb, vf = _density(lo, hi, vol)
    vd, vi, sd, sn = _bin(lo, hi, vb, vf, ds, max(int(gi[0]) - vp_days, 0), int(gi[-1]), vp_days, wg, dg)
    vp[gi] = _rows(vd[gi], vi[gi], wq[:, None], VP_ROWS)
    row = _top(_rows(sd[si_], sn[si_], dq[:, None], SVP_ROWS))
    pc[si_] = dlo[si_] + (row + 0.5) * ((dhi[si_] - dlo[si_]) / SVP_ROWS)
    return vp, pc, lw, hw, good


# ---------------------------------------------------------------- anchor and grid

def _anchor(vp, lo_edge, h, pocs, hz):
    """Per window, the anchor row: the HVN with the most POCs within +-hz of its centre, then the
    higher volume (within TIE: equal), then the lower row; and whether no HVN had any POC that close
    (the fallback). vp (windows, rows) must have volume in every window."""
    w, j = np.nonzero(hvn(vp))                                              # by window, then row
    c = lo_edge[w] + (j + 0.5) * h[w]
    near = (np.abs(pocs[w] - c[:, None]) <= hz[w, None]).sum(1)             # a NaN POC never counts
    vol = vp[w, j]
    starts = np.searchsorted(w, np.arange(len(vp)))
    best = np.maximum.reduceat(near, starts)
    cand = near == best[w]
    top = np.maximum.reduceat(np.where(cand, vol, -1.0), starts)
    cand &= vol >= top[w] - TIE * vp.max(1)[w]
    return np.minimum.reduceat(np.where(cand, j, vp.shape[1]), starts), best == 0


def _walk_one(p, x, r, up):
    """Reference walk for one window, in rows: from the level at row coordinate x, the next level is
    the highest-volume row whose centre is in [x + 0.8 r, x + 1.2 r] (ties, within TIE: the lowest
    row), or x + r if no row centre of the profile is in it; it ends when the window lies above the
    profile. Mirrored downwards. r = step / row height."""
    a_, b_ = WIN_LO * r, WIN_HI * r
    p = list(p)
    tol = TIE * max(p)
    top = len(p) - 1
    out = []
    while True:
        if up:
            a, b = math.ceil(x + a_ - EPS), min(math.floor(x + b_ + EPS), top)
            if a > top:
                return out
        else:
            a, b = max(math.ceil(x - b_ - EPS), 0), math.floor(x - a_ + EPS)
            if b < 0:
                return out
        if a <= b:
            seg = p[a:b + 1]
            mx = max(seg) - tol
            x = float(a + next(k for k, v in enumerate(seg) if v >= mx))
        else:
            x = x + r if up else x - r
        out.append(x)


def _walk(vp, x0, r, up):
    """_walk_one for every window at once (the last few long walks finish one at a time). Returns
    (windows, steps) row coordinates, left-aligned, NaN after each window's last level."""
    nv, rows = vp.shape
    top = rows - 1
    a_, b_ = WIN_LO * r, WIN_HI * r
    tol = TIE * vp.max(1)
    x = np.asarray(x0, float).copy()
    act = np.arange(nv)
    steps = []
    while len(act) > _STRAGGLERS:
        xa = x[act]
        if up:
            a = np.ceil(xa + a_[act] - EPS)
            b = np.minimum(np.floor(xa + b_[act] + EPS), top)
            go = a <= top
        else:
            a = np.maximum(np.ceil(xa - b_[act] - EPS), 0)
            b = np.floor(xa - a_[act] + EPS)
            go = b >= 0
        act, a, b, xa = act[go], a[go].astype(np.intp), b[go].astype(np.intp), xa[go]
        if not len(act):
            break
        snap = a <= b
        si = np.flatnonzero(snap)
        if len(si):
            wid = int((b[si] - a[si]).max()) + 1
            cols = a[si, None] + np.arange(wid)
            vals = vp[act[si, None], np.minimum(cols, top)]
            vals[cols > b[si, None]] = -np.inf
            xa[si] = a[si] + (vals >= (vals.max(1) - tol[act[si]])[:, None]).argmax(1)
        ns = np.flatnonzero(~snap)
        xa[ns] += r[act[ns]] if up else -r[act[ns]]
        x[act] = xa
        rec = np.full(nv, np.nan)
        rec[act] = xa
        steps.append(rec)
    rest = {int(i): _walk_one(vp[i], float(x[i]), float(r[i]), up) for i in act}
    out = np.full((nv, len(steps) + max(map(len, rest.values()), default=0) + 1), np.nan)
    if steps:
        out[:, :len(steps)] = np.array(steps).T
    for i, v in rest.items():
        out[i, len(steps):len(steps) + len(v)] = v
    return out


def _tail(c, s, edge, nb, up):
    """Levels c +- k s (k = 1, 2, ...) until BEYOND of them, with the nb walk levels already there, lie
    beyond the edge. Returns (values (windows, k), how many each window takes)."""
    sg = 1.0 if up else -1.0
    k0 = np.maximum(np.floor(sg * (edge - c) / s) + 1, 1)                  # first k past the edge
    k0 += sg * (c + sg * k0 * s - edge) <= 0
    k0 -= (k0 > 1) & (sg * (c + sg * (k0 - 1) * s - edge) > 0)
    n = (k0 - 1 + np.maximum(BEYOND - nb, 0)).astype(np.intp)
    k = np.arange(1, int(n.max()) + 1 if len(n) else 1)
    return c[:, None] + sg * k * s[:, None], n


def _levels(lo, hi, vol, ds, wdays, vp_days, step_u, zone_u, guards=0):
    """Levels of the session days `wdays` (ascending indices into ds), with per-day step and zone in
    price. Returns (lev, start, count, fb, spacing): day i's sorted level centres are
    lev[start[i] + guards : start[i] + guards + count[i]], between `guards` NaNs on each side; fb is
    the anchor fallback; spacing the median gap between its levels inside the visible range."""
    wdays = np.asarray(wdays, dtype=np.intp)
    nw = len(wdays)
    su = np.broadcast_to(np.asarray(step_u, float), (nw,))
    zu = np.broadcast_to(np.asarray(zone_u, float), (nw,))
    want = np.zeros(len(ds) - 1, dtype=bool)
    want[wdays[np.isfinite(su) & (su > 0) & np.isfinite(zu) & (zu >= 0)]] = True
    vp, pc, lw, hw, good = _profiles(lo, hi, vol, ds, vp_days, want)
    h = (hw - lw) / VP_ROWS
    ok = good[wdays]
    ok[ok] = su[ok] / h[wdays[ok]] >= R_MIN
    oi = np.flatnonzero(ok)
    w = wdays[oi]
    count = np.zeros(nw, dtype=np.intp)
    fb = np.zeros(nw, dtype=bool)
    spacing = np.full(nw, np.nan)
    if len(oi):
        pv = vp[w]
        lo_e, hi_e, hh, s = lw[w], hw[w], h[w], su[oi]
        a, fb[oi] = _anchor(pv, lo_e, hh, pc[w[:, None] - vp_days + np.arange(vp_days)], zu[oi] / 2)
        ups, dns = _walk(pv, a, s / hh, True), _walk(pv, a, s / hh, False)
        nu, nd = (~np.isnan(ups)).sum(1), (~np.isnan(dns)).sum(1)
        ix = np.arange(len(oi))
        up_p = lo_e[:, None] + (ups + 0.5) * hh[:, None]
        dn_p = lo_e[:, None] + (dns + 0.5) * hh[:, None]
        an_p = lo_e + (a + 0.5) * hh
        ce = np.where(nu > 0, up_p[ix, np.maximum(nu - 1, 0)], an_p)
        cb = np.where(nd > 0, dn_p[ix, np.maximum(nd - 1, 0)], an_p)
        tu, ntu = _tail(ce, s, hi_e, (up_p > hi_e[:, None]).sum(1), True)
        td, ntd = _tail(cb, s, lo_e, (dn_p < lo_e[:, None]).sum(1), False)
        count[oi] = ntd + nd + 1 + nu + ntu
    seg = count + 2 * guards
    start = np.cumsum(seg) - seg
    lev = np.full(int(seg.sum()), np.nan)
    if len(oi):
        base = start[oi] + guards
        i, k = np.nonzero(np.arange(td.shape[1]) < ntd[:, None])
        lev[base[i] + ntd[i] - 1 - k] = td[i, k]                            # lowest first
        i, k = np.nonzero(~np.isnan(dns))
        lev[base[i] + ntd[i] + nd[i] - 1 - k] = dn_p[i, k]
        mid = base + ntd + nd
        lev[mid] = an_p
        i, k = np.nonzero(~np.isnan(ups))
        lev[mid[i] + 1 + k] = up_p[i, k]
        i, k = np.nonzero(np.arange(tu.shape[1]) < ntu[:, None])
        lev[mid[i] + 1 + nu[i] + k] = tu[i, k]
        # median gap between consecutive levels inside [L, H], per day
        c = count[oi]
        wid = np.repeat(oi, c)
        x = lev[np.repeat(base - (np.cumsum(c) - c), c) + np.arange(c.sum())]   # each day's levels, in order
        ins = (x >= lw[wdays[wid]]) & (x <= hw[wdays[wid]])
        gv = ins[:-1] & ins[1:] & (wid[:-1] == wid[1:])
        g, gw = np.diff(x)[gv], wid[:-1][gv]
        g = g[np.lexsort((g, gw))]
        m = np.bincount(gw, minlength=nw)
        gs = np.cumsum(m) - m
        has = m > 0
        spacing[has] = (g[gs[has] + (m[has] - 1) // 2] + g[gs[has] + m[has] // 2]) / 2
    return lev, start, count, fb, spacing


def day_levels(lo, hi, vol, day_start, d, vp_days, step_u, zone_u):
    """Sorted level centres for session day d (an index into day_start, the output of `day_starts`)
    and whether its anchor fell back. Empty unless the vp_days days before d are complete and their
    window has range and volume."""
    ds = np.asarray(day_start, dtype=np.int64)
    if not 0 <= d < len(ds) - 1:
        return np.empty(0), False
    lev, start, count, fb, _ = _levels(np.asarray(lo, float), np.asarray(hi, float), np.asarray(vol, float), ds,
                                       [d], int(vp_days), [step_u], [zone_u])
    return lev[start[0]:start[0] + count[0]].copy(), bool(fb[0])


# ---------------------------------------------------------------- per-bar features

def level_features(bars: pd.DataFrame, u_d: np.ndarray, p: dict) -> pd.DataFrame:
    """Per bar, its session day's zones (centre +- zone u / 2, u taken at the day's first bar) seen
    from the close: the nearest centres below and above, their neighbours, the nearest edges strictly
    beyond the close, whether the bar's range overlaps a zone (at_zone) and did on any of the last
    wait_bars bars (touched), the anchor fallback and the median level spacing in u. All NaN on days
    without levels; level columns NaN past the outermost level."""
    n = len(bars)
    out = np.empty((n, len(FEATURES)), order="F")                       # column-major: one block, no copy
    if n == 0:
        return pd.DataFrame(out, index=bars.index, columns=FEATURES)
    lo, hi = bars["low"].to_numpy(float), bars["high"].to_numpy(float)
    close = bars["close"].to_numpy(float)
    ds = day_starts(bars["open_time"].to_numpy(np.int64))
    cnt = np.diff(ds)
    days = np.flatnonzero(cnt > 0)
    per = cnt[days]
    u = np.asarray(u_d, float)[ds[days]]
    zu = p["zone"] * u
    g = 2
    lev, start, count, fb, spacing = _levels(lo, hi, bars["volume"].to_numpy(float), ds, days, int(p["vp_days"]),
                                             p["step"] * u, zu, guards=g)
    has = count > 0
    hzd = zu / 2
    fbd = np.where(has, fb, np.nan)
    sud = np.full(len(days), np.nan)
    sud[has] = spacing[has] / u[has]
    first = start + g                                                       # each day's lowest level
    w = int(p["wait_bars"])
    at = np.empty(n, dtype=bool)
    bar0, bar1 = ds[days], ds[days + 1]
    cuts = np.unique(np.r_[0, np.searchsorted(bar0, np.arange(_BLOCK, n, _BLOCK)), len(days)])
    for k0, k1 in zip(cuts[:-1].tolist(), cuts[1:].tolist()):             # blocks of whole days
        a, b = int(bar0[k0]), int(bar1[k1 - 1])
        pk = per[k0:k1]
        cl = close[a:b]
        i = np.repeat(first[k0:k1], pk)                                     # per bar: its day's lowest level
        for x0, x1, s0, c in zip((bar0[k0:k1] - a).tolist(), (bar1[k0:k1] - a).tolist(), first[k0:k1].tolist(),
                                 count[k0:k1].tolist()):
            if c:
                i[x0:x1] += lev[s0:s0 + c].searchsorted(cl[x0:x1], side="right")
        o, hb, on = out[a:b], np.repeat(hzd[k0:k1], pk), np.repeat(has[k0:k1], pk)
        d1, d2, u1, u2 = lev[i - 1], lev[i - 2], lev[i], lev[i + 1]        # i: the lowest level above the close
        o[:, 0] = d1                                                        # highest centre <= close
        o[:, 1] = d2
        up_lo, dn_hi = u1 - hb, d1 + hb
        np.subtract(u2, hb, out=o[:, 2])                                    # lowest lower edge > close
        np.copyto(o[:, 2], up_lo, where=up_lo > cl)
        o[:, 3] = u1                                                        # lowest centre >= close
        o[:, 4] = u2
        eq = np.flatnonzero(d1 == cl)
        o[eq, 3], o[eq, 4] = d1[eq], u1[eq]
        np.add(d2, hb, out=o[:, 5])                                         # highest upper edge < close
        np.copyto(o[:, 5], dn_hi, where=dn_hi < cl)
        ab = at[a:b]
        np.logical_or(dn_hi >= lo[a:b], up_lo <= hi[a:b], out=ab)          # the nearest zones either side
        ab &= on
        a0 = max(a - w + 1, 0)                                              # touched: at_zone on bars t-w+1 .. t
        cs = np.concatenate(([0], np.cumsum(at[a0:b], dtype=np.int64)))
        j = np.arange(a - a0 + 1, b - a0 + 1)
        seen = cs[j] - cs[np.maximum(j - w, 0)]
        o[:, 6] = ab
        o[:, 7] = seen > 0
        off = ~on
        np.copyto(o[:, 6], np.nan, where=off)
        np.copyto(o[:, 7], np.nan, where=off)
        o[:, 8] = np.repeat(fbd[k0:k1], pk)
        o[:, 9] = np.repeat(sud[k0:k1], pk)
    return pd.DataFrame(out, index=bars.index, columns=FEATURES, copy=False)


# ---------------------------------------------------------------- clouds

def u_bar(close: pd.Series, unit: str, u_d: np.ndarray) -> np.ndarray:
    """Per bar, the u of the cloud slopes and big: 0.01169 x the bar's close ("pct"), or its session
    day's u_d ("atr")."""
    if unit == "pct":
        return PCT_U * np.asarray(close, dtype=float)
    if unit == "atr":
        return np.array(u_d, dtype=float)
    raise ValueError(f"unknown unit {unit!r}: 'pct' or 'atr'")


def _shift(x, k):
    """x delayed by k bars (NaN before)."""
    out = np.full(len(x), np.nan)
    out[k:] = x[:max(len(x) - k, 0)]
    return out


def _signed(pos, neg, *inputs):
    """+1 where pos, -1 where neg (never both), else 0; NaN where any of the inputs is NaN."""
    out = pos.astype(float)
    out -= neg
    nan = np.isnan(inputs[0])
    for x in inputs[1:]:
        nan |= np.isnan(x)
    np.copyto(out, np.nan, where=nan)
    return out


def _state(fast, slow, slope, theta):
    """+1: fast above slow and the cloud rising at theta or more; -1 mirrored; else 0. NaN if any is."""
    return _signed((fast > slow) & (slope >= theta), (fast < slow) & (slope <= -theta), fast, slow, slope)


def _htf(s30, s1h, s2h):
    """Higher timeframe: +1 if at least two of the 30m, 1h and 2h clouds are +1 and none is -1; -1
    mirrored; else 0 (ranging). NaN if any is."""
    up = (s30 == 1).astype(np.int8) + (s1h == 1) + (s2h == 1)
    dn = (s30 == -1).astype(np.int8) + (s1h == -1) + (s2h == -1)
    return _signed((up >= 2) & (dn == 0), (dn >= 2) & (up == 0), s30, s1h, s2h)


def _ltf(s5, s15):
    """Lower timeframe: +1 if the 5m and 15m clouds are both +1; -1 if both -1; else 0 (choppy). NaN if
    either is."""
    return _signed((s5 == 1) & (s15 == 1), (s5 == -1) & (s15 == -1), s5, s15)


def _flip(fast, slow):
    """+1 where the fast EMA ends above the slow from at or below it (bullish), -1 mirrored, else 0. NaN
    where this bar's or the previous bar's EMAs are."""
    f0, s0 = _shift(fast, 1), _shift(slow, 1)
    return _signed((fast > slow) & (f0 <= s0), (fast < slow) & (f0 >= s0), fast, slow, f0, s0)


def cloud_features(close: pd.Series, u: np.ndarray, p: dict) -> pd.DataFrame:
    """Per bar: the HTF and LTF states (and the LTF one bar earlier), the 5m flip, and the 5m cloud's
    EMAs. Cloud k is EMA(ema_fast x M) against EMA(ema_slow x M) of the 5m closes; its slope is the
    change of the cloud's mid over M bars in u per hour. A cloud without u (NaN or not positive) has no
    state. HTF, LTF and LTF(t-1) are NaN until the 2h slow EMA exists."""
    c = pd.Series(np.asarray(close, dtype=float))
    n = len(c)
    u = np.asarray(u, dtype=float)
    has_u = np.isfinite(u) & (u > 0)
    st = {}
    for m in CLOUD_M:
        nf, ns = int(p["ema_fast"]) * m, int(p["ema_slow"]) * m
        fast = c.ewm(span=nf, adjust=False, min_periods=nf).mean().to_numpy()
        slow = c.ewm(span=ns, adjust=False, min_periods=ns).mean().to_numpy()
        mid = (fast + slow) / 2
        slope = np.full(n, np.nan)
        np.divide(mid[m:] - mid[:max(n - m, 0)], u[m:], out=slope[m:], where=has_u[m:])
        slope *= 12 / m
        st[m] = _state(fast, slow, slope, p["theta"])
        if m == 1:
            fast5, slow5 = fast, slow
        if m == M_2H:
            warm = np.isnan(slow)                                           # until the 2h slow EMA exists
    out = np.empty((n, len(CLOUDS)), order="F")                             # column-major: one block, no copy
    out[:, 0] = _htf(st[6], st[12], st[24])
    ltf = out[:, 1]
    ltf[:] = _ltf(st[1], st[3])
    np.copyto(out[:, 0], np.nan, where=warm)
    np.copyto(ltf, np.nan, where=warm)
    out[:, 2] = _shift(ltf, 1)
    out[:, 3] = _flip(fast5, slow5)
    out[:, 4], out[:, 5] = fast5, slow5
    return pd.DataFrame(out, index=getattr(close, "index", None), columns=CLOUDS, copy=False)


# ---------------------------------------------------------------- branches

def close_clock(open_time):
    """New York weekday (Mon = 0) and minute of the day of each 5m bar's close (open time + 5 min)."""
    return ny_weekday_minute(np.asarray(open_time, dtype=np.int64) + _BAR)


def branch_features(bars: pd.DataFrame, lv: pd.DataFrame, cl: pd.DataFrame, u: np.ndarray, p: dict,
                    clock=None) -> pd.DataFrame:
    """Per bar: the flowchart branch (1-6; 0 for none) and its side, the first matching row winning;
    whether the bar's close lies in the entry window; and whether its body is too big to chase
    (|close - open| > big_bar x u; NaN without u). The branch ignores the window, big bars and the
    daily rules (the entry applies them), and is 0 where any of its inputs is NaN. clock: the bars'
    close_clock, if at hand."""
    lo, hi = bars["low"].to_numpy(float), bars["high"].to_numpy(float)
    op, c = bars["open"].to_numpy(float), bars["close"].to_numpy(float)
    htf, ltf, lp, fl, f5, s5 = (cl[k].to_numpy(float) for k in CLOUDS)
    at, tch = lv["at_zone"].to_numpy(float), lv["touched"].to_numpy(float)
    rows = [(htf == 1) & (ltf == 1) & (at == 1) & (lo <= f5) & (c > s5),      # dipped into the cloud, held
            (htf == 1) & (fl == 1) & (tch == 1),
            (htf == 0) & (lp == 1) & (fl == -1) & (tch == 1),
            (htf == 0) & (lp == -1) & (fl == 1) & (tch == 1),
            (htf == -1) & (fl == -1) & (tch == 1),
            (htf == -1) & (ltf == -1) & (at == 1) & (hi >= f5) & (c < s5)]    # ripped into it, failed
    known = np.ones(len(c), dtype=bool)
    for x in (htf, ltf, lp, fl, f5, s5, at, tch, lo, hi, c):
        known &= ~np.isnan(x)
    br = np.select(rows, [1, 2, 3, 4, 5, 6], 0) * known
    wd, mi = close_clock(bars["open_time"]) if clock is None else clock
    inw = (wd < 5) & (mi >= ENTRY_FROM) & (mi <= ENTRY_TO)
    u = np.asarray(u, dtype=float)
    big = np.where(np.isnan(u), np.nan, np.abs(c - op) > p["big_bar"] * u)
    cols = [SIDES[br], br.astype(float), inw.astype(float), big]
    return pd.DataFrame(dict(zip(BRANCHES, cols)), index=bars.index)


# ---------------------------------------------------------------- session

def session_features(bars: pd.DataFrame, clock=None) -> pd.DataFrame:
    """Per bar: eod, 1 when a position is to be closed at this bar's close: its New York close is not
    in [09:40, 15:55) on Mon-Fri, or the bar is in a later session day than the bar before it (after a
    gap over the 15:55 close that ends inside a window); and sday, the bar's session day (18:00 -> 18:00
    New York, epoch days, as ny_session_day). clock: the bars' close_clock, if at hand."""
    t = bars["open_time"].to_numpy(np.int64)
    n = len(t)
    wd, mi = close_clock(t) if clock is None else clock
    ds = day_starts(t)
    sday = np.repeat(np.arange(len(ds) - 1, dtype=np.int64), np.diff(ds))
    if n:
        sday += ny_session_day(t[:1])[0]
    eod = (wd >= 5) | (mi < HELD_FROM) | (mi >= HELD_TO)
    eod[1:] |= sday[1:] != sday[:-1]
    return pd.DataFrame(dict(zip(SESSION_COLS, (eod.astype(float), sday.astype(float)))), index=bars.index)
