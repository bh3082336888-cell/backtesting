"""Event-driven single-symbol backtest. Closed candles only.

Bar t sequence (all decisions use information available at that moment):

  OPEN of t
    1. funding settled in bar t is charged to the position held *into* the open
       (deducted from wallet and from isolated margin → liq price moves)
    2. gap checks for a held position: open beyond liq → liquidated;
       open beyond stop → stopped at the open
    3. pending exit (signal from close of t-1) fills at open, taker + slippage
    4. pending entry fills at open, taker + slippage; sized from the stop
  INTRABAR (OHLC only, no path info → pessimistic)
    5. adverse level (stop or liq, whichever is hit first) checked before target.
       If both stop and target are inside the bar's range, the stop wins.
       Target is a resting limit: needs price to trade *through* it; maker fee.
  CLOSE of t
    6. mark-to-market equity
    7. strategy.on_bar(ctx) → Signal, queued for the open of t+1

Isolated margin, one-way mode. Liquidation loses the remaining isolated margin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from perp_lab.strategy import Ctx, PositionView, Strategy

NAN = math.nan


@dataclass(frozen=True)
class EngineParams:
    taker: float = 0.0005
    maker: float = 0.0002
    slippage: float = 0.0002          # fraction (2 bps)
    equity0: float = 10_000.0
    leverage: float = 3.0
    risk: float = 0.01
    max_margin_frac: float = 1.0
    mmr: float = 0.005
    cum: float = 0.0

    @classmethod
    def from_config(cls, cfg, symbol: str, **over) -> "EngineParams":
        c, a, b = cfg["costs"], cfg["account"], cfg.bracket(symbol)
        kw = dict(taker=c["taker"], maker=c["maker"], slippage=c["slippage_bps"] / 1e4,
                  equity0=a["initial_equity"], leverage=a["leverage"], risk=a["risk_per_trade"],
                  max_margin_frac=a.get("max_margin_frac", 1.0), mmr=b["mmr"], cum=b.get("cum", 0.0))
        kw.update(over)
        return cls(**kw)


def liq_price(side: int, entry: float, qty: float, margin: float, mmr: float, cum: float) -> float:
    """Isolated one-way liquidation price.

    Liquidated when  margin + side*qty*(P - entry) == qty*P*mmr - cum.
    """
    if side > 0:
        return (qty * entry - margin - cum) / (qty * (1.0 - mmr))
    return (qty * entry + margin + cum) / (qty * (1.0 + mmr))


@dataclass
class BacktestResult:
    trades: list[dict]
    equity: np.ndarray            # MTM equity at each bar close, bars [start_i, end_i)
    time: np.ndarray              # open_time ms of those bars
    in_pos: np.ndarray            # bool, position held at bar close
    signals: list[tuple]          # (i, action, stop, target) non-hold signals (for lookahead tests)
    rejected: dict


def build_arrays(bars: pd.DataFrame, feats: pd.DataFrame) -> dict[str, np.ndarray]:
    arr = {c: bars[c].to_numpy() for c in bars.columns}
    for c in feats.columns:
        arr[c] = feats[c].to_numpy()
    return arr


def run_backtest(strategy: Strategy, bars: pd.DataFrame, feats: pd.DataFrame, *,
                 symbol: str, tf: str, ep: EngineParams, start_i: int = 0,
                 end_i: int | None = None, arrays: dict | None = None,
                 record_signals: bool = False) -> BacktestResult:
    A = arrays if arrays is not None else build_arrays(bars, feats)
    n = len(A["open"])
    end_i = n if end_i is None else min(end_i, n)
    O, H, L, C = A["open"], A["high"], A["low"], A["close"]
    T = A["open_time"]
    FC = A.get("funding_charge")
    if FC is None:
        FC = np.zeros(n)
    REG = A.get("regime")

    taker, maker, slip = ep.taker, ep.maker, ep.slippage
    lev, mmr, cum = ep.leverage, ep.mmr, ep.cum

    W = ep.equity0
    side = 0
    qty = entry_px = margin = risk_amt = W_before = 0.0
    stop = target = liq = best = NAN
    entry_i = sig_i = -1
    fees_acc = fund_acc = 0.0
    stop_init = NAN
    tgt_init = NAN

    pend_exit = False
    pend_entry = None  # (side, stop, target, signal_i)

    m = max(end_i - start_i, 0)
    equity = np.empty(m)
    in_pos = np.zeros(m, dtype=bool)
    trades: list[dict] = []
    signals: list[tuple] = []
    rejected = {"liq_inside_stop": 0, "gap_through_stop": 0, "bad_stop": 0}

    ctx = Ctx(A, symbol)
    pv = PositionView()

    def close_pos(i, px, reason, fee_rate):
        nonlocal W, side, qty, margin, fees_acc
        if reason == "liq":
            pnl = -margin
            fee = 0.0
        else:
            pnl = side * qty * (px - entry_px)
            fee = fee_rate * qty * px
        W += pnl - fee
        fees_acc += fee
        net = pnl - fees_acc - fund_acc
        trades.append({
            "symbol": symbol, "tf": tf, "side": side,
            "signal_time": int(T[sig_i]), "entry_time": int(T[entry_i]), "entry_price": entry_px,
            "qty": qty, "notional": qty * entry_px, "leverage": lev, "margin": qty * entry_px / lev,
            "stop_init": stop_init, "target": tgt_init,
            "exit_time": int(T[i]), "exit_price": px, "exit_reason": reason,
            "fees": fees_acc, "funding": fund_acc, "pnl_gross": pnl, "pnl": net,
            "equity_before": W_before, "pnl_pct": net / W_before if W_before else NAN,
            "r_multiple": net / risk_amt if risk_amt else NAN,
            "bars_held": i - entry_i + 1,
            "regime": str(REG[sig_i]) if REG is not None else "unknown",
        })
        side = 0
        qty = margin = 0.0

    last = end_i - 1
    for i in range(start_i, end_i):
        o = O[i]; h = H[i]; l = L[i]; c = C[i]

        # 1. funding on the position held into this open
        if side != 0:
            fr = FC[i]
            if fr != 0.0:
                f = side * qty * o * fr
                W -= f
                margin -= f
                fund_acc += f
                liq = liq_price(side, entry_px, qty, margin, mmr, cum)
            # 2. gaps
            if (side > 0 and o <= liq) or (side < 0 and o >= liq):
                close_pos(i, liq, "liq", 0.0)
                pend_exit = False
            elif (side > 0 and o <= stop) or (side < 0 and o >= stop):
                close_pos(i, o * (1 - side * slip), "stop", taker)
                pend_exit = False

        # 3. pending exit
        if pend_exit and side != 0:
            close_pos(i, o * (1 - side * slip), "signal", taker)
        pend_exit = False

        # 4. pending entry
        if pend_entry is not None and side == 0 and i < last:
            s, st, tg, si = pend_entry
            fill = o * (1 + s * slip)
            dist = (fill - st) * s
            if dist <= 0:
                rejected["gap_through_stop"] += 1
            else:
                W_before = W
                risk_amt = W * ep.risk
                q = risk_amt / dist
                max_notional = W * ep.max_margin_frac * lev
                if q * fill > max_notional:
                    q = max_notional / fill
                    risk_amt = q * dist
                mg = q * fill / lev
                lq = liq_price(s, fill, q, mg, mmr, cum)
                if (s > 0 and lq >= st) or (s < 0 and lq <= st):
                    rejected["liq_inside_stop"] += 1
                else:
                    side, qty, entry_px, margin, liq = s, q, fill, mg, lq
                    stop, target = st, (tg if tg is not None else NAN)
                    stop_init, tgt_init = stop, target
                    entry_i, sig_i = i, si
                    fee = taker * q * fill
                    W -= fee
                    fees_acc, fund_acc = fee, 0.0
                    best = fill
        pend_entry = None

        # 5. intrabar
        if side != 0:
            if side > 0:
                adv = stop if stop > liq else liq
                if l <= adv:
                    if adv == liq and liq >= stop:
                        close_pos(i, liq, "liq", 0.0)
                    else:
                        close_pos(i, stop * (1 - slip), "stop", taker)
                elif target == target and h > target:
                    close_pos(i, target, "target", maker)
            else:
                adv = stop if stop < liq else liq
                if h >= adv:
                    if adv == liq and liq <= stop:
                        close_pos(i, liq, "liq", 0.0)
                    else:
                        close_pos(i, stop * (1 + slip), "stop", taker)
                elif target == target and l < target:
                    close_pos(i, target, "target", maker)

        # end of window: flatten at the close
        if i == last and side != 0:
            close_pos(i, c * (1 - side * slip), "eow", taker)

        # 6. mark to market
        k = i - start_i
        if side != 0:
            equity[k] = W + side * qty * (c - entry_px)
            in_pos[k] = True
            if (side > 0 and c > best) or (side < 0 and c < best):
                best = c
        else:
            equity[k] = W

        if i == last:
            break

        # 7. strategy decision at the close
        ctx.i = i
        if side != 0:
            pv.side = side; pv.entry_price = entry_px; pv.entry_i = entry_i
            pv.stop = stop; pv.target = target; pv.qty = qty
            pv.best_close = best; pv.bars_held = i - entry_i + 1
            ctx.position = pv
        else:
            ctx.position = None
        sig = strategy.on_bar(ctx)
        a = sig.action
        if a == "hold":
            if side != 0:
                ns = sig.stop
                if ns is not None and ns == ns:
                    if (side > 0 and ns > stop) or (side < 0 and ns < stop):
                        stop = ns
                if sig.target is not None:
                    target = sig.target
            continue
        if record_signals:
            signals.append((i, a, sig.stop, sig.target))
        if a == "exit":
            if side != 0:
                pend_exit = True
            continue
        s = 1 if a == "long" else -1
        if side == s:
            continue
        if side != 0:
            pend_exit = True
        st = sig.stop
        if st is None or st != st or (st - c) * s >= 0:
            rejected["bad_stop"] += 1
            continue
        pend_entry = (s, st, sig.target, i)

    return BacktestResult(trades, equity, T[start_i:end_i], in_pos, signals, rejected)
