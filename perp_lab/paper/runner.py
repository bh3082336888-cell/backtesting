"""Paper runner: the same Strategy class on live closed candles.

Differences from the backtest are deliberate and are what the backtest-vs-paper
diff is meant to measure:
  * entries/exits fill at the live top of book (ask to buy, bid to sell) on the
    first quote after the bar closes — the real spread replaces the slippage param
  * stops trigger on live quotes (bid for longs, ask for shorts), filled at that quote
  * targets fill at the target price (maker) once the quote trades through it
  * funding: the predicted rate captured just before each settlement, on mark price
Warmup history comes straight from the exchange (REST), never from the research
loader, so paper never reads the locked holdout through the research path.
Trade log schema == backtest trade log schema (+ source="paper").
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from pathlib import Path
from typing import Callable

import ccxt
import numpy as np
import pandas as pd

from perp_lab.config import TF_MS, ccxt_symbol
from perp_lab.data.live import acall_with_backoff, ohlcv_to_candles, watch_closed
from perp_lab.engine.backtest import EngineParams, liq_price, same_level
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client
from perp_lab.sandbox.errors import PluginError
from perp_lab.strategy import PositionView, Signal

log = logging.getLogger(__name__)


def _finite(w: float) -> float:
    """Equity after a fill or a funding charge. A fill at an infinite or overflowing price (a target
    moved past the price on the wrong side) would leave it not finite: the plugin's `mismatch`, as
    in a replay, raised before anything is booked."""
    if not math.isfinite(w):
        raise PluginError("mismatch", "", "paper equity is not finite")
    return w


class PaperBook:
    def __init__(self, decide: Callable[[pd.DataFrame, PositionView | None], Signal], symbol: str, tf: str,
                 ep: EngineParams):
        self.decide, self.symbol, self.tf, self.ep = decide, symbol, tf, ep
        self.W = ep.equity0
        self.pos: dict | None = None
        self.pending: Signal | None = None
        self.pending_meta: dict = {}
        self.trades: list[dict] = []
        self.bars: pd.DataFrame | None = None
        self.settlements: dict[int, float] = {}

    # -- bar close: decide
    def on_closed_bar(self, bars: pd.DataFrame) -> Signal:
        self.bars = bars
        i = len(bars) - 1
        c = float(bars.close.iloc[-1])
        p = self.pos
        pv = None
        if p:
            if (p["side"] > 0 and c > p["best"]) or (p["side"] < 0 and c < p["best"]):
                p["best"] = c
            pv = PositionView()
            pv.side, pv.entry_price, pv.entry_i = p["side"], p["entry_price"], p["entry_i"]
            pv.stop, pv.target, pv.qty, pv.best_close = p["stop"], p["target"], p["qty"], p["best"]
            pv.bars_held = i - p["entry_i"] + 1
        sig = self.decide(bars, pv)
        if sig.action == "hold":
            if p and sig.stop is not None and sig.stop == sig.stop:
                if (p["side"] > 0 and sig.stop > p["stop"]) or (p["side"] < 0 and sig.stop < p["stop"]):
                    p["stop"] = sig.stop
            if p and sig.target is not None:
                p["target"] = sig.target
            if p and not (same_level(p["stop"], p["level_stop"][-1])
                          and same_level(p["target"], p["level_target"][-1])):
                # decided at this bar's close, in force from then on
                p["level_time"].append(int(bars.open_time.iloc[-1]) + TF_MS[self.tf])
                p["level_stop"].append(p["stop"])
                p["level_target"].append(p["target"])
        else:
            self.pending = sig
            self.pending_meta = {"signal_time": int(bars.open_time.iloc[-1]), "close": c,
                                 "regime": str(bars["regime"].iloc[-1]) if "regime" in bars else "unknown",
                                 "i": i}
        return sig

    # -- quotes: execute pending, then stops/targets/liq
    def on_quote(self, bid: float, ask: float, ts: int):
        ep = self.ep
        if self.pending is not None:
            sig, meta = self.pending, self.pending_meta
            self.pending = None
            side_new = {"long": 1, "short": -1}.get(sig.action, 0)
            if self.pos and (sig.action == "exit" or side_new == -self.pos["side"]):
                self._close(bid if self.pos["side"] > 0 else ask, ts, "signal", ep.taker)
            if side_new and not self.pos:
                self._open(side_new, ask if side_new > 0 else bid, sig, meta, ts)
        p = self.pos
        if not p:
            return
        mark = (bid + ask) / 2
        if (p["side"] > 0 and mark <= p["liq"]) or (p["side"] < 0 and mark >= p["liq"]):
            self._close(p["liq"], ts, "liq", 0.0)
        elif p["side"] > 0 and bid <= p["stop"]:
            self._close(bid, ts, "stop", ep.taker)
        elif p["side"] < 0 and ask >= p["stop"]:
            self._close(ask, ts, "stop", ep.taker)
        elif p["target"] == p["target"] and ((p["side"] > 0 and bid > p["target"]) or
                                             (p["side"] < 0 and ask < p["target"])):
            self._close(p["target"], ts, "target", ep.maker)

    def on_funding(self, rate: float, mark: float):
        p = self.pos
        if not p:
            return
        f = p["side"] * p["qty"] * mark * rate
        self.W = _finite(self.W - f)
        p["margin"] -= f
        p["funding"] += f
        p["liq"] = liq_price(p["side"], p["entry_price"], p["qty"], p["margin"], self.ep.mmr, self.ep.cum)

    def _open(self, side, px, sig: Signal, meta, ts):
        ep = self.ep
        dist = (px - sig.stop) * side
        if dist <= 0:
            log.info("%s paper: price already through stop; entry skipped", self.symbol)
            return
        risk_amt = self.W * ep.risk
        q = risk_amt / dist
        cap = self.W * ep.max_margin_frac * ep.leverage
        if q * px > cap:
            q = cap / px
            risk_amt = q * dist
        mg = q * px / ep.leverage
        lq = liq_price(side, px, q, mg, ep.mmr, ep.cum)
        if (side > 0 and lq >= sig.stop) or (side < 0 and lq <= sig.stop):
            log.info("%s paper: liq inside stop; entry rejected", self.symbol)
            return
        fee = ep.taker * q * px
        self.W = _finite(self.W - fee)
        target_init = sig.target if sig.target is not None else np.nan
        self.pos = {"side": side, "qty": q, "entry_price": px, "stop": sig.stop,
                    "target": target_init, "margin": mg, "liq": lq,
                    "entry_i": meta["i"] + 1, "entry_time": ts, "signal_time": meta["signal_time"],
                    "best": px, "fees": fee, "funding": 0.0, "risk_amt": risk_amt, "W_before": self.W + fee,
                    "stop_init": sig.stop, "target_init": target_init, "regime": meta["regime"],
                    "level_time": [ts], "level_stop": [sig.stop], "level_target": [target_init]}
        log.info("%s paper OPEN %s qty=%.6f @ %.6f stop=%.6f", self.symbol, side, q, px, sig.stop)

    def _close(self, px, ts, reason, fee_rate):
        p = self.pos
        if reason == "liq":
            pnl, fee = -p["margin"], 0.0
        else:
            pnl, fee = p["side"] * p["qty"] * (px - p["entry_price"]), fee_rate * p["qty"] * px
        self.W = _finite(self.W + (pnl - fee))
        fees = p["fees"] + fee
        net = pnl - fees - p["funding"]
        nb = 0
        if self.bars is not None:
            nb = int((self.bars.open_time >= p["entry_time"] - TF_MS[self.tf]).sum())
        self.trades.append({
            "symbol": self.symbol, "tf": self.tf, "side": p["side"], "signal_time": p["signal_time"],
            "entry_time": p["entry_time"], "entry_price": p["entry_price"], "qty": p["qty"],
            "notional": p["qty"] * p["entry_price"], "leverage": self.ep.leverage,
            "margin": p["qty"] * p["entry_price"] / self.ep.leverage, "stop_init": p["stop_init"],
            "target": p["target_init"], "exit_time": ts, "exit_price": px, "exit_reason": reason,
            "fees": fees, "slippage": 0.0, "funding": p["funding"], "pnl_gross": pnl, "pnl": net,
            "equity_before": p["W_before"], "pnl_pct": net / p["W_before"],
            "r_multiple": net / p["risk_amt"], "bars_held": nb, "regime": p["regime"],
            "level_time": p["level_time"], "level_stop": p["level_stop"], "level_target": p["level_target"],
            "source": "paper",
        })
        log.info("%s paper CLOSE %s @ %.6f net=%.2f", self.symbol, reason, px, net)
        self.pos = None


def _no_worker(bars, pv):
    raise RuntimeError("no worker is open for this symbol yet")


class PaperRunner:
    def __init__(self, cfg, src: PluginSource, params: dict, tf: str, symbols: list[str], out_dir: Path,
                 exchange=None):
        self.cfg, self.src, self.params, self.tf, self.symbols = cfg, src, params, tf, symbols
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ex = exchange
        self.eps = {s: EngineParams.from_config(cfg, s) for s in symbols}
        self.books = {s: PaperBook(_no_worker, s, tf, self.eps[s]) for s in symbols}
        self.quote = cfg.get("quote", "USDT")
        self.seen: dict[str, pd.DataFrame] = {}
        self.warmup = cfg["paper"]["warmup_bars"]
        self.dropped: dict[str, str] = {}               # coin -> why the exchange stopped serving it
        self._gone = {s: asyncio.Event() for s in symbols}

    def _drop(self, sym: str, e: Exception):
        """The exchange no longer lists this coin (ccxt BadSymbol): the coin leaves the run, the others carry
        on. Its open position stays as it is: no fill is invented. Its other loops stop at their next turn."""
        if sym not in self.dropped:
            self.dropped[sym] = f"{type(e).__name__}: {e}"
            log.error("%s paper: dropped, the exchange no longer serves it (%s)", sym, self.dropped[sym])
        self._gone[sym].set()

    def _persist(self):
        rows = [t for b in self.books.values() for t in b.trades]
        if rows:
            pd.DataFrame(rows).to_parquet(self.out / "trades.parquet", index=False)
        for s, df in self.seen.items():
            df.to_parquet(self.out / f"bars_{s}.parquet", index=False)
        state = {s: {"W": b.W, "pos": b.pos, **({"dropped": self.dropped[s]} if s in self.dropped else {})}
                 for s, b in self.books.items()}
        (self.out / "state.json").write_text(json.dumps(state, default=float, indent=1))

    async def _warm(self, sym) -> pd.DataFrame:
        since = int(time.time() * 1000) - self.warmup * TF_MS[self.tf]
        rows = await acall_with_backoff(self.ex.fetch_ohlcv, ccxt_symbol(sym, self.quote), self.tf,
                                        since=since, limit=min(self.warmup, 1500))
        df = ohlcv_to_candles(rows, self.tf)
        now = int(time.time() * 1000)
        df = df[df.open_time + TF_MS[self.tf] <= now]
        return self._with_funding(df.reset_index(drop=True), self.books[sym])

    def _with_funding(self, bars: pd.DataFrame, book: "PaperBook") -> pd.DataFrame:
        """Same funding columns as the research loader, from settlements observed live."""
        from perp_lab.data.store import align_funding
        st = pd.DataFrame(sorted(book.settlements.items()), columns=["funding_time", "funding_rate"])
        st["interval_hours"] = np.nan
        return align_funding(bars.drop(columns=["funding_charge", "funding_last", "funding_interval_h"],
                                       errors="ignore"), st, self.tf)

    async def _bars_loop(self, sym, max_bars):
        # The worker is spawned and closed in this thread (its parent-death signal is the spawning thread's).
        book = self.books[sym]
        try:
            bars = await self._warm(sym)
            worker = client.spawn(self.cfg, self.src)
            try:
                await self._stream(sym, book, worker, bars, max_bars)
            finally:
                worker.close()
        except ccxt.BadSymbol as e:
            self._drop(sym, e)

    async def _stream(self, sym, book, worker, bars, max_bars):
        self.seen[sym] = bars.assign(live=False).reset_index(drop=True)
        worker.bars(sym, self.tf, bars.set_axis(pd.to_datetime(bars.open_time.astype("int64"), unit="ms",
                                                                utc=True)))
        worker.stream_open(self.params, self.eps[sym], 0)
        book.decide = lambda window, pv: worker.step(pv, window=window)
        n = 0
        async for c in watch_closed(self.ex, sym, self.tf, self.quote):
            row = pd.DataFrame([c.to_dict()])
            bars = pd.concat([bars, row], ignore_index=True).drop_duplicates("open_time", keep="last")
            bars = self._with_funding(bars.tail(self.warmup).reset_index(drop=True), book)
            bars.index = pd.to_datetime(bars.open_time.astype("int64"), unit="ms", utc=True)
            self.seen[sym] = pd.concat([self.seen[sym], bars.iloc[[-1]].assign(live=True)], ignore_index=True)
            sig = book.on_closed_bar(bars)
            if sig.action != "hold":
                log.info("%s signal %s", sym, sig)
            self._persist()
            n += 1
            if (max_bars and n >= max_bars) or self._gone[sym].is_set():
                return

    async def _quotes_loop(self, sym, stop_evt):
        book = self.books[sym]
        csym = ccxt_symbol(sym, self.quote)
        while not stop_evt.is_set() and not self._gone[sym].is_set():
            # bookTicker stream: Binance USD-M's ticker stream carries no bid/ask.
            try:
                t = (await acall_with_backoff(self.ex.watch_bids_asks, [csym])).get(csym) or {}
            except ccxt.BadSymbol as e:
                self._drop(sym, e)
                return
            bid, ask = t.get("bid"), t.get("ask")
            if bid and ask:
                had = len(book.trades)
                book.on_quote(float(bid), float(ask), int(t.get("timestamp") or time.time() * 1000))
                if len(book.trades) != had:
                    self._persist()

    async def _funding_loop(self, sym, stop_evt, poll_s=30):
        book = self.books[sym]
        nxt, rate = None, None
        while not stop_evt.is_set() and not self._gone[sym].is_set():
            try:
                fr = await acall_with_backoff(self.ex.fetch_funding_rate, ccxt_symbol(sym, self.quote))
            except ccxt.BadSymbol as e:
                self._drop(sym, e)
                return
            now = int(time.time() * 1000)
            if nxt is not None and now >= nxt and rate is not None:
                book.on_funding(rate, float(fr.get("markPrice") or 0.0))
                book.settlements[int(nxt)] = float(rate)
            nxt, rate = fr.get("fundingTimestamp"), fr.get("fundingRate")
            try:
                await asyncio.wait_for(stop_evt.wait(), timeout=poll_s)
            except asyncio.TimeoutError:
                pass

    async def run(self, max_bars: int | None = None):
        if self.ex is None:
            from perp_lab.data.live import make_exchange
            self.ex = make_exchange(pro=True)
        stop_evt = asyncio.Event()
        side = [asyncio.create_task(self._quotes_loop(s, stop_evt)) for s in self.symbols]
        side += [asyncio.create_task(self._funding_loop(s, stop_evt)) for s in self.symbols]
        bars = asyncio.ensure_future(asyncio.gather(*(self._bars_loop(s, max_bars) for s in self.symbols)))
        try:
            # The quote and funding loops run until the bars are done; one that ends first failed
            # (e.g. a fill the book refused as a mismatch), and that stops the run with its error.
            # A dropped coin's side loops end cleanly, so keep waiting on the rest until the bars are done.
            while not bars.done():
                waiting = [t for t in side if not t.done()]
                await asyncio.wait([bars, *waiting], return_when=asyncio.FIRST_COMPLETED)
                for t in side:
                    if t.done():
                        t.result()
            await bars
            if len(self.dropped) == len(self.symbols):
                raise RuntimeError("paper: every coin was dropped: "
                                   + "; ".join(f"{s}: {why}" for s, why in self.dropped.items()))
        finally:
            stop_evt.set()
            bars.cancel()                   # if a side loop failed: each bars loop closes its worker
            for t in side:
                t.cancel()
            await asyncio.gather(bars, *side, return_exceptions=True)
            self._persist()
            close = getattr(self.ex, "close", None)
            if close:
                r = close()
                if asyncio.iscoroutine(r):
                    await r
