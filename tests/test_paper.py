import asyncio
import time

import numpy as np
import pandas as pd
import pytest

from perp_lab.engine.backtest import EngineParams
from perp_lab.paper.runner import PaperBook, PaperRunner
from perp_lab.strategy import EXIT, Signal
from tests.conftest import Scripted, flat_bars

EP = EngineParams()


def test_paper_fills_at_book_and_stops_on_bid():
    b = flat_bars(10)
    book = PaperBook(Scripted({9: Signal("long", stop=95.0)}), "X", "1h", EP)
    book.on_closed_bar(b)
    book.on_quote(99.9, 100.1, 1)
    assert book.pos["entry_price"] == 100.1          # bought the ask
    book.on_quote(96.0, 96.2, 2)
    assert book.pos is not None
    book.on_quote(94.9, 95.1, 3)
    t = book.trades[0]
    assert t["exit_reason"] == "stop" and t["exit_price"] == 94.9 and t["source"] == "paper"


def test_paper_trade_schema_matches_backtest():
    from perp_lab.engine.backtest import run_backtest
    b = flat_bars(20)
    s = Scripted({5: Signal("long", stop=95.0), 8: EXIT})
    bt = run_backtest(s, b, s.prepare(b), symbol="X", tf="1h", ep=EP).trades[0]
    book = PaperBook(Scripted({9: Signal("long", stop=95.0)}), "X", "1h", EP)
    book.on_closed_bar(b.iloc[:10])
    book.on_quote(99.9, 100.1, 1)
    book.pending = EXIT
    book.on_quote(99.9, 100.1, 2)
    pt = book.trades[0]
    assert set(bt) <= set(pt) and set(pt) - set(bt) == {"source"}


class FakeEx:
    """Hourly candles from t0; candle 50 + k is open after k stream updates. Like ccxt.pro binance,
    watch_ohlcv returns only the open candle, so closed ones must come from REST fetch_ohlcv.
    The stream ends after n updates so a runner that never sees a close fails instead of hanging."""
    H = 3_600_000

    def __init__(self, n=12):
        now = int(time.time() * 1000)
        self.t0 = now // self.H * self.H - 60 * self.H
        self.n = n
        self.k = 0

    def _row(self, i):
        return [self.t0 + i * self.H, 100, 101, 99, 100, 1]

    async def fetch_ohlcv(self, sym, tf, since=None, limit=None):
        i0 = max(0, -(-(since - self.t0) // self.H)) if since is not None else 0
        rows = [self._row(i) for i in range(i0, 50 + self.k)]
        return rows[:limit] if limit else rows

    async def watch_ohlcv(self, sym, tf):
        await asyncio.sleep(0)
        if self.k >= self.n:
            raise StreamEnded
        self.k += 1
        return [self._row(50 + self.k)]

    async def watch_ticker(self, sym):
        # Binance USD-M's ticker stream carries no book: ccxt.pro returns bid/ask None (seen live).
        await asyncio.sleep(0.001)
        return {"symbol": sym, "timestamp": self.t0, "last": 100.0, "bid": None, "ask": None}

    async def watch_bids_asks(self, symbols):
        await asyncio.sleep(0.001)
        return {s: {"symbol": s, "timestamp": self.t0, "bid": 99.9, "bidVolume": 3.0,
                    "ask": 100.1, "askVolume": 2.0} for s in symbols}

    async def fetch_funding_rate(self, sym):
        return {"fundingRate": 0.0001, "fundingTimestamp": self.t0 + 10**12, "markPrice": 100.0}


class StreamEnded(Exception):
    pass


class RolloverStream:
    """ccxt.pro binance watch_ohlcv as seen live (BTCUSDT 5m, 2026-10-15): each call returns only
    the candle(s) updated since the last call, so the batch that opens a new candle lacks the one
    that just closed. fetch_ohlcv is REST: final values of closed candles, honouring since/limit."""

    def __init__(self, batches, final):
        self.batches, self.final = list(batches), final

    async def watch_ohlcv(self, sym, tf):
        if not self.batches:
            raise StreamEnded
        return self.batches.pop(0)

    async def fetch_ohlcv(self, sym, tf, since=None, limit=None):
        rows = [r for ts, r in sorted(self.final.items()) if ts >= since]
        return rows[:limit] if limit else rows


def test_watch_closed_yields_candles_missing_from_rollover_batch():
    from perp_lab.data.live import watch_closed
    H = 3_600_000
    t = 1_700_000_000_000 // H * H
    final = {t: [t, 100.0, 102.0, 99.0, 101.0, 12.0],
             t + H: [t + H, 101.0, 105.0, 100.0, 104.0, 9.0]}
    stream = RolloverStream([
        [[t, 100.0, 101.0, 99.0, 100.5, 5.0]],
        [[t, 100.0, 102.0, 99.0, 101.0, 12.0]],
        [[t + H, 101.0, 101.5, 100.5, 101.2, 2.0]],   # t closed; batch holds only t+H
        [[t + 2 * H, 104.0, 104.0, 104.0, 104.0, 0.1]],  # t+H's last ticks never reached us
    ], final)

    async def collect():
        got = []
        try:
            async for c in watch_closed(stream, "BTCUSDT", "1h"):
                got.append(c)
        except StreamEnded:
            pass
        return got

    got = asyncio.run(collect())
    assert [int(c.open_time) for c in got] == [t, t + H]
    assert [(c.high, c.low, c.close, c.volume) for c in got] == [(102.0, 99.0, 101.0, 12.0),
                                                                (105.0, 100.0, 104.0, 9.0)]


class _Proto:
    _reading_paused = False

    def pause_reading(self):
        self._reading_paused = True

    def resume_reading(self):
        self._reading_paused = False


def test_ccxt_ws_drain_keeps_aiohttp_queue_accounting_balanced():
    """ccxt's receive loop drains aiohttp's websocket queue itself and carries only raw message sizes
    onto the last entry. aiohttp 3.14.4 also counts 128 bytes of overhead per message, so each drained
    message leaked into the queue size until reading paused for good: Binance bookTicker went silent
    ~25s in with the connection still open (live, 2026-10-15). Pinned aiohttp<3.14.4 in pyproject."""
    import json

    from aiohttp import WSMessage, WSMsgType
    from aiohttp._websocket.reader_py import WebSocketDataQueue   # the C build compiles this; its _size is hidden
    from ccxt.async_support.base.ws.client import Client

    class Conn:
        closed = False

        def __init__(self, q):
            self._conn = type("C", (), {"protocol": type("P", (), {"_payload": q})()})()
            self.receive = q.read

    async def drain_burst(n):
        loop = asyncio.get_running_loop()
        q = WebSocketDataQueue(_Proto(), 2 ** 16, loop=loop)
        got = []
        noop = lambda *a: None
        c = Client("wss://test", lambda client, m: got.append(m), noop, noop, noop, {"asyncio_loop": loop})
        c.connection = Conn(q)
        for i in range(n):                              # a burst lands before ccxt reads
            data = json.dumps({"i": i})
            q.feed_data(WSMessage(WSMsgType.TEXT, data, None), len(data))
        c.receive_loop()
        while len(got) < n:
            await asyncio.sleep(0)
        return q._size, [m["i"] for m in got]

    size, order = asyncio.run(drain_burst(50))
    assert order == list(range(50))
    assert size == 0          # every byte counted in was counted out


def test_paper_runner_loop_with_fake_exchange(tmp_path):
    from perp_lab.config import load_config
    cfg = load_config()
    cls = type("Always", (Scripted,), {"__init__": lambda self, **kw: Scripted.__init__(
        self, {i: Signal("long", stop=90.0) for i in range(2000)})})
    r = PaperRunner(cfg, cls, {}, "1h", ["BTCUSDT"], tmp_path, exchange=FakeEx())
    asyncio.run(r.run(max_bars=3))
    assert r.books["BTCUSDT"].pos is not None
    assert r.books["BTCUSDT"].pos["entry_price"] == 100.1   # filled at the live ask
    assert (tmp_path / "state.json").exists()
