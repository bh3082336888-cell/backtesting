import asyncio

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
    def __init__(self, n=12):
        self.t0 = 1_700_000_000_000
        self.n = n
        self.k = 0

    async def fetch_ohlcv(self, sym, tf, since=None, limit=None):
        return [[self.t0 + i * 3_600_000, 100, 101, 99, 100, 1] for i in range(50)]

    async def watch_ohlcv(self, sym, tf):
        await asyncio.sleep(0)
        self.k += 1
        ts = self.t0 + (50 + self.k) * 3_600_000
        return [[ts - 3_600_000, 100, 101, 99, 100, 1], [ts, 100, 101, 99, 100, 1]]

    async def watch_ticker(self, sym):
        await asyncio.sleep(0.001)
        return {"bid": 99.9, "ask": 100.1, "timestamp": self.t0}

    async def fetch_funding_rate(self, sym):
        return {"fundingRate": 0.0001, "fundingTimestamp": self.t0 + 10**12, "markPrice": 100.0}


def test_paper_runner_loop_with_fake_exchange(tmp_path):
    from perp_lab.config import load_config
    cfg = load_config()
    cls = type("Always", (Scripted,), {"__init__": lambda self, **kw: Scripted.__init__(
        self, {i: Signal("long", stop=90.0) for i in range(2000)})})
    r = PaperRunner(cfg, cls, {}, "1h", ["BTCUSDT"], tmp_path, exchange=FakeEx())
    asyncio.run(r.run(max_bars=3))
    assert r.books["BTCUSDT"].pos is not None
    assert (tmp_path / "state.json").exists()
