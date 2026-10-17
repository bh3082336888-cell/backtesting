import ctypes
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from perp_lab.data.store import load_bars
from perp_lab.holdout import HoldoutViolation, issue_holdout_token
from perp_lab.judge.exam import cost_hours, run_exam, window_end
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client
from perp_lab.sandbox.errors import PluginError
from tests.conftest import market_config
from tests.sandbox_fixtures import InProcessWorker

END = pd.Timestamp("2026-10-01", tz="UTC")   # funding ends 2026-09-30 16:00 + 8h


def test_window_end_is_the_last_day_with_candles_and_funding(tmp_path):
    cfg = market_config(tmp_path)
    assert window_end(cfg, cfg.symbols, "4h") == END


def test_window_end_follows_the_stalest_symbol(tmp_path):
    cfg = market_config(tmp_path)
    f = cfg.data_dir / "ETHUSDT" / "4h" / "all.parquet"
    df = pd.read_parquet(f)
    df[df.open_time < pd.Timestamp("2026-09-20 16:00", tz="UTC").value // 10**6].to_parquet(f)
    assert window_end(cfg, cfg.symbols, "4h") == pd.Timestamp("2026-09-20", tz="UTC")


def test_exam_refuses_without_a_token(tmp_path):
    cfg = market_config(tmp_path)
    with pytest.raises(HoldoutViolation):
        run_exam(cfg, PluginSource.read("donchian_atr"), "4h", cfg.symbols, {"n": 20}, None, END)


def test_exam_covers_exactly_the_holdout_window(tmp_path):
    cfg = market_config(tmp_path)
    tok = issue_holdout_token("fam", "exam test", tmp_path / "test_log.jsonl")
    res = run_exam(cfg, PluginSource.read("donchian_atr"), "4h", cfg.symbols, {"n": 20, "atr_mult": 3.0}, tok,
                   END)
    assert len(res.portfolio) == 92                       # 31 Jul + 31 Aug + 30 Sep
    assert res.portfolio.index.min() == cfg.holdout_start
    assert res.portfolio.index.max() == pd.Timestamp("2026-09-30", tz="UTC")
    st = pd.to_datetime(res.trades.signal_time, unit="ms", utc=True)
    assert len(res.trades) > 0 and st.min() >= cfg.holdout_start and st.max() < END
    assert set(res.daily) == {"BTCUSDT", "ETHUSDT"} and res.params == {"n": 20, "atr_mult": 3.0}


def test_cost_hours_is_the_median_prepare_time_times_the_window_bars(tmp_path, monkeypatch):
    cfg = market_config(tmp_path)
    times, seen = iter([0.5, 3.0, 1.0]), []

    class Probe(InProcessWorker):
        def bars(self, symbol, tf, df):
            seen.append(("bars", symbol, tf, len(df), df.index.max()))
            super().bars(symbol, tf, df)

        def prepare(self, params):
            seen.append(("prepare", params))
            super().prepare(params)
            return next(times)

        def close(self):
            seen.append(("close",))
    monkeypatch.setattr(client, "spawn", Probe)
    params = {"n": 20, "atr_n": 14, "atr_mult": 3.0}
    hours = cost_hours(cfg, PluginSource.read("donchian_atr"), "4h", cfg.symbols, params, END)
    assert hours == pytest.approx(1.0 * 552 / 3600)         # the median, times 92 days of 4h bars
    research = load_bars(cfg, "BTCUSDT", "4h")                # the first symbol's, all before the holdout
    assert research.index.max() < cfg.holdout_start
    assert seen == [("bars", "BTCUSDT", "4h", len(research), research.index.max()),
                    *[("prepare", params)] * 3, ("close",)]  # one worker of its own


def _spied(monkeypatch):
    """Twins that each report a mapped file of their own, and a spy on the page-cache eviction that
    notes whether every worker was closed by then."""
    spawned, calls = [], []

    class Twin(InProcessWorker):
        def __init__(self, cfg, src):
            super().__init__(cfg, src)
            self.mapped_files = frozenset({f"/nonexistent/worker-{len(spawned)}.so"})
            self.closed = False
            spawned.append(self)

        def close(self):
            self.closed = True
    monkeypatch.setattr(client, "spawn", Twin)
    monkeypatch.setattr(client, "evict_page_cache",
                        lambda paths: calls.append((set(paths), all(w.closed for w in spawned))))
    return spawned, calls


def test_an_exam_evicts_the_page_cache_its_workers_mapped(tmp_path, monkeypatch):
    # A worker can fault in chosen pages of the libraries it maps; a later one could read the
    # pattern back. After the exam, with its workers gone, those files leave the page cache.
    cfg = market_config(tmp_path)
    spawned, calls = _spied(monkeypatch)
    tok = issue_holdout_token("fam", "exam test", tmp_path / "test_log.jsonl")
    run_exam(cfg, PluginSource.read("donchian_atr"), "4h", cfg.symbols, {"n": 20}, tok, END)
    assert len(spawned) == 2                                  # one worker per symbol
    assert calls == [({p for w in spawned for p in w.mapped_files}, True)]


def test_a_failed_exam_evicts_the_page_cache_too(tmp_path, monkeypatch):
    cfg = market_config(tmp_path)
    spawned, calls = _spied(monkeypatch)
    holdout_ms = cfg.holdout_start.value // 10**6
    src = PluginSource.from_text("fails", f'''"""Fails on its first holdout bar."""
from perp_lab.strategy import HOLD, Strategy


class Fails(Strategy):
    def prepare(self, bars):
        return bars[[]]

    def entry(self, ctx):
        if ctx["open_time"] >= {holdout_ms}:
            raise RuntimeError("holdout")
        return HOLD
'''.encode())
    tok = issue_holdout_token("fam", "exam test", tmp_path / "test_log.jsonl")
    with pytest.raises(PluginError):
        run_exam(cfg, src, "4h", cfg.symbols, {}, tok, END)
    assert spawned and calls == [({p for w in spawned for p in w.mapped_files}, True)]


def test_evicting_drops_a_file_from_the_page_cache(caplog):
    # On the real disk (.venv), not a tmpfs, whose pages can't be dropped.
    path = Path(sys.prefix) / f"perp_lab_evict_test_{os.getpid()}_{time.monotonic_ns()}"
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    size = 16 * 4096

    def resident() -> int:
        """Pages of the file in the page cache (mapped here without touching them)."""
        fd = os.open(path, os.O_RDONLY)
        try:
            addr = libc.mmap(None, size, 1, 2, fd, 0)               # PROT_READ, MAP_PRIVATE
            vec = ctypes.create_string_buffer(size // 4096)
            assert libc.mincore(addr, size, vec) == 0, ctypes.get_errno()
            libc.munmap(addr, size)
            return sum(b & 1 for b in vec.raw)
        finally:
            os.close(fd)
    try:
        with open(path, "wb") as f:
            f.write(os.urandom(size))
            f.flush()
            os.fsync(f.fileno())                                   # clean pages are what eviction drops
        path.read_bytes()
        assert resident() == 16
        with caplog.at_level("INFO", logger="perp_lab.sandbox.client"):
            client.evict_page_cache([str(path), "/nonexistent/lib.so", str(path.parent)])
        assert resident() == 0
        assert "/nonexistent/lib.so" in caplog.text               # skipped, not raised
    finally:
        path.unlink(missing_ok=True)
