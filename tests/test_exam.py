import ctypes
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from perp_lab.data.store import load_bars
from perp_lab.engine.backtest import EngineParams
from perp_lab.holdout import HoldoutViolation, issue_holdout_token
from perp_lab.judge.exam import cost_hours, run_exam, window_data_problems, window_end
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client, stream
from perp_lab.sandbox.errors import PluginError, WorkerLost
from tests.conftest import market_config, write_market
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


def _manifest(cfg, symbol, **months):
    """Set klines-manifest entries of `symbol` at 4h: month "2026_08" → {"complete", "source"}."""
    p = cfg.data_dir / symbol / "4h" / "_manifest.json"
    m = json.loads(p.read_text())
    for k, v in months.items():
        if v is None:
            m.pop(k.replace("_", "-"), None)
        else:
            m[k.replace("_", "-")] = {"rows": 0, **v}
    p.write_text(json.dumps(m))


MONTHLY = {"complete": True, "source": "monthly"}


def test_exchange_files_throughout_the_window_raise_nothing(tmp_path):
    cfg = market_config(tmp_path)
    assert window_data_problems(cfg, cfg.symbols, "4h", END) == []
    # a closed month rebuilt from daily files, and the month the window ends in still open
    _manifest(cfg, "BTCUSDT", **{"2026_09": {"complete": True, "source": "daily"},
                                 "2026_10": {"complete": False, "source": "daily"}})
    assert window_data_problems(cfg, cfg.symbols, "4h", pd.Timestamp("2026-10-12", tz="UTC")) == []


def test_candles_stitched_from_ccxt_in_the_window_are_named(tmp_path):
    # ccxt OHLCV has no taker, trade-count or quote-volume columns: a plugin reading them would
    # sit flat through the exam and spend its family's one read on a data gap.
    cfg = market_config(tmp_path)
    _manifest(cfg, "ETHUSDT", **{"2026_08": {"complete": False, "source": "ccxt"},
                                 "2026_09": {"complete": False, "source": "ccxt"}})
    assert window_data_problems(cfg, cfg.symbols, "4h", END) == [
        "ETHUSDT 4h 2026-08: stitched from ccxt (no taker, trade-count or quote-volume columns)",
        "ETHUSDT 4h 2026-09: stitched from ccxt (no taker, trade-count or quote-volume columns)"]


def test_ccxt_months_outside_the_window_do_not_count(tmp_path):
    cfg = market_config(tmp_path)
    _manifest(cfg, "BTCUSDT", **{"2026_10": {"complete": False, "source": "ccxt"}})     # after END
    assert window_data_problems(cfg, cfg.symbols, "4h", END) == []
    _manifest(cfg, "BTCUSDT", **{"2026_06": {"complete": False, "source": "ccxt"}})     # research data
    assert window_data_problems(cfg, cfg.symbols, "4h", END) == []


def test_a_month_the_window_covers_must_be_complete(tmp_path):
    # Daily files missing from a closed month leave days out of the exam; the month the window
    # ends in may still be open.
    cfg = market_config(tmp_path)
    _manifest(cfg, "BTCUSDT", **{"2026_09": {"complete": False, "source": "daily"}})
    assert window_data_problems(cfg, cfg.symbols, "4h", END) == [
        "BTCUSDT 4h 2026-09: incomplete (days missing from data.binance.vision)"]
    assert window_data_problems(cfg, cfg.symbols, "4h", pd.Timestamp("2026-09-20", tz="UTC")) == []


@pytest.mark.parametrize("change, why", [
    ({"2026_07": None}, "no manifest entry"),
    ({"2026_07": {"complete": True, "source": "s3"}}, "unknown source 's3'"),
])
def test_a_month_of_unknown_origin_counts(tmp_path, change, why):
    cfg = market_config(tmp_path)
    _manifest(cfg, "BTCUSDT", **change)
    assert window_data_problems(cfg, cfg.symbols, "4h", END) == [f"BTCUSDT 4h 2026-07: {why}"]


@pytest.mark.parametrize("text, why", [(None, "no manifest"), ("{not json", "manifest unreadable"),
                                       ("[]", "manifest unreadable")])
def test_a_missing_or_unreadable_manifest_counts_for_every_month(tmp_path, text, why):
    cfg = market_config(tmp_path)
    p = cfg.data_dir / "ETHUSDT" / "4h" / "_manifest.json"
    p.unlink() if text is None else p.write_text(text)
    assert window_data_problems(cfg, cfg.symbols, "4h", END) == [
        f"ETHUSDT 4h {m}: {why}" for m in ("2026-07", "2026-08", "2026-09")]


def hourly(tmp_path):
    """The market with 1h candles too: an exam at 1h reads 4h candles as well, for `regime`."""
    cfg = market_config(tmp_path)
    write_market(cfg.data_dir, cfg.symbols, tfs=("4h", "1h"))
    assert cfg["regime"]["timeframe"] == "4h"
    return cfg


def _drop(cfg, symbol, tf, a, b):
    """Remove `symbol`'s `tf` candles that open in [a, b)."""
    f = cfg.data_dir / symbol / tf / "all.parquet"
    df = pd.read_parquet(f)
    lo, hi = (pd.Timestamp(x, tz="UTC").value // 10**6 for x in (a, b))
    df[(df.open_time < lo) | (df.open_time >= hi)].to_parquet(f)


def test_window_end_waits_for_the_regime_timeframe(tmp_path):
    # `regime` comes from 4h candles: past the last of them a 1h exam would carry one label forward.
    cfg = hourly(tmp_path)
    assert window_end(cfg, cfg.symbols, "1h") == END                      # funding ends first here
    _drop(cfg, "ETHUSDT", "4h", "2026-09-20 16:00", "2027-01-01")
    assert window_end(cfg, cfg.symbols, "1h") == pd.Timestamp("2026-09-20", tz="UTC")


def test_a_1h_exam_checks_its_regime_timeframe_too(tmp_path):
    # A 4h month missing or incomplete inside the window would leave stale regime labels; one
    # stitched from ccxt is fine (the labels need OHLC only, which ccxt candles have).
    cfg = hourly(tmp_path)
    assert window_data_problems(cfg, cfg.symbols, "1h", END) == []
    _manifest(cfg, "BTCUSDT", **{"2026_08": {"complete": False, "source": "ccxt"}})
    assert window_data_problems(cfg, cfg.symbols, "1h", END) == []
    _manifest(cfg, "BTCUSDT", **{"2026_08": None, "2026_09": {"complete": False, "source": "daily"}})
    assert window_data_problems(cfg, cfg.symbols, "1h", END) == [
        "BTCUSDT 4h 2026-08: no manifest entry (for the regime labels)",
        "BTCUSDT 4h 2026-09: incomplete (days missing from data.binance.vision) (for the regime labels)"]


def test_whole_days_without_candles_in_the_window_are_named(tmp_path):
    # A daily file that never arrived, say, which the open month's `complete` can't show. Whole UTC
    # days only, so a short exchange outage doesn't hold an exam back for good.
    cfg = market_config(tmp_path)
    _drop(cfg, "BTCUSDT", "4h", "2026-08-10 04:00", "2026-08-10 12:00")     # an outage: 2 bars
    assert window_data_problems(cfg, cfg.symbols, "4h", END) == []
    _drop(cfg, "BTCUSDT", "4h", "2026-09-05", "2026-09-06")
    _drop(cfg, "ETHUSDT", "4h", "2026-09-10", "2026-09-13")
    _drop(cfg, "ETHUSDT", "4h", "2026-10-02", "2026-10-04")                  # after the window
    _drop(cfg, "ETHUSDT", "4h", "2026-06-02", "2026-06-04")                  # research data
    assert window_data_problems(cfg, cfg.symbols, "4h", END) == [
        "BTCUSDT 4h 2026-09-05: no candles",
        "ETHUSDT 4h 2026-09-10 to 2026-09-12: no candles (3 days)"]


def test_whole_days_are_checked_at_the_regime_timeframe_too(tmp_path):
    cfg = hourly(tmp_path)
    _drop(cfg, "ETHUSDT", "4h", "2026-08-01", "2026-08-02")
    assert window_data_problems(cfg, cfg.symbols, "1h", END) == ["ETHUSDT 4h 2026-08-01: no candles"]


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


def _spied(monkeypatch, lose: int | None = None):
    """Twins that each report a mapped file of their own, and a spy on the page-cache eviction that
    notes whether every worker was closed by then. `events` is the order of both: ("spawn", file)
    and ("evict", files, all closed). With `lose`, that worker's first step is a lost worker."""
    spawned, calls, events = [], [], []
    lock = threading.Lock()

    class Twin(InProcessWorker):
        def __init__(self, cfg, src):
            super().__init__(cfg, src)
            self.closed = False
            with lock:
                self.n = len(spawned)
                self.mapped_files = frozenset({f"/nonexistent/worker-{self.n}.so"})
                spawned.append(self)
                events.append(("spawn", f"/nonexistent/worker-{self.n}.so"))

        def step(self, position, **kw):
            if self.n == lose:
                raise WorkerLost("died", "worker exited with code -9")
            return super().step(position, **kw)

        def close(self):
            self.closed = True

    def evict(paths):
        with lock:
            calls.append((set(paths), all(w.closed for w in spawned)))
            events.append(("evict", set(paths), all(w.closed for w in spawned)))
    monkeypatch.setattr(client, "spawn", Twin)
    monkeypatch.setattr(client, "evict_page_cache", evict)
    return spawned, calls, events


def test_an_exam_evicts_the_page_cache_its_workers_mapped(tmp_path, monkeypatch):
    # A worker can fault in chosen pages of the libraries it maps; a later one could read the
    # pattern back. After the exam, with its workers gone, those files leave the page cache.
    cfg = market_config(tmp_path)
    spawned, calls, _ = _spied(monkeypatch)
    tok = issue_holdout_token("fam", "exam test", tmp_path / "test_log.jsonl")
    run_exam(cfg, PluginSource.read("donchian_atr"), "4h", cfg.symbols, {"n": 20}, tok, END)
    assert len(spawned) == 2                                  # one worker per symbol
    assert calls == [({p for w in spawned for p in w.mapped_files}, True)]


def test_a_failed_exam_evicts_the_page_cache_too(tmp_path, monkeypatch):
    cfg = market_config(tmp_path)
    spawned, calls, _ = _spied(monkeypatch)
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


def test_an_exam_in_batches_evicts_each_batch_before_the_next_starts(tmp_path, monkeypatch):
    # Batch 1's workers stream the whole holdout before batch 2's start: marks they left in the
    # page cache would hand batch 2's symbols other symbols' future bars.
    cfg = market_config(tmp_path)
    _, _, events = _spied(monkeypatch)
    monkeypatch.setattr(client, "max_workers", lambda cfg: 1)          # a batch per symbol
    tok = issue_holdout_token("fam", "exam test", tmp_path / "test_log.jsonl")
    run_exam(cfg, PluginSource.read("donchian_atr"), "4h", cfg.symbols, {"n": 20}, tok, END)
    w0, w1 = "/nonexistent/worker-0.so", "/nonexistent/worker-1.so"
    assert events == [("spawn", w0), ("evict", {w0}, True), ("spawn", w1), ("evict", {w1}, True)]


def test_a_rerun_starts_once_the_lost_attempts_files_are_evicted(tmp_path, monkeypatch):
    # The rerun's workers start again from the warm-up, while the lost attempt's had reached later
    # bars: what those left in the page cache goes first.
    cfg = market_config(tmp_path)
    _, _, events = _spied(monkeypatch, lose=0)
    tok = issue_holdout_token("fam", "exam test", tmp_path / "test_log.jsonl")
    run_exam(cfg, PluginSource.read("donchian_atr"), "4h", cfg.symbols, {"n": 20}, tok, END)
    assert [e[0] for e in events] == ["spawn", "spawn", "evict", "spawn", "spawn", "evict"], events
    first, rerun = {e[1] for e in events[:2]}, {e[1] for e in events[3:5]}
    assert events[2] == ("evict", first, True) and events[5] == ("evict", first | rerun, True)   # the batch's


def test_evicting_drops_a_file_from_the_page_cache(caplog):
    # On the real disk (.venv), not a tmpfs, whose pages can't be dropped.
    if not os.access(sys.prefix, os.W_OK):              # a system Python: sys.prefix is /usr
        pytest.skip(f"needs a scratch file in sys.prefix, and {sys.prefix} isn't writable")
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


def test_a_batch_evicts_its_workers_files_itself(tmp_path, monkeypatch):
    # The eviction is run_batch's own job, before a rerun and once it ends, so no caller can leave
    # it out: here it is called directly, with a lost first attempt.
    cfg = market_config(tmp_path)
    _, _, events = _spied(monkeypatch, lose=0)
    tok = issue_holdout_token("fam", "exam test", tmp_path / "test_log.jsonl")
    start_ms = cfg.holdout_start.value // 10**6

    def load(s):
        bars = load_bars(cfg, s, "4h", end=END, holdout=tok)
        return bars, int(np.searchsorted(bars["open_time"].to_numpy(), start_ms)), EngineParams.from_config(cfg, s)
    stream.run_batch(cfg, PluginSource.read("donchian_atr"), "4h", {"n": 20}, cfg.symbols, load)
    assert [e[0] for e in events] == ["spawn", "spawn", "evict", "spawn", "spawn", "evict"], events
    first, rerun = {e[1] for e in events[:2]}, {e[1] for e in events[3:5]}
    assert events[2] == ("evict", first, True) and events[5] == ("evict", first | rerun, True)
