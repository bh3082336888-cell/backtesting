import json

import pandas as pd
import pytest

from perp_lab.data.store import load_bars
from perp_lab.holdout import HoldoutViolation, issue_holdout_token
from perp_lab.judge.exam import cost_hours, run_exam, window_data_problems, window_end
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client
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
