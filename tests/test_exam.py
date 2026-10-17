import pandas as pd
import pytest

from perp_lab.data.store import load_bars
from perp_lab.holdout import HoldoutViolation, issue_holdout_token
from perp_lab.judge.exam import cost_hours, run_exam, window_end
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
