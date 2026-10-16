import pandas as pd
import pytest

from perp_lab.holdout import HoldoutViolation, issue_holdout_token
from perp_lab.judge.exam import run_exam, window_end
from tests.conftest import market_config

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
        run_exam(cfg, "donchian_atr", "4h", cfg.symbols, {"n": 20}, None, END)


def test_exam_covers_exactly_the_holdout_window(tmp_path):
    cfg = market_config(tmp_path)
    tok = issue_holdout_token("fam", "exam test", tmp_path / "test_log.jsonl")
    res = run_exam(cfg, "donchian_atr", "4h", cfg.symbols, {"n": 20, "atr_mult": 3.0}, tok, END)
    assert len(res.portfolio) == 92                       # 31 Jul + 31 Aug + 30 Sep
    assert res.portfolio.index.min() == cfg.holdout_start
    assert res.portfolio.index.max() == pd.Timestamp("2026-09-30", tz="UTC")
    st = pd.to_datetime(res.trades.signal_time, unit="ms", utc=True)
    assert len(res.trades) > 0 and st.min() >= cfg.holdout_start and st.max() < END
    assert set(res.daily) == {"BTCUSDT", "ETHUSDT"} and res.params == {"n": 20, "atr_mult": 3.0}
