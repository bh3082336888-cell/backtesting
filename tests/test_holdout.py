import numpy as np
import pandas as pd
import pytest

from perp_lab.config import Config
from perp_lab.data.store import load_candles
from perp_lab.holdout import HoldoutToken, HoldoutViolation, issue_holdout_token
from perp_lab.validate.walkforward import make_folds


@pytest.fixture
def cfg(tmp_path):
    d = tmp_path / "data" / "BTCUSDT" / "1h"
    d.mkdir(parents=True)
    t0 = int(pd.Timestamp("2026-05-01", tz="UTC").timestamp() * 1000)
    ot = t0 + np.arange(92 * 24, dtype="int64") * 3_600_000
    df = pd.DataFrame({"open_time": ot, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
                       "quote_volume": 1.0, "trades": 1.0, "taker_buy_volume": 1.0,
                       "taker_buy_quote_volume": 1.0, "close_time": ot + 3_599_999})
    df.to_parquet(d / "all.parquet")
    return Config(raw={"data_dir": str(tmp_path / "data"), "runs_dir": str(tmp_path / "runs"),
                       "history_start": "2026-05-01", "holdout_start": "2026-07-01"})


def test_default_load_stops_before_holdout(cfg):
    c = load_candles(cfg, "BTCUSDT", "1h")
    assert c.index.max() < cfg.holdout_start
    assert len(c) == 61 * 24


def test_explicit_read_into_holdout_raises(cfg):
    with pytest.raises(HoldoutViolation):
        load_candles(cfg, "BTCUSDT", "1h", end=pd.Timestamp("2026-07-15", tz="UTC"))


def test_forged_token_rejected(cfg):
    with pytest.raises(HoldoutViolation):
        HoldoutToken("x", "y", "z", object())


def test_minted_token_reads_holdout_and_is_logged(cfg, tmp_path):
    log = tmp_path / "holdout.jsonl"
    tok = issue_holdout_token("lineage-1", "promotion gate", log)
    c = load_candles(cfg, "BTCUSDT", "1h", holdout=tok, end=pd.Timestamp("2026-08-01", tz="UTC"))
    assert c.index.max() >= cfg.holdout_start
    assert "lineage-1" in log.read_text()


def test_walkforward_folds_never_touch_holdout():
    s, h = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")
    folds = make_folds(s, h, 6, 2)
    assert len(folds) == 24
    for f in folds:
        assert f["train_end"] == f["test_start"] and f["test_end"] <= h
        assert f["train_end"] - f["train_start"] >= pd.Timedelta(days=180)
    tests = [(f["test_start"], f["test_end"]) for f in folds]
    assert all(a[1] == b[0] for a, b in zip(tests, tests[1:]))   # contiguous, non-overlapping OOS
