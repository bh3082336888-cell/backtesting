import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from perp_lab.config import load_config
from perp_lab.strategy import HOLD, Signal, Strategy
from perp_lab.validate.causality import synthetic_bars as make_bars  # noqa: F401  (tests import it from here)

H = 3_600_000

# The repo's real plugins, found from this file rather than the working directory, so a test
# run from elsewhere fails loudly instead of collecting an empty parameter set.
REPO = Path(__file__).resolve().parents[1]
REAL_PLUGINS = sorted(p.stem for p in (REPO / "strategies").glob("*.py") if not p.stem.startswith("_"))


def flat_bars(n=50, price=100.0, tf_ms=H, start="2023-01-01"):
    t0 = int(pd.Timestamp(start, tz="UTC").value // 10**6)
    ot = t0 + np.arange(n) * tf_ms
    df = pd.DataFrame({"open_time": ot, "open": price, "high": price, "low": price, "close": price,
                       "volume": 1.0, "close_time": ot + tf_ms - 1, "funding_charge": 0.0,
                       "regime": "ranging"})
    df.index = pd.to_datetime(df.open_time, unit="ms", utc=True)
    return df


class Scripted(Strategy):
    """Test helper: emits signals from a {bar_index: Signal} schedule."""
    name = "scripted"
    params = {}

    def __init__(self, schedule):
        self.p = {}
        self.schedule = schedule

    def prepare(self, bars):
        return bars[[]]

    def on_bar(self, ctx):
        return self.schedule.get(ctx.i, HOLD)


@pytest.fixture(autouse=True)
def _sandbox(request, monkeypatch):
    """Tests marked `sandbox` use real workers, and skip where this machine can't confine one; no
    worker may outlive its test. Every other test gets the in-process twin as `client.spawn` and a
    `client.check` that passes."""
    from perp_lab.sandbox import client, confine
    from tests import sandbox_fixtures

    if request.node.get_closest_marker("sandbox") is None:
        monkeypatch.setattr(client, "spawn", sandbox_fixtures.InProcessWorker)
        monkeypatch.setattr(client, "check", lambda cfg: 8)
        yield
        return
    reason = confine.probe()
    if reason is not None:
        pytest.skip(f"sandbox unavailable: {reason}")
    yield
    left = sandbox_fixtures.running_workers()
    if left:
        sandbox_fixtures.kill_workers(left)
        pytest.fail(f"sandbox workers left running: {left}")


@pytest.fixture
def bars():
    return make_bars()


CANDLE_COLS = ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades",
               "taker_buy_volume", "taker_buy_quote_volume", "close_time"]


def write_market(data_dir, symbols, candles_until="2026-10-15", funding_until="2026-09-30 16:00"):
    """Synthetic 4h candles from 2025-01-01 and 8h funding, laid out like `lab backfill` output,
    with its klines manifest: closed months from the monthly files, the last one (still open on
    `candles_until`) from daily files and incomplete."""
    t0, t1 = pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp(candles_until, tz="UTC")
    n = int((t1 - t0) / pd.Timedelta(hours=4))
    months = pd.period_range(t0.tz_localize(None), t1.tz_localize(None), freq="M")
    manifest = {str(m): {"complete": m != months[-1], "source": "monthly" if m != months[-1] else "daily",
                         "rows": 0} for m in months}
    for i, s in enumerate(symbols):
        b = make_bars(n=n, seed=10 + i, start="2025-01-01", tf_ms=4 * H)
        d = Path(data_dir) / s / "4h"
        d.mkdir(parents=True, exist_ok=True)
        b[CANDLE_COLS].reset_index(drop=True).to_parquet(d / "all.parquet")
        (d / "_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
        ms = pd.date_range("2025-01-01", funding_until, freq="8h", tz="UTC").as_unit("ms").asi8
        fd = Path(data_dir) / s / "funding"
        fd.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"calc_time": ms, "interval_hours": 8.0, "funding_rate": 0.0001,
                      "funding_time": ms}).to_parquet(fd / "all.parquet")


def market_config(tmp_path, **sections):
    """lab.yaml copy on disk over a synthetic 2-symbol market: research 2025-01-01..2026-07-01,
    holdout from 2026-07-01, store and judge files under tmp_path. Dict values update sections."""
    raw = yaml.safe_load(open("config/lab.yaml"))
    raw.update(data_dir=str(tmp_path / "data"), runs_dir=str(tmp_path / "runs"),
               store={"url": f"sqlite:///{tmp_path}/lab.db"}, history_start="2025-01-01",
               universe=["BTC", "ETH"])
    raw["judge"] = {**raw["judge"], "holdout_log": str(tmp_path / "runs" / "holdout_log.jsonl"),
                    "verdicts_dir": str(tmp_path / "runs" / "verdicts")}
    for k, v in sections.items():
        raw[k] = {**raw.get(k, {}), **v} if isinstance(v, dict) else v
    p = tmp_path / "lab.yaml"
    p.write_text(yaml.safe_dump(raw))
    cfg = load_config(p)
    write_market(cfg.data_dir, cfg.symbols)
    return cfg
