import json

import numpy as np
import pandas as pd

from perp_lab.data import vision
from perp_lab.data.store import load_candles
from tests.conftest import market_config

T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
XML = ("<ListBucketResult>"
       "<Contents><Key>data/futures/um/monthly/klines/SOLUSDT/1h/SOLUSDT-1h-2020-09.zip</Key></Contents>"
       "<Contents><Key>data/futures/um/monthly/klines/SOLUSDT/1h/SOLUSDT-1h-2020-09.zip.CHECKSUM</Key></Contents>"
       "<Contents><Key>data/futures/um/monthly/klines/SOLUSDT/1h/SOLUSDT-1h-2020-10.zip</Key></Contents>"
       "</ListBucketResult>")


def stub(monkeypatch, first):
    """The listing says `first`; every fetch 404s. Returns (lookups, fetched urls)."""
    looks, urls = [], []
    monkeypatch.setattr(vision, "first_archive_month", lambda *a, **kw: looks.append(a[:3]) or first)
    monkeypatch.setattr(vision, "_fetch_zip_csv", lambda url, *a, **kw: urls.append(url) or None)
    return looks, urls


def test_months_before_the_first_archive_file_are_absent_and_never_fetched(tmp_path, monkeypatch):
    cfg = market_config(tmp_path)
    looks, urls = stub(monkeypatch, pd.Period("2025-03", "M"))
    run = lambda: vision.backfill(cfg, ["SOLUSDT"], ["1h"], ["klines"], start=T("2025-01-01"),  # noqa: E731
                                  end=T("2025-03-31"), workers=1)
    assert run()["absent"] == 2 and len(looks) == 1
    assert not [u for u in urls if "2025-01" in u or "2025-02" in u]
    man = json.loads((cfg.data_dir / "SOLUSDT" / "1h" / "_manifest.json").read_text())
    assert man["2025-01"] == man["2025-02"] == {"complete": True, "source": "absent", "rows": 0}
    urls.clear()
    again = run()
    assert again["absent"] == 0 and again["skip"] == 2
    assert not [u for u in urls if "2025-01" in u or "2025-02" in u]


def test_a_complete_group_needs_no_listing(tmp_path, monkeypatch):
    cfg = market_config(tmp_path)
    looks, _ = stub(monkeypatch, pd.Period("2025-03", "M"))
    m = vision.Manifest(cfg.data_dir / "SOLUSDT" / "1h")
    for k in ("2025-01", "2025-02"):
        m.mark(k, complete=True, source="monthly", rows=1)
    vision.backfill(cfg, ["SOLUSDT"], ["1h"], ["klines"], start=T("2025-01-01"), end=T("2025-02-28"), workers=1)
    assert looks == []


def test_metrics_months_before_the_first_daily_file_are_absent(tmp_path, monkeypatch):
    cfg = market_config(tmp_path)
    _, urls = stub(monkeypatch, pd.Period("2025-02", "M"))
    s = vision.backfill(cfg, ["SOLUSDT"], [], ["metrics"], start=T("2025-01-01"), end=T("2025-02-28"), workers=1)
    assert s["absent"] == 1 and not [u for u in urls if "2025-01-" in u]


def test_listing_lookups_use_a_small_retry_budget_of_their_own(tmp_path, monkeypatch):
    cfg = market_config(tmp_path)
    seen = []
    monkeypatch.setattr(vision, "first_archive_month", lambda *a, **kw: seen.append(a[3]) or None)
    monkeypatch.setattr(vision, "_fetch_zip_csv", lambda url, *a, **kw: None)
    vision.backfill(cfg, ["SOLUSDT", "ETHUSDT"], ["1h"], ["klines"], start=T("2025-01-01"),
                    end=T("2025-01-31"), workers=2)
    assert len(seen) == 2 and max(seen) <= 2


def test_an_unreadable_listing_marks_nothing_absent(tmp_path, monkeypatch):
    cfg = market_config(tmp_path)
    _, urls = stub(monkeypatch, None)
    s = vision.backfill(cfg, ["SOLUSDT"], ["1h"], ["klines"], start=T("2025-01-01"), end=T("2025-01-31"), workers=1)
    assert s["absent"] == 0 and s["missing"] == 1 and any("2025-01" in u for u in urls)   # today's behaviour


def test_first_archive_month_reads_the_earliest_zip(monkeypatch):
    seen = []
    monkeypatch.setattr(vision, "_get", lambda url, r=6: seen.append(url) or XML.encode())
    assert vision.first_archive_month("SOLUSDT", "klines", "1h", 1) == pd.Period("2020-09", "M")
    assert "prefix=data/futures/um/monthly/klines/SOLUSDT/1h/" in seen[0]
    vision.first_archive_month("SOLUSDT", "metrics", None, 1)
    assert "prefix=data/futures/um/daily/metrics/SOLUSDT/" in seen[1]


def test_first_archive_month_is_none_when_the_listing_cant_be_read(monkeypatch):
    for body in (None, b"<ListBucketResult></ListBucketResult>", b"not xml"):
        monkeypatch.setattr(vision, "_get", lambda url, r=6, b=body: b)
        assert vision.first_archive_month("SOLUSDT", "klines", "1h", 1) is None

    bad = XML.replace("2020-09", "2020-13").replace("2020-10", "2020-13")
    monkeypatch.setattr(vision, "_get", lambda url, r=6: bad.encode())
    assert vision.first_archive_month("SOLUSDT", "klines", "1h", 1) is None   # no ValueError

    def down(url, r=6):
        raise vision.FetchError("HTTP 503")
    monkeypatch.setattr(vision, "_get", down)
    assert vision.first_archive_month("SOLUSDT", "klines", "1h", 1) is None


def write_1h(cfg, trades):
    t0 = T("2025-01-01").value // 10**6
    ot = t0 + np.arange(len(trades)) * 3_600_000
    tr = np.array(trades, dtype=float)
    d = cfg.data_dir / "BTCUSDT" / "1h"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"open_time": ot, "open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0,
                  "volume": np.where(tr == 0, 0.0, 1.0), "quote_volume": np.where(tr == 0, 0.0, 100.0),
                  "trades": tr, "taker_buy_volume": 0.0, "taker_buy_quote_volume": 0.0,
                  "close_time": ot + 3_599_999}).to_parquet(d / "all.parquet")


def test_zero_trade_candles_after_the_last_trade_load_as_absent(tmp_path):
    cfg = market_config(tmp_path)
    write_1h(cfg, [5, 5, 0, 0, 5, 5, 0, 0, 0])
    assert load_candles(cfg, "BTCUSDT", "1h").trades.tolist() == [5, 5, 0, 0, 5, 5]   # the halt stays


def test_candles_without_a_trade_count_are_never_dropped(tmp_path):
    cfg = market_config(tmp_path)
    write_1h(cfg, [5, 5, np.nan, np.nan])                  # ccxt-stitched
    assert len(load_candles(cfg, "BTCUSDT", "1h")) == 4
