"""Calibrate the per-coin slippage model from pre-holdout bookTicker spreads.

Run from the repo root:  .venv/bin/python docs/research/slippage-calibration/calibrate.py
Downloads about 3 GB one file at a time and keeps only calibration.csv on disk.
Method and results: docs/superpowers/specs/2026-10-18-wider-data-design.md, "Slippage model".
"""

from __future__ import annotations

import hashlib
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from perp_lab.data import vision                                            # noqa: E402
from perp_lab.engine.slippage import (ADV_REF_USD, DAY_MS, fit_spread_curve,  # noqa: E402
                                      half_spread_bps, slippage_bps)

COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK", "ADA", "DOT",
         "LTC", "BCH", "FIL", "AAVE", "XLM", "ATOM", "ETC"]
DAYS = ["2023-06-15", "2023-09-15", "2023-12-15", "2024-03-15"]
TABLE_DAYS = ["2020-08-01", "2022-02-01", "2026-06-30"]
ADV_DAYS = 30
FLOOR_BPS = 2.0
MAX_BPS = 50.0
HOLDOUT = pd.Timestamp("2026-07-01")
BASE = vision.BASE
OUT = Path(__file__).resolve().parent / "calibration.csv"

_klines: dict[tuple[str, str], pd.DataFrame | None] = {}


def _check_day(day: pd.Timestamp) -> None:
    assert day < HOLDOUT, f"refusing to read {day.date()}: on or after the holdout"


def monthly_klines(sym: str, month: str) -> pd.DataFrame | None:
    key = (sym, month)
    if key not in _klines:
        assert pd.Period(month, "M").end_time < HOLDOUT
        url = f"{BASE}/monthly/klines/{sym}/1d/{sym}-1d-{month}.zip"
        blob = vision._get(url)
        if blob is None:
            _klines[key] = None
        else:
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                raw = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None, dtype=str)
            _klines[key] = vision.parse_klines(raw)
    return _klines[key]


def adv_before(sym: str, day: str, days: int = ADV_DAYS) -> float:
    """Mean daily quote volume over the `days` UTC days before `day`; NaN when no day has data."""
    d = pd.Timestamp(day)
    _check_day(d)
    lo = d - pd.Timedelta(days=days)
    parts = []
    for m in pd.period_range(lo, d - pd.Timedelta(days=1), freq="M"):
        k = monthly_klines(sym, str(m))
        if k is not None:
            parts.append(k)
    if not parts:
        return float("nan")
    k = pd.concat(parts)
    lo_ms, hi_ms = int(lo.timestamp() * 1000), int(d.timestamp() * 1000)
    k = k[(k["open_time"] >= lo_ms) & (k["open_time"] < hi_ms)]
    assert hi_ms <= int(HOLDOUT.timestamp() * 1000)
    return float(k["quote_volume"].mean()) if len(k) else float("nan")


def day_half_spread(sym: str, day: str) -> float:
    d = pd.Timestamp(day)
    _check_day(d)
    url = f"{BASE}/daily/bookTicker/{sym}/{sym}-bookTicker-{day}.zip"
    blob = vision._get(url)
    if blob is None:
        raise RuntimeError(f"no bookTicker file: {url}")
    chk = vision._get(url + ".CHECKSUM")
    if chk is not None and chk.decode().split()[0].strip().lower() != hashlib.sha256(blob).hexdigest():
        raise vision.FetchError(f"checksum mismatch for {url}")
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            head = f.readline().decode().strip().lower()
        has_header = head.startswith("update_id")
        with z.open(name) as f:
            df = pd.read_csv(f, header=None, skiprows=1 if has_header else 0,
                             usecols=[1, 3, 5], dtype="float64")
    del blob
    bid, ask, t = df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy(), df.iloc[:, 2].to_numpy()
    del df
    if len(t) and t.max() >= 1e14:       # microsecond timestamps
        t = t // 1000
    order = np.argsort(t, kind="stable")
    start = int(d.timestamp() * 1000)
    return half_spread_bps(bid[order], ask[order], t[order], start + DAY_MS)


def main() -> None:
    rows = []
    for c in COINS:
        sym = f"{c}USDT"
        for day in DAYS:
            hs = day_half_spread(sym, day)
            adv = adv_before(sym, day)
            rows.append({"coin": c, "day": day, "adv_usd": adv, "half_spread_bps": hs})
            print(f"{c} {day} adv={adv:,.0f} half_spread={hs:.4f} bp", flush=True)
    cal = pd.DataFrame(rows)
    cal.to_csv(OUT, index=False)
    print(f"\nwrote {OUT} ({len(cal)} rows)")

    fit = fit_spread_curve(cal["adv_usd"].to_numpy(), cal["half_spread_bps"].to_numpy())
    print("fit:", fit)
    model = {"adv_days": ADV_DAYS, "spread_bps_at_100m": fit["spread_bps_at_100m"],
             "alpha": fit["alpha"], "max_bps": MAX_BPS}

    print(f"\nslippage_bps per coin (floor {FLOOR_BPS}), bar on:")
    print(f"{'coin':6}" + "".join(f"{d:>14}" for d in TABLE_DAYS))
    for c in COINS:
        cells = []
        for d in TABLE_DAYS:
            adv = adv_before(f"{c}USDT", d)
            if not np.isfinite(adv) or adv <= 0:
                cells.append("–")
            else:
                cells.append(f"{slippage_bps(np.array([adv]), FLOOR_BPS, model)[0]:.2f}")
        print(f"{c:6}" + "".join(f"{x:>14}" for x in cells))


if __name__ == "__main__":
    main()
