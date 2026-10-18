"""Backfill from data.binance.vision → parquet.

Layout:
    data/{SYMBOL}/{tf}/{YYYY-MM}.parquet        klines
    data/{SYMBOL}/funding/{YYYY-MM}.parquet     funding settlements
    data/{SYMBOL}/metrics/{YYYY-MM}.parquet     OI / long-short metrics (5m)
    data/{SYMBOL}/{dataset}/_manifest.json      {month: {complete, source, rows}}

Idempotent + resumable: complete months are skipped; partial months (current
month, built from daily files) are rebuilt on every run. Checksums verified.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE = "https://data.binance.vision/data/futures/um"
LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
CANDLE_SCHEMA = [
    "open_time", "open", "high", "low", "close", "volume", "quote_volume",
    "trades", "taker_buy_volume", "taker_buy_quote_volume", "close_time",
]


class FetchError(RuntimeError):
    pass


_session_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_session_local, "s", None)
    if s is None:
        s = requests.Session()
        _session_local.s = s
    return s


def _get(url: str, max_retries: int = 6) -> bytes | None:
    """GET with exponential backoff. 404 → None. Rate limits are logged, never swallowed silently."""
    for attempt in range(max_retries):
        try:
            r = _session().get(url, timeout=60)
        except requests.RequestException as e:
            wait = min(60, 2 ** attempt) + random.random()
            log.warning("network error %s (%s); retry %d in %.1fs", url, e, attempt + 1, wait)
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r.content
        if r.status_code == 404:
            return None
        if r.status_code in (418, 429) or r.status_code >= 500:
            ra = r.headers.get("Retry-After")
            wait = float(ra) if ra else min(120, 2 ** (attempt + 1)) + random.random()
            log.warning("HTTP %s on %s; backing off %.1fs (attempt %d)", r.status_code, url, wait, attempt + 1)
            time.sleep(wait)
            continue
        raise FetchError(f"HTTP {r.status_code} for {url}")
    raise FetchError(f"gave up after {max_retries} retries: {url}")


def _fetch_zip_csv(url: str, max_retries: int) -> pd.DataFrame | None:
    blob = _get(url, max_retries)
    if blob is None:
        return None
    chk = _get(url + ".CHECKSUM", max_retries)
    if chk is not None:
        expected = chk.decode().split()[0].strip().lower()
        actual = hashlib.sha256(blob).hexdigest()
        if expected != actual:
            raise FetchError(f"checksum mismatch for {url}")
    else:
        log.warning("no CHECKSUM for %s", url)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = z.namelist()[0]
        raw = z.read(name)
    return pd.read_csv(io.BytesIO(raw), header=None, dtype=str)


def _strip_header(df: pd.DataFrame, first_col_name: str) -> pd.DataFrame:
    if len(df) and str(df.iloc[0, 0]).strip() == first_col_name:
        df = df.iloc[1:]
    return df.reset_index(drop=True)


def _to_ms(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s).astype("int64")
    # Guard against microsecond timestamps (Binance moved some datasets to µs in 2025).
    return v.where(v < 10**14, v // 1000)


def parse_klines(df: pd.DataFrame) -> pd.DataFrame:
    df = _strip_header(df, "open_time")
    df = df.iloc[:, : len(KLINE_COLS)]
    df.columns = KLINE_COLS
    out = pd.DataFrame({
        "open_time": _to_ms(df["open_time"]),
        "close_time": _to_ms(df["close_time"]),
    })
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_volume", "taker_buy_quote_volume"]:
        out[c] = pd.to_numeric(df[c]).astype("float64")
    out["trades"] = pd.to_numeric(df["trades"]).astype("float64")
    return out[CANDLE_SCHEMA]


def parse_funding(df: pd.DataFrame) -> pd.DataFrame:
    df = _strip_header(df, "calc_time")
    out = pd.DataFrame({
        "calc_time": _to_ms(df.iloc[:, 0]),
        "interval_hours": pd.to_numeric(df.iloc[:, 1]).astype("float64"),
        "funding_rate": pd.to_numeric(df.iloc[:, 2]).astype("float64"),
    })
    # Settlement times carry a few ms of jitter; snap to the minute.
    out["funding_time"] = (out["calc_time"] + 30_000) // 60_000 * 60_000
    return out


def parse_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = _strip_header(df, "create_time")
    out = pd.DataFrame({
        # explicit unit: pandas 3 infers second resolution from these strings
        "create_time": pd.to_datetime(df.iloc[:, 0], utc=True).dt.as_unit("ms").astype("int64"),
        "oi": pd.to_numeric(df.iloc[:, 2], errors="coerce"),
        "oi_value": pd.to_numeric(df.iloc[:, 3], errors="coerce"),
        "top_ls_count": pd.to_numeric(df.iloc[:, 4], errors="coerce"),
        "top_ls_pos": pd.to_numeric(df.iloc[:, 5], errors="coerce"),
        "ls_count": pd.to_numeric(df.iloc[:, 6], errors="coerce"),
        "taker_ls_vol": pd.to_numeric(df.iloc[:, 7], errors="coerce"),
    })
    return out


@dataclass
class Job:
    symbol: str
    dataset: str          # klines | fundingRate | metrics
    tf: str | None
    month: pd.Period
    first: pd.Period | None = None   # earliest month the archive holds; earlier months are absent


def first_archive_month(symbol: str, dataset: str, tf: str | None, max_retries: int) -> pd.Period | None:
    """Earliest month with a .zip in the archive listing, or None when it can't be told.

    S3 lists keys in ascending order, so the first page holds the earliest file.
    """
    prefix = {"klines": f"data/futures/um/monthly/klines/{symbol}/{tf}/",
              "fundingRate": f"data/futures/um/monthly/fundingRate/{symbol}/",
              "metrics": f"data/futures/um/daily/metrics/{symbol}/"}[dataset]
    try:
        body = _get(f"{LIST_URL}?delimiter=/&prefix={prefix}", max_retries)
        if body is None:
            log.warning("no archive listing for %s %s %s", symbol, dataset, tf or "")
            return None
        root = ET.fromstring(body)
    except (FetchError, ET.ParseError) as e:
        log.warning("archive listing unreadable for %s %s %s: %s", symbol, dataset, tf or "", e)
        return None
    try:
        months = []
        for el in root.iter():
            if el.tag.rsplit("}", 1)[-1] == "Key" and el.text and el.text.endswith(".zip"):
                m = re.search(r"(\d{4}-\d{2})(?:-\d{2})?\.zip$", el.text)
                if m:
                    months.append(pd.Period(m.group(1), "M"))
    except ValueError as e:      # a key with an impossible month
        log.warning("archive listing for %s %s %s has a bad key: %s", symbol, dataset, tf or "", e)
        return None
    if not months:
        log.warning("archive listing for %s %s %s holds no file", symbol, dataset, tf or "")
        return None
    return min(months)


def _months(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Period]:
    return list(pd.period_range(start.tz_localize(None).to_period("M"),
                                end.tz_localize(None).to_period("M"), freq="M"))


def dataset_dir(data_dir: Path, symbol: str, dataset: str, tf: str | None) -> Path:
    sub = {"klines": tf, "fundingRate": "funding", "metrics": "metrics"}[dataset]
    return data_dir / symbol / sub


class Manifest:
    _lock = threading.Lock()

    def __init__(self, d: Path):
        self.path = d / "_manifest.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def is_complete(self, month: str) -> bool:
        return bool(self.data.get(month, {}).get("complete"))

    def mark(self, month: str, complete: bool, source: str, rows: int):
        with self._lock:
            # Re-read under the lock: other jobs write other months of the same dataset.
            if self.path.exists():
                self.data = json.loads(self.path.read_text())
            self.data[month] = {"complete": complete, "source": source, "rows": rows}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True))
            tmp.replace(self.path)


def _monthly_url(job: Job) -> str:
    m = job.month.strftime("%Y-%m")
    s = job.symbol
    if job.dataset == "klines":
        return f"{BASE}/monthly/klines/{s}/{job.tf}/{s}-{job.tf}-{m}.zip"
    if job.dataset == "fundingRate":
        return f"{BASE}/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"
    raise ValueError(job.dataset)


def _daily_url(job: Job, day: pd.Timestamp) -> str:
    d = day.strftime("%Y-%m-%d")
    s = job.symbol
    if job.dataset == "klines":
        return f"{BASE}/daily/klines/{s}/{job.tf}/{s}-{job.tf}-{d}.zip"
    if job.dataset == "metrics":
        return f"{BASE}/daily/metrics/{s}/{s}-metrics-{d}.zip"
    raise ValueError(job.dataset)


_PARSERS = {"klines": parse_klines, "fundingRate": parse_funding, "metrics": parse_metrics}
_SORT_KEY = {"klines": "open_time", "fundingRate": "funding_time", "metrics": "create_time"}


def run_job(job: Job, data_dir: Path, max_retries: int, today: pd.Timestamp) -> tuple[str, int]:
    d = dataset_dir(data_dir, job.symbol, job.dataset, job.tf)
    man = Manifest(d)
    mkey = job.month.strftime("%Y-%m")
    if man.is_complete(mkey):
        return "skip", 0
    if job.first is not None and job.month < job.first:
        man.mark(mkey, complete=True, source="absent", rows=0)
        return "absent", 0
    parse = _PARSERS[job.dataset]
    month_done = job.month.end_time.tz_localize("UTC") < today.normalize()
    frames, source = [], None

    if job.dataset != "metrics" and month_done:
        raw = _fetch_zip_csv(_monthly_url(job), max_retries)
        if raw is not None:
            frames.append(parse(raw))
            source = "monthly"

    if not frames:
        if job.dataset == "fundingRate":
            # No daily funding files on vision; current month comes from the ccxt tail.
            return "missing", 0
        source = "daily"
        days = pd.date_range(job.month.start_time, job.month.end_time.normalize(), freq="D", tz="UTC")
        missing = 0
        for day in days:
            if day >= today.normalize():
                missing += 1
                continue
            raw = _fetch_zip_csv(_daily_url(job, day), max_retries)
            if raw is None:
                missing += 1
                continue
            frames.append(parse(raw))
        month_done = month_done and missing == 0

    if not frames:
        return "missing", 0
    df = pd.concat(frames, ignore_index=True)
    key = _SORT_KEY[job.dataset]
    df = df.drop_duplicates(subset=[key]).sort_values(key).reset_index(drop=True)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{mkey}.parquet.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(d / f"{mkey}.parquet")
    man.mark(mkey, complete=month_done, source=source, rows=len(df))
    return ("ok" if month_done else "partial"), len(df)


def backfill(cfg, symbols: list[str], tfs: list[str], datasets: list[str],
             start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
             workers: int | None = None) -> dict:
    today = pd.Timestamp.now(tz="UTC")
    start = start or cfg.history_start
    end = end or today
    months = _months(start, end)
    jobs: list[Job] = []
    for s in symbols:
        for ds in datasets:
            if ds == "klines":
                jobs += [Job(s, ds, tf, m) for tf in tfs for m in months]
            elif ds in ("fundingRate", "metrics"):
                jobs += [Job(s, ds, None, m) for m in months]
            else:
                log.warning("dataset %s not supported; skipping", ds)
    workers = workers or cfg["backfill"]["workers"]
    retries = cfg["backfill"]["max_retries"]
    # One listing per (symbol, dataset, tf) group that still has work; a complete group needs none.
    # Lookups run on the pool with a small retry budget of their own: a slow or down listing
    # must not stall the jobs, and a failed one just leaves the group as it was before.
    mans: dict[tuple, Manifest] = {}
    todo: list[tuple] = []
    for j in jobs:
        g = (j.symbol, j.dataset, j.tf)
        if g not in mans:
            mans[g] = Manifest(dataset_dir(cfg.data_dir, j.symbol, j.dataset, j.tf))
        if g not in todo and not mans[g].is_complete(j.month.strftime("%Y-%m")):
            todo.append(g)
    stats = {"absent": 0, "ok": 0, "partial": 0, "skip": 0, "missing": 0, "error": 0}
    log.info("backfill: %d jobs, %d workers", len(jobs), workers)
    with ThreadPoolExecutor(workers) as ex:
        looks = {g: ex.submit(first_archive_month, *g, min(2, retries)) for g in todo}
        firsts = {}
        for g, f in looks.items():
            try:
                firsts[g] = f.result()
            except Exception as e:  # a lookup never aborts the backfill
                log.warning("archive listing failed for %s: %s", g, e)
        for j in jobs:
            j.first = firsts.get((j.symbol, j.dataset, j.tf))
        futs = {ex.submit(run_job, j, cfg.data_dir, retries, today): j for j in jobs}
        for f in as_completed(futs):
            j = futs[f]
            try:
                status, rows = f.result()
            except Exception as e:  # logged and counted, never swallowed silently
                log.error("job failed %s %s %s %s: %s", j.symbol, j.dataset, j.tf, j.month, e)
                status = "error"
            stats[status] += 1
            if status == "missing":
                log.info("missing %s %s %s %s", j.symbol, j.dataset, j.tf or "", j.month)
    log.info("backfill done: %s", stats)
    return stats
