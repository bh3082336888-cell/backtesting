"""ccxt tail (REST gap-stitch) + live closed-candle stream (ccxt.pro watchOHLCV).

Same candle schema as the vision parquet. Fields ccxt doesn't provide
(quote_volume, trades, taker_buy_*) are NaN — explicit, not invented.
Rate-limit / network errors are logged and backed off; never swallowed.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pandas as pd

from perp_lab.config import TF_MS, ccxt_symbol
from perp_lab.data.vision import CANDLE_SCHEMA, Manifest, dataset_dir

log = logging.getLogger(__name__)

def make_exchange(pro: bool = False):
    import ccxt
    opts = {"enableRateLimit": True, "options": {"defaultType": "swap"}}
    if pro:
        import ccxt.pro as ccxtpro
        return ccxtpro.binance(opts)
    return ccxt.binance(opts)


def _retryable(e: Exception) -> bool:
    import ccxt
    return isinstance(e, (ccxt.RateLimitExceeded, ccxt.DDoSProtection, ccxt.NetworkError))


def call_with_backoff(fn, *a, max_retries: int = 6, **kw):
    for attempt in range(max_retries):
        try:
            return fn(*a, **kw)
        except Exception as e:
            if not _retryable(e) or attempt == max_retries - 1:
                log.error("ccxt call %s failed: %s: %s", getattr(fn, "__name__", fn), type(e).__name__, e)
                raise
            wait = min(60, 2 ** (attempt + 1))
            log.warning("ccxt %s: %s; backing off %ss (attempt %d)", type(e).__name__, e, wait, attempt + 1)
            time.sleep(wait)


async def acall_with_backoff(fn, *a, max_retries: int = 6, **kw):
    for attempt in range(max_retries):
        try:
            return await fn(*a, **kw)
        except Exception as e:
            if not _retryable(e) or attempt == max_retries - 1:
                log.error("ccxt call failed: %s: %s", type(e).__name__, e)
                raise
            wait = min(60, 2 ** (attempt + 1))
            log.warning("ccxt %s: %s; backing off %ss (attempt %d)", type(e).__name__, e, wait, attempt + 1)
            await asyncio.sleep(wait)


def ohlcv_to_candles(rows: list[list], tf: str) -> pd.DataFrame:
    step = TF_MS[tf]
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume"])
    df["open_time"] = df.open_time.astype("int64")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype("float64")
    for c in ["quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[c] = float("nan")
    df["close_time"] = df.open_time + step - 1
    return df[CANDLE_SCHEMA]


def closed_only(df: pd.DataFrame, tf: str, now_ms: int) -> pd.DataFrame:
    return df[df.open_time + TF_MS[tf] <= now_ms]


def fetch_recent(exchange, symbol: str, tf: str, since_ms: int, quote: str = "USDT",
                 now_ms: int | None = None) -> pd.DataFrame:
    """Paginated fetchOHLCV from since_ms to now, closed candles only."""
    now_ms = now_ms or int(time.time() * 1000)
    sym = ccxt_symbol(symbol, quote)
    out, since = [], since_ms
    while since < now_ms:
        rows = call_with_backoff(exchange.fetch_ohlcv, sym, tf, since=since, limit=1500)
        if not rows:
            break
        out += rows
        nxt = rows[-1][0] + TF_MS[tf]
        if nxt <= since:
            break
        since = nxt
    if not out:
        return ohlcv_to_candles([], tf)
    df = ohlcv_to_candles(out, tf).drop_duplicates("open_time").sort_values("open_time")
    return closed_only(df, tf, now_ms).reset_index(drop=True)


def tail(cfg, symbol: str, tf: str, exchange=None) -> int:
    """Stitch the gap between the last parquet candle and now. Returns rows written."""
    exchange = exchange or make_exchange()
    d = dataset_dir(cfg.data_dir, symbol, "klines", tf)
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet for {symbol} {tf}; backfill first")
    last = int(pd.read_parquet(files[-1], columns=["open_time"]).open_time.max())
    new = fetch_recent(exchange, symbol, tf, last + TF_MS[tf], cfg.get("quote", "USDT"))
    if new.empty:
        return 0
    man = Manifest(d)
    new["_m"] = pd.to_datetime(new.open_time, unit="ms", utc=True).dt.strftime("%Y-%m")
    for m, part in new.groupby("_m"):
        part = part.drop(columns="_m")
        f = d / f"{m}.parquet"
        if f.exists():
            part = pd.concat([pd.read_parquet(f), part]).drop_duplicates("open_time", keep="first")
        part = part.sort_values("open_time").reset_index(drop=True)
        part.to_parquet(f, index=False)
        man.mark(m, complete=False, source="ccxt", rows=len(part))
    return len(new)


async def watch_closed(exchange, symbol: str, tf: str, quote: str = "USDT"):
    """Yield each candle once it is closed (the next candle has started)."""
    sym = ccxt_symbol(symbol, quote)
    pending_ts = None
    while True:
        rows = await acall_with_backoff(exchange.watch_ohlcv, sym, tf)
        if not rows:
            continue
        by_ts = {int(r[0]): r for r in rows}
        latest = max(by_ts)
        if pending_ts is not None and latest > pending_ts:
            if pending_ts in by_ts:
                yield ohlcv_to_candles([by_ts[pending_ts]], tf).iloc[0]
            else:
                log.warning("%s %s: closed candle %s fell out of the stream cache; refetch on restart",
                            symbol, tf, pending_ts)
        if pending_ts is None or latest > pending_ts:
            pending_ts = latest
