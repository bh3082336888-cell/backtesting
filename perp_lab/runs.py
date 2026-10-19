"""Run recording. Every lab command writes a run directory; Phase 2 ingests these
into the results store (schema below mirrors the planned SQLite row)."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MIN_OOS_TRADES_PER_COIN = 30 / 8        # today's 30 at 8 coins


def per_coin_min(per_coin: float, n_coins: int) -> int:
    """A trade minimum that scales with the number of coins behind it."""
    return math.ceil(per_coin * n_coins)


def data_identity(cfg, symbols: list[str]) -> dict:
    """What a run's numbers depend on besides the plugin: history, coins and costs."""
    from perp_lab.engine.slippage import COST_MODEL_VERSION
    return {"history_start": str(cfg.history_start.date()), "universe": list(symbols),
            "cost_model": COST_MODEL_VERSION, "costs": cfg["costs"],
            "brackets": {s: cfg.bracket(s) for s in symbols}}


def _canon(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return float(x)
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    return x


def data_id(identity: dict) -> str:
    """16 hex characters of the identity's hash; 2 and 2.0 hash alike."""
    blob = json.dumps(_canon(identity), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def new_run_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def repo_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) else (str(f) if math.isinf(f) else f)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def verdict(oos: dict, rnd: dict, sens: dict, n_coins: int) -> tuple[str, list[str]]:
    """noise | candidate. Every failed check is listed — no single headline number decides."""
    fails = []
    if oos.get("sharpe", 0) <= 0:
        fails.append(f"OOS Sharpe {oos.get('sharpe', 0):.2f} <= 0")
    if not rnd.get("beats_random"):
        fails.append(f"does not beat random-entry p{rnd.get('percentile', '?')} OOS")
    if sens.get("narrow_island"):
        fails.append(f"narrow parameter island (worst drop {sens.get('worst_drop_at_20pct', 0):.0%} at ±20%)")
    t, m = oos.get("trades", 0), per_coin_min(MIN_OOS_TRADES_PER_COIN, n_coins)
    if t < m:
        fails.append(f"only {t} OOS trades (< {m}: {MIN_OOS_TRADES_PER_COIN:g} per coin × {n_coins} coins)")
    return ("noise" if fails else "candidate"), fails


def write_run(cfg, record: dict, frames: dict[str, pd.DataFrame], report_md: str | None = None) -> Path:
    d = cfg.runs_dir / record["run_id"]
    d.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        if df is not None and len(df):
            df.to_parquet(d / f"{name}.parquet")
    ident = data_identity(cfg, record["universe"])
    record = {**record, "data": ident, "data_id": data_id(ident), "artifacts": {n: str(d / f"{n}.parquet") for n, df in frames.items()
                                       if df is not None and len(df)}}
    (d / "summary.json").write_text(json.dumps(_clean(record), indent=1))
    if report_md:
        (d / "report.md").write_text(report_md)
    from perp_lab.store.db import record_run
    record_run(cfg, record, d)       # no run exists that isn't recorded
    return d
