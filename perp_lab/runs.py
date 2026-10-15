"""Run recording. Every lab command writes a run directory; Phase 2 ingests these
into the results store (schema below mirrors the planned SQLite row)."""

from __future__ import annotations

import json
import math
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MIN_OOS_TRADES = 30


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


def verdict(oos: dict, rnd: dict, sens: dict) -> tuple[str, list[str]]:
    """noise | candidate. Every failed check is listed — no single headline number decides."""
    fails = []
    if oos.get("sharpe", 0) <= 0:
        fails.append(f"OOS Sharpe {oos.get('sharpe', 0):.2f} <= 0")
    if not rnd.get("beats_random"):
        fails.append(f"does not beat random-entry p{rnd.get('percentile', '?')} OOS")
    if sens.get("narrow_island"):
        fails.append(f"narrow parameter island (worst drop {sens.get('worst_drop_at_20pct', 0):.0%} at ±20%)")
    if oos.get("trades", 0) < MIN_OOS_TRADES:
        fails.append(f"only {oos.get('trades', 0)} OOS trades (< {MIN_OOS_TRADES})")
    return ("noise" if fails else "candidate"), fails


def write_run(cfg, record: dict, frames: dict[str, pd.DataFrame], report_md: str | None = None) -> Path:
    d = cfg.runs_dir / record["run_id"]
    d.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        if df is not None and len(df):
            df.to_parquet(d / f"{name}.parquet")
    record = {**record, "artifacts": {n: str(d / f"{n}.parquet") for n, df in frames.items()
                                       if df is not None and len(df)}}
    (d / "summary.json").write_text(json.dumps(_clean(record), indent=1))
    if report_md:
        (d / "report.md").write_text(report_md)
    from perp_lab.store.db import record_run
    record_run(cfg, record, d)       # no run exists that isn't recorded
    return d
