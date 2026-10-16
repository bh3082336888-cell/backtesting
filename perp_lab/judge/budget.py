"""Holdout budget: the only code allowed to unlock the holdout.

The audit log written by issue_holdout_token is the record of reads. One read = one distinct
judgement_id (entries without one count by nonce). judgement_id must be an int (not None, not
a bool). Under an exclusive file lock: a judgement that already holds a read may retry (its
result was never published) only for the same lineage_id; a judgement_id logged under another
lineage is not a retry and meets every rule below. Otherwise the exam
window must be at least min_holdout_days, total reads stay under holdout_max_reads, and each
family gets one read. A missing or malformed log refuses; it never means "zero reads".
"""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from perp_lab.holdout import HoldoutToken, issue_holdout_token


class BudgetLogError(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetCheck:
    status: str     # mint | pending | blocked
    code: str       # ok | retry | too_short | budget_spent | family_examined


def log_path(cfg) -> Path:
    return Path(cfg["judge"]["holdout_log"])


def init_log(cfg) -> Path:
    p = log_path(cfg)
    if p.exists():
        raise FileExistsError(f"{p} already exists; the holdout read record is never reset")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        raise BudgetLogError(f"holdout log {path} is missing; create it once with `lab holdout-init`")
    entries = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError as ex:
            raise BudgetLogError(f"holdout log {path} line {i} is not JSON: {ex}") from None
        if not isinstance(e, dict) or "lineage_id" not in e or "nonce" not in e:
            raise BudgetLogError(f"holdout log {path} line {i} is not a holdout read entry")
        entries.append(e)
    return entries


def _read_key(e: dict) -> tuple:
    return ("judgement", e["judgement_id"]) if e.get("judgement_id") is not None else ("nonce", e["nonce"])


def reads_used(cfg) -> int:
    return len({_read_key(e) for e in read_log(log_path(cfg))})


def check(cfg, entries: list[dict], lineage_id: str, judgement_id: int, window_days: int) -> BudgetCheck:
    if not isinstance(judgement_id, int) or isinstance(judgement_id, bool):
        raise TypeError(f"judgement_id must be an int, got {judgement_id!r}")
    jc = cfg["judge"]
    if any(e.get("judgement_id") == judgement_id and e["lineage_id"] == lineage_id for e in entries):
        return BudgetCheck("mint", "retry")
    if window_days < jc["min_holdout_days"]:
        return BudgetCheck("pending", "too_short")
    if len({_read_key(e) for e in entries}) >= jc["holdout_max_reads"]:
        return BudgetCheck("blocked", "budget_spent")
    if any(e["lineage_id"] == lineage_id for e in entries):
        return BudgetCheck("blocked", "family_examined")
    return BudgetCheck("mint", "ok")


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{path}.lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def mint(cfg, lineage_id: str, judgement_id: int, window_days: int) -> tuple[BudgetCheck, HoldoutToken | None]:
    """Check the budget and, if allowed, unlock the holdout. The read is logged before any data is read."""
    p = log_path(cfg)
    with _locked(p):
        chk = check(cfg, read_log(p), lineage_id, judgement_id, window_days)
        if chk.status != "mint":
            return chk, None
        return chk, issue_holdout_token(lineage_id, f"judge exam, judgement {judgement_id}", p,
                                        judgement_id=judgement_id)
