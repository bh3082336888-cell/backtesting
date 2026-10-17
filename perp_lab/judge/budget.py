"""Holdout budget: the only code allowed to unlock the holdout.

The audit log written by issue_holdout_token is the record of reads. One read = one distinct
judgement_id (entries without one count by nonce). judgement_id must be an int (not None, not
a bool). Under an exclusive file lock: a judgement that already holds a read may retry (its
result was never published) only for the same lineage_id; a judgement_id logged under another
lineage is not a retry and meets every rule below. Otherwise the exam
window must be at least min_holdout_days, total reads stay under holdout_max_reads, and each
family gets one read. A missing or malformed log refuses; it never means "zero reads".

Each mint logs the end of the window it unlocks. A read's window is fixed at its first mint: a
retry examines the same window (`stored_window_end`), however much data has arrived since. A read
gets `judge.exam_max_attempts` attempts, one log entry each; past that, a retry is refused under the
same lock as the mint (`failed: exam_error`), so two exams can't both take the last attempt.
"""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from perp_lab.holdout import HoldoutToken, issue_holdout_token


class BudgetLogError(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetCheck:
    status: str     # mint | pending | blocked | failed
    code: str       # ok | retry | too_short | budget_spent | family_examined | exam_error


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


def _of(entries: list[dict], judgement_id: int, lineage_id: str | None) -> list[dict]:
    """A judgement's entries (under lineage_id, when given): its read's first attempt and retries."""
    return [e for e in entries if e.get("judgement_id") == judgement_id
            and (lineage_id is None or e["lineage_id"] == lineage_id)]


def _holds(entries: list[dict], judgement_id: int, lineage_id: str) -> bool:
    return bool(_of(entries, judgement_id, lineage_id))


def holds_read(cfg, judgement_id: int, lineage_id: str) -> bool:
    """Whether this judgement already holds its family's read (it was minted, whatever happened
    after). Raises BudgetLogError if the log is missing or malformed: callers must not guess."""
    return _holds(read_log(log_path(cfg)), judgement_id, lineage_id)


def attempts(cfg, judgement_id: int, lineage_id: str | None = None) -> int:
    """How many times this judgement's read was minted (its first attempt and each retry); only
    its entries under `lineage_id`, when given. Raises BudgetLogError like holds_read."""
    return len(_of(read_log(log_path(cfg)), judgement_id, lineage_id))


def stored_window_end(cfg, judgement_id: int, lineage_id: str | None = None) -> pd.Timestamp | None:
    """The window end fixed when this judgement's read was minted: the first of its entries (under
    `lineage_id`, when given) that has one. None when none has (no read yet, or 3a entries only)."""
    return _fixed_end(_of(read_log(log_path(cfg)), judgement_id, lineage_id), log_path(cfg))


def _fixed_end(entries: list[dict], path: Path) -> pd.Timestamp | None:
    """The window end of the first of a read's entries that has one."""
    for e in entries:
        if e.get("window_end") is not None:
            try:
                end = pd.Timestamp(e["window_end"])
            except (TypeError, ValueError) as ex:
                raise BudgetLogError(f"holdout log {path}: bad window_end {e['window_end']!r}") from ex
            if end.tz is None:
                raise BudgetLogError(f"holdout log {path}: window_end {e['window_end']!r} has no timezone")
            return end.tz_convert("UTC")
    return None


def check(cfg, entries: list[dict], lineage_id: str, judgement_id: int, window_days: int) -> BudgetCheck:
    if not isinstance(judgement_id, int) or isinstance(judgement_id, bool):
        raise TypeError(f"judgement_id must be an int, got {judgement_id!r}")
    jc = cfg["judge"]
    held = _of(entries, judgement_id, lineage_id)
    if held:                                    # a retry on the read it holds, within its attempts
        if len(held) >= jc["exam_max_attempts"]:
            return BudgetCheck("failed", "exam_error")
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


@contextmanager
def exam_lock(cfg, judgement_id: int):
    """One `lab exam` at a time per judgement: a non-blocking exclusive lock on
    `<holdout_log>.exam-<id>.lock`, held for the whole attempt. Yields whether this caller holds
    it; False while another process or thread examines the judgement. A crashed exam's lock goes
    with its process."""
    if not isinstance(judgement_id, int) or isinstance(judgement_id, bool):
        raise TypeError(f"judgement_id must be an int, got {judgement_id!r}")
    p = Path(f"{log_path(cfg)}.exam-{judgement_id}.lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def mint(cfg, lineage_id: str, judgement_id: int, window_days: int,
         window_end: pd.Timestamp) -> tuple[BudgetCheck, HoldoutToken | None]:
    """Check the budget and, if allowed, unlock the holdout. The read is logged, with the end of the
    window it examines, before any data is read. A retry examines the window fixed at its read's
    first mint, so that is the end it logs (whatever `window_end` says); 3a entries have none."""
    if not isinstance(window_end, pd.Timestamp) or window_end.tz is None:
        raise TypeError(f"window_end must be a tz-aware pd.Timestamp, got {window_end!r}")
    p = log_path(cfg)
    with _locked(p):
        entries = read_log(p)
        chk = check(cfg, entries, lineage_id, judgement_id, window_days)
        if chk.status != "mint":
            return chk, None
        fixed = _fixed_end(_of(entries, judgement_id, lineage_id), p) if chk.code == "retry" else None
        if fixed is not None:
            window_end = fixed
        return chk, issue_holdout_token(lineage_id, f"judge exam, judgement {judgement_id}", p,
                                        judgement_id=judgement_id, window_end=window_end.isoformat())
