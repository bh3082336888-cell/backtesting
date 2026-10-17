"""Holdout lock.

The holdout window (>= holdout_start) is unreadable from Phase 1–2 code paths.
Reading it requires a HoldoutToken, which can only be minted by
`issue_holdout_token`, and every mint is appended to an audit log. Phase 3's
judge is the only intended caller. The proposer never gets one — that is
enforced by the judge process, not by a prompt.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class HoldoutViolation(RuntimeError):
    pass


_MINT_KEY = object()  # module-private; tokens built without it are rejected


@dataclass(frozen=True)
class HoldoutToken:
    lineage_id: str
    reason: str
    nonce: str
    _key: object

    def __post_init__(self):
        if self._key is not _MINT_KEY:
            raise HoldoutViolation("HoldoutToken must be minted via issue_holdout_token()")


def issue_holdout_token(lineage_id: str, reason: str, log_path: Path,
                        judgement_id: int | None = None, window_end: str | None = None) -> HoldoutToken:
    """Mint a token, logging the read first. `window_end` (ISO) is the end of the window this read
    examines, fixed at its first mint."""
    tok = HoldoutToken(lineage_id, reason, secrets.token_hex(8), _MINT_KEY)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(),
            "lineage_id": lineage_id, "reason": reason, "nonce": tok.nonce, "judgement_id": judgement_id,
            "window_end": window_end,
        }) + "\n")
    return tok


def check_token(token) -> None:
    if not isinstance(token, HoldoutToken) or token._key is not _MINT_KEY:
        raise HoldoutViolation("invalid holdout token")
