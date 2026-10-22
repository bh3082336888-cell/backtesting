"""A judge and an exam never run at once.

Plugin code running while a holdout exam runs could observe the exam through a shared side channel
(the sandbox spec's residual R6, page-cache residue). So every command that runs plugin code holds
`plugin_runs` for its whole run, and an exam holds `exam_exclusive`.

Two flock files under `runs_dir`: R (the readers-writer lock) and T (a turnstile). A shared caller
takes T, then R shared, then drops T; an exam takes T, then R exclusive, and keeps both. While an
exam waits for R it holds T, so new shared callers queue behind it: a stream of judges from the
unattended proposer can't starve it. flock locks belong to the open file description, so each call
opens its own files and the locks hold across threads as well as processes.

Not covered, an accepted residual: `lab paper`, and the dashboard's paper page (dashboard/app.py
replay_divergence, which reaches the sandbox) run plugin code outside this lock.

The locks are not re-entrant: never nest `plugin_runs`, and never take it under `exam_exclusive`.
A nested take deadlocks against itself as soon as an exam is waiting (it holds the turnstile).
"""
from __future__ import annotations

import fcntl
from contextlib import contextmanager


def _open(cfg, name: str):
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    return open(cfg.runs_dir / name, "a+")


@contextmanager
def plugin_runs(cfg):
    """Shared: any number of plugin-running commands at once, never alongside an exam."""
    t, r = _open(cfg, ".turnstile.lock"), _open(cfg, ".plugin-runs.lock")
    try:
        try:
            fcntl.flock(t, fcntl.LOCK_EX)
            fcntl.flock(r, fcntl.LOCK_SH)
        finally:
            t.close()                       # closing releases T (and is safe if it was never taken)
        yield
    finally:
        r.close()


@contextmanager
def exam_exclusive(cfg):
    """Exclusive: waits for running plugin commands to finish, and holds new ones off meanwhile."""
    t, r = _open(cfg, ".turnstile.lock"), _open(cfg, ".plugin-runs.lock")
    try:
        fcntl.flock(t, fcntl.LOCK_EX)
        fcntl.flock(r, fcntl.LOCK_EX)
        yield
    finally:
        r.close()
        t.close()
