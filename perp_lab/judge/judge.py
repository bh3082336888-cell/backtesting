"""The judge: promote or kill a strategy plugin against its parent, with no human involved.

Spec: docs/superpowers/specs/2026-10-16-judge-design.md. Flow: pre-checks -> idempotency ->
lookahead -> parent run -> child walk-forward -> gates -> decision -> judgement + lineage edge ->
(optional holdout exam via `attempt`) -> public verdict.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from perp_lab.engine.metrics import summarize
from perp_lab.judge import budget, gates
from perp_lab.judge.exam import run_exam, window_end
from perp_lab.runs import new_run_id, repo_commit, write_run
from perp_lab.store import db
from perp_lab.strategy import list_plugins, load_strategy_class, plugin_hash, plugin_parent
from perp_lab.validate.causality import causality_failures
from perp_lab.validate.pipeline import validate_strategy

log = logging.getLogger(__name__)

PARENT_BOOT = 5_000
VERDICT_KEYS = {"strategy", "timeframe", "strategy_hash", "lineage_id", "parent_run_id", "decision",
                "reasons", "dsr", "n_trials", "sr0", "parent_passed", "beats_parent_frac",
                "holdout_status", "holdout_reason", "walkforward"}


class JudgeRefused(Exception):
    """A pre-check failed; nothing was recorded."""


def oos_portfolio(run_dir) -> pd.Series:
    return pd.read_parquet(Path(run_dir) / "oos_daily.parquet")["PORTFOLIO"].dropna()


def run_summary(run_dir) -> dict:
    return json.loads((Path(run_dir) / "summary.json").read_text())


def verdict_path(cfg, strategy: str, tf: str, strategy_hash: str) -> Path:
    return Path(cfg["judge"]["verdicts_dir"]) / f"{strategy}@{tf}@{strategy_hash[:10]}.json"


def write_verdict(cfg, j: dict) -> Path:
    """The proposer's only window on a judgement. Holdout outcome as status + fixed code only:
    never a number computed from holdout data, never the holdout run's id."""
    wf = None
    if j["child_run_id"]:
        run_dir = db.run_row(cfg, j["child_run_id"])["run_dir"]
        wf = {"run_id": j["child_run_id"], "metrics": run_summary(run_dir).get("metrics"),
              "report": str(Path(run_dir) / "report.md")}
    out = {k: j[k] for k in VERDICT_KEYS - {"walkforward"}}
    out["walkforward"] = wf
    p = verdict_path(cfg, j["strategy"], j["timeframe"], j["strategy_hash"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=str))
    return p


def finish(cfg, jid: int) -> dict:
    j = db.get_judgement_by_id(cfg, jid)
    return {**j, "verdict_path": str(write_verdict(cfg, j))}


def resolve_family(name: str) -> str:
    """The family of a plugin: that of the root of its PARENT chain, by name, so it never splits."""
    path, cur = [name], name
    while (p := plugin_parent(cur)) is not None:
        if p in path:
            raise JudgeRefused(f"PARENT chain loops: {' → '.join(path + [p])}")
        if p not in list_plugins():
            raise JudgeRefused(f"PARENT '{p}' is not a plugin in strategies/")
        path.append(p)
        cur = p
    return f"family:{cur}"


def judge(cfg, name: str, tf: str, symbols: list[str] | None = None, workers: int | None = None, *,
          validate: Callable = validate_strategy, attempt: Callable | None = None) -> dict:
    jc = cfg["judge"]
    parent = plugin_parent(name)
    if parent is not None:
        if parent == name:
            raise JudgeRefused(f"{name} names itself as its PARENT")
        if parent not in list_plugins():
            raise JudgeRefused(f"PARENT '{parent}' is not a plugin in strategies/")
    h = plugin_hash(name)
    if parent is not None and plugin_hash(parent) == h:
        raise JudgeRefused(f"{name} is byte-identical to its parent {parent}; not a variation")

    existing = db.get_judgement(cfg, h, tf)
    if existing:
        return {**existing, "verdict_path": str(verdict_path(cfg, name, tf, h)), "existing": True}

    family = resolve_family(name)
    fails = causality_failures(load_strategy_class(name))
    if fails:
        jid = db.record_judgement(cfg, {
            "strategy": name, "timeframe": tf, "strategy_hash": h, "child_run_id": None,
            "parent_run_id": None, "lineage_id": family, "decision": "kill",
            "reasons": [f"lookahead: {f}" for f in fails], "holdout_status": "n/a", "holdout_reason": "n/a"})
        return finish(cfg, jid)

    universe = symbols or cfg.symbols
    parent_row = None
    if parent is not None:
        parent_row = db.latest_wf_run(cfg, parent, tf, plugin_hash(parent), universe,
                                      lineage_id=family)
        if parent_row is None:
            prec, _ = validate(cfg, parent, tf, symbols, workers=workers, lineage_id=family)
            parent_row = db.run_row(cfg, prec["run_id"])

    rec, _ = validate(cfg, name, tf, symbols, workers=workers,
                      parent_run_id=parent_row["run_id"] if parent_row else None, lineage_id=family)
    child = db.run_row(cfg, rec["run_id"])
    n, v = db.trial_stats(cfg)
    cr = oos_portfolio(child["run_dir"])
    d = gates.deflated_sharpe(cr, n, v)
    reasons = list(rec.get("fail_reasons") or [])
    if d["dsr"] < jc["dsr_min"]:
        reasons.append(f"Deflated Sharpe {d['dsr']:.2f} < {jc['dsr_min']} (N={n})")

    parent_passed, frac = None, None
    if parent_row is not None:
        pr = oos_portfolio(parent_row["run_dir"])
        parent_passed = bool(parent_row["verdict"] == "candidate"
                             and gates.deflated_sharpe(pr, n, v)["dsr"] >= jc["dsr_min"])
        if parent_passed:
            frac = gates.beats_parent(cr, pr, block=jc["bootstrap_block_days"], n_boot=PARENT_BOOT,
                                      seed=gates.seed_from(child["run_id"]))
            if frac < jc["parent_beat_frac"]:
                reasons.append(f"beats its passing parent in {frac:.0%} of resamples "
                               f"< {jc['parent_beat_frac']:.0%}")

    decision = "kill" if reasons else "promote"
    jid = db.record_judgement(cfg, {
        "strategy": name, "timeframe": tf, "strategy_hash": h, "child_run_id": child["run_id"],
        "parent_run_id": parent_row["run_id"] if parent_row else None, "lineage_id": family,
        "decision": decision, "reasons": reasons, "dsr": d["dsr"], "n_trials": n, "sr0": d["sr0"],
        "parent_passed": parent_passed, "beats_parent_frac": frac,
        "holdout_status": "pending" if decision == "promote" else "n/a",
        "holdout_reason": "awaiting_exam" if decision == "promote" else "n/a"})
    if parent_row is not None:
        db.add_edge(cfg, parent_row["run_id"], child["run_id"], decision,
                    "; ".join(reasons) or "passed every gate", decided_by="harness")
    if decision == "promote" and attempt is not None:
        attempt(cfg, jid)
    return finish(cfg, jid)


HOLDOUT_BOOT = 10_000


def _record_holdout_run(cfg, j: dict, s: dict, res, ok: bool, reasons: list[str], detail: dict) -> str:
    rid = new_run_id()
    rec = {"run_id": rid, "strategy": j["strategy"], "strategy_hash": j["strategy_hash"],
           "hypothesis": s.get("hypothesis"), "repo_commit": repo_commit(), "params": res.params,
           "universe": list(res.daily), "timeframes": [j["timeframe"]], "validation_mode": "holdout",
           "date_start": str(res.start.date()), "date_end": str(res.end.date()),
           "status": "passed" if ok else "failed", "fail_reasons": reasons,
           "metrics": {**summarize(res.portfolio, res.trades), "floor": detail["floor"]},
           "lineage_id": j["lineage_id"], "parent_run_id": None,
           "created_at": datetime.now(timezone.utc).isoformat(),
           "notes": f"holdout exam of {j['child_run_id']} (judgement {j['id']})"}
    daily = pd.DataFrame({**res.daily, "PORTFOLIO": res.portfolio})
    write_run(cfg, rec, {"holdout_trades": res.trades, "holdout_daily": daily})
    return rid


def attempt_exam(cfg, jid: int, *, run: Callable = run_exam, end_fn: Callable = window_end) -> dict:
    """Try the holdout exam for a pending judgement, under the budget rules."""
    j = db.get_judgement_by_id(cfg, jid)
    if j is None or j["holdout_status"] != "pending":
        return j
    child = db.run_row(cfg, j["child_run_id"])
    s = run_summary(child["run_dir"])
    syms, tf = s["universe"], j["timeframe"]
    end = end_fn(cfg, syms, tf)
    days = max(0, (end - cfg.holdout_start).days)
    chk, token = budget.mint(cfg, j["lineage_id"], jid, days)
    if token is None:
        db.update_holdout(cfg, jid, chk.status, chk.code)
        return finish(cfg, jid)
    try:
        res = run(cfg, j["strategy"], tf, syms, s["sensitivity"]["base_params"], token, end)
    except Exception:
        db.update_holdout(cfg, jid, "pending", "exam_error")   # same judgement may retry on the same read
        raise
    jc = cfg["judge"]
    ok, reasons, detail = gates.holdout_consistent(
        res.portfolio, len(res.trades), oos_portfolio(child["run_dir"]),
        min_trades=jc["holdout_min_trades"], floor_pct=jc["holdout_floor_pct"],
        block=jc["bootstrap_block_days"], n_boot=HOLDOUT_BOOT, seed=gates.seed_from(child["run_id"]))
    hrid = _record_holdout_run(cfg, j, s, res, ok, reasons, detail)
    db.update_holdout(cfg, jid, "passed" if ok else "failed", "passed" if ok else "failed", hrid)
    return finish(cfg, jid)


def exam_pending(cfg, **kw) -> list[dict]:
    """Retry every pending exam, oldest first."""
    return [attempt_exam(cfg, j["id"], **kw) for j in db.pending_judgements(cfg)]
