"""The judge: promote or kill a strategy plugin against its parent, with no human involved.

Spec: docs/superpowers/specs/2026-10-16-judge-design.md. Flow: pre-checks -> idempotency ->
lookahead (the child's, then the parent's) -> parent run -> child walk-forward -> gates -> decision ->
judgement + lineage edge -> (a promotion supersedes its parent's pending exam) -> public verdict.

`judge` never runs an exam: a promotion waits as `pending` (`awaiting_exam`) for `lab exam`
(`exam_pending` -> `attempt_exam`). The proposer runs `lab judge`, so it must not be able to time
an exam (docs/superpowers/specs/2026-10-17-plugin-sandbox-design.md, "Streamed exam").
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
from perp_lab.judge.exam import cost_hours, run_exam, window_data_problems, window_end
from perp_lab.plugins import PluginMetaError, PluginSource, UndeclaredTimeframe, require_timeframe
from perp_lab.sandbox import client
from perp_lab.sandbox.errors import PluginError, SandboxUnavailable
from perp_lab.runs import new_run_id, repo_commit, write_run
from perp_lab.store import db
from perp_lab.strategy import list_plugins, plugin_hash
from perp_lab.validate.causality import causality_failures, synthetic_bars
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


def read_plugin(name: str) -> PluginSource:
    """One read of a plugin's bytes; computed metadata is a refusal."""
    try:
        return PluginSource.read(name)
    except PluginMetaError as e:
        raise JudgeRefused(f"{name}: {e}") from e


def _require_timeframe(src: PluginSource, tf: str, role: str = "") -> None:
    """A timeframe the plugin doesn't declare is refused before anything runs or is recorded: the
    walk-forward would be a trial that can't be taken back."""
    try:
        require_timeframe(src, tf, role)
    except UndeclaredTimeframe as e:
        raise JudgeRefused(str(e)) from e


def plugin_parent(name: str, src: PluginSource | None = None) -> str | None:
    """The PARENT a plugin declares, read from its source (`src` when already read)."""
    return (src or read_plugin(name)).meta.parent


def resolve_family(cfg, name: str, src: PluginSource | None = None) -> str:
    """The family of a plugin. Walk the PARENT chain from `name` to its root: the first plugin
    name on the way that already has a judgement gives its earliest family, so a judged plugin
    keeps its family (and its descendants with it) even after its PARENT line is removed or
    edited. With no judgement on the chain, the family is `family:<root>`."""
    path, cur = [name], name
    while (fam := db.family_of(cfg, cur)) is None:
        p = plugin_parent(cur, src if cur == name else None)
        if p is None:
            return f"family:{cur}"
        if p in path:
            raise JudgeRefused(f"PARENT chain loops: {' → '.join(path + [p])}")
        if p not in list_plugins():
            raise JudgeRefused(f"PARENT '{p}' is not a plugin in strategies/")
        path.append(p)
        cur = p
    return fam


DATED_LEAD_DAYS = 60       # dated synthetic bars start this long before the holdout
DATED_FIRST_CUT_DAYS = 45  # first cut point, before the holdout start
DATED_CUT_EVERY_DAYS = 30
DATED_AHEAD_DAYS = 30      # at judge time, cover the holdout up to this far past today
_HOUR = pd.Timedelta(hours=1)


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def dated_lookahead_failures(cfg, src, until: pd.Timestamp, params: dict | None = None) -> list[str]:
    """The causality check on synthetic hourly bars dated from DATED_LEAD_DAYS before the holdout
    start to `until`, cut every DATED_CUT_EVERY_DAYS from DATED_FIRST_CUT_DAYS before the holdout
    start and always at `until` itself, so lookahead keyed to any date before `until` is seen (the
    end cut catches any peek that starts before it; the spaced cuts catch peeks confined to a
    stretch). `params` (e.g. the exam's frozen params) is checked as one more variant. No real
    data. The no-signals rule does not apply: a plugin may honestly be quiet in this span. A span
    that hasn't reached the holdout has nothing to check."""
    if until <= cfg.holdout_start:
        return []
    start = cfg.holdout_start - pd.Timedelta(days=DATED_LEAD_DAYS)
    cuts, d = [], cfg.holdout_start - pd.Timedelta(days=DATED_FIRST_CUT_DAYS)
    while d < until:
        cuts.append(int((d - start) / _HOUR))
        d += pd.Timedelta(days=DATED_CUT_EVERY_DAYS)
    cuts.append(int((until - start) / _HOUR))
    bars = synthetic_bars(n=cuts[-1] + 24, seed=1, start=str(start.date()))
    return [f"holdout-dated: {f}"
            for f in causality_failures(cfg, src, bars, cuts=cuts, require_signals=False, params=params)]


def lookahead_failures(cfg, src) -> list[str]:
    """The causality check at the synthetic bars' default dates, plus the dated check over the
    holdout up to DATED_AHEAD_DAYS past today. attempt_exam repeats the dated check over the
    actual exam window just before the read, so dates after the judged span are covered too."""
    until = max(_now(), cfg.holdout_start) + pd.Timedelta(days=DATED_AHEAD_DAYS)
    return list(dict.fromkeys(causality_failures(cfg, src) + dated_lookahead_failures(cfg, src, until)))


def supersede(cfg, parent: str, tf: str, family: str) -> None:
    """A promoted child either beat its passing parent or its parent did not pass, so the
    parent's queued exam must not spend the family's single read: block it as superseded.
    Only within the child's own family (a re-pointed PARENT must not cancel another family's
    exam), and never a parent that already holds the family's read according to the audit log
    (an exam that crashed, was interrupted or later waited as plugin_changed): it retries on that
    read. If the log can't be read, whether it holds a read is unknown, so it is left queued.

    Under the parent's exam lock, taken without waiting: while `lab exam` examines the parent, it
    may be about to mint the family's read, so the parent is left alone (its exam goes ahead)."""
    pj = db.get_judgement(cfg, plugin_hash(parent), tf)
    if pj is None or pj["holdout_status"] != "pending" or pj["lineage_id"] != family:
        return
    with budget.exam_lock(cfg, pj["id"]) as mine:
        if not mine:
            return
        pj = db.get_judgement_by_id(cfg, pj["id"])          # as it is now, under the lock
        if pj["holdout_status"] != "pending":
            return
        try:
            if budget.holds_read(cfg, pj["id"], pj["lineage_id"]):
                return
        except budget.BudgetLogError:
            return
        db.update_holdout(cfg, pj["id"], "blocked", "superseded")
    finish(cfg, pj["id"])


def judge(cfg, name: str, tf: str, symbols: list[str] | None = None, workers: int | None = None, *,
          validate: Callable = validate_strategy) -> dict:
    jc = cfg["judge"]
    src = read_plugin(name)              # read once: this src and h are what is checked, run and recorded
    h = src.hash
    _require_timeframe(src, tf)
    parent = src.meta.parent
    psrc = None
    if parent is not None:
        if parent == name:
            raise JudgeRefused(f"{name} names itself as its PARENT")
        if parent not in list_plugins():
            raise JudgeRefused(f"PARENT '{parent}' is not a plugin in strategies/")
        psrc = read_plugin(parent)
        if psrc.hash == h:
            raise JudgeRefused(f"{name} is byte-identical to its parent {parent}; not a variation")
        _require_timeframe(psrc, tf, "parent ")   # its walk-forward runs at tf too, and is a trial
    # The refusals above read only plugin source (no code runs), so they come first, on any machine.
    client.check(cfg)                    # no sandbox, no judging: SandboxUnavailable propagates

    existing = db.get_judgement(cfg, h, tf)
    if existing:   # (re)write its verdict under the judged strategy's name, even for a renamed copy
        return {**finish(cfg, existing["id"]), "existing": True}

    family = resolve_family(cfg, name, src)
    fails = lookahead_failures(cfg, src)
    if fails:
        jid = db.record_judgement(cfg, {
            "strategy": name, "timeframe": tf, "strategy_hash": h, "child_run_id": None,
            "parent_run_id": None, "lineage_id": family, "decision": "kill",
            "reasons": [f"lookahead: {f}" for f in fails], "holdout_status": "n/a", "holdout_reason": "n/a"})
        return finish(cfg, jid)

    if psrc is not None and lookahead_failures(cfg, psrc):
        raise JudgeRefused(f"parent {parent} fails the lookahead check; fix or judge it first")

    universe = symbols or cfg.symbols
    parent_row = None
    if parent is not None:
        parent_row = db.latest_wf_run(cfg, parent, tf, psrc.hash, universe, lineage_id=family)
        if parent_row is None:
            try:
                prec, _ = validate(cfg, parent, tf, symbols, workers=workers, lineage_id=family, src=psrc)
            except PluginError as e:
                log.warning("parent %s failed in the sandbox: %s\n%s", parent, e.reason(),
                            e.stderr.decode("utf-8", "replace"))
                raise JudgeRefused(f"parent {parent} fails in the sandbox; fix or judge it first") from e
            parent_row = db.run_row(cfg, prec["run_id"])

    try:
        rec, _ = validate(cfg, name, tf, symbols, workers=workers,
                          parent_run_id=parent_row["run_id"] if parent_row else None, lineage_id=family, src=src)
    except PluginError as e:
        log.warning("%s failed in the sandbox: %s\n%s", name, e.reason(), e.stderr.decode("utf-8", "replace"))
        jid = db.record_judgement(cfg, {
            "strategy": name, "timeframe": tf, "strategy_hash": h, "child_run_id": None,
            "parent_run_id": None, "lineage_id": family, "decision": "kill", "reasons": [e.reason()],
            "holdout_status": "n/a", "holdout_reason": "n/a"})
        return finish(cfg, jid)
    child = db.run_row(cfg, rec["run_id"])
    n, _ = db.trial_stats(cfg)          # N store-wide; the spread is the no-edge one (gates)
    cr = oos_portfolio(child["run_dir"])
    d = gates.deflated_sharpe(cr, n)
    reasons = list(rec.get("fail_reasons") or [])
    if d["dsr"] < jc["dsr_min"]:
        reasons.append(f"Deflated Sharpe {d['dsr']:.2f} < {jc['dsr_min']} (N={n})")

    parent_passed, frac = None, None
    if parent_row is not None:
        pr = oos_portfolio(parent_row["run_dir"])
        parent_passed = bool(parent_row["verdict"] == "candidate"
                             and gates.deflated_sharpe(pr, n)["dsr"] >= jc["dsr_min"])
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
    if decision == "promote" and parent is not None:
        try:
            supersede(cfg, parent, tf, family)
        except Exception:
            finish(cfg, jid)   # the verdict file must exist even when this step fails
            raise
    return finish(cfg, jid)    # a promotion stays pending (awaiting_exam): only `lab exam` examines


HOLDOUT_BOOT = 10_000
BACKFILL_HINT = ("run `lab backfill --tfs {tf}` (it rebuilds every incomplete month from data.binance.vision, a "
                 "closed one from its monthly file and the current one from daily files up to yesterday), then "
                 "`lab exam` again")


def holdout_report(j: dict, res, ok: bool, reasons: list[str], m: dict, floor_pct: float) -> str:
    """The holdout run's report.md, for the human (it stays in the run directory)."""
    floor = "n/a" if m["floor"] is None else f"{m['floor']:.1%}"
    return "\n".join([
        f"# Holdout exam: {j['strategy']} @ {j['timeframe']}",
        "",
        f"Judgement {j['id']}, walk-forward run {j['child_run_id']}, family {j['lineage_id']}.",
        "",
        f"- Window: {res.start.date()} to {res.end.date()} (end exclusive), {m['days']} days",
        f"- Frozen params (walk-forward modal): {json.dumps(res.params, sort_keys=True, default=str)}",
        f"- Symbols: {', '.join(res.daily)}",
        "",
        f"## {'PASSED' if ok else 'FAILED'}",
        "",
        *([f"- {r}" for r in reasons] or ["- every check passed"]),
        "",
        "## Key metrics",
        "",
        "| metric | value |",
        "|---|---|",
        f"| Sharpe | {m['sharpe']:.2f} |",
        f"| total return | {m['total_return']:.1%} |",
        f"| max drawdown | {m['max_dd']:.1%} |",
        f"| trades | {m['trades']} |",
        f"| win rate | {m['win_rate']:.0%} |",
        f"| walk-forward p{floor_pct:g} floor for this length | {floor} |",
    ]) + "\n"


def _record_holdout_run(cfg, j: dict, s: dict, res, ok: bool, reasons: list[str], detail: dict) -> str:
    rid = new_run_id()
    metrics = {**summarize(res.portfolio, res.trades), "floor": detail["floor"]}
    rec = {"run_id": rid, "strategy": j["strategy"], "strategy_hash": j["strategy_hash"],
           "hypothesis": s.get("hypothesis"), "repo_commit": repo_commit(), "params": res.params,
           "universe": list(res.daily), "timeframes": [j["timeframe"]], "validation_mode": "holdout",
           "date_start": str(res.start.date()), "date_end": str(res.end.date()),
           "status": "passed" if ok else "failed", "fail_reasons": reasons,
           "metrics": metrics,
           "lineage_id": j["lineage_id"], "parent_run_id": None,
           "created_at": datetime.now(timezone.utc).isoformat(),
           "notes": f"holdout exam of {j['child_run_id']} (judgement {j['id']})"}
    daily = pd.DataFrame({**res.daily, "PORTFOLIO": res.portfolio})
    write_run(cfg, rec, {"holdout_trades": res.trades, "holdout_daily": daily},
              holdout_report(j, res, ok, reasons, metrics, cfg["judge"]["holdout_floor_pct"]))
    return rid


def exam_stderr_dir(cfg, jid: int) -> Path:
    """Where an exam's workers leave their stderr, for humans."""
    return cfg.runs_dir / "_exam_stderr" / f"judgement-{jid}"


def failed_holdout_report(j: dict, s: dict, start: pd.Timestamp, end: pd.Timestamp, err: PluginError,
                          stderr_dir: Path) -> str:
    """The report.md of an exam the plugin failed, for the human (the verdict says only plugin_error)."""
    params = json.dumps(s["sensitivity"]["base_params"], sort_keys=True, default=str)
    return "\n".join([
        f"# Holdout exam: {j['strategy']} @ {j['timeframe']}",
        "",
        f"Judgement {j['id']}, walk-forward run {j['child_run_id']}, family {j['lineage_id']}.",
        "",
        f"- Window: {start.date()} to {end.date()} (end exclusive), {(end - start).days} days",
        f"- Frozen params (walk-forward modal): {params}",
        f"- Symbols: {', '.join(s['universe'])}",
        "",
        "## FAILED: the plugin failed in the exam (plugin_error)",
        "",
        "The read is spent. The public verdict says only `failed` / `plugin_error`.",
        "",
        f"- Error: {err.reason()}",
        f"- Kind: {err.kind}",
        f"- Type: {err.type or '-'}",
        f"- Message: {err.message or '-'}",
        f"- Worker stderr: {stderr_dir}",
    ]) + "\n"


def _record_failed_holdout_run(cfg, j: dict, s: dict, end: pd.Timestamp, err: PluginError) -> str:
    """The holdout run of an exam the plugin failed: no metrics, the error in its reasons and report."""
    rid = new_run_id()
    start = cfg.holdout_start
    reasons = [err.reason()]
    rec = {"run_id": rid, "strategy": j["strategy"], "strategy_hash": j["strategy_hash"],
           "hypothesis": s.get("hypothesis"), "repo_commit": repo_commit(), "params": s["sensitivity"]["base_params"],
           "universe": list(s["universe"]), "timeframes": [j["timeframe"]], "validation_mode": "holdout",
           "date_start": str(start.date()), "date_end": str(end.date()),
           "status": "failed", "fail_reasons": reasons, "metrics": {},
           "lineage_id": j["lineage_id"], "parent_run_id": None,
           "created_at": datetime.now(timezone.utc).isoformat(),
           "notes": f"holdout exam of {j['child_run_id']} (judgement {j['id']}): plugin error"}
    write_run(cfg, rec, {}, failed_holdout_report(j, s, start, end, err, exam_stderr_dir(cfg, j["id"])))
    return rid


def attempt_exam(cfg, jid: int, *, run: Callable = run_exam, end_fn: Callable = window_end,
                 cost_fn: Callable = cost_hours, data_fn: Callable = window_data_problems) -> dict:
    """Try the holdout exam for a pending judgement, under the budget rules. Run by `lab exam`
    only. In order (the sandbox spec's "Streamed exam"):

    1. read the plugin once; an edit since it was judged waits as plugin_changed;
    2. a judgement already holding its read: failed / exam_error after exam_max_attempts, else the
       window end stored at the mint, and on to 3b, then 6 (the read is spent and the window fixed);
    3. the window end (shorter than min_holdout_days: pending / too_short, before any worker
       starts); 3b. where the window's candles came from, from the manifests only: a month not
       from exchange files (stitched from ccxt, incomplete, unknown) waits as pending /
       incomplete_data, with the months under `detail`, so no read is spent and no attempt used
       on candles a plugin may not be able to trade on; 4. the dated lookahead check over it;
       5. the cost probe (too_slow, or plugin_error if the plugin fails it); 6. the sandbox
       check; 7. the judgement read again (no longer pending, e.g. superseded meanwhile: stop,
       no read), then mint, storing the window end (refused, failed / exam_error, once the
       read's attempts are used up); 8. the streamed exam.

    SandboxUnavailable at any step before the mint is pending / exam_error, re-raised: nothing is
    minted and no attempt used. A plugin error in the exam itself is failed / plugin_error (the read
    is spent); its kind and message go only to the holdout run's report, never to the verdict.

    One attempt per judgement at a time (`budget.exam_lock`, held throughout), and one exam at a
    time on this machine (`budget.exams_lock`, likewise): while another `lab exam` examines this
    judgement or any other (an exam can take hours), this returns it unchanged, marked
    `in_progress`, and spends nothing.
    """
    with budget.exam_lock(cfg, jid) as mine, budget.exams_lock(cfg) as alone:
        j = db.get_judgement_by_id(cfg, jid)     # read under the lock: an exam that just ended is final
        if j is None or j["holdout_status"] != "pending":
            return j
        if not (mine and alone):
            return {**j, "in_progress": True}
        return _attempt(cfg, j, run, end_fn, cost_fn, data_fn)


def _attempt(cfg, j: dict, run: Callable, end_fn: Callable, cost_fn: Callable, data_fn: Callable) -> dict:
    """attempt_exam's steps, under the judgement's exam lock."""
    jid = j["id"]
    try:                                                     # 1. these bytes are checked and examined
        src = PluginSource.read(j["strategy"])
    except PluginMetaError:
        src = None
    if src is None or src.hash != j["strategy_hash"]:
        db.update_holdout(cfg, jid, "pending", "plugin_changed")   # never exam code that was not judged
        return finish(cfg, jid)
    child = db.run_row(cfg, j["child_run_id"])
    s = run_summary(child["run_dir"])
    syms, tf, params, fam = s["universe"], j["timeframe"], s["sensitivity"]["base_params"], j["lineage_id"]
    jc = cfg["judge"]
    try:                                    # the sandbox down at any step before the mint: no read, no attempt
        retry = budget.holds_read(cfg, jid, fam)
        if retry:                                            # 2. a retry on the same read
            if budget.attempts(cfg, jid, fam) >= jc["exam_max_attempts"]:   # (the mint re-checks, locked)
                db.update_holdout(cfg, jid, "failed", "exam_error")
                return finish(cfg, jid)
            end = budget.stored_window_end(cfg, jid, fam)
            if end is None:                                  # minted before window ends were stored
                end = end_fn(cfg, syms, tf)
        else:
            end = end_fn(cfg, syms, tf)                      # 3
            if (end - cfg.holdout_start).days < jc["min_holdout_days"]:   # too short: no worker started
                db.update_holdout(cfg, jid, "pending", "too_short")       # (the mint checks again, locked)
                return finish(cfg, jid)
        problems = data_fn(cfg, syms, tf, end)               # 3b. manifests only: no candle is read
        if problems:
            log.warning("exam of judgement %s waits for exchange data:\n  %s", jid, "\n  ".join(problems))
            db.update_holdout(cfg, jid, "pending", "incomplete_data")
            return {**finish(cfg, jid), "detail": [*problems, BACKFILL_HINT.format(tf=tf)]}
        if not retry:
            if dated_lookahead_failures(cfg, src, end, params=params):     # 4
                db.update_holdout(cfg, jid, "blocked", "lookahead")   # caught over the real exam window: no read
                return finish(cfg, jid)
            try:
                hours = cost_fn(cfg, src, tf, syms, params, end)     # 5
            except PluginError as err:                       # on research bars: no read is spent
                log.warning("cost probe for judgement %s: the plugin failed: %s\n%s", jid, err.reason(),
                            err.stderr.decode("utf-8", "replace"))
                db.update_holdout(cfg, jid, "blocked", "plugin_error")
                return finish(cfg, jid)
            if hours > cfg["sandbox"]["exam_cpu_hours"]:
                db.update_holdout(cfg, jid, "blocked", "too_slow")    # no read
                return finish(cfg, jid)
        client.check(cfg)                                    # 6. no sandbox, no mint
    except SandboxUnavailable:
        db.update_holdout(cfg, jid, "pending", "exam_error")
        finish(cfg, jid)
        raise
    if db.get_judgement_by_id(cfg, jid)["holdout_status"] != "pending":   # e.g. superseded since step 1:
        return finish(cfg, jid)                                          # no read
    days = max(0, (end - cfg.holdout_start).days)
    chk, token = budget.mint(cfg, fam, jid, days, window_end=end)   # 7; failed / exam_error past the cap
    if token is None:
        db.update_holdout(cfg, jid, chk.status, chk.code)
        return finish(cfg, jid)
    if chk.code == "retry":                                  # the read was minted after step 2 looked:
        stored = budget.stored_window_end(cfg, jid, fam)     # its window is the one fixed then
        if stored is not None:
            end = stored
    try:                                                     # 8. in this process: the token never leaves it
        try:
            res = run(cfg, src, tf, syms, params, token, end, stderr_dir=exam_stderr_dir(cfg, jid))
        except PluginError as err:                           # the plugin's failure: the read is spent
            hrid = _record_failed_holdout_run(cfg, j, s, end, err)
            log.warning("exam of judgement %s: the plugin failed; see holdout run %s", jid, hrid)
            db.update_holdout(cfg, jid, "failed", "plugin_error", hrid)
        else:
            ok, reasons, detail = gates.holdout_consistent(
                res.portfolio, len(res.trades), oos_portfolio(child["run_dir"]),
                min_trades=jc["holdout_min_trades"], floor_pct=jc["holdout_floor_pct"],
                block=jc["bootstrap_block_days"], n_boot=HOLDOUT_BOOT, seed=gates.seed_from(child["run_id"]))
            hrid = _record_holdout_run(cfg, j, s, res, ok, reasons, detail)
            db.update_holdout(cfg, jid, "passed" if ok else "failed", "passed" if ok else "failed", hrid)
    except BaseException:                                       # a crash, or Ctrl-C: the read is spent
        db.update_holdout(cfg, jid, "pending", "exam_error")   # same judgement may retry on the same read
        finish(cfg, jid)                                        # the verdict file must still exist
        raise
    return finish(cfg, jid)


def withdraw(cfg, jid: int, reason: str) -> dict:
    """Withdraw a pending promotion before its read is minted (`lab withdraw`): it is never examined,
    and its family keeps its one read for a later promotion. A human's decision on the walk-forward
    record (a pre-registered check that failed, for one); the 3b-2 proposer must not be able to run
    it. Refused (budget.WithdrawRefused, nothing changed) without a reason, for a judgement that
    isn't pending, while `lab exam` examines it (its exam lock, taken without waiting) and once its
    read is minted or when the log can't tell (`budget.no_read_held`, under the mint's lock). One
    line goes to `budget.withdrawals_path` before the judgement changes; its verdict is rewritten,
    status and code only (`withdrawn`)."""
    reason = (reason or "").strip()
    if not reason:
        raise budget.WithdrawRefused("a reason is required")
    with budget.exam_lock(cfg, jid) as mine:
        if not mine:
            raise budget.WithdrawRefused(f"judgement {jid} is being examined (or superseded) right now")
        j = db.get_judgement_by_id(cfg, jid)       # read under the lock: only its holder changes its status
        if j is None:
            raise budget.WithdrawRefused(f"no judgement {jid}")
        if j["holdout_status"] != "pending":
            raise budget.WithdrawRefused(f"judgement {jid} is {j['holdout_status']} ({j['holdout_reason']}), "
                                         "not pending")
        with budget.no_read_held(cfg, jid):
            budget.record_withdrawal(cfg, {
                "time": datetime.now(timezone.utc).isoformat(), "judgement_id": jid, "strategy": j["strategy"],
                "timeframe": j["timeframe"], "strategy_hash": j["strategy_hash"], "lineage_id": j["lineage_id"],
                "reason": reason})
            db.update_holdout(cfg, jid, "withdrawn", "withdrawn")
    return finish(cfg, jid)


def exam_pending(cfg, **kw) -> list[dict]:
    """Retry every pending exam, oldest first; one crashing exam never blocks the rest. While
    another `lab exam` is examining, each comes back unchanged, marked `in_progress`."""
    out = []
    for j in db.pending_judgements(cfg):
        try:
            out.append(attempt_exam(cfg, j["id"], **kw))
        except Exception as e:
            log.exception("exam for judgement %s failed", j["id"])
            out.append({**finish(cfg, j["id"]), "error": f"{type(e).__name__}: {e}"})
    return out
