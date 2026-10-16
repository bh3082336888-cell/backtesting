"""Results store. SQLAlchemy Core: SQLite now, Postgres by changing `store.url`.

Every run directory written by the lab is also a row here (write_run calls
record_run). `ingest_runs` backfills any run dir that isn't recorded yet, so
"no run exists that isn't recorded" holds even after a crash.

Status model:
    computed  : noise | candidate        (from the run verdict — the harness decides)
    approved  : paper                    (only via approve_paper, only from candidate)
    live      : never set by this system — live is manual, outside the lab.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa

log = logging.getLogger(__name__)

md = sa.MetaData()

runs = sa.Table(
    "runs", md,
    sa.Column("run_id", sa.String(64), primary_key=True),
    sa.Column("strategy", sa.String(128), nullable=False, index=True),
    sa.Column("strategy_hash", sa.String(64)),
    sa.Column("hypothesis", sa.Text),
    sa.Column("params", sa.JSON),
    sa.Column("universe", sa.JSON),
    sa.Column("timeframes", sa.JSON),
    sa.Column("timeframe", sa.String(8), index=True),
    sa.Column("date_start", sa.String(10)),
    sa.Column("date_end", sa.String(10)),
    sa.Column("oos_start", sa.String(10)),
    sa.Column("validation_mode", sa.String(16), nullable=False, index=True),  # IS|walkforward|holdout|paper
    sa.Column("verdict", sa.String(16)),
    sa.Column("fail_reasons", sa.JSON),
    sa.Column("sharpe", sa.Float), sa.Column("sortino", sa.Float), sa.Column("max_dd", sa.Float),
    sa.Column("total_return", sa.Float), sa.Column("cagr", sa.Float),
    sa.Column("profit_factor", sa.Float), sa.Column("win_rate", sa.Float), sa.Column("avg_r", sa.Float),
    sa.Column("exposure", sa.Float), sa.Column("trades", sa.Integer), sa.Column("liquidations", sa.Integer),
    sa.Column("is_sharpe", sa.Float),
    sa.Column("random_pctile", sa.Float), sa.Column("beats_random", sa.Boolean),
    sa.Column("narrow_island", sa.Boolean),
    sa.Column("metrics", sa.JSON),
    sa.Column("trade_log_path", sa.Text),
    sa.Column("run_dir", sa.Text),
    sa.Column("parent_run_id", sa.String(64), sa.ForeignKey("runs.run_id"), nullable=True),
    sa.Column("lineage_id", sa.String(64), index=True),
    sa.Column("repo_commit", sa.String(64)),
    sa.Column("created_at", sa.DateTime(timezone=True)),
    sa.Column("notes", sa.Text),
    sa.Column("paper_started_at", sa.DateTime(timezone=True)),
    sa.Column("paper_regime_changes", sa.Integer),
    sa.Column("paper_regime_detail", sa.JSON),
    sa.Column("paper_eligible", sa.Boolean),
    sa.Column("paper_checked_at", sa.DateTime(timezone=True)),
)

run_folds = sa.Table(
    "run_folds", md,
    sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.run_id"), primary_key=True),
    sa.Column("fold", sa.Integer, primary_key=True),
    sa.Column("train_start", sa.String(10)), sa.Column("train_end", sa.String(10)),
    sa.Column("test_start", sa.String(10)), sa.Column("test_end", sa.String(10)),
    sa.Column("combo", sa.Text), sa.Column("params", sa.JSON),
    sa.Column("is_score", sa.Float), sa.Column("oos_return", sa.Float), sa.Column("oos_sharpe", sa.Float),
)

run_symbols = sa.Table(
    "run_symbols", md,
    sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.run_id"), primary_key=True),
    sa.Column("symbol", sa.String(32), primary_key=True),
    sa.Column("sharpe", sa.Float), sa.Column("max_dd", sa.Float), sa.Column("total_return", sa.Float),
    sa.Column("trades", sa.Integer), sa.Column("profit_factor", sa.Float), sa.Column("win_rate", sa.Float),
)

run_regimes = sa.Table(
    "run_regimes", md,
    sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.run_id"), primary_key=True),
    sa.Column("regime", sa.String(16), primary_key=True),
    sa.Column("trades", sa.Integer), sa.Column("win_rate", sa.Float), sa.Column("avg_r", sa.Float),
    sa.Column("sum_r", sa.Float), sa.Column("profit_factor", sa.Float), sa.Column("sum_pnl_pct", sa.Float),
)

lineage_edges = sa.Table(
    "lineage_edges", md,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("parent_run_id", sa.String(64), sa.ForeignKey("runs.run_id"), nullable=False),
    sa.Column("child_run_id", sa.String(64), sa.ForeignKey("runs.run_id"), nullable=False),
    sa.Column("decision", sa.String(16), nullable=False),     # promote | kill | manual
    sa.Column("reason", sa.Text, nullable=False),
    sa.Column("decided_by", sa.String(32), nullable=False),   # harness | human
    sa.Column("created_at", sa.DateTime(timezone=True)),
    sa.UniqueConstraint("parent_run_id", "child_run_id"),
)

status_log = sa.Table(
    "status_log", md,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("strategy", sa.String(128), nullable=False, index=True),
    sa.Column("timeframe", sa.String(8), nullable=False),
    sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.run_id")),
    sa.Column("status", sa.String(16), nullable=False),        # paper | retired
    sa.Column("set_by", sa.String(32), nullable=False),
    sa.Column("note", sa.Text),
    sa.Column("created_at", sa.DateTime(timezone=True)),
)

judgements = sa.Table(
    "judgements", md,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("strategy", sa.String(128), nullable=False, index=True),
    sa.Column("timeframe", sa.String(8), nullable=False),
    sa.Column("strategy_hash", sa.String(64), nullable=False),
    sa.Column("child_run_id", sa.String(64), sa.ForeignKey("runs.run_id"), nullable=True),
    sa.Column("parent_run_id", sa.String(64), sa.ForeignKey("runs.run_id"), nullable=True),
    sa.Column("lineage_id", sa.String(64), index=True),
    sa.Column("decision", sa.String(16), nullable=False),          # promote | kill
    sa.Column("reasons", sa.JSON, nullable=False),
    sa.Column("dsr", sa.Float), sa.Column("n_trials", sa.Integer), sa.Column("sr0", sa.Float),
    sa.Column("parent_passed", sa.Boolean), sa.Column("beats_parent_frac", sa.Float),
    sa.Column("holdout_status", sa.String(16), nullable=False),    # n/a | pending | passed | failed | blocked
    sa.Column("holdout_reason", sa.String(32), nullable=False),    # fixed code, never a number
    sa.Column("holdout_run_id", sa.String(64), sa.ForeignKey("runs.run_id"), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True)),
    sa.Column("updated_at", sa.DateTime(timezone=True)),
    sa.UniqueConstraint("strategy_hash", "timeframe"),
)

DECISIONS = {"promote", "kill"}
HOLDOUT_STATUSES = {"n/a", "pending", "passed", "failed", "blocked"}

APPROVABLE_STATUSES = {"paper", "retired"}   # "live" deliberately absent


def _now():
    return datetime.now(timezone.utc)


@lru_cache(maxsize=8)
def _engine(url: str) -> sa.Engine:
    if url.startswith("sqlite:///"):
        Path(url[len("sqlite:///"):]).parent.mkdir(parents=True, exist_ok=True)
    eng = sa.create_engine(url, future=True)
    md.create_all(eng)
    return eng


def engine(cfg) -> sa.Engine:
    url = os.path.expandvars(cfg.get("store", {}).get("url", "sqlite:///runs/lab.db"))
    return _engine(url)


def _f(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(v) or np.isinf(v)) else v


def _ts(x):
    if x is None:
        return None
    t = pd.Timestamp(x)
    return (t.tz_localize("UTC") if t.tzinfo is None else t).to_pydatetime()


def _fold_oos(daily_path: str | None, folds: list[dict]) -> dict[int, tuple[float, float]]:
    from perp_lab.engine.metrics import sharpe
    if not daily_path or not Path(daily_path).exists():
        return {}
    d = pd.read_parquet(daily_path)
    if "PORTFOLIO" not in d:
        return {}
    p = d["PORTFOLIO"].dropna()
    out = {}
    for f in folds:
        a, b = pd.Timestamp(f["test_start"]), pd.Timestamp(f["test_end"])
        a = a if a.tzinfo else a.tz_localize("UTC")
        b = b if b.tzinfo else b.tz_localize("UTC")
        w = p[(p.index >= a) & (p.index < b)]
        out[int(f["fold"])] = (float((1 + w).prod() - 1), sharpe(w))
    return out


def record_run(cfg, rec: dict, run_dir: str | Path | None = None) -> None:
    """Upsert one run (and its child rows). Idempotent."""
    m = rec.get("metrics") or {}
    rnd = rec.get("random_entry") or {}
    sens = rec.get("sensitivity") or {}
    art = rec.get("artifacts") or {}
    tfs = rec.get("timeframes") or []
    row = {
        "run_id": rec["run_id"], "strategy": rec["strategy"], "strategy_hash": rec.get("strategy_hash"),
        "hypothesis": rec.get("hypothesis"), "params": rec.get("params"), "universe": rec.get("universe"),
        "timeframes": tfs, "timeframe": tfs[0] if tfs else None,
        "date_start": rec.get("date_start"), "date_end": rec.get("date_end"), "oos_start": rec.get("oos_start"),
        "validation_mode": rec.get("validation_mode", "IS"),
        "verdict": rec.get("status"), "fail_reasons": rec.get("fail_reasons"),
        "sharpe": _f(m.get("sharpe")), "sortino": _f(m.get("sortino")), "max_dd": _f(m.get("max_dd")),
        "total_return": _f(m.get("total_return")), "cagr": _f(m.get("cagr")),
        "profit_factor": _f(m.get("profit_factor")), "win_rate": _f(m.get("win_rate")),
        "avg_r": _f(m.get("avg_r")), "exposure": _f(m.get("exposure")),
        "trades": int(m["trades"]) if m.get("trades") is not None else None,
        "liquidations": int(m["liquidations"]) if m.get("liquidations") is not None else None,
        "is_sharpe": _f((rec.get("is_metrics") or {}).get("sharpe")),
        "random_pctile": _f(rnd.get("strategy_pctile_vs_random")),
        "beats_random": rnd.get("beats_random"),
        "narrow_island": sens.get("narrow_island"),
        "metrics": m,
        "trade_log_path": art.get("oos_trades") or art.get("trades"),
        "run_dir": str(run_dir) if run_dir else rec.get("run_dir"),
        "parent_run_id": rec.get("parent_run_id"),
        "lineage_id": rec.get("lineage_id") or (None if rec.get("parent_run_id") else rec["run_id"]),
        "repo_commit": rec.get("repo_commit"),
        "created_at": _ts(rec.get("created_at")) or _now(),
        "notes": rec.get("notes"),
        "paper_started_at": _ts(rec.get("paper_started_at")),
    }
    if row["parent_run_id"] and not row["lineage_id"]:
        row["lineage_id"] = lineage_of(cfg, row["parent_run_id"])
    folds = rec.get("folds") or []
    fo = _fold_oos(art.get("oos_daily"), folds)
    eng = engine(cfg)
    with eng.begin() as c:
        for t in (run_folds, run_symbols, run_regimes):
            c.execute(t.delete().where(t.c.run_id == row["run_id"]))
        exists = c.execute(sa.select(runs.c.run_id).where(runs.c.run_id == row["run_id"])).first()
        if exists:
            keep = {"paper_regime_changes", "paper_regime_detail", "paper_eligible", "paper_checked_at"}
            c.execute(runs.update().where(runs.c.run_id == row["run_id"]).values(
                **{k: v for k, v in row.items() if k not in keep}))
        else:
            c.execute(runs.insert().values(**row))
        if folds:
            c.execute(run_folds.insert(), [{
                "run_id": row["run_id"], "fold": int(f["fold"]),
                "train_start": str(f["train_start"])[:10], "train_end": str(f["train_end"])[:10],
                "test_start": str(f["test_start"])[:10], "test_end": str(f["test_end"])[:10],
                "combo": f.get("combo"), "params": f.get("params"), "is_score": _f(f.get("is_score")),
                "oos_return": fo.get(int(f["fold"]), (None, None))[0],
                "oos_sharpe": fo.get(int(f["fold"]), (None, None))[1],
            } for f in folds])
        sy = rec.get("by_symbol") or {}
        if sy:
            c.execute(run_symbols.insert(), [{
                "run_id": row["run_id"], "symbol": s, "sharpe": _f(x.get("sharpe")), "max_dd": _f(x.get("max_dd")),
                "total_return": _f(x.get("total_return")), "trades": int(x.get("trades") or 0),
                "profit_factor": _f(x.get("profit_factor")), "win_rate": _f(x.get("win_rate")),
            } for s, x in sy.items()])
        rg = rec.get("regime_split") or []
        if rg:
            c.execute(run_regimes.insert(), [{
                "run_id": row["run_id"], "regime": x["regime"], "trades": int(x.get("trades") or 0),
                "win_rate": _f(x.get("win_rate")), "avg_r": _f(x.get("avg_r")), "sum_r": _f(x.get("sum_r")),
                "profit_factor": _f(x.get("profit_factor")), "sum_pnl_pct": _f(x.get("sum_pnl_pct")),
            } for x in rg])


def ingest_runs(cfg) -> list[str]:
    """Record every run dir under runs_dir that isn't in the store yet."""
    eng = engine(cfg)
    with eng.connect() as c:
        have = {r[0] for r in c.execute(sa.select(runs.c.run_id))}
    added = []
    for sj in sorted(cfg.runs_dir.glob("*/summary.json")):
        rec = json.loads(sj.read_text())
        if rec.get("run_id") in have:
            continue
        record_run(cfg, rec, sj.parent)
        added.append(rec["run_id"])
    return added


def lineage_of(cfg, run_id: str) -> str | None:
    with engine(cfg).connect() as c:
        r = c.execute(sa.select(runs.c.lineage_id).where(runs.c.run_id == run_id)).first()
    return r[0] if r else None


def add_edge(cfg, parent_run_id: str, child_run_id: str, decision: str, reason: str,
             decided_by: str = "human") -> None:
    if decision not in {"promote", "kill", "manual"}:
        raise ValueError("decision must be promote | kill | manual")
    if not reason.strip():
        raise ValueError("every lineage edge needs a reason")
    eng = engine(cfg)
    with eng.begin() as c:
        ids = {r[0] for r in c.execute(sa.select(runs.c.run_id).where(
            runs.c.run_id.in_([parent_run_id, child_run_id])))}
        if ids != {parent_run_id, child_run_id}:
            raise ValueError(f"unknown run(s): {({parent_run_id, child_run_id} - ids)}")
        lin = c.execute(sa.select(runs.c.lineage_id).where(runs.c.run_id == parent_run_id)).scalar()
        c.execute(runs.update().where(runs.c.run_id == child_run_id).values(
            parent_run_id=parent_run_id, lineage_id=lin))
        c.execute(lineage_edges.delete().where(sa.and_(lineage_edges.c.parent_run_id == parent_run_id,
                                                       lineage_edges.c.child_run_id == child_run_id)))
        c.execute(lineage_edges.insert().values(parent_run_id=parent_run_id, child_run_id=child_run_id,
                                                decision=decision, reason=reason, decided_by=decided_by,
                                                created_at=_now()))


# ---------------------------------------------------------------- judgements (Phase 3)

def _row(r) -> dict | None:
    return dict(r._mapping) if r is not None else None


def record_judgement(cfg, row: dict) -> int:
    """Insert one judgement. A second judgement of the same (strategy_hash, timeframe), e.g. two
    judges racing, returns the first one's id instead of failing."""
    if row["decision"] not in DECISIONS:
        raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
    if row["holdout_status"] not in HOLDOUT_STATUSES:
        raise ValueError(f"holdout_status must be one of {sorted(HOLDOUT_STATUSES)}")
    now = _now()
    try:
        with engine(cfg).begin() as c:
            return int(c.execute(judgements.insert().values(**row, created_at=now, updated_at=now))
                       .inserted_primary_key[0])
    except sa.exc.IntegrityError:
        existing = get_judgement(cfg, row["strategy_hash"], row["timeframe"])
        if existing is None:
            raise
        return existing["id"]


def _judgement_where(cfg, where) -> dict | None:
    with engine(cfg).connect() as c:
        return _row(c.execute(sa.select(judgements).where(where)).first())


def get_judgement(cfg, strategy_hash: str, timeframe: str) -> dict | None:
    return _judgement_where(cfg, sa.and_(judgements.c.strategy_hash == strategy_hash,
                                         judgements.c.timeframe == timeframe))


def get_judgement_by_id(cfg, jid: int) -> dict | None:
    return _judgement_where(cfg, judgements.c.id == jid)


def judgement_for_run(cfg, run_id: str) -> dict | None:
    return _judgement_where(cfg, judgements.c.child_run_id == run_id)


def update_holdout(cfg, jid: int, status: str, reason: str, holdout_run_id: str | None = None) -> None:
    if status not in HOLDOUT_STATUSES:
        raise ValueError(f"holdout_status must be one of {sorted(HOLDOUT_STATUSES)}")
    vals = {"holdout_status": status, "holdout_reason": reason, "updated_at": _now()}
    if holdout_run_id is not None:
        vals["holdout_run_id"] = holdout_run_id
    with engine(cfg).begin() as c:
        c.execute(judgements.update().where(judgements.c.id == jid).values(**vals))


def pending_judgements(cfg) -> list[dict]:
    with engine(cfg).connect() as c:
        rows = c.execute(sa.select(judgements).where(judgements.c.holdout_status == "pending")
                         .order_by(judgements.c.created_at, judgements.c.id))
        return [dict(r._mapping) for r in rows]


def run_row(cfg, run_id: str) -> dict | None:
    with engine(cfg).connect() as c:
        return _row(c.execute(sa.select(runs).where(runs.c.run_id == run_id)).first())


def latest_wf_run(cfg, strategy: str, timeframe: str, strategy_hash: str, universe: list[str]) -> dict | None:
    """Most recent walk-forward run of exactly this plugin version, timeframe and symbol universe."""
    with engine(cfg).connect() as c:
        rows = c.execute(sa.select(runs).where(sa.and_(
            runs.c.strategy == strategy, runs.c.timeframe == timeframe,
            runs.c.strategy_hash == strategy_hash, runs.c.validation_mode == "walkforward"))
            .order_by(runs.c.created_at.desc())).all()
    for r in rows:
        if list(r.universe or []) == list(universe):
            return dict(r._mapping)
    return None


def family_of(cfg, strategy: str) -> str | None:
    """A plugin name's family: that of its first judgement, else of its first walk-forward run.
    Looked up by name so editing or re-running a root never starts a new family."""
    with engine(cfg).connect() as c:
        j = c.execute(sa.select(judgements.c.lineage_id).where(sa.and_(
            judgements.c.strategy == strategy, judgements.c.lineage_id.isnot(None)))
            .order_by(judgements.c.created_at, judgements.c.id).limit(1)).first()
        if j:
            return j[0]
        r = c.execute(sa.select(runs.c.lineage_id).where(sa.and_(
            runs.c.strategy == strategy, runs.c.validation_mode == "walkforward",
            runs.c.lineage_id.isnot(None))).order_by(runs.c.created_at).limit(1)).first()
    return r[0] if r else None


def trial_stats(cfg) -> tuple[int, float]:
    """N = distinct (strategy_hash, timeframe) among walk-forward runs, store-wide; V = sample
    variance of the daily Sharpes of the latest run of each (0.0 when N < 2)."""
    with engine(cfg).connect() as c:
        df = pd.read_sql(sa.select(runs.c.strategy_hash, runs.c.timeframe, runs.c.sharpe, runs.c.created_at)
                         .where(sa.and_(runs.c.validation_mode == "walkforward", runs.c.sharpe.isnot(None))), c)
    df = df.dropna(subset=["strategy_hash", "timeframe"])
    if df.empty:
        return 0, 0.0
    latest = df.sort_values("created_at").groupby(["strategy_hash", "timeframe"]).tail(1)
    n = len(latest)
    v = float((latest.sharpe / np.sqrt(365)).var(ddof=1)) if n >= 2 else 0.0
    return n, v


# ---------------------------------------------------------------- status

def latest_runs(cfg, mode: str = "walkforward") -> pd.DataFrame:
    """Most recent run per (strategy, timeframe) for a validation mode."""
    with engine(cfg).connect() as c:
        df = pd.read_sql(sa.select(runs).where(runs.c.validation_mode == mode), c)
    if df.empty:
        return df
    return (df.sort_values("created_at").groupby(["strategy", "timeframe"], as_index=False).tail(1)
            .reset_index(drop=True))


def current_status(cfg, strategy: str, timeframe: str, verdict: str | None) -> str:
    with engine(cfg).connect() as c:
        r = c.execute(sa.select(status_log.c.status).where(sa.and_(
            status_log.c.strategy == strategy, status_log.c.timeframe == timeframe))
            .order_by(status_log.c.id.desc()).limit(1)).first()
    return r[0] if r else (verdict or "noise")


def approve_paper(cfg, run_id: str, approver: str, note: str = "") -> None:
    """Human approval gate. Only a walk-forward 'candidate' can enter paper."""
    with engine(cfg).begin() as c:
        r = c.execute(sa.select(runs.c.strategy, runs.c.timeframe, runs.c.verdict, runs.c.validation_mode)
                      .where(runs.c.run_id == run_id)).first()
        if r is None:
            raise ValueError(f"unknown run {run_id}")
        if r.validation_mode != "walkforward" or r.verdict != "candidate":
            raise PermissionError(f"run {run_id} is '{r.verdict}' ({r.validation_mode}); only walk-forward "
                                  "candidates can be approved for paper")
        c.execute(status_log.insert().values(strategy=r.strategy, timeframe=r.timeframe, run_id=run_id,
                                             status="paper", set_by=approver, note=note, created_at=_now()))


def set_status(cfg, strategy: str, timeframe: str, status: str, set_by: str, note: str = "") -> None:
    if status not in APPROVABLE_STATUSES - {"paper"}:
        raise PermissionError(f"status '{status}' can't be set here (paper → approve_paper; live → never)")
    with engine(cfg).begin() as c:
        c.execute(status_log.insert().values(strategy=strategy, timeframe=timeframe, status=status,
                                             set_by=set_by, note=note, created_at=_now()))


def update_paper_tracking(cfg, run_id: str, changes: int, detail: dict, now=None) -> dict:
    now = now or _now()
    with engine(cfg).begin() as c:
        started = c.execute(sa.select(runs.c.paper_started_at).where(runs.c.run_id == run_id)).scalar()
        if started is None:
            raise ValueError(f"{run_id} is not a paper run")
        started = _ts(started)
        days = (now - started).total_seconds() / 86400
        eligible = bool(days >= cfg["paper"]["min_paper_days"] and changes >= 1)
        c.execute(runs.update().where(runs.c.run_id == run_id).values(
            paper_regime_changes=int(changes), paper_regime_detail=detail,
            paper_eligible=eligible, paper_checked_at=now))
    return {"days": days, "changes": changes, "eligible": eligible}


def read_table(cfg, table: sa.Table, where=None) -> pd.DataFrame:
    q = sa.select(table)
    if where is not None:
        q = q.where(where)
    with engine(cfg).connect() as c:
        return pd.read_sql(q, c)
