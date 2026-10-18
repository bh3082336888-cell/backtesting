"""lab CLI: backfill | tail | backtest | walkforward | paper | strategies"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import typer

from perp_lab.config import load_config

app = typer.Typer(add_completion=False, no_args_is_help=True)
log = logging.getLogger("lab")

CfgOpt = typer.Option(None, "--config", "-c", help="path to lab.yaml")


@contextmanager
def _sandbox_required():
    """Plugin code runs only in the sandbox: where this machine can't provide one, say why and exit
    1, without a traceback."""
    from perp_lab.sandbox.errors import SandboxUnavailable
    try:
        yield
    except SandboxUnavailable as e:
        typer.secho(f"sandbox unavailable: {e}", fg="red")
        raise typer.Exit(1)


def _syms(cfg, symbols: Optional[str]) -> list[str]:
    if not symbols:
        return cfg.symbols
    q = cfg.get("quote", "USDT")
    return [s if s.endswith(q) else f"{s}{q}" for s in symbols.split(",")]


@app.command()
def backfill(symbols: Optional[str] = typer.Option(None, help="comma list, e.g. BTC,ETH"),
             tfs: Optional[str] = typer.Option(None, help="comma list of timeframes"),
             datasets: Optional[str] = typer.Option(None, help="klines,fundingRate,metrics"),
             start: Optional[str] = None, workers: Optional[int] = None, config: Optional[str] = CfgOpt):
    """Backfill from data.binance.vision → parquet (idempotent, resumable)."""
    from perp_lab.data.vision import backfill as bf
    cfg = load_config(config)
    stats = bf(cfg, _syms(cfg, symbols), tfs.split(",") if tfs else cfg["timeframes"],
               datasets.split(",") if datasets else cfg["backfill"]["datasets"],
               start=pd.Timestamp(start, tz="UTC") if start else None, workers=workers)
    typer.echo(json.dumps(stats))
    if stats["error"]:
        raise typer.Exit(1)


@app.command()
def tail(symbols: Optional[str] = None, tfs: Optional[str] = None, config: Optional[str] = CfgOpt):
    """Stitch parquet → now via ccxt fetchOHLCV."""
    from perp_lab.data.live import make_exchange, tail as do_tail
    cfg = load_config(config)
    ex = make_exchange()
    for s in _syms(cfg, symbols):
        for tf in (tfs.split(",") if tfs else cfg["timeframes"]):
            n = do_tail(cfg, s, tf, ex)
            typer.echo(f"{s} {tf}: +{n} candles")


@app.command()
def strategies():
    """List strategy plugins and their hypotheses."""
    from perp_lab.plugins import PluginSource
    from perp_lab.strategy import list_plugins
    for n in list_plugins():
        s = PluginSource.read(n)
        typer.echo(f"{n} [{s.hash[:10]}] tfs={s.meta.timeframes}\n  {s.meta.hypothesis}")


@app.command()
def backtest(strategy: str, tf: str = "1h", symbols: Optional[str] = None,
             params: str = typer.Option("{}", help='JSON params, e.g. \'{"n": 40}\''),
             config: Optional[str] = CfgOpt):
    """Single fixed-param backtest over the research span. This is IN-SAMPLE by definition."""
    from perp_lab.data.store import load_bars, research_symbols
    from perp_lab.engine.backtest import EngineParams
    from perp_lab.engine.metrics import daily_returns, portfolio_returns, summarize
    from perp_lab.plugins import PluginSource
    from perp_lab.runs import new_run_id, repo_commit, write_run
    from perp_lab.sandbox import client

    cfg = load_config(config)
    src = PluginSource.read(strategy)
    p = json.loads(params)
    per, trades = {}, []
    for s in research_symbols(cfg, _syms(cfg, symbols), tf):
        bars = load_bars(cfg, s, tf)
        with _sandbox_required(), client.Batch(cfg, src, s, tf, bars) as batch:
            r = batch.backtest({**src.meta.params, **p}, EngineParams.from_config(cfg, s))
        per[s] = daily_returns(r.equity, r.time)
        trades += r.trades
    tr = pd.DataFrame(trades)
    m = summarize(portfolio_returns(per), tr)
    rid = new_run_id()
    rec = {"run_id": rid, "strategy": strategy, "strategy_hash": src.hash,
           "hypothesis": src.meta.hypothesis, "repo_commit": repo_commit(), "params": {**src.meta.params, **p},
           "universe": list(per), "timeframes": [tf], "validation_mode": "IS",
           "date_start": str(cfg.history_start.date()), "date_end": str(cfg.holdout_start.date()),
           "metrics": m, "parent_run_id": None, "created_at": datetime.now(timezone.utc).isoformat(),
           "notes": "fixed-param full-span backtest: in-sample, not evidence"}
    d = write_run(cfg, rec, {"trades": tr.assign(source="backtest", run_id=rid) if len(tr) else tr,
                             "daily": pd.DataFrame(per)})
    typer.secho("IN-SAMPLE (not evidence of anything):", dim=True)
    typer.secho(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}), dim=True)
    typer.echo(f"run: {d}")


@app.command()
def walkforward(strategy: str, tf: str = "1h", symbols: Optional[str] = None,
                seeds: Optional[int] = typer.Option(None, help="random-entry seeds (default: config)"),
                workers: Optional[int] = None, notes: str = "",
                parent_run: Optional[str] = typer.Option(None, help="parent run_id (lineage)"),
                config: Optional[str] = CfgOpt):
    """Walk-forward + sensitivity + random-entry + MC + regime split. OOS headline only."""
    from perp_lab.plugins import UndeclaredTimeframe
    from perp_lab.validate.pipeline import validate_strategy
    cfg = load_config(config)
    try:
        with _sandbox_required():
            rec, report = validate_strategy(cfg, strategy, tf, _syms(cfg, symbols), seeds, workers, notes,
                                            parent_run_id=parent_run)
    except UndeclaredTimeframe as e:            # a trial at a timeframe it doesn't declare: nothing ran
        typer.secho(f"refused: {e}", fg="red")
        raise typer.Exit(2)
    o, i = rec["metrics"], rec["is_metrics"]
    typer.secho(f"{strategy} @ {tf}  →  {rec['status'].upper()}", bold=True,
                fg="red" if rec["status"] == "noise" else "green")
    typer.echo(f"OOS  Sharpe {o['sharpe']:.2f}  MaxDD {o['max_dd']:.1%}  Return {o['total_return']:.1%}  "
               f"PF {o['profit_factor']:.2f}  trades {o['trades']}")
    typer.secho(f"IS   Sharpe {i['sharpe']:.2f}  MaxDD {i['max_dd']:.1%}  (reference only)", dim=True)
    for f in rec["fail_reasons"]:
        typer.echo(f"  ✗ {f}")
    typer.echo(f"run: {rec['run_dir']}/report.md")


@app.command()
def paper(from_run: str = typer.Argument(..., help="walk-forward run_id approved for paper in the dashboard"),
          symbols: Optional[str] = None, max_bars: Optional[int] = None, config: Optional[str] = CfgOpt):
    """Forward-test an APPROVED walk-forward run on live closed candles (modal params)."""
    import sqlalchemy as sa

    from perp_lab.paper.runner import PaperRunner
    from perp_lab.runs import new_run_id, repo_commit
    from perp_lab.store.db import current_status, engine, record_run, runs
    from perp_lab.plugins import PluginSource
    cfg = load_config(config)
    with engine(cfg).connect() as c:
        r = c.execute(sa.select(runs).where(runs.c.run_id == from_run)).mappings().first()
    if r is None:
        raise typer.BadParameter(f"unknown run {from_run}")
    if current_status(cfg, r["strategy"], r["timeframe"], r["verdict"]) != "paper":
        typer.secho(f"{r['strategy']} @ {r['timeframe']} is not approved for paper. "
                    "Approve it in the dashboard (only walk-forward candidates qualify).", fg="red")
        raise typer.Exit(2)
    params = (r["metrics"] or {}).get("modal_params") or json.loads(
        (cfg.runs_dir / from_run / "summary.json").read_text())["sensitivity"]["base_params"]
    src = PluginSource.read(r["strategy"])
    rid = new_run_id()
    out = cfg.runs_dir / rid
    syms = _syms(cfg, symbols) if symbols else r["universe"]
    started = datetime.now(timezone.utc)
    record_run(cfg, {"run_id": rid, "strategy": r["strategy"], "strategy_hash": src.hash,
                     "hypothesis": src.meta.hypothesis, "params": params, "universe": syms,
                     "timeframes": [r["timeframe"]], "validation_mode": "paper", "repo_commit": repo_commit(),
                     "parent_run_id": from_run, "lineage_id": r["lineage_id"], "created_at": started.isoformat(),
                     "paper_started_at": started.isoformat(), "metrics": {},
                     "artifacts": {"trades": str(out / "trades.parquet")}}, out)
    typer.echo(f"paper run {rid} → {out}")
    pr = PaperRunner(cfg, src, params, r["timeframe"], syms, out)
    with _sandbox_required():
        asyncio.run(pr.run(max_bars=max_bars))


@app.command("paper-refresh")
def paper_refresh(run_id: str, config: Optional[str] = CfgOpt):
    """Recompute regime changes + paper_eligible for a paper run (fetches live 4h bars)."""
    from perp_lab.paper.tracking import refresh_paper_run
    typer.echo(json.dumps(refresh_paper_run(load_config(config), run_id), default=str))


@app.command()
def ingest(config: Optional[str] = CfgOpt):
    """Record any run directory not yet in the results store."""
    from perp_lab.store.db import ingest_runs
    added = ingest_runs(load_config(config))
    typer.echo(f"ingested {len(added)} run(s): {added}")


@app.command()
def link(parent: str, child: str, decision: str = typer.Option(..., help="promote | kill | manual"),
         reason: str = typer.Option(..., help="why — shown on the lineage edge"),
         config: Optional[str] = CfgOpt):
    """Record a hand-made lineage edge (Phase 3's judge writes these itself)."""
    from perp_lab.store.db import add_edge
    try:
        add_edge(load_config(config), parent, child, decision, reason, decided_by="human")
    except PermissionError as e:
        typer.secho(f"refused: {e}", fg="red")
        raise typer.Exit(1)
    typer.echo(f"{parent} → {child} [{decision}] {reason}")


@app.command("holdout-init")
def holdout_init(config: Optional[str] = CfgOpt):
    """Create the empty holdout audit log (once). Exams refuse to run without it."""
    from perp_lab.judge.budget import init_log
    try:
        p = init_log(load_config(config))
    except FileExistsError as e:
        typer.secho(str(e), fg="red")
        raise typer.Exit(1)
    typer.echo(f"created {p}")


@app.command()
def judge(strategy: str, tf: str = "4h", symbols: Optional[str] = None, workers: Optional[int] = None,
          config: Optional[str] = CfgOpt):
    """Promote or kill a plugin against its PARENT; promoted plugins queue for the holdout exam,
    which only `lab exam` runs."""
    from perp_lab.judge.judge import JudgeRefused
    from perp_lab.judge.judge import judge as run_judge
    cfg = load_config(config)
    try:
        with _sandbox_required():
            j = run_judge(cfg, strategy, tf, _syms(cfg, symbols) if symbols else None, workers)
    except JudgeRefused as e:
        typer.secho(f"refused: {e}", fg="red")
        raise typer.Exit(2)
    again = " (already judged)" if j.get("existing") else ""
    typer.secho(f"{j['strategy']} @ {j['timeframe']}  →  {j['decision'].upper()}{again}", bold=True,
                fg="green" if j["decision"] == "promote" else "red")
    for r in j["reasons"]:
        typer.echo(f"  ✗ {r}")
    dsr = "—" if j["dsr"] is None else f"{j['dsr']:.2f}"
    typer.echo(f"Deflated Sharpe {dsr} (N={j['n_trials']})   holdout: {j['holdout_status']} ({j['holdout_reason']})")
    typer.echo(f"verdict: {j['verdict_path']}")


@app.command()
def withdraw(strategy: Optional[str] = typer.Argument(None, help="the plugin, by the name it was judged under"),
             tf: Optional[str] = typer.Option(None, "--tf", help="the timeframe it was judged at"),
             reason: str = typer.Option(..., help="why: recorded in withdrawals.jsonl next to the holdout log"),
             judgement: Optional[int] = typer.Option(None, "--id", help="the judgement id: picks it alone (several "
                                                     "versions pending, or a plugin file renamed since)"),
             config: Optional[str] = CfgOpt):
    """Withdraw a pending promotion before its holdout read: it is never examined, and its family
    keeps its one read. A human's decision on the walk-forward record; the proposer never runs it.
    Name it by PLUGIN and --tf, or by --id (a name or --tf given with it must match)."""
    from perp_lab.judge.budget import WithdrawRefused
    from perp_lab.judge.judge import withdraw as run_withdraw
    from perp_lab.store.db import get_judgement_by_id, pending_judgements
    cfg = load_config(config)

    def refuse(msg: str):
        typer.secho(f"refused: {msg}", fg="red")
        raise typer.Exit(2)
    if judgement is not None:
        j = get_judgement_by_id(cfg, judgement)
        if j is None:
            refuse(f"no judgement {judgement}")
        if strategy not in (None, j["strategy"]) or tf not in (None, j["timeframe"]):
            refuse(f"judgement {judgement} is {j['strategy']} @ {j['timeframe']}")
    elif strategy is None or tf is None:
        refuse("name the plugin and --tf, or pass --id")
    else:
        mine = [j for j in pending_judgements(cfg) if j["strategy"] == strategy and j["timeframe"] == tf]
        if not mine:
            refuse(f"no pending promotion of {strategy} @ {tf}")
        if len(mine) > 1:
            ids = ", ".join(f"{j['id']} ({j['strategy_hash'][:10]})" for j in mine)
            refuse(f"several versions of {strategy} @ {tf} are pending: {ids}; pass --id")
        j = mine[0]
    try:
        w = run_withdraw(cfg, j["id"], reason)
    except WithdrawRefused as e:
        refuse(str(e))
    typer.echo(f"{w['strategy']} @ {w['timeframe']}: withdrawn (judgement {w['id']}); family {w['lineage_id']} keeps "
               "its holdout read")
    typer.echo(f"verdict: {w['verdict_path']}")


@app.command()
def exam(config: Optional[str] = CfgOpt):
    """Attempt the holdout exam for every pending promotion (budget rules apply). The only command
    that runs exams; the proposer never runs it."""
    from perp_lab.judge.judge import exam_pending
    done = exam_pending(load_config(config))
    if not done:
        typer.echo("no pending exams")
    for j in done:
        if j.get("in_progress"):
            typer.echo(f"{j['strategy']} @ {j['timeframe']}: in progress elsewhere (another `lab exam` is running)")
            continue
        err = f" ERROR: {j['error']}" if "error" in j else ""
        typer.echo(f"{j['strategy']} @ {j['timeframe']}: {j['holdout_status']} ({j['holdout_reason']}){err}")
        for line in j.get("detail", []):          # e.g. the months of an incomplete_data wait
            typer.echo(f"  {line}")
    if any("error" in j for j in done):
        raise typer.Exit(1)


@app.command()
def dashboard(port: int = 8501,
              address: str = typer.Option("localhost", help="bind address; 0.0.0.0 exposes the dashboard "
                                                             "(and its paper-approval button) to the network"),
              config: Optional[str] = CfgOpt):
    """Launch the Streamlit dashboard (localhost only unless --address says otherwise)."""
    import subprocess
    import sys
    args = [sys.executable, "-m", "streamlit", "run", "dashboard/app.py", "--server.port", str(port),
            "--server.address", address]
    if config:
        args += ["--", "--config", config]
    raise typer.Exit(subprocess.call(args))


if __name__ == "__main__":
    app()
