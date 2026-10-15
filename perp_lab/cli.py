"""lab CLI: backfill | tail | backtest | walkforward | paper | strategies"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import typer

from perp_lab.config import load_config

app = typer.Typer(add_completion=False, no_args_is_help=True)
log = logging.getLogger("lab")

CfgOpt = typer.Option(None, "--config", "-c", help="path to lab.yaml")


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
    from perp_lab.strategy import list_plugins, load_strategy_class, plugin_hash
    for n in list_plugins():
        c = load_strategy_class(n)
        typer.echo(f"{n} [{plugin_hash(n)[:10]}] tfs={c.timeframes}\n  {c.hypothesis()}")


@app.command()
def backtest(strategy: str, tf: str = "1h", symbols: Optional[str] = None,
             params: str = typer.Option("{}", help='JSON params, e.g. \'{"n": 40}\''),
             config: Optional[str] = CfgOpt):
    """Single fixed-param backtest over the research span. This is IN-SAMPLE by definition."""
    from perp_lab.data.store import load_bars
    from perp_lab.engine.backtest import EngineParams, run_backtest
    from perp_lab.engine.metrics import daily_returns, portfolio_returns, summarize
    from perp_lab.runs import new_run_id, repo_commit, write_run
    from perp_lab.strategy import load_strategy_class, plugin_hash

    cfg = load_config(config)
    cls = load_strategy_class(strategy)
    p = json.loads(params)
    per, trades = {}, []
    for s in _syms(cfg, symbols):
        bars = load_bars(cfg, s, tf)
        st = cls(**p)
        r = run_backtest(st, bars, st.prepare(bars), symbol=s, tf=tf, ep=EngineParams.from_config(cfg, s))
        per[s] = daily_returns(r.equity, r.time)
        trades += r.trades
    tr = pd.DataFrame(trades)
    m = summarize(portfolio_returns(per), tr)
    rid = new_run_id()
    rec = {"run_id": rid, "strategy": strategy, "strategy_hash": plugin_hash(strategy),
           "hypothesis": cls.hypothesis(), "repo_commit": repo_commit(), "params": {**cls.params, **p},
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
                workers: Optional[int] = None, notes: str = "", config: Optional[str] = CfgOpt):
    """Walk-forward + sensitivity + random-entry + MC + regime split. OOS headline only."""
    from perp_lab.validate.pipeline import validate_strategy
    cfg = load_config(config)
    rec, report = validate_strategy(cfg, strategy, tf, _syms(cfg, symbols), seeds, workers, notes)
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
def paper(strategy: str, tf: str = "1h", symbols: Optional[str] = None,
          params: str = typer.Option("{}", help="JSON params (use the walk-forward modal params)"),
          max_bars: Optional[int] = None, config: Optional[str] = CfgOpt):
    """Forward-test on live closed candles; writes the same trade-log schema."""
    from perp_lab.paper.runner import PaperRunner
    from perp_lab.strategy import load_strategy_class
    cfg = load_config(config)
    cls = load_strategy_class(strategy)
    out = cfg.runs_dir / "paper" / f"{strategy}_{tf}"
    r = PaperRunner(cfg, cls, json.loads(params), tf, _syms(cfg, symbols), out)
    asyncio.run(r.run(max_bars=max_bars))


if __name__ == "__main__":
    app()
