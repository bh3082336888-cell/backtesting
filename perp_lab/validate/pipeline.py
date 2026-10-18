"""Full validation of one strategy on one timeframe, recorded as a run."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd

from perp_lab.plugins import PluginSource, require_timeframe
from perp_lab.runs import new_run_id, repo_commit, verdict, write_run
from perp_lab.validate.benchmarks import buy_and_hold, random_entry, regime_split
from perp_lab.validate.montecarlo import mc_drawdown
from perp_lab.validate.sensitivity import sensitivity
from perp_lab.validate.walkforward import walkforward

log = logging.getLogger(__name__)


def _f(x, pct=False):
    if x is None:
        return "—"
    if isinstance(x, str):
        return x
    return f"{x:.1%}" if pct else f"{x:.2f}"


def render_report(rec: dict, regime: pd.DataFrame, monthly: pd.DataFrame) -> str:
    o, i, r, s, m, b = (rec["metrics"], rec["is_metrics"], rec["random_entry"], rec["sensitivity"],
                        rec["monte_carlo"], rec["buy_and_hold"])
    L = [f"# {rec['strategy']} @ {rec['timeframes'][0]} — walk-forward OOS",
         "", f"**Hypothesis:** {rec['hypothesis']}", "",
         f"run `{rec['run_id']}` · plugin `{rec['strategy_hash'][:10]}` · {rec['date_start']} → {rec['date_end']} "
         f"(holdout from {rec['holdout_start']} untouched) · {len(rec['folds'])} folds "
         f"({rec['wf_train_months']}m train / {rec['wf_test_months']}m test)", "",
         f"## Verdict: **{rec['status'].upper()}**", ""]
    L += [f"- {x}" for x in rec["fail_reasons"]] or ["- all checks passed (now go look for the bug)"]
    L += ["", "## OOS (headline)", "",
          "| Sharpe | Sortino | MaxDD | Return | CAGR | PF | Win% | avg R | Trades | Exposure | Liqs |",
          "|---|---|---|---|---|---|---|---|---|---|---|",
          f"| {_f(o['sharpe'])} | {_f(o['sortino'])} | {_f(o['max_dd'], 1)} | {_f(o['total_return'], 1)} | "
          f"{_f(o['cagr'], 1)} | {_f(o['profit_factor'])} | {_f(o['win_rate'], 1)} | {_f(o['avg_r'])} | "
          f"{o['trades']} | {_f(o.get('exposure'), 1)} | {o['liquidations']} |", "",
          f"<span style='color:grey'>IS (train windows of selected params, for reference only): "
          f"Sharpe {_f(i['sharpe'])}, MaxDD {_f(i['max_dd'], 1)}, Return {_f(i['total_return'], 1)}</span>", "",
          "## Benchmarks (OOS span)", "",
          f"- Buy & hold (1x long, funding paid, equal-weight): Sharpe {_f(b['sharpe'])}, "
          f"MaxDD {_f(b['max_dd'], 1)}, Return {_f(b['total_return'], 1)}",
          f"- Random entry, same exits, matched frequency ({r.get('seeds', 0)} seeds): "
          f"mean Sharpe {_f(r.get('random_sharpe_mean'))}, p{r.get('percentile', '')} "
          f"{_f(r.get('random_sharpe_p' + str(r.get('percentile', ''))))}; strategy sits at the "
          f"{_f(r.get('strategy_pctile_vs_random'), 1)} percentile → beats random: **{r.get('beats_random')}**", "",
          "## Robustness", "",
          f"- Sensitivity (fixed modal params {s['base_params']}): base Sharpe {_f(s['base_sharpe'])}, "
          f"narrow island: **{s['narrow_island']}** {s.get('note', '')}",
          f"- Monte Carlo trade order: realised DD {_f(m.get('actual_dd'), 1)}, median {_f(m.get('p50_dd'), 1)}, "
          f"5th pct {_f(m.get('p5_dd'), 1)}, 1st pct {_f(m.get('p1_dd'), 1)}", "",
          "## Per symbol (OOS)", "", "| Symbol | Sharpe | MaxDD | Return | Trades | PF |", "|---|---|---|---|---|---|"]
    for sym, x in rec["by_symbol"].items():
        L.append(f"| {sym} | {_f(x['sharpe'])} | {_f(x['max_dd'], 1)} | {_f(x['total_return'], 1)} | "
                 f"{x['trades']} | {_f(x['profit_factor'])} |")
    L += ["", "## Regime split (OOS trades, regime at signal)", ""]
    L.append(regime.round(3).to_markdown() if len(regime) else "_no trades_")
    L += ["", "## Monthly OOS returns (portfolio)", ""]
    L.append((monthly * 100).round(1).to_markdown() if len(monthly) else "_none_")
    L += ["", "## Param selection per fold", "", "| Fold | Test window | Params | IS score |", "|---|---|---|---|"]
    for f in rec["folds"]:
        L.append(f"| {f['fold']} | {str(f['test_start'])[:10]} → {str(f['test_end'])[:10]} | {f['combo']} | "
                 f"<span style='color:grey'>{_f(f['is_score'])}</span> |")
    return "\n".join(L) + "\n"


def validate_strategy(cfg, name: str, tf: str, symbols=None, seeds=None, workers=None,
                      notes: str = "", parent_run_id: str | None = None,
                      lineage_id: str | None = None, src: PluginSource | None = None) -> tuple[dict, str]:
    """Walk-forward, sensitivity and benchmarks of one plugin version, recorded as a run. The bytes
    are read once (or given as `src`): every stage runs them, and the run records their hash."""
    from perp_lab.engine.metrics import monthly_table

    src = PluginSource.read(name) if src is None else src
    if src.name != name:
        raise ValueError(f"source of {src.name} given to validate {name}")
    require_timeframe(src, tf)                  # a trial at a timeframe it doesn't declare: nothing runs
    rid = new_run_id()
    wf = walkforward(cfg, src, tf, symbols, workers, cfg.runs_dir / rid / "sandbox_stderr")
    log.info("walkforward done; OOS Sharpe %.2f", wf.oos_summary["sharpe"])
    sens = sensitivity(cfg, wf, workers)
    rnd = random_entry(cfg, wf, seeds, workers)
    mc = mc_drawdown(wf.oos_trades, len(wf.symbols), cfg["validation"]["mc_iterations"])
    oa, ob = wf.raw["oos_span"]
    bh = buy_and_hold(cfg, wf.symbols, tf, oa, ob)
    reg = regime_split(wf.oos_trades)
    monthly = monthly_table(wf.oos_portfolio)
    status, fails = verdict(wf.oos_summary, rnd, sens)

    rec = {
        "run_id": rid, "strategy": name, "strategy_hash": src.hash,
        "hypothesis": src.meta.hypothesis, "repo_commit": repo_commit(),
        "params": {"grid": src.meta.grid, "defaults": src.meta.params},
        "universe": wf.symbols, "timeframes": [tf],
        "date_start": str(wf.folds[0]["train_start"].date()), "date_end": str(ob.date()),
        "oos_start": str(oa.date()), "holdout_start": str(cfg.holdout_start.date()),
        "validation_mode": "walkforward",
        "wf_train_months": cfg["walkforward"]["train_months"], "wf_test_months": cfg["walkforward"]["test_months"],
        "metrics": wf.oos_summary, "is_metrics": wf.is_summary, "by_symbol": wf.oos_by_symbol,
        "sensitivity": sens, "random_entry": {k: v for k, v in rnd.items() if k != "sharpes"},
        "monte_carlo": mc, "buy_and_hold": bh["portfolio"],
        "regime_split": reg.reset_index().to_dict("records") if len(reg) else [],
        "folds": [{k: v for k, v in f.items()} for f in wf.selected],
        "status": status, "fail_reasons": fails,
        "costs": cfg["costs"], "account": cfg["account"],
        "parent_run_id": parent_run_id, "lineage_id": lineage_id,
        "created_at": datetime.now(timezone.utc).isoformat(), "notes": notes,
    }
    daily = pd.DataFrame({**wf.oos_daily, "PORTFOLIO": wf.oos_portfolio})
    frames = {
        "oos_trades": wf.oos_trades.assign(source="backtest", run_id=rid) if len(wf.oos_trades) else wf.oos_trades,
        "oos_daily": daily, "grid": wf.grid_table,
        "sensitivity": pd.DataFrame(sens["rows"]),
        "random_sharpes": pd.DataFrame({"sharpe": rnd.get("sharpes", [])}),
    }
    report = render_report(rec, reg, monthly)
    d = write_run(cfg, rec, frames, report)
    rec["run_dir"] = str(d)
    return rec, report
