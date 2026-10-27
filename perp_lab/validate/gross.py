"""The signal before costs: the walk-forward's selected combos and the random-entry benchmark,
re-run with fees and slippage removed (funding kept). A plugin whose edge only shows here has a
signal the costs eat; one that shows nowhere has none. Sizing compounds on equity, so this is
a re-run, not the net result plus the costs."""

from __future__ import annotations

from perp_lab.engine.metrics import portfolio_returns, sharpe
from perp_lab.validate.benchmarks import random_entry
from perp_lab.validate.walkforward import WFResult, combo_key, run_many, stitch_oos

GROSS = {"taker": 0.0, "maker": 0.0, "slippage": 0.0, "slippage_scale": 0.0}


def before_costs(cfg, wf: WFResult, seeds=None, workers=None) -> dict:
    combos = {x["combo"]: x["params"] for x in wf.selected}
    jobs = [(p, s, wf.tf, GROSS, None, None) for p in combos.values() for s in wf.symbols]
    R = {(combo_key(r["params"]), r["symbol"]): r for r in run_many(cfg, wf.src, jobs, workers, wf.stderr_dir)}
    sh = sharpe(portfolio_returns(stitch_oos(R, wf.selected, wf.symbols)))
    rnd = random_entry(cfg, wf, seeds, workers, overrides=GROSS, strat_sharpe=sh)
    pct = cfg["validation"]["random_percentile"]
    return {"sharpe": sh, "random_mean": rnd.get("random_sharpe_mean"),
            "random_p95": rnd.get(f"random_sharpe_p{pct}"),
            "pctile": rnd.get("strategy_pctile_vs_random"), "long_share": rnd.get("long_share")}
