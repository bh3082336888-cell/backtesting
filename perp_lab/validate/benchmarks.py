"""Benchmarks: buy-and-hold (with funding) and random-entry (same exits).

A strategy that does not beat the random-entry distribution OOS is noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from perp_lab.data.store import load_bars
from perp_lab.engine.metrics import portfolio_returns, sharpe, summarize
from perp_lab.validate.walkforward import WFResult, _window, modal_params, run_many


def buy_and_hold(cfg, symbols, tf, a, b) -> dict:
    per = {}
    for s in symbols:
        bars = load_bars(cfg, s, tf, with_regime=False, with_metrics=False)
        d = pd.DataFrame({"close": bars.close, "f": bars.funding_charge}).resample("1D").agg(
            {"close": "last", "f": "sum"})
        r = d.close.pct_change() - d.f          # 1x long pays positive funding
        per[s] = _window(r.dropna(), a, b)
    port = portfolio_returns(per)
    return {"portfolio": summarize(port), "per_symbol": {s: summarize(r) for s, r in per.items()},
            "daily": port}


def random_entry(cfg, wf: WFResult, seeds: int | None = None, workers=None) -> dict:
    seeds = seeds or cfg["validation"]["random_seeds"]
    params = modal_params(wf)
    probs = {}
    for s in wf.symbols:
        n = int((wf.oos_trades.symbol == s).sum()) if len(wf.oos_trades) else 0
        probs[s] = n / max(wf.oos_flat_bars[s], 1.0)
    jobs = [(wf.strategy, params, s, wf.tf, None, seed, probs[s])
            for seed in range(seeds) for s in wf.symbols if probs[s] > 0]
    if not jobs:
        return {"note": "strategy took no OOS trades; benchmark undefined", "beats_random": False}
    res = run_many(cfg, jobs, workers)
    oa, ob = wf.raw["oos_span"]
    by_seed: dict[int, dict] = {}
    trades_by_seed: dict[int, int] = {}
    for r in res:
        by_seed.setdefault(r["seed"], {})[r["symbol"]] = _window(r["daily"], oa, ob)
        t = r["trades"]
        if len(t):
            st = pd.to_datetime(t.signal_time, unit="ms", utc=True)
            trades_by_seed[r["seed"]] = trades_by_seed.get(r["seed"], 0) + int(((st >= oa) & (st < ob)).sum())
    sh = np.array([sharpe(portfolio_returns(v)) for v in by_seed.values()])
    pct = cfg["validation"]["random_percentile"]
    strat_sh = wf.oos_summary["sharpe"]
    thr = float(np.percentile(sh, pct))
    return {
        "seeds": int(len(sh)), "percentile": pct, "params": params, "entry_prob": probs,
        "random_sharpe_mean": float(sh.mean()), "random_sharpe_p50": float(np.median(sh)),
        f"random_sharpe_p{pct}": thr, "strategy_oos_sharpe": strat_sh,
        "strategy_pctile_vs_random": float((sh < strat_sh).mean()),
        "beats_random": bool(strat_sh > thr),
        "random_trades_mean": float(np.mean(list(trades_by_seed.values()))) if trades_by_seed else 0.0,
        "sharpes": sh.tolist(),
    }


def regime_split(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or len(trades) == 0:
        return pd.DataFrame()

    def agg(g):
        w = g.pnl[g.pnl > 0].sum()
        l = -g.pnl[g.pnl < 0].sum()
        return pd.Series({"trades": len(g), "win_rate": (g.pnl > 0).mean(),
                          "avg_r": g.r_multiple.mean(), "sum_r": g.r_multiple.sum(),
                          "profit_factor": w / l if l > 0 else np.nan,
                          "sum_pnl_pct": g.pnl_pct.sum()})
    return trades.groupby("regime").apply(agg, include_groups=False)
