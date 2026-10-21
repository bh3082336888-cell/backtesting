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


def random_entry(cfg, wf: WFResult, seeds: int | None = None, workers=None, *,
                 overrides: dict | None = None, strat_sharpe: float | None = None) -> dict:
    """Random entries with the plugin's exits, frequency and overall OOS long share: a long-biased
    plugin must beat random longs, not 50/50, or it passes on market drift alone."""
    seeds = seeds or cfg["validation"]["random_seeds"]
    params = modal_params(wf)
    probs = {}
    for s in wf.symbols:
        n = int((wf.oos_trades.symbol == s).sum()) if len(wf.oos_trades) else 0
        probs[s] = n / max(wf.oos_flat_bars[s], 1.0)
    long_share = float((wf.oos_trades.side > 0).mean()) if len(wf.oos_trades) else 0.5
    jobs = [(params, s, wf.tf, overrides, seed, (probs[s], long_share))
            for seed in range(seeds) for s in wf.symbols if probs[s] > 0]
    if not jobs:
        return {"note": "strategy took no OOS trades; benchmark undefined", "beats_random": False}
    res = run_many(cfg, wf.src, jobs, workers, wf.stderr_dir)
    oa, ob = wf.raw["oos_span"]
    # A coin the strategy never traded gets no job, but it stays in every random portfolio as
    # zeros on its listed days, so the portfolio averages the same coins on the same days.
    idle = {s: pd.Series(0.0, index=_window(wf.oos_daily[s], oa, ob).index)
            for s in wf.symbols if probs[s] == 0}
    by_seed: dict[int, dict] = {seed: dict(idle) for seed in range(seeds)}
    trades_by_seed: dict[int, int] = {}
    for r in res:
        by_seed.setdefault(r["seed"], {})[r["symbol"]] = _window(r["daily"], oa, ob)
        t = r["trades"]
        if len(t):
            st = pd.to_datetime(t.signal_time, unit="ms", utc=True)
            trades_by_seed[r["seed"]] = trades_by_seed.get(r["seed"], 0) + int(((st >= oa) & (st < ob)).sum())
    sh = np.array([sharpe(portfolio_returns(v)) for v in by_seed.values()])
    pct = cfg["validation"]["random_percentile"]
    strat_sh = wf.oos_summary["sharpe"] if strat_sharpe is None else strat_sharpe
    thr = float(np.percentile(sh, pct))
    return {
        "seeds": int(len(sh)), "percentile": pct, "params": params, "entry_prob": probs, "long_share": long_share,
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
