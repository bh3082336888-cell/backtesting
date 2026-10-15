"""Parameter sensitivity: does the edge survive ±10/20% moves of each parameter?

Base = the modal walk-forward selection. Each numeric param is moved one at a
time; every variant runs fixed over the OOS span. This is a stability diagnostic,
not a performance claim. A strat is a "narrow island" if any ±20% move drops
portfolio Sharpe by more than `island_drop` (or flips its sign).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from perp_lab.engine.metrics import portfolio_returns, sharpe
from perp_lab.strategy import load_strategy_class
from perp_lab.validate.walkforward import WFResult, _window, combo_key, modal_params, run_many


def _variants(params: dict, steps: list[float]) -> list[tuple[str, float, dict]]:
    out = []
    for k, v in params.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        seen = {v}
        for st in steps:
            nv = v * (1 + st)
            nv = int(round(nv)) if isinstance(v, int) else round(float(nv), 6)
            if nv in seen or (isinstance(v, int) and nv < 1):
                continue
            seen.add(nv)
            out.append((k, st, {**params, k: nv}))
    return out


def sensitivity(cfg, wf: WFResult, workers=None) -> dict:
    cls = load_strategy_class(wf.strategy)
    base = {**cls.params, **modal_params(wf)}
    steps = cfg["validation"]["sensitivity_steps"]
    var = _variants(base, steps)
    all_params = [base] + [p for _, _, p in var]
    jobs = [(wf.strategy, p, s, wf.tf, None, None, None) for p in all_params for s in wf.symbols]
    res = run_many(cfg, jobs, workers)
    oa, ob = wf.raw["oos_span"]
    by = {}
    for r in res:
        by.setdefault(combo_key(r["params"]), {})[r["symbol"]] = _window(r["daily"], oa, ob)
    sh = {k: sharpe(portfolio_returns(v)) for k, v in by.items()}
    base_sh = sh[combo_key(base)]
    drop_lim = cfg["validation"]["island_drop"]
    rows, island, worst = [], False, None
    for k, st, p in var:
        s = sh[combo_key(p)]
        rows.append({"param": k, "step": st, "value": p[k], "sharpe": s})
        if abs(abs(st) - 0.2) < 1e-9 and base_sh > 0:
            drop = 1 - s / base_sh
            if drop > drop_lim:
                island = True
                worst = max(worst or 0.0, drop)
    return {
        "base_params": base, "base_sharpe": base_sh,
        "rows": rows,
        "narrow_island": (island if base_sh > 0 else None),
        "worst_drop_at_20pct": worst,
        "note": "n/a: base Sharpe <= 0, nothing to be an island of" if base_sh <= 0 else "",
    }
