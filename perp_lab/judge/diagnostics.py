"""What a verdict says about its walk-forward run beyond the decision: aggregates only.

The verdict file is the proposer's only window on a judgement, and "kill, Sharpe -1.26" doesn't say
whether costs ate a real signal, one side or one coin carried it, or the selection jumped about.
These numbers do. They are never series: no per-month or per-coin returns, no trade lists, no dates
of the best or worst stretches (docs/superpowers/specs/2026-10-19-proposer-design.md, "Verdict
diagnostics"). The model knows the OOS span's market history, and the finer the feedback, the
easier it is to fit that span instead of finding a mechanism.

Computed from a walk-forward run directory only (`summary.json`, `oos_trades`, `oos_daily`), never
from a holdout run: a verdict never carries a number derived from holdout data.

Per trade, in bp of the trade's notional. The engine's `pnl_gross` is after slippage (its fills
include it) and before fees and funding, so the gross edge is `pnl_gross + slippage`, and gross -
fees - slippage - funding is the net `pnl`. A liquidation is kept as it is: its `pnl_gross` is
-margin, so its gross is -margin plus its entry slippage, which is its realised P&L before fees and
slippage like any other trade's.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from perp_lab.overlap import nearest, positions

DATA = {"judged", "refreshed"}       # the judged run, or a refresh on today's data
POSITION_COLUMNS = ["symbol", "side", "entry_time", "exit_time"]


def _num(x) -> float | None:
    """A finite float, else None: JSON has no NaN or infinity (and the summary writes inf as a string)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _int(x) -> int | None:
    return None if x is None else int(x)


def _trades(run_dir: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """The run's OOS trades; absent when it had none."""
    p = run_dir / "oos_trades.parquet"
    return pd.read_parquet(p, columns=columns) if p.exists() else pd.DataFrame(columns=columns or [])


def _portfolio(run_dir: Path) -> pd.Series | None:
    p = run_dir / "oos_daily.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    return d["PORTFOLIO"].dropna() if "PORTFOLIO" in d else None


def _gates(cfg, s: dict) -> dict:
    """The gates' inputs: net Sharpe against the net random-entry benchmark, and the trade minimum."""
    m, r = s.get("metrics") or {}, s.get("random_entry") or {}
    pct = r.get("percentile", cfg["validation"]["random_percentile"])
    return {"net_sharpe": _num(m.get("sharpe")), "random_mean": _num(r.get("random_sharpe_mean")),
            "random_p95": _num(r.get(f"random_sharpe_p{pct}")), "pctile": _num(r.get("strategy_pctile_vs_random")),
            "long_share": _num(r.get("long_share")), "oos_trades": _int(m.get("trades")),
            "min_oos_trades": _int(s.get("min_oos_trades"))}


def _before_costs(s: dict) -> dict | None:
    g = s.get("before_costs")
    if g is None:               # a run from before the gross pass
        return None
    return {k: _num(g.get(k)) for k in ("sharpe", "random_mean", "random_p95", "pctile")}


def _mean(x: pd.Series) -> float | None:
    return _num(x.mean()) if len(x) else None


def _breadth(t: pd.DataFrame, n_folds: int) -> dict:
    """How many coins and folds made money, and how much of the total the best one supplied."""
    by_coin = t.groupby("symbol")["pnl"].sum() if len(t) else pd.Series(dtype=float)
    by_fold = t.groupby("fold")["pnl"].sum() if len(t) else pd.Series(dtype=float)
    total = float(t["pnl"].sum()) if len(t) else 0.0
    return {"coins_net_pos": int((by_coin > 0).sum()), "coins": int(len(by_coin)),
            "folds_net_pos": int((by_fold > 0).sum()), "folds": n_folds,
            "top_coin_share": _num(by_coin.max() / total) if total > 0 else None,
            "top_fold_share": _num(by_fold.max() / total) if total > 0 else None}


def _overlap(t: pd.DataFrame, daily: pd.Series | None, earlier: list[dict]) -> dict:
    """The nearest earlier runs. `nearest` keys them by (strategy, timeframe): `earlier` is oldest
    first, so a plugin name judged in two versions at one timeframe counts with its latest."""
    others = {}
    for e in earlier:
        d = Path(e["run_dir"])
        others[(e["strategy"], e["timeframe"])] = (positions(_trades(d, POSITION_COLUMNS)), _portfolio(d))
    return nearest(positions(t), daily, others)


def diagnostics(cfg, run_dir: str | Path, *, data: str, earlier: list[dict]) -> dict:
    """The verdict's `diagnostics` block for the walk-forward run in `run_dir`. `data` says whether it
    is the judged run or a refresh on today's data; `earlier` is the runs of earlier judgements, each
    {"strategy", "timeframe", "run_dir"}, for the overlap lists."""
    if data not in DATA:
        raise ValueError(f"data must be one of {sorted(DATA)}, not {data!r}")
    run_dir = Path(run_dir)
    s = json.loads((run_dir / "summary.json").read_text())
    if s.get("validation_mode") != "walkforward":     # a verdict never carries a number from a holdout run
        raise ValueError(f"diagnostics come from a walk-forward run, not a {s.get('validation_mode')} run")
    t = _trades(run_dir)
    if len(t):
        n = t["notional"].astype(float)
        slip = t["slippage"] if "slippage" in t else pd.Series(np.nan, index=t.index)   # runs before it was kept
        gross, net = (t["pnl_gross"] + slip) / n * 1e4, t["pnl"] / n * 1e4
        fees, slip_bp, funding = t["fees"] / n * 1e4, slip / n * 1e4, t["funding"] / n * 1e4
        long, short = t["side"] > 0, t["side"] < 0
        share = t["exit_reason"].astype(str).value_counts(normalize=True)
        regimes = {str(k): {"trades": int(len(g)), "gross_bp": _mean(gross[g.index])}
                   for k, g in t.groupby(t["regime"].astype(str))}
    else:
        gross = net = fees = slip_bp = funding = pd.Series(dtype=float)
        long = short = pd.Series(dtype=bool)
        share, regimes = pd.Series(dtype=float), {}
    m = s.get("metrics") or {}
    return {
        "data": data,
        "gates": _gates(cfg, s),
        "before_costs": _before_costs(s),
        "per_trade_bp": {"gross": _mean(gross), "fees": _mean(fees), "slippage": _mean(slip_bp),
                         "funding": _mean(funding), "net": _mean(net)},
        "sides": {"long": int(long.sum()), "short": int(short.sum()),
                  "long_share": _num(long.mean()) if len(t) else None,
                  "long_gross_bp": _mean(gross[long]), "long_net_bp": _mean(net[long]),
                  "short_gross_bp": _mean(gross[short]), "short_net_bp": _mean(net[short])},
        "exits": {"share": {str(k): float(v) for k, v in sorted(share.items())},
                  "mean_bars_held": _mean(t["bars_held"]) if len(t) else None},
        "regimes": regimes,
        "breadth": _breadth(t, len(s.get("folds") or [])),
        "is_vs_oos": {"is_sharpe": _num((s.get("is_metrics") or {}).get("sharpe")),
                      "oos_sharpe": _num(m.get("sharpe"))},
        "selection": dict(Counter(str(f.get("combo")) for f in s.get("folds") or [])),
        "overlap": _overlap(t, _portfolio(run_dir), earlier),
    }
