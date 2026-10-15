"""Paper-run tracking: regime-change count (for paper_eligible) and backtest-vs-paper divergence.

Both read only what the paper runner itself recorded (or fetches from the
exchange) — never the research loader — so the holdout lock is not involved.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from perp_lab.config import TF_MS, ccxt_symbol
from perp_lab.validate.regime import regime_frame


def count_regime_changes(labels: pd.Series, start: pd.Timestamp, min_days: int) -> dict:
    """Persisted regime changes on/after `start`.

    labels: 4h regime labels indexed by bar time (include pre-start context).
    The daily dominant label is taken; runs shorter than `min_days` are noise and
    dropped; a change is a kept run whose label differs from the previous kept
    run and which starts on/after `start`. An ongoing run only counts once it
    has already lasted `min_days`.
    """
    lab = labels[labels != "unknown"]
    if lab.empty:
        return {"changes": 0, "sequence": []}
    daily = lab.groupby(lab.index.floor("D")).agg(lambda x: x.value_counts().index[0])
    rid = (daily != daily.shift()).cumsum()
    runs = pd.DataFrame({"label": daily.groupby(rid).first(),
                         "start": daily.index.to_series().groupby(rid).first(),
                         "days": daily.groupby(rid).size()})
    kept = runs[runs.days >= max(min_days, 1)].reset_index(drop=True)
    start_d = pd.Timestamp(start).floor("D")
    changes, seq = 0, []
    prev = None
    for _, r in kept.iterrows():
        if prev is not None and r.label != prev and r.start >= start_d:
            changes += 1
            seq.append({"from": prev, "to": r.label, "at": str(r.start.date()), "days": int(r.days)})
        prev = r.label
    return {"changes": changes, "sequence": seq}


def fetch_4h_regimes(cfg, exchange, symbol: str, since_ms: int) -> pd.Series:
    """Live 4h candles from the exchange → regime labels (same function as research)."""
    from perp_lab.data.live import call_with_backoff, ohlcv_to_candles
    rows, since = [], since_ms
    now = int(time.time() * 1000)
    while since < now:
        part = call_with_backoff(exchange.fetch_ohlcv, ccxt_symbol(symbol, cfg.get("quote", "USDT")), "4h",
                                 since=since, limit=1500)
        if not part:
            break
        rows += part
        nxt = part[-1][0] + TF_MS["4h"]
        if nxt <= since:
            break
        since = nxt
    df = ohlcv_to_candles(rows, "4h").drop_duplicates("open_time")
    df = df[df.open_time + TF_MS["4h"] <= now]
    df.index = pd.to_datetime(df.open_time, unit="ms", utc=True)
    return regime_frame(df, cfg["regime"]).regime


def paper_regime_status(cfg, labels_by_symbol: dict[str, pd.Series], started: pd.Timestamp) -> tuple[int, dict]:
    k = cfg["paper"]["regime_change_min_days"]
    detail = {s: count_regime_changes(l, started, k) for s, l in labels_by_symbol.items()}
    changes = min((d["changes"] for d in detail.values()), default=0)
    return changes, detail


def refresh_paper_run(cfg, run_id: str, exchange=None) -> dict:
    """Recompute regime changes for a paper run and update paper_eligible in the store."""
    import sqlalchemy as sa

    from perp_lab.store.db import engine, runs, update_paper_tracking
    with engine(cfg).connect() as c:
        r = c.execute(sa.select(runs.c.paper_started_at, runs.c.universe).where(runs.c.run_id == run_id)).first()
    if r is None or r.paper_started_at is None:
        raise ValueError(f"{run_id} is not a paper run")
    started = pd.Timestamp(r.paper_started_at)
    started = started if started.tzinfo else started.tz_localize("UTC")
    if exchange is None:
        from perp_lab.data.live import make_exchange
        exchange = make_exchange()
    # regime needs ~180d of 4h history for the vol rank, plus context before the paper start
    since = int((started - pd.Timedelta(days=200)).timestamp() * 1000)
    labels = {s: fetch_4h_regimes(cfg, exchange, s, since) for s in r.universe}
    changes, detail = paper_regime_status(cfg, labels, started)
    return update_paper_tracking(cfg, run_id, changes, detail)


def replay_divergence(cfg, paper_dir: Path, strategy_cls, params: dict, tf: str) -> pd.DataFrame:
    """Re-run the backtest engine on exactly the bars the paper runner saw.

    Returns daily cumulative pnl (% of starting equity, summed over symbols) for
    paper vs replay, and their difference. Same strategy, same bars — the gap is
    execution: spread vs slippage model, live stop fills, timing.
    """
    from perp_lab.engine.backtest import EngineParams, run_backtest
    paper_dir = Path(paper_dir)
    tp = paper_dir / "trades.parquet"
    paper = pd.read_parquet(tp) if tp.exists() else pd.DataFrame()
    rep_rows = []
    for f in sorted(paper_dir.glob("bars_*.parquet")):
        sym = f.stem[len("bars_"):]
        bars = pd.read_parquet(f)
        bars.index = pd.to_datetime(bars.open_time, unit="ms", utc=True)
        live = np.flatnonzero(bars["live"].to_numpy())
        if len(live) == 0:
            continue
        s = strategy_cls(**params)
        res = run_backtest(s, bars, s.prepare(bars), symbol=sym, tf=tf,
                           ep=EngineParams.from_config(cfg, sym), start_i=int(live[0]))
        rep_rows += res.trades
    rep = pd.DataFrame(rep_rows)

    def cum(t):
        if t is None or t.empty:
            return pd.Series(dtype=float)
        d = pd.to_datetime(t.exit_time, unit="ms", utc=True).dt.floor("D")
        return t.pnl_pct.groupby(d).sum().cumsum()

    out = pd.concat({"paper": cum(paper), "replay": cum(rep)}, axis=1).sort_index().ffill().fillna(0.0)
    if out.empty:
        return out
    out["divergence"] = out["paper"] - out["replay"]
    return out
