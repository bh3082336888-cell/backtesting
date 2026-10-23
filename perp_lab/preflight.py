"""`lab preflight FILE --tf TF`: the checks a plugin can fail before it costs a trial, on synthetic bars only.

The proposer stages a plugin and runs this before `lab judge`. Plugin code runs only in the sandbox
and only on synthetic bars: never `load_bars`, never anything under data/. So the whole report, the
error lines included, carries no market data and may go back to the model. Nothing is recorded and
no trial is spent (spec: docs/superpowers/specs/2026-10-19-proposer-design.md, "Preflight").

Three checks, each run whenever the plugin can run, so one report names every defect at once:
1. the plugin's metadata literals and the judge's source-only refusals (`judge.precheck`);
2. the lookahead check exactly as the judge runs it that UTC day (`judge.lookahead_failures`);
3. position overlap on the shared synthetic market (`validate.synthetic_market`), at defaults and at the
   asked timeframe, against every other plugin in strategies/ at each timeframe it declares. A mirror
   of a judged plugin is refused: its OOS result would be close to minus one already seen. Near-copies
   are only reported, since which of them are refused depends on ideas, and the orchestrator applies
   those rules.
Only metadata that can't be read stops the rest: without it the plugin can't run. A check that could
not be made is named under `skipped`, with the reason, so an empty list always means "checked, nothing
found". A skip fails preflight, as a failed check does: the plugin failing on the synthetic market, or
a timeframe the market can't make. An earlier plugin that fails there holds nothing.

Positions are cached per (plugin bytes, timeframe, market version) under `proposer.state_dir`, so in
steady state only the new plugin runs; a change to the bars or to how the engine replays them needs a
new MARKET_VERSION.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from perp_lab.engine.backtest import EngineParams
from perp_lab.judge.judge import JudgeRefused, lookahead_failures, precheck
from perp_lab.overlap import mirror_overlap, overlap, positions
from perp_lab.plugins import PluginMetaError, PluginSource
from perp_lab.sandbox import client
from perp_lab.sandbox.errors import PluginError
from perp_lab.sandbox.replay import replay
from perp_lab.store import db
from perp_lab.strategy import list_plugins
from perp_lab.validate.synthetic_market import MARKET_VERSION, NoMarket, market

log = logging.getLogger(__name__)

LIST_FLOOR = 0.1      # an entry is listed when its overlap or its mirror overlap reaches this
LIST_TOP = 10         # and only the closest this many
_NAMES = ["symbol", "hour"]


def blank_report() -> dict:
    """`skipped` maps a check that could not be made ("lookahead", "overlap") to why."""
    return {"ok": False, "meta": [], "lookahead": [], "overlap": [], "mirror_of": [], "skipped": {}, "error": None}


def cache_path(cfg, strategy_hash: str, tf: str) -> Path:
    return (Path(cfg["proposer"]["state_dir"]) / "cache" / "positions"
            / f"{strategy_hash}@{tf}@v{MARKET_VERSION}.parquet")


def _series(symbols, hours, sides) -> pd.Series:
    """Positions as `overlap.positions` builds them: (symbol, hour) -> side, int8."""
    idx = pd.MultiIndex.from_arrays([np.asarray(symbols, dtype=object), np.asarray(hours, dtype="int64")],
                                    names=_NAMES)
    return pd.Series(np.asarray(sides, dtype="int8"), index=idx)


def _read_cache(p: Path) -> pd.Series | None:
    try:
        df = pd.read_parquet(p)
        return _series(df["symbol"].to_numpy(dtype=object), df["hour"].to_numpy(), df["side"].to_numpy())
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError) as e:          # torn or foreign: measure again
        log.warning("position cache %s is unreadable (%s); running the plugin again", p.name, e)
        return None


def _write_cache(p: Path, pos: pd.Series) -> None:
    """Written whole, then renamed into place: a concurrent preflight reads the old file or the new one."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    pd.DataFrame({"symbol": pos.index.get_level_values("symbol").to_numpy(dtype=object),
                  "hour": pos.index.get_level_values("hour").to_numpy(dtype="int64"),
                  "side": pos.to_numpy(dtype="int8")}).to_parquet(tmp, index=False)
    os.replace(tmp, p)


def positions_of(cfg, src: PluginSource, tf: str) -> pd.Series:
    """The plugin's positions on the synthetic market at tf, at its defaults: each symbol run in a fresh
    sandbox worker, its decisions replayed by the trusted engine. PluginError if it fails on any symbol
    (nothing is cached then); NoMarket for a timeframe the market can't make."""
    p = cache_path(cfg, src.hash, tf)
    cached = _read_cache(p)
    if cached is not None:
        return cached
    bars = market(tf)
    ep = EngineParams()

    def trades(sym: str) -> list[dict]:
        out = client.fresh_run(cfg, src, sym, tf, bars[sym], dict(src.meta.params), ep)
        return replay(bars[sym], out.decisions, symbol=sym, tf=tf, ep=ep).trades

    with ThreadPoolExecutor(min(len(bars), client.max_workers(cfg))) as pool:
        held = [t for ts in pool.map(trades, bars) for t in ts]
    pos = positions(pd.DataFrame(held))
    _write_cache(p, pos)
    return pos


def _earlier(cfg, name: str) -> list[tuple[str, PluginSource, pd.Series]]:
    """(timeframe, source, positions) of an earlier plugin at each timeframe it declares. One that fails
    holds nothing; one whose metadata can't be read, or a timeframe without a market, is not compared."""
    try:
        src = PluginSource.read(name)
    except PluginMetaError as e:
        log.warning("%s: its metadata can't be read (%s); not compared", name, e)
        return []
    out = []
    for tf in src.meta.timeframes:
        try:
            pos = positions_of(cfg, src, tf)
        except NoMarket as e:
            log.warning("%s @ %s: %s; not compared", name, tf, e)
            continue
        except PluginError as e:
            log.warning("%s @ %s fails on the synthetic market (%s); it counts as holding nothing",
                        name, tf, e.reason())
            pos = _series([], [], [])
        out.append((tf, src, pos))
    return out


def _done(report: dict) -> dict:
    report["ok"] = not (report["meta"] or report["lookahead"] or report["mirror_of"] or report["skipped"]) \
        and report["error"] is None
    return report


def _overlap(cfg, src: PluginSource, tf: str, report: dict) -> None:
    """The overlap check into `report`: the listing and `mirror_of`, or why it could not be made."""
    try:
        mine = positions_of(cfg, src, tf)
    except NoMarket as e:
        report["skipped"]["overlap"] = str(e)
        return
    except PluginError as e:
        report["skipped"]["overlap"] = f"the plugin fails on the synthetic market @ {tf}: {e.reason()}"
        return
    entries = []                          # rounded once: the rules here and the orchestrator's read these
    for name in list_plugins():
        if name == src.name:                              # this file itself, already in strategies/
            continue
        for otf, osrc, pos in _earlier(cfg, name):
            entries.append({"strategy": name, "timeframe": otf, "overlap": round(overlap(mine, pos), 3),
                            "mirror": round(mirror_overlap(mine, pos), 3),
                            "judged": db.get_judgement(cfg, osrc.hash, otf) is not None})
    report["mirror_of"] = [f"{e['strategy']}@{e['timeframe']}" for e in entries
                           if e["judged"] and e["mirror"] >= cfg["proposer"]["overlap_max"]]
    listed = [e for e in entries if max(e["overlap"], e["mirror"]) >= LIST_FLOOR]
    listed.sort(key=lambda e: (-max(e["overlap"], e["mirror"]), e["strategy"], e["timeframe"]))
    report["overlap"] = listed[:LIST_TOP]


def preflight(cfg, path: Path, tf: str) -> dict:
    """The report on the plugin file at `path`, by its file name, at tf. Raises SandboxUnavailable (and
    any harness failure) for the caller to report; `lab preflight` turns those into `error`. Without a
    sandbox only unreadable metadata is answered: every other plugin goes on to run code."""
    report = blank_report()
    try:
        src = PluginSource.from_text(path.stem, path.read_bytes())
    except PluginMetaError as e:
        report["meta"].append(f"{e.name}: {e}")           # as the judge words it
        why = "its metadata can't be read, so the plugin can't run"
        report["skipped"].update(lookahead=why, overlap=why)
        return _done(report)
    try:
        precheck(src, tf)
    except JudgeRefused as e:                             # the judge would refuse it; it can still run
        report["meta"].append(str(e))
    client.check(cfg)
    report["lookahead"] = lookahead_failures(cfg, src)
    _overlap(cfg, src, tf, report)
    return _done(report)
