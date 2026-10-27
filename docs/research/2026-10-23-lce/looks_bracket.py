"""LCE look before the trials: the share of in-session bars with a valid bracket. Counts only.

Why: the session-calibrated random benchmark fires at the strategy's rate over every flat in-session bar
(walkforward._compact's flat_bars), but RandomEntry enters only where the plugin's bracket is valid for the
side it draws. The bracket is NaN on days without levels (warm-up, missing days, lce_atr's ATR warm-up)
and where the close is beyond the outermost level. So the benchmark's expected trade count is about the
strategy's times the valid share. This look measures that share before the trials (the final whole-branch
review's finding).

For both plugins at defaults, on each coin's research 5m bars, the plugin runs in the sandbox exactly as
in looks.py: a sealed worker, and only the run's features are read. The run's decisions, fingerprint,
trades and equity are never read, and nothing is joined to later prices. The look takes the bars whose
open_time lies in the walk-forward's OOS span [2020-07-01, 2026-07-01) with in_window == 1. It reports the
share of them with a valid long bracket, with a valid short bracket, and with both. Each is recomputed
exactly as LcePct.bracket does, via looks.valid_bracket: finite stop and target, stop < close < target
strictly for a long (target < close < stop for a short), and in the window. It then gives the totals
pooled over the coins and the minimum across coins. One diagnostic column, "no levels", is the share of
those bars on a session day without levels (anchor_fb is NaN: warm-up, a listing, missing days). The rest
of the invalid share is a close beyond the outermost level.

Like looks.py, it raises `sandbox.max_output_mb` in memory to 512 MB for its own workers. A coin's
features are about 150 MB, over the configured 64 MB.

Run from the repo root:
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/research/2026-10-23-lce/looks_bracket.py > docs/research/2026-10-23-lce/looks_bracket.out
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from looks import HOLDOUT, OUTPUT_MB, PLUGINS, TF, THREADS, valid_bracket
from perp_lab.config import load_config
from perp_lab.data.store import load_bars
from perp_lab.engine.backtest import EngineParams
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client

OOS_START = pd.Timestamp("2020-07-01", tz="UTC")
FLOOR = 0.95                     # the review's stop rule: a coin under it, long or short, goes to the user
KINDS = ("long", "short", "both")
COLS = KINDS + ("nolev",)


def look(cfg, srcs: dict, sym: str) -> dict:
    bars = load_bars(cfg, sym, TF)
    t = bars.open_time.to_numpy(np.int64)
    assert len(t) and t.max() < HOLDOUT.value // 10**6, sym
    close = bars.close.to_numpy(float)
    span = (t >= OOS_START.value // 10**6) & (t < HOLDOUT.value // 10**6)
    ones = np.ones(len(t))
    out = {}
    for name, src in srcs.items():
        with client.spawn(cfg, src) as w:
            w.bars(sym, TF, bars)
            f = w.run(dict(src.meta.params), EngineParams.from_config(cfg, sym), features=True).features
        assert f.index.equals(bars.index), (sym, name)
        sel = span & (f.in_window.to_numpy() == 1)
        lng = valid_bracket(f, close, ones) & sel
        sht = valid_bracket(f, close, -ones) & sel
        out[name] = {"n": int(sel.sum()), "long": int(lng.sum()), "short": int(sht.sum()),
                     "both": int((lng & sht).sum()), "nolev": int((sel & np.isnan(f.anchor_fb.to_numpy())).sum())}
        del f
    return out


def main():
    cfg = load_config()
    cfg["sandbox"]["max_output_mb"] = OUTPUT_MB
    srcs = {name: PluginSource.read(name) for name in PLUGINS}
    syms = cfg.symbols
    with ThreadPoolExecutor(THREADS) as pool:      # each thread spawns and closes its own workers
        res = dict(zip(syms, pool.map(lambda s: look(cfg, srcs, s), syms)))

    print(f"LCE look: in-window bars of the OOS span {OOS_START:%Y-%m-%d} .. {HOLDOUT:%Y-%m-%d} (exclusive) with a "
          f"valid bracket; research {TF} bars, defaults, counts only")
    print("plugins: " + ", ".join(f"{n} {s.hash[:12]}" for n, s in srcs.items()))
    for name in PLUGINS:
        print(f"\n{name}\n{'coin':<9} {'in-window':>9} {'long':>6} {'short':>6} {'both':>6} {'no levels':>9}")
        tot = dict.fromkeys(("n",) + COLS, 0)
        for sym in syms:
            p = res[sym][name]
            print(f"{sym:<9} {p['n']:>9,} " + " ".join(f"{p[k] / p['n']:>6.3f}" for k in KINDS)
                  + f" {p['nolev'] / p['n']:>9.3f}")
            for k in tot:
                tot[k] += p[k]
        print(f"{'total':<9} {tot['n']:>9,} " + " ".join(f"{tot[k] / tot['n']:>6.3f}" for k in KINDS)
              + f" {tot['nolev'] / tot['n']:>9.3f}")
        lows = []
        for k in KINDS:
            sym = min(syms, key=lambda s: res[s][name][k] / res[s][name]["n"])
            lows.append(f"{k} {res[sym][name][k] / res[sym][name]['n']:.3f} ({sym})")
        print("minimum across coins: " + ", ".join(lows))
        under = [f"{s} {k} {res[s][name][k] / res[s][name]['n']:.3f}" for s in syms for k in ("long", "short")
                 if res[s][name][k] / res[s][name]["n"] < FLOOR]
        print(f"coins under {FLOOR} (long or short): " + (", ".join(under) if under else "none"))


if __name__ == "__main__":
    main()
