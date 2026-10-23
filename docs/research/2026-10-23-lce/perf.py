"""lce_pct's prepare on BTCUSDT's full research 5m bars at vp_days = 10, in a sandbox worker.

The spec's performance budget ("Performance budget"): the median of 5 `prepare` calls in a sealed
worker, in CPU seconds the worker spent, measured from outside (`Worker.prepare`, as the exam's cost
probe does). Recorded, not a gate: on 2026-10-23 the user decided to continue and settle the exam
budget only if LCE is promoted.

Timing only. The bars are research bars (`load_bars` stops at holdout_start); nothing here computes
returns, P&L, trades or any other outcome.

Two placements on this hybrid CPU: the scheduler's (as an exam runs), and pinned to the performance
cores (the worker inherits the parent's CPU affinity), which is how Tasks 6 and 7 reported their
numbers.

Run: .venv/bin/python docs/research/2026-10-23-lce/perf.py > docs/research/2026-10-23-lce/perf.out
"""

import os
import platform
import statistics
import subprocess
from pathlib import Path

import pandas as pd

from perp_lab.config import load_config
from perp_lab.data.store import load_bars
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client

SYMBOL, TF, RUNS, BUDGET_S = "BTCUSDT", "5m", 5, 0.30


def fast_cpus() -> set[int]:
    """The cores within 10% of the fastest's maximum clock (a hybrid CPU's performance cores)."""
    cpus = sorted(os.sched_getaffinity(0))

    def khz(c):
        try:
            return int(Path(f"/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq").read_text())
        except (OSError, ValueError):
            return 0
    top = max(map(khz, cpus))
    return {c for c in cpus if khz(c) >= 0.9 * top}


def measure(cfg, src, bars, params, cpus=None) -> list[float]:
    old = os.sched_getaffinity(0)
    if cpus:
        os.sched_setaffinity(0, cpus)
    try:
        w = client.spawn(cfg, src)
        try:
            w.bars(SYMBOL, TF, bars)
            return [w.prepare(params) for _ in range(RUNS)]
        finally:
            w.close()
    finally:
        os.sched_setaffinity(0, old)


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def main():
    cfg = load_config()
    src = PluginSource.read("lce_pct")
    bars = load_bars(cfg, SYMBOL, TF)
    params = {**src.meta.params, "vp_days": 10}
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    span = pd.to_datetime(bars.open_time.iloc[[0, -1]], unit="ms", utc=True)
    fast = fast_cpus()
    print(f"lce_pct prepare, {SYMBOL} {TF} research bars, vp_days = 10, in a sandbox worker")
    print(f"plugin {src.hash[:12]} at commit {commit}; {cpu_model()}; {len(fast)} of "
          f"{len(os.sched_getaffinity(0))} cores are performance cores")
    print(f"bars: {len(bars):,} ({span.iloc[0]:%Y-%m-%d} .. {span.iloc[1]:%Y-%m-%d}), {bars.shape[1]} columns")
    print(f"load average before: {' '.join(f'{x:.2f}' for x in os.getloadavg())}")
    print(f"budget (spec; recorded, not a gate): {BUDGET_S:.2f} s")
    print()
    for label, cpus in (("scheduler's placement", None), ("performance cores", fast)):
        t = measure(cfg, src, bars, params, cpus)
        print(f"{label}: median {statistics.median(t):.3f} s CPU over {RUNS} runs "
              f"({', '.join(f'{x:.3f}' for x in t)})")


if __name__ == "__main__":
    main()
