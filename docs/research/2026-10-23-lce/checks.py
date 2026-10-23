"""The LCE trials' pre-registered checks (spec, "Research note (after the two verdicts)"), fixed before
the first `lab judge`. Read-only on the judge's walk-forward run directory and its verdict file; no
strategy is run and no bar is loaded.

  The verdict: the judge's decision, every reason, Deflated Sharpe, N and sr0 (and the parent
    comparison, for lce_atr, whose PARENT is lce_pct).
  The kill kind, by this rule on summary.json's `before_costs` and `random_entry` blocks:
    "promoted"               the decision is promote; else
    "no signal"              before_costs sharpe <= before_costs random_p95 (the before-costs pass's OOS
                             Sharpe doesn't beat its own random-entry p95); else
    "signal eaten by costs"  random_entry beats_random is false (the net run doesn't beat its random p95);
    "other: <reasons>"       otherwise: both beat random, and the failing reasons name the gate.
  Per trade, in bp of notional, means over the OOS trades: gross = (pnl_gross + slippage) / notional
    (before fees, slippage and funding: `pnl_gross` is after slippage), fees, slippage, funding, and
    net = pnl; gross - fees - slippage - funding = net, trade by trade.
  Against Tradewriter's claims: trades per coin per session day (session day = ny_session_day of the
    signal time), win rate (pnl > 0), the share of target exits and mean R (`r_multiple`); and the
    count of OOS entries outside the declared session (summary.json's `outside_session`).
  Every timestamp read is asserted to be before the holdout start (2026-07-01).

The lab's before-costs pass removes fees and slippage but keeps funding; the spec's "No signal" said zero
funding. LCE trades are flat by 15:55 New York, so a trade meets at most the 16:00 UTC settlement (11:00
or 12:00 New York, inside the session; more on a coin Binance moved to 4-hour funding), about 1 bp of
notional at typical rates. The per-trade funding printed here measures it; the difference is expected to
be immaterial next to fees and slippage.

A run written before these fields existed (trial 7's) has no `before_costs`, no `outside_session` and no
`slippage` column: those print "n/a". Smoke-tested on trial 7's run, runs/20261019-065932-b571ea, before
any LCE trial.

Run from the repo root, on the run directory the verdict names:
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/research/2026-10-23-lce/checks.py runs/<run_id>
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from perp_lab.indicators import ny_session_day

HOLDOUT = pd.Timestamp("2026-07-01", tz="UTC")
COLS = ["symbol", "side", "signal_time", "entry_time", "exit_time", "exit_reason", "notional", "fees", "slippage",
        "funding", "pnl_gross", "pnl", "r_multiple"]


def num(x) -> float | None:
    """A finite number, or None."""
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def fmt(x, spec=".3f") -> str:
    return "n/a" if num(x) is None else format(x, spec)


def bp(x: pd.Series) -> pd.Series:
    """Per trade, in bp of the trade's notional."""
    return x / t.notional * 1e4


def kill_kind(v: dict, s: dict) -> str:
    if v["decision"] == "promote":
        return "promoted"
    bc, rnd = s.get("before_costs"), s.get("random_entry") or {}
    if not bc:
        return "n/a (the run has no before_costs block)"
    sh, p95 = num(bc.get("sharpe")), num(bc.get("random_p95"))
    if sh is None or p95 is None:
        return f"n/a (before_costs sharpe {fmt(bc.get('sharpe'))}, random_p95 {fmt(bc.get('random_p95'))})"
    if sh <= p95:
        return "no signal"
    if not rnd.get("beats_random"):
        return "signal eaten by costs"
    return "other: " + "; ".join(v["reasons"])


d = Path(sys.argv[1])
s = json.loads((d / "summary.json").read_text())
v = json.loads((d.parent / "verdicts" / f"{s['strategy']}@{s['timeframes'][0]}@{s['strategy_hash'][:10]}.json")
               .read_text())
assert v["walkforward"]["run_id"] == d.name
have = set(pq.read_schema(d / "oos_trades.parquet").names)
t = pd.read_parquet(d / "oos_trades.parquet", columns=[c for c in COLS if c in have])
daily = pd.read_parquet(d / "oos_daily.parquet")

for col in ("signal_time", "entry_time", "exit_time"):
    assert (pd.to_datetime(t[col], unit="ms", utc=True) < HOLDOUT).all(), col
assert (daily.index < HOLDOUT).all(), "oos_daily"

print(f"verdict {v['strategy']} @ {v['timeframe']} @ {v['strategy_hash'][:10]}: {v['decision'].upper()}")
for r in v["reasons"]:
    print(f"  - {r}")
print(f"  Deflated Sharpe {v['dsr']:.2e}, N = {v['n_trials']}, sr0 = {v['sr0']:.4f} (daily); "
      f"holdout {v['holdout_status']} ({v['holdout_reason']})")
if v.get("parent_run_id"):
    print(f"  parent run {v['parent_run_id']}: passed {v.get('parent_passed')}, "
          f"beats it in {fmt(v.get('beats_parent_frac'), '.0%')} of resamples")

m, bc, rnd = s["metrics"], s.get("before_costs") or {}, s.get("random_entry") or {}
pct = rnd.get("percentile", 95)
print(f"\nrun {d.name}, plugin {s['strategy_hash'][:10]}, OOS {s['oos_start']} .. {s['date_end']}, "
      f"{len(s['folds'])} folds, {len(s['universe'])} coins, walk-forward status {s['status']}")
print(f"  net: OOS Sharpe {fmt(m.get('sharpe'), '.2f')}, random p{pct} "
      f"{fmt(rnd.get(f'random_sharpe_p{pct}'), '.2f')}, beats random {rnd.get('beats_random')}")
print(f"  before costs (fees and slippage removed, funding kept): Sharpe {fmt(bc.get('sharpe'), '.2f')}, "
      f"random p95 {fmt(bc.get('random_p95'), '.2f')}")
print(f"kill kind: {kill_kind(v, s)}")

if t.empty:
    sys.exit("\nno OOS trades")
print(f"\n{len(t)} OOS trades, signal {pd.to_datetime(t.signal_time.min(), unit='ms', utc=True).date()} .. "
      f"exit {pd.to_datetime(t.exit_time.max(), unit='ms', utc=True)}")
slip = t["slippage"] if "slippage" in t else None
gross = None if slip is None else bp(t.pnl_gross + slip)
print("per trade, bp of notional (mean): "
      f"gross {fmt(None if gross is None else gross.mean(), '.2f')}, fees {bp(t.fees).mean():.2f}, "
      f"slippage {fmt(None if slip is None else bp(slip).mean(), '.2f')}, funding {bp(t.funding).mean():.2f}, "
      f"net {bp(t.pnl).mean():.2f}")
if gross is not None:
    gap = (gross - bp(t.fees) - bp(slip) - bp(t.funding) - bp(t.pnl)).abs().max()
    print(f"  gross - fees - slippage - funding - net, largest per trade: {gap:.1e} bp")

day = ny_session_day(t.signal_time.to_numpy(np.int64))
per_day = t.groupby([t.symbol.to_numpy(), day]).size()
listed = daily.drop(columns="PORTFOLIO").notna()
listed = int(listed[listed.index.dayofweek < 5].to_numpy().sum())
print(f"\ntrades per coin per session day: {len(t) / listed:.3f} over the {listed:,} Mon-Fri coin-days listed "
      f"in the OOS span; {per_day.mean():.3f} on the {len(per_day):,} coin-days with a trade (most {per_day.max()})")
print(f"win rate (pnl > 0) {(t.pnl > 0).mean():.1%}; target exits {(t.exit_reason == 'target').mean():.1%}; "
      f"mean R {t.r_multiple.mean():.3f}")
print("exits: " + ", ".join(f"{k} {n}" for k, n in t.exit_reason.value_counts().items()))
o = s.get("outside_session", None)
print(f"OOS entries outside the declared session: {'n/a' if o is None else o}")
