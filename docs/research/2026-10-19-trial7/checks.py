"""Trial 7's result and the pre-registered checks (section 10 of the note). Read-only on the judge's
walk-forward run directory and its verdict file; no strategy is run and no bar is loaded.

  The verdict: the judge's decision, reasons, Deflated Sharpe, N and sr0, and the holdout status.
  The headline: OOS and IS Sharpe, trades, random-entry percentile, narrow island, buy and hold.
  The checks, with the rules exactly as sections 7 and 9 fix them:
    fewer than 64 OOS trades (a broken data path, not a result);
    gross OOS P&L (`pnl_gross`: after slippage, before fees and funding) <= 0;
    gross > 0 but net (`pnl`) < 0;
    IS Sharpe above OOS Sharpe by more than about 1;
    sides: `pnl_gross` summed by `side`, either side <= 0 fails;
    concentration: one symbol or one 2-month fold (the walk-forward's `fold`, by `signal_time`)
      supplying more than half of the OOS net P&L;
    crash months: the sum of `pnl` over OOS trades whose `exit_time` falls, by UTC month, outside the
      eight crash months of section 9 is <= 0.
  Every timestamp read is asserted to be before the holdout start (2026-07-01).

Run from the repo root:
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/research/2026-10-19-trial7/checks.py runs/20261019-065932-b571ea
"""
import json
import sys
from pathlib import Path

import pandas as pd

HOLDOUT = pd.Timestamp("2026-07-01", tz="UTC")
CRASH = {"2021-06", "2022-01", "2022-04", "2022-05", "2022-06", "2024-04", "2025-02", "2026-06"}

d = Path(sys.argv[1])
t = pd.read_parquet(d / "oos_trades.parquet")
daily = pd.read_parquet(d / "oos_daily.parquet")
s = json.loads((d / "summary.json").read_text())
v = json.loads((d.parent / "verdicts" / f"{s['strategy']}@{s['timeframes'][0]}@{s['strategy_hash'][:10]}.json").read_text())
assert v["walkforward"]["run_id"] == d.name

for col in ("signal_time", "entry_time", "exit_time"):
    assert pd.to_datetime(t[col], unit="ms", utc=True).max() < HOLDOUT, col
assert daily.index.max() < HOLDOUT, "oos_daily"

print(f"verdict {v['strategy']} @ {v['timeframe']} @ {v['strategy_hash'][:10]}: {v['decision'].upper()}")
for r in v["reasons"]:
    print(f"  - {r}")
print(f"  Deflated Sharpe {v['dsr']:.2e}, N = {v['n_trials']}, sr0 = {v['sr0']:.4f} (daily); "
      f"holdout {v['holdout_status']} ({v['holdout_reason']})")

m, im, re, se = s["metrics"], s["is_metrics"], s["random_entry"], s["sensitivity"]
print(f"\nrun {d.name}, plugin {s['strategy_hash'][:10]}, OOS {s['oos_start']} .. {s['date_end']}, "
      f"{len(s['folds'])} folds, {len(s['universe'])} coins, walk-forward status {s['status']}")
print(f"  OOS: Sharpe {m['sharpe']:.2f}, return {m['total_return']:.1%}, max DD {m['max_dd']:.1%}, "
      f"PF {m['profit_factor']:.2f}, win {m['win_rate']:.1%}, avg R {m['avg_r']:.3f}, trades {m['trades']}")
print(f"  IS (selected params' train windows): Sharpe {im['sharpe']:.2f}")
print(f"  random entry ({re['seeds']} seeds, params {re['params']}): mean {re['random_sharpe_mean']:.2f}, "
      f"p95 {re['random_sharpe_p95']:.2f}, strategy at the {re['strategy_pctile_vs_random']:.0%} percentile")
print(f"  narrow island: {se['narrow_island']} ({se['note']}); sensitivity base Sharpe {se['base_sharpe']:.2f}")
print(f"  buy and hold: Sharpe {s['buy_and_hold']['sharpe']:.2f}")
picks = pd.Series([f["combo"] for f in s["folds"]]).value_counts()
print("  combo picked per fold: " + ", ".join(f"{k} x{n}" for k, n in picks.items()))

et = pd.to_datetime(t.exit_time, unit="ms", utc=True)
print(f"\n{len(t)} OOS trades, signal {pd.to_datetime(t.signal_time.min(), unit='ms', utc=True).date()}"
      f" .. exit {et.max()}; oos_daily {daily.index.min().date()} .. {daily.index.max().date()}")
gross, net = t.pnl_gross.sum(), t.pnl.sum()
print(f"totals: pnl_gross {gross:,.2f}  fees {t.fees.sum():,.2f}  funding {t.funding.sum():,.2f}  pnl {net:,.2f}")

print("\n1. trades < 64 (broken data path):", len(t), "->", "FAIL" if len(t) < 64 else "pass")
print("2. gross <= 0:", f"{gross:,.2f}", "->", "FAIL" if gross <= 0 else "pass")
print("3. gross > 0 but net < 0:", f"gross {gross:,.2f}, net {net:,.2f}", "->",
      "FAIL" if gross > 0 and net < 0 else "pass")
print(f"4. IS - OOS Sharpe > ~1: IS {im['sharpe']:.3f}, OOS {m['sharpe']:.3f}, gap {im['sharpe'] - m['sharpe']:.3f} ->",
      "FAIL" if im["sharpe"] - m["sharpe"] > 1 else "pass")

sides = t.groupby("side").agg(trades=("pnl", "size"), pnl_gross=("pnl_gross", "sum"), pnl=("pnl", "sum"))
sides["share"] = sides.trades / len(t)
sides["gross_per_trade"] = sides.pnl_gross / sides.trades
print("\n5. sides (either side's gross <= 0 fails):\n" + sides.to_string(float_format=lambda x: f"{x:,.3f}"))
print("   ->", "FAIL" if (sides.pnl_gross <= 0).any() or len(sides) < 2 else "pass")

print("\n6. concentration (one symbol or one 2-month fold > half of OOS net P&L fails):")
for key in ("symbol", "fold"):
    g = t.groupby(key).agg(trades=("pnl", "size"), pnl_gross=("pnl_gross", "sum"), pnl=("pnl", "sum"))
    g["share_of_net"] = g.pnl / net if net else float("nan")
    g = g.sort_values("pnl", ascending=False)
    if key == "fold":
        f = pd.DataFrame(s["folds"]).set_index("fold")
        g.insert(0, "test", f.test_start.str[:7].reindex(g.index) + ".." + f.test_end.str[:7].reindex(g.index))
        g.insert(1, "combo", f.combo.reindex(g.index))
    print(f"   by {key}:\n" + g.to_string(float_format=lambda x: f"{x:,.3f}"))
    top = g.iloc[0]
    if net <= 0:
        print(f"   -> top {key} {g.index[0]}: net {top.pnl:,.2f}; total net is {net:,.2f} <= 0, so no share is defined")
    else:
        print(f"   -> top {key} {g.index[0]}: {top.share_of_net:.1%} of net ->",
              "FAIL" if top.share_of_net > 0.5 else "pass")
    print(f"   {key}s with net > 0: {(g.pnl > 0).sum()} of {len(g)}")

mo = et.dt.strftime("%Y-%m")
crash = mo.isin(CRASH)
print("\n7. crash months (net pnl outside the eight months, by exit_time UTC month, <= 0 rules out any child):")
print(f"   inside : {crash.sum():5d} trades, pnl_gross {t.pnl_gross[crash].sum():,.2f}, pnl {t.pnl[crash].sum():,.2f}")
print(f"   outside: {(~crash).sum():5d} trades, pnl_gross {t.pnl_gross[~crash].sum():,.2f}, pnl {t.pnl[~crash].sum():,.2f}")
per = t[crash].groupby(mo[crash]).agg(trades=("pnl", "size"), pnl=("pnl", "sum"))
print("   per crash month:\n" + per.to_string(float_format=lambda x: f"{x:,.2f}"))
print("   ->", "outside <= 0: all the profit comes from the crash months" if t.pnl[~crash].sum() <= 0
      else "outside > 0")
