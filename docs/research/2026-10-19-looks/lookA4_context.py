"""Look A4 (market context on the wider data; no candidate is run): three numbers the note's
arithmetic and kill rules lean on, re-derived for the 17 coins from 2020.

  The Deflated Sharpe bar: the annualised OOS Sharpe at which DSR = 0.95 for N = 7 trials over the
    OOS span, normal returns, with the judge's no-edge spread 1/(T-1) (perp_lab.judge.gates).
  The drift a long bias collects: the equal-weight 1x-long buy-and-hold Sharpe, funding paid, over
    the OOS span and its two parts (perp_lab.validate.benchmarks.buy_and_hold, as every walk-forward
    report shows it).
  Crash months, by a rule fixed before this ran: a calendar month is a crash month when the mean,
    over the coins with a close at the end of the previous month, of their close-to-close monthly
    return is -20% or worse. The list over the OOS span pins section 7's "the 2022 crash months"
    and gives their 2020-21 counterpart. The five worst months above the line are printed too.
These are unconditional market statistics: no candidate signal appears in them."""

import math

import numpy as np
import pandas as pd

from common import CFG, SYMBOLS, bars
from perp_lab.judge.gates import dsr_from_moments
from perp_lab.validate.benchmarks import buy_and_hold
from perp_lab.validate.walkforward import make_folds

FOLDS = make_folds(CFG.history_start, CFG.holdout_start, 6, 2)
A, B = FOLDS[0]["test_start"], FOLDS[-1]["test_end"]
N_TRIALS = 7
T = (B - A).days


def bar(n: int, t: int) -> float:
    lo, hi = 0.0, 0.5
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if dsr_from_moments(mid, t, 0.0, 3.0, n, 1.0 / (t - 1)) < 0.95 else (lo, mid)
    return hi * math.sqrt(365)


print(f"OOS span {A.date()} -> {B.date()} ({T} days)")
print(f"Deflated Sharpe bar (DSR 0.95, normal returns): N=7 {bar(N_TRIALS, T):.2f}; "
      f"for reference N=6 {bar(6, T):.2f}, N=50 {bar(50, T):.2f}, and N=7 over 4 years {bar(7, 1461):.2f}")

mid = pd.Timestamp("2022-07-01", tz="UTC")
for a, b in [(A, B), (A, mid), (mid, B)]:
    bh = buy_and_hold(CFG, SYMBOLS, "1h", a, b)["portfolio"]
    print(f"Buy and hold, equal weight, {a.date()} -> {b.date()}: Sharpe {bh['sharpe']:.2f}, "
          f"total return {bh['total_return']:.0%}, max drawdown {bh['max_dd']:.0%}")

closes = pd.DataFrame({s: bars(s, "1h").close.resample("1ME").last() for s in SYMBOLS})
monthly = closes.pct_change(fill_method=None)
mean = monthly.mean(axis=1, skipna=True)
n = monthly.notna().sum(axis=1)
oos = mean[(mean.index >= A) & (mean.index < B)]
crash = oos[oos <= -0.20]
print(f"\nCrash months over the OOS span (equal-weight mean monthly return <= -20%): {len(crash)}")
for m, r in crash.items():
    print(f"  {m:%Y-%m}: {r:.0%} over {int(n[m])} coins")
rest = oos[oos > -0.20].nsmallest(5)
print("Worst five months above the line: " + ", ".join(f"{m:%Y-%m} {r:.0%}" for m, r in rest.items()))

# Added after the first run, for the addendum's blind-spot paragraph; the rule above is unchanged.
print("\nMonths the addendum names as falling outside the rule (same measure; rank among the OOS months):")
for mo in ["2021-05", "2022-11"]:
    m = mean.index[mean.index.strftime("%Y-%m") == mo][0]
    print(f"  {mo}: BTC {monthly.loc[m, 'BTCUSDT']:.1%}, mean {mean[m]:.1%}, "
          f"rank {int((oos <= mean[m]).sum())} of {len(oos)}")
