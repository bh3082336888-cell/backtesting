# LCE trials 8 and 9: what the verdicts mean

Spec: [`docs/superpowers/specs/2026-10-23-lce-model-design.md`](../../superpowers/specs/2026-10-23-lce-model-design.md).
The two plugins code Tradewriter's LCE day-trading model (Level → Confirmation → Execution: six
EMA-cloud flowchart branches at volume-profile levels). They trade 17 Binance perps on 5m bars in US
equity hours: entries 09:40–15:45 New York, Mon–Fri, flat by 15:55. They differ only in how an ES
distance becomes a crypto distance:
- `lce_pct`: fixed percentages of price;
- `lce_atr`: multiples of each coin's daily ATR.

Both trials ran on 2026-10-27 from `main` at 2191314. N went from 7 to 9. **Both plugins are killed,
so LCE does not go into the lab's crypto book**, which is the one decision the spec says this work
makes.

In short:
- **`lce_pct` loses before costs:** −2.47 bp per trade gross.
- **`lce_atr` has a small positive gross:** +2.42 bp per trade. Binance's costs, about 14 bp per
  trade, swamp it.
  - It is not an edge the data can tell from luck. Its t clustered by date is 1.19.
  - Ten market-wide sell-off days carry almost all of it: without them it is +0.06 bp.
- Neither result looks like Tradewriter's account of his own trading:
  - The bracket's planned reward to risk, at the signal bar's close, has a median of 0.558
    (`lce_pct`) and 0.666 (`lce_atr`), against his 1.5–2R.
  - The win rate is 36.5% and 45.2%, against his "85%".

The run directories are git-ignored. Most numbers below come from a file in this folder:

| file | what it is |
|---|---|
| [`checks.py`](checks.py), [`checks_lce_pct.out`](checks_lce_pct.out), [`checks_lce_atr.out`](checks_lce_atr.out) | the pre-registered checks (committed before the first `lab judge`) and their output on the two runs |
| [`attribution.py`](attribution.py), [`attribution.out`](attribution.out) | after the verdicts: per-branch attribution, planned R:R, which zone the signal bar touched, and other per-trade diagnostics from the runs' OOS trades |
| [`looks.py`](looks.py), [`looks.out`](looks.out), [`looks_bracket.py`](looks_bracket.py), [`looks_bracket.out`](looks_bracket.out) | the pre-trial looks at real data (counts only) |
| [`preflight_lce_pct.json`](preflight_lce_pct.json), [`preflight_lce_atr.json`](preflight_lce_atr.json) | `lab preflight` at 5m |
| [`perf.py`](perf.py), [`perf.out`](perf.out) | the speed record for `prepare` |

A few numbers come from the verdict files themselves (`runs/verdicts/lce_pct@5m@5458edefef.json` and
`runs/verdicts/lce_atr@5m@181835ba54.json`, `diagnostics`), or from the runs' `report.md`. Those
places are marked "(verdict)" or "(report)". "bp" means basis points of the trade's notional, as a
mean per trade. "Gross" means before fees, slippage and funding, computed as
`(pnl_gross + slippage) / notional`.

## 1. The verdicts

Both runs are walk-forwards over the OOS span 2020-07-01 → 2026-07-01: 36 folds (6 months train, 2
months test), 17 coins, and 4 grid combos (`vp_days` ∈ {5, 10} × `theta` ∈ {0.01, 0.02}). No entry
fell outside the declared session in either run, and the holdout was never read.

| | trial 8, `lce_pct` | trial 9, `lce_atr` |
|---|---|---|
| plugin | `5458edefef` | `181835ba54` (`PARENT = "lce_pct"`) |
| run | `20261027-071510-8626d9` | `20261027-075940-4464e6` |
| decision | **KILL** | **KILL** |
| reasons | OOS Sharpe −17.39 ≤ 0; Deflated Sharpe 0.00 < 0.95 (N=8) | OOS Sharpe −2.89 ≤ 0; Deflated Sharpe 0.00 < 0.95 (N=9) |
| Deflated Sharpe, sr0 (daily) | 0.00e+00, 0.0312 | 0.00e+00, 0.0325 |
| holdout | n/a (nothing queued) | n/a (nothing queued) |
| parent | none | `lce_pct`'s run, passed False; "beats it" n/a |
| OOS Sharpe, net | −17.39 | −2.89 |
| random entry, same exits, 200 seeds: p95 | −20.13: beats random | −5.31: beats random |
| random trades per seed against the strategy's | 23,950.2 / 25,652 = 0.934 | 11,211.7 / 11,337 = 0.989 |
| before costs (fees and slippage removed, funding kept): Sharpe against random p95 | −4.43 against −5.68 | 0.67 against 0.49 |
| OOS trades | 25,652 | 11,337 |
| per trade, bp: gross, fees, slippage, funding, net | −2.47, 8.21, 4.74, 0.01, −15.43 | 2.42, 8.62, 5.26, 0.15, −11.61 |
| win rate (pnl > 0); target exits; mean R | 36.5%; 59.7%; −0.435 | 45.2%; 45.8%; −0.073 |
| exits | target 15,312, stop 10,190, signal 150 | target 5,194, stop 3,577, signal 2,566 |
| return, max drawdown (report) | −99.7%, −99.7% | −39.0%, −39.2% |
| coins / folds net positive (verdict) | 0 of 17 / 0 of 36 | 0 of 17 / 4 of 36 |
| mean bars held (verdict) | 2.03 (about 10 minutes) | 16.06 (about 80 minutes) |
| buy and hold, same span (report) | Sharpe 0.72 | Sharpe 0.72 |

The walk-forward status is `noise` for both. The narrow-island test was n/a for both, because the
base Sharpe was ≤ 0. `lce_pct` lost in every one of the 72 OOS months (report, monthly table), and
every coin's Sharpe ran from −6.37 to −8.07.

## 2. What kind of kill

The rule was fixed in [`checks.py`](checks.py) before the first `lab judge`. Its last change,
66d433f, was committed at 06:56 UTC on 2026-10-27; trial 8 started at 07:15 UTC. The rule reads the
run's `before_costs` and `random_entry` blocks, in this order:
- **promoted:** the decision is promote;
- **no signal:** the before-costs Sharpe is at or below its own random-entry p95 (same exits, no
  fees, no slippage);
- **signal eaten by costs:** the net run doesn't beat its random p95;
- **other:** both beat random, and another gate kills it. The failing reasons name the gate.

**Both kills are "other."** Each plugin beats its random benchmark net and before costs, and each is
killed by OOS Sharpe ≤ 0 and Deflated Sharpe 0.00 < 0.95.

The label is correct by the rule, but it hides how different the two results are. The rule's two
benchmark comparisons ask whether the entries beat random entries in the same bracket. They don't
ask whether the trades make money before costs. Read plainly, the numbers say this.

**`lce_pct` loses before costs.**
- Gross is −2.47 bp per trade, and the before-costs Sharpe is −4.43.
- It beats random only because random entries in the same bracket lose more: their before-costs
  Sharpe has a mean of −6.33 (verdict).
- Its target is hit on 59.7% of trades, but only 36.5% of trades make money:
  - target exits average +0.160 R net;
  - stop exits average −1.327 R (attribution).
- The targets are so close that one exit at the target pays little more than the costs. Fees are
  8.21 bp and slippage 4.74 bp, 12.95 bp a trade in all.
- Gross is negative on both sides (verdict): long −2.94 bp, short −2.06 bp.

**`lce_atr` has a small positive gross, and costs swamp it.**
- Gross is +2.42 bp per trade. The before-costs Sharpe is +0.67, against a random p95 of +0.49,
  at the 97.5th percentile of the random runs (verdict).
- Fees (8.62 bp), slippage (5.26 bp) and funding (0.15 bp) add up to 14.03 bp a trade, so net is
  −11.61 bp.
- The positive gross is all on the short side (verdict): short +7.55 bp, long −2.56 bp.
- **It is statistically weak and comes from a few days** (section 3, "By date"):
  - The before-costs Sharpe of 0.67 over 2,191 days is a t of about 1.64, from one comparison
    among N = 9 trials, with no deflation applied.
  - The gross per trade has a t of 1.19 when clustered by date.
  - Each date weighted equally gives −3.95 bp.
- By the regime at the signal (verdict), gross is:

  | regime | gross, bp | trades |
  |---|---|---|
  | ranging | +5.19 | 2,444 |
  | transition | +6.02 | 1,912 |
  | trending | +1.37 | 4,579 |
  | high volatility | −1.44 | 2,374 |

Two properties of the rule, for the record:
- The order of its tests sends "net beats random, before costs doesn't" to "no signal". That case
  didn't arise here.
- The lab's before-costs pass keeps funding; the spec's "no signal" line said zero funding. Funding
  is 0.01 and 0.15 bp a trade here, so the difference is immaterial (see section 6, gap 12).

## 3. Per branch

**Method (spec).** Each plugin is rerun at its **modal params** with features on, one coin at a
time, in a sealed sandbox worker on research bars, and `sig_branch` is joined at each OOS trade's
`signal_time`.
- The modal params are the combo selected in the most folds, as
  `walkforward.modal_params` computes it. For both plugins that is `theta = 0.02, vp_days = 5`,
  which equals each run's sensitivity base.
- For `lce_atr` that is a tie:
  - `vp_days = 5` and `vp_days = 10` (both at `theta = 0.02`) were each selected in 17 folds.
  - `modal_params` takes the first combo `value_counts` lists, and `value_counts` keeps tied combos
    in their order of first appearance.
  - Fold 0 selected `vp_days = 5`, so the tie went to it. The judge's sensitivity check and
    random-entry benchmark used the same params.
- Modal-param branches can be wrong for trades from folds that picked another combo: 94.0%
  (`lce_pct`) and 73.2% (`lce_atr`) of those trades get the same branch. So `attribution.py` also
  attributes **each trade under its own fold's combo**, rerunning every combo some fold selected.
  That cross-check is exact: under its own combo, every one of the 36,989 trades' signal bars has a
  branch on the trade's side.

| | `lce_pct` | `lce_atr` |
|---|---|---|
| trades from folds that selected the modal combo | 19,744 (77.0%; 28 of 36 folds) | 5,139 (45.3%; 17 of 36 folds) |
| trades from other folds | 5,908 (23.0%) | 6,198 (54.7%) |
| signal bars with no branch under the modal params | 349 (1.4%) | 1,651 (14.6%) |
| same branch under the modal params as under the trade's own combo | 98.6% | 85.4% |

For `lce_pct` the two attributions nearly agree; the table below uses the modal params. For
`lce_atr` the modal attribution leaves 14.6% of the trades unassigned, so read the exact one.

The branches (spec):
1. HTF bullish, LTF bullish: buy the dip into the cloud at a zone.
2. HTF bullish: a bullish 5m flip after a touch.
3. HTF ranging, LTF was bullish: a bearish flip, short.
4. HTF ranging, LTF was bearish: a bullish flip, long.
5. HTF bearish: a bearish flip, short.
6. HTF bearish, LTF bearish: sell the rip into the cloud at a zone.

In the tables, R:R p50 is the median planned R:R at the fill (section 4), and the last five columns
are bp per trade.

**`lce_pct`, modal params:**

| branch | trades | share | win | target | mean R | R:R p50 | gross | fees | slippage | funding | net |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 9,698 | 0.378 | 35.6% | 58.6% | −0.460 | 0.41 | −2.81 | 8.24 | 4.77 | 0.01 | −15.83 |
| 2 | 2,146 | 0.084 | 34.6% | 57.7% | −0.471 | 0.41 | −3.25 | 8.26 | 4.86 | 0.03 | −16.41 |
| 3 | 43 | 0.002 | 34.9% | 60.5% | −0.512 | 0.30 | −2.39 | 8.19 | 6.14 | 0.06 | −16.78 |
| 4 | 42 | 0.002 | 28.6% | 47.6% | −0.584 | 0.35 | −5.92 | 8.56 | 4.79 | 0.03 | −19.30 |
| 5 | 2,324 | 0.091 | 34.1% | 58.6% | −0.450 | 0.39 | −3.51 | 8.25 | 4.89 | −0.01 | −16.64 |
| 6 | 11,050 | 0.431 | 38.2% | 61.4% | −0.403 | 0.42 | −1.82 | 8.16 | 4.65 | −0.00 | −14.63 |
| none | 349 | 0.014 | 35.8% | 57.6% | −0.438 | 0.44 | −1.83 | 8.27 | 4.99 | 0.00 | −15.09 |
| total | 25,652 | 1 | 36.5% | 59.7% | −0.435 | 0.41 | −2.47 | 8.21 | 4.74 | 0.01 | −15.43 |

No branch of `lce_pct` makes money before costs. Under the exact attribution, the four branches with
more than 2,000 trades have gross from −1.78 bp (branch 6) to −3.42 bp (branch 5).

**`lce_atr`, each trade under its own fold's combo (exact):**

| branch | trades | share | win | target | mean R | R:R p50 | gross | fees | slippage | funding | net |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5,058 | 0.446 | 44.6% | 44.9% | −0.102 | 0.64 | −1.07 | 8.64 | 5.31 | 0.32 | −15.33 |
| 2 | 668 | 0.059 | 44.9% | 48.7% | −0.159 | 0.54 | −14.03 | 8.52 | 5.41 | 0.34 | −28.30 |
| 3 | 17 | 0.001 | 58.8% | 47.1% | −0.124 | 0.60 | 24.60 | 8.60 | 4.87 | −0.08 | 11.21 |
| 4 | 24 | 0.002 | 45.8% | 50.0% | −0.093 | 0.52 | 1.70 | 8.48 | 4.54 | 0.35 | −11.67 |
| 5 | 630 | 0.056 | 44.3% | 45.2% | −0.126 | 0.62 | −12.57 | 8.66 | 5.05 | 0.04 | −26.33 |
| 6 | 4,940 | 0.436 | 45.9% | 46.4% | −0.025 | 0.65 | 10.06 | 8.62 | 5.22 | −0.04 | −3.75 |
| total | 11,337 | 1 | 45.2% | 45.8% | −0.073 | 0.64 | 2.42 | 8.62 | 5.26 | 0.15 | −11.61 |

The modal-params table for `lce_atr` (in `attribution.out`) tells the same story less sharply:
- branch 6: +9.80 bp gross;
- branch 1: −2.35 bp;
- 1,651 trades with no branch: +8.20 bp.

**Branch 6 carries `lce_atr`'s positive gross.** Branch 6 sells the rip into the cloud at a zone,
with both clouds bearish: +10.06 bp over 4,940 trades. Against it:
- branch 1, its long mirror, is −1.07 bp over 5,058 trades;
- the trend-flip branches 2 and 5 lose 14.03 and 12.57 bp gross;
- branches 3 and 4 have 41 trades between them, too few to read.

Branch 3 is the only branch that is net positive, on 17 trades. Branch 6 nets −3.75 bp.

The positive gross is not stable across folds or years (attribution):

| folds that selected | folds | trades | gross, bp | net, bp | branch 6 gross, bp |
|---|---|---|---|---|---|
| `theta = 0.02, vp_days = 5` (modal) | 17 | 5,139 | −2.76 | −16.50 | +1.80 |
| `theta = 0.02, vp_days = 10` | 17 | 5,044 | +5.40 | −8.56 | +16.97 |
| `theta = 0.01, vp_days = 5` | 2 | 1,154 | +12.50 | −3.18 | +13.80 |

Each fold covers its own two months, so the combo and the calendar are confounded here.

By year of the signal, gross runs:

| year | 2020 (from July) | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (to June) |
|---|---|---|---|---|---|---|---|
| gross, bp | +8.62 | +3.71 | −7.73 | +5.14 | +5.06 | −0.38 | +3.99 |

Net is negative in every year, from −8.65 to −21.31 bp.

**By date** (attribution, "how concentrated the gross is"). Each trade is dated by the New York date
of its signal. "Clustered t" treats the trades of one date as one draw, because the 17 coins trade
the same session. The market's move is the equal-weight mean, over the coins, of
close(15:55) / close(09:40) − 1. Its rank is among the 1,565 weekdays of the OOS span, where 1 is
the worst.

| | `lce_atr`, all trades | `lce_atr`, branch 6 | `lce_pct`, all trades |
|---|---|---|---|
| trades, dates | 11,337 on 1,438 | 4,940 on 766 | 25,652 on 1,551 |
| gross per trade, bp (t clustered by date) | +2.42 (1.19) | +10.06 (2.86) | −2.47 (−11.04) |
| each date weighted equally, bp (t) | −3.95 (−1.92) | −4.50 (−1.37) | −2.18 (−9.39) |
| without the top 10 dates | +0.06 | +5.02 | −2.60 |
| without the top 20 dates | −1.58 | +1.58 | −2.72 |
| trades on the top 10 dates; short share | 174; 98.9% | 175; 100.0% | 167; 76.0% |

What it shows:
- **`lce_atr`'s ten best dates carry almost all of its gross.** Their shares of the summed gross
  are 20.3%, 10.3%, 10.1%, 9.2%, 8.2%, 8.1%, 8.1%, 8.0%, 7.8% and 7.6%. Every one is a session in
  which 15 to 17 of the 17 coins fell, and nine of the ten rank among the worst 160 of 1,565:

  | date | market's move | rank (of 1,565) | note |
  |---|---|---|---|
  | 2021-05-21 | −15.64% | 1 | |
  | 2021-06-18 | −4.96% | 55 | |
  | 2022-11-09 | −6.77% | 16 | the FTX collapse |
  | 2022-05-09 | −10.25% | 6 | the LUNA collapse |
  | 2021-05-13 | −5.50% | 40 | |
  | 2024-12-19 | −9.18% | 7 | |
  | 2025-10-30 | −2.85% | 153 | |
  | 2024-12-10 | −1.12% | 427 | |
  | 2026-06-05 | −3.70% | 86 | |
  | 2025-03-10 | −6.66% | 18 | |

  On those days the plugin took 12 to 22 trades across the coins, 94% to 100% of them short. That
  is one correlated bet on a falling market, not that many independent trades.
- **On a typical date `lce_atr` loses before costs.** Weighting each date equally gives −3.95 bp.
- **Branch 6's +10.06 bp follows the same pattern.** Its clustered t is 2.86, but weighting each
  date equally gives −4.50 bp, and its top 10 dates include seven of the ten above. It is also one
  of six branches, singled out after the result.
- **`lce_pct`'s loss is broad, not a few bad days.** It loses on a typical date (−2.18 bp, t −9.39),
  and removing its best dates makes the gross worse (−2.60, −2.72). Its top dates hold modest sums
  (328 bp at most) and include ordinary market days.

## 4. Against his claims

| | his claim | `lce_pct` | `lce_atr` |
|---|---|---|---|
| reward to risk | 1.5–2R | planned R:R at the signal close: median 0.558, 3.7% of trades ≥ 1.5. At the fill: median 0.412 | at the close: median 0.666, 5.4% ≥ 1.5. At the fill: median 0.636 |
| R realised | | mean −0.435 R. Target exits +0.160 R, stop exits −1.327 R | mean −0.073 R. Target exits +0.435 R, stop exits −0.736 R |
| win rate | "85%" | 36.5% (target exits 59.7%) | 45.2% (target exits 45.8%) |
| trades a day | at most 2 a day in total; stop after a win; usually done before noon | 0.976 per coin per Mon–Fri coin-day; 1.325 on coin-days with a trade (most 2); 16.39 per Mon–Fri date across the 17 coins; 64.6% of fills before 12:00 New York | 0.431; 1.135 (most 2); 7.24; 53.9% |

The spec quotes "85%" as his claim and sets it against the win rate, so it is read here as a win
rate. Even the share of trades that reach their target, which is the most generous reading of a
"win", is 59.7% and 45.8%.

**Planned R:R.** The planned R:R is (target − entry) / (entry − stop), signed by the side, from each
trade's own prices.
- At the signal bar's close, it is the bracket's own plan.
- At the fill, it comes after the gap to the next open and the entry slippage. The ratio is signed:
  10.3% of `lce_pct`'s trades (2,632) and 3.1% of `lce_atr`'s (350) filled at or past their target.
- Below 1.0 at the close: 83.6% of `lce_pct`'s trades and 77.9% of `lce_atr`'s. Below 0.5 at the
  fill: 59.4% and 33.5%.

The bracket's geometry allows this. The arithmetic below uses the spec's constants. For a long with
levels one step (0.303 u) apart and zones 0.103 u wide:
- A close on the base level's centre gives:
  - a target 0.303 − 0.0515 = 0.2515 u away (the next zone's lower edge);
  - a stop 0.1515 u away (halfway to the level below);
  - so R:R 1.66.
- As the close rises toward the next zone, the target distance shrinks to 0 and the stop distance
  grows to about 0.40 u.

The spec says "a stop about half a step away and a target about one step, which matches the 1.5–2R
he states". That holds only for a close on the base level. The bracket requires only
stop < close < target, so a close anywhere in the gap is taken.

An example is `lce_pct`'s first BTC trade (attribution), a branch 1 long whose signal bar opened at
2020-07-01 16:00 UTC:
- it closed at 9,242.43 and filled at 9,244.63;
- the target was 5.38 above the fill and the stop 31.40 below: planned R:R 0.171 at the fill
  (0.260 at the close);
- it exited at the target and still lost: R −0.035.

## 5. Levels

Most of these come from the pre-trial looks at the defaults. The levels don't depend on `theta`,
and `vp_days = 5` is the modal value for both plugins, so these are the modal params' levels. The
spacing quantiles come from `attribution.py`'s modal reruns, which reproduce the looks' medians.

| | `lce_pct` | `lce_atr` |
|---|---|---|
| weekday session days with levels on the anchor fallback (no POC within ±zone/2 of any HVN) | 1,067 of 27,501 (0.039); per coin 0.008 (BTC) to 0.068 (DOGE) | **0 of 27,327** |
| the day's median level spacing, in u (step 0.303): median over coin session days (looks); p10, p25, p50, p75, p90 (attribution, 38,495 and 38,252 days) | 0.279 (per coin 0.273–0.282); 0.250, 0.261, 0.279, 0.301, 0.320 | 0.271 (per coin 0.268–0.273); 0.247, 0.255, 0.271, 0.292, 0.313 |
| signals not on a big bar (body > 0.303 u) | 0.810 | 0.997 |
| signals that are not big and have a valid bracket | 161,026 of 201,112 (0.801) | 86,218 of 86,538 (0.996) |
| in-session OOS bars with a valid bracket (looks_bracket), long / short | 0.938 / 0.937 pooled; lowest 0.917 / 0.915 (SOL) | 0.992 / 0.992 pooled; lowest 0.977 / 0.977 (FIL) |
| in-session OOS bars on a day without levels | 0.002 | 0.006 |

How to read it:
- **The anchor.** `lce_atr`'s anchor never falls back. Its zones are 0.103 × the coin's daily ATR
  wide, where BTC's ATR is about 3% of price (spec), against 0.103 × 1.169% in `lce_pct`. The POC
  tolerance is half a zone, so some HVN nearly always has a POC in reach. Task 9 checked that this
  comes from the constants and is not a bug. It means the fallback path never ran in trial 9.
- **The spacing.** Its middle 80% (p10 to p90) lies inside the spec's search window,
  [0.8, 1.2] × step = 0.242–0.364 u. It is below the 0.303 step on at least three quarters of the
  days (p75 0.301 and 0.292).
- **The invalid brackets.** `lce_pct`'s in-session bars without a valid bracket are mostly closes
  beyond the outermost level. Its big-bar rule blocks 19% of its signals (0.810 not big): bars
  whose body is over 0.303 u, about 0.35% of price.

## 6. Where this differs from Tradewriter, and what could hide an edge

The spec lists seven differences (gaps 1–7). The trials and the attribution found seven more (gaps
8–14). "Could it hide an edge?" asks whether coding the model closer to how he trades could
plausibly turn these results into a pass.

| # | difference | what these runs show | could it hide an edge? |
|---|---|---|---|
| 1 | **Levels** are redrawn daily from the last 5–10 days of 5m volume. His are drawn by eye from "tests and rejections" and kept for weeks. | The levels behave as specified (section 5). | **Plausibly.** The levels are the core of the model, and ours are a mechanical stand-in. But his published levels are drawn with the later price already on screen, so no point-in-time test can reproduce them. |
| 2 | **Cloud states** use one slope threshold θ (two values on the grid). No "mature" or "healthy" judgement, no Goldilocks distance. | `theta = 0.02` was selected in 34 of 36 folds for both plugins. Every `lce_pct` branch is negative gross. | Unlikely to hide a large one. |
| 3 | **Entries** are market orders at the next 5m open. He uses alerts, sometimes limits. | 10.3% of `lce_pct`'s fills (3.1% of `lce_atr`'s) were at or past the target, and lost 5.49 bp (1.06 bp) gross. The median planned R:R falls from 0.558 at the close to 0.412 at the fill (0.666 to 0.636). | It costs `lce_pct` some of its gross, but not all: its buckets above R:R 0 are negative too, up to R:R 2 (below). Limit entries would also miss trades. |
| 4 | **No A+/B grading**, and none of the flowchart's "additional context". | Not testable. | **Plausibly.** Discretion can select trades, but as taught it can't be falsified. |
| 5 | **Crypto has no 9:30 open auction.** The window is clock time only. | 64.6% / 53.9% of fills came before noon. | **Plausibly.** ES's open dynamics don't exist on crypto. This is what the ES leg tests. |
| 6 | **Volume profiles** come from 5m bars, not intrabar data. | The spacing lands near the step (median 0.279 / 0.271 u); the anchor falls back on 3.9% / 0% of days. | Unlikely on its own. |
| 7 | **Several coins.** He takes at most 2 trades a day in all; we allow 2 a day per coin, across 17 coins. | 16.39 and 7.24 trades per weekday across the book, on average; 0 of 17 coins net positive in either trial. | No. It changes the count and the weights, not the bp per trade, and no coin was positive. |
| 8 | Found: **"at a level" ignores the side.** Branches 1 and 6 need `at_zone`: the bar overlaps the zone just below the close *or* the one just above. "Buy dips into the cloud at demand" means the zone on the stop side. A long can fire on a bar that touched only the supply above it. Its target is then that zone's lower edge when the close is below the zone, or the next zone's lower edge when the close is inside it. Branches 2–5's `touched` ignores the side in the same way (not measured here). | Among branch 1 and 6 trades: far zone only 23.7% for `lce_pct` and 47.3% for `lce_atr`; both zones 39.5% and 1.8%; near zone only 36.8% and 50.9%. Median planned R:R is 0.27 for far only and 0.56 for near only (`lce_pct`); 0.52 and 0.87 (`lce_atr`). | **Not on this evidence.** The trades that touched the zone the flowchart means (near, with or without far: 76.3% of `lce_pct`'s branch 1 and 6 trades, 52.7% of `lce_atr`'s) do no better before costs than the far-only ones: −2.69 against −0.99 bp (`lce_pct`), +2.48 against +6.60 bp (`lce_atr`). |
| 9 | Found: **no minimum room to the target.** The bracket needs only stop < close < target, so trades with almost no room to the next zone are taken. He states 1.5–2R and would skip them. | See the table below. | **Not on crypto.** No slice by planned R:R is net positive. In `lce_pct`, gross rises with R:R but stays below costs. A gross of a few bp would matter at ES's costs, so the ES spec should state a rule here before any ES result. |
| 10 | Found: **the 3×-equity notional cap.** A trade whose stop is closer than a third of a percent of price can't risk the full 1% of equity; he risks the same on every trade. | Risk at entry was under 0.99% of equity on 34.7% of `lce_pct`'s trades (p10 0.721%) and 0.0% of `lce_atr`'s. | No. It changes the weights in the equity curve, not the bp per trade. |
| 11 | Found: **the ±10/20% sensitivity tests fewer knobs than it lists.** `max_trades = 2` rounds back to 2, so it has no rows. `atr_n` doesn't enter `lce_pct`: its four rows all equal the base Sharpe, −17.39 (summary.json's sensitivity rows, in the git-ignored run directory). `vp_days` and `wait_bars` get two distinct values each, and in `lce_pct` the `wait_bars = 7` row also equals the base exactly. | The island test was n/a for both, since the base Sharpe was ≤ 0. | No. It limits the robustness check, which didn't decide anything here. |
| 12 | Found: **funding is paid.** Binance's 16:00 UTC settlement is 11:00 or 12:00 New York, inside the session; the plan ([`docs/superpowers/plans/2026-10-23-lce-model.md`](../../superpowers/plans/2026-10-23-lce-model.md)) had called it immaterial for trades flat by 15:55. | Nonzero funding on 1.5% / 17.8% of trades; 0.01 / 0.15 bp a trade. | No. |
| 13 | Found: **the random benchmark takes fewer trades** than the strategy, because it fires on every in-session flat bar but can enter only where the bracket is valid. Accepted by the user on 2026-10-27 (spec, "Known residual"). | Ratios 0.934 and 0.989. With costs, fewer random trades make the benchmark harder to beat, and both plugins beat it anyway. | No. The kills come from Sharpe ≤ 0 and the Deflated Sharpe, not from the benchmark. |
| 14 | Found (final code review): **the plugin's win reconstruction leaves out liquidation.** This is safe while the daily ATR stays under about 58% of price. | 0 liquidations in either run (verdict). | No. |

**Planned R:R at the fill against outcomes** (gap 9; attribution). The last three columns are bp per
trade.

| plugin | R:R bucket | trades | win | target | mean R | gross | net |
|---|---|---|---|---|---|---|---|
| `lce_pct` | ≤ 0 | 2,632 | 0.0% | 79.8% | −0.374 | −5.49 | −18.08 |
| | 0–0.5 | 12,617 | 36.2% | 66.8% | −0.396 | −3.05 | −15.63 |
| | 0.5–1 | 8,471 | 48.3% | 48.3% | −0.461 | −1.12 | −14.56 |
| | 1–2 | 1,926 | 36.0% | 35.9% | −0.665 | −0.52 | −14.38 |
| | ≥ 2 | 6 | 33.3% | 33.3% | −0.572 | 2.91 | −11.34 |
| `lce_atr` | ≤ 0 | 350 | 0.0% | 92.3% | −0.058 | −1.06 | −12.82 |
| | 0–0.5 | 3,445 | 57.4% | 70.5% | −0.066 | −1.51 | −13.94 |
| | 0.5–1 | 5,350 | 44.1% | 34.3% | −0.053 | 6.94 | −7.90 |
| | 1–2 | 2,170 | 35.8% | 27.8% | −0.138 | −1.86 | −16.81 |
| | ≥ 2 | 22 | 36.4% | 18.2% | −0.073 | −2.99 | −19.07 |

This table is a diagnostic, not a rule. Selecting trades by planned R:R, by side, by branch or by
which zone was touched, after seeing these tables, would be a new hypothesis and a new trial
(anti-goal 1). That is the user's call.

**Which differences could hide an edge.** Only the levels (1), the discretion (4) and the ES open
(5) plausibly could. None of them can be tested on crypto without a new spec and a new trial, and
the first two can't be coded faithfully at all. The entries (3) and the missing room-to-target rule
(9) shape the bracket's economics. They would matter more at ES's costs than at crypto's. Every
other difference either doesn't touch the bp per trade, or was checked here and showed no better
subset.

## 7. The looks before the trials

The rules for these looks are in the spec ("Looks at real data before the trials"):
- no forward returns, P&L, hit rate or R;
- research bars only: `load_bars` stops at 2026-07-01. `looks.py` and `looks_bracket.py` also
  assert it; `perf.py` relies on `load_bars` alone, and computes only timing;
- plugin code only inside the sandbox.

No look found a bug, so no rule, constant or plugin line changed after them. The plugin hashes in
`looks.out`, `looks_bracket.out` and `perf.out` are the judged ones (`5458edefef10`,
`181835ba5477`).

| look | what it computed | why | what it found |
|---|---|---|---|
| `lab preflight --tf 5m` ([`preflight_lce_pct.json`](preflight_lce_pct.json), [`preflight_lce_atr.json`](preflight_lce_atr.json)) | The lab's metadata, lookahead, overlap and mirror checks, on the synthetic 5m market. | The pre-trial check the plan requires. Synthetic bars only. | `ok: true` for both, with every list empty. |
| [`perf.py`](perf.py) → [`perf.out`](perf.out) | The median CPU time of 5 `prepare` calls in a sealed worker, on BTCUSDT's research 5m bars at `vp_days = 10`, under two CPU placements. | The spec's performance budget (0.30 s). Timing only: no outcome is computed. | `lce_pct` 0.640 s (scheduler's placement) / 0.630 s (performance cores); `lce_atr` 0.600 / 0.610 s. Over budget. **The user decided on 2026-10-23 to continue**; the exam budget is settled only if LCE is promoted, which it wasn't. |
| [`looks.py`](looks.py) → [`looks.out`](looks.out) | Per coin and plugin, at the defaults: signals per branch in the window; the share not on big bars; the valid-bracket count and share; the anchor-fallback share; the median spacing; the Mon–Fri dates without a bar closing at 15:55. | The spec's allowed list: branch counts, the valid-bracket share, the anchor fallback and spacing, 15:55 bars, listings and gaps. **The extra columns** (bars, in-window bars, "small": the share not on big bars) let the valid share be read. **Two clock checks** were a look for bugs: `sday` equals the trusted `ny_session_day`, and `in_window` equals the trusted `in_session`, on every bar. | Every branch fires on every coin; the rarest is `lce_atr`'s branch 4, with 1 signal each on SOL and AVAX. Both clock checks pass on every bar. 10 coin-dates have no 15:55 bar: 2022-02-28 and 2022-04-01 on SOL, XRP, LTC, FIL and XLM, the store's known candle gaps, where the position exits at the first bar after. |
| [`looks_bracket.py`](looks_bracket.py) → [`looks_bracket.out`](looks_bracket.out) | Per coin: the share of in-session OOS bars with a valid long bracket, a valid short one, and both; and the share on days without levels. | The final whole-branch review found that the session-calibrated random benchmark can enter only where the bracket is valid, so it takes fewer trades than the strategy. This is the spec's allowed "share with a valid bracket". | 0.938 / 0.992 pooled. `lce_pct` was under the 0.95 stop line on 16 of 17 coins, so the question went to the user, who accepted and recorded the shortfall on 2026-10-27. The trials then measured it: 0.934 and 0.989 (section 1). |

A features run over a coin's whole research history returns about 150 MB, over the configured
`sandbox.max_output_mb` of 64 MB. So `looks.py`, `looks_bracket.py` and, after the verdicts,
`attribution.py` each **raise that bound in memory to 512 MB for their own workers only**. Nothing
else in the config changes, and the workers stay sealed under the configured memory and CPU limits.

`checks.py` is not a look: it reads only the judged run. It was committed before trial 8 in
691058a and fixed in e1e3417 (a missing benchmark prints n/a, never a kill kind). 66d433f added the
random trade-count line. Its output on both runs is in 7a60077.

## 8. What the ES leg needs next

The spec leaves ES/SPX to its own spec ("ES leg"). It needs:
- **ES data with real exchange volume at 1m or 5m.** The levels are volume profiles, so CFD
  tick-count feeds won't do. The user is looking for a source.
- **An ES bar loader.**
- **An ES cost model:** commission plus 1 tick.
- **The CME calendar:** holidays and early closes.
- **A third, literal "points" translation:** u = 82.4 points, fixed.

It would run the same plugin code, so the comparison is fair.

**The translations on ES.** The three translations start close together on ES. The spec's
calibration gives ES's median ATR(14) as 82.4 points and its median close as 6,923, and states
u = 1.169% of price.
- 1.169% is not 82.4 / 6,923, which is 1.19%. It is presumably the median of the daily ATR / close
  ratio, which need not equal the ratio of the two medians.
- So at ES's median close, `lce_pct`'s u (0.01169 × 6,923 ≈ 80.9 points) is about 2% smaller than
  the literal translation's 82.4 points.
- `lce_atr`'s u follows ES's own ATR from day to day.

They differ by that 2%, and by how far ES's price and volatility move from the medians. So the ES
leg tests the model more than the translation.

**What trial 9 says about the ES leg.** An ES round trip costs about 0.5 bp, against about 14 bp a
trade on Binance:
- one tick of 0.25 points at the median close of 6,923 is 0.36 bp;
- commission adds roughly 0.1 bp on a full contract, and more per dollar on the micros he trades.

The ES spec fixes the exact number. At first sight, `lce_atr`'s +2.42 bp gross would clear that
several times over. **But these trials give little basis for that hypothesis:**
- **It is weak statistically.** The gross per trade has a t of 1.19 when clustered by date. The
  before-costs Sharpe is a t of about 1.64: one comparison among N = 9 trials, not deflated.
  Weighting each date equally, the gross is −3.95 bp.
- **It comes from a few crypto crash days.** Without its ten best dates it is +0.06 bp. On those
  dates the plugin was short most of the 17 coins at once, into market-wide crypto sell-offs (FTX,
  LUNA and others; section 3, "By date"). That is correlated short exposure across coins, not
  timing skill trade by trade. ES has neither the 17-coin stack nor these crypto-specific crashes.
- **It rests on one branch and one side.** Branch 6 and the shorts carry it. It is positive in the
  folds that selected `vp_days = 10` or `theta = 0.01`, −2.76 bp in the modal folds, and negative
  in 2022 and 2025.
- **The other translation lost.** `lce_pct` lost before costs, broadly: t −11.04 by date.

So "LCE has a small gross edge that ES's low costs would let through" has little support here. The
ES leg can still answer the question the spec asked: whether LCE has a signal on the market it was
built for. It should be argued from that question, not from trial 9. It is new trials, counted in
N. Its rules should be fixed before any ES result is seen, including whether to require room to the
target (gap 9).

## 9. What this ends

Under the spec's anti-goals:
- Both plugins are killed, and LCE does not go into the crypto book.
- No constant, threshold, branch rule, session time or bracket rule changes after these results.
- A re-test on crypto would need a new spec and a new trial, and that is the user's call. That
  covers a side-aware `at_zone`, a minimum R:R, shorts only, branch 6 only, or `vp_days = 10` only.
- The holdout was not read, and the family `lce_pct` keeps its unused holdout read.

What it teaches: on these 17 coins over 2020-07 → 2026-07, Tradewriter's flowchart, coded as
taught, has no edge net of Binance's costs.
- In fixed-percentage units (`lce_pct`), its brackets lose even before costs. They take many trades
  with almost no room to the target.
- In daily-ATR units (`lce_atr`), it shows a small positive gross: 2.42 bp, about a sixth of the
  14.03 bp a trade costs. That gross comes from short trades on a handful of market-wide sell-off
  days. On a typical date it loses even before costs.
