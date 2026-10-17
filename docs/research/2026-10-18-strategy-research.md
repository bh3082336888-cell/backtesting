# Strategy research, 2026-10-18: one hypothesis to implement

Status: research note, written by us without an outside brief. Section 5 is the spec that the
next task implements, as written. Implementing the plugin spends no trial. Running its
walk-forward does: N goes from 6 to 7 store-wide, so that run is the user's call.

How this was done:
- No internet of any kind. Everything below comes from the repo, from memory (each citation is
  marked "from memory"), and from market data read only through `perp_lab.data.store.load_bars`
  with the research config. Every read ended at the holdout start (2026-07-01).
- No candidate's forward returns, P&L or hit rate were measured. The looks at the data were
  limited to bar ranges, unconditional move sizes, how often conditions occur, feature
  distributions, funding regimes and data gaps. Section 7 lists each one, with the reason it is
  not a backtest.

**Outcome.** Implement `flow_shock_fade` at **1h**. It fades an hour of extreme one-sided taker
flow that moved price. Its exits are chosen so that random entries earn nothing from them, so
only the entry can beat the random benchmark. We think it is more likely to be killed than
promoted. Section 4 says why we chose it anyway, and section 6 says which results should end
this line of research.

## 1. What the six baselines teach

The numbers come from the walk-forward table in the plan's constraint 5 (OOS 2022-07 to
2026-07, 8 symbols, 10,000 USDT per symbol book). Net = gross − fees − funding. Gross already
includes slippage, because slippage is in the fill prices; fees are exchange fees only.

| plugin | tf | trades | gross | fees | funding | net | fees per trade | fees vs gross | OOS (IS) Sharpe | vs random |
|---|---|---|---|---|---|---|---|---|---|---|
| donchian_atr | 4h | 1390 | +13,060 | 3,102 | 1,159 | +8,799 | 2.23 | 24% of gross | 0.45 (1.21) | p48 |
| donchian_atr_wide | 4h | 1160 | +17,449 | 1,814 | 1,314 | +14,321 | 1.56 | 10% of gross | 0.81 (1.16) | p77 |
| donchian_atr | 1h | 4793 | −372 | 15,362 | 1,669 | −17,403 | 3.21 | 41× the gross loss | −0.46 (0.63) | p49 |
| bb_vwap_reversion | 15m | 8435 | −21,466 | 26,366 | −165 | −47,667 | 3.13 | 1.2× the gross loss | −4.30 (−4.02) | — |
| bb_vwap_reversion | 1h | 3759 | −21,548 | 10,584 | −621 | −31,511 | 2.82 | 0.5× the gross loss | −2.18 (−1.83) | p72 |
| funding_fade | 1h | 542 | −621 | 1,293 | −572 | −1,342 | 2.39 | 2.1× the gross loss | −0.30 (1.35) | p68 |

1. **Costs per timeframe.** At 4h, fees were 10–24% of gross, and funding took another 7–9%
   because trend trades are held for days. Costs did not kill the 4h runs. At 1h the same trend
   rule had no gross edge (−372), and 15,362 of fees turned that into −17.4k. At 15m,
   `bb_vwap_reversion` paid 26k of fees on top of a 21k gross loss. In dollars, fees were
   1.6–3.2 per trade on books that risk about 100 per trade. The unit that matters is R: section 2
   shows that a round trip with a 2-ATR stop costs 0.135 R at 15m, 0.064 R at 1h and 0.030 R at 4h.
2. **Profit came from the exit, not the entry.** Both 4h Donchian runs made money gross, but they
   sat at p48 and p77 of the random-entry Sharpe distribution, and those random entries used the
   same ATR chandelier exit. Coin-flip entries with that trail earned about as much. In a market
   that trends at the scale of a 3–4 ATR trail on 4h bars, a trailing stop earns whatever the
   entry direction, and the breakout timing added nothing. The lesson for any new candidate:
   - Its exits should earn about zero on random entries: a fixed time exit, a wide stop, no
     trail. Then the only way to beat random p95 is entry timing or direction.
   - Its exits must not depend on the trigger bar. A target "back at the shock bar's open" would
     give random entries different and worse exits, and make the benchmark easy to beat for the
     wrong reason.
3. **Fading a stretched price had no gross edge.** `bb_vwap_reversion` lost 21.5k gross at both
   15m and 1h, before costs. At 1h it still beat 72% of random entries, which means its VWAP-target
   exit loses on random entries too. Any new reversion candidate has to say why its trigger is
   more than "price is stretched".
4. **Regime behaviour.** `donchian_atr` 4h lost in `high_vol` (average R −0.135), and
   `donchian_atr_wide` earned in `trending` (average R +0.283). Trend rules pay in trends and
   bleed in high-volatility chop. A candidate that trades in high-volatility moments is a
   different exposure. But a regime filter added now would be fitted to these very numbers, so
   we add none.
5. **In-sample overfitting.** Every IS Sharpe exceeded its OOS Sharpe:
   - by 0.35 for `donchian_atr_wide` (3 combos);
   - by 0.76 for `donchian_atr` 4h (9 combos);
   - by 1.09 for `donchian_atr` 1h;
   - by 1.65 for `funding_fade` (12 combos on a rare trigger, 542 trades, flagged as a narrow
     island).

   The rarer the trigger and the bigger the grid, the more the 6-month training windows picked
   noise. What follows for us: a small grid, plenty of trades per training window even at the
   strictest grid point, and thresholds on a continuous statistic, so that a ±20% move changes
   the trade set gradually.
6. **The bar.** Buy and hold had a Sharpe of 0.59 with a −67% drawdown, so a net-long rule rides
   that drift. For normal returns over a 4-year OOS track, the Deflated Sharpe gate needs about
   1.47 annualised at N = 6 and about 1.52 at N = 7, which this plugin's run would make it. That
   is `perp_lab.judge.gates.expected_max_sharpe` plus the 0.95 quantile; fat tails raise the bar
   further. None of the six came within 0.6 of it.

## 2. Cost hurdle by timeframe

Costs come from the engine's fill rules:
- A market entry costs 7 bp (5 bp taker plus 2 bp slippage), and so does a stop, signal or time
  exit.
- A target exit costs 2 bp (maker, no slippage).
- A round trip is therefore **14 bp** when it exits on a stop or on time, and **9 bp** when it
  exits on a target.
- Funding: the median settled rate was 0.6–1.0 bp per 8 h on six symbols, 0.47 bp on SOL and
  −0.45 bp on BNB. A long paid 0.8–2.2 bp a day on those six and was paid about 1.5 bp a day on
  SOL and BNB (look 2). Over a hold of a day or less that is a few bp at most, which is small next
  to taker costs. Over multi-day holds it adds up: the Donchian runs paid 7–9% of their gross.

The median over the 8 symbols of each symbol's own median, in bp, over every bar from 2022-01 to
2026-07 (look 1). |move h| is the median absolute log move from bar t's close to bar t+h's close,
over all bars, with no condition.

| tf | bar range | ATR14 | \|move 1\| | \|move 2\| | \|move 4\| | \|move 8\| | \|move 12\| | \|move 20\| |
|---|---|---|---|---|---|---|---|---|
| 15m | 46 | 52 | 19 | 26 | 36 | 50 | 62 | 82 |
| 1h | 95 | 109 | 36 | 51 | 73 | 106 | 134 | 181 |
| 4h | 199 | 232 | 74 | 107 | 158 | 230 | 279 | 378 |

The table below sets a 14 bp round trip against those sizes. "R cost" is the round trip divided
by the stop distance, which is what a trade sized to risk 1% pays.

| tf | cost / bar range | R cost, 1 / 2 / 3 ATR stop | cost / \|move\| at 1, 4, 8, 20 bars |
|---|---|---|---|
| 15m | 0.31 | 0.27 / 0.135 / 0.090 | 0.74 / 0.39 / 0.28 / 0.17 |
| 1h | 0.15 | 0.128 / 0.064 / 0.043 | 0.39 / 0.19 / 0.13 / 0.08 |
| 4h | 0.07 | 0.060 / 0.030 / 0.020 | 0.19 / 0.09 / 0.06 / 0.04 |

Scale differs by symbol: a 1h ATR is 67 bp on BTC and 123–128 bp on SOL, AVAX and LINK, so BTC
pays nearly twice as much per R as the alts.

What this says, by timeframe:
- **15m cannot pay.** A round trip is a third of a median bar's range and 17% of a median
  20-bar (5-hour) move. With a 2-ATR stop every trade starts 0.135 R down, so a rule needs about
  0.14 R gross per trade just to break even. We can't name an intraday timing edge that large on
  ordinary bars. `bb_vwap_reversion` 15m is the worked example.
- **1h can, narrowly.** A round trip is 0.04–0.06 R with a 2–3 ATR stop, and 13% of a median
  8-hour move. An edge of roughly 15–30 bp per trade over a 6–24 bar hold would pay its costs;
  clearing the Deflated Sharpe bar takes more (section 4 works it out for the chosen plugin). 1h
  is the shortest timeframe where costs can be paid.
- **4h is cheap** at 0.02–0.03 R per round trip. The 4h baselines failed on timing, not on costs.

## 3. Candidates

All five read one symbol's bars only. None needs open interest or a reference series. Each one
uses something none of the six baselines used: taker flow (A, E), volatility compression (B),
time of week (C), or funding as a regime filter rather than a trigger (D).

### A. Forced-flow fade (1h): fade an hour of extreme net taker flow that moved price

- **Mechanism, and who pays.**
  - What happens: a liquidation is a market order. When a move trips leveraged positions, the
    exchange's liquidation engine and stop orders send a burst of one-sided taker volume that
    doesn't care about price. Market makers widen or pull quotes in that burst, because their
    inventory risk jumps. So price overshoots, and recovers part of the way once the forced flow
    stops and liquidity comes back.
  - Who pays: the forced, price-insensitive seller (or buyer, in a short squeeze). Who earns: the
    liquidity provider who takes the other side afterwards.
  - Theory, from memory: Grossman & Miller (1988, *Journal of Finance*, "Liquidity and Market
    Structure") price immediacy as a temporary concession. Campbell, Grossman & Wang (1993,
    *QJE*, "Trading Volume and Serial Correlation in Stock Returns") find that price changes on
    high volume reverse more. Brunnermeier & Pedersen (2009, *RFS*, "Market Liquidity and Funding
    Liquidity") explain margin spirals: forced deleveraging overshoots. Nagel (2012, *RFS*,
    "Evaporating Liquidity") reads short-term reversal returns as pay for providing liquidity,
    and that pay is highest when volatility is high.
- **Why it should persist.** Perps are traded with high leverage, and liquidation is mechanical.
  Liquidity providers have risk limits that bind exactly when the burst arrives. The premium is
  pay for warehousing risk at the worst moment, not a mispricing that a cleverer trader can
  remove at no risk.
- **Why the entry carries it.** The exits are an ATR stop and a fixed time exit. They have no
  target, no trail, and nothing that depends on the trigger bar. A random entry with those exits
  is a coin flip held for a fixed time, worth about zero minus costs. To beat the random p95, the
  trigger bars themselves must be followed by reversion.
- **Expected frequency.** At the default thresholds (net flow ≥ 0.5 of a median bar's volume, and
  a move of ≥ 1 ATR in the flow's direction), a trigger fires about 105 times per symbol per year
  (70–178). At the strict grid value (net flow ≥ 1.0), about 22 times per symbol per year (11–34).
  That gives roughly 600–3,000 OOS trades in total, and at least 46 in every half-year of the OOS
  span, pooled over symbols. Even the weakest training half-year (2022 H1) has 36 triggers at the
  strict threshold. So ≥ 30 OOS trades and ≥ 20 trades per training window are both safe (looks
  5 and 6).
- **Data needed.** `open`, `close`, `high`, `low`, `volume` and `taker_buy_volume`. The local
  data has no NaN taker volume (look 2). OI is not needed.
- **Why net taker flow, not a volume multiple.** The obvious trigger, volume ≥ 3–5× its
  trailing median together with a large bar, has two problems.
  - The judge can't verify it. The judge's lookahead check runs the plugin's default parameters
    on synthetic bars and kills a plugin that never signals there. Those bars have i.i.d. uniform
    volume, which never exceeds about 1.8× its median, so a volume-multiple trigger never fires.
    The judge would kill it as unverifiable before any walk-forward.
  - Adding the taker side by sign alone tells us nothing new: 98.6% of 1h volume-shock down bars
    already have a taker-buy share of 0.5 or less (look 3).

  Net taker flow in units of a median bar's volume, (taker buys − taker sells) / median volume,
  puts both the volume shock and the one-sidedness into one continuous number. It reaches the
  defaults' threshold on the judge's synthetic bars (13 trades; look 7).
- **How it most likely fails.**
  1. The flow is informed: news, or a large trader with information (Kyle 1985, *Econometrica*,
     "Continuous Auctions and Insider Trading", from memory). Then price continues and the fade
     loses. Llorente, Michaely, Saar & Wang (2002, *RFS*, "Dynamic Volume-Return Relation of
     Individual Stocks", from memory) find exactly this split: high-volume moves reverse when
     trading is hedging-driven and continue when it is informed. Without OI or liquidation data
     we can't tell the two apart.
  2. Fast market makers capture the reversion inside the shock hour, and nothing is left at the
     next open.
  3. Deleveraging lasts several hours and runs through the stop.
  4. About two thirds of the triggers are longs, because the down tail of net flow is fatter
     (look 5). A crash year could hit the long side hard, and all eight books at once.

### B. Volatility-compression breakout (4h)

- **Mechanism, and who pays.** Volatility clusters and mean-reverts (Engle 1982, *Econometrica*;
  Bollerslev 1986, *Journal of Econometrics*; from memory). In a quiet stretch, market makers
  quote tight and range traders fade the edges, with stops just outside the range. A break of
  the range fires those stops, and the expansion that follows runs one way for a while. The range
  faders and short-volatility liquidity providers pay. This is the "squeeze" of Bollinger (2001,
  *Bollinger on Bollinger Bands*, a book; from memory).
- **Why it should persist.** Volatility clustering is one of the most robust stylized facts, and
  stops cluster at obvious levels.
- **Why the entry might carry it.** The claim is that breakouts from compression run further, in
  units of the ATR at entry, than moves from random times.
  - With a trailing exit, the random benchmark already earns from trends: the Donchian runs at
    p48 and p77 show that. So B must beat a raised bar.
  - With a time exit instead, B becomes a claim of directional continuation, and the Donchian
    evidence says breakout direction was worth about the same as a coin flip.
- **Expected frequency.** A Bollinger width in the bottom 10% of its trailing 180 bars, followed
  by a break of the 20-bar range, fires 22–33 times per symbol per year (look 4). That is about
  100 per symbol over the OOS span.
- **Data needed.** OHLC.
- **How it most likely fails.** False breakouts. It is also the Donchian mechanism with a filter
  added, and the lab has already measured breakout timing at about random. Honestly, it would be
  a child of `donchian_atr`.

### C. Weekend-move reversal (4h)

- **Mechanism, and who pays.**
  - From Friday's US close to Sunday evening, the capital tied to traditional markets is
    offline: CME futures basis traders, and the authorised participants of the US spot ETFs (BTC
    since January 2024, ETH since July 2024). Weekend flow meets thinner books, so weekend moves
    carry more temporary price pressure, which is corrected when that capital returns.
  - The weekend trader who takes liquidity from a thin book pays; whoever provides liquidity
    across the gap earns. The Monday weekend effect in equities (French 1980, *Journal of
    Financial Economics*, from memory) is a distant cousin of the idea.
- **Why it should persist.** Market hours are institutional facts. One caution, from memory and
  unverified here (we have no internet): CME announced in 2025 that it planned to move its crypto
  futures to 24/7 trading in early 2026. If that happened, part of this mechanism disappears
  exactly in the holdout.
- **Why the entry carries it.** The entry comes at a fixed weekly time, and the exits are an ATR
  stop and a 24-hour time exit. A random entry with the same exits is worth about zero.
- **Expected frequency.** One trigger per symbol per weekend, about 208 per symbol over the OOS
  span if every weekend is traded. A size threshold cuts that hard: only 2–7% of weekends move
  more than one 2-day ATR (look 4). So it has to trade almost every weekend.
- **Data needed.** OHLC and timestamps.
- **How it most likely fails.**
  - Breadth. All eight symbols move together at weekends, so this is close to one bet a week. Even
    a real edge caps the Sharpe somewhere around 1.
  - Weekend news (hacks, regulatory action) is information, and price continues after it.
  - "CME gaps fill" is well-worn folklore, so if the effect were real it would be crowded.

### D. Time-series momentum with a funding-crowding filter (4h)

- **Mechanism, and who pays.**
  - Time-series momentum: past returns predict future returns over weeks, through underreaction
    and trend-chasing flows. Moskowitz, Ooi & Pedersen (2012, *Journal of Financial Economics*,
    "Time Series Momentum", from memory), and Liu & Tsyvinski (2021, *RFS*, "Risks and Returns of
    Cryptocurrency", from memory), who report it in crypto.
  - The funding filter: skip a trend trade when that side's funding is high. A crowded,
    carry-paying trend is the one most exposed to a liquidation reversal.
  - Late, leveraged trend chasers pay, through funding and through liquidations.
- **Why it should persist.** Underreaction and leverage cycles are behavioural and
  institutional.
- **Why the entry carries it.** The direction comes from the sign of the past N-day return, and
  the exit is a time exit with no trail, so a random 50/50 entry is worth about zero.
- **Expected frequency.** Close to continuous. With one-week holds, about 200 trades per symbol.
  The filter rarely binds: funding is above 2 bp on only 3–5% of bars (look 4), and BNB's
  settled rate is exactly 0 at 55% of settlements (look 2b), so the filter behaves differently on
  BNB.
- **Data needed.** `close` and `funding_last`.
- **How it most likely fails.** Crypto trends are market-wide, so the eight books are about two
  independent bets. Breakout direction was already measured as no better than random.

### E. Taker-flow persistence (4h)

- **Mechanism, and who pays.** Informed traders split their orders (Kyle 1985, from memory), so
  persistent net taker buying over a day reveals accumulation that isn't fully priced yet. The
  liquidity providers who sell to them are adversely selected and pay.
- **Why it should persist.** An informed trader always splits orders, to limit price impact.
- **Why the entry carries it.** The direction comes from the 24-hour net taker imbalance, the
  exit is a time exit, and random entries are worth about zero.
- **Expected frequency.** The 24-hour imbalance is beyond ±5% on 5–14% of 4h bars (look 4), so
  triggers are frequent.
- **Data needed.** `taker_buy_volume` and `volume`.
- **How it most likely fails.**
  - Its sign is A's sign reversed, on the same data. Persistent taker buying can just as well be
    retail chasing that reverts.
  - Perp taker share leans below 0.5 (median 0.49, look 2), a level bias that would have to be
    removed first.
  - Market makers see the same flow and price it immediately.

## 4. Choice

**Implement A, `flow_shock_fade`, at 1h. The runner-up is C, weekend-move reversal.**

| gate (constraint 4) | A flow fade 1h | B compression 4h | C weekend 4h | D TSMOM + funding 4h | E flow persistence 4h |
|---|---|---|---|---|---|
| judge can verify it on synthetic bars at the defaults | yes (13 trades) | yes | yes | yes | yes |
| ≥ 30 OOS trades, ≥ 20 per training window, ≥ 10 in a 6-month holdout | yes (600–3,000; at least 46 per 6 months) | yes | yes, if every weekend is traded | yes | yes |
| beats random p95 with the same exits | only if the reversion is real; the exits are neutral, so the test is fair | must beat a benchmark already lifted by trend capture | fair test | fair test, but trend direction already measured at about random | fair test; its sign is the opposite of A's |
| no narrow parameter island | two thresholds on one continuous statistic; thousands of trades at the loose corner | moderate | few parameters | few parameters | moderate |
| Deflated Sharpe ≥ 0.95 (about 1.52 annualised) | needs a gross reversion of at least ~28–31 bp per trade at the defaults and ~44–50 bp at `flow_x = 1.0`; floors, not estimates (below) | ceiling is the Donchian 0.5–0.8 | ceiling around 1, from breadth | ceiling around 1, correlated books | unknown |
| holdout exam | needs exchange-file candles for the window (section 5, harness notes) | — | CME 24/7 risk | — | same data caveat as A |

Why A:

- **It is the only one with a counterparty we can name and a payoff that can reach the bar.** A
  rough sum, which is arithmetic and not a measurement, for `hold_bars` = 12:

  | step | defaults (`flow_x` = 0.5) | strict (`flow_x` = 1.0) |
  |---|---|---|
  | trades a year over the 8 books (look 6 lower bound ÷ 4 years) | 685 | 162 |
  | share of triggers on distinct hours (look 5) | 2,352 of 3,998 = 59% | 535 of 812 = 66% |
  | independent trades a year, about | 400 | 107 |
  | Sharpe per trade for 1.52 a year (1.52 ÷ √n) | 0.076 | 0.147 |
  | net edge per trade (× the 0.61 R spread below) | 0.046 R = 0.14 ATR | 0.090 R = 0.27 ATR |
  | ATR on trigger bars, pooled median (look 8) | 102 bp | 112 bp |
  | net edge per trade in bp | 14 bp | 30 bp |
  | **gross reversion per trade** (net + the 14 bp round trip) | **28 bp** (0.28 ATR) | **44 bp** (0.39 ATR) |
  | the same at the median symbol's trigger-bar ATR (121 and 134 bp, look 8) | 31 bp | 50 bp |

  - The 0.61 R is the standard deviation of a 12-bar outcome against a 3-ATR stop. The median
    absolute 12-bar move is 1.23 ATR (134 ÷ 109 bp, look 1). For normal returns the standard
    deviation is 1.48 times the median absolute value: 1.82 ATR, which is 0.61 R.
  - **These are floors, not estimates.** Three things the sum leaves out raise the hurdle:
    - Fat tails. For fat-tailed returns the standard deviation is more than 1.48 times the median
      absolute move, so the spread is wider than 0.61 R. The 1.52 bar itself assumes normal
      returns too (section 1, lesson 6).
    - Correlation across symbols in adjacent hours. The distinct-hour count merges only triggers
      in the same hour, so a cascade that reaches symbols an hour or two apart still counts as
      several independent trades.
    - Post-shock volatility above a lagging ATR. Volatility clusters, so the hours after a shock
      move more than ordinary hours, while the Wilder ATR at the trigger bar gives the shock bar a
      weight of only 1/14. In units of the stop's ATR, the spread after a trigger is then wider
      than the all-bar ratio above.
  - One simplification cuts the other way: the 3-ATR stop caps a loss near 1 R, but only on the
    trades that travel that far against the entry. Funding over a 12-hour hold, about 1 bp at
    most, is left out.
  - So the reversion after the next open has to be at least about 28–31 bp per trade at the
    defaults and 44–50 bp at the strict threshold: 0.28 and 0.39 of the trigger bar's ATR, after
    a trigger move of at least one pre-shock ATR. Those are plausible sizes for the reversion of a
    forced move. They are also exactly zero if the mechanism isn't there.
- **Its triggers are spread out.** 31% of the default triggers share their hour with three or
  more other symbols (22% at `flow_x = 1.0`), against 47% for plain volume shocks (looks 3 and 5).
  Weekends (C) and trends (D) are close to one bet across all books.
- **Its test is clean.** Its exits earn nothing on random entries, so a pass could only come from
  the entry. B must beat a benchmark the lab has already seen lifted by the trailing exit, and D
  repeats a direction signal the Donchian runs already measured at about random.
- **It leans toward the hours where the trend baselines lost.** Within each symbol, the median
  trigger bar's ATR is 0.98–1.19 times that symbol's median ATR at the defaults, and 1.05–1.36
  times at `flow_x = 1.0` (look 8). Pooled over the symbols, the defaults' lean is all in the
  tail: the trigger-bar ATR is 102 bp at the median and 211 bp at p90, against 103 and 191 bp
  for all bars. At `flow_x = 1.0` it is 112 and 261 bp. (Pooled quantiles mix the symbols in
  proportion to their trigger counts. Section 2's 109 bp for all bars is the median of the
  per-symbol medians; on that measure trigger bars are at 121 bp at the defaults and 134 bp at
  `flow_x = 1.0`.) High volatility is where `donchian_atr` 4h averaged −0.135 R. If this plugin
  works, it is partly a different exposure, more so at the strict threshold.

Said plainly: **we don't expect it to pass the judge.** The most likely result is a gross
reversion near zero: the shock either kept going or had already reverted inside the hour. That
would mean a kill at the random-entry or Deflated Sharpe gate. Its single biggest risk is the
sign. The theory supports fading forced flow and following informed flow, and with no OI or
liquidation data we can't tell which kind a shock was. We chose it because, if any of the five
can clear a 1.5 Sharpe bar, this is the one, and because a kill would still teach something
specific: forced-flow reversion at 1h doesn't survive these costs.

Why C is the runner-up: nothing the lab has already measured bears on it, its exits are as fair
as A's, and it is the cheapest to trade (4h). Its breadth of about one bet a week caps it below
the bar even if the effect is real. B and D sit in the trend family whose timing the lab has
already measured at about random. E is A with the sign flipped. Choosing E after seeing A's
result would mean choosing the sign from an OOS outcome.

## 5. Plugin spec: `flow_shock_fade`

**File:** `strategies/flow_shock_fade.py`. It defines one class, `FlowShockFade(Strategy)`.
Imports, and nothing else (plugins import only numpy, pandas and `perp_lab`): `numpy as np`,
`from perp_lab.indicators import atr`, and `from perp_lab.strategy import EXIT, HOLD, Strategy, enter`.
Add nothing to `perp_lab/indicators.py`: the flow statistic is specific to this plugin, and `atr`
already exists.

**Hypothesis (the module docstring's first line, verbatim):**

> An hour of extreme one-sided taker flow that moves price marks forced or impatient trading (liquidation and stop cascades), so fading it at the next open earns the partial reversion over the following hours, net of costs.

The rest of the module docstring summarises the rules (use this text as written):

```
Root of a new family (order-flow shock). flow = (taker buys - taker sells) on the bar, divided by
the median bar volume of the previous vol_n bars; mv = (close - open) / ATR at t-1. Long when
flow <= -flow_x and mv <= -move_k; short when flow >= flow_x and mv >= move_k. Stop stop_atr x ATR
from the close, no target, exit after hold_bars bars. The exits ignore the trigger on purpose:
random entries with the same exits earn nothing, so only the entry can beat them.
```

**`PARENT`: none. This is a new family, so the module has no `PARENT` line.** The closest
existing plugin is `bb_vwap_reversion`, but its trigger is price-only, its target is VWAP, and
its mechanism is "stretched price", not forced flow. Making it the parent would put this plugin
in that family and spend that family's one holdout read on a different idea.

**Literals** (class attributes, exactly as written):

```python
params = {"flow_x": 0.5, "move_k": 1.0, "hold_bars": 12, "stop_atr": 3.0, "atr_n": 14, "vol_n": 168}
grid = {"flow_x": [0.5, 1.0], "hold_bars": [6, 12, 24]}
timeframes = ["1h"]
```

Where each value comes from:

| param | value | why |
|---|---|---|
| `flow_x` | 0.5 (grid 0.5, 1.0) | Net one-sided taker volume of half, or all, of a typical bar's total volume. At 1h, net flow is at or below −0.5 on 1.1% of bars and at or above +0.5 on 0.6%; for ±1.0 the figures are 0.2% and 0.1% (look 5). That is rare enough to mean an abnormal burst and common enough for ≥ 46 trades per 6 months. 2.0 is too rare: with `move_k` = 1 it gives 1–23 triggers per pooled half-year, under the 20-trade training minimum in 8 of 9 half-years. |
| `move_k` | 1.0 | The flow must have moved price by at least one pre-shock ATR. With no concession, there is nothing to revert. |
| `hold_bars` | 12 (grid 6, 12, 24) | A quarter, a half and a whole day. The mechanism says hours, not days, but doesn't say how many. |
| `stop_atr` | 3.0 | Wide enough that the post-shock volatility rarely stops the trade before the time exit, so the test is about the reversion over the hold. If price goes on by another shock's worth, the cascade isn't over. |
| `atr_n` | 14 | The lab's standard ATR. |
| `vol_n` | 168 | Seven days of 1h bars, so the median spans the weekly volume cycle. |

The default `flow_x` is the loose end of its grid on purpose. The judge's lookahead check needs the
defaults to signal on its synthetic bars, and `flow_x = 1.0` never does there (look 7). Don't
raise the defaults. The walk-forward selects `flow_x` and `hold_bars` from the grid in any case.

The sensitivity check moves every numeric param in `params` by ±10% and ±20%, so all six are
exposed to the island test. That is intended.

**`prepare(bars)`** returns `bars[[]].assign(flow=..., mv=..., atr=...)`: three float columns on
`bars.index`. With `p = self.p`:

```python
a    = atr(bars, p["atr_n"])                                         # Wilder ATR of true range, includes bar t
vmed = bars.volume.shift(1).rolling(p["vol_n"], min_periods=p["vol_n"]).median()  # bars t-vol_n .. t-1
flow = (2.0 * bars.taker_buy_volume - bars.volume) / vmed.replace(0.0, np.nan)
mv   = (bars.close - bars.open) / a.shift(1).replace(0.0, np.nan)    # ATR at t-1: excludes bar t
```

- `flow`:
  - Meaning: taker buy volume minus taker sell volume on bar t (taker sell = `volume −
    taker_buy_volume`), in units of the median volume of the previous `vol_n` bars.
  - Units: base-asset `volume` and `taker_buy_volume`. Never the quote columns.
  - Warmup: NaN for rows 0 … `vol_n − 1` (the first 168 rows).
  - NaN: if `taker_buy_volume` is NaN the result is NaN, and a zero median gives NaN.
  - The column always exists in lab bars, holding NaN when unknown (ccxt candles, the test
    synthetic bars), so no handling for a missing column is needed. Hand-built test bars must
    include it: `tests/conftest.py::flat_bars` doesn't.
- `mv`: the signed body of bar t in units of the ATR at t−1. It is NaN for rows 0 … `atr_n − 1`,
  because `atr` first has a value at row `atr_n − 1`.
- `atr`: the ATR at t, used by the stop. It is NaN for rows 0 … `atr_n − 2`.
- Causality: every term reads bars ≤ t, so `tests/test_lookahead.py` must pass. No column after t
  is touched, and no `shift(-k)` is used.

**`entry(ctx)`**, called when flat. Read `f, m, a = ctx["flow"], ctx["mv"], ctx["atr"]`:

1. If any of `f`, `m`, `a` is not finite (`np.isfinite`), return `HOLD`. This covers warmup
   and missing taker data.
2. **Long:** `f <= -flow_x and m <= -move_k` → `enter(1, *self.bracket(ctx, 1))`.
3. **Short:** `f >= flow_x and m >= move_k` → `enter(-1, *self.bracket(ctx, -1))`.
4. Otherwise `HOLD`.

Both comparisons are inclusive. Because `flow_x > 0`, the long and short rules can't both be
true.

**`bracket(ctx, side)`** returns `(ctx["close"] - side * stop_atr * ctx["atr"], None)`: the stop is
`stop_atr` ATRs (3.0 by default) from the signal bar's close, and there is **no target**.

Don't override `on_bar`: the base class already dispatches to `entry` when flat and to `manage`
when in a position.

**`manage(ctx)`** has no trail, no target and no stop change:
- If `ctx.position.bars_held >= hold_bars`, return `EXIT`.
- Otherwise return `HOLD`.

A trigger on the opposite side while in a position is ignored: the engine calls `manage`, not
`entry`, while a position is open. There is no reversal.

**Facts the unit tests can pin** (engine semantics, checked on synthetic bars in look 7):

- **Timing.**
  - A trigger at the close of bar t fills at the open of bar e = t + 1.
  - At the close of bar e, `bars_held` is 1. `EXIT` is returned at the close of bar
    e + hold_bars − 1 and fills at the open of bar e + hold_bars, so the position is held for
    exactly `hold_bars` bars.
  - The trade record counts the exit bar, so it shows `bars_held = hold_bars + 1` and
    `exit_reason = "signal"` (for example 13 when `hold_bars` is 12).
  - The next entry can be signalled at the close of bar e + hold_bars at the earliest.
- **Stop.**
  - For a long, the stop is `close_t − stop_atr·atr_t`; for a short, `close_t + stop_atr·atr_t`
    (3·atr_t at the default).
  - `bracket` never returns a target, and `manage` never moves the stop.
- **Warmup.** With 1h bars, nothing can fire before row `vol_n` (168).
- **Missing taker data.** A bar with NaN `taker_buy_volume` never fires.
- **Synthetic bars** (look 7). On `with_moving_columns(synthetic_bars(seed=1))`:
  - the defaults make 13 trades, `hold_bars=6` makes 13 and `hold_bars=24` makes 12;
  - `flow_x=1.0` makes none;
  - raw `synthetic_bars(seed=1)` gives none at any setting, because its taker columns are NaN.

**Expected trades per symbol, 2022-07 to 2026-07** (look 6). The lower bound skips triggers while
a position is held for its full `hold_bars`; the upper bound counts every trigger. A stop exit can
only add trades.

| flow_x | hold_bars | per symbol, lower bound: median [min, max] | upper bound | all 8 symbols, lower bound | long share |
|---|---|---|---|---|---|
| 0.5 | 6 | 357 [246, 593] | 442 [293, 754] | 3,013 | 0.64 |
| 0.5 | 12 | 327 [229, 530] | 442 [293, 754] | 2,740 | 0.64 |
| 0.5 | 24 | 277 [203, 429] | 442 [293, 754] | 2,330 | 0.64 |
| 1.0 | 6 | 75 [44, 131] | 93 [50, 150] | 673 | 0.67 |
| 1.0 | 12 | 71 [43, 127] | 93 [50, 150] | 647 | 0.68 |
| 1.0 | 24 | 67 [43, 121] | 93 [50, 150] | 609 | 0.68 |

Every combo has at least 46 trades in every half-year of the OOS span, pooled over symbols, and
the 2022 H1 training half-year has 36 triggers even at `flow_x = 1.0` (look 5). So
`min_train_trades = 20` should never bind, and a 6-month holdout should see well over 10. A quiet
2-month fold can hold as few as 5 at `flow_x = 1.0`.

**Harness notes for the implementer:**

1. **A test change, in its own commit.** `tests/test_lookahead.py::test_plugin_trades_on_synthetic_data`
   runs plugins on raw `make_bars(seed=1)`, whose taker, trades and quote-volume columns are NaN.
   This plugin makes no trades there, so that test fails.
   - Change it to run on `with_moving_columns(make_bars(seed=1))` (imported from
     `perp_lab.validate.causality`). Those are the bars the judge's own lookahead check and
     `tests/test_worker.py::test_sandboxed_batch_equals_in_process` already use.
   - For the four existing plugins this changes nothing, because none of them reads the columns
     that `with_moving_columns` fills.
   - Commit it separately, for example `test(lookahead): check trades on the judge's moving-column bars`.
2. **Don't raise the defaults.** A default of `flow_x = 1.0` makes no signal on the judge's
   synthetic bars, and the judge would then kill the plugin as `defaults: no signals on synthetic
   data`. Look 7b ran the harness's own check on a draft and saw exactly that. At the specified
   defaults, the check passes.
3. **For the user, not for the implementer.** Candles stitched by `lab tail` and the live feed
   come through ccxt, whose OHLCV carries NaN taker columns (`perp_lab/data/live.py`). This plugin
   never fires on such candles.
   - `lab paper` couldn't trade it without first extending the live feed.
   - Before `lab exam`, every candle in the exam window must come from data.binance.vision.
     `lab backfill` after the window's last month has closed does that, and complete funding needs
     that run anyway. Otherwise the exam can come up short of 10 trades and spend the family's
     one read for nothing.

## 6. What would kill it

Each of these ends this line of research. None of them is a reason to write a tweaked child.

1. **The walk-forward verdict is `noise`.**
   - OOS Sharpe ≤ 0, or no better than random-entry p95. The exits earn about zero on random
     entries by construction, so failing this gate means the trigger carries no timing
     information at 1h.
   - A narrow island. The edge depends on the exact threshold, so it isn't a mechanism.
   - Fewer than 30 OOS trades. We expect 600–3,000, and look 2 found no missing taker data, so
     this would point to a broken data path rather than a result. Find the cause before anything
     else: the judge doesn't re-judge a file version it has already judged, so there is no free
     retry.
2. **The gross OOS P&L, before fees and funding, is ≤ 0.** The reversion isn't there, whatever
   the other gates say. **Don't flip the sign into a continuation child.** Picking the sign after
   seeing the OOS result is a hidden trial. A continuation hypothesis (candidate E) would need its
   own case made from mechanism, and the two must be counted as two trials.
3. **Gross is positive but costs make the net negative.** The edge is smaller than 1h costs.
   - Moving to 4h would dilute an hourly mechanism, and it would be a new trial chosen by
     outcome.
   - Moving to 15m roughly doubles the cost per R.
   - So neither move.
4. **Every walk-forward gate passes, but the Deflated Sharpe is below 0.95.** The effect is real
   but too small to clear the multiple-testing bar. Changing thresholds or the hold after seeing
   the OOS result is exactly what the deflation charges for. Record it and stop.
5. **The in-sample Sharpe exceeds the OOS Sharpe by more than about 1** (the `funding_fade`
   pattern). The training-window selection is fitting noise.
6. **Before the exam: concentration.** The judge promotes on its own, but `lab exam` is a
   separate command. Before running it, check the per-symbol and per-fold OOS results. If one
   symbol or one 2-month fold supplies more than half of the OOS net P&L, treat the effect as not
   established, and don't spend the family's read on it.
7. **The holdout exam fails.** The family ends; by rule it gets one read.

Two results to note, which kill nothing by themselves but rule out any child:
- The long side makes money while the short side loses. That is drift (about two thirds of the
  trades are long), not the symmetric mechanism.
- All the profit comes from the 2022 crash months.

## 7. Data looks

How the looks were done:
- Every look ran from the worktree as
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/burhan/backtesting-research /home/burhan/backtesting/.venv/bin/python <script>`.
  Without `PYTHONPATH`, a script run by path resolves `perp_lab` to the main checkout's editable
  install.
- All market data went through `common.py`. It loads the research config
  `/tmp/claude-1000/-home-burhan-backtesting/464fe316-f53a-4226-89ab-e02f2b7426f3/scratchpad/research/lab.yaml`
  and calls `perp_lab.data.store.load_bars(cfg, symbol, tf)` with no `end` and no token, so every
  read ends at `holdout_start` (2026-07-01). It also asserts that the last bar is before the
  holdout start.
- No script opens a data file, calls `load_funding` directly, or touches the holdout log, the
  budget or `perp_lab/holdout.py`.
- `load_bars` itself reads each symbol's funding file whole, as every lab read does. It keeps only
  the settlements that fall inside, or before, the returned bars, so no post-holdout rate reaches
  a bar.
- The scripts and their outputs (`*.out`) are in the scratch directory and are not committed.

Scratch directory: `/tmp/claude-1000/-home-burhan-backtesting/464fe316-f53a-4226-89ab-e02f2b7426f3/scratchpad/research/`

| # | script | what was computed | why it is not a performance measurement |
|---|---|---|---|
| 1 | `look1_costs.py` | For 15m, 1h and 4h and each symbol: the medians of bar range, ATR14 and the absolute log move over 1, 2, 4, 8, 12 and 20 bars, across all bars; then the cost ratios (section 2). | Unconditional and unsigned. It describes how big moves are on every bar. No rule, signal or direction appears in it, so it can't say whether anything predicts them. |
| 2 | `look2_data.py` | Candle gaps; zero-volume bars; NaN shares of `taker_buy_volume`, `trades` and `oi`; the taker-buy share distribution over all bars; settled funding rates (mean, median, carry per day, share at 1 bp, above 1 bp and below 0); funding intervals; regime label shares. | Data quality and feature distributions only. No price after any bar is read. |
| 2b | `look2b_bnb_funding.py` | Nonzero funding settlements per half-year per symbol, and whether BNB's missing charges are zero rates or missing rows. | A data-coverage check. |
| 3 | `look3_shock_freq.py` | For 15m and 1h: how often bars with volume ≥ v× their 7-day median and a body ≥ k pre-shock ATRs occur; the taker-buy share on those bars; pooled counts per 6-month window; how many symbols share a shock bar. | How often a condition occurs, and what the flow looks like on the same bar. Nothing after the shock bar is read. |
| 4 | `look4_other_freq.py` | How often candidates B–E trigger: a compression breakout (4h); the size of the Friday-to-Sunday move against the ATR (that is the weekend move itself, not what follows it); how often funding sits above 2 bp or below 0; the 24-hour net taker imbalance distribution. | Frequencies and distributions of conditions at or before the trigger bar. |
| 5 | `look5_flow_freq.py` | Quantiles of the net-flow statistic; how often triggers fire for flow_x ∈ {0.25, 0.5, 1, 2} × move_k ∈ {0, 1, 2, 3} at 15m and 1h; pooled 6-month counts, including 2022 H1; the long share; how many symbols share a trigger bar; ATR on trigger bars against all bars (for flow_x = 1, move_k = 2 only; look 8 redoes it for the spec's triggers); the share of bars beyond each grid threshold; and how often the defaults fire on the harness's synthetic bars. | How often a condition occurs, plus contemporaneous feature values. No forward price is read. The synthetic part isn't market data. |
| 6 | `look6_trade_bounds.py` | Lower and upper bounds on trades per symbol and in pooled windows for each grid combo over 2022-07 to 2026-07: every trigger, and triggers taken greedily with a `hold_bars` refractory period. | Neither bound needs to know how a trade ended, so no price after a trigger is read. These are counts, not outcomes. |
| 7 | `look7_synthetic_check.py` (with `flow_shock_fade_draft.py`, a scratch draft of the spec) | On synthetic bars only (`synthetic_bars(seed=1)`, raw and with moving columns): whether the spec parses as plugin metadata, signals at its defaults and grid corners, keeps its time exit, and stays causal at the judge's cut points. | Loads no market data. It checks the spec against the harness, not against markets. |
| 7b | `look7b_judge_check.py` | The harness's own `causality_failures` (the judge's lookahead check) on the draft, using the in-process worker twin from the test suite: no failures. The same draft with the default `flow_x` raised to 1.0 is reported as `defaults: no signals on synthetic data`. | Synthetic bars only, as for look 7. |
| 8 | `look8_trigger_atr.py` | At 1h: the ATR14 at bar t, in bp of bar t's close, on the bars where the spec's trigger fires at (flow_x, move_k) = (0.5, 1) and (1.0, 1), with look 5's (1.0, 2) repeated as a cross-check, against all bars. Pooled p10/p50/p90, the median of the per-symbol medians, each symbol's median, and its ratio to that symbol's all-bar median, over the whole span; and the pooled median over 2022-07 to 2026-07 alone (100 and 109 bp for the two spec triggers). This is the ATR the stop is set from, so it converts section 4's hurdle from R into bp. | A same-bar feature distribution: the trigger and the ATR both use bars ≤ t, and nothing after the trigger bar is read. It says how wide the stop is on trigger bars, not what happens after them. |

Findings that the sections above rely on:

- **Data quality.**
  - There are no NaN `taker_buy_volume` or `trades` values anywhere in the research span.
  - SOL and XRP have 2 candle gaps each, both in late February 2022.
  - `oi` is NaN everywhere (not backfilled).
  - BNB's settled funding rate is exactly 0 at 2,693 of its 4,926 settlements. These are recorded
    zero rates, not missing rows.
  - SOL's funding interval was 8h, 4h or 2h at different times; every other symbol stayed at 8h.
- **Taker share.** Over all bars it centres at 0.49–0.50. On volume-shock bars it almost always
  agrees with the move's direction: at 1h with v = 4 and k = 2, 98.6% of down shocks have a share
  of 0.5 or less.
- **Net-flow statistic at 1h.** p0.1 −1.31, p1 −0.52, p5 −0.23, p50 −0.01, p95 0.17, p99 0.39,
  p99.9 0.97. The down tail is fatter, so about two thirds of the triggers are longs.
- **Breadth.** Share of triggers whose bar also has a trigger in at least one, three or five
  other symbols:
  - at the defaults (flow_x = 0.5, move_k = 1): 60%, 31% and 14%, with 2,352 distinct trigger
    hours for 3,998 triggers;
  - at flow_x = 1, move_k = 1: 51%, 22% and 10%;
  - at flow_x = 1, move_k = 2: 54%, 23% and 7%;
  - for 1h volume shocks (v = 4, k = 2): 68%, 47% and 32%.
- **ATR on trigger bars at 1h** (look 8), in bp of price: p10/p50/p90 pooled over symbols, then
  the median of the per-symbol medians.
  - All bars: 54/103/191, and 109.
  - The defaults (flow_x = 0.5, move_k = 1), 3,998 triggers: 55/102/211, and 121.
  - flow_x = 1, move_k = 1, 812 triggers: 57/112/261, and 134.
  - flow_x = 1, move_k = 2 (look 5's trigger), 634 triggers: 57/111/255, and 137.
  - Per symbol, the median trigger bar's ATR is 0.98–1.19 times the symbol's all-bar median at
    the defaults, and 1.05–1.36 times at flow_x = 1, move_k = 1.
- **Synthetic bars.** By construction, volume is uniform on 100–1000 and never exceeds about 1.8×
  its median, so a volume-multiple trigger is unverifiable there. The net-flow statistic peaks at 0.80 in absolute value, so the
  defaults (0.5, 1.0) fire and `flow_x = 1.0` doesn't.
