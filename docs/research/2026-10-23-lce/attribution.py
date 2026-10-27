"""LCE trials 8 and 9, after the verdicts: each OOS trade's flowchart branch, and its planned reward to risk.
A diagnostic of two finished trials (spec, "Research note (after the two verdicts)"); it spends no trial,
runs no judge, exam or backtest, and reads no holdout data.

For each judged plugin (lce_pct, then lce_atr):
  The run: the verdict file of the plugin's current hash (runs/verdicts/<name>@5m@<hash10>.json), whose
    walk-forward run directory holds oos_trades.parquet. The plugin's hash must equal the run's
    strategy_hash, so the reruns use the judged code.
  The modal params: the combo selected in the most folds, computed from the run's folds by
    perp_lab.validate.walkforward.modal_params itself (ties as it breaks them), over the plugin's
    defaults. Asserted equal to the run's own sensitivity base params, which the judge computed the same
    way.
  The reruns: per coin, the plugin runs at those params in a sealed sandbox worker on the coin's research
    5m bars (`load_bars` stops at holdout_start; every open_time is asserted before 2026-07-01) with
    features on, exactly as looks.py does; only the features are read (the run's decisions, fingerprint,
    trades and equity are never read). The walk-forward runs every combo continuously over the whole
    research span, so these are the features the judged runs saw. Each other combo some fold selected is
    rerun the same way, for a cross-check.
  The join: sig_branch (0 none, 1-6 the flowchart rows) of the bar whose open_time is the trade's
    signal_time.
    Primary (the spec's method): every trade under the modal params. A trade from a fold that selected
      another combo may sit on a bar with another branch, or none, under the modal params: the script
      prints the share of trades from such folds and the share of trades whose signal bar has no branch
      under the modal params.
    Cross-check: every trade under its own fold's combo (`combo` in oos_trades). There every signal bar
      must have a branch whose side is the trade's side; the script prints the count that doesn't (0
      expected) and the share of trades whose branch agrees with the modal-params branch.
  Per branch (1-6 and "none"): trades, win rate (pnl > 0), target-exit share, mean R (`r_multiple`), the
    median planned R:R at the fill (below), and the means per trade in bp of notional of gross
    ((pnl_gross + slippage) / notional: before fees, slippage and funding, as checks.py), fees, slippage,
    funding and net (pnl). Once under the modal params, once under each trade's own combo, and once for the
    modal folds' trades alone. Then gross and net by the combo the trade's fold selected (with gross by
    exact branch), and by the signal's year.
  Planned R:R, from the trade's own prices: (target - entry_price) / (entry_price - stop_init), each
    times the trade's side, at the fill (the next open after the signal, with entry slippage). Signed,
    not absolute: a fill at or past the target (the next open gapped to it) has a planned R:R <= 0, and
    an absolute value would count it as room to the target. The same ratio at the signal bar's close
    (the bracket's own plan, before the gap to the next open) is printed for comparison. Quartiles, the
    shares below 0.5 and 1.0, and per bucket (<= 0, 0-0.5, 0.5-1, 1-2, >= 2; and < 0.5 together):
    trades, win rate, target-exit share, mean R, gross and net bp per trade.
  Which zone the signal bar touched, for branches 1 and 6 (whose condition includes at_zone: the bar's
    range overlaps the zone just below the close or the one just above it, either side). From each trade's
    own combo's level features at the signal bar (z_dn_c, z_up_lo, z_up_c, z_up2_c, from which the two
    zones and their half-width follow as level_features draws them) and the bar's low and high. "Near":
    the zone on the stop side, which the flowchart means ("buy dips into the cloud at demand": for a long,
    the zone at or below the close). "Far": the zone on the target side (for a long, supply above). Per
    branch: near only, far only, both, and near with or without far (near only + both: the bar touched
    the zone the flowchart means), with the per-group line.
  How concentrated the gross is, by date: each trade's date is the New York date of its signal (the
    session is 09:40-15:55, so one date per session). For all trades and for exact branch 6 alone: the
    dates and trades; gross per trade with its t clustered by date (sum over dates of (S_d - n_d m)^2,
    S_d a date's summed gross bp, n_d its trades, m the mean; no small-sample correction); the mean of
    the per-date mean gross, weighting each date equally, and its t across dates; gross per trade with
    the top 10 and top 20 dates removed (dates ranked by S_d), with the removed trades and their short
    share; and the top 10 dates, each with its trades, short share, S_d, its share of the total, and the
    market's move that session: the equal-weight mean over the coins with both bars of close(15:55) /
    close(09:40) - 1, New York bar closes, how many of them fell, and the date's rank among all Mon-Fri
    dates in the OOS span (1 = the worst move). The before-costs Sharpe as a t: Sharpe x sqrt(OOS days /
    365), from summary.json (the lab annualises daily Sharpe by sqrt(365)).
  From oos_trades alone, for the note's claims and fidelity sections: risk at entry as a share of equity
    (qty x |entry_price - stop_init| / equity_before; 1% unless the 3x-equity notional cap binds),
    mean R by exit reason, the share of trades that paid or received funding, the share of fills before
    12:00 New York, trades per Mon-Fri date across the book, and the first BTC trade's prices.
  Levels at the modal params, from the modal reruns' features: the distribution over session days with
    levels of the day's median level spacing (spacing_u, in u). Every coin's research span, as looks.py's
    medians, which it should reproduce.

Like looks.py, it raises `sandbox.max_output_mb` in memory to 512 MB for its own workers: a coin's
features are about 150 MB, over the configured 64 MB. Nothing else in the config changes, and the workers
stay sealed under the configured memory and CPU limits.

Run from the repo root:
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/research/2026-10-23-lce/attribution.py > docs/research/2026-10-23-lce/attribution.out
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from looks import HOLDOUT, NY, OUTPUT_MB, PLUGINS, TF, THREADS
from perp_lab.config import load_config
from perp_lab.data.store import load_bars
from perp_lab.engine.backtest import EngineParams
from perp_lab.engine.metrics import ANN
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client
from perp_lab.validate.walkforward import combo_key, modal_params

RUNS = {"lce_pct": "20261027-071510-8626d9", "lce_atr": "20261027-075940-4464e6"}     # trials 8 and 9
SIDES = np.array([0, 1, 1, -1, 1, -1, -1])                    # by branch, as the plugin's SIDES
LEVELS = ["z_dn_c", "z_up_lo", "z_up_c", "z_up2_c"]          # the signal bar's nearest levels (features)
COLS = ["symbol", "side", "signal_time", "entry_time", "exit_time", "entry_price", "qty", "notional",
        "stop_init", "target", "exit_reason", "fees", "slippage", "funding", "pnl_gross", "pnl",
        "equity_before", "r_multiple", "fold", "combo"]
BUCKETS = {"<= 0": lambda x: x <= 0, "0-0.5": lambda x: (x > 0) & (x < 0.5), "0.5-1": lambda x: (x >= 0.5) & (x < 1),
           "1-2": lambda x: (x >= 1) & (x < 2), ">= 2": lambda x: x >= 2, "< 0.5 (both)": lambda x: x < 0.5}


def judged(cfg, src: PluginSource) -> tuple[dict, dict, Path]:
    """The verdict of this plugin version, its walk-forward run's summary, and the run directory."""
    v = json.loads((Path(cfg["judge"]["verdicts_dir"]) / f"{src.name}@{TF}@{src.hash[:10]}.json").read_text())
    d = cfg.runs_dir / v["walkforward"]["run_id"]
    s = json.loads((d / "summary.json").read_text())
    assert d.name == RUNS[src.name], (src.name, d.name)
    assert s["strategy_hash"] == src.hash == v["strategy_hash"], src.name
    return v, s, d


def features(cfg, src: PluginSource, sym: str, bars: pd.DataFrame, params: dict, cols: list) -> pd.DataFrame:
    """The plugin's features at params on the coin's bars, from a sealed worker: only `cols` kept."""
    with client.spawn(cfg, src) as w:
        w.bars(sym, TF, bars)
        f = w.run(params, EngineParams.from_config(cfg, sym), features=True).features
    assert f.index.equals(bars.index), (sym, src.name)
    return f[cols].copy()


def per_coin(cfg, plans: dict, sym: str) -> dict:
    """For each plugin: the branch and the nearest levels at each of the coin's trades' signal bars under
    every combo in its plan, the signal bars' close, low and high, and (modal combo) the session days'
    level spacing; and, under "move", the coin's session move per New York date."""
    bars = load_bars(cfg, sym, TF)
    t = bars.open_time.to_numpy(np.int64)
    assert len(t) and t.max() < HOLDOUT.value // 10**6, sym
    close, low, high = (bars[c].to_numpy(float) for c in ("close", "low", "high"))
    out = {"move": session_move(t, close)}
    for name, (src, combos, modal, trades) in plans.items():
        sig = trades.loc[trades.symbol == sym, "signal_time"].to_numpy(np.int64)
        i = np.searchsorted(t, sig)
        assert (i < len(t)).all() and (t[np.minimum(i, len(t) - 1)] == sig).all(), (name, sym)
        res = {"index": trades.index[trades.symbol == sym], "close": close[i], "low": low[i], "high": high[i],
               "branch": {}, "levels": {}}
        for key, params in combos.items():
            cols = ["sig_branch"] + LEVELS + (["sday", "anchor_fb", "spacing_u"] if key == modal else [])
            f = features(cfg, src, sym, bars, params, cols)
            res["branch"][key] = f.sig_branch.to_numpy()[i].astype(int)
            res["levels"][key] = f[LEVELS].to_numpy()[i]
            if key == modal:
                days = f.groupby("sday", sort=True)[["anchor_fb", "spacing_u"]].first()
                res["spacing"] = days.loc[days.anchor_fb.notna(), "spacing_u"].dropna().to_numpy()
            del f
        out[name] = res
    return out


def session_move(open_time: np.ndarray, close: np.ndarray) -> pd.Series:
    """Per Mon-Fri New York date with both bars: close at the bar closing 15:55 / close at the bar closing
    09:40, minus 1 (New York times of the bars' closes, open_time + 5 min)."""
    ct = pd.to_datetime(open_time + 300_000, unit="ms", utc=True).tz_convert(NY)
    hm = ct.hour * 60 + ct.minute
    wk = ct.dayofweek < 5
    a = pd.Series(close[wk & (hm == 9 * 60 + 40)], index=ct[wk & (hm == 9 * 60 + 40)].date)
    b = pd.Series(close[wk & (hm == 15 * 60 + 55)], index=ct[wk & (hm == 15 * 60 + 55)].date)
    return (b / a - 1).dropna()


def bp(x: pd.Series, t: pd.DataFrame) -> pd.Series:
    return x / t.notional * 1e4


def concentration(t: pd.DataFrame, move: pd.DataFrame, label: str) -> None:
    """How the gross per trade depends on a few dates (see the docstring)."""
    g = bp(t.pnl_gross + t.slippage, t)
    day = pd.to_datetime(t.signal_time, unit="ms", utc=True).dt.tz_convert(NY).dt.date
    S, n = g.groupby(day).sum(), g.groupby(day).size()
    m = S.sum() / n.sum()
    t_cl = m / np.sqrt(((S - n * m) ** 2).sum()) * n.sum()
    pm = S / n
    t_eq = pm.mean() / pm.std(ddof=1) * np.sqrt(len(pm))
    print(f"  {label}: {n.sum():,} trades on {len(S):,} dates; gross per trade {m:.2f} bp, t clustered by date"
          f" {t_cl:.2f}; each date weighted equally {pm.mean():.2f} bp, t {t_eq:.2f}")
    top = S.sort_values(ascending=False)
    for k in (10, 20):
        out = day.isin(top.index[:k])
        print(f"    without the top {k} dates: gross per trade {g[~out].mean():.2f} bp over {(~out).sum():,} trades"
              f" (removed {out.sum():,} trades, short share {(t.side[out] == -1).mean():.1%})")
    ew = move.mean(axis=1)
    rank = ew.rank(method="min")
    print(f"    top 10 dates by summed gross ({S.sum():,.0f} bp summed over all its trades):")
    print(f"      {'date':<10} {'trades':>6} {'short':>6} {'gross bp':>9} {'share':>6} {'market':>7} {'down':>6}"
          f" {'rank':>9}")
    for dt in top.index[:10]:
        x = t[day == dt]
        mk = move.loc[dt] if dt in move.index else pd.Series(dtype=float)
        print(f"      {dt!s:<10} {len(x):>6} {(x.side == -1).mean():>6.1%} {S[dt]:>9,.0f} {S[dt] / S.sum():>6.1%}"
              f" {ew.get(dt, np.nan):>7.2%} {int((mk < 0).sum()):>2} of {int(mk.notna().sum()):>2}"
              f" {int(rank.get(dt, 0)):>4} of {len(ew):,}")


def stats(t: pd.DataFrame) -> dict:
    """The per-group line: counts, rates and bp per trade."""
    if t.empty:
        return {"trades": 0}
    return {"trades": len(t), "win": (t.pnl > 0).mean(), "target": (t.exit_reason == "target").mean(),
            "r": t.r_multiple.mean(), "rr": t.rr_fill.median(), "gross": bp(t.pnl_gross + t.slippage, t).mean(),
            "fees": bp(t.fees, t).mean(), "slip": bp(t.slippage, t).mean(), "fund": bp(t.funding, t).mean(),
            "net": bp(t.pnl, t).mean()}


def table(t: pd.DataFrame, col: str) -> None:
    head = (f"  {'branch':<7} {'trades':>7} {'share':>6} {'win':>6} {'target':>7} {'mean R':>7} {'R:R p50':>7}"
            f" {'gross':>7} {'fees':>6} {'slip':>6} {'fund':>6} {'net':>7}")
    print(head)
    groups = [(str(k), t[t[col] == k]) for k in range(1, 7)] + [("none", t[t[col] == 0]), ("total", t)]
    for label, g in groups:
        s = stats(g)
        if not s["trades"]:
            print(f"  {label:<7} {0:>7}")
            continue
        print(f"  {label:<7} {s['trades']:>7,} {s['trades'] / len(t):>6.3f} {s['win']:>6.1%} {s['target']:>7.1%}"
              f" {s['r']:>7.3f} {s['rr']:>7.2f} {s['gross']:>7.2f} {s['fees']:>6.2f} {s['slip']:>6.2f}"
              f" {s['fund']:>6.2f} {s['net']:>7.2f}")


def q(x: pd.Series | np.ndarray, ps=(0.25, 0.5, 0.75)) -> str:
    return ", ".join(f"p{int(p * 100)} {np.quantile(x, p):.3f}" for p in ps)


def report(name: str, src: PluginSource, v: dict, s: dict, d: Path, t: pd.DataFrame, modal: str,
           res: dict) -> None:
    folds = pd.Series([f["combo"] for f in s["folds"]]).value_counts()
    print(f"\n==== {name}: verdict {v['decision'].upper()}, run {d.name}, plugin {src.hash[:12]}")
    print(f"combos selected per fold: " + ", ".join(f"{k} x{n}" for k, n in folds.items()))
    print(f"modal params (walkforward.modal_params on the run's folds): {modal}, selected in {folds[modal]} of "
          f"{len(s['folds'])} folds; equal to the run's sensitivity base params (asserted)")
    tied = folds[folds == folds.iloc[0]].index.tolist()
    if len(tied) > 1:
        first = {k: next(f["fold"] for f in s["folds"] if f["combo"] == k) for k in tied}
        print(f"  a tie: {', '.join(f'{k} (first selected in fold {first[k]})' for k in tied)} are each selected in"
              f" {folds.iloc[0]} folds; modal_params takes value_counts' first, and value_counts keeps tied combos"
              f" in their order of first appearance, so the tie goes to {modal}")
        assert modal == min(tied, key=first.get), name

    n = len(t)
    from_modal = t.combo == modal
    none_modal = t.br_modal == 0
    print(f"\n{n:,} OOS trades; from folds that selected the modal combo {from_modal.sum():,} ({from_modal.mean():.1%}),"
          f" from other folds {(~from_modal).sum():,} ({(~from_modal).mean():.1%})")
    print(f"under the modal params, signal bars with no branch (sig_branch == 0): {none_modal.sum():,} of {n:,} "
          f"({none_modal.mean():.1%}); of the modal folds' trades {(none_modal & from_modal).sum():,}, of the other "
          f"folds' {(none_modal & ~from_modal).sum():,} ({(none_modal & ~from_modal).sum() / max((~from_modal).sum(), 1):.1%}"
          f" of them)")
    bad_own = (t.br_own == 0) | (SIDES[t.br_own.to_numpy()] != t.side.to_numpy())
    agree = t.br_own == t.br_modal
    print(f"under each trade's own fold's combo: signal bars with no branch, or a branch of the other side: "
          f"{bad_own.sum()} (0 expected)")
    print(f"the branch under the modal params equals the branch under the trade's own combo for {agree.sum():,} "
          f"of {n:,} trades ({agree.mean():.1%}); of the other folds' trades {(agree & ~from_modal).sum():,} "
          f"({(agree & ~from_modal).sum() / max((~from_modal).sum(), 1):.1%})")

    print("\nper branch, every OOS trade under the modal params (the spec's attribution); bp of notional per trade;"
          "\nwin = pnl > 0; target = exits at the target; R:R p50 = median planned R:R at the fill")
    table(t, "br_modal")
    print("\nper branch, each trade under its own fold's combo (cross-check; exact)")
    table(t, "br_own")
    print("\nper branch, the modal folds' trades only (modal params = own combo)")
    table(t[from_modal], "br_modal")

    print("\nby the combo the trade's fold selected (exact branches): trades, gross and net bp per trade; gross by branch"
          " (trades)")
    for k, g in t.groupby("combo", sort=False):
        st = stats(g)
        br = ", ".join(f"{b} {stats(x)['gross']:.2f} ({len(x):,})" for b, x in g.groupby("br_own"))
        print(f"  {k:<22} {folds[k]:>2} folds {st['trades']:>6,} trades, gross {st['gross']:>6.2f}, net {st['net']:>7.2f};"
              f" by branch: {br}")
    print("by year of the signal (UTC): trades, gross and net bp per trade")
    for y, g in t.groupby(pd.to_datetime(t.signal_time, unit="ms", utc=True).dt.year):
        st = stats(g)
        print(f"  {y} {st['trades']:>6,} trades, gross {st['gross']:>6.2f}, net {st['net']:>7.2f}")

    print("\nplanned R:R = (target - entry_price) / (entry_price - stop_init), times the side")
    for label, col in (("at the fill (entry_price, after the gap to the next open and entry slippage)", "rr_fill"),
                       ("at the signal bar's close (the bracket's own plan)", "rr_close")):
        x = t[col]
        print(f"  {label}:\n    {q(x)}, mean {x.mean():.3f}; < 0.5 {(x < 0.5).mean():.1%}, < 1.0 {(x < 1).mean():.1%},"
              f" <= 0 {(x <= 0).sum():,} ({(x <= 0).mean():.1%}), >= 1.5 {(x >= 1.5).mean():.1%}, max {x.max():.3f}")
    print("  by bucket of the planned R:R at the fill:")
    print(f"    {'R:R':<13} {'trades':>7} {'share':>6} {'win':>6} {'target':>7} {'mean R':>7} {'gross':>7} {'net':>7}")
    for label, rule in BUCKETS.items():
        st = stats(t[rule(t.rr_fill)])
        if not st["trades"]:
            print(f"    {label:<13} {0:>7}")
            continue
        print(f"    {label:<13} {st['trades']:>7,} {st['trades'] / n:>6.3f} {st['win']:>6.1%} {st['target']:>7.1%}"
              f" {st['r']:>7.3f} {st['gross']:>7.2f} {st['net']:>7.2f}")

    print("\nwhich zone the signal bar touched, branches 1 and 6 (their at_zone), each trade under its own combo:"
          "\nnear = the zone on the stop side (for a long the one at or below the close: demand; for a short the one"
          "\nabove: supply), far = the zone on the target side (for a long: supply above)")
    print(f"  {'branch':<7} {'touched':<10} {'trades':>7} {'share':>6} {'win':>6} {'target':>7} {'mean R':>7}"
          f" {'R:R p50':>7} {'gross':>7} {'net':>7}")
    for label, br in (("1", [1]), ("6", [6]), ("1 and 6", [1, 6])):
        g = t[t.br_own.isin(br)]
        for kind, m in (("near only", g.near & ~g.far), ("far only", g.far & ~g.near), ("both", g.near & g.far),
                        ("near, any", g.near)):
            st = stats(g[m])
            print(f"  {label:<7} {kind:<10} {st['trades']:>7,} {st['trades'] / len(g):>6.3f} {st['win']:>6.1%}"
                  f" {st['target']:>7.1%} {st['r']:>7.3f} {st['rr']:>7.2f} {st['gross']:>7.2f} {st['net']:>7.2f}")
    g = t[t.br_own.isin([1, 6])]
    print(f"  consistency: branch 1 and 6 trades touching neither zone {int((~g.near & ~g.far).sum())},"
          f" with a NaN level {int(g.lv_nan.sum())} (0 and 0 expected); \"near, any\" = near only + both")

    bc, days = s["before_costs"]["sharpe"], s["metrics"]["days"]
    print(f"\nhow concentrated the gross is, by New York date of the signal (Mon-Fri dates in the OOS span:"
          f" {len(res['move']):,} with a market move)")
    print(f"  before-costs Sharpe {bc:.2f} over {days:,} days: as a t, {bc:.2f} x sqrt({days} / {ANN:.0f})"
          f" = {bc * np.sqrt(days / ANN):.2f} (one comparison; no deflation for N)")
    concentration(t, res["move"], "all trades")
    concentration(t[t.br_own == 6], res["move"], "branch 6 (exact)")

    risk = t.qty * (t.entry_price - t.stop_init).abs() / t.equity_before * 100
    print(f"\nrisk at entry, % of equity: {q(risk, (0.1, 0.25, 0.5, 0.75))}; under 0.99% (the 3x notional cap binds) "
          f"{(risk < 0.99).mean():.1%}")
    print("mean R by exit reason: " + ", ".join(f"{k} {g.r_multiple.mean():.3f} ({len(g):,})"
                                                for k, g in t.groupby("exit_reason")))
    print(f"trades with nonzero funding: {(t.funding != 0).mean():.1%}")
    fill = pd.to_datetime(t.entry_time, unit="ms", utc=True).dt.tz_convert(NY)
    print(f"fills before 12:00 New York: {(fill.dt.hour < 12).mean():.1%}")
    first, last = (pd.Timestamp(x, unit="ms", tz="UTC").tz_convert(NY).date()
                   for x in (t.signal_time.min(), t.signal_time.max()))
    wd = len(pd.bdate_range(first, last))
    print(f"trades per Mon-Fri date across the 17 coins: {n / wd:.2f} ({n:,} over {wd:,} dates, {first} .. {last})")
    b = t[t.symbol == "BTCUSDT"].sort_values("signal_time").iloc[0]
    print(f"first BTC trade: signal bar opens {pd.Timestamp(int(b.signal_time), unit='ms', tz='UTC')}, side {b.side:+d},"
          f" branch {b.br_own} (own combo); close {b.close:.2f}, fill {b.entry_price:.2f}, stop {b.stop_init:.2f}"
          f" ({b.entry_price - b.stop_init:+.2f}), target {b.target:.2f} ({b.target - b.entry_price:+.2f});"
          f" planned R:R {b.rr_fill:.3f} at the fill, {b.rr_close:.3f} at the close; exit {b.exit_reason}, R {b.r_multiple:.3f}")

    sp = np.concatenate(res["spacing"])
    print(f"\nlevels at the modal params: the day's median level spacing, in u, over {len(sp):,} coin session days with "
          f"levels: {q(sp, (0.1, 0.25, 0.5, 0.75, 0.9))} (step 0.303)")


def main():
    cfg = load_config()
    cfg["sandbox"]["max_output_mb"] = OUTPUT_MB
    plans, meta = {}, {}
    for name in PLUGINS:
        src = PluginSource.read(name)
        v, s, d = judged(cfg, src)
        modal = combo_key(modal_params(SimpleNamespace(selected=s["folds"])))
        base = {k: x for k, x in s["sensitivity"]["base_params"].items()}
        assert base == {**src.meta.params, **modal_params(SimpleNamespace(selected=s["folds"]))}, name
        combos = {f["combo"]: {**src.meta.params, **f["params"]} for f in s["folds"]}
        t = pd.read_parquet(d / "oos_trades.parquet", columns=COLS)
        for col in ("signal_time", "entry_time", "exit_time"):
            assert (pd.to_datetime(t[col], unit="ms", utc=True) < HOLDOUT).all(), (name, col)
        assert len(t) == s["metrics"]["trades"], name
        plans[name] = (src, combos, modal, t)
        meta[name] = (src, v, s, d)

    syms = cfg.symbols
    with ThreadPoolExecutor(THREADS) as pool:      # each thread spawns and closes its own workers
        res = dict(zip(syms, pool.map(lambda s: per_coin(cfg, plans, s), syms)))
    move = pd.DataFrame({sym: res[sym]["move"] for sym in syms}).sort_index()
    span = (move.index >= pd.Timestamp("2020-07-01").date()) & (move.index < HOLDOUT.date())
    move = move[span]

    print(f"LCE per-branch attribution and planned R:R, trials 8 and 9 (after the verdicts; no trial spent)")
    print(f"research {TF} bars before {HOLDOUT:%Y-%m-%d}; features-on reruns in sealed sandbox workers; "
          f"OOS trades from the judged runs")
    for name in PLUGINS:
        src, v, s, d = meta[name]
        _, combos, modal, t = plans[name]
        t = t.copy()
        t["br_modal"], t["br_own"] = 0, 0
        t["close"], t["low"], t["high"] = np.nan, np.nan, np.nan
        lv = np.full((len(t), len(LEVELS)), np.nan)
        for sym in syms:
            r = res[sym][name]
            idx = r["index"]
            t.loc[idx, "br_modal"] = r["branch"][modal]
            t.loc[idx, ["close", "low", "high"]] = np.c_[r["close"], r["low"], r["high"]]
            own = t.loc[idx, "combo"].to_numpy()
            t.loc[idx, "br_own"] = np.select([own == k for k in r["branch"]], list(r["branch"].values()), -1)
            pos = t.index.get_indexer(idx)
            for k, x in r["levels"].items():
                lv[pos[own == k]] = x[own == k]
        assert (t.br_own >= 0).all(), name
        t["br_modal"], t["br_own"] = t.br_modal.astype(int), t.br_own.astype(int)
        risk = (t.entry_price - t.stop_init) * t.side
        assert (risk > 0).all(), name
        t["rr_fill"] = (t.target - t.entry_price) * t.side / risk
        t["rr_close"] = (t.target - t.close) * t.side / ((t.close - t.stop_init) * t.side)
        # The two zones at_zone looks at, as level_features draws them: the one whose centre d1 is the
        # highest <= close, and the one whose centre u1 is the lowest > close; half-width hb (zone u / 2).
        dn, up_lo, up_c, up2_c = lv.T
        hb = np.where(up_lo < up_c, up_c - up_lo, up2_c - up_lo)        # z_up_lo is u1 - hb, else u2 - hb
        u1 = np.where(up_c > t.close, up_c, up2_c)                      # z_up_c is d1 when d1 == close
        below, above = t.low.to_numpy() <= dn + hb, t.high.to_numpy() >= u1 - hb
        long = t.side.to_numpy() == 1
        t["near"], t["far"] = np.where(long, below, above), np.where(long, above, below)
        t["lv_nan"] = np.isnan(dn + hb + u1)
        report(name, src, v, s, d, t, modal, {"spacing": [res[sym][name]["spacing"] for sym in syms], "move": move})


if __name__ == "__main__":
    main()
