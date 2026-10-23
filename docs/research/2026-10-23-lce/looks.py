"""LCE's pre-trial looks at real research data: counts only (spec, "Looks at real data before the trials").

For each coin of the universe and each LCE plugin at its defaults, the plugin runs in a sealed sandbox
worker on the coin's research 5m bars (`load_bars` stops at holdout_start; every open_time is asserted
before 2026-07-01), and only the run's features (what `prepare` returned) are read. The run's decisions,
fingerprint, trades and equity are never read, printed, stored or used. Nothing here computes returns,
P&L, hit rate or R, or joins anything to later prices; the one price used is each bar's own close, which
the bracket compares with its levels.

Per coin and plugin:
  bars; in-window bars (in_window == 1)
  b1 .. b6: signals per branch, bars with sig_branch == k and in_window == 1
  small: the share of those signals on a bar that isn't big (big == 0)
  valid: of those signals, the ones on a bar that isn't big (big == 0) with a valid bracket, recomputed
    from the feature columns exactly as LcePct.bracket defines it (long: stop = (z_dn_c + z_dn2_c) / 2,
    target z_up_lo, stop < close < target; short: stop = (z_up_c + z_up2_c) / 2, target z_dn_hi,
    target < close < stop; all finite, in the window). The daily rules are not applied: they depend on
    fills and wins.
  fb: of the weekday session days (sday, Mon-Fri by the trading date it ends on) with levels, the share
    whose anchor_fb is 1
  spacing: the median spacing_u over the session days with levels (all of them)
  no 15:55: Mon-Fri dates, from the first bar's close to the last's, without a bar that closes at 15:55
    New York time, by trusted code (`session_window.in_session`, a 15:55-15:55 window on open_time + 5 min)
then the totals across coins (shares pooled over every coin's signals or days), and the dates without a
15:55 bar.

Two consistency checks against trusted clock code, a look for bugs: the plugin's sday equals
`ny_session_day(open_time)`, and its in_window equals `in_session` of the bar's close in the plugin's
declared session. A mismatch is printed per coin.

A run's reply carries every feature of every bar, about 150 MB for a coin's 690k bars: over the 64 MB a
worker may send by the config (`sandbox.max_output_mb`, a bound on the frames the parent reads). This
script raises that bound in memory to 512 MB for its own workers; nothing else in the config changes,
and the workers stay sealed under the configured memory and CPU limits.

Run from the repo root:
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python docs/research/2026-10-23-lce/looks.py > docs/research/2026-10-23-lce/looks.out
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from perp_lab.config import load_config
from perp_lab.data.store import load_bars
from perp_lab.engine.backtest import EngineParams
from perp_lab.indicators import ny_session_day
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client
from perp_lab.session_window import in_session

PLUGINS = ("lce_pct", "lce_atr")
TF, BAR_MS = "5m", 300_000
HOLDOUT = pd.Timestamp("2026-07-01", tz="UTC")
NY = "America/New_York"
AT_1555 = {"tz": NY, "days": [0, 1, 2, 3, 4], "start": "15:55", "end": "15:55"}
OUTPUT_MB = 512
THREADS = 4
SIDES = np.array([0.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0])      # by branch, as the plugin's SIDES


def valid_bracket(f: pd.DataFrame, close: np.ndarray, side: np.ndarray) -> np.ndarray:
    """LcePct.bracket's validity per bar, for the side of its branch (0: none, so never valid)."""
    long_stop, long_tgt = (f.z_dn_c + f.z_dn2_c).to_numpy() / 2, f.z_up_lo.to_numpy()
    short_stop, short_tgt = (f.z_up_c + f.z_up2_c).to_numpy() / 2, f.z_dn_hi.to_numpy()
    with np.errstate(invalid="ignore"):
        ok_long = (long_stop < close) & (close < long_tgt) & np.isfinite(long_stop) & np.isfinite(long_tgt)
        ok_short = (short_tgt < close) & (close < short_stop) & np.isfinite(short_stop) & np.isfinite(short_tgt)
    ok = np.where(side == 1, ok_long, np.where(side == -1, ok_short, False))
    return ok & (f.in_window.to_numpy() == 1) & np.isfinite(close)


def missing_1555(open_time: np.ndarray) -> list:
    """Mon-Fri New York dates whose 15:55 falls between the first and the last bar's close, without a
    bar that closes then."""
    close = open_time + BAR_MS
    at = in_session(close, AT_1555)
    have = set(pd.to_datetime(close[at], unit="ms", utc=True).tz_convert(NY).date)
    first, last = (pd.Timestamp(int(x), unit="ms", tz="UTC") for x in close[[0, -1]])
    dates = pd.bdate_range(first.tz_convert(NY).date(), last.tz_convert(NY).date())
    at1555 = (dates + pd.Timedelta(hours=15, minutes=55)).tz_localize(NY)
    keep = (at1555 >= first) & (at1555 <= last)
    return [d.date() for d in dates[keep] if d.date() not in have]


def look(cfg, srcs: dict, sym: str) -> dict:
    bars = load_bars(cfg, sym, TF)
    t = bars.open_time.to_numpy(np.int64)
    assert len(t) and t.max() < HOLDOUT.value // 10**6, sym
    close = bars.close.to_numpy(float)
    out = {"bars": len(bars), "first": pd.Timestamp(int(t[0]), unit="ms", tz="UTC").date(),
           "missing": missing_1555(t)}
    sday_ref = ny_session_day(t)
    for name, src in srcs.items():
        with client.spawn(cfg, src) as w:
            w.bars(sym, TF, bars)
            f = w.run(dict(src.meta.params), EngineParams.from_config(cfg, sym), features=True).features
        assert f.index.equals(bars.index), (sym, name)
        br, inw, big = f.sig_branch.to_numpy(), f.in_window.to_numpy() == 1, f.big.to_numpy()
        sig = inw & (br >= 1)
        side = SIDES[np.where(np.isnan(br), 0, br).astype(int)]
        valid = sig & (big == 0) & valid_bracket(f, close, side)
        sday = f.sday.to_numpy()
        days = pd.DataFrame({"sday": sday, "fb": f.anchor_fb.to_numpy(), "sp": f.spacing_u.to_numpy()})
        days = days.groupby("sday", sort=True).first()
        lev = days[days.fb.notna()]
        wk = lev[((lev.index.to_numpy(np.int64) + 3) % 7) < 5]
        out[name] = {
            "in_window": int(inw.sum()),
            "branch": [int((sig & (br == k)).sum()) for k in range(1, 7)],
            "signals": int(sig.sum()), "small": int((sig & (big == 0)).sum()), "valid": int(valid.sum()),
            "wk_days": len(wk), "fb_days": int((wk.fb == 1).sum()),
            "spacing": lev.sp.dropna().to_numpy(),
            "sday_bad": int((sday != sday_ref).sum()),
            "inw_bad": int((inw != in_session(t + BAR_MS, src.meta.session)).sum()),
        }
        del f
    return out


def share(a, b) -> str:
    return f"{a / b:.3f}" if b else "n/a"


def median(x) -> str:
    return f"{np.median(x):.3f}" if len(x) else "n/a"


def row(label: str, p: dict) -> str:
    return (f"{label:<9} {p['bars']:>10,} {p['in_window']:>9,} " + " ".join(f"{x:>7,}" for x in p["branch"])
            + f" {share(p['small'], p['signals']):>6} {p['valid']:>7,} {share(p['valid'], p['signals']):>5}"
            f" {share(p['fb_days'], p['wk_days']):>6} {median(p['spacing']):>7} {p['missing']:>8}")


def main():
    cfg = load_config()
    cfg["sandbox"]["max_output_mb"] = OUTPUT_MB
    srcs = {name: PluginSource.read(name) for name in PLUGINS}
    syms = cfg.symbols
    with ThreadPoolExecutor(THREADS) as pool:      # each thread spawns and closes its own workers
        res = dict(zip(syms, pool.map(lambda s: look(cfg, srcs, s), syms)))

    print(f"LCE pre-trial looks: research {TF} bars before {HOLDOUT:%Y-%m-%d}, defaults, counts only")
    print("plugins: " + ", ".join(f"{n} {s.hash[:12]}" for n, s in srcs.items()))
    head = (f"{'coin':<9} {'bars':>10} {'in-window':>9} " + " ".join(f"{'b' + str(k):>7}" for k in range(1, 7))
            + f" {'small':>6} {'valid':>13} {'fb':>6} {'spacing':>7} {'no 15:55':>8}")
    for name in PLUGINS:
        print(f"\n{name}\n{head}")
        tot = {k: 0 for k in ("bars", "missing", "in_window", "signals", "small", "valid", "wk_days", "fb_days")}
        tot["branch"], tot["spacing"] = np.zeros(6, int), []
        for sym in syms:
            r, p = res[sym], res[sym][name]
            print(row(sym, {**p, "bars": r["bars"], "missing": len(r["missing"])}))
            tot["bars"] += r["bars"]
            tot["missing"] += len(r["missing"])
            for k in ("in_window", "signals", "small", "valid", "wk_days", "fb_days"):
                tot[k] += p[k]
            tot["branch"] += p["branch"]
            tot["spacing"].append(p["spacing"])
        print(row("total", {**tot, "spacing": np.concatenate(tot["spacing"])}))
        print(f"signals {tot['signals']:,}; weekday session days with levels {tot['wk_days']:,}, "
              f"on the anchor fallback {tot['fb_days']:,}")
        bad = [f"{s} sday {res[s][name]['sday_bad']} in_window {res[s][name]['inw_bad']}" for s in syms
               if res[s][name]["sday_bad"] or res[s][name]["inw_bad"]]
        print("checks: sday == ny_session_day and in_window == in_session on every bar: "
              + ("ok" if not bad else "MISMATCH " + ", ".join(bad)))

    print("\nfirst bar and Mon-Fri dates without a bar closing at 15:55 New York")
    for sym in syms:
        m = res[sym]["missing"]
        print(f"{sym:<9} first {res[sym]['first']}; {len(m)}" + (": " + ", ".join(map(str, m)) if m else ""))


if __name__ == "__main__":
    main()
