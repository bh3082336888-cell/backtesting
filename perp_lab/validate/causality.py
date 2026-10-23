"""Causality (lookahead) check, usable at judge time as well as in the test suite.

For several cut points k: features computed on bars[:k] must equal features computed on
the full series for rows < k, and the engine must emit identical signals for bars < k-1.
Any use of future data breaks one of the two.

At judge time every column a plugin can read moves (a peek at an all-NaN or constant column
would go unseen), and a plugin that never signals is reported, since the check proves nothing
about it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from perp_lab.engine.backtest import EngineParams
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client
from perp_lab.sandbox.client import RunOutput
from perp_lab.sandbox.errors import PluginError

H = 3_600_000
CUTS = [400, 1100, 1777, 2500]


def synthetic_bars(n=3000, seed=0, start="2023-01-01", funding_every=8, tf_ms=H, vol_scale=1.0):
    """Synthetic perp bars with regimes, trends and funding extremes. vol_scale multiplies the drift,
    return and high/low noise magnitudes (1.0 leaves every draw as it was)."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 0.002 * vol_scale, n // 200 + 1), 200)[:n]
    r = drift + rng.normal(0, 0.01 * vol_scale, n)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[100.0, close[:-1]]
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004 * vol_scale, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004 * vol_scale, n)))
    t0 = int(pd.Timestamp(start, tz="UTC").value // 10**6)
    ot = t0 + np.arange(n) * tf_ms
    fc = np.zeros(n)
    idx = np.arange(0, n, funding_every)
    fc[idx] = 0.0001 + rng.normal(0, 0.00005, len(idx))
    spikes = rng.choice(idx, size=max(len(idx) // 15, 1), replace=False)
    fc[spikes] = rng.choice([-1, 1], size=len(spikes)) * rng.uniform(0.0008, 0.002, len(spikes))
    df = pd.DataFrame({
        "open_time": ot, "open": open_, "high": hi, "low": lo, "close": close,
        "volume": rng.uniform(100, 1000, n), "quote_volume": np.nan, "trades": np.nan,
        "taker_buy_volume": np.nan, "taker_buy_quote_volume": np.nan, "close_time": ot + tf_ms - 1,
        "funding_charge": fc, "funding_last": pd.Series(np.where(fc != 0, fc, np.nan)).ffill().values,
        "funding_interval_h": 8.0, "oi": np.nan, "oi_value": np.nan,
        "regime": rng.choice(["trending", "ranging", "high_vol"], n), "regime_adx": 20.0,
    })
    df.index = pd.to_datetime(df.open_time, unit="ms", utc=True)
    return df


def compare(full: RunOutput, cut: RunOutput, k: int) -> None:
    """Raises AssertionError when the run on bars[:k] differs from the full run before the cut."""
    pd.testing.assert_frame_equal(full.features.iloc[:k], cut.features, check_exact=False, rtol=1e-9,
                                  atol=1e-12, check_dtype=False, obj=f"features at cut {k}")
    if not full.decisions.before(k - 1).same(cut.decisions.before(k - 1)):
        raise AssertionError(f"signals diverge before cut {k}")


EXTRAS_SEED = 7
NO_SIGNALS = "defaults: no signals on synthetic data; causality cannot be verified"


def with_moving_columns(bars: pd.DataFrame, seed: int = EXTRAS_SEED) -> pd.DataFrame:
    """A copy of bars in which each exchange column that is missing, all-NaN or constant gets
    seeded, non-constant values. Each value at row t depends only on rows <= t."""
    out = bars.copy()
    n = len(out)
    rng = np.random.default_rng(seed)
    close = out["close"].to_numpy(dtype=float)
    volume = out["volume"].to_numpy(dtype=float)
    oi = 1e6 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    taker = volume * rng.uniform(0.3, 0.7, n)
    fills = {
        "oi": oi, "oi_value": oi * close,
        "quote_volume": volume * close, "trades": rng.integers(50, 5000, n).astype(float),
        "taker_buy_volume": taker, "taker_buy_quote_volume": taker * close,
        "regime_adx": np.clip(25 + np.cumsum(rng.normal(0, 1, n)), 5, 80),
        "funding_interval_h": np.where((np.arange(n) // 500) % 2 == 0, 8.0, 4.0),
    }
    for c, v in fills.items():
        if c not in out or out[c].nunique() <= 1:
            out[c] = v
    return out


def causality_failures(cfg, src: PluginSource, bars: pd.DataFrame | None = None, *,
                       cuts: list[int] | None = None, require_signals: bool = True,
                       params: dict | None = None) -> list[str]:
    """The cut-vs-full comparison on the default params and every grid corner; one line per failing variant.

    Every compared run happens in its own fresh sandbox worker that is sent only its own bars, so a
    plugin cannot carry what it saw in a longer run into a shorter one. Runs go in waves of
    increasing length (each cut, then the full series): no longer run starts before every shorter
    one has finished. The comparison is done here, in the parent.

    Runs on a copy of the bars in which every column moves. A crash on synthetic data
    disqualifies a plugin as surely as a peek at the future, and so does never signalling
    with its default params: then nothing about its causality is verified. Callers checking
    extra date ranges pass require_signals=False: a plugin may honestly be quiet there."""
    bars = with_moving_columns(synthetic_bars(seed=1) if bars is None else bars)
    variants = [{}] + [{k: v} for k, vals in src.meta.grid.items() for v in (vals[0], vals[-1])]
    if params and params not in variants:   # e.g. the walk-forward's frozen exam params
        variants.append(dict(params))
    labels = [", ".join(f"{k}={v}" for k, v in kw.items()) or "defaults" for kw in variants]
    n = len(bars)
    lengths = sorted({k for k in (CUTS if cuts is None else cuts) if k < n}) + [n]
    runs: dict[tuple[int, int], RunOutput] = {}     # (variant, length) -> run
    failed: dict[int, str] = {}
    with ThreadPoolExecutor(client.max_workers(cfg)) as pool:
        for length in lengths:
            alive = [v for v in range(len(variants)) if v not in failed]
            futures = {v: pool.submit(client.fresh_run, cfg, src, "X", "1h", bars.iloc[:length],
                                      variants[v], EngineParams(), features=True) for v in alive}
            for v, fut in futures.items():      # every run of this wave has finished before the next
                try:                            # wave starts: result() waits for each in turn
                    runs[v, length] = fut.result()
                except PluginError as e:
                    failed[v] = (f"{labels[v]}: {e.type}: {e.message}" if e.kind == "exception"
                                 else f"{labels[v]}: plugin_error: {e.kind}")
    out = []
    for v, kw in enumerate(variants):
        if v in failed:
            out.append(failed[v])
            continue
        full = runs[v, n]
        try:
            for k in lengths[:-1]:
                compare(full, runs[v, k], k)
        except Exception as e:
            msg = str(e).splitlines()[0] if str(e) else ""
            out.append(f"{labels[v]}: {type(e).__name__}: {msg}" if msg else f"{labels[v]}: {type(e).__name__}")
            continue
        if require_signals and not kw and full.decisions.n_actions() == 0:
            out.append(NO_SIGNALS)
    return out
