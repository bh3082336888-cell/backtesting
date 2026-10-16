"""Causality (lookahead) check, usable at judge time as well as in the test suite.

For several cut points k: features computed on bars[:k] must equal features computed on
the full series for rows < k, and the engine must emit identical signals for bars < k-1.
Any use of future data breaks one of the two.

At judge time every column a plugin can read moves (a peek at an all-NaN or constant column
would go unseen), and a plugin that never signals is reported, since the check proves nothing
about it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from perp_lab.engine.backtest import EngineParams, run_backtest

H = 3_600_000
CUTS = [400, 1100, 1777, 2500]


def synthetic_bars(n=3000, seed=0, start="2023-01-01", funding_every=8, tf_ms=H):
    """Synthetic perp bars with regimes, trends and funding extremes."""
    rng = np.random.default_rng(seed)
    drift = np.repeat(rng.normal(0, 0.002, n // 200 + 1), 200)[:n]
    r = drift + rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[100.0, close[:-1]]
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
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


def assert_causal(strat_factory, bars, cuts: list[int] | None = None):
    """Raises AssertionError on a lookahead; returns the full-series run (with its signals)."""
    full = strat_factory()
    f_full = full.prepare(bars)
    assert f_full.index.equals(bars.index), "prepare() must return features on the bars index"
    r_full = run_backtest(full, bars, f_full, symbol="X", tf="1h", ep=EngineParams(), record_signals=True)
    for k in (CUTS if cuts is None else cuts):
        cut = bars.iloc[:k]
        s = strat_factory()
        f_cut = s.prepare(cut)
        pd.testing.assert_frame_equal(f_full.iloc[:k], f_cut, check_exact=False, rtol=1e-9, atol=1e-12,
                                      obj=f"features at cut {k}")
        r_cut = run_backtest(s, cut, f_cut, symbol="X", tf="1h", ep=EngineParams(), record_signals=True)
        a = [x for x in r_full.signals if x[0] < k - 1]
        b = [x for x in r_cut.signals if x[0] < k - 1]
        assert a == b, f"signals diverge before cut {k}"
    return r_full


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


def causality_failures(cls, bars: pd.DataFrame | None = None, *, cuts: list[int] | None = None,
                       require_signals: bool = True, params: dict | None = None) -> list[str]:
    """assert_causal on the default params and every grid corner; one line per failing variant.

    Runs on a copy of the bars in which every column moves. A crash on synthetic data
    disqualifies a plugin as surely as a peek at the future, and so does never signalling
    with its default params: then nothing about its causality is verified. Callers checking
    extra date ranges pass require_signals=False: a plugin may honestly be quiet there."""
    bars = with_moving_columns(synthetic_bars(seed=1) if bars is None else bars)
    variants = [{}] + [{k: v} for k, vals in cls.grid.items() for v in (vals[0], vals[-1])]
    if params and params not in variants:   # e.g. the walk-forward's frozen exam params
        variants.append(dict(params))
    out = []
    for kw in variants:
        try:
            r = assert_causal(lambda: cls(**kw), bars, cuts)
        except Exception as e:
            label = ", ".join(f"{k}={v}" for k, v in kw.items()) or "defaults"
            msg = str(e).splitlines()[0] if str(e) else ""
            out.append(f"{label}: {type(e).__name__}: {msg}" if msg else f"{label}: {type(e).__name__}")
            continue
        if require_signals and not kw and not r.signals:
            out.append(NO_SIGNALS)
    return out
