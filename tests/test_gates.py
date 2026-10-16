import math

import numpy as np
import pandas as pd
import pytest

from perp_lab.judge.gates import (beats_parent, deflated_sharpe, dsr_from_moments, expected_max_sharpe,
                                  holdout_consistent, moments)


def _series(vals, start="2024-01-01"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq="D", tz="UTC"))


def _noise(n, seed, mu=0.0, sd=0.01):
    return _series(np.random.default_rng(seed).normal(mu, sd, n))


V = (0.5 / math.sqrt(365)) ** 2          # trials' Sharpes spread by 0.5 annualised


def test_dsr_reference_value():
    # donchian_atr 4h today: OOS Sharpe 0.45 over 4 years, one trial
    assert dsr_from_moments(0.45 / math.sqrt(365), 1461, 0.0, 3.0, 1, 0.0) == pytest.approx(0.815907, abs=1e-6)


def test_sr0_is_zero_for_one_trial_and_grows_with_trials():
    assert expected_max_sharpe(1, V) == 0.0
    assert expected_max_sharpe(10, V) == pytest.approx(0.04120912, abs=1e-8)


def test_dsr_falls_as_trials_rise():
    assert dsr_from_moments(1.0 / math.sqrt(365), 1461, 0.0, 3.0, 10, V) == pytest.approx(0.664622, abs=1e-6)


def test_moments_match_hand_computed_values():
    sr, t, skew, kurt = moments(_series([0.01, -0.005, 0.02, 0.0, 0.005]))
    assert t == 5
    assert sr == pytest.approx(0.62385029, abs=1e-8)
    assert skew == pytest.approx(0.39587034, abs=1e-8)
    assert kurt == pytest.approx(1.99452155, abs=1e-8)


def test_deflated_sharpe_on_a_series():
    d = deflated_sharpe(_series([0.01, -0.005, 0.02, 0.0, 0.005]), 1, 0.0)
    assert d["dsr"] == pytest.approx(0.912049, abs=1e-6) and d["sr0"] == 0.0


def test_flat_series_scores_zero():
    assert deflated_sharpe(_series([0.0] * 100), 1, 0.0)["dsr"] == 0.0     # never trades
    assert deflated_sharpe(_series([0.01, 0.02]), 1, 0.0)["dsr"] == 0.0    # too short


def test_identical_series_never_beats_parent():
    p = _noise(500, 1)
    assert beats_parent(p, p.copy(), block=10, n_boot=500, seed=1) == 0.0   # ties are not wins


def test_same_spread_higher_mean_always_beats_and_lower_never_does():
    p = _noise(500, 1)
    assert beats_parent(p + 0.002, p, block=10, n_boot=500, seed=1) == 1.0
    assert beats_parent(p - 0.002, p, block=10, n_boot=500, seed=1) == 0.0


def test_too_little_overlap_scores_zero():
    parent = _noise(100, 1)                                   # 2024-01-01 .. 2024-04-09
    child = _series(np.full(100, 0.01), start="2024-04-05")  # overlaps 5 days
    assert beats_parent(child, parent, block=10, n_boot=500, seed=1) == 0.0


def test_beats_parent_is_reproducible_for_a_seed():
    p = _noise(500, 1)
    c = 0.5 * p + _noise(500, 2)
    assert beats_parent(c, p, block=10, n_boot=500, seed=7) == beats_parent(c, p, block=10, n_boot=500, seed=7)


WF = _noise(1000, 3, mu=0.003)
KW = dict(min_trades=10, floor_pct=5, block=10, n_boot=2000, seed=1)


def test_holdout_like_the_walk_forward_record_passes():
    ok, reasons, d = holdout_consistent(_noise(120, 4, mu=0.003), 25, WF, **KW)
    assert ok and reasons == [] and d["days"] == 120 and d["floor"] is not None


def test_too_few_holdout_trades_fail_even_when_positive():
    ok, reasons, _ = holdout_consistent(_noise(120, 4, mu=0.003), 5, WF, **KW)
    assert not ok and reasons == ["not enough evidence: 5 holdout trades < 10"]


def test_negative_holdout_fails():
    ok, reasons, _ = holdout_consistent(_noise(120, 5, mu=-0.003), 25, WF, **KW)
    assert not ok and reasons[0].startswith("holdout Sharpe")


def test_positive_but_far_below_the_walk_forward_record_fails():
    ok, reasons, d = holdout_consistent(_noise(120, 6, mu=0.0003, sd=0.001), 25, WF, **KW)
    assert not ok and d["sharpe"] > 0 and d["cum_return"] < d["floor"]
    assert reasons == [r for r in reasons if r.startswith("holdout return")] and len(reasons) == 1


def test_default_spread_is_the_no_edge_sampling_spread():
    # with no var given, SR0 uses Var(SR) = 1/(T-1): the spread of Sharpe estimates of no-edge
    # strategies over the candidate's own track length (T = 1461 days → 1/1460)
    d = deflated_sharpe(_noise(1461, 7, mu=0.001), 10)
    assert d["sr0"] == pytest.approx(0.041209121, abs=1e-9)
