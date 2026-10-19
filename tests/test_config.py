import yaml

from perp_lab.config import load_config

CFG = load_config()
RAW = yaml.safe_load(open("config/lab.yaml"))


def test_the_slippage_floor_is_todays_cost_and_the_model_is_set():
    c = CFG["costs"]
    assert c["slippage_bps"] == 2.0
    m = c["slippage_model"]
    assert set(m) == {"adv_days", "spread_bps_at_100m", "alpha", "max_bps"} and m["adv_days"] == 30
    assert m["alpha"] > 0 and m["spread_bps_at_100m"] > 0 and m["max_bps"] == 50.0


def test_the_universe_rule_states_the_survivor_bias():
    assert "survivor bias" in CFG["universe_rule"].lower() and "2026-10-18" in CFG["universe_rule"]
