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


COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK",
         "ADA", "DOT", "LTC", "BCH", "FIL", "AAVE", "XLM", "ATOM", "ETC"]
BEFORE = {"BTCUSDT": 0.004, "ETHUSDT": 0.005, "SOLUSDT": 0.0065, "BNBUSDT": 0.0065, "XRPUSDT": 0.0065,
          "DOGEUSDT": 0.0065, "AVAXUSDT": 0.010, "LINKUSDT": 0.010}
NEW = {"ADAUSDT": 0.010, "DOTUSDT": 0.010, "LTCUSDT": 0.010, "BCHUSDT": 0.010, "FILUSDT": 0.015,
       "AAVEUSDT": 0.010, "XLMUSDT": 0.015, "ATOMUSDT": 0.010, "ETCUSDT": 0.010}


def test_the_universe_is_the_17_coins_from_2020():
    assert RAW["universe"] == COINS
    assert (RAW["history_start"], RAW["holdout_start"]) == ("2020-01-01", "2026-07-01")


def test_every_coin_has_its_own_bracket_and_none_is_kinder_than_before():
    br = RAW["margin_brackets"]
    assert {f"{c}USDT" for c in COINS} <= set(br)
    assert {s: br[s]["mmr"] for s in BEFORE} == BEFORE and {s: br[s]["mmr"] for s in NEW} == NEW
    assert all(br[s]["mmr"] >= br["default"]["mmr"] for s in NEW) and all(br[s]["cum"] == 0.0 for s in NEW)


def test_5m_is_not_a_research_timeframe():
    assert RAW["timeframes"] == ["15m", "1h", "4h"]
