import json
import re

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa

from perp_lab.judge.diagnostics import diagnostics
from perp_lab.runs import new_run_id, write_run
from perp_lab.store import db
from tests.test_judgements_store import jrow
from tests.test_store import cfg  # noqa: F401

DAYS = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
H_MS = 3_600_000
T0 = DAYS[0].value // 1_000_000                      # epoch ms

# symbol, side, notional, pnl_gross, slippage, fees, funding, exit_reason, bars_held, regime, fold
TRADES = [("BTCUSDT", 1, 1000.0, 20.0, 1.0, 1.2, 0.3, "stop", 4, "trending", 1),
          ("BTCUSDT", 1, 2000.0, -10.0, 2.0, 2.4, -0.4, "signal", 2, "ranging", 2),
          ("ETHUSDT", -1, 500.0, -5.0, 0.5, 0.6, 0.1, "stop", 6, "trending", 1),
          ("ETHUSDT", 1, 1000.0, 5.0, 1.0, 1.2, 0.0, "target", 8, "trending", 2)]
COLS = ["symbol", "side", "notional", "pnl_gross", "slippage", "fees", "funding", "exit_reason", "bars_held",
        "regime", "fold"]


def trades(rows=TRADES, sign=1):
    t = pd.DataFrame(rows, columns=COLS)
    t["side"] *= sign
    t["pnl"] = t.pnl_gross - t.fees - t.funding          # as the engine nets a trade
    t["entry_time"] = T0 + np.arange(len(t)) * 24 * H_MS  # a day apart, each held 4 hours
    t["exit_time"] = t.entry_time + 4 * H_MS
    return t


SUMMARY = {
    "metrics": {"sharpe": 0.3, "trades": 4}, "is_metrics": {"sharpe": 1.2},
    "random_entry": {"seeds": 10, "percentile": 95, "random_sharpe_mean": 0.1, "random_sharpe_p95": 0.9,
                     "strategy_pctile_vs_random": 0.6, "beats_random": False, "long_share": 0.75},
    "before_costs": {"sharpe": 0.8, "random_mean": 0.2, "random_p95": 1.1, "pctile": 0.7, "long_share": 0.75},
    "min_oos_trades": 8,
}


def recorded_run(cfg, run_id, t=None, sign=1, drop=(), **over):
    """A walk-forward run on disk and in the store: three folds (the third without trades)."""
    rng = np.random.default_rng(7)
    daily = pd.DataFrame({s: sign * rng.normal(0, 0.01, len(DAYS)) for s in ("BTCUSDT", "ETHUSDT")}, index=DAYS)
    daily["PORTFOLIO"] = daily.mean(axis=1)
    folds = [{"fold": k, "train_start": str(DAYS[0].date()), "train_end": str(DAYS[30 * k].date()),
              "test_start": str(DAYS[30 * k].date()), "test_end": str(DAYS[30 * k + 29].date()),
              "combo": combo, "params": {"n": int(combo[2:])}, "is_score": 1.0}
             for k, combo in ((1, "n=20"), (2, "n=10"), (3, "n=20"))]
    rec = {"run_id": run_id, "strategy": "s1", "strategy_hash": "h1", "hypothesis": "h", "params": {},
           "universe": ["BTCUSDT", "ETHUSDT"], "timeframes": ["4h"], "validation_mode": "walkforward",
           "status": "noise", "fail_reasons": ["x"], "folds": folds,
           "created_at": "2026-10-01T00:00:00+00:00", **SUMMARY, **over}
    for k in drop:
        rec.pop(k)
    frames = {"oos_daily": daily}
    if t is not None:
        frames["oos_trades"] = t
    return write_run(cfg, rec, frames)


def test_diagnostics_from_a_recorded_run(cfg):
    d = diagnostics(cfg, recorded_run(cfg, "r1", trades()), data="judged", earlier=[])
    assert set(d) == {"data", "gates", "before_costs", "per_trade_bp", "sides", "exits", "regimes", "breadth",
                      "is_vs_oos", "selection", "overlap"}
    assert d["data"] == "judged"
    assert d["gates"] == {"net_sharpe": 0.3, "random_mean": 0.1, "random_p95": 0.9, "pctile": 0.6,
                          "long_share": 0.75, "oos_trades": 4, "min_oos_trades": 8}
    assert d["before_costs"] == {"sharpe": 0.8, "random_mean": 0.2, "random_p95": 1.1, "pctile": 0.7}
    # by hand, in bp of each trade's notional: gross (210, -40, -90, 60), net (185, -60, -114, 38)
    assert d["per_trade_bp"] == pytest.approx({"gross": 35.0, "fees": 12.0, "slippage": 10.0, "funding": 0.75,
                                               "net": 12.25})
    s = d["sides"]
    assert (s["long"], s["short"]) == (3, 1)
    assert {k: v for k, v in s.items() if k not in ("long", "short")} == pytest.approx({
        "long_share": 0.75, "long_gross_bp": 230 / 3, "long_net_bp": 163 / 3,
        "short_gross_bp": -90.0, "short_net_bp": -114.0})
    assert d["exits"]["share"] == pytest.approx({"stop": 0.5, "signal": 0.25, "target": 0.25})
    assert d["exits"]["mean_bars_held"] == pytest.approx(5.0)
    assert d["regimes"].keys() == {"trending", "ranging"}
    assert d["regimes"]["trending"]["trades"] == 3 and d["regimes"]["trending"]["gross_bp"] == pytest.approx(60.0)
    assert d["regimes"]["ranging"]["trades"] == 1 and d["regimes"]["ranging"]["gross_bp"] == pytest.approx(-40.0)
    b = d["breadth"]
    # net per coin: BTC 18.5 - 12 = 6.5, ETH -5.7 + 3.8 = -1.9; per fold: 12.8, -8.2 and none; total 4.6
    assert (b["coins_net_pos"], b["coins"], b["folds_net_pos"], b["folds"]) == (1, 2, 1, 3)
    assert b["top_coin_share"] == pytest.approx(6.5 / 4.6) and b["top_fold_share"] == pytest.approx(12.8 / 4.6)
    assert d["is_vs_oos"] == {"is_sharpe": 1.2, "oos_sharpe": 0.3}
    assert d["selection"] == {"n=20": 2, "n=10": 1}
    assert d["overlap"] == {"position": [], "mirror": [], "correlation": []}


def test_concentration_is_undefined_when_the_total_net_is_not_positive(cfg):
    losing = [r[:3] + (-abs(r[3]),) + r[4:] for r in TRADES]
    b = diagnostics(cfg, recorded_run(cfg, "r1", trades(losing)), data="judged", earlier=[])["breadth"]
    assert (b["coins_net_pos"], b["folds_net_pos"]) == (0, 0)
    assert b["top_coin_share"] is None and b["top_fold_share"] is None


def test_diagnostics_carry_no_dates_or_run_ids(cfg):
    run_id, other = new_run_id(), new_run_id()
    earlier = [{"strategy": "s0", "timeframe": "1h", "run_dir": str(recorded_run(cfg, other, trades()))}]
    d = diagnostics(cfg, recorded_run(cfg, run_id, trades()), data="judged", earlier=earlier)
    s = json.dumps(d, allow_nan=False)                   # also: plain JSON, no NaN or infinity
    assert d["overlap"]["position"] == [{"strategy": "s0", "timeframe": "1h", "value": 1.0}]
    assert not re.search(r"\b20\d\d-\d\d", s) and run_id not in s and other not in s


def test_overlap_against_earlier_runs(cfg):
    earlier = [{"strategy": "same", "timeframe": "4h", "run_dir": str(recorded_run(cfg, "a", trades()))},
               {"strategy": "flipped", "timeframe": "1h",
                "run_dir": str(recorded_run(cfg, "b", trades(sign=-1), sign=-1))},
               {"strategy": "idle", "timeframe": "15m", "run_dir": str(recorded_run(cfg, "c"))}]
    o = diagnostics(cfg, recorded_run(cfg, "r1", trades()), data="refreshed", earlier=earlier)["overlap"]
    assert o["position"] == [{"strategy": "same", "timeframe": "4h", "value": 1.0}]
    assert o["mirror"] == [{"strategy": "flipped", "timeframe": "1h", "value": 1.0}]
    assert sorted((x["strategy"], x["value"]) for x in o["correlation"]) == [("flipped", -1.0), ("idle", 1.0),
                                                                               ("same", 1.0)]


def test_a_run_without_trades(cfg):
    rnd = {"note": "strategy took no OOS trades; benchmark undefined", "beats_random": False, "long_share": 0.5}
    gross = {"sharpe": 0.0, "random_mean": None, "random_p95": None, "pctile": None, "long_share": 0.5}
    d = diagnostics(cfg, recorded_run(cfg, "r1", random_entry=rnd, before_costs=gross,
                                      metrics={"sharpe": 0.0, "trades": 0}), data="judged", earlier=[])
    assert d["gates"] == {"net_sharpe": 0.0, "random_mean": None, "random_p95": None, "pctile": None,
                          "long_share": 0.5, "oos_trades": 0, "min_oos_trades": 8}
    assert d["before_costs"] == {"sharpe": 0.0, "random_mean": None, "random_p95": None, "pctile": None}
    assert d["per_trade_bp"] == dict.fromkeys(("gross", "fees", "slippage", "funding", "net"))
    assert d["sides"] == {"long": 0, "short": 0, "long_share": None, "long_gross_bp": None, "long_net_bp": None,
                          "short_gross_bp": None, "short_net_bp": None}
    assert d["exits"] == {"share": {}, "mean_bars_held": None} and d["regimes"] == {}
    assert d["breadth"] == {"coins_net_pos": 0, "coins": 0, "folds_net_pos": 0, "folds": 3,
                            "top_coin_share": None, "top_fold_share": None}
    assert d["overlap"] == {"position": [], "mirror": [], "correlation": []}
    json.dumps(d, allow_nan=False)


def test_a_run_from_before_the_gross_pass(cfg):
    d = diagnostics(cfg, recorded_run(cfg, "r1", trades(), drop=("before_costs", "min_oos_trades")),
                    data="judged", earlier=[])
    assert d["before_costs"] is None and d["gates"]["min_oos_trades"] is None
    assert d["gates"]["net_sharpe"] == 0.3


def test_diagnostics_name_where_their_data_came_from(cfg):
    d = recorded_run(cfg, "r1", trades())
    assert diagnostics(cfg, d, data="refreshed", earlier=[])["data"] == "refreshed"
    with pytest.raises(ValueError, match="data"):
        diagnostics(cfg, d, data="holdout", earlier=[])


def test_diagnostics_refuse_a_holdout_run(cfg):
    d = recorded_run(cfg, "h1", trades(), validation_mode="holdout")
    with pytest.raises(ValueError, match="walk-forward"):
        diagnostics(cfg, d, data="judged", earlier=[])


def test_earlier_runs_are_the_runs_of_earlier_judgements(cfg):
    dirs = {r: str(recorded_run(cfg, r, trades())) for r in ("r1", "r2", "r2b", "r3")}
    j1 = db.record_judgement(cfg, jrow(strategy="a", strategy_hash="ha", child_run_id="r1"))
    db.record_judgement(cfg, jrow(strategy="b", strategy_hash="hb"))            # a kill without a run
    j3 = db.record_judgement(cfg, jrow(strategy="c", timeframe="1h", strategy_hash="hc", child_run_id="r2"))
    j4 = db.record_judgement(cfg, jrow(strategy="d", strategy_hash="hd", child_run_id="r3"))
    assert db.earlier_runs(cfg, j1) == []
    assert db.earlier_runs(cfg, j4) == [{"strategy": "a", "timeframe": "4h", "run_dir": dirs["r1"]},
                                        {"strategy": "c", "timeframe": "1h", "run_dir": dirs["r2"]}]
    db.set_diagnostics(cfg, j3, {"data": "refreshed"}, "r2b")                  # a refresh: its run is the latest
    assert db.earlier_runs(cfg, j4)[1] == {"strategy": "c", "timeframe": "1h", "run_dir": dirs["r2b"]}


def test_earlier_runs_never_name_a_holdout_run(cfg):
    recorded_run(cfg, "r1", trades())
    recorded_run(cfg, "h1", trades(), validation_mode="holdout")
    j1 = db.record_judgement(cfg, jrow(strategy="a", strategy_hash="ha", child_run_id="r1"))
    db.set_diagnostics(cfg, j1, {"data": "refreshed"}, "h1")
    j2 = db.record_judgement(cfg, jrow(strategy="b", strategy_hash="hb", child_run_id="r1"))
    assert db.earlier_runs(cfg, j2) == []


def test_set_diagnostics_stores_the_block_and_its_run(cfg):
    recorded_run(cfg, "r1", trades())
    jid = db.record_judgement(cfg, jrow(child_run_id="r1"))
    assert db.get_judgement_by_id(cfg, jid)["diagnostics"] is None
    diag = diagnostics(cfg, cfg.runs_dir / "r1", data="judged", earlier=[])
    db.set_diagnostics(cfg, jid, diag, "r1")
    j = db.get_judgement_by_id(cfg, jid)
    assert j["diagnostics"] == diag and j["diagnostics_run_id"] == "r1"


def test_an_old_store_gains_the_diagnostics_columns(cfg):
    db.record_judgement(cfg, jrow())
    with db.engine(cfg).begin() as c:
        c.execute(sa.text("ALTER TABLE judgements DROP COLUMN diagnostics"))
        c.execute(sa.text("ALTER TABLE judgements DROP COLUMN diagnostics_run_id"))
    db._engine.cache_clear()
    j = db.get_judgement(cfg, "h1", "4h")
    assert j["diagnostics"] is None and j["diagnostics_run_id"] is None
