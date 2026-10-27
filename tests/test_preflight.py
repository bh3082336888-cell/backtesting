"""`lab preflight`: the judge's checks on synthetic bars only, before any trial."""
import json
import logging
import sys

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from perp_lab import preflight as pf
from perp_lab.cli import app
from perp_lab.config import load_config
from perp_lab.judge.judge import JudgeRefused, judge, precheck
from perp_lab.overlap import overlap
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client
from perp_lab.sandbox.errors import PluginError, SandboxUnavailable
from perp_lab.store import db
from perp_lab.strategy import plugin_hash
from perp_lab.validate.synthetic_market import MARKET_VERSION
from tests.conftest import REPO
from tests.judge_fixtures import CHEATER, WIDE_TEXT, StubValidate, add_plugin, plugins  # noqa: F401
from tests.test_runlock import _try_exclusive_r

BASE = (REPO / "strategies" / "donchian_atr.py").read_text()     # params n 40, atr_n 14, atr_mult 3.0; 1h, 4h
SOUND = BASE.replace('"n": 40,', '"n": 44,')                      # a parameter change
COPY = BASE + "\n# the same plugin, other bytes\n"
DELAYED = (BASE.replace('c = ctx["close"]', 'c = ctx.prev("close")')
           .replace('ctx["dc_hi"]', 'ctx.prev("dc_hi")').replace('ctx["dc_lo"]', 'ctx.prev("dc_lo")'))
TWIN_1H = (BASE.replace('params = {"n": 40, "atr_n": 14, "atr_mult": 3.0}',
                        'params = {"n": 160, "atr_n": 56, "atr_mult": 6.0}')   # 4 x the bars, 2 x the ATRs
           .replace('timeframes = ["1h", "4h"]', 'timeframes = ["1h"]'))

# donchian_atr turned over. Negating the side in entry() alone is not enough: that breakout fade
# exits on its own trail and holds other hours (mirror overlap about 0.33 on the synthetic market).
# A mirror holds donchian_atr's cells on the other side, so it is out where donchian_atr is out:
# donchian_atr's chandelier stop is its target, and its own stop is far away.
MIRROR = '''"""donchian_atr turned over: short its breakouts, long its breakdowns, flat when it is flat."""
from perp_lab.indicators import atr, donchian_prior
from perp_lab.strategy import HOLD, Signal, Strategy, enter


class Mirror(Strategy):
    params = {"n": 40, "atr_n": 14, "atr_mult": 3.0}
    grid = {"n": [20, 40, 80], "atr_mult": [2.0, 3.0, 4.0]}
    timeframes = ["4h"]

    def prepare(self, bars):
        hi, lo = donchian_prior(bars, self.p["n"])
        return bars[[]].assign(dc_hi=hi, dc_lo=lo, atr=atr(bars, self.p["atr_n"]))

    def bracket(self, ctx, side):
        return ctx["close"] * (1 - side * 0.25), ctx["close"] + side * self.p["atr_mult"] * ctx["atr"]

    def entry(self, ctx):
        c = ctx["close"]
        if c > ctx["dc_hi"]:
            return enter(-1, *self.bracket(ctx, -1))
        if c < ctx["dc_lo"]:
            return enter(1, *self.bracket(ctx, 1))
        return HOLD

    def manage(self, ctx):
        pos = ctx.position
        a = ctx["atr"]
        if a != a:
            return HOLD
        closes = ctx.hist("close", pos.bars_held)
        if pos.side < 0:
            return Signal("hold", target=max(pos.target, closes.max() - self.p["atr_mult"] * a))
        return Signal("hold", target=min(pos.target, closes.min() + self.p["atr_mult"] * a))
'''

QUIET = '''"""Never trades."""
from perp_lab.strategy import HOLD, Strategy


class Quiet(Strategy):
    timeframes = ["4h"]

    def prepare(self, bars):
        return bars[[]]

    def entry(self, ctx):
        return HOLD
'''

KEYS = {"ok", "meta", "lookahead", "overlap", "mirror_of", "skipped", "error"}
ENTRY_KEYS = {"strategy", "timeframe", "overlap", "mirror", "judged"}


@pytest.fixture
def cfg(tmp_path):
    """A temp config: store, runs, data and the proposer's state (the position cache) all under tmp_path."""
    raw = yaml.safe_load(open("config/lab.yaml"))
    raw.update(runs_dir=str(tmp_path / "runs"), data_dir=str(tmp_path / "data"),
               store={"url": f"sqlite:///{tmp_path}/lab.db"})
    raw["judge"] = {**raw["judge"], "holdout_log": str(tmp_path / "runs" / "holdout_log.jsonl"),
                    "verdicts_dir": str(tmp_path / "runs" / "verdicts")}
    raw["proposer"] = {**raw["proposer"], "state_dir": str(tmp_path / "proposer")}
    p = tmp_path / "lab.yaml"
    p.write_text(yaml.safe_dump(raw))
    return load_config(p)


def stage(tmp_path, name: str, text: str):
    d = tmp_path / "staging"
    d.mkdir(exist_ok=True)
    (d / f"{name}.py").write_text(text)
    return d / f"{name}.py"


def run_cli(cfg, path, tf):
    r = CliRunner().invoke(app, ["preflight", str(path), "--tf", tf, "--config", str(cfg.path)])
    return r.exit_code, json.loads(r.stdout)


def entry(report, strategy, tf):
    [e] = [e for e in report["overlap"] if (e["strategy"], e["timeframe"]) == (strategy, tf)]
    return e


def nothing_recorded(cfg):
    return (db.read_table(cfg, db.judgements).empty and db.read_table(cfg, db.runs).empty
            and db.trial_stats(cfg) == (0, 0.0))


def judged(cfg, name, tf):
    db.record_judgement(cfg, {"strategy": name, "timeframe": tf, "strategy_hash": plugin_hash(name),
                              "child_run_id": None, "parent_run_id": None, "lineage_id": f"family:{name}",
                              "decision": "kill", "reasons": ["x"], "holdout_status": "n/a",
                              "holdout_reason": "n/a"})


@pytest.fixture
def no_lookahead(monkeypatch):
    monkeypatch.setattr(pf, "lookahead_failures", lambda c, s: [])


# ---------------------------------------------------------------- through the real sandbox

@pytest.mark.sandbox
def test_preflight_passes_a_sound_plugin_and_reports_its_nearest(cfg, plugins, tmp_path):
    code, report = run_cli(cfg, stage(tmp_path, "sound", SOUND), "4h")
    assert code == 0 and set(report) == KEYS
    assert report["ok"] is True and report["error"] is None
    assert report["meta"] == report["lookahead"] == report["mirror_of"] == [] and report["skipped"] == {}
    top = report["overlap"][0]
    assert set(top) == ENTRY_KEYS and (top["strategy"], top["timeframe"]) == ("donchian_atr", "4h")
    assert top["overlap"] >= 0.5 and top["mirror"] < 0.1 and top["judged"] is False
    scores = [max(e["overlap"], e["mirror"]) for e in report["overlap"]]
    assert scores == sorted(scores, reverse=True) and min(scores) >= 0.1
    assert nothing_recorded(cfg)                                          # no trial, no judgement, no run
    cache = tmp_path / "proposer" / "cache" / "positions"
    assert (cache / f"{PluginSource.from_text('sound', SOUND).hash}@4h@v{MARKET_VERSION}.parquet").exists()
    assert (cache / f"{plugin_hash('donchian_atr')}@1h@v{MARKET_VERSION}.parquet").exists()   # each declared tf


@pytest.mark.sandbox
def test_preflight_fails_a_lookahead_plugin_as_the_judge_does(cfg, plugins, tmp_path):
    code, report = run_cli(cfg, stage(tmp_path, "cheater", CHEATER), "4h")
    assert code == 3 and report["ok"] is False and report["error"] is None
    assert report["lookahead"] and report["meta"] == [] and report["skipped"] == {}   # the overlap ran too
    cache = tmp_path / "proposer" / "cache" / "positions"
    assert (cache / f"{PluginSource.from_text('cheater', CHEATER).hash}@4h@v{MARKET_VERSION}.parquet").exists()
    assert nothing_recorded(cfg)
    add_plugin(plugins, "cheater", CHEATER)
    j = judge(cfg, "cheater", "4h", validate=StubValidate({}))      # the same UTC day: the same span
    assert j["decision"] == "kill" and j["reasons"] == [f"lookahead: {f}" for f in report["lookahead"]]


@pytest.mark.sandbox
def test_a_refused_plugin_is_still_checked_for_lookahead_and_overlap(cfg, plugins, tmp_path):
    hourly = SOUND.replace('timeframes = ["1h", "4h"]', 'timeframes = ["1h"]')
    code, report = run_cli(cfg, stage(tmp_path, "hourly", hourly), "4h")
    assert code == 3 and report["meta"] == ["hourly declares timeframes 1h, not 4h"]
    assert report["lookahead"] == [] and report["skipped"] == {} and report["error"] is None   # it ran: causal
    assert entry(report, "donchian_atr", "4h")["overlap"] >= 0.5                                # and a near-copy


@pytest.mark.sandbox
def test_preflight_refuses_a_mirror_of_a_judged_plugin(cfg, plugins, tmp_path):
    judged(cfg, "donchian_atr", "4h")
    code, report = run_cli(cfg, stage(tmp_path, "mirror", MIRROR), "4h")
    assert code == 3 and report["ok"] is False and report["mirror_of"] == ["donchian_atr@4h"]
    assert report["meta"] == report["lookahead"] == [] and report["error"] is None
    e = entry(report, "donchian_atr", "4h")
    assert e["mirror"] >= cfg["proposer"]["overlap_max"] and e["overlap"] < 0.1 and e["judged"] is True
    assert entry(report, "donchian_atr", "1h")["judged"] is False      # judged at 4h only


@pytest.mark.sandbox
def test_overlap_sees_a_byte_different_copy_a_one_bar_delay_and_a_param_change(cfg, plugins, tmp_path,
                                                                             no_lookahead):
    # The lookahead check is the cheater test's; here only the positions matter.
    reports = {name: pf.preflight(cfg, stage(tmp_path, name, text), "4h")
               for name, text in (("copy", COPY), ("delayed", DELAYED), ("sound", SOUND))}
    for name, report in reports.items():
        assert report["ok"] is True, name                          # a near-copy is reported, not refused
        assert entry(report, "donchian_atr", "4h")["overlap"] >= 0.5, name
    assert entry(reports["copy"], "donchian_atr", "4h")["overlap"] == 1.0


@pytest.mark.sandbox
def test_a_1h_plugin_and_its_4h_twin_overlap_on_the_shared_market(cfg, plugins, tmp_path, no_lookahead):
    report = pf.preflight(cfg, stage(tmp_path, "twin", TWIN_1H), "1h")
    top = report["overlap"][0]
    assert (top["strategy"], top["timeframe"]) == ("donchian_atr", "4h") and top["overlap"] >= 0.5
    assert entry(report, "donchian_atr", "1h")["overlap"] < top["overlap"]


@pytest.mark.sandbox
def test_preflight_never_reads_data(cfg, plugins, tmp_path, monkeypatch):
    import perp_lab.data.store as store
    data = cfg.data_dir
    (data / "BTCUSDT" / "4h").mkdir(parents=True)
    (data / "BTCUSDT" / "4h" / "all.parquet").write_bytes(b"not for plugins")

    def refuse(*a, **k):
        raise AssertionError("preflight read market data")
    for fn in ("load_bars", "research_symbols"):           # and every module that imported it by name
        real = getattr(store, fn)
        for mod in list(sys.modules.values()):
            if getattr(mod, fn, None) is real:
                monkeypatch.setattr(mod, fn, refuse)
    data.chmod(0)
    try:
        report = pf.preflight(cfg, stage(tmp_path, "sound", SOUND), "4h")
    finally:
        data.chmod(0o755)
    assert report["ok"] is True and report["overlap"]


# ---------------------------------------------------------------- the report, with stubbed runs

def held(cells: int, side: int = 1) -> pd.Series:
    """Positions on SYN1 for the first `cells` hours."""
    hours = np.arange(cells, dtype="int64")
    idx = pd.MultiIndex.from_arrays([np.array(["SYN1"] * cells, dtype=object), hours], names=["symbol", "hour"])
    return pd.Series(np.full(cells, side, dtype="int8"), index=idx)


def stub_runs(monkeypatch, table: dict):
    """positions_of by (plugin, tf) from `table`: a Series, or an exception to raise; others hold nothing."""
    calls = []

    def fake(cfg, src, tf):
        calls.append((src.name, tf))
        got = table.get((src.name, tf), held(0))
        if isinstance(got, Exception):
            raise got
        return got
    monkeypatch.setattr(pf, "positions_of", fake)
    monkeypatch.setattr(pf, "lookahead_failures", lambda c, s: [])
    return calls


def test_a_mirror_is_refused_only_when_its_original_was_judged(cfg, plugins, tmp_path, monkeypatch):
    stub_runs(monkeypatch, {("cand", "4h"): held(100), ("donchian_atr", "4h"): held(100, side=-1)})
    path = stage(tmp_path, "cand", SOUND)
    report = pf.preflight(cfg, path, "4h")
    assert report["ok"] is True and report["mirror_of"] == []
    assert report["overlap"] == [{"strategy": "donchian_atr", "timeframe": "4h", "overlap": 0.0, "mirror": 1.0,
                                  "judged": False}]
    judged(cfg, "donchian_atr", "4h")
    report = pf.preflight(cfg, path, "4h")
    assert report["ok"] is False and report["mirror_of"] == ["donchian_atr@4h"]
    assert report["overlap"][0]["judged"] is True


@pytest.mark.parametrize("cells,refused", [(49, False), (50, True)])
def test_a_mirror_is_one_at_overlap_max_or_above(cfg, plugins, tmp_path, monkeypatch, cells, refused):
    stub_runs(monkeypatch, {("cand", "4h"): held(100), ("donchian_atr", "4h"): held(cells, side=-1)})
    judged(cfg, "donchian_atr", "4h")
    report = pf.preflight(cfg, stage(tmp_path, "cand", SOUND), "4h")
    assert report["overlap"][0]["mirror"] == cells / 100
    assert report["ok"] is not refused and report["mirror_of"] == (["donchian_atr@4h"] if refused else [])


def test_the_report_lists_the_top_10_at_or_above_a_tenth(cfg, plugins, tmp_path, monkeypatch):
    others = {f"other{k:02d}": k * 5 for k in range(14)}           # overlaps 0.0, 0.05, ..., 0.65
    for name in others:
        add_plugin(plugins, name, BASE.replace('timeframes = ["1h", "4h"]', 'timeframes = ["4h"]')
                   + f"\n# {name}\n")
    table = {(n, "4h"): held(c) for n, c in others.items()}
    stub_runs(monkeypatch, {**table, ("cand", "4h"): held(100), ("donchian_atr_wide", "4h"): held(30, side=-1)})
    report = pf.preflight(cfg, stage(tmp_path, "cand", SOUND), "4h")
    got = [(e["strategy"], e["overlap"], e["mirror"]) for e in report["overlap"]]
    assert got == [(f"other{k:02d}", k * 5 / 100, 0.0) for k in (13, 12, 11, 10, 9, 8, 7)] + [
        ("donchian_atr_wide", 0.0, 0.3), ("other06", 0.3, 0.0), ("other05", 0.25, 0.0)]   # a tie: by name


def test_a_judged_mirror_is_refused_even_outside_the_top_10(cfg, plugins, tmp_path, monkeypatch):
    others = {f"near{k:02d}": 60 + k for k in range(10)}           # ten near-copies, all above the mirror
    for name in others:
        add_plugin(plugins, name, BASE.replace('timeframes = ["1h", "4h"]', 'timeframes = ["4h"]')
                   + f"\n# {name}\n")
    stub_runs(monkeypatch, {**{(n, "4h"): held(c) for n, c in others.items()}, ("cand", "4h"): held(100),
                            ("donchian_atr", "4h"): held(55, side=-1)})
    judged(cfg, "donchian_atr", "4h")
    report = pf.preflight(cfg, stage(tmp_path, "cand", SOUND), "4h")
    assert [e["strategy"] for e in report["overlap"]] == sorted(others, reverse=True)
    assert report["mirror_of"] == ["donchian_atr@4h"] and report["ok"] is False


def test_every_other_plugin_runs_at_each_timeframe_it_declares(cfg, plugins, tmp_path, monkeypatch):
    calls = stub_runs(monkeypatch, {})
    add_plugin(plugins, "sound", SOUND)                              # the file's own name: not compared
    pf.preflight(cfg, stage(tmp_path, "sound", SOUND), "4h")
    assert sorted(calls) == [("donchian_atr", "1h"), ("donchian_atr", "4h"), ("donchian_atr_wide", "4h"),
                             ("sound", "4h")]


def test_an_earlier_plugin_that_fails_holds_nothing_and_is_logged(cfg, plugins, tmp_path, monkeypatch, caplog):
    stub_runs(monkeypatch, {("cand", "4h"): held(100), ("donchian_atr", "4h"): held(100),
                            ("donchian_atr_wide", "4h"): PluginError("exception", "ZeroDivisionError", "boom")})
    add_plugin(plugins, "computed", BASE.replace('timeframes = ["1h", "4h"]', 'timeframes = ["1h"] * 2'))
    with caplog.at_level(logging.WARNING, logger="perp_lab.preflight"):
        report = pf.preflight(cfg, stage(tmp_path, "cand", SOUND), "4h")
    assert report["ok"] is True and [e["strategy"] for e in report["overlap"]] == ["donchian_atr"]
    assert "donchian_atr_wide @ 4h" in caplog.text and "ZeroDivisionError" in caplog.text
    assert "computed" in caplog.text and "metadata" in caplog.text


def test_a_plugin_that_fails_on_the_synthetic_market_fails_preflight(cfg, plugins, tmp_path, monkeypatch):
    calls = stub_runs(monkeypatch, {("cand", "4h"): PluginError("exception", "ZeroDivisionError", "boom")})
    report = pf.preflight(cfg, stage(tmp_path, "cand", SOUND), "4h")
    assert report["ok"] is False and report["overlap"] == report["lookahead"] == [] and calls == [("cand", "4h")]
    assert report["skipped"] == {"overlap": "the plugin fails on the synthetic market @ 4h: "
                                            "plugin_error: exception: ZeroDivisionError: boom"}


def test_a_lookahead_failing_near_copy_still_reports_its_overlap_and_mirror_of(cfg, plugins, tmp_path,
                                                                              monkeypatch):
    # The model hears every defect at once: a peeking near-copy, and a mirror, in one report.
    stub_runs(monkeypatch, {("cand", "4h"): held(100), ("donchian_atr", "4h"): held(100),
                            ("donchian_atr_wide", "4h"): held(100, side=-1)})
    monkeypatch.setattr(pf, "lookahead_failures", lambda c, s: ["defaults: AssertionError: features at cut 400"])
    judged(cfg, "donchian_atr_wide", "4h")
    report = pf.preflight(cfg, stage(tmp_path, "cand", SOUND), "4h")
    assert report["ok"] is False and report["lookahead"] == ["defaults: AssertionError: features at cut 400"]
    assert [(e["strategy"], e["overlap"], e["mirror"]) for e in report["overlap"]] == [
        ("donchian_atr", 1.0, 0.0), ("donchian_atr_wide", 0.0, 1.0)]
    assert report["mirror_of"] == ["donchian_atr_wide@4h"] and report["skipped"] == {}


@pytest.mark.parametrize("name,text,tf", [
    ("computed", BASE.replace('"n": 40,', '"n": 4 * 10,'), "4h"),
    ("broken", BASE.replace("def entry(self, ctx):", "def entry(self, ctx)"), "4h"),
], ids=["computed", "broken"])
def test_unreadable_metadata_skips_every_later_stage_and_says_so(cfg, plugins, tmp_path, monkeypatch,
                                                                 name, text, tf):
    # Nothing can run without literal metadata, so nothing does, on any machine; the report says why.
    monkeypatch.setattr(pf, "lookahead_failures", lambda *a: pytest.fail("ran the lookahead check"))
    monkeypatch.setattr(pf, "positions_of", lambda *a: pytest.fail("ran the plugin"))
    monkeypatch.setattr(client, "check", lambda cfg: pytest.fail("checked the sandbox"))
    report = pf.preflight(cfg, stage(tmp_path, name, text), tf)
    assert report["ok"] is False and len(report["meta"]) == 1 and report["meta"][0].startswith(f"{name}: ")
    why = "its metadata can't be read, so the plugin can't run"
    assert report["skipped"] == {"lookahead": why, "overlap": why}
    assert report["lookahead"] == report["overlap"] == report["mirror_of"] == [] and report["error"] is None


@pytest.mark.parametrize("name,text,tf,extra,msg", [
    ("hourly", SOUND, "15m", {}, "hourly declares timeframes 1h, 4h, not 15m"),
    ("orphan", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "nope"'), "4h", {},
     "PARENT 'nope' is not a plugin in strategies/"),
    ("selfish", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "selfish"'), "4h", {},
     "selfish names itself as its PARENT"),
    ("twin_a", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "twin_b"'), "4h",
     {"twin_b": WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "twin_b"')},
     "twin_a is byte-identical to its parent twin_b; not a variation"),
    ("kid", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "hourly_root"'), "4h",
     {"hourly_root": BASE.replace('timeframes = ["1h", "4h"]', 'timeframes = ["1h"]')},
     "parent hourly_root declares timeframes 1h, not 4h"),
], ids=["undeclared-tf", "orphan", "selfish", "byte-identical", "parent-undeclared-tf"])
def test_a_source_refusal_still_runs_the_lookahead_check_and_the_overlap(cfg, plugins, tmp_path, monkeypatch,
                                                                       name, text, tf, extra, msg):
    # The judge would refuse it, but the plugin can run: the rest is checked too, at the asked timeframe.
    for n, t in extra.items():
        add_plugin(plugins, n, t)
    calls = stub_runs(monkeypatch, {(name, tf): held(100), ("donchian_atr", "4h"): held(100)})
    monkeypatch.setattr(pf, "lookahead_failures", lambda c, s: calls.append(("lookahead", s.name)) or ["x: y"])
    report = pf.preflight(cfg, stage(tmp_path, name, text), tf)
    assert report["ok"] is False and report["meta"] == [msg] and report["skipped"] == {}
    assert report["lookahead"] == ["x: y"] and calls[:2] == [("lookahead", name), (name, tf)]
    got = [(e["strategy"], e["timeframe"], e["overlap"]) for e in report["overlap"]]
    assert got == [("donchian_atr", "4h", 1.0)]


def test_a_timeframe_without_a_market_skips_the_overlap_and_says_so(cfg, plugins, tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "lookahead_failures", lambda c, s: [])
    onem = SOUND.replace('timeframes = ["1h", "4h"]', 'timeframes = ["1m"]')
    report = pf.preflight(cfg, stage(tmp_path, "onem", onem), "1m")
    assert report["ok"] is False and report["meta"] == report["lookahead"] == report["overlap"] == []
    assert report["skipped"] == {"overlap": "no synthetic market at 1m: its bars are 5m, resampled to multiples "
                                            "of 5m"}


def test_precheck_is_the_judges_source_only_refusals(plugins):
    assert precheck(PluginSource.read("donchian_atr_wide"), "4h") is None      # a sound child passes
    assert precheck(PluginSource.read("donchian_atr"), "1h") is None           # and a root
    with pytest.raises(JudgeRefused, match="donchian_atr_wide declares timeframes 4h, not 1h"):
        precheck(PluginSource.read("donchian_atr_wide"), "1h")
    with pytest.raises(JudgeRefused, match="PARENT 'nope' is not a plugin in strategies/"):
        precheck(PluginSource.from_text("orphan", WIDE_TEXT.replace('"donchian_atr"', '"nope"')), "4h")


def test_preflight_without_a_sandbox_raises_unless_nothing_can_run(cfg, plugins, tmp_path, monkeypatch):
    def no(cfg):
        raise SandboxUnavailable("no landlock")
    monkeypatch.setattr(client, "check", no)
    monkeypatch.setattr(pf, "lookahead_failures", lambda *a: pytest.fail("ran the lookahead check"))
    computed = BASE.replace('"n": 40,', '"n": 4 * 10,')
    assert pf.preflight(cfg, stage(tmp_path, "computed", computed), "4h")["meta"]     # answered on any machine
    for name, text, tf in (("hourly", SOUND, "15m"), ("sound", SOUND, "4h")):        # a refusal still runs code
        with pytest.raises(SandboxUnavailable):
            pf.preflight(cfg, stage(tmp_path, name, text), tf)


# ---------------------------------------------------------------- the position cache

def test_the_position_cache_is_keyed_by_hash_timeframe_and_market_version(cfg, monkeypatch, tmp_path):
    runs = []
    real = client.fresh_run
    monkeypatch.setattr(client, "fresh_run", lambda *a, **k: runs.append(a[3]) or real(*a, **k))
    src = PluginSource.from_text("donchian_atr", BASE)
    cache = tmp_path / "proposer" / "cache" / "positions"

    first = pf.positions_of(cfg, src, "4h")
    assert len(first) > 0 and runs == ["4h"] * 4                   # one run per synthetic symbol
    assert (cache / f"{src.hash}@4h@v{MARKET_VERSION}.parquet").exists()
    again = pf.positions_of(cfg, src, "4h")
    assert runs == ["4h"] * 4                                       # read back, nothing ran
    pd.testing.assert_series_equal(again, first)

    pf.positions_of(cfg, src, "1h")                                 # another timeframe
    other = PluginSource.from_text("donchian_atr", COPY)            # other bytes, the same name
    pf.positions_of(cfg, other, "4h")
    monkeypatch.setattr(pf, "MARKET_VERSION", MARKET_VERSION + 1)  # another market
    pd.testing.assert_series_equal(pf.positions_of(cfg, src, "4h"), first)
    assert runs == ["4h"] * 4 + ["1h"] * 4 + ["4h"] * 8
    assert sorted(p.name for p in cache.iterdir()) == sorted([
        f"{src.hash}@4h@v{MARKET_VERSION}.parquet", f"{src.hash}@1h@v{MARKET_VERSION}.parquet",
        f"{other.hash}@4h@v{MARKET_VERSION}.parquet", f"{src.hash}@4h@v{MARKET_VERSION + 1}.parquet"])


def test_a_plugin_that_never_trades_is_cached_as_holding_nothing(cfg, monkeypatch):
    src = PluginSource.from_text("quiet", QUIET)
    empty = pf.positions_of(cfg, src, "4h")
    monkeypatch.setattr(client, "fresh_run", lambda *a, **k: pytest.fail("ran again"))
    pd.testing.assert_series_equal(pf.positions_of(cfg, src, "4h"), empty)
    assert len(empty) == 0 and overlap(empty, empty) == 0.0


def test_a_failed_run_is_not_cached(cfg, monkeypatch, tmp_path):
    def boom(*a, **k):
        raise PluginError("exception", "ZeroDivisionError", "boom")
    monkeypatch.setattr(client, "fresh_run", boom)
    with pytest.raises(PluginError):
        pf.positions_of(cfg, PluginSource.from_text("x", BASE), "4h")
    assert not (tmp_path / "proposer" / "cache" / "positions").exists() or not any(
        (tmp_path / "proposer" / "cache" / "positions").iterdir())


# ---------------------------------------------------------------- the command

@pytest.mark.parametrize("outcome,code", [
    ({"ok": True}, 0),
    ({"ok": False, "lookahead": ["defaults: x"]}, 3),
    ({"ok": False, "mirror_of": ["donchian_atr@4h"]}, 3),
    ({"ok": False, "skipped": {"overlap": "no synthetic market at 5m"}}, 3),
    (SandboxUnavailable("no landlock"), 1),
    (RuntimeError("boom"), 1),
])
def test_cli_preflight_exit_codes(cfg, monkeypatch, tmp_path, outcome, code):
    seen = []

    def stub(cfg_, path, tf):
        seen.append((path, tf, _try_exclusive_r(cfg_)))
        if isinstance(outcome, Exception):
            raise outcome
        return {**pf.blank_report(), **outcome}
    monkeypatch.setattr(pf, "preflight", stub)
    path = stage(tmp_path, "x", SOUND)
    r = CliRunner().invoke(app, ["preflight", str(path), "--tf", "4h", "--config", str(cfg.path)])
    assert r.exit_code == code and seen == [(path, "4h", False)]          # run under the shared lock
    report = json.loads(r.stdout)                                          # stdout is the report alone
    assert set(report) == KEYS and report["ok"] is (code == 0)
    if code == 1:
        assert report["error"] == f"{type(outcome).__name__}: {outcome}"
    else:
        assert report["error"] is None
    assert _try_exclusive_r(cfg)                                           # and released afterwards


def test_the_proposer_config_is_the_specs():
    p = load_config()["proposer"]
    assert (p["model"], str(p["model_training_cutoff"]), p["claude_bin"], p["effort"], p["state_dir"]) == (
        "claude-opus-5-5", "2026-06-30", "claude", "high", "proposer")
    assert (p["timeframes"], p["trials_per_day"], p["trials_total"], p["proposals_per_idea"],
            p["fixups_per_proposal"], p["max_grid_combos"]) == (["15m", "1h", "4h"], 12, 100, 4, 3, 16)
    assert (p["overlap_max"], p["corr_max"], p["max_input_tokens_per_day"], p["max_output_tokens_per_day"],
            p["call_timeout_s"]) == (0.5, 0.8, 4_000_000, 500_000, 1800)
