import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from perp_lab.cli import app
from perp_lab.config import Config
from perp_lab.sandbox import client
from perp_lab.sandbox.errors import PluginError, SandboxUnavailable
from perp_lab.judge.judge import VERDICT_KEYS, JudgeRefused, TransientFailure, judge, lookahead_failures
from perp_lab.store import db
from perp_lab.strategy import plugin_hash
from tests.judge_fixtures import (CHEATER, NOISE, STRONG, WEAK, WIDE_TEXT, StubValidate, add_plugin,
                                  plugins, series)  # noqa: F401
from tests.conftest import REAL_PLUGINS, REPO, market_config
from tests.test_judge_exam import fresh_wf_caches  # noqa: F401
from tests.test_store import cfg  # noqa: F401


def test_root_noise_is_killed_with_its_reasons(cfg, plugins):
    j = judge(cfg, "donchian_atr", "4h", validate=StubValidate({"donchian_atr": ("noise", NOISE)}))
    assert j["decision"] == "kill" and "does not beat random-entry p95 OOS" in j["reasons"]
    assert (j["holdout_status"], j["holdout_reason"]) == ("n/a", "n/a") and j["parent_run_id"] is None
    assert db.read_table(cfg, db.lineage_edges).empty
    out = json.loads(Path(j["verdict_path"]).read_text())
    assert set(out) == VERDICT_KEYS and out["decision"] == "kill" and out["walkforward"]["run_id"] == j["child_run_id"]


def test_strong_root_is_promoted_and_awaits_its_exam(cfg, plugins):
    j = judge(cfg, "donchian_atr", "4h", validate=StubValidate({"donchian_atr": ("candidate", STRONG)}))
    assert j["decision"] == "promote" and j["reasons"] == []
    assert (j["holdout_status"], j["holdout_reason"]) == ("pending", "awaiting_exam")
    assert j["dsr"] > 0.99 and j["n_trials"] == 1 and j["sr0"] == 0.0


def test_candidate_with_a_weak_deflated_sharpe_is_killed(cfg, plugins):
    j = judge(cfg, "donchian_atr", "4h", validate=StubValidate({"donchian_atr": ("candidate", WEAK)}))
    assert j["decision"] == "kill" and len(j["reasons"]) == 1
    assert j["reasons"][0].startswith("Deflated Sharpe 0.78 < 0.95")


def test_child_must_beat_a_passing_parent(cfg, plugins):
    v = StubValidate({"donchian_atr": ("candidate", STRONG), "donchian_atr_wide": ("candidate", STRONG - 0.001)})
    j = judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert v.calls == ["donchian_atr", "donchian_atr_wide"]          # the parent's walk-forward runs first
    assert j["decision"] == "kill" and j["parent_passed"] is True and j["beats_parent_frac"] == 0.0
    assert j["reasons"] == ["beats its passing parent in 0% of resamples < 95%"]
    e = db.read_table(cfg, db.lineage_edges)
    assert e.decision.tolist() == ["kill"] and e.decided_by.tolist() == ["harness"]


def test_child_that_beats_a_passing_parent_is_promoted_into_its_family(cfg, plugins):
    v = StubValidate({"donchian_atr": ("candidate", STRONG), "donchian_atr_wide": ("candidate", STRONG + 0.001)})
    j = judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert j["decision"] == "promote" and j["beats_parent_frac"] == 1.0
    assert j["lineage_id"] == db.run_row(cfg, j["parent_run_id"])["lineage_id"] == "family:donchian_atr"


def test_a_noise_parent_neither_lowers_nor_raises_the_bar(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE), "donchian_atr_wide": ("candidate", STRONG)})
    j = judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert j["decision"] == "promote" and j["parent_passed"] is False and j["beats_parent_frac"] is None


def test_an_existing_parent_run_is_reused(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE), "donchian_atr_wide": ("candidate", STRONG)})
    v(cfg, "donchian_atr", "4h", lineage_id="family:donchian_atr")
    judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert v.calls == ["donchian_atr", "donchian_atr_wide"]


def test_rejudging_an_unchanged_strategy_returns_the_existing_judgement(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE)})
    a = judge(cfg, "donchian_atr", "4h", validate=v)
    b = judge(cfg, "donchian_atr", "4h", validate=v)
    assert a["id"] == b["id"] and v.calls == ["donchian_atr"] and b["existing"] is True


def test_a_lookahead_cheater_is_killed_without_a_walk_forward_run(cfg, plugins):
    add_plugin(plugins, "cheater", CHEATER)
    v = StubValidate({})
    j = judge(cfg, "cheater", "4h", validate=v)
    assert j["decision"] == "kill" and j["child_run_id"] is None and v.calls == []
    assert j["reasons"] and all(r.startswith("lookahead: ") for r in j["reasons"])
    assert db.trial_stats(cfg) == (0, 0.0)
    assert json.loads(Path(j["verdict_path"]).read_text())["walkforward"] is None


@pytest.mark.parametrize("name,parent,msg", [("orphan", "nope", "not a plugin"), ("selfish", "selfish", "itself")])
def test_unknown_or_self_parent_is_refused(cfg, plugins, name, parent, msg):
    add_plugin(plugins, name, WIDE_TEXT.replace('PARENT = "donchian_atr"', f'PARENT = "{parent}"'))
    with pytest.raises(JudgeRefused, match=msg):
        judge(cfg, name, "4h", validate=StubValidate({}))


def test_a_byte_identical_child_is_refused(cfg, plugins):
    text = WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "twin_b"')
    add_plugin(plugins, "twin_a", text)
    add_plugin(plugins, "twin_b", text)
    with pytest.raises(JudgeRefused, match="byte-identical"):
        judge(cfg, "twin_a", "4h", validate=StubValidate({}))


def test_a_root_keeps_its_family_across_timeframes(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE)})
    a = judge(cfg, "donchian_atr", "4h", validate=v)
    b = judge(cfg, "donchian_atr", "1h", validate=v)
    assert a["lineage_id"] is not None and a["lineage_id"] == b["lineage_id"]


def test_judge_refuses_a_timeframe_the_plugin_does_not_declare(cfg, plugins, monkeypatch):
    # A walk-forward is a trial and can't be taken back: donchian_atr_wide declares 4h only, so
    # 1h is refused before any check runs or anything is recorded or written.
    monkeypatch.setattr("perp_lab.judge.judge.lookahead_failures", lambda *a, **k: pytest.fail("checked"))
    v = StubValidate({"donchian_atr_wide": ("candidate", STRONG)})
    with pytest.raises(JudgeRefused, match=r"donchian_atr_wide declares timeframes 4h, not 1h"):
        judge(cfg, "donchian_atr_wide", "1h", validate=v)
    assert v.calls == [] and db.read_table(cfg, db.judgements).empty and db.read_table(cfg, db.runs).empty
    assert not Path(cfg["judge"]["verdicts_dir"]).exists()
    r = CliRunner().invoke(app, ["judge", "donchian_atr_wide", "--tf", "1h", "--config", str(cfg.path)])
    assert r.exit_code == 2 and "refused: donchian_atr_wide declares timeframes 4h, not 1h" in r.output
    assert db.read_table(cfg, db.judgements).empty and not Path(cfg["judge"]["verdicts_dir"]).exists()


def test_judge_refuses_a_timeframe_the_parent_does_not_declare(cfg, plugins):
    # The parent's walk-forward runs at the child's timeframe, and it is a trial too.
    root = (plugins / "donchian_atr.py").read_text()
    assert 'timeframes = ["1h", "4h"]' in root
    add_plugin(plugins, "hourly_root", root.replace('timeframes = ["1h", "4h"]', 'timeframes = ["1h"]'))
    add_plugin(plugins, "kid", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "hourly_root"'))
    v = StubValidate({"hourly_root": ("candidate", STRONG), "kid": ("candidate", STRONG)})
    with pytest.raises(JudgeRefused, match=r"parent hourly_root declares timeframes 1h, not 4h"):
        judge(cfg, "kid", "4h", validate=v)
    assert v.calls == [] and db.read_table(cfg, db.judgements).empty and db.read_table(cfg, db.runs).empty


def test_judge_at_a_declared_timeframe_goes_ahead(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE)})
    assert judge(cfg, "donchian_atr", "1h", validate=v)["timeframe"] == "1h" and v.calls == ["donchian_atr"]


def test_cli_refusal_exits_2(cfg, plugins):
    add_plugin(plugins, "orphan", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "nope"'))
    r = CliRunner().invoke(app, ["judge", "orphan", "--tf", "4h", "--config", str(cfg.path)])
    assert r.exit_code == 2 and "refused" in r.output


def test_a_plain_walk_forward_run_from_another_family_is_not_reused(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE), "donchian_atr_wide": ("candidate", STRONG)})
    v(cfg, "donchian_atr", "1h")
    v(cfg, "donchian_atr", "4h")
    j = judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert v.calls == ["donchian_atr", "donchian_atr", "donchian_atr", "donchian_atr_wide"]
    assert j["lineage_id"] == "family:donchian_atr"
    assert db.run_row(cfg, j["parent_run_id"])["lineage_id"] == "family:donchian_atr"
    assert judge(cfg, "donchian_atr", "4h", validate=v)["lineage_id"] == "family:donchian_atr"


def test_a_grandchild_judged_first_shares_the_root_family(cfg, plugins):
    text = WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "donchian_atr_wide"') + "# grandchild variant\n"
    add_plugin(plugins, "grandchild", text)
    v = StubValidate({"donchian_atr_wide": ("noise", NOISE), "grandchild": ("candidate", STRONG),
                      "donchian_atr": ("noise", NOISE)})
    a = judge(cfg, "grandchild", "4h", validate=v)
    b = judge(cfg, "donchian_atr", "4h", validate=v)
    assert a["lineage_id"] == b["lineage_id"] == "family:donchian_atr"


def test_a_parent_chain_loop_is_refused(cfg, plugins):
    add_plugin(plugins, "loop_a", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "loop_b"') + "# a\n")
    add_plugin(plugins, "loop_b", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "loop_a"') + "# b\n")
    with pytest.raises(JudgeRefused, match="loops"):
        judge(cfg, "loop_a", "4h", validate=StubValidate({}))


def test_a_lookahead_kill_belongs_to_its_family(cfg, plugins):
    add_plugin(plugins, "cheater", CHEATER)
    j = judge(cfg, "cheater", "4h", validate=StubValidate({}))
    assert j["decision"] == "kill" and j["lineage_id"] == "family:donchian_atr"


def test_rejudging_rewrites_a_missing_verdict(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE)})
    a = judge(cfg, "donchian_atr", "4h", validate=v)
    Path(a["verdict_path"]).unlink()
    b = judge(cfg, "donchian_atr", "4h", validate=v)
    assert b["existing"] is True and b["id"] == a["id"] and Path(b["verdict_path"]).exists()


def test_a_renamed_identical_copy_returns_the_original_judgement(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE)})
    a = judge(cfg, "donchian_atr", "4h", validate=v)
    add_plugin(plugins, "copy_root", (plugins / "donchian_atr.py").read_text())
    b = judge(cfg, "copy_root", "4h", validate=v)
    assert b["existing"] is True and b["id"] == a["id"] and b["strategy"] == "donchian_atr"
    assert b["verdict_path"] == a["verdict_path"] and Path(b["verdict_path"]).exists()


def test_cli_names_the_judgement_it_returns(cfg, plugins):
    judge(cfg, "donchian_atr", "4h", validate=StubValidate({"donchian_atr": ("noise", NOISE)}))
    add_plugin(plugins, "copy_root", (plugins / "donchian_atr.py").read_text())
    r = CliRunner().invoke(app, ["judge", "copy_root", "--tf", "4h", "--config", str(cfg.path)])
    assert r.exit_code == 0 and "donchian_atr @ 4h" in r.output and "already judged" in r.output


def test_a_judged_child_keeps_its_family_when_its_parent_line_is_removed(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE), "donchian_atr_wide": ("noise", NOISE),
                      "grandchild": ("noise", NOISE)})
    a = judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert a["lineage_id"] == "family:donchian_atr"
    f = plugins / "donchian_atr_wide.py"
    f.write_text(f.read_text().replace('PARENT = "donchian_atr"\n', "") + "# now a root\n")
    b = judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert b["id"] != a["id"] and b["parent_run_id"] is None and b["lineage_id"] == "family:donchian_atr"
    add_plugin(plugins, "grandchild", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "donchian_atr_wide"')
               + "# grandchild variant\n")
    assert judge(cfg, "grandchild", "4h", validate=v)["lineage_id"] == "family:donchian_atr"


def test_a_parent_that_fails_the_lookahead_check_is_refused(cfg, plugins):
    add_plugin(plugins, "cheater", CHEATER)
    add_plugin(plugins, "honest_child", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "cheater"'))
    v = StubValidate({"cheater": ("candidate", STRONG), "honest_child": ("candidate", STRONG)})
    with pytest.raises(JudgeRefused, match="parent cheater fails the lookahead check; fix or judge it first"):
        judge(cfg, "honest_child", "4h", validate=v)
    assert v.calls == [] and db.read_table(cfg, db.judgements).empty and db.read_table(cfg, db.runs).empty


def test_a_broken_trial_does_not_raise_the_bar(cfg, plugins):
    # a cost-ruined trial (Sharpe far below zero) widens the spread of past trials' Sharpes, but
    # SR0 uses the no-edge sampling spread for the candidate's track length: 1/(1461-1) at N=2
    v = StubValidate({"donchian_atr_wide": ("noise", series(-0.02, seed=9)), "donchian_atr": ("candidate", STRONG)})
    v(cfg, "donchian_atr_wide", "4h")
    j = judge(cfg, "donchian_atr", "4h", validate=v)
    assert j["n_trials"] == 2 and j["sr0"] == pytest.approx(0.0136026191, abs=1e-9)


DATE_PEEK = '''"""Honest momentum on ordinary dates; peeks at the next close on holdout dates."""
import numpy as np
import pandas as pd
from perp_lab.indicators import atr
from perp_lab.strategy import HOLD, Strategy, enter

PARENT = "donchian_atr"


class DatePeek(Strategy):
    params = {}
    timeframes = ["4h"]

    def prepare(self, bars):
        late = bars.index >= pd.Timestamp("2026-07-01", tz="UTC")
        honest = np.sign(bars.close - bars.close.shift(1))
        peek = np.sign(bars.close.shift(-1) - bars.close)
        return bars[[]].assign(sig=honest.where(~late, peek), atr=atr(bars, 14))

    def bracket(self, ctx, side):
        return ctx["close"] - side * 2 * ctx["atr"], None

    def entry(self, ctx):
        s = ctx["sig"]
        if s != s or s == 0:
            return HOLD
        side = int(s)
        return enter(side, *self.bracket(ctx, side))
'''


def test_lookahead_keyed_to_holdout_dates_is_caught_before_any_run(cfg, plugins):
    add_plugin(plugins, "datepeek", DATE_PEEK)
    v = StubValidate({})
    j = judge(cfg, "datepeek", "4h", validate=v)
    assert j["decision"] == "kill" and j["child_run_id"] is None and v.calls == []
    assert any(r.startswith("lookahead: holdout-dated: ") and "features at cut" in r for r in j["reasons"])
    assert not any(r.startswith("lookahead: defaults") for r in j["reasons"])   # honest on 2023 dates


@pytest.mark.parametrize("name", REAL_PLUGINS)
def test_real_plugins_pass_the_holdout_dated_check(cfg, name, monkeypatch):
    from perp_lab.plugins import PluginSource
    monkeypatch.setattr("perp_lab.strategy.STRATEGY_DIR", REPO / "strategies")
    assert lookahead_failures(cfg, PluginSource.read(name)) == []


SEASONAL = '''"""Honest momentum, active November to April only (sell in May and go away)."""
import numpy as np
from perp_lab.indicators import atr
from perp_lab.strategy import HOLD, Strategy, enter


class Seasonal(Strategy):
    params = {}
    timeframes = ["4h"]

    def prepare(self, bars):
        winter = bars.index.month.isin([11, 12, 1, 2, 3, 4])
        mom = np.sign(bars.close - bars.close.shift(1)).where(winter, 0.0)
        return bars[[]].assign(sig=mom, atr=atr(bars, 14))

    def bracket(self, ctx, side):
        return ctx["close"] - side * 2 * ctx["atr"], None

    def entry(self, ctx):
        s = ctx["sig"]
        if s != s or s == 0:
            return HOLD
        return enter(int(s), *self.bracket(ctx, int(s)))
'''



def date_peek(day: str, parent: bool = True) -> str:
    text = DATE_PEEK.replace('"2026-07-01"', f'"{day}"')
    return text if parent else text.replace('PARENT = "donchian_atr"\n', "")


TODAY = pd.Timestamp("2026-10-16", tz="UTC")


@pytest.mark.parametrize("day", ["2026-07-01", "2026-08-20", "2026-09-10", "2026-11-01"])
def test_the_dated_check_covers_the_holdout_up_to_a_month_past_today(cfg, plugins, monkeypatch, day):
    import perp_lab.judge.judge as jm
    from perp_lab.plugins import PluginSource
    monkeypatch.setattr(jm, "_now", lambda: TODAY, raising=False)
    add_plugin(plugins, "peek", date_peek(day))
    fails = lookahead_failures(cfg, PluginSource.read("peek"))
    assert any(f.startswith("holdout-dated: ") and "features at cut" in f for f in fails)


@pytest.mark.parametrize("name", REAL_PLUGINS)
def test_real_plugins_pass_a_long_dated_window(cfg, name, monkeypatch):
    from perp_lab.judge.judge import dated_lookahead_failures
    from perp_lab.plugins import PluginSource
    monkeypatch.setattr("perp_lab.strategy.STRATEGY_DIR", REPO / "strategies")
    assert dated_lookahead_failures(cfg, PluginSource.read(name), pd.Timestamp("2027-06-30", tz="UTC")) == []


def test_the_dated_check_reaches_the_end_of_its_span(cfg, plugins):
    # a peek keyed to the last days before `until` must be seen: the span always ends on a cut
    from perp_lab.judge.judge import dated_lookahead_failures
    from perp_lab.plugins import PluginSource
    add_plugin(plugins, "peek", date_peek("2027-02-20"))
    fails = dated_lookahead_failures(cfg, PluginSource.read("peek"), pd.Timestamp("2027-03-08", tz="UTC"))
    assert any("features at cut" in f for f in fails)


def test_the_dated_check_is_empty_before_the_holdout_starts(cfg, plugins):
    # an exam window that hasn't reached the holdout has nothing to check (and must not crash)
    from perp_lab.judge.judge import dated_lookahead_failures
    from perp_lab.plugins import PluginSource
    early = cfg.holdout_start - pd.Timedelta(days=70)
    assert dated_lookahead_failures(cfg, PluginSource.read("donchian_atr"), early) == []


PARAM_PEEK = '''"""Peeks at the next close only with k=2, a value between its grid corners."""
import numpy as np
from perp_lab.indicators import atr
from perp_lab.strategy import HOLD, Strategy, enter


class ParamPeek(Strategy):
    params = {"k": 1}
    grid = {"k": [1, 3]}

    def prepare(self, bars):
        honest = np.sign(bars.close - bars.close.shift(1))
        peek = np.sign(bars.close.shift(-1) - bars.close)
        return bars[[]].assign(sig=peek if self.p["k"] == 2 else honest, atr=atr(bars, 14))

    def bracket(self, ctx, side):
        return ctx["close"] - side * 2 * ctx["atr"], None

    def entry(self, ctx):
        s = ctx["sig"]
        if s != s or s == 0:
            return HOLD
        return enter(int(s), *self.bracket(ctx, int(s)))
'''


def test_the_dated_check_also_tries_the_exam_params(cfg, plugins):
    from perp_lab.judge.judge import dated_lookahead_failures
    from perp_lab.plugins import PluginSource
    add_plugin(plugins, "parampeek", PARAM_PEEK)
    src, until = PluginSource.read("parampeek"), pd.Timestamp("2026-12-28", tz="UTC")
    assert dated_lookahead_failures(cfg, src, until) == []          # corners k=1, k=3 are honest
    assert any("k=2" in f for f in dated_lookahead_failures(cfg, src, until, params={"k": 2}))


def test_a_seasonal_plugin_quiet_across_the_whole_dated_span_is_not_killed(cfg, plugins, monkeypatch):
    # judged on 2026-08-01 the dated span is May-August: the plugin never signals there
    import perp_lab.judge.judge as jm
    from perp_lab.plugins import PluginSource
    monkeypatch.setattr(jm, "_now", lambda: pd.Timestamp("2026-08-01", tz="UTC"))
    add_plugin(plugins, "seasonal", SEASONAL)
    assert lookahead_failures(cfg, PluginSource.read("seasonal")) == []


def _failing(v, failing, err):
    def run(cfg, name, *a, **kw):
        if name == failing:
            v.calls.append(name)
            raise err
        return v(cfg, name, *a, **kw)
    return run


def test_a_plugin_error_in_the_walk_forward_is_a_recorded_kill(cfg, plugins, caplog):
    v = StubValidate({})
    err = PluginError("exception", "ValueError", "boom", b"Traceback: boom\n")
    with caplog.at_level("WARNING", logger="perp_lab.judge.judge"):
        j = judge(cfg, "donchian_atr", "4h", validate=_failing(v, "donchian_atr", err))
    assert j["decision"] == "kill" and j["reasons"] == ["plugin_error: exception: ValueError: boom"]
    assert json.loads(Path(j["verdict_path"]).read_text())["reasons"] == ["plugin_error: exception: ValueError"]
    assert j["child_run_id"] is None and (j["holdout_status"], j["holdout_reason"]) == ("n/a", "n/a")
    assert "Traceback: boom" in caplog.text
    assert db.read_table(cfg, db.judgements).shape[0] == 1
    assert "Traceback" not in Path(j["verdict_path"]).read_text()
    assert db.trial_stats(cfg)[0] == 1


def test_the_verdict_names_a_builtin_exception_type_only(cfg, plugins):
    err = PluginError("exception", "ValueError", "secret 123")
    j = judge(cfg, "donchian_atr", "4h", validate=_failing(StubValidate({}), "donchian_atr", err))
    v = json.loads(Path(j["verdict_path"]).read_text())
    assert v["reasons"] == ["plugin_error: exception: ValueError"] and "secret" not in json.dumps(v)
    assert "verdict_reasons" not in v
    assert db.get_judgement_by_id(cfg, j["id"])["reasons"] == ["plugin_error: exception: ValueError: secret 123"]


def test_a_plugin_defined_exception_type_becomes_other():
    assert PluginError("exception", "Btc_up_3pct", "m").public_reason() == "plugin_error: exception: other"


def test_a_kind_without_an_exception_keeps_only_its_kind():
    assert PluginError("protocol", "", "x").public_reason() == "plugin_error: protocol"


def test_a_plugin_error_after_real_bars_raises_n(cfg, plugins):
    err = PluginError("exception", "ValueError", "boom")
    judge(cfg, "donchian_atr", "4h", validate=_failing(StubValidate({}), "donchian_atr", err))
    assert db.trial_stats(cfg)[0] == 1


def test_a_harness_crash_after_real_bars_raises_n(cfg, plugins):
    with pytest.raises(RuntimeError):
        judge(cfg, "donchian_atr", "4h", validate=_failing(StubValidate({}), "donchian_atr", RuntimeError("engine")))
    assert db.trial_stats(cfg)[0] == 1 and db.read_table(cfg, db.judgements).empty


def test_judging_the_same_bytes_again_adds_no_trial(cfg, plugins):
    with pytest.raises(RuntimeError):
        judge(cfg, "donchian_atr", "4h", validate=_failing(StubValidate({}), "donchian_atr", RuntimeError("x")))
    judge(cfg, "donchian_atr", "4h", validate=StubValidate({"donchian_atr": ("noise", NOISE)}))
    assert db.trial_stats(cfg)[0] == 1
    assert db.trial_attempts(cfg, plugin_hash("donchian_atr"), "4h") == 2


def test_a_lookahead_kill_registers_no_trial(cfg, plugins):
    add_plugin(plugins, "cheater", CHEATER)
    judge(cfg, "cheater", "4h", validate=StubValidate({}))
    assert db.trial_stats(cfg)[0] == 0 and db.read_table(cfg, db.trials).empty


def test_a_plugin_error_in_a_child_with_a_parent_records_no_edge(cfg, plugins):
    v = StubValidate({"donchian_atr": ("candidate", STRONG)})
    j = judge(cfg, "donchian_atr_wide", "4h",
              validate=_failing(v, "donchian_atr_wide", PluginError("exception", "KeyError", "x")))
    assert j["decision"] == "kill" and j["reasons"] == ["plugin_error: exception: KeyError: x"]
    assert j["child_run_id"] is None and db.read_table(cfg, db.lineage_edges).empty


def test_a_parent_that_fails_in_the_sandbox_refuses_the_child(cfg, plugins):
    v = StubValidate({"donchian_atr_wide": ("candidate", STRONG)})
    with pytest.raises(JudgeRefused, match="parent donchian_atr fails in the sandbox; fix or judge it first"):
        judge(cfg, "donchian_atr_wide", "4h", validate=_failing(v, "donchian_atr", PluginError("died")))
    assert db.read_table(cfg, db.judgements).empty


def test_judge_refuses_to_start_without_a_sandbox(cfg, plugins, monkeypatch):
    def no(cfg):
        raise SandboxUnavailable("no landlock")
    monkeypatch.setattr(client, "check", no)
    v = StubValidate({"donchian_atr": ("candidate", STRONG)})
    with pytest.raises(SandboxUnavailable, match="no landlock"):
        judge(cfg, "donchian_atr", "4h", validate=v)
    assert v.calls == [] and db.read_table(cfg, db.judgements).empty
    r = CliRunner().invoke(app, ["judge", "donchian_atr", "--tf", "4h", "--config", str(cfg.path)])
    assert r.exit_code == 1 and "sandbox unavailable: no landlock" in r.output


def test_a_wrong_timeframe_is_refused_even_without_a_sandbox(cfg, plugins, monkeypatch):
    # The refusals that read only plugin source (an undeclared timeframe, a missing or identical
    # parent) come before the sandbox check: a wrong --tf is refused on any machine, as such.
    monkeypatch.setattr(client, "check", lambda cfg: pytest.fail("checked the sandbox first"))
    with pytest.raises(JudgeRefused, match=r"donchian_atr_wide declares timeframes 4h, not 1h"):
        judge(cfg, "donchian_atr_wide", "1h", validate=StubValidate({}))
    add_plugin(plugins, "orphan", WIDE_TEXT.replace('PARENT = "donchian_atr"', 'PARENT = "nope"'))
    with pytest.raises(JudgeRefused, match="PARENT 'nope' is not a plugin"):
        judge(cfg, "orphan", "4h", validate=StubValidate({}))
    r = CliRunner().invoke(app, ["judge", "donchian_atr_wide", "--tf", "1h", "--config", str(cfg.path)])
    assert r.exit_code == 2 and "refused: donchian_atr_wide declares timeframes 4h, not 1h" in r.output


def test_an_edit_during_judging_does_not_change_what_is_judged(cfg, plugins):
    from perp_lab.strategy import plugin_hash
    ph, ch = plugin_hash("donchian_atr"), plugin_hash("donchian_atr_wide")
    v = StubValidate({"donchian_atr": ("noise", NOISE), "donchian_atr_wide": ("candidate", STRONG)})

    def editing(cfg, name, *a, **kw):
        for n in ("donchian_atr", "donchian_atr_wide"):
            f = plugins / f"{n}.py"
            f.write_text(f.read_text() + "\n# edited mid-run\n")
        return v(cfg, name, *a, **kw)
    j = judge(cfg, "donchian_atr_wide", "4h", validate=editing)
    assert j["strategy_hash"] == ch
    runs = db.read_table(cfg, db.runs).set_index("strategy")["strategy_hash"]
    assert runs["donchian_atr"] == ph and runs["donchian_atr_wide"] == ch


@pytest.mark.sandbox
def test_end_to_end_judge_through_the_real_sandbox(tmp_path, plugins, fresh_wf_caches):
    cfg = market_config(tmp_path, judge={"min_holdout_days": 30},
                        validation={"random_seeds": 5, "mc_iterations": 200})
    j = judge(cfg, "donchian_atr", "4h", workers=2)
    assert j["decision"] in {"promote", "kill"} and j["n_trials"] == 1
    if j["decision"] == "kill":
        assert j["reasons"] and j["holdout_status"] == "n/a"
    else:
        assert (j["holdout_status"], j["holdout_reason"]) == ("pending", "awaiting_exam")
    child = db.run_row(cfg, j["child_run_id"])
    assert child["validation_mode"] == "walkforward" and len(db.read_table(cfg, db.run_folds)) == 6
    assert set(json.loads(Path(j["verdict_path"]).read_text())) == VERDICT_KEYS


def test_the_parent_is_re_run_when_its_run_used_other_data(cfg, plugins):
    v = StubValidate({"donchian_atr": ("candidate", STRONG), "donchian_atr_wide": ("candidate", STRONG + 0.001)})
    elsewhere = Config(raw={**cfg.raw, "history_start": "2019-01-01"}, path=cfg.path)
    old, _ = v(elsewhere, "donchian_atr", "4h", lineage_id="family:donchian_atr")
    j = judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert v.calls == ["donchian_atr", "donchian_atr", "donchian_atr_wide"] and j["parent_run_id"] != old["run_id"]


class SkippingValidate(StubValidate):
    """A walk-forward that skips coins with no bars, as research_symbols does: `skip` maps a plugin
    to the coins its run drops from the universe it is asked for. Records what each call asked for."""

    def __init__(self, outcomes: dict, skip: dict):
        super().__init__(outcomes)
        self.skip, self.asked = skip, []

    def __call__(self, cfg, name, tf, symbols=None, **kw):
        self.asked.append((name, symbols))
        ran = [s for s in (symbols or cfg.symbols) if s not in self.skip.get(name, ())]
        return super().__call__(cfg, name, tf, ran, **kw)


def test_the_parent_is_re_run_on_the_childs_data_when_the_child_skips_a_coin(cfg, plugins):
    # The parent's run on the full universe is found by the requested universe, but the child's run
    # dropped a coin: comparing them would pair two different data sets (spec change 6).
    v = SkippingValidate({"donchian_atr": ("candidate", STRONG), "donchian_atr_wide": ("candidate", STRONG + 0.001)},
                         skip={"donchian_atr_wide": ("ETCUSDT",)})
    old, _ = v(cfg, "donchian_atr", "4h", lineage_id="family:donchian_atr")
    j = judge(cfg, "donchian_atr_wide", "4h", validate=v)
    child = db.run_row(cfg, j["child_run_id"])
    assert child["universe"] == [s for s in cfg.symbols if s != "ETCUSDT"] and len(child["universe"]) == 16
    assert v.calls == ["donchian_atr", "donchian_atr_wide", "donchian_atr"]
    assert v.asked[-1] == ("donchian_atr", child["universe"])
    assert j["parent_run_id"] not in (None, old["run_id"])
    parent = db.run_row(cfg, j["parent_run_id"])
    assert parent["data_id"] == child["data_id"] and parent["universe"] == child["universe"]
    assert j["decision"] == "promote" and j["beats_parent_frac"] == 1.0
    e = db.read_table(cfg, db.lineage_edges)
    assert e.parent_run_id.tolist() == [j["parent_run_id"]] and e.child_run_id.tolist() == [j["child_run_id"]]


def test_the_judge_refuses_when_the_parent_cannot_run_on_the_childs_data(cfg, plugins):
    # The parent's re-run on the child's coins drops another one: the two runs still differ, so
    # nothing is compared and no judgement or lineage edge is recorded.
    v = SkippingValidate({"donchian_atr": ("candidate", STRONG), "donchian_atr_wide": ("candidate", STRONG + 0.001)},
                         skip={"donchian_atr_wide": ("ETCUSDT",), "donchian_atr": ("ATOMUSDT",)})
    with pytest.raises(JudgeRefused, match="parent donchian_atr .* on the data its child ran on"):
        judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert v.calls == ["donchian_atr", "donchian_atr_wide", "donchian_atr"]
    assert db.read_table(cfg, db.judgements).empty and db.read_table(cfg, db.lineage_edges).empty


def test_a_transient_failure_records_nothing_but_the_trial(cfg, plugins):
    with pytest.raises(TransientFailure):
        judge(cfg, "donchian_atr", "4h", validate=_failing(StubValidate({}), "donchian_atr", PluginError("died")))
    assert db.read_table(cfg, db.judgements).empty
    assert db.trial_attempts(cfg, plugin_hash("donchian_atr"), "4h") == 1 and db.trial_stats(cfg)[0] == 1


def test_a_transient_failure_on_the_third_attempt_is_a_kill(cfg, plugins):
    run = _failing(StubValidate({}), "donchian_atr", PluginError("timeout"))
    for _ in range(2):
        with pytest.raises(TransientFailure):
            judge(cfg, "donchian_atr", "4h", validate=run)
    j = judge(cfg, "donchian_atr", "4h", validate=run)
    assert j["decision"] == "kill" and j["reasons"] == ["plugin_error: timeout"]


def test_a_transient_failure_then_a_clean_run_is_judged(cfg, plugins):
    with pytest.raises(TransientFailure):
        judge(cfg, "donchian_atr", "4h", validate=_failing(StubValidate({}), "donchian_atr", PluginError("cpu_limit")))
    j = judge(cfg, "donchian_atr", "4h", validate=StubValidate({"donchian_atr": ("noise", NOISE)}))
    assert j["child_run_id"] is not None and db.trial_stats(cfg)[0] == 1


def test_cli_transient_failure_exits_1(cfg, plugins, monkeypatch):
    def boom(*a, **k): raise TransientFailure("died")
    monkeypatch.setattr("perp_lab.judge.judge.judge", boom)
    r = CliRunner().invoke(app, ["judge", "donchian_atr", "--tf", "4h", "--config", str(cfg.path)])
    assert r.exit_code == 1 and "the trial is registered" in r.output
