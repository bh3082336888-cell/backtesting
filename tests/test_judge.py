import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from perp_lab.cli import app
from perp_lab.judge.judge import VERDICT_KEYS, JudgeRefused, judge, lookahead_failures
from perp_lab.store import db
from tests.judge_fixtures import (CHEATER, NOISE, STRONG, WEAK, WIDE_TEXT, StubValidate, add_plugin,
                                  plugins, series)  # noqa: F401
from tests.conftest import REAL_PLUGINS, REPO
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


def test_a_plugin_edited_while_it_is_judged_gets_no_judgement(cfg, plugins):
    v = StubValidate({"donchian_atr": ("candidate", STRONG)})
    f = plugins / "donchian_atr.py"

    def editing(cfg, name, *a, **kw):
        f.write_text(f.read_text() + "\n# edited mid-run\n")
        return v(cfg, name, *a, **kw)
    with pytest.raises(RuntimeError, match="donchian_atr changed while it was being judged; re-run lab judge"):
        judge(cfg, "donchian_atr", "4h", validate=editing)
    assert v.calls == ["donchian_atr"] and db.read_table(cfg, db.judgements).empty


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
    from perp_lab.strategy import load_strategy_class
    monkeypatch.setattr("perp_lab.strategy.STRATEGY_DIR", REPO / "strategies")
    assert lookahead_failures(cfg, load_strategy_class(name)) == []


def test_a_parent_edited_while_its_walk_forward_runs_gets_no_judgement(cfg, plugins):
    v = StubValidate({"donchian_atr": ("noise", NOISE), "donchian_atr_wide": ("candidate", STRONG)})
    f = plugins / "donchian_atr.py"

    def editing(cfg, name, *a, **kw):
        if name == "donchian_atr":
            f.write_text(f.read_text() + "\n# edited mid-run\n")
        return v(cfg, name, *a, **kw)
    with pytest.raises(RuntimeError, match="parent donchian_atr changed while it was being checked or run; re-run lab judge"):
        judge(cfg, "donchian_atr_wide", "4h", validate=editing)
    assert v.calls == ["donchian_atr"] and db.read_table(cfg, db.judgements).empty


def test_a_parent_edited_during_its_lookahead_check_gets_no_judgement(cfg, plugins, monkeypatch):
    # the parent's hash is taken before its lookahead check, so code edited after the check
    # (and never checked) can't be walked forward and accepted as the baseline
    import perp_lab.judge.judge as jm
    real, f = jm.lookahead_failures, plugins / "donchian_atr.py"

    def editing(cfg, cls):
        out = real(cfg, cls)
        if cls.name == "donchian_atr":
            f.write_text(f.read_text() + "\n# edited after its check\n")
        return out

    monkeypatch.setattr(jm, "lookahead_failures", editing)
    v = StubValidate({"donchian_atr": ("noise", NOISE), "donchian_atr_wide": ("candidate", STRONG)})
    with pytest.raises(RuntimeError, match="parent donchian_atr changed while it was being checked or run"):
        judge(cfg, "donchian_atr_wide", "4h", validate=v)
    assert db.read_table(cfg, db.judgements).empty


SEASONAL = '''"""Honest momentum, active November to April only (sell in May and go away)."""
import numpy as np
from perp_lab.indicators import atr
from perp_lab.strategy import HOLD, Strategy, enter


class Seasonal(Strategy):
    params = {}

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
    from perp_lab.strategy import load_strategy_class
    monkeypatch.setattr(jm, "_now", lambda: TODAY, raising=False)
    add_plugin(plugins, "peek", date_peek(day))
    fails = lookahead_failures(cfg, load_strategy_class("peek"))
    assert any(f.startswith("holdout-dated: ") and "features at cut" in f for f in fails)


@pytest.mark.parametrize("name", REAL_PLUGINS)
def test_real_plugins_pass_a_long_dated_window(cfg, name, monkeypatch):
    from perp_lab.judge.judge import dated_lookahead_failures
    from perp_lab.strategy import load_strategy_class
    monkeypatch.setattr("perp_lab.strategy.STRATEGY_DIR", REPO / "strategies")
    assert dated_lookahead_failures(cfg, load_strategy_class(name), pd.Timestamp("2027-06-30", tz="UTC")) == []


def test_the_dated_check_reaches_the_end_of_its_span(cfg, plugins):
    # a peek keyed to the last days before `until` must be seen: the span always ends on a cut
    from perp_lab.judge.judge import dated_lookahead_failures
    from perp_lab.strategy import load_strategy_class
    add_plugin(plugins, "peek", date_peek("2027-02-20"))
    fails = dated_lookahead_failures(cfg, load_strategy_class("peek"), pd.Timestamp("2027-03-08", tz="UTC"))
    assert any("features at cut" in f for f in fails)


def test_the_dated_check_is_empty_before_the_holdout_starts(cfg, plugins):
    # an exam window that hasn't reached the holdout has nothing to check (and must not crash)
    from perp_lab.judge.judge import dated_lookahead_failures
    from perp_lab.strategy import load_strategy_class
    early = cfg.holdout_start - pd.Timedelta(days=70)
    assert dated_lookahead_failures(cfg, load_strategy_class("donchian_atr"), early) == []


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
    from perp_lab.strategy import load_strategy_class
    add_plugin(plugins, "parampeek", PARAM_PEEK)
    cls, until = load_strategy_class("parampeek"), pd.Timestamp("2026-12-28", tz="UTC")
    assert dated_lookahead_failures(cfg, cls, until) == []          # corners k=1, k=3 are honest
    assert any("k=2" in f for f in dated_lookahead_failures(cfg, cls, until, params={"k": 2}))


def test_a_seasonal_plugin_quiet_across_the_whole_dated_span_is_not_killed(cfg, plugins, monkeypatch):
    # judged on 2026-08-01 the dated span is May-August: the plugin never signals there
    import perp_lab.judge.judge as jm
    from perp_lab.strategy import load_strategy_class
    monkeypatch.setattr(jm, "_now", lambda: pd.Timestamp("2026-08-01", tz="UTC"))
    add_plugin(plugins, "seasonal", SEASONAL)
    assert lookahead_failures(cfg, load_strategy_class("seasonal")) == []
