"""Walk-forward, sensitivity and the random-entry benchmark run plugins through the sandbox: jobs
go out in one-symbol chunks, each run in a `client.Batch` and replayed by the trusted engine.
Unmarked tests run on the in-process twin; the `sandbox` one spawns real workers."""

import pickle
from pathlib import Path

import pandas as pd
import pytest

from perp_lab.data.store import load_bars
from perp_lab.engine.backtest import EngineParams, run_backtest
from perp_lab.engine.metrics import daily_returns
from perp_lab.plugins import PluginSource
from perp_lab.sandbox.errors import PluginError, ProtocolError
from perp_lab.store import db
from perp_lab.strategy import RandomEntry
from perp_lab.validate import pipeline
from perp_lab.validate import walkforward as wfm
from tests.conftest import REPO, market_config
from tests.judge_fixtures import add_plugin, plugins  # noqa: F401
from tests.sandbox_fixtures import load_class

DONCHIAN = (REPO / "strategies" / "donchian_atr.py").read_text()
PREPARE = "    def prepare(self, bars):\n"
LOUD = DONCHIAN.replace(PREPARE, PREPARE + '        print("hello")\n')

ONE_SYMBOL = '''"""Remembers the first close of every series it prepares, and refuses a second series."""
from perp_lab.strategy import HOLD, Strategy

FIRST_CLOSES = []


class OneSymbol(Strategy):
    params = {"n": 1}
    grid = {"n": [1, 2, 3]}

    def prepare(self, bars):
        FIRST_CLOSES.append(float(bars.close.iloc[0]))
        print(f"prepared {len(FIRST_CLOSES)}")
        if len(set(FIRST_CLOSES)) > 1:
            raise RuntimeError("this worker was sent a second symbol")
        return bars[[]]

    def entry(self, ctx):
        return HOLD
'''


def failing(body: str) -> PluginSource:
    """A plugin whose prepare() prints a line and then runs `body`."""
    text = f'''"""Fails in prepare."""
from perp_lab.sandbox.errors import PluginError
from perp_lab.strategy import HOLD, Strategy


class Failing(Strategy):
    def prepare(self, bars):
        print("about to fail")
        {body}

    def entry(self, ctx):
        return HOLD
'''
    return PluginSource.from_text("failing", text)


@pytest.fixture(autouse=True)
def fresh_wf_caches():
    wfm._cfg.cache_clear()
    wfm._bars.cache_clear()
    yield
    wfm._cfg.cache_clear()
    wfm._bars.cache_clear()


def test_chunks_never_mix_symbols_and_spread_work():
    jobs = [({"n": k}, s, "4h", None, None, None) for k in range(10) for s in ("A", "B", "C")]
    cs = wfm.chunks(jobs, 4)
    assert len(cs) >= 8
    for symbol, tf, js in cs:
        assert js and {j[1] for j in js} == {symbol} and {j[2] for j in js} == {tf}
    for s in "ABC":   # every job exactly once, each symbol's in their original order
        assert [j for _, _, js in cs for j in js if j[1] == s] == [j for j in jobs if j[1] == s]
    assert sum(len(js) for _, _, js in cs) == len(jobs)


LEVELS = ("level_time", "level_stop", "level_target")


@pytest.mark.parametrize("workers", [1, 2])
def test_run_many_matches_a_direct_backtest(tmp_path, workers):
    cfg = market_config(tmp_path)
    src = PluginSource.from_text("donchian_atr", DONCHIAN)
    cls = load_class("donchian_atr")
    jobs = [(c, s, "4h", None, None, None) for c in wfm.grid_combos(src.meta.grid) for s in cfg.symbols]
    jobs += [({"n": 20}, s, "4h", None, seed, 0.02) for seed in (0, 1) for s in cfg.symbols]
    res = wfm.run_many(cfg, src, jobs, workers, keep_levels=True)
    assert [(r["params"], r["symbol"], r["seed"]) for r in res] == [(j[0], j[1], j[4]) for j in jobs]
    for (params, s, tf, _, seed, prob), r in zip(jobs, res):
        bars = load_bars(cfg, s, tf)
        strat = cls(**params)
        if seed is not None:
            strat = RandomEntry(strat, prob, seed)
        ref = run_backtest(strat, bars, strat.prepare(bars), symbol=s, tf=tf, ep=EngineParams.from_config(cfg, s))
        assert len(ref.trades) > 0
        pd.testing.assert_series_equal(r["daily"], daily_returns(ref.equity, ref.time))
        assert set(LEVELS) <= set(r["trades"].columns)
        pd.testing.assert_frame_equal(r["trades"], pd.DataFrame(ref.trades))


@pytest.mark.parametrize("keep", [False, True])
def test_only_jobs_that_ask_for_it_keep_the_level_columns(tmp_path, keep):
    # A sensitivity-style job (no seed) reads only its daily returns, so by default it carries no
    # stop/target path through the pool; the walk-forward grid asks for it (its trades are charted).
    cfg = market_config(tmp_path)
    src = PluginSource.from_text("donchian_atr", DONCHIAN)
    jobs = [({"n": 20}, s, "4h", None, None, None) for s in cfg.symbols]
    res = wfm.run_many(cfg, src, jobs, 1, **({"keep_levels": True} if keep else {}))
    full = [run_backtest(s, b, s.prepare(b), symbol=sym, tf="4h", ep=EngineParams.from_config(cfg, sym)).trades
            for sym in cfg.symbols for s, b in [(load_class("donchian_atr")(n=20), load_bars(cfg, sym, "4h"))]]
    for r, want in zip(res, full):
        assert len(r["trades"]) > 0 and (set(LEVELS) <= set(r["trades"].columns)) == keep
        assert keep or not set(LEVELS) & set(r["trades"].columns)
        want = pd.DataFrame(want)
        pd.testing.assert_frame_equal(r["trades"], want if keep else want.drop(columns=list(LEVELS)))


def test_only_the_walkforward_grid_asks_for_the_level_columns(tmp_path, plugins, monkeypatch):
    # Its trades become oos_trades (and the trade chart); sensitivity reads daily returns only and
    # the random-entry benchmark signal_time only.
    from perp_lab.validate import benchmarks, sensitivity
    cfg = market_config(tmp_path, validation={"random_seeds": 2, "mc_iterations": 50})
    asked = {}

    def spy(who, real):
        def run_many(*a, **kw):
            asked.setdefault(who, set()).add(kw.get("keep_levels", False))
            return real(*a, **kw)
        return run_many
    monkeypatch.setattr(wfm, "run_many", spy("walkforward", wfm.run_many))
    monkeypatch.setattr(sensitivity, "run_many", spy("sensitivity", sensitivity.run_many))
    monkeypatch.setattr(benchmarks, "run_many", spy("random_entry", benchmarks.run_many))
    rec, _ = pipeline.validate_strategy(cfg, "donchian_atr", "4h", workers=1)
    assert asked == {"walkforward": {True}, "sensitivity": {False}, "random_entry": {False}}
    oos = pd.read_parquet(Path(rec["run_dir"]) / "oos_trades.parquet")
    assert len(oos) and set(LEVELS) <= set(oos.columns)


def test_plugin_output_lands_in_the_run_directory(tmp_path, plugins):
    cfg = market_config(tmp_path, validation={"random_seeds": 2, "mc_iterations": 50})
    add_plugin(plugins, "loud", LOUD)
    rec, report = pipeline.validate_strategy(cfg, "loud", "4h", workers=2)
    run_dir = Path(rec["run_dir"])
    files = sorted((run_dir / "sandbox_stderr").glob("*.txt"))
    assert files and all(f.name.split("-")[0] in rec["universe"] for f in files)
    assert all("hello" in f.read_text() for f in files)
    assert "hello" not in (run_dir / "summary.json").read_text() and "hello" not in report


@pytest.mark.sandbox
def test_a_worker_never_sees_a_second_symbol(tmp_path):
    cfg = market_config(tmp_path)
    first = {s: float(load_bars(cfg, s, "4h").close.iloc[0]) for s in cfg.symbols}
    assert len(set(first.values())) == len(first)          # a second symbol would be noticed
    src = PluginSource.from_text("one_symbol", ONE_SYMBOL)
    jobs = [({"n": n}, s, "4h", None, None, None) for n in (1, 2, 3) for s in cfg.symbols]
    res = wfm.run_many(cfg, src, jobs, workers=2, stderr_dir=tmp_path / "err")
    assert [(r["params"], r["symbol"]) for r in res] == [(j[0], j[1]) for j in jobs]
    out = [f.read_text() for f in (tmp_path / "err").glob("*.txt")]
    assert any("prepared 2" in o for o in out)              # a worker did run several jobs


@pytest.mark.parametrize("body, kind, type_, message", [
    ('raise ValueError("boom")', "exception", "ValueError", "boom"),
    ('raise PluginError("timeout", "", "retry me")', "protocol", "", "worker reported a timeout error: retry me"),
])
def test_a_plugin_error_in_a_pool_worker_reaches_the_caller_intact(tmp_path, body, kind, type_, message):
    cfg = market_config(tmp_path)
    jobs = [({}, s, "4h", None, None, None) for s in cfg.symbols]
    with pytest.raises(PluginError) as e:
        wfm.run_many(cfg, failing(body), jobs, workers=2)
    assert (e.value.kind, e.value.type, e.value.message) == (kind, type_, message)
    assert b"about to fail" in e.value.stderr
    assert isinstance(e.value, ProtocolError) == (kind == "protocol")


@pytest.mark.parametrize("err", [PluginError("exception", "ValueError", "boom", b"tail"),
                                 ProtocolError("bad frame", b"tail")])
def test_plugin_errors_pickle_with_every_field(err):
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is type(err)
    assert (back.kind, back.type, back.message, back.stderr) == (err.kind, err.type, err.message, err.stderr)


def test_a_walk_forward_at_an_undeclared_timeframe_is_refused_before_anything_runs(tmp_path, plugins, monkeypatch):
    # It would be a trial (N counts (version, timeframe) pairs): donchian_atr_wide declares 4h only.
    from typer.testing import CliRunner

    from perp_lab.cli import app
    from perp_lab.plugins import UndeclaredTimeframe
    cfg = market_config(tmp_path)
    monkeypatch.setattr(pipeline, "walkforward", lambda *a, **k: pytest.fail("walked forward"))
    with pytest.raises(UndeclaredTimeframe, match=r"donchian_atr_wide declares timeframes 4h, not 1h"):
        pipeline.validate_strategy(cfg, "donchian_atr_wide", "1h", workers=1)
    r = CliRunner().invoke(app, ["walkforward", "donchian_atr_wide", "--config", str(cfg.path)])   # default 1h
    assert r.exit_code == 2 and "refused: donchian_atr_wide declares timeframes 4h, not 1h" in r.output
    assert not cfg.runs_dir.exists() or not any(cfg.runs_dir.iterdir())
    assert db.read_table(cfg, db.runs).empty


def test_an_edit_during_validation_does_not_change_what_is_recorded(tmp_path, plugins, monkeypatch):
    cfg = market_config(tmp_path, validation={"random_seeds": 2, "mc_iterations": 50})
    f = plugins / "donchian_atr.py"
    h0 = PluginSource.read("donchian_atr").hash
    real = wfm.run_many

    def editing(*a, **kw):   # an edit that breaks any run that reads the file again
        f.write_text(f.read_text() + '\nraise RuntimeError("edited mid-run")\n')
        return real(*a, **kw)

    monkeypatch.setattr(wfm, "run_many", editing)
    rec, _ = pipeline.validate_strategy(cfg, "donchian_atr", "4h", workers=1)
    assert PluginSource.read("donchian_atr").hash != h0
    assert rec["strategy_hash"] == h0 == db.run_row(cfg, rec["run_id"])["strategy_hash"]
