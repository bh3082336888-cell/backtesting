"""The judge-exam lock: shared plugin runs, an exclusive exam, writer preference."""
import fcntl
import threading

from typer.testing import CliRunner

from perp_lab import runlock
from perp_lab.cli import app
from tests.judge_fixtures import plugins  # noqa: F401
from tests.test_store import cfg  # noqa: F401

WAIT = 2.0


def _in_thread(fn):
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    return t


def _try_exclusive_r(cfg):
    """Whether the plugin-runs lock could be taken exclusively right now (no turnstile)."""
    with open(cfg.runs_dir / ".plugin-runs.lock", "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False


def test_no_exam_starts_while_a_judge_runs(cfg):
    entered, left = threading.Event(), threading.Event()
    with runlock.plugin_runs(cfg):
        def exam():
            with runlock.exam_exclusive(cfg):
                entered.set()
        t = _in_thread(exam)
        assert not entered.wait(0.3)
        left.set()
    assert entered.wait(WAIT)
    t.join(WAIT)


def test_no_judge_starts_while_an_exam_runs(cfg):
    entered = threading.Event()
    with runlock.exam_exclusive(cfg):
        def judge():
            with runlock.plugin_runs(cfg):
                entered.set()
        t = _in_thread(judge)
        assert not entered.wait(0.3)
    assert entered.wait(WAIT)
    t.join(WAIT)


def test_judges_run_together(cfg):
    with runlock.plugin_runs(cfg):
        got = threading.Event()

        def second():
            with runlock.plugin_runs(cfg):
                got.set()
        t = _in_thread(second)
        assert got.wait(WAIT)
        t.join(WAIT)


def test_a_stream_of_judges_does_not_starve_a_waiting_exam(cfg):
    order, b_in = [], threading.Event()
    exam_waiting = threading.Event()
    a = runlock.plugin_runs(cfg)
    a.__enter__()                                   # judge A holds shared

    def exam():
        # signal just before blocking; the turnstile is taken first thing inside exam_exclusive
        exam_waiting.set()
        with runlock.exam_exclusive(cfg):
            order.append("exam")

    te = _in_thread(exam)
    assert exam_waiting.wait(WAIT)
    # wait deterministically until the exam holds the turnstile (it is then blocked on R)
    with open(cfg.runs_dir / ".turnstile.lock", "a+") as f:
        for _ in range(200):
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(f, fcntl.LOCK_UN)
            except BlockingIOError:
                break
            threading.Event().wait(0.01)
        else:
            raise AssertionError("the exam never took the turnstile")

    def judge_b():
        with runlock.plugin_runs(cfg):
            order.append("B")
            b_in.set()

    tb = _in_thread(judge_b)
    assert not b_in.wait(0.3)                       # B queues behind the waiting exam
    a.__exit__(None, None, None)
    assert b_in.wait(WAIT)
    te.join(WAIT)
    tb.join(WAIT)
    assert order == ["exam", "B"]


def test_the_lock_is_released_when_the_body_raises(cfg):
    try:
        with runlock.plugin_runs(cfg):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert _try_exclusive_r(cfg)


def test_cli_judge_holds_the_shared_lock(cfg, plugins, monkeypatch):
    seen = {}

    def stub(cfg_, *a, **k):
        seen["free"] = _try_exclusive_r(cfg_)
        raise RuntimeError("stop")

    monkeypatch.setattr("perp_lab.judge.judge.judge", stub)
    r = CliRunner().invoke(app, ["judge", "donchian_atr", "--tf", "4h", "--config", str(cfg.path)])
    assert seen == {"free": False}
    assert isinstance(r.exception, RuntimeError)
    assert _try_exclusive_r(cfg)                    # and released afterwards
