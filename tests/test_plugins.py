import subprocess

import pytest

from perp_lab.judge.judge import JudgeRefused, judge
from perp_lab.plugins import PluginMetaError, PluginSource, blob_hash
from tests.sandbox_fixtures import load_class
from tests.conftest import REAL_PLUGINS, REPO
from tests.judge_fixtures import StubValidate, add_plugin, plugins, series  # noqa: F401
from tests.test_store import cfg  # noqa: F401


@pytest.mark.parametrize("name", REAL_PLUGINS)
def test_metadata_from_the_source_equals_the_loaded_class(name, monkeypatch):
    monkeypatch.setattr("perp_lab.strategy.STRATEGY_DIR", REPO / "strategies")
    src, cls = PluginSource.read(name), load_class(name)
    assert (src.meta.params, src.meta.grid, src.meta.timeframes) == (cls.params, cls.grid, cls.timeframes)
    assert src.meta.hypothesis == cls.hypothesis() and len(src.meta.hypothesis) > 20


@pytest.mark.parametrize("name", REAL_PLUGINS)
def test_hash_equals_git_hash_object(name):
    p = REPO / "strategies" / f"{name}.py"
    assert blob_hash(p.read_bytes()) == subprocess.check_output(["git", "hash-object", str(p)], text=True).strip()


def test_parent_is_read_without_running_the_module():
    text = '"""h"""\nraise SystemExit("ran")\nPARENT = "donchian_atr"\nclass A(Strategy):\n    params = {}\n'
    assert PluginSource.from_text("x", text).meta.parent == "donchian_atr"


def test_defaults_when_metadata_is_absent():
    m = PluginSource.from_text("x", "class A(Strategy):\n    pass\n").meta
    assert (m.parent, m.hypothesis, m.params, m.grid, m.timeframes) == (None, "", {}, {}, ["1h"])


@pytest.mark.parametrize("snippet", [
    'PARENT = "a" + "b"', 'PARENT = "a"\nPARENT = "b"', 'if True:\n    PARENT = "a"', 'PARENT = 3'])
def test_a_parent_that_is_not_one_string_literal_is_refused(snippet):
    with pytest.raises(PluginMetaError):
        PluginSource.from_text("x", f'"""h"""\n{snippet}\nclass A(Strategy):\n    pass\n')


@pytest.mark.parametrize("body", ['grid = {"n": list(range(3))}', 'params = dict(n=1)', 'timeframes = TFS'])
def test_class_metadata_must_be_literal(body):
    with pytest.raises(PluginMetaError):
        PluginSource.from_text("x", f'"""h"""\nclass A(Strategy):\n    {body}\n')


def test_exactly_one_strategy_class():
    with pytest.raises(PluginMetaError):
        PluginSource.from_text("x", '"""h"""\nclass A:\n    pass\n')
    with pytest.raises(PluginMetaError):
        PluginSource.from_text("x", '"""h"""\nclass A(Strategy):\n    pass\nclass B(Strategy):\n    pass\n')


def test_the_error_names_the_plugin_and_its_hash():
    with pytest.raises(PluginMetaError) as e:
        PluginSource.from_text("x", b'class A:\n    pass\n')
    assert e.value.name == "x" and e.value.hash == blob_hash(b'class A:\n    pass\n')


def test_a_judge_of_a_plugin_with_computed_metadata_is_refused(cfg, plugins):
    text = (plugins / "donchian_atr.py").read_text()
    old = '    grid = {"n": [20, 40, 80], "atr_mult": [2.0, 3.0, 4.0]}'
    assert old in text
    add_plugin(plugins, "computed", text.replace(old, '    grid = {"n": list(range(3))}'))
    with pytest.raises(JudgeRefused, match="computed"):
        judge(cfg, "computed", "4h", validate=StubValidate({}))


_LCE_SESSION = {"tz": "America/New_York", "days": [0, 1, 2, 3, 4], "start": "09:40", "end": "15:45"}


def test_session_literal_parsed():
    m = PluginSource.from_text("x", f'"""h"""\nclass A(Strategy):\n    session = {_LCE_SESSION!r}\n').meta
    assert m.session == _LCE_SESSION


def test_session_default_empty():
    assert PluginSource.from_text("x", "class A(Strategy):\n    pass\n").meta.session == {}


def test_session_must_be_literal():
    with pytest.raises(PluginMetaError):
        PluginSource.from_text("x", '"""h"""\nclass A(Strategy):\n    session = dict(tz="UTC")\n')


@pytest.mark.parametrize("value", ['{"tz": "UTC"}', '{"tz": "UTC", "days": [7], "start": "09:00", "end": "10:00"}', '[]'])
def test_session_bad_shape_refused(value):
    with pytest.raises(PluginMetaError):
        PluginSource.from_text("x", f'"""h"""\nclass A(Strategy):\n    session = {value}\n')
