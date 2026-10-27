import io
import math

import numpy as np
import pandas as pd
import pytest

from perp_lab.sandbox.decisions import Decisions
from perp_lab.sandbox.errors import ProtocolError
from perp_lab.sandbox.protocol import (
    MAX_HEADER, WORKER_DTYPES, Arrow, decisions_from, check_random, decisions_sections, read_frame, table_bytes,
    table_frame, write_frame)
from perp_lab.strategy import Signal
from tests.conftest import make_bars

MIB = 1 << 20


def frame(header, sections=None):
    b = io.BytesIO()
    write_frame(b, header, sections)
    return b.getvalue()


def raw(header_obj, body=b""):
    import json
    h = json.dumps(header_obj).encode()
    return len(h).to_bytes(4, "big") + h + body


class Counting(io.BytesIO):
    """Counts the bytes handed out, so a test can see what was read."""
    consumed = 0

    def read(self, n=-1):
        d = super().read(n)
        self.consumed += len(d)
        return d


def test_round_trip_of_every_worker_dtype_and_nan_in_header():
    secs = {"a": np.arange(5, dtype="<i8"), "b": np.array([-1, 0, 3], dtype="<i1"),
            "c": np.array([1.5, math.nan, -2.0], dtype="<f8")}
    h, got = read_frame(io.BytesIO(frame({"type": "x", "v": math.nan}, secs)), max_bytes=MIB, from_worker=True)
    assert h["type"] == "x" and math.isnan(h["v"])
    assert set(WORKER_DTYPES) == {"<i8", "<i1", "<f8"}
    for k, a in secs.items():
        assert got[k].dtype == a.dtype and np.array_equal(got[k], a, equal_nan=True)


def test_empty_array_round_trips():
    _, got = read_frame(io.BytesIO(frame({}, {"a": np.zeros(0, "<f8")})), max_bytes=MIB, from_worker=True)
    assert len(got["a"]) == 0 and got["a"].dtype == np.dtype("<f8")


def test_bytes_and_arrow_come_from_the_parent_only():
    data = frame({}, {"raw": b"abc", "tbl": Arrow(b"xyz")})
    _, got = read_frame(io.BytesIO(data), max_bytes=MIB, from_worker=False)
    assert got == {"raw": b"abc", "tbl": b"xyz"} and type(got["raw"]) is bytes
    for sec in ({"raw": b"abc"}, {"tbl": Arrow(b"xyz")}):
        with pytest.raises(ProtocolError):
            read_frame(io.BytesIO(frame({}, sec)), max_bytes=MIB, from_worker=True)


def test_oversized_declaration_is_refused_before_any_section_byte_is_read():
    n = 65 * MIB // 8
    f = Counting(raw({"sections": [{"name": "a", "dtype": "<f8", "n": n}]}, b"\0" * 4096))
    with pytest.raises(ProtocolError):
        read_frame(f, max_bytes=64 * MIB, from_worker=True)
    assert f.consumed == 4 + int.from_bytes(f.getvalue()[:4], "big")   # prefix and header only


def test_total_across_sections_is_checked():
    h = {"sections": [{"name": "a", "dtype": "<i8", "n": 100}, {"name": "b", "dtype": "<i8", "n": 100}]}
    with pytest.raises(ProtocolError):
        read_frame(Counting(raw(h)), max_bytes=1599, from_worker=True)


@pytest.mark.parametrize("data", [
    b"\x00\x00",                                   # truncated length
    (10).to_bytes(4, "big") + b"{}",               # truncated header
    (0).to_bytes(4, "big"),                        # zero-length header
    (MAX_HEADER + 1).to_bytes(4, "big") + b"{}",   # header too long
    raw([1, 2]),                                   # not an object
    (2).to_bytes(4, "big") + b"\xff\xfe",          # not UTF-8
    (3).to_bytes(4, "big") + b"{x}",               # not JSON
    (MAX_HEADER).to_bytes(4, "big") + b"[" * MAX_HEADER,  # deep nesting
    raw({"sections": [{"name": "a", "dtype": "<i8", "n": 2}]}, b"\0" * 8),  # truncated section
])
def test_malformed_frames_are_protocol_errors(data):
    with pytest.raises(ProtocolError):
        read_frame(io.BytesIO(data), max_bytes=MIB, from_worker=True)


@pytest.mark.parametrize("sections", [
    "x", [1], [{"name": "a", "dtype": "<i8"}], [{"name": 1, "dtype": "<i8", "n": 0}],
    [{"name": "a", "dtype": "<i8", "n": -1}], [{"name": "a", "dtype": "<i8", "n": 1.0}],
    [{"name": "a", "dtype": "<i8", "n": True}], [{"name": "a", "dtype": "<i4", "n": 0}],
    [{"name": "a", "dtype": "<i8", "n": 0, "x": 1}],
    [{"name": "a", "dtype": "<i8", "n": 0}, {"name": "a", "dtype": "<i8", "n": 0}],
])
def test_bad_section_entries_are_protocol_errors(sections):
    with pytest.raises(ProtocolError):
        read_frame(io.BytesIO(raw({"sections": sections})), max_bytes=MIB, from_worker=True)


def test_empty_input_is_eof():
    with pytest.raises(EOFError):
        read_frame(io.BytesIO(b""), max_bytes=MIB, from_worker=True)


def test_short_reads_from_a_pipe_are_reassembled():
    class Dribble(io.BytesIO):
        def read(self, n=-1):
            return super().read(min(n, 3))
    secs = {"a": np.arange(10, dtype="<i8")}
    h, got = read_frame(Dribble(frame({"k": 1}, secs)), max_bytes=MIB, from_worker=True)
    assert h["k"] == 1 and np.array_equal(got["a"], secs["a"])


def test_write_refuses_a_foreign_dtype_and_an_oversized_header():
    with pytest.raises(ValueError):
        frame({}, {"a": np.zeros(2, dtype="<i4")})
    with pytest.raises(ValueError):
        frame({"x": "y" * MAX_HEADER})


def test_table_round_trip_keeps_index_and_regime():
    df = make_bars()
    out = table_frame(table_bytes(df))
    pd.testing.assert_frame_equal(out, df)
    assert "regime" in out.columns


def test_decisions_round_trip_and_validate():
    dec = Decisions.from_log([(2, Signal("long", 9.0, None)), (5, Signal("hold", None, math.nan))])
    secs = decisions_sections(dec)
    _, got = read_frame(io.BytesIO(frame({}, secs)), max_bytes=MIB, from_worker=True)
    assert decisions_from(got, 10).same(dec)
    with pytest.raises(ProtocolError):
        decisions_from(got, 5)            # bar 5 is out of range
    with pytest.raises(ProtocolError):
        decisions_from({k: v for k, v in got.items() if k != "i"}, 10)


@pytest.mark.parametrize("bad", [1.5, -0.1, None, "0.5", True, float("nan")])
def test_the_protocol_refuses_a_long_share_outside_0_1(bad):
    with pytest.raises(ProtocolError):
        check_random({"seed": 1, "prob": 0.1, "long": bad})


def test_the_protocol_takes_a_random_spec_in_range():
    spec = {"seed": 1, "prob": 0.1, "long": 1.0}
    assert check_random(spec) == spec and check_random(None) is None
