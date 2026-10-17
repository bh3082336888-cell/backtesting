"""Framed pipe protocol between the trusted parent and a confined worker.

Frame: `>I` header length H (1 <= H <= MAX_HEADER), H bytes of a UTF-8 JSON object (NaN allowed),
then, for each entry of `header["sections"]` (a list of {"name", "dtype", "n"}), exactly
`n * itemsize` bytes.

Trust: `read_frame(..., from_worker=True)` is the parent's boundary. It never unpickles and never
parses Arrow; it accepts only the raw little-endian dtypes in WORKER_DTYPES, and checks every
declared size (and their total) against `max_bytes` before it reads a single section byte.

Arrow: `table_bytes` encodes a DataFrame to Arrow IPC; the parent uses it to send bars to the
worker. `table_frame` decodes one and is called only inside the worker. The parent never parses
Arrow from a worker, and `read_frame` refuses an "arrow" section from one.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pandas as pd

from perp_lab.sandbox.decisions import Decisions
from perp_lab.sandbox.errors import ProtocolError

MAX_HEADER = 65536
WORKER_DTYPES = {"<i8", "<i1", "<f8"}
_ITEMSIZE = {"<i8": 8, "<i1": 1, "<f8": 8, "bytes": 1, "arrow": 1}
_DECISION_KEYS = ("i", "action", "stop", "target", "has_stop", "has_target")


class Arrow(bytes):
    """Marks a section as an Arrow IPC table (parent to worker only)."""


def write_frame(f, header: dict, sections: dict | None = None) -> None:
    entries, bodies = [], []
    for name, v in (sections or {}).items():
        if isinstance(v, Arrow):
            dtype, body, n = "arrow", bytes(v), len(v)
        elif isinstance(v, (bytes, bytearray, memoryview)):
            dtype, body, n = "bytes", bytes(v), len(v)
        elif isinstance(v, np.ndarray) and v.ndim == 1:
            dtype = next((d for d in WORKER_DTYPES if v.dtype == np.dtype(d)), None)
            if dtype is None:
                raise ValueError(f"section {name!r}: dtype {v.dtype} is not one of {sorted(WORKER_DTYPES)}")
            body, n = np.ascontiguousarray(v).tobytes(), len(v)
        else:
            raise ValueError(f"section {name!r} must be bytes or a 1-d array")
        entries.append({"name": name, "dtype": dtype, "n": n})
        bodies.append(body)
    h = json.dumps({**header, "sections": entries}, allow_nan=True).encode("utf-8")
    if len(h) > MAX_HEADER:
        raise ValueError(f"header is {len(h)} bytes, over {MAX_HEADER}")
    f.write(struct.pack(">I", len(h)) + h)
    for body in bodies:
        f.write(body)
    f.flush()


def _read_exactly(f, n: int, what: str) -> bytes:
    chunks, got = [], 0
    while got < n:
        c = f.read(n - got)
        if not c:
            raise ProtocolError(f"stream ended inside the {what}")
        chunks.append(c)
        got += len(c)
    return b"".join(chunks)


def _parse_sections(header: dict, *, max_bytes: int, from_worker: bool) -> list[tuple[str, str, int]]:
    raw = header.get("sections", [])
    if not isinstance(raw, list):
        raise ProtocolError("header sections must be a list")
    allowed = WORKER_DTYPES if from_worker else set(_ITEMSIZE)
    out, seen, total = [], set(), 0
    for e in raw:
        if not isinstance(e, dict) or set(e) != {"name", "dtype", "n"}:
            raise ProtocolError("malformed section entry")
        name, dtype, n = e["name"], e["dtype"], e["n"]
        if not isinstance(name, str) or name in seen:
            raise ProtocolError("section name must be a unique string")
        if not isinstance(dtype, str) or dtype not in allowed:
            raise ProtocolError(f"section dtype {dtype!r} is not allowed here")
        if type(n) is not int or n < 0:
            raise ProtocolError("section length must be a non-negative int")
        seen.add(name)
        total += n * _ITEMSIZE[dtype]
        if total > max_bytes:
            raise ProtocolError(f"frame declares more than {max_bytes} bytes")
        out.append((name, dtype, n))
    return out


def read_frame(f, *, max_bytes: int, from_worker: bool) -> tuple[dict, dict]:
    first = f.read(4)
    if not first:
        raise EOFError("stream closed before a frame")
    hlen = struct.unpack(">I", first + _read_exactly(f, 4 - len(first), "length prefix"))[0]
    if not 1 <= hlen <= MAX_HEADER:
        raise ProtocolError(f"header length {hlen} out of range")
    body = _read_exactly(f, hlen, "header")
    try:
        header = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise ProtocolError(f"header is not valid JSON: {type(e).__name__}") from None
    if not isinstance(header, dict):
        raise ProtocolError("header must be a JSON object")
    layout = _parse_sections(header, max_bytes=max_bytes, from_worker=from_worker)
    sections = {}
    for name, dtype, n in layout:
        data = _read_exactly(f, n * _ITEMSIZE[dtype], f"section {name!r}")
        sections[name] = data if dtype in ("bytes", "arrow") else np.frombuffer(data, dtype=dtype)
    return header, sections


def table_bytes(df: pd.DataFrame) -> bytes:
    import pyarrow as pa
    t = pa.Table.from_pandas(df, preserve_index=True)
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, t.schema) as w:
        w.write_table(t)
    return sink.getvalue().to_pybytes()


def table_frame(b: bytes) -> pd.DataFrame:
    """Decode Arrow IPC. Worker side only: the parent never parses Arrow from a worker."""
    import pyarrow as pa
    return pa.ipc.open_stream(pa.py_buffer(b)).read_all().to_pandas()


def decisions_sections(dec: Decisions) -> dict:
    return {k: getattr(dec, k) for k in _DECISION_KEYS}


def decisions_from(sections: dict, n_bars: int) -> Decisions:
    missing = [k for k in _DECISION_KEYS if k not in sections]
    if missing:
        raise ProtocolError(f"decision sections missing: {missing}")
    dec = Decisions(*(sections[k] for k in _DECISION_KEYS))
    dec.validate(n_bars)
    return dec
