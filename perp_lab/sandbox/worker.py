"""A confined worker: `python -s -P -m perp_lab.sandbox.worker --deny-probe PATH --memory-mb N --cpu-s N
--parent-pid PID`.

Order, which is the point of this module:

  a. move the protocol to private descriptors and point fds 0 and 1 at stderr, so a plugin's
     print() cannot corrupt the framing;
  b. read the limits the parent passes (`--cpu-s` is only a lifetime backstop: the parent
     enforces each request's limits itself), and die with the parent: the parent-death signal
     is SIGKILL (seccomp later stops plugin code from clearing it), and a worker whose parent
     is already gone (getppid() is not `--parent-pid`) exits at once. EOF reaches only an idle
     worker; this reaches a busy one, which may hold holdout bars;
  c. confine this process (`confine.confine`), while it has one thread and has imported nothing
     but the standard library and `perp_lab.sandbox.confine`;
  d. import `session` (numpy, pandas, pyarrow, the engine), whose threads inherit the confinement;
  e. self-test: opening the deny-probe (the parent's config file), creating a socket, fork and
     shmget must each fail;
  f. report ready with the Landlock ABI;
  g. serve requests until EOF on the protocol descriptor, then exit 0 at once (no wait for
     threads a plugin may have started).

A failure in a-e is the host's, not the plugin's: an "unavailable" error, exit 3. Until (g) no
plugin byte has been read. `session` is the only module that executes plugin code.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import signal
import struct
import sys

from perp_lab.sandbox import confine

UNAVAILABLE_EXIT = 3
ORPHANED_EXIT = 5                             # the parent was gone before the worker could follow it
_PR_SET_PDEATHSIG = 1
_SHM_FLAGS = 0o1000 | 0o600                   # IPC_CREAT | 0600
_IPC_RMID = 0


def _send(f, header: dict) -> None:
    """One frame with no sections, in the protocol's format, using the standard library only."""
    h = json.dumps({**header, "sections": []}).encode()
    f.write(struct.pack(">I", len(h)) + h)
    f.flush()


def _args(argv: list[str]) -> tuple[str, int, int, int]:
    opts = dict(zip(argv[::2], argv[1::2]))
    if len(argv) % 2 or set(opts) != {"--deny-probe", "--memory-mb", "--cpu-s", "--parent-pid"}:
        raise ValueError(f"usage: --deny-probe PATH --memory-mb N --cpu-s N --parent-pid PID, got {argv}")
    return opts["--deny-probe"], int(opts["--memory-mb"]), int(opts["--cpu-s"]), int(opts["--parent-pid"])


def _follow_parent(parent_pid: int) -> None:
    """SIGKILL this process when the parent goes; exit now if it already has."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        raise RuntimeError(f"prctl(PR_SET_PDEATHSIG): {os.strerror(ctypes.get_errno())}")
    if os.getppid() != parent_pid:
        os._exit(ORPHANED_EXIT)


def _self_test(deny_probe: str) -> None:
    """Each of these must fail now; if one works, confinement did not take."""
    try:
        open(deny_probe, "rb").close()
    except PermissionError:
        pass
    except OSError as e:                      # e.g. ENOENT would prove nothing
        raise RuntimeError(f"deny-probe {deny_probe}: expected EACCES, got {errno.errorcode.get(e.errno, e.errno)}")
    else:
        raise RuntimeError(f"deny-probe {deny_probe} was readable")
    import socket
    try:
        socket.socket().close()
    except OSError:
        pass
    else:
        raise RuntimeError("socket() worked")
    try:
        pid = os.fork()
    except OSError:
        pass
    else:
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
        raise RuntimeError("fork() worked")
    libc = ctypes.CDLL(None, use_errno=True)
    sid = libc.shmget(0, 4096, _SHM_FLAGS)    # IPC_PRIVATE
    if sid != -1:
        libc.shmctl(sid, _IPC_RMID, None)
        raise RuntimeError("shmget() worked")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # a. The protocol moves to private descriptors; fds 0 and 1 become stderr.
    try:
        proto_in, proto_out = os.dup(0), os.dup(1)
        os.dup2(2, 1)
        os.dup2(2, 0)
        sys.stdout.reconfigure(line_buffering=True)   # a plugin's print reaches stderr as it happens
    except OSError:
        return UNAVAILABLE_EXIT
    out = os.fdopen(proto_out, "wb")
    inp = os.fdopen(proto_in, "rb")
    try:
        deny_probe, memory_mb, cpu_s, parent_pid = _args(argv)                       # b
        _follow_parent(parent_pid)
        abi = confine.confine(memory_mb=memory_mb, cpu_s=cpu_s, read_paths=confine.allowlist())  # c
        from perp_lab.sandbox import session                                         # d
        _self_test(deny_probe)                                                       # e
    except BaseException as e:
        _send(out, {"error": {"kind": "unavailable", "type": type(e).__name__, "message": str(e)[:1000]}})
        return UNAVAILABLE_EXIT
    _send(out, {"ok": True, "ready": True, "abi": abi})                              # f
    return _serve(inp, out, memory_mb << 20, session)                                # g


def _handle(session, sess, header: dict, sections: dict) -> tuple[dict, dict]:
    from perp_lab.sandbox.errors import ProtocolError
    from perp_lab.sandbox.protocol import table_frame

    op = header.get("op")
    if op == "load":
        return {"ok": True, **sess.load(header["name"], sections["source"])}, {}
    if op == "bars":
        return {"ok": True, "n": sess.bars(header["symbol"], header["tf"], table_frame(sections["table"]))}, {}
    if op == "run":
        result = sess.run(header["params"], header["ep"], header["k"], header["start_i"], header["random"],
                          header["features"])
        return session.run_reply(result, header["features"])
    if op == "stream_open":
        return session.stream_open_reply(sess.stream_open(header["params"], header["ep"], header["warmup_k"]))
    if op == "step":
        table = table_frame(sections["table"]) if "table" in sections else None
        return session.step_reply(sess.step(header["position"], header["mode"], table))
    raise ProtocolError(f"unknown op {op!r}")


def _error(session, exc: BaseException) -> dict:
    try:
        return session.error_reply(exc)
    except BaseException:                     # an exception class built to break even this
        return {"error": {"kind": "exception", "type": "", "message": ""}}


def _flush() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except BaseException:                 # a plugin may have replaced or closed them
            pass


def _serve(inp, out, max_bytes: int, session) -> int:
    from perp_lab.sandbox.errors import PluginError
    from perp_lab.sandbox.protocol import read_frame, write_frame

    sess = session.Session()
    try:
        while True:
            try:
                header, sections = read_frame(inp, max_bytes=max_bytes, from_worker=False)
            except EOFError:
                return 0                      # the parent closed the pipe, or went away
            except PluginError as e:          # the trusted side sent garbage: nothing to sync on
                write_frame(out, _error(session, e))
                return 4
            try:
                reply, secs = _handle(session, sess, header, sections)
            except BaseException as e:        # plugin code may raise anything, SystemExit included
                reply, secs = _error(session, e), {}
            _flush()                          # what the plugin printed goes out before the reply
            try:
                write_frame(out, reply, secs)
            except (ValueError, TypeError) as e:  # a reply that can't be framed; nothing was written
                write_frame(out, _error(session, PluginError("protocol", "", f"reply not sendable: {e}")))
    except OSError:                           # the parent's end of the pipe is gone
        return 0


if __name__ == "__main__":
    code = main()
    _flush()
    os._exit(code)                            # without waiting for any thread a plugin started
