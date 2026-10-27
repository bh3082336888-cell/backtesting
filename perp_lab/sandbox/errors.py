"""Error types shared by the sandbox's parent and worker sides."""

from __future__ import annotations

import builtins

KINDS = ("exception", "cpu_limit", "protocol", "mismatch", "metadata", "features", "died", "timeout")
# Kinds the machine can cause (a killed worker, a slow or loaded host), not only the plugin
TRANSIENT_KINDS = ("died", "timeout", "cpu_limit")
# Exception names a verdict may carry; a plugin picks its own class names, so any other becomes "other"
BUILTIN_EXCEPTIONS = frozenset(n for n, o in vars(builtins).items()
                               if isinstance(o, type) and issubclass(o, BaseException))
MESSAGE_CHARS = 200                     # an error's type and message, as they may reach a verdict

_BYTES = bytes(c if 0x20 <= c < 0x7F or c in (9, 10) else 0x3F for c in range(256))


def ascii_text(s, limit: int = MESSAGE_CHARS) -> str:
    """At most `limit` printable ASCII characters: whitespace becomes a space, anything else '?'.

    Applied to every string a worker supplies, on both sides of the pipe. Anything but a plain str
    (a subclass could override slicing) becomes ''.
    """
    if type(s) is not str:
        return ""
    return "".join(c if " " <= c <= "~" else (" " if c in "\t\n\r" else "?") for c in s[:limit])


def ascii_bytes(b: bytes) -> bytes:
    """Printable ASCII, newlines and tabs kept; every other byte becomes '?'. For worker stderr."""
    return bytes(b).translate(_BYTES)


class SandboxUnavailable(RuntimeError):
    """The host cannot confine plugin code, so plugin code does not run."""


class PluginError(Exception):
    def __init__(self, kind: str, type_: str = "", message: str = "", stderr: bytes = b""):
        super().__init__(kind, type_, message)
        self.kind = kind
        self.type = type_
        self.message = message
        self.stderr = stderr

    def reason(self) -> str:
        r = f"plugin_error: {self.kind}"
        if self.kind == "exception":
            r += f": {self.type}: {self.message}"
        return r

    def public_reason(self) -> str:
        """The reason a verdict may carry: the kind and a builtin exception type, never the message."""
        r = f"plugin_error: {self.kind}"
        if self.kind == "exception":
            r += f": {self.type if self.type in BUILTIN_EXCEPTIONS else 'other'}"
        return r

    def __reduce__(self):               # every field, so it crosses a process pool intact
        return PluginError, (self.kind, self.type, self.message, self.stderr)


class ProtocolError(PluginError):
    def __init__(self, message: str = "", stderr: bytes = b""):
        super().__init__("protocol", "", message, stderr)

    def __reduce__(self):
        return ProtocolError, (self.message, self.stderr)


class WorkerLost(Exception):
    """The worker died or ran out of wall-clock time. Retried once in a fresh worker."""

    def __init__(self, kind: str, message: str = "", stderr: bytes = b""):
        if kind not in ("died", "timeout"):
            raise ValueError(f"WorkerLost kind must be died or timeout, not {kind!r}")
        super().__init__(kind, message)
        self.kind = kind
        self.message = message
        self.stderr = stderr
