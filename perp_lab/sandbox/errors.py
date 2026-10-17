"""Error types shared by the sandbox's parent and worker sides."""

from __future__ import annotations

KINDS = ("exception", "cpu_limit", "protocol", "mismatch", "metadata", "features", "died", "timeout")


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


class ProtocolError(PluginError):
    def __init__(self, message: str = "", stderr: bytes = b""):
        super().__init__("protocol", "", message, stderr)


class WorkerLost(Exception):
    def __init__(self, kind: str, message: str = ""):
        if kind not in ("died", "timeout"):
            raise ValueError(f"WorkerLost kind must be died or timeout, not {kind!r}")
        super().__init__(kind, message)
        self.kind = kind
        self.message = message
