"""YAML config → plain dict-backed object. Kept deliberately thin."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_CONFIG = Path("config/lab.yaml")

TF_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


@dataclass
class Config:
    raw: dict
    path: Path = field(default=DEFAULT_CONFIG)

    def __getitem__(self, k):
        return self.raw[k]

    def get(self, k, default=None):
        return self.raw.get(k, default)

    @property
    def data_dir(self) -> Path:
        return Path(self.raw["data_dir"])

    @property
    def runs_dir(self) -> Path:
        return Path(self.raw["runs_dir"])

    @property
    def symbols(self) -> list[str]:
        q = self.raw.get("quote", "USDT")
        return [f"{b}{q}" for b in self.raw["universe"]]

    @property
    def history_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.raw["history_start"], tz="UTC")

    @property
    def holdout_start(self) -> pd.Timestamp:
        return pd.Timestamp(self.raw["holdout_start"], tz="UTC")

    def bracket(self, symbol: str) -> dict:
        b = self.raw.get("margin_brackets", {})
        return b.get(symbol, b.get("default", {"mmr": 0.01, "cum": 0.0}))


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p) as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, path=p)


def ccxt_symbol(symbol: str, quote: str = "USDT") -> str:
    """BTCUSDT -> BTC/USDT:USDT (ccxt unified perp symbol)."""
    base = symbol[: -len(quote)]
    return f"{base}/{quote}:{quote}"
