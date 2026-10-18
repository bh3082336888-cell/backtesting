"""Shared loader for the research looks. Data only through perp_lab.data.store with the research
config; the default end is the holdout start (2026-07-01), and no end or token is ever passed."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from perp_lab.config import load_config
from perp_lab.data.store import load_bars

CFG_PATH = Path(__file__).with_name("lab.yaml")     # run from the repo root: its data_dir is ./data
CFG = load_config(CFG_PATH)
SYMBOLS = CFG.symbols
TFS = ["15m", "1h", "4h"]
HOLDOUT = CFG.holdout_start


@lru_cache(maxsize=32)
def bars(symbol: str, tf: str) -> pd.DataFrame:
    b = load_bars(CFG, symbol, tf)          # end defaults to holdout_start
    assert b.index.max() < HOLDOUT, "research look must not see the holdout"
    return b
