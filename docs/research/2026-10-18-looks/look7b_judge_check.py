"""Look 7b (SYNTHETIC bars only; no market data): run the harness's own causality_failures on the
scratch draft, with the in-process worker twin the test suite uses, as the judge's lookahead check
would. Confirms the spec's defaults are verifiable (signal on the judge's synthetic bars)."""
import sys
sys.path.insert(0, "/home/burhan/backtesting-research")
from perp_lab.config import load_config
from perp_lab.plugins import PluginSource
from perp_lab.sandbox import client
from perp_lab.validate.causality import causality_failures
from tests import sandbox_fixtures

client.spawn = sandbox_fixtures.InProcessWorker
client.check = lambda cfg: 8
cfg = load_config("/tmp/claude-1000/-home-burhan-backtesting/464fe316-f53a-4226-89ab-e02f2b7426f3/scratchpad/research/lab.yaml")
text = open("/tmp/claude-1000/-home-burhan-backtesting/464fe316-f53a-4226-89ab-e02f2b7426f3/scratchpad/research/flow_shock_fade_draft.py", "rb").read()
src = PluginSource.from_text("flow_shock_fade", text)
print("judge-style causality_failures (default synthetic bars):", causality_failures(cfg, src) or "none")
# Counter-check: the same draft with the default flow_x raised to 1.0 must be reported unverifiable.
raised = PluginSource.from_text("flow_shock_fade", text.replace(b'"flow_x": 0.5, "move_k"', b'"flow_x": 1.0, "move_k"'))
print("same check with default flow_x = 1.0:", causality_failures(cfg, raised) or "none")
