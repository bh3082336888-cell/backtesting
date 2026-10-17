"""Look 7 (SYNTHETIC bars only; no market data is loaded): does the drafted spec run in the engine,
signal at its defaults on the judge's synthetic bars, keep its time exit exactly, and stay causal
at the judge's cut points? Uses perp_lab.validate.causality.synthetic_bars / with_moving_columns,
the same generator the judge's lookahead check uses. In-process (not the sandbox)."""

import sys

import pandas as pd

sys.path.insert(0, "/tmp/claude-1000/-home-burhan-backtesting/464fe316-f53a-4226-89ab-e02f2b7426f3/scratchpad/research")
from flow_shock_fade_draft import FlowShockFade  # noqa: E402

from perp_lab.engine.backtest import EngineParams, run_backtest  # noqa: E402
from perp_lab.plugins import parse_meta  # noqa: E402
from perp_lab.validate.causality import CUTS, synthetic_bars, with_moving_columns  # noqa: E402

src = open(sys.modules["flow_shock_fade_draft"].__file__, "rb").read()
meta = parse_meta(src, "flow_shock_fade")
print("meta:", meta.hypothesis[:60], "...", meta.params, meta.grid, meta.timeframes, "parent:", meta.parent)

for label, bars in [("raw synthetic_bars(seed=1)", synthetic_bars(seed=1)),
                    ("with_moving_columns(synthetic_bars(seed=1))", with_moving_columns(synthetic_bars(seed=1)))]:
    for kw in [{}, {"flow_x": 1.0}, {"hold_bars": 6}, {"hold_bars": 24}]:
        s = FlowShockFade(**kw)
        f = s.prepare(bars)
        r = run_backtest(s, bars, f, symbol="X", tf="1h", ep=EngineParams(), record_signals=True)
        tr = pd.DataFrame(r.trades)
        held = sorted(tr.bars_held.unique().tolist()) if len(tr) else []
        reasons = tr.exit_reason.value_counts().to_dict() if len(tr) else {}
        print(f"{label} {kw or 'defaults'}: signals {len(r.signals)}, trades {len(tr)}, "
              f"exit reasons {reasons}, bars_held values {held[:6]}{'...' if len(held) > 6 else ''}, "
              f"first finite flow row {int(f.flow.notna().values.argmax()) if f.flow.notna().any() else None}, "
              f"first finite mv row {int(f.mv.notna().values.argmax())}, rejected {r.rejected}")

# causality at the judge's cuts, defaults and grid corners, in process
bars = with_moving_columns(synthetic_bars(seed=1))
bad = []
for kw in [{}, {"flow_x": 0.5}, {"flow_x": 1.0}, {"hold_bars": 6}, {"hold_bars": 24}]:
    full = FlowShockFade(**kw)
    ff = full.prepare(bars)
    rf = run_backtest(full, bars, ff, symbol="X", tf="1h", ep=EngineParams(), record_signals=True)
    for k in CUTS:
        s = FlowShockFade(**kw)
        fc = s.prepare(bars.iloc[:k])
        try:
            pd.testing.assert_frame_equal(ff.iloc[:k], fc, check_exact=False, rtol=1e-9, atol=1e-12)
        except AssertionError as e:
            bad.append((kw, k, "features", str(e)[:80]))
        rc = run_backtest(s, bars.iloc[:k], fc, symbol="X", tf="1h", ep=EngineParams(), record_signals=True)
        if [x for x in rf.signals if x[0] < k - 1] != [x for x in rc.signals if x[0] < k - 1]:
            bad.append((kw, k, "signals"))
print("causality failures at cuts", CUTS, ":", bad or "none")
