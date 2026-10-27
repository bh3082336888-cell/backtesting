"""Perp Lab dashboard. `lab dashboard` or `streamlit run dashboard/app.py`.

Reads the results store + run artifacts. The only write it can perform is the
human paper-approval gate (and retire). It cannot set anything to live.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sqlalchemy as sa
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import trade_chart as tc  # noqa: E402
from perp_lab.config import load_config  # noqa: E402
from perp_lab.store import db  # noqa: E402

st.set_page_config(page_title="Perp Lab", layout="wide")


def _cfg_path() -> str | None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    a, _ = p.parse_known_args(sys.argv[1:])
    return a.config or os.environ.get("PERP_LAB_CONFIG")


CFG = load_config(_cfg_path())

# ---------------------------------------------------------------- palette (dataviz reference instance)
DARK = getattr(getattr(st.context, "theme", None), "type", None) == "dark"
C = {
    "s1": "#3987e5" if DARK else "#2a78d6",          # series 1 / positive pole
    "s2": "#d95926" if DARK else "#eb6834",          # series 2
    "neg": "#e66767" if DARK else "#e34948",         # negative pole
    "pos": "#0ca30c",                                # status good: trade charts' reward box and TP
    "mid": "#383835" if DARK else "#f0efec",         # diverging midpoint
    "muted": "#898781",                              # IS / reference ink
    "grid": "#2c2c2a" if DARK else "#e1e0d9",
    "band": "rgba(255,255,255,0.04)" if DARK else "rgba(11,11,11,0.035)",
    # trade charts: neutral candles so the trades stand out
    "train": "rgba(255,255,255,0.08)" if DARK else "rgba(11,11,11,0.07)",
    "ink": "#e8e6e1" if DARK else "#1f1f1d",
    "candle": "#6b6a66" if DARK else "#a9a7a0",
    "liq": "#a35cff" if DARK else "#8e2dff",
    "eow": "#2bb39a" if DARK else "#1f9e89",
    "bg": "#0e1117" if DARK else "#ffffff",         # the page behind the chart: dot rings, label plates
}
DIVERGING = [[0.0, C["neg"]], [0.5, C["mid"]], [1.0, C["s1"]]]


def style(fig: go.Figure, h=320, **kw) -> go.Figure:
    title = kw.pop("title", None)
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=64, b=8), hovermode=kw.pop("hovermode", "closest"),
                      title=dict(text=title, y=0.98, yanchor="top") if title else None,
                      legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0), **kw)
    fig.update_xaxes(showgrid=False, linecolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"], gridwidth=1, zerolinecolor=C["muted"], zerolinewidth=1)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", theme="streamlit")


# ---------------------------------------------------------------- data access

@st.cache_data(ttl=10)
def all_runs() -> pd.DataFrame:
    db.ingest_runs(CFG)
    return db.read_table(CFG, db.runs).sort_values("created_at", ascending=False)


@st.cache_data(ttl=10)
def table(name: str, run_id: str | None = None) -> pd.DataFrame:
    t = getattr(db, name)
    return db.read_table(CFG, t, t.c.run_id == run_id if run_id is not None else None)


@st.cache_data(ttl=10)
def edges() -> pd.DataFrame:
    return db.read_table(CFG, db.lineage_edges)


@st.cache_data(ttl=60)
def artifact(run_dir: str, name: str) -> pd.DataFrame | None:
    p = Path(run_dir) / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_data(ttl=60)
def summary(run_dir: str) -> dict:
    p = Path(run_dir) / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(ttl=60)
def chart_trades(mode: str, run_dir: str) -> pd.DataFrame:
    return tc.load_trades(mode, run_dir)


@st.cache_data(ttl=60)
def chart_span(mode: str, run_dir: str, symbol: str, tf: str, date_start, date_end):
    return tc.chart_span(CFG, mode, run_dir, symbol, tf, date_start, date_end)


@st.cache_data(ttl=60)
def chart_candles(mode: str, run_dir: str, symbol: str, tf: str, start, end) -> pd.DataFrame:
    return tc.run_candles(CFG, mode, run_dir, symbol, tf, start, end)


def status_of(r) -> str:
    return db.current_status(CFG, r["strategy"], r["timeframe"], r["verdict"])


def fmt_pct(x):
    return "—" if x is None or pd.isna(x) else f"{x:.1%}"


def fmt2(x):
    return "—" if x is None or pd.isna(x) else f"{x:.2f}"


def latest_wf(runs: pd.DataFrame) -> pd.DataFrame:
    wf = runs[runs.validation_mode == "walkforward"]
    return wf.sort_values("created_at").groupby(["strategy", "timeframe"], as_index=False).tail(1)


# ---------------------------------------------------------------- pages

def page_strategies():
    st.header("Strategies")
    st.caption("Latest walk-forward run per strategy × timeframe. All metrics are out-of-sample. "
               "IS Sharpe is shown greyed for reference only.")
    runs = all_runs()
    wf = latest_wf(runs)
    if wf.empty:
        st.info("No walk-forward runs yet. Run `lab walkforward <strategy> --tf <tf>`.")
        return
    reg = db.read_table(CFG, db.run_regimes)
    rows = []
    for _, r in wf.iterrows():
        rr = reg[reg.run_id == r.run_id].set_index("regime")
        row = {"strategy": r.strategy, "tf": r.timeframe, "status": status_of(r),
               "OOS Sharpe": r.sharpe, "Max DD": r.max_dd, "trades": r.trades,
               "vs random (pctile)": r.random_pctile, "narrow island": r.narrow_island}
        for g in ["trending", "ranging", "high_vol", "transition"]:
            row[f"avg R {g}"] = rr.avg_r.get(g, np.nan) if len(rr) else np.nan
        row["IS Sharpe"] = r.is_sharpe
        row["last run"] = pd.Timestamp(r.created_at).strftime("%Y-%m-%d %H:%M")
        row["run_id"] = r.run_id
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("OOS Sharpe", ascending=False)
    stat = st.multiselect("Status", ["noise", "candidate", "paper", "retired", "live"],
                          default=["noise", "candidate", "paper", "live"])
    df = df[df.status.isin(stat)]
    sty = (df.style
           .format({"OOS Sharpe": "{:.2f}", "Max DD": "{:.1%}", "vs random (pctile)": "{:.0%}",
                    "IS Sharpe": "{:.2f}", **{c: "{:.3f}" for c in df.columns if c.startswith("avg R")}},
                   na_rep="—")
           .map(lambda _: f"color: {C['muted']}", subset=["IS Sharpe"]))
    st.dataframe(sty, hide_index=True, width="stretch")


def _fold_bands(fig, folds: pd.DataFrame):
    for i, f in folds.reset_index(drop=True).iterrows():
        if i % 2 == 0:
            fig.add_vrect(x0=f.test_start, x1=f.test_end, fillcolor=C["band"], line_width=0, layer="below")


def _trade_label(t) -> str:
    return (f"{pd.to_datetime(t.entry_time, unit='ms', utc=True):%Y-%m-%d %H:%M} · "
            f"{'long' if t.side > 0 else 'short'} · {t.exit_reason} · R {t.r_multiple:+.2f}")


def _picked(dk, wk, jk):
    st.session_state[wk] = st.session_state[dk]
    st.session_state[jk] = None        # no longer a jumped-to trade, so picking it again jumps again


def _jump(jk, wk, vk, trades, tf, span):
    i = st.session_state.get(jk)
    if i is not None:
        st.session_state[wk] = tc.trade_window(trades.iloc[i], tf, span)
        st.session_state[vk] += 1      # a new date picker, showing the trade's dates


def trades_chart_section(r, rid, folds=None):
    """One symbol's candles over a date range with this run's trades on them. Walk-forward and paper only."""
    mode = r.validation_mode
    if mode not in tc.CHARTABLE:
        st.caption(f"Trade charts cover walk-forward and paper runs; this run is '{mode}'.")
        return
    trades = chart_trades(mode, r.run_dir)
    symbols = tc.chart_symbols(mode, r.run_dir, r.universe)
    if not symbols:
        st.caption("This run saved no bars to chart.")
        return
    n = trades.symbol.value_counts() if len(trades) else pd.Series(dtype=int)
    c = st.columns([2, 3, 4])
    sym = c[0].selectbox("Symbol", symbols, key=f"tc_sym_{rid}",
                         format_func=lambda s: f"{s} · {int(n.get(s, 0))} trades")
    try:
        span = chart_span(mode, r.run_dir, sym, r.timeframe, r.date_start, r.date_end)
    except FileNotFoundError as e:
        st.warning(str(e))
        return
    lo, hi = tc.as_dates(*span)
    wk, vk = f"tc_win_{rid}", f"tc_ver_{rid}"
    if wk not in st.session_state:
        st.session_state[wk], st.session_state[vk] = tc.default_window(mode, span, folds), 0
    held = tuple(st.session_state[wk])
    fit = tuple(min(max(d, lo), hi) for d in held)
    if fit != held:                    # another symbol's span (paper): show the part that fits
        st.session_state[wk], st.session_state[vk] = fit, st.session_state[vk] + 1
    dk, jk = f"tc_dates_{rid}_{st.session_state[vk]}", f"tc_jump_{rid}_{sym}"
    c[1].date_input("Dates (UTC)", value=st.session_state[wk], min_value=lo, max_value=hi, key=dk,
                    on_change=_picked, args=(dk, wk, jk))
    mine = (trades[trades.symbol == sym].sort_values("entry_time").reset_index(drop=True)
            if len(trades) else trades)
    c[2].selectbox("Jump to trade", [None, *range(len(mine))], key=jk,
                   format_func=lambda i: "—" if i is None else _trade_label(mine.iloc[i]),
                   on_change=_jump, args=(jk, wk, vk, mine, r.timeframe, span))
    start, end = tc.picked_window(st.session_state[wk], span)
    try:
        candles = chart_candles(mode, r.run_dir, sym, r.timeframe, start, end)
    except FileNotFoundError as e:
        st.warning(str(e))
        return
    if candles.empty:
        st.caption("No candles in this range.")
        return
    fig = tc.trade_figure(candles, mine, r.timeframe, palette=C, folds=folds if mode == "walkforward" else None,
                          focus=st.session_state.get(jk))       # the jumped-to trade: a row label of `mine`
    last = end - pd.Timedelta(1, "ns")
    show(style(fig, h=560, title=f"{sym} {r.timeframe} · {start:%Y-%m-%d} → {last:%Y-%m-%d} UTC"))
    shown = tc.display_tf(r.timeframe, len(candles))
    here = tc.trades_in(mine, start, end)
    src = ("the research store, the bars this walk-forward read" if mode == "walkforward"
           else "the bars this paper run saved")
    coarse = (f" Drawn as {shown} candles to stay readable; narrow the dates to see {r.timeframe}."
              if shown != r.timeframe else "")
    shade = (" Shaded: every other test fold (hover a fold label for its train window and params); "
             "every trade here is out-of-sample." if mode == "walkforward"
             else " Shaded: warmup bars from before the run started.")
    lines = ("stop (red) and target (green), as they moved" if tc.has_levels(here)
             else "initial stop (red) and initial target (green)")
    st.caption(f"{len(here)} trade{'' if len(here) == 1 else 's'} in range · candles from {src}.{coarse} "
               "Each trade: a green box from the entry to the target and a red one to the stop, which closes "
               "once the stop reaches the entry (no fixed target: the move to the exit, green for a gain, red "
               "for a loss) · bold blue line and dot: the entry at the fill · dot on the exit candle, coloured "
               f"by reason · ○ close of the signal bar · dotted: {lines}. Price and time tags (UTC) and the R:R "
               f"label show for the trade picked in Jump to trade, or for every trade when {tc.TAG_ALL} or "
               f"fewer are in range; hover a box for the whole trade.{shade}")
    if len(here):
        with st.expander(f"Trades in range ({len(here)})"):
            cols = [x for x in ["side", "signal_time", "entry_time", "entry_price", "exit_time", "exit_price",
                                "exit_reason", "pnl", "r_multiple", "regime", "fees", "funding", "fold"] if x in here]
            tbl = here[cols].copy()
            for x in ("signal_time", "entry_time", "exit_time"):
                tbl[x] = pd.to_datetime(tbl[x], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M")
            tbl["side"] = np.where(tbl.side > 0, "long", "short")
            st.dataframe(tbl.style.format({"entry_price": "{:,.6g}", "exit_price": "{:,.6g}", "pnl": "{:+,.2f}",
                                           "r_multiple": "{:+.2f}", "fees": "{:,.2f}", "funding": "{:,.2f}"},
                                          na_rep="—"),
                         hide_index=True, width="stretch")


def page_detail():
    runs = all_runs()
    wf = runs[runs.validation_mode.isin(["walkforward", "IS"])]
    if wf.empty:
        st.info("No runs yet.")
        return
    keys = sorted({f"{s} @ {t}" for s, t in zip(wf.strategy, wf.timeframe)})
    pick = st.selectbox("Strategy", keys, key="detail_pick")
    s, tf = pick.split(" @ ")
    hist = wf[(wf.strategy == s) & (wf.timeframe == tf)].sort_values("created_at", ascending=False)
    rid = st.selectbox("Run", hist.run_id.tolist(), format_func=lambda x: f"{x} ({hist.set_index('run_id').loc[x, 'validation_mode']})")
    r = hist.set_index("run_id").loc[rid]
    sm = summary(r.run_dir)
    status = db.current_status(CFG, s, tf, r.verdict)

    st.header(f"{s} @ {tf}")
    st.markdown(f"**Hypothesis:** {r.hypothesis}")
    judgement_section(rid)
    st.caption(f"run `{rid}` · plugin `{(r.strategy_hash or '')[:10]}` · mode **{r.validation_mode}** · "
               f"{r.date_start} → {r.date_end} · status **{status}**")
    if r.validation_mode == "IS":
        st.warning("In-sample run: not evidence of anything. Use walk-forward runs for decisions.")
    if r.fail_reasons:
        st.error("Verdict: NOISE — " + " · ".join(r.fail_reasons))
    elif r.verdict == "candidate":
        st.success("Verdict: CANDIDATE — passed every Phase 1 check. Now look for the bug.")

    m = st.columns(6)
    m[0].metric("OOS Sharpe", fmt2(r.sharpe))
    m[1].metric("Max DD", fmt_pct(r.max_dd))
    m[2].metric("Return", fmt_pct(r.total_return))
    m[3].metric("Profit factor", fmt2(r.profit_factor))
    m[4].metric("Trades", "—" if pd.isna(r.trades) else int(r.trades))
    m[5].metric("vs random-entry", fmt_pct(r.random_pctile))
    if not pd.isna(r.is_sharpe):
        st.markdown(f"<span style='color:{C['muted']}'>IS Sharpe {r.is_sharpe:.2f} (train windows of the "
                    f"selected params — reference only)</span>", unsafe_allow_html=True)

    folds = table("run_folds", rid)
    daily = artifact(r.run_dir, "oos_daily")
    if daily is not None and "PORTFOLIO" in daily:
        st.subheader("Walk-forward OOS equity")
        eq = (1 + daily["PORTFOLIO"].fillna(0)).cumprod()
        fig = go.Figure(go.Scatter(x=eq.index, y=eq.values, mode="lines", line=dict(color=C["s1"], width=2),
                                   name="OOS portfolio", hovertemplate="%{x|%Y-%m-%d}<br>equity %{y:.3f}<extra></extra>"))
        if len(folds):
            _fold_bands(fig, folds)
        show(style(fig, title="Stitched OOS equity (shaded = alternate test folds)", hovermode="x unified",
                   showlegend=False))
    if len(folds):
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_bar(x=folds.fold, y=folds.oos_sharpe, name="OOS Sharpe", marker_color=C["s1"],
                        hovertemplate="fold %{x}<br>OOS Sharpe %{y:.2f}<extra></extra>")
            fig.add_scatter(x=folds.fold, y=folds.is_score, name="IS score (reference)", mode="markers",
                            marker=dict(color=C["muted"], size=8, symbol="diamond"),
                            hovertemplate="fold %{x}<br>IS %{y:.2f}<extra></extra>")
            show(style(fig, title="Per fold: OOS Sharpe vs IS score", xaxis_title="fold"))
        with c2:
            fig = go.Figure()
            for i, f in folds.iterrows():
                w = daily["PORTFOLIO"][(daily.index >= pd.Timestamp(f.test_start, tz="UTC")) &
                                       (daily.index < pd.Timestamp(f.test_end, tz="UTC"))] if daily is not None else None
                if w is None or w.empty:
                    continue
                e = (1 + w.fillna(0)).cumprod() - 1
                fig.add_scatter(x=np.arange(len(e)), y=e.values, mode="lines", showlegend=False,
                                line=dict(color=C["s1"] if e.iloc[-1] >= 0 else C["neg"], width=1.5),
                                opacity=0.7, name=f"fold {f.fold}",
                                hovertemplate=f"fold {f.fold} ({f.test_start})<br>day %{{x}}<br>%{{y:.2%}}<extra></extra>")
            show(style(fig, title="Each fold's OOS curve, rebased (blue ends up, red ends down)",
                       xaxis_title="days into test window", yaxis_tickformat=".0%"))
        with st.expander("Param selection per fold"):
            st.dataframe(folds[["fold", "test_start", "test_end", "combo", "is_score", "oos_sharpe", "oos_return"]],
                         hide_index=True, width="stretch")

    st.subheader("Trades on the chart")
    trades_chart_section(r, rid, folds)

    sy = table("run_symbols", rid)
    if len(sy):
        st.subheader("Per pair (OOS)")
        c1, c2 = st.columns([3, 2])
        with c1:
            sy = sy.sort_values("sharpe")
            fig = go.Figure(go.Bar(x=sy.sharpe, y=sy.symbol, orientation="h",
                                   marker_color=[C["s1"] if v >= 0 else C["neg"] for v in sy.sharpe],
                                   hovertemplate="%{y}<br>Sharpe %{x:.2f}<extra></extra>"))
            show(style(fig, h=300, title="OOS Sharpe by pair", showlegend=False))
        with c2:
            st.dataframe(sy[["symbol", "sharpe", "max_dd", "total_return", "trades", "profit_factor"]]
                         .sort_values("sharpe", ascending=False)
                         .style.format({"sharpe": "{:.2f}", "max_dd": "{:.1%}", "total_return": "{:.1%}",
                                        "profit_factor": "{:.2f}"}, na_rep="—"),
                         hide_index=True, width="stretch")

    grid = artifact(r.run_dir, "grid")
    sens = artifact(r.run_dir, "sensitivity")
    if grid is not None or sens is not None:
        st.subheader("Parameter sensitivity")
        st.caption("Fixed params over the whole OOS span — a stability diagnostic, not an OOS result.")
        c1, c2 = st.columns(2)
        gp = [k for k in (sm.get("params", {}).get("grid") or {})]
        if grid is not None and len(gp) >= 2:
            with c1:
                pv = grid.pivot_table(index=gp[0], columns=gp[1], values="sharpe")
                lim = float(np.nanmax(np.abs(pv.values))) or 1.0
                fig = go.Figure(go.Heatmap(z=pv.values, x=[str(x) for x in pv.columns], y=[str(y) for y in pv.index],
                                           colorscale=DIVERGING, zmin=-lim, zmax=lim, xgap=2, ygap=2,
                                           text=np.round(pv.values, 2), texttemplate="%{text}",
                                           hovertemplate=f"{gp[0]}=%{{y}}, {gp[1]}=%{{x}}<br>Sharpe %{{z:.2f}}<extra></extra>",
                                           colorbar=dict(title="Sharpe")))
                fig.update_xaxes(type="category")
                fig.update_yaxes(type="category")
                show(style(fig, title=f"Grid: {gp[0]} × {gp[1]}", xaxis_title=gp[1], yaxis_title=gp[0]))
        if sens is not None and len(sens):
            with c2:
                pv = sens.pivot_table(index="param", columns="step", values="sharpe")
                base = (sm.get("sensitivity") or {}).get("base_sharpe")
                lim = float(np.nanmax(np.abs(pv.values))) or 1.0
                fig = go.Figure(go.Heatmap(z=pv.values, x=[f"{c:+.0%}" for c in pv.columns], y=list(pv.index),
                                           colorscale=DIVERGING, zmin=-lim, zmax=lim, xgap=2, ygap=2,
                                           text=np.round(pv.values, 2), texttemplate="%{text}",
                                           hovertemplate="%{y} %{x}<br>Sharpe %{z:.2f}<extra></extra>",
                                           colorbar=dict(title="Sharpe")))
                fig.update_xaxes(type="category")
                isl = (sm.get("sensitivity") or {}).get("narrow_island")
                show(style(fig, title=f"One-at-a-time moves (base Sharpe {fmt2(base)}, narrow island: {isl})"))

    reg = table("run_regimes", rid)
    rnd = artifact(r.run_dir, "random_sharpes")
    c1, c2 = st.columns(2)
    if len(reg):
        with c1:
            st.subheader("Regime split (OOS trades)")
            st.dataframe(reg.drop(columns=["run_id"]), hide_index=True, width="stretch")
    if rnd is not None and len(rnd):
        with c2:
            st.subheader("vs random entry")
            re = sm.get("random_entry", {})
            pct = re.get("percentile", 95)
            fig = go.Figure(go.Histogram(x=rnd.sharpe, nbinsx=40, marker_color=C["s1"], opacity=0.85,
                                         name="random-entry seeds",
                                         hovertemplate="Sharpe %{x}<br>%{y} seeds<extra></extra>"))
            fig.add_vline(x=r.sharpe, line_color=C["s2"], line_width=2,
                          annotation_text=f"strategy {r.sharpe:.2f}", annotation_position="top")
            thr = re.get(f"random_sharpe_p{pct}")
            if thr is not None:
                fig.add_vline(x=thr, line_color=C["muted"], line_dash="dash", line_width=1,
                              annotation_text=f"p{pct} {thr:.2f}", annotation_position="bottom")
            show(style(fig, title=f"{re.get('seeds', len(rnd))} random-entry seeds, same exits",
                       xaxis_title="OOS Sharpe", showlegend=False, bargap=0.05))

    paper_section(r, rid, status)


def judgement_section(rid):
    j = db.judgement_for_run(CFG, rid)
    if j is None:
        st.caption("Not judged. `lab judge` decides promote or kill; `lab exam` runs the holdout exam of a promoted one.")
        return
    st.subheader("Judge")
    c = st.columns(4)
    c[0].metric("Decision", j["decision"].upper())
    c[1].metric("Deflated Sharpe", fmt2(j["dsr"]), help=f"N = {j['n_trials']} trials, SR0 = {fmt2(j['sr0'])} (daily)")
    c[2].metric("Beats parent", "—" if j["beats_parent_frac"] is None else f"{j['beats_parent_frac']:.0%}")
    c[3].metric("Holdout", j["holdout_status"], help=j["holdout_reason"])
    for x in j["reasons"]:
        st.markdown(f"- {x}")
    if j["holdout_run_id"]:
        h = db.run_row(CFG, j["holdout_run_id"])  # uncached: the exam run may be newer than the cache
        if h is None:
            st.caption(f"Holdout run {j['holdout_run_id']} not found in the store.")
        else:
            trades = int(h["trades"]) if h["trades"] is not None and pd.notna(h["trades"]) else 0
            st.caption(f"Holdout {h['date_start']} → {h['date_end']}: Sharpe {fmt2(h['sharpe'])}, "
                       f"return {fmt_pct(h['total_return'])}, {trades} trades (shown to you, never to the proposer)")


def paper_section(r, rid, status):
    st.subheader("Paper")
    runs = all_runs()
    papers = runs[(runs.validation_mode == "paper") & (runs.parent_run_id == rid)]
    min_days = CFG["paper"]["min_paper_days"]
    if r.validation_mode == "walkforward":
        if status == "paper":
            st.info("Approved for paper. Start it with `lab paper " + rid + "`.")
        elif r.verdict == "candidate" and status != "retired":
            j = db.judgement_for_run(CFG, rid)
            if j is None or j["holdout_status"] != "passed":
                held = "not judged" if j is None else f"{j['holdout_status']} ({j['holdout_reason']})"
                st.caption(f"Paper approval needs a passed holdout exam. Holdout: {held}.")
            else:
                with st.form(f"approve_{rid}"):
                    who = st.text_input("Approver")
                    note = st.text_input("Note")
                    if st.form_submit_button("Approve for paper trading"):
                        if not who.strip():
                            st.error("Approver name required.")
                        else:
                            try:
                                db.approve_paper(CFG, rid, who.strip(), note)
                            except PermissionError as e:
                                st.error(str(e))
                            else:
                                st.cache_data.clear()
                                st.rerun()
        else:
            st.caption(f"Paper approval is only offered for walk-forward candidates. This run is '{r.verdict}'.")
    if papers.empty:
        st.caption("No paper runs from this run.")
        return
    for _, p in papers.iterrows():
        started = pd.Timestamp(p.paper_started_at)
        started = started if started.tzinfo else started.tz_localize("UTC")
        days = (datetime.now(timezone.utc) - started).total_seconds() / 86400
        left = max(0.0, min_days - days)
        ch = p.paper_regime_changes
        c = st.columns(4)
        c[0].metric("Paper days", f"{days:.1f}", help=f"started {started:%Y-%m-%d %H:%M} UTC")
        c[1].metric("Countdown", "done" if left == 0 else f"{left:.1f} days left")
        c[2].metric("Regime changes", "not checked" if pd.isna(ch) else int(ch),
                    help=f"persisted ≥{CFG['paper']['regime_change_min_days']}d, min across pairs")
        ok_days, ok_reg = left == 0, (not pd.isna(ch) and ch >= 1)
        c[3].metric("Paper-complete eligible", "yes" if (ok_days and ok_reg) else "no")
        st.caption(f"{'✅' if ok_days else '⏳'} ≥ {min_days} days · {'✅' if ok_reg else '⏳'} ≥ 1 regime change "
                   f"(both required). Refresh regime count: `lab paper-refresh {p.run_id}`")
        try:
            from perp_lab.paper.tracking import replay_divergence
            from perp_lab.plugins import PluginSource
            dv = replay_divergence(CFG, Path(p.run_dir), PluginSource.read(p.strategy), p.params or {}, p.timeframe)
        except Exception as e:  # shown, not swallowed
            st.warning(f"divergence unavailable: {e}")
            dv = pd.DataFrame()
        if len(dv):
            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure()
                fig.add_scatter(x=dv.index, y=dv.paper, name="paper", line=dict(color=C["s1"], width=2))
                fig.add_scatter(x=dv.index, y=dv.replay, name="backtest replay", line=dict(color=C["s2"], width=2))
                show(style(fig, title="Cumulative P&L: paper vs backtest on the same bars",
                           yaxis_tickformat=".1%", hovermode="x unified"))
            with c2:
                fig = go.Figure(go.Scatter(x=dv.index, y=dv.divergence, line=dict(color=C["s1"], width=2),
                                           name="paper − replay"))
                show(style(fig, title="Divergence (paper − replay)", yaxis_tickformat=".2%",
                           hovermode="x unified", showlegend=False))
        st.markdown(f"**Paper trades on the chart** · run `{p.run_id}`")
        trades_chart_section(p, p.run_id)


def page_pairs():
    st.header("Pairs")
    st.caption("OOS Sharpe of each strategy's latest walk-forward run, per pair. A pair where most strategies "
               "'work' is more likely regime luck than edge.")
    wf = latest_wf(all_runs())
    if wf.empty:
        st.info("No walk-forward runs yet.")
        return
    sy = db.read_table(CFG, db.run_symbols)
    sy = sy[sy.run_id.isin(wf.run_id)]
    lab = dict(zip(wf.run_id, wf.strategy + " @ " + wf.timeframe))
    sy["strat"] = sy.run_id.map(lab)
    pv = sy.pivot_table(index="strat", columns="symbol", values="sharpe")
    lim = float(np.nanmax(np.abs(pv.values))) or 1.0
    fig = go.Figure(go.Heatmap(z=pv.values, x=list(pv.columns), y=list(pv.index), colorscale=DIVERGING,
                               zmin=-lim, zmax=lim, xgap=2, ygap=2, text=np.round(pv.values, 2),
                               texttemplate="%{text}", colorbar=dict(title="OOS Sharpe"),
                               hovertemplate="%{y}<br>%{x}: Sharpe %{z:.2f}<extra></extra>"))
    show(style(fig, h=120 + 44 * len(pv), title="Strategy × pair"))
    # one vote per lineage (a parent and its children are one idea, not independent evidence)
    lin = dict(zip(wf.strategy + " @ " + wf.timeframe, wf.lineage_id))
    votes = pv.groupby(pv.index.map(lambda x: lin.get(x, x))).mean()
    frac = (votes > 0).sum() / votes.notna().sum()
    out = pd.DataFrame({"lineages positive": (votes > 0).sum(), "lineages": votes.notna().sum(),
                        "share positive": frac, "mean Sharpe": votes.mean()})
    out["flag"] = np.where((out["lineages"] >= 3) & (out["share positive"] >= 2 / 3),
                           "⚠ everything works here — suspect regime luck", "")
    st.dataframe(out.sort_values("share positive", ascending=False)
                 .style.format({"share positive": "{:.0%}", "mean Sharpe": "{:.2f}"}),
                 width="stretch")


def page_lineage():
    st.header("Lineage")
    runs = all_runs()
    e = edges()
    lins = runs.groupby("lineage_id").size()
    lins = lins[lins.index.isin(set(runs.loc[runs.run_id.isin(set(e.parent_run_id) | set(e.child_run_id)),
                                             "lineage_id"]))] if len(e) else lins.iloc[0:0]
    if lins.empty:
        st.info("No lineage edges yet. Create one with `lab link PARENT CHILD --decision ... --reason ...`; "
                "Phase 3's judge writes them automatically.")
        return
    rmap = runs.set_index("run_id")
    pick = st.selectbox("Lineage (root run)", list(lins.index),
                        format_func=lambda x: f"{rmap.loc[x, 'strategy']} @ {rmap.loc[x, 'timeframe']} · {x}"
                        if x in rmap.index else x)
    members = runs[runs.lineage_id == pick]
    ee = e[e.parent_run_id.isin(members.run_id) | e.child_run_id.isin(members.run_id)]
    colors = {"promote": C["s1"], "kill": C["neg"], "manual": C["muted"]}
    dot = ["digraph G {", 'rankdir=TB; bgcolor="transparent";',
           'node [shape=box, style="rounded", fontname="sans-serif", fontsize=11];',
           'edge [fontname="sans-serif", fontsize=10];']
    for _, m in members.iterrows():
        lbl = (f"{m.strategy} @ {m.timeframe}\\n{m.run_id}\\nOOS Sharpe {fmt2(m.sharpe)} · "
               f"{m.validation_mode} · {m.verdict or ''}")
        dot.append(f'"{m.run_id}" [label="{lbl}"];')
    for _, x in ee.iterrows():
        reason = str(x.reason).replace('"', "'")
        reason = reason if len(reason) <= 60 else reason[:57] + "..."
        dot.append(f'"{x.parent_run_id}" -> "{x.child_run_id}" [label="{x.decision}: {reason}", '
                   f'color="{colors.get(x.decision, C["muted"])}", fontcolor="{colors.get(x.decision, C["muted"])}"];')
    dot.append("}")
    st.graphviz_chart("\n".join(dot), width="content")
    st.dataframe(ee[["parent_run_id", "child_run_id", "decision", "reason", "decided_by", "created_at"]],
                 hide_index=True, width="stretch")
    st.dataframe(members[["run_id", "strategy", "timeframe", "validation_mode", "verdict", "sharpe", "max_dd",
                          "trades", "parent_run_id", "created_at"]], hide_index=True, width="stretch")


PAGES = {"Strategies": page_strategies, "Strategy detail": page_detail, "Pairs": page_pairs,
         "Lineage": page_lineage}
page = st.sidebar.radio("Perp Lab", list(PAGES), key="page")


def holdout_caption() -> str:
    from perp_lab.judge.budget import BudgetLogError, reads_used
    jc = CFG.get("judge") or {}
    opens = (CFG.holdout_start + pd.Timedelta(days=jc.get("min_holdout_days", 0))).date()
    try:
        used = f"{reads_used(CFG)}/{jc.get('holdout_max_reads')}"
    except BudgetLogError:
        used = "log not initialised (`lab holdout-init`)"
    return (f"holdout reads: {used}  \nearliest exam date: {opens} (funding data arrives a month "
            "at a time, so it can be a few days later)")


st.sidebar.caption(f"store: {CFG.get('store', {}).get('url')}  \nholdout from {CFG.holdout_start.date()} (locked)")
st.sidebar.caption(holdout_caption())
st.sidebar.caption(CFG.get("universe_rule") or "Universe rule not set in this config: survivor bias not stated.")
PAGES[page]()
