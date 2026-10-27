// The trade chart component: Streamlit's component protocol (by hand, over postMessage) and the chart on
// TradingView Lightweight Charts (the vendored standalone build, `window.LightweightCharts`).
//
// Each render brings a payload from dashboard/tv_chart.py. The candles are set only when they change, so the
// user's pan and zoom survive the page's reruns; a newly focused trade is scrolled into view only if it isn't on
// screen. Navigation is the library's own: drag to pan, wheel or pinch to zoom, drag an axis to stretch it, and the
// price axis fits the candles in view.

import { focusTrade } from "./geometry.mjs";
import { Scale, ShadeLayer, TradeLayer } from "./primitives.js";

const LWC = window.LightweightCharts;

// ---------------------------------------------------------------- Streamlit's component protocol (API v1)

function post(type, data) {
  window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, "*");
}

const Streamlit = {
  ready: () => post("streamlit:componentReady", { apiVersion: 1 }),
  setFrameHeight: (height) => post("streamlit:setFrameHeight", { height }),
  setComponentValue: (value) => post("streamlit:setComponentValue", { value, dataType: "json" }),
};

// ---------------------------------------------------------------- state

const $ = (id) => document.getElementById(id);
const el = { title: $("title"), ohlc: $("ohlc"), key: $("key"), signal: $("signal"), goto: $("goto"),
             fit: $("fit"), chart: $("chart"), tip: $("tip") };

let chart = null;
let series = null;
let scale = null;
let shade = null;
let trades = null;
let markers = null;
let lines = [];             // the price lines on the axis
let payload = null;         // the payload drawn
let candlesKey = null;      // which candles are on the chart
let focused = null;         // the trade last focused
let height = 0;
let font = "system-ui, sans-serif";

function build() {
  chart = LWC.createChart(el.chart, {
    autoSize: true,
    layout: { fontFamily: font, fontSize: 11 },
    timeScale: { timeVisible: true, secondsVisible: false, minBarSpacing: 0.01, rightOffset: 4 },
    crosshair: { mode: LWC.CrosshairMode.Normal },
  });
  series = chart.addSeries(LWC.CandlestickSeries, { priceLineVisible: false, lastValueVisible: false });
  scale = new Scale(chart, series);
  shade = new ShadeLayer(scale);
  trades = new TradeLayer(scale);
  trades.keepClear = logoBox;
  series.attachPrimitive(shade);
  series.attachPrimitive(trades);
  markers = LWC.createSeriesMarkers(series, []);
  chart.subscribeCrosshairMove(onMove);
  chart.subscribeClick(onClick);
  chart.subscribeDblClick(fit);
  el.fit.addEventListener("click", fit);
  el.goto.addEventListener("click", () => goTo(payload, true));
  el.signal.addEventListener("change", () => payload && drawTrades(payload));
}

/** TradingView's attribution logo (kept on, as its licence asks) in pane pixels, so time tags stay clear of it. */
function logoBox() {
  const logo = el.chart.querySelector("#tv-attr-logo");
  if (!logo || !logo.parentElement) return null;
  const r = logo.getBoundingClientRect();
  const pane = logo.parentElement.getBoundingClientRect();
  if (!r.width) return null;
  return { left: r.left - pane.left - 2, right: r.right - pane.left + 2, top: r.top - pane.top - 2,
           bottom: r.bottom - pane.top + 2 };
}

// ---------------------------------------------------------------- drawing a payload

function render(args, theme) {
  const p = args.payload;
  if (theme && theme.font) font = theme.font;
  const h = args.height || 560;
  if (h !== height) {
    height = h;
    document.body.style.height = `${h}px`;
    Streamlit.setFrameHeight(h);
  }
  if (!chart) build();
  paint(p.palette);
  el.title.textContent = p.title || "";
  exitKey(p.exits);

  const n = p.candles.length;
  const key = n ? `${p.shown}|${n}|${p.candles[0].time}|${p.candles[n - 1].time}` : "";
  const fresh = key !== candlesKey;
  if (fresh) {
    series.setData(p.candles);
    scale.setBars(p.candles.map((c) => c.time * 1000), p.bar_ms);
    candlesKey = key;
  }
  series.applyOptions({ priceFormat: p.price_format });
  payload = p;
  shade.font = font;
  shade.setData({ spans: p.folds, palette: p.palette });
  drawTrades(p);
  markers.setMarkers(p.markers.map((m) => ({
    time: m.time, position: "atPriceMiddle", price: m.price, shape: "circle", color: m.color,
    id: `${m.kind}:${m.id}`,
  })));
  priceLines(p);
  readout(null);
  view(p, fresh);
}

function paint(p) {
  const root = document.documentElement.style;
  for (const k of ["bg", "ink", "muted", "grid"]) root.setProperty(`--${k}`, p[k]);
  chart.applyOptions({
    layout: { background: { type: LWC.ColorType.Solid, color: p.bg }, textColor: p.ink, fontFamily: font },
    grid: { vertLines: { color: p.grid }, horzLines: { color: p.grid } },
    rightPriceScale: { borderColor: p.grid },
    timeScale: { borderColor: p.grid },
    crosshair: { vertLine: { labelBackgroundColor: p.muted }, horzLine: { labelBackgroundColor: p.muted } },
  });
  // neutral candles, hollow up and filled down, so the trades stand out
  series.applyOptions({
    upColor: "rgba(0,0,0,0)", borderUpColor: p.candle, wickUpColor: p.candle,
    downColor: p.candle, borderDownColor: p.candle, wickDownColor: p.candle,
  });
}

function drawTrades(p) {
  trades.font = font;
  trades.setData({ trades: p.trades, window: p.window, palette: p.palette, signal: el.signal.checked });
}

/** Entry, TP and SL of the tagged trades, as price lines labelled on the price axis. */
function priceLines(p) {
  for (const line of lines) series.removePriceLine(line);
  lines = [];
  for (const tr of p.trades) {
    for (const pl of tr.price_lines) {
      lines.push(series.createPriceLine({
        price: pl.price, color: pl.color, title: pl.title, axisLabelVisible: true,
        lineStyle: LWC.LineStyle.Dashed, lineWidth: 1,
      }));
    }
  }
}

function focusedTrade(p) {
  return p && p.focus != null ? p.trades.find((t) => t.id === p.focus) || null : null;
}

/** New candles: show the focused trade with context, else the whole range. A new focus on the same candles:
 *  scroll to it only if it isn't whole on screen. Anything else leaves the user's view as it is. */
function view(p, fresh) {
  const changed = p.focus !== focused;
  focused = p.focus;
  const tr = focusedTrade(p);
  el.goto.hidden = !tr;
  if (!fresh && !(tr && changed)) return;
  requestAnimationFrame(() => {             // once the chart has taken the new data
    if (tr) goTo(p, fresh);
    else fit();
  });
}

/** Bring the focused trade on screen with context bars either side; `fresh` ignores the current view. Also the
 *  "focused trade" button's action, as picking the same trade again in Jump to trade changes nothing. */
function goTo(p, fresh) {
  const tr = focusedTrade(p);
  if (!tr) return;
  const current = fresh ? null : chart.timeScale().getVisibleLogicalRange();
  const range = focusTrade(tr, p.window, (t) => scale.logical(t), current, p.context);
  if (range) chart.timeScale().setVisibleLogicalRange(range);
  chart.priceScale("right").applyOptions({ autoScale: true });
}

function fit() {
  chart.timeScale().fitContent();
  chart.priceScale("right").applyOptions({ autoScale: true });
}

// ---------------------------------------------------------------- the bar above the chart

function exitKey(exits) {
  const items = Object.entries(exits).map(([reason, color]) => {
    const s = document.createElement("span");
    s.style.setProperty("--c", color);
    s.textContent = reason;
    return s;
  });
  el.key.replaceChildren(...(items.length ? [document.createTextNode("exit:"), ...items] : []));
  el.key.title = "exit dots, coloured by exit reason";
}

/** The bar under the crosshair (else the last one): its OHLC. Its time is on the time axis's crosshair label. */
function readout(param) {
  let bar = param && param.seriesData ? param.seriesData.get(series) : null;
  if (!bar && payload && payload.candles.length) bar = payload.candles[payload.candles.length - 1];
  if (!bar) {
    el.ohlc.replaceChildren();
    return;
  }
  const f = (v) => series.priceFormatter().format(v);
  const parts = [];
  for (const [k, v] of [["O", bar.open], ["H", bar.high], ["L", bar.low], ["C", bar.close]]) {
    const b = document.createElement("b");
    b.textContent = f(v);
    parts.push(document.createTextNode(`${parts.length ? "  " : ""}${k} `), b);
  }
  el.ohlc.replaceChildren(...parts);
  el.ohlc.title = `${new Date(bar.time * 1000).toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

// ---------------------------------------------------------------- hover and click

function onMove(param) {
  readout(param);
  if (!param.point || !payload) {
    el.tip.hidden = true;
    return;
  }
  const { x, y } = param.point;
  const fold = shade.labelAt(x, y);
  if (fold) {
    tip(fold.split("\n"), x, y);
    return;
  }
  const id = trades.tradeAt(x, y);
  const tr = id == null ? null : payload.trades.find((t) => t.id === id);
  if (tr) tip(tr.tooltip, x, y);
  else el.tip.hidden = true;
}

/** The floating panel by the cursor (pane coordinates x, y), kept inside the frame. */
function tip(textLines, x, y) {
  el.tip.replaceChildren(...textLines.map((t) => {
    const d = document.createElement("div");
    d.textContent = t;
    return d;
  }));
  el.tip.hidden = false;
  el.tip.style.left = "0px";                // measure at full width (up to its max-width), then place it
  el.tip.style.top = "0px";
  const box = el.chart.getBoundingClientRect();
  const w = el.tip.offsetWidth;
  const h = el.tip.offsetHeight;
  let left = box.left + x + 14;
  let top = box.top + y + 14;
  if (left + w > window.innerWidth - 4) left = Math.max(4, box.left + x - 14 - w);
  if (top + h > window.innerHeight - 4) top = Math.max(4, box.top + y - 14 - h);
  el.tip.style.left = `${left}px`;
  el.tip.style.top = `${top}px`;
}

/** A click inside a trade's box focuses it: the page sets Jump to trade from the value. `at` makes every click a
 *  new value, so clicking the same box again still reaches the page. */
function onClick(param) {
  if (!param.point) return;
  const id = trades.tradeAt(param.point.x, param.point.y);
  if (id != null) Streamlit.setComponentValue({ clicked: id, at: Date.now() });
}

window.addEventListener("message", (event) => {
  if (event.source !== window.parent || !event.data || event.data.type !== "streamlit:render") return;
  render(event.data.args, event.data.theme);
});
Streamlit.ready();
