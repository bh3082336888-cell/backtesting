// Series primitives for the trade chart, drawn on the candle series' pane in CSS pixels.
//
// `Layer` is the base: once attached to the series it draws its `data` in two passes, z = "bottom" (under the
// candles: shading and boxes) and z = "normal" (over them: lines, labels, tags), with the shared `Scale` mapping
// times (epoch ms) and prices to pixels. A layer asks for a redraw when its data changes. The trade chart's own
// layers are `ShadeLayer` (folds, warmup) and `TradeLayer` (positions).

import {
  BOX_ALPHA, EDGE_ALPHA, hitTrade, labelSpot, slotsUnder, spanRects, stackTags, stepLines, timeToLogical,
  tradeShapes, visibleTags, withAlpha,
} from "./geometry.mjs";

const TAG_H = 16;            // time tag height, px
const TAG_STEP = TAG_H + 2;  // one row of time tags to the next
const FONT = "system-ui, sans-serif";

/** Times (epoch ms) and prices to CSS pixels on the chart's main pane. Times between bars land between them,
 *  so a trade keeps its exact times on coarser candles. */
export class Scale {
  constructor(chart, series) {
    this.chart = chart;
    this.series = series;
    this.times = [];
    this.step = 1;
    this.x = (t) => {
      const l = timeToLogical(this.times, t, this.step);
      return l == null ? null : this.chart.timeScale().logicalToCoordinate(l);
    };
    this.y = (price) => this.series.priceToCoordinate(price);
  }

  /** The drawn candles' open times (ms) and the bar interval (ms). */
  setBars(times, step) {
    this.times = times;
    this.step = step;
  }

  logical(t) {
    return timeToLogical(this.times, t, this.step);
  }
}

class PaneView {
  constructor(layer, z) {
    this.z = z;
    this.r = { draw: (target) => target.useMediaCoordinateSpace(({ context, mediaSize }) => {
      if (layer.data) layer.draw(context, mediaSize, z);
    }) };
  }

  zOrder() {
    return this.z;
  }

  renderer() {
    return this.r;
  }
}

export class Layer {
  constructor(scale) {
    this.scale = scale;
    this.data = null;
    this.font = FONT;
    this.views = [new PaneView(this, "bottom"), new PaneView(this, "normal")];
    this.request = null;
  }

  attached({ requestUpdate }) {
    this.request = requestUpdate;
  }

  detached() {
    this.request = null;
  }

  paneViews() {
    return this.views;
  }

  updateAllViews() {}

  setData(data) {
    this.data = data;
    this.update();
  }

  update() {
    if (this.request) this.request();
  }

  /** Draw `this.data` on the pane: `size` is {width, height}, `z` the pass ("bottom" or "normal"). */
  draw(ctx, size, z) {} // eslint-disable-line no-unused-vars
}

// ---------------------------------------------------------------- drawing helpers

function fillBox(ctx, r, color) {
  const x = Math.min(r.x0, r.x1);
  const y = Math.min(r.y0, r.y1);
  const w = Math.max(Math.abs(r.x1 - r.x0), 1);
  const h = Math.max(Math.abs(r.y1 - r.y0), 1);
  ctx.fillStyle = withAlpha(color, BOX_ALPHA);
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = withAlpha(color, EDGE_ALPHA);
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
}

function polyline(ctx, pts, color, width, dash = []) {
  if (pts.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (const [px, py] of pts.slice(1)) ctx.lineTo(px, py);
  ctx.stroke();
  ctx.setLineDash([]);
}

/** A filled tag with white text, its left edge at `left`. */
function tag(ctx, text, left, y, w, color, font) {
  ctx.font = `11px ${font}`;
  ctx.fillStyle = color;
  ctx.fillRect(left, y, w, TAG_H);
  ctx.fillStyle = "#ffffff";
  ctx.textBaseline = "middle";
  ctx.textAlign = "left";
  ctx.fillText(text, left + 4, y + TAG_H / 2 + 0.5);
}

/** The R:R label: a plate holding two lines, the R:R in bold and the result in `tone`, centred on x but kept
 *  inside the pane's `width`; it hangs from y (`below`) or stands on it. */
function plate(ctx, lines, tone, x, y, below, width, p, font) {
  const fonts = [`600 11px ${font}`, `11px ${font}`];
  const widths = lines.map((text, i) => {
    ctx.font = fonts[i];
    return ctx.measureText(text).width;
  });
  const w = Math.max(...widths) + 10;
  const h = lines.length * 14 + 6;
  const left = Math.max(0, Math.min(x - w / 2, width - w));
  const top = below ? y + 3 : y - 3 - h;
  ctx.fillStyle = withAlpha(p.bg, 0.85);
  ctx.fillRect(left, top, w, h);
  ctx.strokeStyle = p.muted;
  ctx.lineWidth = 1;
  ctx.strokeRect(left + 0.5, top + 0.5, w - 1, h - 1);
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  lines.forEach((text, i) => {
    ctx.font = fonts[i];
    ctx.fillStyle = i === 0 ? p.ink : tone;
    ctx.fillText(text, left + 5, top + 4 + i * 14);
  });
}

// ---------------------------------------------------------------- layers

/** Shaded spans: walk-forward test folds (every other one shaded, each labelled) and a paper run's warmup.
 *  data: {spans: [{from, to, fill, label, hover}], palette}. */
export class ShadeLayer extends Layer {
  constructor(scale) {
    super(scale);
    this.labels = [];
  }

  draw(ctx, size, z) {
    const { spans, palette: p } = this.data;
    const shown = spanRects(spans, this.scale.x, size.width);
    if (z === "bottom") {
      for (const s of shown) {
        if (!s.fill) continue;
        ctx.fillStyle = s.fill;
        ctx.fillRect(s.x0, 0, s.x1 - s.x0, size.height);
      }
      return;
    }
    ctx.font = `10px ${this.font}`;
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillStyle = p.muted;
    const labels = [];
    let free = -Infinity;                   // where the next label may start without overprinting the last
    for (const s of shown) {
      const lx = Math.max(s.lx, free);
      const w = ctx.measureText(s.label).width;
      if (lx + w > Math.min(s.x1, size.width)) continue;     // no room in its band
      ctx.fillText(s.label, lx, 4);
      labels.push({ x0: lx, x1: lx + w, hover: s.hover });
      free = lx + w + 8;
    }
    this.labels = labels;
  }

  /** The hover text of the label under (px, py), or null. */
  labelAt(px, py) {
    if (py > 18) return null;
    const hit = this.labels.find((l) => px >= l.x0 - 2 && px <= l.x1 + 2);
    return hit ? hit.hover : null;
  }
}

/** The trades as TradingView's position tool draws them. data: {trades, window, palette, signal (bool)}.
 *  Bottom pass: the reward, risk and no-target boxes. Normal pass: the stop and target step lines, the bold entry
 *  line, the signal bar's close joined to the fill, and for tagged trades the R:R label, the breakeven note and
 *  the time tags with their dotted verticals. The dots are series markers; the price tags are price lines. */
export class TradeLayer extends Layer {
  constructor(scale) {
    super(scale);
    this.hits = [];
    this.keepClear = () => null;    // the pane box time tags must not cover (TradingView's logo), if any
  }

  draw(ctx, size, z) {
    const { trades, window: win, palette: p } = this.data;
    const { x, y } = this.scale;
    const shapes = trades.map((tr) => tradeShapes(tr, x, y, win));
    if (z === "bottom") {
      this.hits = trades.map((tr, i) => {
        const s = shapes[i];
        if (s.result) fillBox(ctx, s.result, s.gain ? p.pos : p.neg);
        for (const r of s.reward) fillBox(ctx, r, p.pos);
        for (const r of s.risk) fillBox(ctx, r, p.neg);
        return { id: tr.id, rects: [...s.reward, ...s.risk, ...(s.result ? [s.result] : []),
                                    ...(s.entry ? [s.entry] : [])] };
      });
      return;
    }
    trades.forEach((tr, i) => {
      for (const line of stepLines(tr.stops, x, y, win)) polyline(ctx, line, p.neg, 1, [2, 3]);
      for (const line of stepLines(tr.targets, x, y, win)) polyline(ctx, line, p.pos, 1, [2, 3]);
      const e = shapes[i].entry;
      if (e) polyline(ctx, [[e.x0, e.y0], [e.x1, e.y0]], p.s1, 2.5);
      if (this.data.signal && tr.signal) this.signal(ctx, tr, p);
    });
    const tagged = trades.filter((tr) => tr.tagged);
    for (const tr of tagged) {
      const spot = labelSpot(tr, x, y, win, size.width);
      if (spot) plate(ctx, [tr.rr, tr.result], tr.tone, spot.x, spot.y, spot.below, size.width, p, this.font);
      if (tr.be) {
        const bx = x(tr.be.time);
        const by = y(tr.be.price);
        if (bx != null && by != null) {
          ctx.font = `10px ${this.font}`;
          ctx.fillStyle = p.s2;
          ctx.textAlign = "left";
          ctx.textBaseline = tr.side > 0 ? "top" : "bottom";
          ctx.fillText(tr.be.text, bx + 3, by + (tr.side > 0 ? 3 : -3));
        }
      }
    }
    // time tags last, at the foot of the pane over a dotted vertical at their time; ones that would overprint
    // another stack upwards
    ctx.font = `11px ${this.font}`;
    const shown = tagged.flatMap((tr) => visibleTags(tr.time_tags, x, size.width))
      .map((t) => ({ ...t, w: ctx.measureText(t.text).width + 8 }));
    const foot = size.height - TAG_STEP;
    const logo = this.keepClear();
    const keepClear = logo ? slotsUnder(logo, foot, TAG_STEP, TAG_H) : [];
    for (const t of stackTags(shown, size.width, { keepClear })) {
      const top = foot - t.row * TAG_STEP;
      ctx.globalAlpha = 0.6;
      polyline(ctx, [[t.x, 0], [t.x, top]], t.color, 1, [1, 3]);
      ctx.globalAlpha = 1;
      tag(ctx, t.text, t.left, top, t.w, t.color, this.font);
    }
  }

  signal(ctx, tr, p) {
    const { x, y } = this.scale;
    const sx = x(tr.signal.time);
    const sy = y(tr.signal.price);
    const ex = x(tr.entry.time);
    const ey = y(tr.entry.price);
    if (sx == null || sy == null) return;
    if (ex != null && ey != null) polyline(ctx, [[sx, sy], [ex, ey]], p.ink, 1, [1, 2]);
    ctx.strokeStyle = p.ink;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(sx, sy, 4, 0, 2 * Math.PI);
    ctx.stroke();
  }

  /** The id of the trade whose box is under (px, py), as last drawn. */
  tradeAt(px, py) {
    return hitTrade(this.hits, px, py);
  }

  /** The library's hit test: a pointer over a box, to say it can be clicked. */
  hitTest(px, py) {
    const id = this.tradeAt(px, py);
    return id == null ? null : { externalId: `trade:${id}`, zOrder: "bottom", cursorStyle: "pointer" };
  }
}
