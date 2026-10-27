// Pure geometry for the trade chart's primitives: no DOM, no chart library, so `node --test` covers it.
//
// Times are epoch ms (the payload's trade times), prices are prices. `x(t)` and `y(price)` map them to CSS pixels
// in the pane and return null when they can't. `win` is the payload's window [first bar, last bar + tf): nothing
// is drawn outside it, as nothing outside it was loaded.

export const BOX_ALPHA = 0.18;        // box fill
export const EDGE_ALPHA = 0.6;        // box outline
const SLACK = 3;                      // px of slack around a box for hover and click

/** The fractional logical index of time `t` among ascending bar open times `times`: a bar's own time maps to its
 *  index (its candle's centre); a time between two bars maps between them; before the first or past the last bar
 *  it goes on at `step` (the bar interval) per index. Null without bars. */
export function timeToLogical(times, t, step) {
  const n = times.length;
  if (!n) return null;
  if (t <= times[0]) return (t - times[0]) / step;
  if (t >= times[n - 1]) return n - 1 + (t - times[n - 1]) / step;
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= t) lo = mid;
    else hi = mid;
  }
  return lo + (t - times[lo]) / (times[hi] - times[lo]);
}

/** [t0, t1] cut to the window, or null when nothing of it is inside. */
export function clipSpan(t0, t1, win) {
  const a = Math.max(t0, win[0]);
  const b = Math.min(t1, win[1]);
  return b > a ? [a, b] : null;
}

/** A box from time t0 to t1 between prices p0 (the entry side) and p1, in pixels, cut to the window. */
export function rect(t0, t1, p0, p1, x, y, win) {
  const span = clipSpan(t0, t1, win);
  if (!span) return null;
  const r = { x0: x(span[0]), x1: x(span[1]), y0: y(p0), y1: y(p1) };
  return Object.values(r).some((v) => v == null || !Number.isFinite(v)) ? null : r;
}

/** Boxes [t0, t1, entry, level] (as `trade_boxes` gives them) in pixels; any wholly outside the window are left out. */
export function boxRects(boxes, x, y, win) {
  return boxes.map(([t0, t1, p0, p1]) => rect(t0, t1, p0, p1, x, y, win)).filter(Boolean);
}

/** One trade's shapes in pixels: the reward and risk boxes, the no-target shading (entry → exit price over the
 *  trade, `gain` when the move went the trade's way), and the bold entry line as a flat box. */
export function tradeShapes(trade, x, y, win) {
  const b = trade.boxes;
  const [t0, t1] = b.span;
  const entry = b.entry[1];
  let result = null;
  let gain = null;
  if (b.no_target) {
    const exit = b.exit[1];
    result = rect(t0, t1, entry, exit, x, y, win);
    gain = trade.side * (exit - entry) > 0;
  }
  return {
    reward: boxRects(b.reward, x, y, win),
    risk: boxRects(b.risk, x, y, win),
    result,
    gain,
    entry: rect(t0, t1, entry, entry, x, y, win),
  };
}

/** Level steps [t0, t1, price] as polylines: a level runs to the next one's time with a vertical join; a gap
 *  (a missing level) starts a new line. A level that held for an instant is kept, so its join is drawn. */
export function stepLines(segs, x, y, win) {
  const lines = [];
  let cur = null;
  let last = null;
  for (const [t0, t1, v] of segs) {
    const a = Math.max(t0, win[0]);
    const b = Math.min(t1, win[1]);
    if (b < a || (b === a && t1 > t0)) {
      cur = null;
      continue;
    }
    const px0 = x(a);
    const px1 = x(b);
    const py = y(v);
    if (px0 == null || px1 == null || py == null) {
      cur = null;
      continue;
    }
    if (cur && last === a) cur.push([px0, py], [px1, py]);
    else lines.push((cur = [[px0, py], [px1, py]]));
    last = b;
  }
  return lines;
}

/** The tags whose time is on screen (0 ≤ x ≤ width), each with its x. */
export function visibleTags(tags, x, width) {
  const out = [];
  for (const tag of tags) {
    const px = x(tag.time);
    if (px != null && px >= 0 && px <= width) out.push({ ...tag, x: px });
  }
  return out;
}

/** Time tags laid out at the foot of the pane so none overprints another. Each tag {x, w, side} sits beside its
 *  time's line (an entry tag ends at it, an exit tag starts at it), kept inside [0, width], and takes the lowest
 *  row (0 at the foot) where it clears the tags already there, left to right. Returns the tags with `left` and
 *  `row`. */
export function stackTags(tags, width, gap = 2) {
  const rows = [];
  return [...tags].sort((a, b) => a.x - b.x).map((t) => {
    const left = Math.max(0, Math.min(t.side === "entry" ? t.x - t.w : t.x, width - t.w));
    const right = left + t.w;
    let row = 0;
    while ((rows[row] || []).some(([l, r]) => left < r + gap && right > l - gap)) row += 1;
    (rows[row] = rows[row] || []).push([left, right]);
    return { ...t, left, row };
  });
}

/** Where a trade's R:R label goes: centred on the part of the trade in view, at the reward box's far edge and
 *  hanging into it (`below` the edge for a long, above it for a short); with no target, on the reward side of the
 *  entry line, at the exit if the price went further. Null when the trade is off screen. */
export function labelSpot(trade, x, y, win, width) {
  const b = trade.boxes;
  const span = clipSpan(b.span[0], b.span[1], win);
  if (!span) return null;
  const a = x(span[0]);
  const z = x(span[1]);
  if (a == null || z == null) return null;
  const x0 = Math.max(a, 0);
  const x1 = Math.min(z, width);
  if (x1 < x0) return null;
  const long = trade.side > 0;
  const far = long ? Math.max : Math.min;
  const inward = b.reward.length > 0;
  const price = inward ? far(...b.reward.map((r) => r[3])) : far(b.entry[1], b.exit[1]);
  const py = y(price);
  if (py == null) return null;
  return { x: (x0 + x1) / 2, y: py, below: inward ? long : !long };
}

function inside(r, px, py, slack) {
  return (px >= Math.min(r.x0, r.x1) - slack && px <= Math.max(r.x0, r.x1) + slack &&
          py >= Math.min(r.y0, r.y1) - slack && py <= Math.max(r.y0, r.y1) + slack);
}

/** The id of the trade whose box is under (px, py), or null. `items` are [{id, rects}] in drawing order, so the
 *  last drawn (on top) wins. */
export function hitTrade(items, px, py, slack = SLACK) {
  for (let i = items.length - 1; i >= 0; i--) {
    if (items[i].rects.some((r) => inside(r, px, py, slack))) return items[i].id;
  }
  return null;
}

/** The logical range that shows the trade [l0, l1] with `pad` bars either side, or null when the `current` range
 *  already shows it whole. When the current zoom has room, it keeps the zoom and centres the trade. */
export function focusRange(l0, l1, current, pad) {
  if (current && l0 >= current.from && l1 <= current.to) return null;
  const width = current ? current.to - current.from : 0;
  if (width >= l1 - l0 + 2 * pad) {
    const mid = (l0 + l1) / 2;
    return { from: mid - width / 2, to: mid + width / 2 };
  }
  return { from: l0 - pad, to: l1 + pad };
}

/** The logical range that brings a trade's part in the window on screen with `pad` bars either side (as
 *  `focusRange`), or null when it is already on screen or not in the window. `logical(t)` maps a time to a
 *  logical index. */
export function focusTrade(trade, win, logical, current, pad) {
  const span = clipSpan(trade.boxes.span[0], trade.boxes.span[1], win);
  if (!span) return null;
  const l0 = logical(span[0]);
  const l1 = logical(span[1]);
  return l0 == null || l1 == null ? null : focusRange(l0, l1, current, pad);
}

/** Shaded spans (folds, warmup) on screen, each with its pixel edges and `lx`, where its label starts. */
export function spanRects(spans, x, width) {
  const out = [];
  for (const s of spans) {
    const x0 = x(s.from);
    const x1 = x(s.to);
    if (x0 == null || x1 == null || x1 <= 0 || x0 >= width) continue;
    out.push({ ...s, x0, x1, lx: Math.max(x0, 0) + 4 });
  }
  return out;
}

/** '#rrggbb' at an opacity, as rgba; any other colour (already rgba) as it is. */
export function withAlpha(color, alpha) {
  const m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(color);
  if (!m) return color;
  return `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${alpha})`;
}
