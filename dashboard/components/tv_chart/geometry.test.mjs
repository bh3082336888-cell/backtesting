// The trade chart's pure geometry: times and prices to pixels, with a given scale. `node --test` runs this.
import { test } from "node:test";
import assert from "node:assert/strict";

import {
  boxRects, focusRange, hitTrade, labelSpot, spanRects, stackTags, stepLines, timeToLogical, tradeShapes,
  visibleTags, withAlpha,
} from "./geometry.mjs";

const x = (t) => (t - 1000) / 10;      // 10 ms per pixel, t = 1000 at the left edge
const y = (p) => 500 - 2 * p;          // 2 px per price unit; up the price is up the screen
const ALL = [0, 1e9];                   // a window that holds everything

// A long from 1000 at 100: target 106, stop 97 moved to the entry (breakeven) at 2500, out at the target at 4000.
const LONG = {
  id: 7, side: 1,
  boxes: {
    reward: [[1000, 4000, 100, 106]], risk: [[1000, 2500, 100, 97]], entry: [1000, 100],
    exit: [4000, 106, "target"], be_time: 2500, no_target: false, span: [1000, 4000],
  },
};

test("box rectangles map time/price to pixels with a given scale", () => {
  assert.deepEqual(boxRects([[1000, 2000, 100, 110]], x, y, ALL), [{ x0: 0, x1: 100, y0: 300, y1: 280 }]);
  // a box that starts before the window is drawn from the window's start, and one wholly outside not at all
  assert.deepEqual(boxRects([[1000, 2000, 100, 110]], x, y, [1500, 1800]), [{ x0: 50, x1: 80, y0: 300, y1: 280 }]);
  assert.deepEqual(boxRects([[1000, 2000, 100, 110]], x, y, [3000, 4000]), []);
  // a time the chart can't place draws nothing rather than a box to nowhere
  assert.deepEqual(boxRects([[1000, 2000, 100, 110]], () => null, y, ALL), []);
});

test("risk box ends at the breakeven time", () => {
  const s = tradeShapes(LONG, x, y, ALL);
  assert.equal(s.risk.length, 1);
  assert.deepEqual(s.risk[0], { x0: 0, x1: x(2500), y0: y(100), y1: y(97) });
  assert.deepEqual(s.reward, [{ x0: 0, x1: x(4000), y0: y(100), y1: y(106) }]);   // the target box runs to the exit
  assert.deepEqual(s.entry, { x0: 0, x1: x(4000), y0: y(100), y1: y(100) });      // the bold entry line
  assert.equal(s.result, null);
});

test("time tags skip times outside the visible range", () => {
  const tags = [{ time: 500, text: "a" }, { time: 1500, text: "b" }, { time: 3500, text: "c" }];
  // x(500) = -50 and x(3500) = 250 are off a 200 px wide pane
  assert.deepEqual(visibleTags(tags, x, 200).map((t) => [t.text, t.x]), [["b", 50]]);
  assert.deepEqual(visibleTags(tags, () => null, 200), []);
});

test("time tags that would overprint stack in rows above the foot", () => {
  const tags = [
    { x: 100, w: 80, side: "entry", text: "13:55 entry" },     // 20..100
    { x: 110, w: 80, side: "exit", text: "14:05 exit" },       // 110..190: clear of the first
    { x: 150, w: 80, side: "entry", text: "14:30 entry" },     // 70..150: hits both, a row up
    { x: 150, w: 80, side: "exit", text: "14:30 exit" },       // 150..230: hits the second, a row up
    { x: 10, w: 80, side: "entry", text: "early entry" },      // pushed inside the pane: 0..80, hits the first
  ];
  assert.deepEqual(stackTags(tags, 400).map((t) => [t.text, t.left, t.row]), [
    ["early entry", 0, 0], ["13:55 entry", 20, 1], ["14:05 exit", 110, 0], ["14:30 entry", 70, 2],
    ["14:30 exit", 150, 1],
  ]);
  assert.equal(stackTags([{ x: 395, w: 80, side: "exit" }], 400)[0].left, 320);   // kept inside on the right
});

test("a time maps to a logical index between bars, and past the ends at the bar interval", () => {
  const times = [0, 10, 20, 40];                       // a missing bar between 20 and 40
  assert.equal(timeToLogical(times, 10, 10), 1);
  assert.equal(timeToLogical(times, 5, 10), 0.5);
  assert.equal(timeToLogical(times, 30, 10), 2.5);
  assert.equal(timeToLogical(times, -10, 10), -1);
  assert.equal(timeToLogical(times, 60, 10), 5);
  assert.equal(timeToLogical([100], 130, 10), 3);
  assert.equal(timeToLogical([], 5, 10), null);
});

test("a no-target trade shades entry to exit, green for a gain and red for a loss", () => {
  const free = (side, exit) => ({
    id: 1, side,
    boxes: { reward: [], risk: [[1000, 3000, 100, 97]], entry: [1000, 100], exit: [3000, exit, "signal"],
             be_time: null, no_target: true, span: [1000, 3000] },
  });
  const gain = tradeShapes(free(1, 104), x, y, ALL);
  assert.deepEqual(gain.result, { x0: 0, x1: 200, y0: y(100), y1: y(104) });
  assert.equal(gain.gain, true);
  assert.equal(gain.risk.length, 1);                     // the stop is still the risk
  assert.equal(tradeShapes(free(1, 98), x, y, ALL).gain, false);
  assert.equal(tradeShapes(free(-1, 98), x, y, ALL).gain, true);   // a short gains as the price falls
});

test("step lines join a level to the next and break where one is missing", () => {
  const segs = [[1000, 2000, 5], [2000, 3000, 6], [4000, 5000, 7]];
  assert.deepEqual(stepLines(segs, x, y, ALL), [
    [[0, 490], [100, 490], [100, 488], [200, 488]],
    [[300, 486], [400, 486]],
  ]);
  assert.deepEqual(stepLines(segs, x, y, [2500, 4500]), [[[150, 488], [200, 488]], [[300, 486], [350, 486]]]);
});

test("the label hangs inside the reward box from its far edge, mirrored for a short", () => {
  assert.deepEqual(labelSpot(LONG, x, y, ALL, 1000), { x: 150, y: y(106), below: true });
  const short = { ...LONG, side: -1, boxes: { ...LONG.boxes, reward: [[1000, 4000, 100, 94]] } };
  assert.deepEqual(labelSpot(short, x, y, ALL, 1000), { x: 150, y: y(94), below: false });
  // no target: on the reward side of the entry line, at the exit if it went further
  const free = { ...LONG, boxes: { ...LONG.boxes, reward: [], no_target: true, exit: [4000, 103, "signal"] } };
  assert.deepEqual(labelSpot(free, x, y, ALL, 1000), { x: 150, y: y(103), below: false });
  // centred on the part in view, and gone once the trade is off screen
  assert.equal(labelSpot(LONG, x, y, ALL, 100).x, 50);
  assert.equal(labelSpot(LONG, (t) => t - 9000, y, ALL, 1000), null);
});

test("a click inside a box finds its trade, the one drawn last on top", () => {
  const items = [
    { id: 1, rects: [{ x0: 0, x1: 100, y0: 300, y1: 200 }] },
    { id: 2, rects: [{ x0: 50, x1: 150, y0: 250, y1: 280 }] },
  ];
  assert.equal(hitTrade(items, 20, 220), 1);
  assert.equal(hitTrade(items, 60, 260), 2);
  assert.equal(hitTrade(items, 140, 100), null);
  assert.equal(hitTrade(items, 102, 220), 1);            // a few pixels of slack, for thin boxes
});

test("focus leaves a trade in view alone, else brings it in with context", () => {
  assert.equal(focusRange(10, 20, { from: 0, to: 100 }, 40), null);
  // off screen and the zoom has room: keep the zoom, centre the trade
  assert.deepEqual(focusRange(300, 310, { from: 0, to: 100 }, 40), { from: 255, to: 355 });
  // the zoom is too tight for the trade and its context (or there is no view yet): show both
  assert.deepEqual(focusRange(300, 310, { from: 0, to: 50 }, 40), { from: 260, to: 350 });
  assert.deepEqual(focusRange(300, 310, null, 40), { from: 260, to: 350 });
});

test("fold shading spans the pane, its label at the band's visible start", () => {
  const folds = [{ from: 500, to: 1500, fill: "#eee", label: "fold 0", hover: "h" },
                 { from: 1500, to: 9000, fill: null, label: "fold 1", hover: "h1" },
                 { from: 9000, to: 9500, fill: "#eee", label: "fold 2", hover: "h2" }];
  assert.deepEqual(spanRects(folds, x, 200).map((s) => [s.label, s.x0, s.x1, s.lx]), [
    ["fold 0", -50, 50, 4], ["fold 1", 50, 800, 54]]);        // fold 2 starts past the pane's right edge
});

test("colours take an opacity", () => {
  assert.equal(withAlpha("#0ca30c", 0.18), "rgba(12,163,12,0.18)");
  assert.equal(withAlpha("rgba(1,2,3,0.5)", 0.18), "rgba(1,2,3,0.5)");   // already has one: left alone
});
