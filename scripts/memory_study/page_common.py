# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Look and JS helpers shared by the Kimi K3 study pages (memory and roofline).

Each page template carries ``/*__CSS__*/`` inside its ``<style>`` block and
``/*__HELPERS__*/`` at the top of its script, after ``const DATA``; ``render``
fills both plus ``/*__DATA__*/null``. The helpers expect a ``<div id="tip">``.
"""

import json

PAGE_CSS = r""".viz-root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface-1: #fcfcfb;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --muted: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --border: rgba(11, 11, 11, 0.10);
  --wash: rgba(11, 11, 11, 0.04);
  --s-experts: #2a78d6;
  --s-attention: #eb6834;
  --s-other: #1baf7a;
  --s-acts: #eda100;
  --s-acts-full: #9a6400;
  --s-pergpu: #52514e;
  --s-weights: #e87ba4;
  --s-grads: #008300;
  --s-optstate: #4a3aa7;
  --s-tiny: #c3c2b7;
  --note-bg: #fdf1d6;
}
html, body { margin: 0; }
.viz-root {
  background: var(--page);
  color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  min-height: 100vh;
}
.wrap { max-width: 1240px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 24px; font-weight: 600; margin: 0 0 4px; }
h2 { font-size: 24px; font-weight: 700; margin: 48px 0 6px; }
.sub { color: var(--text-secondary); margin: 0 0 16px; max-width: 860px; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.topbar { display: flex; justify-content: space-between; align-items: start; gap: 16px; }
button, .seg label {
  font: inherit; color: var(--text-primary); background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px; cursor: pointer;
}
button:hover, .seg label:hover { background: var(--wash); }
:focus-visible { outline: 2px solid var(--s-experts); outline-offset: 2px; }

/* stat tiles */
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0 8px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.tile .label { color: var(--text-secondary); font-size: 13px; }
.tile .value { font-size: 26px; font-weight: 600; margin-top: 2px; }
.tile .note { color: var(--muted); font-size: 12px; }

/* legend */
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 8px 0; color: var(--text-secondary); font-size: 13px; }
.legend .item { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }

/* layer anatomy */
.layers { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }
.card h3 { margin: 0; font-size: 16px; font-weight: 600; }
.card .meta { color: var(--text-secondary); font-size: 13px; margin: 2px 0 10px; }
.card .ids { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
.strip { display: flex; gap: 2px; height: 14px; margin: 8px 0 4px; }
.strip > div { border-radius: 2px; min-width: 2px; }
.strip-labels { display: flex; justify-content: space-between; font-size: 12px; color: var(--text-secondary); margin-bottom: 12px; gap: 12px; }
.block { border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }
.block.glue { background: transparent; }
.block-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; margin-bottom: 4px; }
.block-title { font-weight: 600; display: inline-flex; align-items: center; gap: 6px; }
.block-total { color: var(--text-secondary); font-size: 12px; white-space: nowrap; }
/* context-length toggle, pinned to the top-right while scrolling */
.ctx-float {
  position: fixed; top: 12px; right: 16px; z-index: 20;
  display: flex; align-items: center; gap: 8px;
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 6px 8px 6px 12px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.10);
}
.ctx-float .ctx-label { color: var(--text-secondary); font-size: 13px; }
.ctx-float .ctx-batch { color: var(--text-secondary); font-size: 12px; line-height: 1.35; white-space: nowrap; }
.subhead { font-size: 17px; font-weight: 600; margin: 24px 0 4px; }
.subtitle { color: var(--muted); font-size: 12px; margin: 6px 0 2px; }
.flow { color: var(--muted); font-size: 12px; text-align: center; line-height: 1; padding: 3px 0; }
.flow::before { content: ""; display: block; width: 1px; height: 8px; background: var(--axis); margin: 0 auto 3px; }
.note-step {
  color: var(--text-secondary); font-size: 12px; background: var(--wash);
  border-radius: 6px; padding: 4px 8px; margin: 6px 0;
}
.row {
  display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr) 64px 112px;
  align-items: center; gap: 8px; padding: 2px 4px; border-radius: 4px; cursor: default;
}
.row:hover, .row:focus { background: var(--wash); outline: none; }
.row .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.row .formula { color: var(--text-secondary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.row .num { text-align: right; font-variant-numeric: tabular-nums; }
.row.tiny .name, .row.tiny .num { color: var(--muted); }
.minibar { height: 8px; background: transparent; position: relative; }
.minibar > div { height: 8px; border-radius: 0 3px 3px 0; }
.tiny-summary { color: var(--muted); font-size: 12px; padding: 2px 4px; }
.toolbar { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin: 4px 0 12px; }
.toolbar label.check { display: inline-flex; align-items: center; gap: 6px; color: var(--text-secondary); }

/* chart section */
.panel { background: var(--surface-1); border: 1px solid var(--border); border-radius: 12px; padding: 16px 16px 8px; }
.seg { display: inline-flex; gap: 0; }
.seg input { position: absolute; opacity: 0; pointer-events: none; }
.seg label { border-radius: 0; margin-left: -1px; }
.seg label:first-of-type { border-radius: 6px 0 0 6px; margin-left: 0; }
.seg label:last-of-type { border-radius: 0 6px 6px 0; }
.seg input:checked + label { background: var(--text-primary); color: var(--surface-1); border-color: var(--text-primary); }
.seg input:focus-visible + label { outline: 2px solid var(--s-experts); outline-offset: 2px; }
.chart-title { font-weight: 600; margin: 6px 0 2px; }
.chart-note { color: var(--muted); font-size: 12px; margin: 0 0 8px; }
#bigstrip, #bigstrip4 { display: flex; gap: 2px; height: 28px; margin: 6px 0 4px; }
#bigstrip > div, #bigstrip4 > div { border-radius: 3px; min-width: 2px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; overflow: visible; white-space: nowrap; }
svg text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
svg .tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
svg .rowlabel { fill: var(--text-primary); font-size: 13px; }
svg .vallabel { fill: var(--text-secondary); font-size: 12px; font-variant-numeric: tabular-nums;
  paint-order: stroke; stroke: var(--surface-1); stroke-width: 4px; stroke-linejoin: round; }
svg .hit { fill: transparent; }
svg g.bar:hover .hit, svg g.bar:focus .hit { fill: var(--wash); }
svg g.bar:focus { outline: none; }
/* linear horizontal bars (section 2) */
.hbars { display: grid; grid-template-columns: max-content 1fr; gap: 6px 12px; align-items: center; }
.hbars .lbl { font-size: 13px; white-space: nowrap; }
.hbars .lbl .muted { color: var(--muted); }
.hbars .track { display: flex; align-items: center; gap: 8px; min-width: 0; border-radius: 4px; padding: 2px 0; }
.hbars .track:hover, .hbars .track:focus { background: var(--wash); outline: none; }
.hbars .bar { height: 14px; border-radius: 0 4px 4px 0; min-width: 2px; flex: none; }
.hbars .val { color: var(--text-secondary); font-size: 12px; white-space: nowrap; font-variant-numeric: tabular-nums; }
.callout {
  margin: 12px 0 4px; padding: 10px 14px; border-radius: 8px; background: var(--wash);
  font-size: 14px;
}
.callout b { font-weight: 600; }
.callout.note {
  background: var(--note-bg); border-left: 4px solid var(--s-acts);
  margin-top: 16px; padding: 12px 16px;
}
.callout.note .note-label { font-weight: 700; margin-right: 6px; }
details { margin-top: 12px; }
summary { cursor: pointer; color: var(--text-secondary); }
table { border-collapse: collapse; margin-top: 8px; width: 100%; }
th, td { text-align: left; padding: 4px 10px; border-bottom: 1px solid var(--grid); font-size: 13px; }
td.r, th.r { text-align: right; font-variant-numeric: tabular-nums; }
/* section 6: cells that changed from the row above */
td.down { background: rgba(12, 163, 12, 0.10); }
td.up { background: rgba(208, 59, 59, 0.10); }
td .change { display: block; font-size: 11px; font-weight: 600; }
td .row-detail { display: block; font-size: 11px; color: var(--text-secondary); }
#zeroTable td.r, #zeroTable th.r, #zeroBufTable td.r, #zeroBufTable th.r,
#quantZeroTable td.r, #quantZeroTable th.r { white-space: nowrap; }
td.down .change { color: #006300; }
td.up .change { color: #b02a2a; }
.foot { color: var(--muted); font-size: 12px; margin-top: 32px; }

/* tooltip */
#tip {
  position: fixed; z-index: 10; pointer-events: none; display: none;
  background: var(--surface-1); color: var(--text-primary);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12); font-size: 12px; max-width: 360px;
}
#tip .v { font-size: 15px; font-weight: 600; }
#tip .k { color: var(--text-secondary); }
#tip .line { display: flex; align-items: center; gap: 6px; }
#tip .key { width: 12px; height: 2px; border-radius: 1px; display: inline-block; }
@media (max-width: 900px) {
  .layers { grid-template-columns: 1fr; }
  .tiles { grid-template-columns: 1fr 1fr; }
}
"""

PAGE_JS_HELPERS = r"""// ---------- formatting ----------
function fmt(n) {
  const units = [["T", 1e12], ["B", 1e9], ["M", 1e6], ["K", 1e3]];
  for (const [u, s] of units) if (n >= s) return (n / s).toPrecision(4) + u;
  return String(n);
}
const exact = n => n.toLocaleString("en-US");
function pct(x) {
  if (x === 0) return "0%";
  if (x < 0.01) return "<0.01%";
  return x.toFixed(2) + "%";
}
// Binary-unit label for 2^e: 2^20 -> "1Mi", 2^33 -> "8Gi".
function binaryUnit(e) {
  const [u, base] = e >= 40 ? ["Ti", 40] : e >= 30 ? ["Gi", 30] : e >= 20 ? ["Mi", 20] : ["Ki", 10];
  return 2 ** (e - base) + u;
}
const log2s = n => "2^" + Math.log2(n).toFixed(2);
const gib = n => (n / 2 ** 30).toFixed(n / 2 ** 30 < 10 ? 2 : 1) + " GiB";
// Bytes in binary units, 4 significant digits: 5.056 TiB, 445.4 GiB, 96.00 MiB.
function fmtBytes(n) {
  for (const [u, s] of [["TiB", 2 ** 40], ["GiB", 2 ** 30], ["MiB", 2 ** 20], ["KiB", 2 ** 10]])
    if (n >= s) return (n / s).toPrecision(4) + " " + u;
  return n + " B";
}
// Decimal units, for comparing against the GB200's 192 GB.
function fmtDecimal(n) {
  for (const [u, s] of [["TB", 1e12], ["GB", 1e9], ["MB", 1e6]])
    if (n >= s) return (n / s).toPrecision(3) + " " + u;
  return n + " B";
}
const GB200_BYTES = 192e9;

function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "style") e.style.cssText = v;
    else e.setAttribute(k, v);
  }
  if (text !== undefined) e.textContent = text;
  return e;
}
const SVGNS = "http://www.w3.org/2000/svg";
function sel(tag, attrs = {}, text) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}

// ---------- tooltip ----------
const tip = document.getElementById("tip");
function showTip(lines, x, y) {
  tip.replaceChildren();
  for (const ln of lines) {
    const d = el("div", { class: "line" });
    if (ln.color) d.append(el("span", { class: "key", style: `background:${ln.color}` }));
    const t = el("span", { class: ln.cls || "" }, ln.text);
    d.append(t);
    tip.append(d);
  }
  tip.style.display = "block";
  const r = tip.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
  tip.style.left = Math.max(8, left) + "px";
  tip.style.top = Math.max(8, top) + "px";
}
function hideTip() { tip.style.display = "none"; }
function attachTip(node, linesFn) {
  node.addEventListener("pointermove", ev => showTip(linesFn(), ev.clientX, ev.clientY));
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("focus", () => {
    const r = node.getBoundingClientRect();
    showTip(linesFn(), r.right - 40, r.top + r.height / 2);
  });
  node.addEventListener("blur", hideTip);
}

"""


def render(template: str, data: dict) -> str:
    """Fill a page template's CSS, JS helpers and embedded data."""
    return (
        template.replace("/*__CSS__*/", PAGE_CSS)
        .replace("/*__HELPERS__*/", PAGE_JS_HELPERS)
        .replace("/*__DATA__*/null", json.dumps(data))
    )
