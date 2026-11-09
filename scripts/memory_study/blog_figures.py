# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static SVG figures for the blog drafts, referenced as `::: figure <name>`."""

import html

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
KDA, MLA, STACK, PARTIAL = "#2a78d6", "#eb6834", "#4a3aa7", "#1baf7a"
FONT = 'font-family="system-ui, -apple-system, Segoe UI, sans-serif"'

NUM_LAYERS, BLOCK = 93, 12
MLA_LAYERS = set(range(3, 92, 4)) | {92}


def _text(x, y, s, size=12, fill=INK2, anchor="start", weight=400):
    return (f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" {FONT}>{html.escape(s)}</text>')


def layer_strip() -> str:
    """93 layers in 8 attention-residual blocks of 12 (one outlined row per
    block), KDA vs MLA, between the token embedding and the output (a last
    mix over the stack and the partial, final norm, LM head); each finished
    block's sum becomes a stack entry."""
    cw, ch, gap, pad, rgap, left, top = 46, 28, 3, 5, 12, 150, 8
    rows = (NUM_LAYERS + BLOCK - 1) // BLOCK
    bar_w = BLOCK * (cw + gap) - gap
    chip_x = left + bar_w + pad + 26
    chip_w = 96
    w = chip_x + chip_w + 8
    rh = ch + 2 * pad  # one block's outlined row
    h = top + 2 * (ch + rgap) + rows * (rh + rgap) + 62
    out = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
        f'aria-label="Kimi K3: token embedding, 93 layers (69 KDA and 24 MLA) in 8 attention-residual '
        f'blocks of 12, each block\'s sum joining the stack, then a final mix, norm and LM head">',
        '<defs><marker id="ls-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
        f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{STACK}"/></marker></defs>',
    ]

    def end_bar(y, label):
        out.append(f'<rect x="{left}" y="{y}" width="{bar_w}" height="{ch}" rx="4" fill="{MUTED}" fill-opacity="0.35"/>')
        out.append(_text(left + bar_w / 2, y + ch / 2 + 4, label, 12, INK, "middle", 600))

    def chip(y, label):
        out.append(f'<rect x="{chip_x}" y="{y}" width="{chip_w}" height="{ch}" rx="6" fill="{STACK}" '
                   f'fill-opacity="0.12" stroke="{STACK}" stroke-width="1.3"/>')
        out.append(_text(chip_x + chip_w / 2, y + ch / 2 + 4, label, 11, INK, "middle", 600))

    end_bar(top, "Token embedding")
    out.append(f'<path d="M{left + bar_w + 2},{top + ch / 2} L{chip_x - 2},{top + ch / 2}" stroke="{STACK}" '
               'stroke-width="1.3" marker-end="url(#ls-arr)"/>')
    chip(top, "stack entry 0")
    top += ch + rgap
    for r in range(rows):
        y = top + r * (rh + rgap)
        first, last = r * BLOCK, min(NUM_LAYERS, (r + 1) * BLOCK) - 1
        n = last - first + 1
        bw = n * (cw + gap) - gap + 2 * pad
        out.append(f'<rect x="{left - pad}" y="{y}" width="{bw}" height="{rh}" rx="7" fill="{STACK}" '
                   f'fill-opacity="0.05" stroke="{STACK}" stroke-width="1.3"/>')
        out.append(_text(left - pad - 10, y + rh / 2 - 2, f"AttnRes block {r}", 12, INK, "end", 600))
        out.append(_text(left - pad - 10, y + rh / 2 + 13, f"layers {first}-{last}", 11, MUTED, "end"))
        for c in range(n):
            i = first + c
            x = left + c * (cw + gap)
            kind = "MLA" if i in MLA_LAYERS else "KDA"
            tip = f"Layer {i}: {kind} attention, {'dense FFN' if i == 0 else 'MoE'}"
            out.append(
                f'<g><title>{tip}</title><rect x="{x}" y="{y + pad}" width="{cw}" height="{ch}" rx="4" '
                f'fill="{MLA if kind == "MLA" else KDA}" fill-opacity="{0.9 if kind == "MLA" else 0.8}"/>'
            )
            out.append(_text(x + cw / 2, y + pad + ch / 2 + 4, str(i) + ("*" if i == 0 else ""),
                             11, "#ffffff", "middle", 600))
            out.append("</g>")
        cy = y + rh / 2
        if r < rows - 1:  # the block's sum joins the stack when the next block starts
            out.append(f'<path d="M{left - pad + bw + 2},{cy} L{chip_x - 2},{cy}" stroke="{STACK}" '
                       'stroke-width="1.3" marker-end="url(#ls-arr)"/>')
            chip(cy - ch / 2, f"stack entry {r + 1}")
        else:
            out.append(_text(chip_x + chip_w / 2, cy + 4, "(the partial)", 11, MUTED, "middle"))
    end_bar(top + rows * (rh + rgap), "Mix of all stack entries and the partial, final norm, LM head")
    ly = h - 34
    out.append(f'<rect x="{left}" y="{ly - 10}" width="12" height="12" rx="2" fill="{KDA}" fill-opacity="0.8"/>')
    out.append(_text(left + 18, ly, "KDA (linear attention), 69 layers", 12, INK2))
    out.append(f'<rect x="{left + 250}" y="{ly - 10}" width="12" height="12" rx="2" fill="{MLA}" fill-opacity="0.9"/>')
    out.append(_text(left + 268, ly, "MLA (full attention), 24 layers", 12, INK2))
    out.append(f'<rect x="{left + 490}" y="{ly - 10}" width="12" height="12" rx="2" fill="{STACK}" '
               f'fill-opacity="0.12" stroke="{STACK}" stroke-width="1.2"/>')
    out.append(_text(left + 508, ly, "attention-residual block / stack", 12, INK2))
    out.append(_text(left, ly + 20, "* layer 0 has a dense FFN instead of MoE", 12, MUTED))
    out.append("</svg>")
    return "\n".join(out)


def attention_residuals() -> str:
    """How a sublayer inside block 2 builds its input: score every source
    (stack entries and the block's partial sum) with a learned vector, softmax
    over sources per token, mix; its output is added to the partial. At each
    block boundary the partial joins the stack and a new partial starts."""
    w, h = 900, 420
    out = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
        'aria-label="Attention residuals: a sublayer in block 2 mixes the embedding, the sums of blocks 0 '
        'and 1 and the partial sum of block 2, weighted by a softmax over learned scores; its output is '
        'added to the partial, which joins the stack when the block ends">',
        '<defs><marker id="ar-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>'
        '<marker id="ar-arr-g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{PARTIAL}"/></marker>'
        '<marker id="ar-arr-s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{STACK}"/></marker></defs>',
    ]

    def box(x, y, bw, bh, color=INK2, fill=SURFACE, dash=False, op=1.0):
        d = ' stroke-dasharray="5 3"' if dash else ""
        out.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="7" fill="{fill}" '
                   f'fill-opacity="{op}" stroke="{color}" stroke-width="1.5"{d}/>')

    def arrow(d, color=MUTED, marker="ar-arr"):
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.4" marker-end="url(#{marker})"/>')

    # --- sources: the stack and the partial
    sx, sw, sh, sy = 20, 200, 34, 52
    out.append(_text(sx, 24, "Sources", 13, INK, weight=600))
    out.append(_text(sx, 40, "for a sublayer in block 2", 11, MUTED))
    src = [("Embedding", STACK, False), ("Block 0: sum of layers 0-11", STACK, False),
           ("Block 1: sum of layers 12-23", STACK, False), ("Partial: block 2 so far", PARTIAL, True)]
    ys = []
    for k, (label, color, dash) in enumerate(src):
        y = sy + k * (sh + 8) + (14 if dash else 0)
        ys.append(y + sh / 2)
        box(sx, y, sw, sh, color, color, dash, 0.12)
        out.append(_text(sx + 10, y + sh / 2 + 4, label, 12, INK))
    out.append(f'<rect x="{sx - 6}" y="{sy - 6}" width="{sw + 12}" height="{3 * (sh + 8) + 4}" rx="9" '
               f'fill="none" stroke="{STACK}" stroke-width="1" stroke-dasharray="2 3"/>')
    out.append(_text(sx + sw + 10, sy + 6, "stack", 11, STACK, weight=600))

    # --- scoring, softmax, mix
    cx, cw_, cy, chh = 300, 250, 46, 172
    box(cx, cy, cw_, chh)
    out.append(_text(cx + cw_ / 2, cy + 24, "Mix the sources", 13, INK, "middle", 600))
    lines = [("score each source:", INK2), ("s_i = w \u00b7 RMSNorm(source_i)", INK),
             ("(w: learned, one per sublayer)", MUTED), ("", INK2),
             ("weights: softmax of s over sources", INK2), ("(per token)", MUTED), ("", INK2),
             ("input = \u03a3 weight_i \u00d7 source_i", INK)]
    for k, (t, col) in enumerate(lines):
        if t:
            out.append(_text(cx + cw_ / 2, cy + 46 + k * 14, t, 12, col, "middle"))
    for y in ys:
        arrow(f"M{sx + sw},{y} C{sx + sw + 40},{y} {cx - 40},{cy + chh / 2} {cx},{cy + chh / 2}")

    # --- the sublayer
    lx, lw = 610, 260
    box(lx, cy + 30, lw, 90)
    out.append(_text(lx + lw / 2, cy + 70, "Norm, then attention", 13, INK, "middle", 600))
    out.append(_text(lx + lw / 2, cy + 90, "(or the FFN / MoE)", 12, INK2, "middle"))
    arrow(f"M{cx + cw_},{cy + chh / 2} L{lx},{cy + chh / 2}")

    # --- output added to the partial
    pb = ys[3] + sh / 2
    route = pb + 26
    out.append(f'<path d="M{lx + lw / 2},{cy + 120} L{lx + lw / 2},{route} L{sx + sw / 2},{route} '
               f'L{sx + sw / 2},{pb + 2}" fill="none" stroke="{PARTIAL}" stroke-width="1.5" '
               'marker-end="url(#ar-arr-g)"/>')
    out.append(_text(lx + lw / 2 - 10, route - 8, "output is added to the partial (the residual stream)",
                     12, INK2, "end"))

    # --- block boundary
    by = route + 40
    out.append(f'<line x1="{sx}" y1="{by - 14}" x2="{w - 20}" y2="{by - 14}" stroke="{GRID}"/>')
    out.append(_text(sx, by + 6, "At each block boundary (every 12 layers)", 13, INK, weight=600))
    bx, bw2, bh2, row = sx, 150, 32, by + 26
    box(bx, row, bw2, bh2, PARTIAL, PARTIAL, True, 0.12)
    out.append(_text(bx + bw2 / 2, row + 20, "Partial: all of block 2", 11, INK, "middle"))
    arrow(f"M{bx + bw2},{row + bh2 / 2} L{bx + bw2 + 60},{row + bh2 / 2}", STACK, "ar-arr-s")
    sx2 = bx + bw2 + 64
    box(sx2, row, 180, bh2, STACK, STACK, False, 0.12)
    out.append(_text(sx2 + 90, row + 20, "becomes stack entry 3", 11, INK, "middle"))
    out.append(_text(sx2 + 200, row + 14, "Block 3's first sublayer mixes 4 stack entries and no partial;",
                     12, INK2))
    out.append(_text(sx2 + 200, row + 30, "its output starts a new partial.", 12, INK2))
    out.append(_text(sx, h - 30, "Every layer does this twice, before its attention and before its FFN / MoE, "
                     "each with its own w.", 12, INK2))
    out.append(_text(sx, h - 12, "With 93 layers a sublayer mixes at most 9 sources: the embedding, "
                     "7 finished blocks and the partial.", 12, INK2))
    out.append("</svg>")
    return "\n".join(out)


FIGURES = {"layer-strip": layer_strip, "attention-residuals": attention_residuals}


# ---------------------------------------------------------------- log2 memory bars
import functools as _functools
import json as _json
import math as _math
from pathlib import Path as _Path

GB200_BYTES = 192e9
COLORS = {"experts": "#2a78d6", "attention": "#eb6834", "other": "#1baf7a", "acts": "#eda100",
          "acts_full": "#9a6400", "weights": "#e87ba4", "grads": "#008300", "optstate": "#4a3aa7"}


def _data():
    return _json.loads((_Path(__file__).resolve().parent / "kimi_k3_data.json").read_text())


def _dec(n: float) -> str:
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6)):
        if n >= div:
            v = n / div
            return f"{v:.2f} {unit}" if v < 10 else (f"{v:.1f} {unit}" if v < 100 else f"{v:.0f} {unit}")
    return f"{n:.0f} B"


def _bin(e: int) -> str:
    return {30: "1 GiB", 40: "1 TiB"}.get(e) or (f"{2 ** (e - 40)} TiB" if e > 40 else f"{2 ** (e - 30)} GiB"
                                                if e > 30 else f"{2 ** (e - 20)} MiB")


def log2_bars(rows, aria, lo=30, hi=45, legend=None) -> str:
    """Horizontal bars on a log2 axis (one gridline per power of 2, labels every
    4x) with a line at one GB200's 192 GB. rows: (label, bytes, color), or
    (label, [(bytes, color, part label), ...], None) for a bar stacked from parts
    (each part's length on the log axis is the increase it adds)."""
    lab, w, bar, gap, top = 250, 760, 22, 10, 24  # w leaves room for the last axis label
    leg_rows, cur, cw = [], [], 40
    for name, color in legend or []:
        iw = 18 + 6.3 * len(name) + 22
        if cur and cw + iw > w - 10:
            leg_rows.append(cur); cur, cw = [], 40
        cur.append((name, color, cw)); cw += iw
    if cur:
        leg_rows.append(cur)
    h = top + len(rows) * (bar + gap) + 40 + 22 * len(leg_rows)
    axis_y = h - 18 - 22 * len(leg_rows)
    x = lambda b: lab + (w - lab - 64) * (_math.log2(max(b, 2 ** lo)) - lo) / (hi - lo)
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{html.escape(aria)}">']
    for e in range(lo, hi + 1):
        out.append(f'<line x1="{x(2 ** e):.1f}" x2="{x(2 ** e):.1f}" y1="{top - 6}" y2="{axis_y - 16}" stroke="{GRID}" '
                   f'stroke-width="{1 if e % 2 else 1.4}"/>')
        if e % 2 == 0:
            out.append(_text(x(2 ** e), axis_y, _bin(e), 11, MUTED, "middle"))
    gx = x(GB200_BYTES)
    out.append(f'<line x1="{gx:.1f}" x2="{gx:.1f}" y1="{top - 14}" y2="{axis_y - 16}" stroke="{INK}" stroke-width="1.5" '
               'stroke-dasharray="4 3"/>')
    out.append(_text(gx + 4, top - 12, "one GB200: 192 GB", 11, INK))
    for k, (label, b, color) in enumerate(rows):
        y = top + k * (bar + gap)
        out.append(_text(lab - 10, y + bar / 2 + 4, label, 12, INK2, "end"))
        if isinstance(b, list):
            parts, b = b, sum(p[0] for p in b)
            tip = "; ".join(f"{pl} {_dec(pb)}" for pb, _, pl in parts)
            out.append(f'<g><title>{html.escape(label)}: {_dec(b)} ({html.escape(tip)})</title>')
            start = 0.0
            for i, (pb, pc, _) in enumerate(parts):
                x0, x1 = (lab if i == 0 else x(start)), x(start + pb)
                out.append(f'<rect x="{x0:.1f}" y="{y}" width="{x1 - x0:.1f}" height="{bar}" '
                           f'rx="{4 if i in (0, len(parts) - 1) else 0}" fill="{pc}"/>')
                start += pb
            out.append("</g>")
        else:
            out.append(f'<g><title>{html.escape(label)}: {_dec(b)} (2^{_math.log2(b):.2f})</title>'
                       f'<rect x="{lab}" y="{y}" width="{x(b) - lab:.1f}" height="{bar}" rx="4" fill="{color}"/></g>')
        out.append(_text(x(b) + 6, y + bar / 2 + 4, _dec(b), 12, INK, weight=600)
                   .replace("<text ", '<text stroke="#f9f9f7" stroke-width="4" paint-order="stroke" ', 1))
    for r, entries in enumerate(leg_rows):
        ly = h - 22 * (len(leg_rows) - r) + 10
        for name, color, lx in entries:
            out.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" rx="2" fill="{color}"/>')
            out.append(_text(lx + 18, ly, name, 12, INK2))
    out.append("</svg>")
    return "\n".join(out)


def memory_weights_bf16() -> str:
    """fp32 weights by component and one 8K sequence's saved activations."""
    d = _data()
    w = d["persistent"]["model"]["weights"]  # fp32 bytes
    acts = d["contexts"]["8K"]["activations"]["total"]
    return log2_bars([("Routed experts (weights)", w["routed"], COLORS["experts"]),
                      ("Saved activations, one 8K sequence", acts, COLORS["acts"]),
                      ("Attention (weights)", w["attention"], COLORS["attention"]),
                      ("Shared experts (weights)", w["shared"], COLORS["other"]),
                      ("Everything else (weights)", w["others"], "#c3c2b7")],
                     "fp32 weights by component and saved activations, log2 axis")


def memory_by_kind(ctx: str = "8K") -> str:
    """Persistent state by kind and one sequence's saved activations."""
    d = _data()
    P = d["persistent"]["model"]
    tot = lambda k: sum(P[k].values())
    acts = d["contexts"][ctx]["activations"]["total"]
    return log2_bars([("Master weights + optimizer (fp32)", tot("weights") + tot("state"), COLORS["optstate"]),
                      ("Gradients (fp32)", tot("grads"), COLORS["grads"]),
                      (f"Stashed activations, one {ctx} sequence", acts, COLORS["acts"])],
                     f"persistent state by kind and saved activations at {ctx}, log2 axis")


def memory_remat(ctx: str = "8K") -> str:
    """Saved activations of one sequence at each remat level; full remat as its
    stash plus the recompute working set at the backward peak."""
    d = _data()
    c = d["contexts"][ctx]
    full = c["quant"]["full"]["off"]["total"]
    rws = recompute_working_set(ctx, c["seq_len"])
    return log2_bars([("No remat", c["activations"]["total"], COLORS["acts"]),
                      ("Default remat", c["remat"]["default"]["total"], COLORS["acts"]),
                      ("Full remat", [(full, COLORS["acts_full"], "stashed"),
                                      (rws, "#e87ba4", "recompute peak")], None)],
                     f"stashed activations of one {ctx} sequence by remat level, log2 axis",
                     legend=[("Stashed (no / default remat)", COLORS["acts"]), ("Stashed (full remat)", COLORS["acts_full"]),
                             ("Recompute peak (full remat)", "#e87ba4")])


def linear_stacked_bars(rows, aria, legend, top_gb=None, clip=()) -> str:
    """Stacked horizontal bars on a linear axis (GB), with a line at one GB200.
    rows: (label, [(bytes, color, part label), ...]). Rows whose label is in
    `clip` are left out of the axis range and drawn cut off at the right edge,
    with a break mark and their true total."""
    lab, w, bar, gap, top = 250, 760, 22, 10, 24
    # legend entries, wrapped into rows that fit the width
    leg_rows, cur, cw = [], [], 40
    for name, color in legend:
        iw = 18 + 6.3 * len(name) + 22
        if cur and cw + iw > w - 10:
            leg_rows.append(cur); cur, cw = [], 40
        cur.append((name, color, cw)); cw += iw
    leg_rows.append(cur)
    h = top + len(rows) * (bar + gap) + 44 + 20 * len(leg_rows)
    axis_y = h - 22 - 20 * len(leg_rows)
    vmax = top_gb * 1e9 if top_gb else max(sum(p[0] for p in parts) for l, parts in rows if l not in clip) * 1.15
    vmax = max(vmax, GB200_BYTES * 1.3)  # room for the one-GPU label
    x = lambda b: lab + (w - lab - 64) * b / vmax
    step = 50e9 if vmax <= 400e9 else 100e9
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{html.escape(aria)}">']
    t = 0.0
    while t <= vmax:
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top - 6}" y2="{axis_y - 16}" stroke="{GRID}"/>')
        out.append(_text(x(t), axis_y, f"{t / 1e9:.0f} GB", 11, MUTED, "middle"))
        t += step
    gx = x(GB200_BYTES)
    out.append(f'<line x1="{gx:.1f}" x2="{gx:.1f}" y1="{top - 14}" y2="{axis_y - 16}" stroke="{INK}" '
               'stroke-width="1.5" stroke-dasharray="4 3"/>')
    out.append(_text(gx + 4, top - 12, "one GB200: 192 GB", 11, INK))
    for k, (label, parts) in enumerate(rows):
        y = top + k * (bar + gap)
        total = sum(p[0] for p in parts)
        out.append(_text(lab - 10, y + bar / 2 + 4, label, 12, INK2, "end"))
        tip = "; ".join(f"{pl} {_dec(pb)}" for pb, _, pl in parts)
        out.append(f'<g><title>{html.escape(label)}: {_dec(total)} ({html.escape(tip)})</title>')
        start, edge = 0.0, x(vmax)
        for pb, pc, _ in parts:
            x0, x1 = min(x(start), edge), min(x(start + pb), edge)
            if x1 > x0:
                out.append(f'<rect x="{x0:.1f}" y="{y}" width="{max(0.5, x1 - x0 - 1):.1f}" height="{bar}" fill="{pc}"/>')
            start += pb
        clipped = x(total) > edge
        if clipped:  # break mark: two slanted white lines across the bar near its cut end
            for dx in (-14, -8):
                out.append(f'<line x1="{edge + dx - 3:.1f}" y1="{y + bar + 2}" x2="{edge + dx + 3:.1f}" y2="{y - 2}" '
                           f'stroke="#f9f9f7" stroke-width="3"/>')
        out.append("</g>")
        lx_val = edge + 6 if clipped else x(total) + 6
        out.append(_text(lx_val, y + bar / 2 + 4, _dec(total), 12, INK, weight=600)
                   .replace("<text ", '<text stroke="#f9f9f7" stroke-width="4" paint-order="stroke" ', 1))
    for r, entries in enumerate(leg_rows):
        ly = h - 20 * (len(leg_rows) - r) + 4
        for name, color, lx in entries:
            out.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" rx="2" fill="{color}"/>')
            out.append(_text(lx + 18, ly, name, 12, INK2))
    out.append("</svg>")
    return "\n".join(out)


def memory_distmoe(ctx: str = "8K") -> str:
    """No remat, full remat, and full remat plus DistMoE's pieces, stacked."""
    d = _data()
    c = d["contexts"][ctx]
    dm = d["cp_sweep"][ctx]["1"]["nopp_f4"]
    full = c["quant"]["full"]["off"]["total"]
    rws = recompute_working_set(ctx, c["seq_len"])
    remat = [(full, COLORS["acts_full"], "stashed"), (rws, "#e87ba4", "recompute peak")]
    moe = [(dm["saved"], "#2a78d6", "DistMoE stashed inputs"), (dm["output"], "#4a3aa7", "DistMoE outputs"),
           (dm["scratch"], "#1baf7a", "DistMoE scratch")]
    no_remat = [(c["activations"]["total"], COLORS["acts"], "stashed")]
    return linear_stacked_bars([("No remat", no_remat), ("Full remat", remat), ("Full remat + DistMoE", remat + moe)],
                               f"no remat, full remat and DistMoE memory for one {ctx} sequence, linear axis",
                               clip=("No remat",),
                               legend=[("No remat stash", COLORS["acts"]), ("Full remat stash", COLORS["acts_full"]),
                                       ("Recompute peak", "#e87ba4"), ("DistMoE inputs", "#2a78d6"),
                                       ("DistMoE outputs", "#4a3aa7"), ("DistMoE scratch", "#1baf7a")])


def memory_summary(ctx: str = "8K") -> str:
    """Where Section 3 leaves us: persistent state vs activations after full
    remat + DistMoE."""
    d = _data()
    P = d["persistent"]["model"]
    tot = lambda k: sum(P[k].values())
    c = d["contexts"][ctx]
    dm = d["cp_sweep"][ctx]["1"]["nopp_f4"]
    acts = (c["quant"]["full"]["off"]["total"] + recompute_working_set(ctx, c["seq_len"])
            + dm["saved"] + dm["output"] + dm["scratch"])
    return log2_bars([("Master weights + optimizer (fp32)", tot("weights") + tot("state"), COLORS["optstate"]),
                      ("Gradients (fp32)", tot("grads"), COLORS["grads"]),
                      (f"Activations: full remat + DistMoE ({ctx})", acts, COLORS["experts"])],
                     f"persistent state and activations at {ctx}, log2 axis")


MXFP8_WEIGHT_BYTES = 1 + 1 / 1024  # e4m3 data + one scale per 32x32 block
# MXFP8 for the routed experts only (DistMoE); the dense linears run bf16, as in
# Kimi K3's pre-training. True restores MXFP8 dense linears everywhere.
DENSE_MXFP8 = False


def dense_quantized_params() -> int:
    """Dense params per layer whose compute copy is MXFP8 (0 with bf16 dense)."""
    return _data()["mxfp8_linear_params"]["kda_block"] if DENSE_MXFP8 else 0
TOP_K, NUM_EXPERTS, MOE_LAYERS = 16, 896, 92


def routing_metadata(ep: int, local_experts: float, local_tokens: float) -> float:
    """Bytes DistMoE stashes per MoE layer per microbatch for its backward:
    three int64 pointer arrays (forward gather, backward gather, scatter), each
    sized for the worst case of every EP rank's tokens picking every local
    expert: ep x local tokens x min(top-k, local experts) entries."""
    return 3 * 8 * ep * local_tokens * min(TOP_K, local_experts)


def expert_padding(num_experts: float, degree: int) -> float:
    """FSDP shards the expert dimension; an uneven split is padded up to a
    multiple of the FSDP degree (14 experts over 8 GPUs -> 16)."""
    return -(-num_experts // degree) * degree / num_experts


def recompute_working_set(ctx: str, local_tokens: float) -> float:
    """In backward, full remat recomputes the layer being backpropagated: its
    intermediate activations (one KDA layer without remat, routed experts
    excluded -- DistMoE recomputes those in its own scratch) and their
    gradients, built as the intermediates are freed: about 1.5x the layer's
    stash."""
    d = _data()
    c = d["contexts"][ctx]
    layer = {x["key"]: x["bytes"] for x in c["activations"]["kda_layer"]["components"]}
    stash = sum(v for k, v in layer.items() if k != "routed")
    return 1.5 * stash * local_tokens / c["seq_len"]


def padded_shards(ctx_data: dict, degree: int, pad: float) -> tuple[float, float]:
    """(gradients, master + optimizer) per GPU when sharded `degree` ways, with
    the expert part padded by `pad`."""
    P = ctx_data["persistent"]["model"]
    g = (sum(P["grads"].values()) - P["grads"]["routed"] + pad * P["grads"]["routed"]) / degree
    mo_r = P["weights"]["routed"] + P["state"]["routed"]
    mo = (sum(P["weights"].values()) + sum(P["state"].values()) - mo_r + pad * mo_r) / degree
    return g, mo


def zero_parts(ctx: str = "8K") -> list:
    """Per-GPU memory over 512 GPUs at each sharding step (data parallel, one
    sequence per microbatch, full remat + DistMoE), with the compute weights in
    MXFP8 (FSDP releases the bf16 all-gather output once the MXFP8 copy is
    built). Rows: (label, compute weights, gradients, master + optimizer,
    activations, peak from the chunked-loss loop or 0, total, activation parts)."""
    d = _data()
    P = d["persistent"]["model"]
    tot = lambda k: sum(P[k].values())
    n = sum(d["optimizers"]["params"].values())
    S = d["symbols"]
    T = d["contexts"][ctx]["seq_len"]
    c = d["contexts"][ctx]
    dm = d["cp_sweep"][ctx]["1"]["nopp_f4"]
    base_acts = c["quant"]["full"]["off"]["total"] + dm["saved"] + dm["output"] + dm["scratch"]
    quantized = d["mxfp8_linear_params"]["model"] * DENSE_MXFP8 + P["weights"]["routed"] / 4  # params in quantized matrices
    copy = MXFP8_WEIGHT_BYTES * quantized + 2 * (n - quantized)  # whole compute copy
    blk = {r["key"]: r["numel"] for r in d["scopes"]["kda_layer"]["stored"]}
    q_blk = dense_quantized_params()
    layer_w = lambda routed: MXFP8_WEIGHT_BYTES * (q_blk + routed) + 2 * (sum(blk.values()) - blk["routed"] - q_blk)
    layer_g = lambda routed: 4 * (sum(blk.values()) - blk["routed"] + routed)  # one layer's fp32 gradient
    chunks = d["chunked_loss"]
    lm = S["V"] * S["D"]
    chunk_loop = 6 * lm + T / chunks * (442e3 + 4 * S["V"]) + 4 * T * S["D"]  # 442 KB: a chunk's saves per token
    rws = recompute_working_set(ctx, T)
    G, MO = tot("grads"), tot("weights") + tot("state")
    pad = expert_padding(NUM_EXPERTS, 512)  # without EP, each layer's 896 experts over 512 GPUs
    g_shard, mo_shard = padded_shards(d, 512, pad)
    meta1 = MOE_LAYERS * routing_metadata(1, NUM_EXPERTS, T)
    acts1 = base_acts + rws + meta1  # full remat's stash and recompute peak, DistMoE, routing indices
    out = []
    for label, w, g, mo in (("No sharding", 0, G, MO), ("ZeRO-1", copy, G, mo_shard), ("ZeRO-2", copy, g_shard, mo_shard)):
        out.append((label, w, g, mo, acts1, 0, w + g + mo + acts1, (base_acts, meta1, rws)))
    # FSDP overlaps communication with compute, so two layers are in flight: in
    # backward, the current and the prefetched layer's compute weights, and the
    # current layer's gradient plus the previous one's, still being reduce-scattered;
    # the layer being backpropagated is also recomputed in full
    for label, routed, ep in (("ZeRO-3 / FSDP", blk["routed"], 1), ("FSDP + EP 64", blk["routed"] / 64, 64)):
        meta = MOE_LAYERS * routing_metadata(ep, NUM_EXPERTS / ep, T)
        g_s, mo_s = padded_shards(d, 512, expert_padding(NUM_EXPERTS / ep, 512 // ep))
        acts = base_acts + rws + meta
        w, g = 2 * layer_w(routed), g_s + 2 * layer_g(routed)
        backward_peak = w + g + mo_s + acts
        loss_peak = g_s + mo_s + acts - rws + chunk_loop  # the loss runs before the recompute peak
        out.append((label, w, g, mo_s, acts, chunk_loop if loss_peak > backward_peak else 0,
                    max(backward_peak, loss_peak), (base_acts, meta, rws)))
    return out


def zero_steps(ctx: str = "8K") -> list:
    return [(r[0], r[6]) for r in zero_parts(ctx)]


def sharding_steps(ctx: str = "8K") -> str:
    rows = zero_steps(ctx)
    colors = ["#c3c2b7", "#4a3aa7", "#4a3aa7", "#2a78d6", "#1baf7a"]
    return log2_bars([(label, b, col) for (label, b), col in zip(rows, colors)],
                     f"per-GPU memory at each sharding step at {ctx}, log2 axis")


def zero_table(ctx: str = "8K") -> str:
    """The per-GPU table behind sharding-steps."""
    rows = zero_parts(ctx)
    G512 = rows[2][2]  # ZeRO-2's gradient shard
    head = ["Per GPU", "Master + optimizer", "Gradients", "Activations", "Compute weights (MXFP8)",
            "Total", "x GPU"]
    w_notes = {"ZeRO-1": "whole model", "ZeRO-2": "whole model",
               "ZeRO-3 / FSDP": "2 x <b>896 experts</b>",
               "FSDP + EP 64": "2 x <b>14 experts</b>"}
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    prev = None

    def shade(v, pv):
        """Green if this row needs less than the row above, red if more."""
        if pv is None or abs(v - pv) <= 1e-6 * max(v, pv, 1):
            return ""
        return ' style="background: rgba(0, 131, 0, 0.12)"' if v < pv else ' style="background: rgba(227, 73, 72, 0.15)"'

    for label, w, g, mo, a, loop, total, (base_a, meta, rws) in rows:
        cell = lambda v, note="": (f"{_dec(v) if v else '-'}" + (f'<span class="sub">{note}</span>' if note else ""))
        a_note = f"incl. {_dec(meta)} DistMoE indices" if meta > 1e9 else ""
        if label in ("ZeRO-3 / FSDP", "FSDP + EP 64"):
            experts = "896 experts" if label == "ZeRO-3 / FSDP" else "14 experts"
            g_note = f"{_dec(G512)} + {_dec(g - G512)} (2 x <b>{experts}</b>)"
        else:
            g_note = {"No sharding": "", "ZeRO-1": "unsharded", "ZeRO-2": "1/512"}[label]
        tot = f"<b>{_dec(total)}</b>" + ('<span class="sub">peak: chunked-loss loop</span>' if loop else "")
        p = prev or (None,) * 5
        out.append(f"<tr><td>{label}</td><td{shade(mo, p[0])}>{cell(mo, '' if label == 'No sharding' else '1/512')}</td>"
                   f"<td{shade(g, p[1])}>{cell(g, g_note)}</td><td{shade(a, p[2])}>{cell(a, a_note)}</td>"
                   f"<td{shade(w, p[3])}>{cell(w, w_notes.get(label, ''))}</td>"
                   f"<td{shade(total, p[4])}>{tot}</td><td>{total / GB200_BYTES:.2f}</td></tr>")
        prev = (mo, g, a, w, total)
    out.append("</table>")
    return "".join(out)


def fsdp_ep_parts(ctx: str = "8K", cp: int = 1, tp: int = 1) -> dict:
    """FSDP + EP 64 per GPU with each sequence split over cp x tp GPUs; same
    rules as zero_parts. CP ranks fold into FSDP; TP (with sequence
    parallelism) also splits the dense weights and the vocab-parallel LM head
    tp ways, while the routed experts fold TP into EP. The shards stay 1/512."""
    d = _data()
    P = d["persistent"]["model"]
    tot = lambda k: sum(P[k].values())
    S = d["symbols"]
    T = d["contexts"][ctx]["seq_len"]
    split = cp * tp
    full = d["contexts"][ctx]["quant"]["full"]["off"]["total"]
    dm = d["cp_sweep"][ctx][str(split)]["nopp_f4"]
    tl = T / split
    meta = MOE_LAYERS * routing_metadata(64, NUM_EXPERTS / 64, tl)
    acts = full / split + recompute_working_set(ctx, tl) + dm["saved"] + dm["output"] + dm["scratch"] + meta
    blk = {r["key"]: r["numel"] for r in d["scopes"]["kda_layer"]["stored"]}
    q_blk = dense_quantized_params()
    routed = blk["routed"] / 64
    dense = sum(blk.values()) - blk["routed"]
    layer_w = MXFP8_WEIGHT_BYTES * (q_blk / tp + routed) + 2 * (dense - q_blk) / tp
    layer_g = 4 * (dense / tp + routed)
    shards = sum(padded_shards(d, 512, expert_padding(NUM_EXPERTS / 64, 8)))
    lm = S["V"] * S["D"]
    chunk_loop = 6 * lm / tp + tl / d["chunked_loss"] * (442e3 + 4 * S["V"] / tp) + 4 * tl * S["D"]
    buffers = 2 * (layer_w + layer_g)
    seqs = d["global_tokens"] / T
    return {"cp": cp, "tp": tp, "tokens": tl, "microbatches": seqs * split / 512, "acts": acts, "meta": meta,
            "shards": shards,
            "buffers": max(buffers, chunk_loop), "loop": chunk_loop > buffers,
            "total": shards + acts + max(buffers, chunk_loop)}


def cp_table(ctx: str = "8K") -> str:
    """FSDP + EP 64 at CP 1 / 2 / 4 / 8, shaded against the row above."""
    rows = [fsdp_ep_parts(ctx, cp) for cp in (1, 2, 4)]
    head = ["FSDP + EP 64", "Tokens / GPU", "Microbatches", "Activations", "Shards", "Largest temporary",
            "Total", "x GPU"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    prev = None
    for r in rows:
        sh = lambda k: ("" if prev is None or abs(r[k] - prev[k]) <= 1e-6 * max(r[k], 1) else
                        ' style="background: rgba(0, 131, 0, 0.12)"' if r[k] < prev[k]
                        else ' style="background: rgba(227, 73, 72, 0.15)"')
        fits = r["total"] <= GB200_BYTES
        buf = _dec(r["buffers"]) + f'<span class="sub">{"chunked-loss loop" if r["loop"] else "2 layers in flight"}</span>'
        out.append(f'<tr><td>CP {r["cp"]}</td><td>{r["tokens"]:,.0f}</td><td>{r["microbatches"]:g}</td>'
                   f'<td{sh("acts")}>{_dec(r["acts"])}<span class="sub">incl. {_dec(r["meta"])} DistMoE indices</span></td>'
                   f'<td>{_dec(r["shards"])}</td><td{sh("buffers")}>{buf}</td>'
                   f'<td{sh("total")}><b>{_dec(r["total"])}</b><span class="sub">{"fits" if fits else "over 192 GB"}</span></td>'
                   f'<td>{r["total"] / GB200_BYTES:.2f}</td></tr>')
        prev = r
    out.append("</table>")
    return "".join(out)


def tp_table(ctx: str = "8K") -> str:
    """TP next to CP of the same degree; each TP row shaded against its CP row."""
    pairs = [(fsdp_ep_parts(ctx, cp=k), fsdp_ep_parts(ctx, tp=k)) for k in (2, 4)]
    head = ["FSDP + EP 64", "Tokens / GPU", "Activations", "Largest temporary", "Total", "x GPU"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for c, t in pairs:
        for r, ref in ((c, None), (t, c)):
            sh = lambda k: ("" if ref is None or abs(r[k] - ref[k]) <= 1e-3 * max(r[k], 1) else
                            ' style="background: rgba(0, 131, 0, 0.12)"' if r[k] < ref[k]
                            else ' style="background: rgba(227, 73, 72, 0.15)"')
            name = f"CP {r['cp']}" if r["tp"] == 1 else f"TP {r['tp']}"
            delta = "" if ref is None else f'<span class="sub">{_dec(ref["total"] - r["total"])} less than CP {ref["cp"]}</span>'
            buf = _dec(r["buffers"]) + f'<span class="sub">{"chunked-loss loop" if r["loop"] else "2 layers in flight"}</span>'
            out.append(f'<tr><td>{name}</td><td>{r["tokens"]:,.0f}</td><td{sh("acts")}>{_dec(r["acts"])}</td>'
                       f'<td{sh("buffers")}>{buf}</td><td{sh("total")}><b>{_dec(r["total"])}</b>{delta}</td>'
                       f'<td>{r["total"] / GB200_BYTES:.2f}</td></tr>')
    out.append("</table>")
    return "".join(out)


def split_steps(ctx: str = "8K") -> str:
    """FSDP + EP 64 per-GPU totals with each sequence split by CP or TP."""
    rows = [("FSDP + EP 64", fsdp_ep_parts(ctx), "#1baf7a"),
            ("+ CP 2", fsdp_ep_parts(ctx, cp=2), "#2a78d6"), ("+ CP 4", fsdp_ep_parts(ctx, cp=4), "#2a78d6"),
            ("+ TP 2", fsdp_ep_parts(ctx, tp=2), "#eb6834"), ("+ TP 4", fsdp_ep_parts(ctx, tp=4), "#eb6834")]
    return log2_bars([(label, r["total"], color) for label, r, color in rows],
                     f"FSDP + EP 64 per-GPU memory with CP or TP at {ctx}, log2 axis")


def rank0_chunk_layers(P: int, v: int) -> float:
    """Mean layers in rank 0's chunks: 93 layers split over P x v virtual stages,
    the first 93 % (P v) one layer longer; rank 0 runs stages 0, P, 2P, ..."""
    base, extra = divmod(93, P * v)
    sizes = [base + (1 if r * P < extra else 0) for r in range(v)]
    return sum(sizes) / v


def pp_inflight(P: int, v: int, M: int) -> int:
    """Chunk-microbatches rank 0 holds at its peak: P with 1F1B (v = 1); with
    Interleaved 1F1B (PyTorch's _get_warmup_ops), (v - 1) x P + 2 (P - 1)
    warmup forward chunks plus the steady state's next forward."""
    if v == 1:
        return P
    return min((v - 1) * P + 2 * (P - 1) + 1, M * v)


def pp_parts(ctx: str = "8K", P: int = 8, zero: int = 1, cp: int = 1, tp: int = 1, v: int = 1) -> dict:
    """Per GPU on the first PP stage (the worst one under 1F1B), as on the
    memory page but with MXFP8 compute weights and two layers in flight.
    Each stage holds ceil(93 / P) layers on 512 / P GPUs, with FSDP and EP
    (EP = the stage's GPUs) inside the stage. ZeRO-1 (FSDP's default under PP)
    keeps the stage's compute weights gathered and its gradients unsharded all
    step; ZeRO-3 reshards every microbatch. 1F1B keeps P microbatches in flight
    on the first stage."""
    d = _data()
    lay = {r["key"]: r["numel"] for r in d["layers"][0]["scope"]["stored"]}
    layer_n, routed_n = sum(lay.values()), lay["routed"]
    dense_n = layer_n - routed_n
    q_blk = dense_quantized_params()
    g = 512 // P
    ls = -(-93 // P)
    act = d["contexts"][ctx]["quant"]["full"]["off"]
    head = sum(r["bytes"] for r in act["regions"] if r["label"].startswith("Head") or r["label"] == "Embedding")
    split = cp * tp
    seqs = d["global_tokens"] / act["seq_len"]
    m = seqs * P * split / 512
    dm = d["cp_sweep"][ctx][str(split)][f"pp{P}_f4"]
    tl = act["seq_len"] / split
    # layer-microbatches in flight on rank 0: P microbatches x the stage's
    # layers with 1F1B; with interleaving, more chunks of fewer layers
    held = round(pp_inflight(P, v, int(m)) * rank0_chunk_layers(P, v))
    meta = held * routing_metadata(g, NUM_EXPERTS / g, tl)
    dm_scale = held / (P * ls)  # DistMoE's saved inputs / outputs are per layer-microbatch too
    acts = (held * (act["total"] - head) / 93 / split + recompute_working_set(ctx, tl)
            + (dm["saved"] + dm["output"]) * dm_scale + dm["scratch"] + meta)
    mo = 8 * ls * layer_n / g
    block_w = MXFP8_WEIGHT_BYTES * (q_blk / tp + routed_n / g) + 2 * (dense_n - q_blk) / tp
    block_g = 4 * (dense_n / tp + routed_n / g)
    if zero == 1:
        w, gr, temp = ls * block_w, ls * block_g, 0.0
    else:
        w, gr, temp = 0.0, 4 * ls * layer_n / g, 2 * (block_w + block_g)
    total = mo + gr + acts + w + temp
    return {"P": P,
        "v": v,
        "held": held, "v": v, "held": held, "zero": zero, "cp": cp, "tp": tp, "layers": ls, "gpus": g, "experts": 896 // g,
            "microbatches": m, "mo": mo, "grads": gr, "acts": acts, "meta": meta, "weights": w, "temp": temp,
            "total": total,
            "valid": m >= P}


BF16_PEAK, MXFP8_PEAK = 2.5e15, 5e15
BF16_EFF, MXFP8_EFF = 0.70, 0.60
DENSE_SHAPES = "dense" if DENSE_MXFP8 else "dense_bf16"  # section of bench/shapes_GB300.json
DENSE_PEAK, DENSE_EFF = (MXFP8_PEAK, MXFP8_EFF) if DENSE_MXFP8 else (BF16_PEAK, BF16_EFF)
DENSE_PREC = "MXFP8" if DENSE_MXFP8 else "bf16"
KDA_US_PER_TOKEN = 0.45  # forward + remat forward + backward, per token per KDA layer (measured model)


def roofline_ops(ctx: str = "8K") -> list:
    """Roofline compute per GPU per step: (op, FLOPs or None, precision,
    seconds). Every GPU processes global_tokens / 512 tokens per step whatever
    the parallelism, so these are the same for every config."""
    d = _data()
    S = d["symbols"]
    sc = {k: {r["key"]: r["numel"] for r in v["stored"]} for k, v in d["scopes"].items()}
    kda, mla = sc["kda_layer"], sc["mla_layer"]
    T = 8192 if ctx == "8K" else 32768
    tg = d["global_tokens"] / 512
    dense = lambda l: l["attention"] + l["shared"] + l["others"]
    layer0 = kda["attention"] + 3 * S["D"] * S["F"]
    n_kda, n_mla = 69, 24
    dense_n = layer0 + (n_kda - 1) * dense(kda) + n_mla * dense(mla)
    expert_n = S["k"] * kda["routed"] / S["E"] * (n_kda + n_mla - 1)
    f8, b16 = MXFP8_EFF * MXFP8_PEAK, BF16_EFF * BF16_PEAK
    ops = [
        ("Dense linears", 8 * tg * dense_n, DENSE_PREC, 8 * tg * dense_n / (DENSE_EFF * DENSE_PEAK)),
        ("Routed experts", 8 * tg * expert_n, "MXFP8", 8 * tg * expert_n / f8),
        ("LM head", 6 * tg * S["V"] * S["D"], "bf16", 6 * tg * S["V"] * S["D"] / b16),
    ]
    mla_f = 4.5 * tg * T * S["H"] * (S["Dn"] + S["Dr"] + S["Dv"]) * n_mla
    ops.append(("MLA attention", mla_f, "bf16", mla_f / b16))
    ops.append(("KDA attention", None, "bf16", tg * KDA_US_PER_TOKEN * 1e-6 * n_kda))
    nm = sum(nonmatmul_bytes(ctx).values())
    ops.append(("Non-matmul (memory-bound; FSDP 512 + EP 64)", None, "HBM", nm / HBM_BW))
    return ops


def roofline_table(ctx: str = "8K") -> str:
    ops = roofline_ops(ctx)
    total = sum(o[3] for o in ops)
    head = ["Op", "TFLOPs per GPU per step", "Precision (efficiency)", "Time", "Share"]
    eff = {"MXFP8": f"MXFP8 ({MXFP8_EFF:.0%} of 5 PF)", "bf16": f"bf16 ({BF16_EFF:.0%} of 2.5 PF)"}
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for name, fl, prec, t in ops:
        fl_s = f"{fl / 1e12:,.0f}" if fl else ("-" if prec == "HBM" else "measured")
        prec_s = eff[prec] if fl else ("bytes / 8 TB/s" if prec == "HBM" else "bf16 (measured time)")
        out.append(f"<tr><td>{name}</td><td>{fl_s}</td><td>{prec_s}</td>"
                   f"<td>{t:.2f} s</td><td>{t / total:.0%}</td></tr>")
    out.append(f"<tr><td><b>Total</b></td><td></td><td></td><td><b>{total:.1f} s</b></td><td></td></tr>")
    out.append("</table>")
    return "\n".join(out)


SPLITS = [(1, 1), (2, 1), (4, 1), (1, 2), (1, 4)]
_GREEN = ' style="background: rgba(0, 131, 0, 0.12)"'
_RED = ' style="background: rgba(227, 73, 72, 0.15)"'


def _bench(name: str) -> dict:
    return _json.loads((_Path(__file__).resolve().parent / "bench" / name).read_text())


def _log_interp(pts: list, x: float) -> float:
    """Piecewise-linear in log x through sorted (x, y) points, clamped at the ends."""
    import math
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * math.log(x / x0) / math.log(x1 / x0)


@_functools.lru_cache(maxsize=None)
def distmoe_eff(tl: int, ep: int, precision: str = "mxfp8") -> float:
    """DistMoE's routed-expert kernels (grouped GEMMs with dispatch and combine
    fused in) as a fraction of MXFP8 peak, at tl local tokens per microbatch,
    counting 8 FLOPs per parameter per token (forward, remat, backward).

    EP 64: measured in the FSDP 512 + EP 64 fake-process-group steps, whose
    collectives move no data, so this is the kernels alone (averaged over the
    configs with the same local tokens). EP 32: scaled by the single-GPU
    DistMoE benchmark's EP 32 / EP 64 ratio at the same local tokens.

    precision="bf16": the same fake-process-group steps with bf16 experts
    (bench/expert_bf16/), as a fraction of bf16 peak; EP 64 only."""
    import re
    S = _data()["symbols"]
    n_expert = {r["key"]: r["numel"] for r in _data()["scopes"]["kda_layer"]["stored"]}["routed"] / S["E"]
    flops = 8 * _data()["global_tokens"] / 512 * S["k"] * n_expert * 92  # per GPU per step
    bench = _Path(__file__).resolve().parent / "bench"
    folder, peak = {"mxfp8": (bench, MXFP8_PEAK), "bf16": (bench / "expert_bf16", BF16_PEAK)}[precision]
    by_tl = {}
    for f in sorted(folder.glob("step_sweep_*k_cp*_tp*.json")):
        ctx, cp, tp = re.match(r"step_sweep_(\d+)k_cp(\d)_tp(\d)", f.stem).groups()
        t = _json.loads(f.read_text())["fwd_bwd_by_class_s"]["DistMoE grouped GEMM (+fused dispatch/combine)"]
        by_tl.setdefault(int(ctx) * 1024 // (int(cp) * int(tp)), []).append(flops / t / peak)
    eff = _log_interp(sorted((k, sum(v) / len(v)) for k, v in by_tl.items()), tl)
    if ep == 64:
        return eff
    assert ep == 32 and precision == "mxfp8", (ep, precision)
    rows = [r for r in _bench("dist_moe_GB300_ep1.json")["results"] if r["dtype"] == "mxfp8"]
    tf = lambda layout: sorted((r["local_tokens"], r["fwd_bwd_tflops"]) for r in rows if r["layout"] == layout)
    return eff * _log_interp(tf("ep32_as32"), tl) / _log_interp(tf("ep64_as16"), tl)


def _eff_cell(eff: float, assumed: float) -> str:
    """% of peak, green at or above the assumption, red more than 5 points below."""
    sh = _GREEN if eff >= assumed - 0.005 else _RED if eff < assumed - 0.05 else ""
    return f"<td{sh}>{eff:.0%}</td>"


def ideal_gap_table(ctx: str = "8K") -> str:
    """Measured time over the ideal roofline's, per compute op and split: dense
    GEMMs and routed experts (MXFP8, assumed 60%), MLA (FA4 bf16, assumed 70%),
    KDA (its CP 1, TP 1 time per token, scaled perfectly)."""
    T = 8192 if ctx == "8K" else 32768
    sh = _bench("shapes_GB300.json")
    fa = _bench("mla_fa4_GB300.json")["mla_fa4"]["data"]["heads"][str(T)]
    head = ["Split", "Dense GEMMs", "Routed experts (EP 64)", "MLA attention", "KDA attention"]
    out = [f'<table class="dtable"><caption>Slowdown against the ideal roofline at {ctx}: measured time / the '
           'ideal roofline\'s time (1.00x: the ideal roofline is right; red: more than 5% slower)</caption><tr>'
           + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]

    def cell(x):
        # neutral within 5% of the ideal, red tint growing with the gap
        a = 0.0 if x < 1.05 else min(0.45, 0.10 + 0.35 * (x - 1.05) / 1.0)
        style = f' style="background: rgba(227, 73, 72, {a:.2f})"' if a else ""
        return f"<td{style}>{x:.2f}x</td>"

    for cp, tp in SPLITS:
        dn = sh[DENSE_SHAPES]["data"][f"m{T // cp}_tp{tp}"]
        dense = (69 * DENSE_EFF / dn["kda_layer"]["eff"] + 24 * DENSE_EFF / dn["mla_layer"]["eff"]) / 93
        ex = MXFP8_EFF / distmoe_eff(T // (cp * tp), 64)
        mla = BF16_EFF / (fa[str(96 // (cp * tp))]["fwd_bwd_tflops"] * 1e12 / BF16_PEAK)
        cpf = 1.0
        if cp > 1:
            r = {x["seq_len"]: x for x in _bench(f"kda_cp_GB300_cp{cp}.json")["results"]}[T]
            cpf = r["cp_fwd_bwd_ms"] / r["local_fwd_bwd_ms"]
        kda = cpf * tp / KDA_TP_SPEEDUP[ctx][tp]
        out.append(f"<tr><td>CP {cp}, TP {tp}</td>{cell(dense)}{cell(ex)}{cell(mla)}{cell(kda)}</tr>")
    out.append("</table>")
    return "\n".join(out)


def dense_shapes_table(ctx: str = "8K") -> str:
    """Measured dense GEMM efficiency at each split's shapes."""
    d = _bench("shapes_GB300.json")[DENSE_SHAPES]["data"]
    T = 8192 if ctx == "8K" else 32768
    head = ["Split", "<code>M</code> (tokens)", "<code>N</code> (column-parallel) or <code>K</code> (row-parallel)",
            "KDA + MoE layer", "MLA + MoE layer"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for cp, tp in SPLITS:
        r = d[f"m{T // cp}_tp{tp}"]
        out.append(f"<tr><td>CP {cp}, TP {tp}</td><td>{T // cp:,}</td><td>{'1/' + str(tp) if tp > 1 else 'whole'}</td>"
                   + _eff_cell(r["kda_layer"]["eff"], DENSE_EFF) + _eff_cell(r["mla_layer"]["eff"], DENSE_EFF) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def expert_shapes_table(ctx: str = "8K") -> str:
    """DistMoE's routed-expert kernels per split and EP, as a fraction of MXFP8 peak."""
    d = _bench("shapes_GB300.json")["experts"]["data"]
    T = 8192 if ctx == "8K" else 32768
    head = ["Split", "Local tokens", "EP 64 (14 experts per GPU): <code>M</code> per expert", "EP 64",
            "EP 32 (28 experts per GPU): <code>M</code> per expert", "EP 32"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for cp, tp in SPLITS:
        tl = T // (cp * tp)
        a, b = d[f"t{tl}_ep64"], d[f"t{tl}_ep32"]
        out.append(f"<tr><td>CP {cp}, TP {tp}</td><td>{tl:,}</td><td>{a['rows_per_expert']:,}</td>"
                   + _eff_cell(distmoe_eff(tl, 64), MXFP8_EFF) + f"<td>{b['rows_per_expert']:,}</td>"
                   + _eff_cell(distmoe_eff(tl, 32), MXFP8_EFF) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def mla_table(ctx: str = "8K") -> str:
    """Measured FA4 MLA efficiency at each split's heads (Ulysses CP and TP both
    split heads; the sequence stays whole)."""
    d = _bench("mla_fa4_GB300.json")["mla_fa4"]["data"]["heads"]
    T = 8192 if ctx == "8K" else 32768
    head = ["Split", "Heads per GPU", "Forward", "Forward + backward"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for cp, tp in SPLITS:
        h = 96 // (cp * tp)
        r = d[str(T)][str(h)]
        out.append(f"<tr><td>CP {cp}, TP {tp}</td><td>{h}</td>"
                   + _eff_cell(r["fwd_tflops"] * 1e12 / BF16_PEAK, BF16_EFF)
                   + _eff_cell(r["fwd_bwd_tflops"] * 1e12 / BF16_PEAK, BF16_EFF) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


HBM_BW = 8e12  # GB200 HBM3e
CTX_TOKENS = {"4K": 4096, "8K": 8192, "32K": 32768}
MX_BYTES = 2 + 1 + 1 / 32  # MXFP8 quantization: read bf16, write fp8 + one scale per 32
# (in, out, TP layout) of each MXFP8 dense linear in a KDA + MoE / MLA + MoE layer
_KDA_LIN = [(7168, 12288, "col")] * 4 + [(12288, 7168, "row"), (7168, 12288, "col"), (6144, 7168, "row"),
                                         (7168, 3584, "rep_local"), (3584, 7168, "rep_local")]
_MLA_LIN = [(7168, 1536, "rep"), (1536, 18432, "col"), (7168, 576, "rep"), (512, 24576, "col"),
            (7168, 12288, "col"), (12288, 7168, "row"), (7168, 12288, "col"), (6144, 7168, "row"),
            (7168, 3584, "rep_local"), (3584, 7168, "rep_local")]


def nonmatmul_bytes(ctx: str = "8K", cp: int = 1, tp: int = 1, ep: int = 64, P: int = 1,
                    split_weights: bool = False, *, rows: int = 1, expert_mxfp8: bool = True) -> dict:
    """HBM bytes per GPU per step of the memory-bound work, by class (summed
    over all 93 layers: with PP each GPU runs 1/P of the layers on P times the
    tokens). Full remat: forward ops run twice (forward + remat), backward
    moves about twice a forward's bytes.

    - AttnRes (TorchTitan's fused Triton kernel), per mix over n sources of
      width D: forward and remat each read the n bf16 sources and write the
      output (2n + 2 B x D); backward reads them and the output gradient and
      writes their gradients (4n + 2): (8n + 6) x D in total. Each block's first
      layer also concatenates the stack (3 passes).
    - MXFP8 quantization (MX_BYTES per element): input x three times (forward,
      remat, weight gradient), output gradient twice; weights three times per
      microbatch under ZeRO-3, once per step in both orientations under PP's
      ZeRO-1 (the gathered stage stays alive).
    - FSDP buffer copies: per gather, copy-out of the bf16 gathered weights
      (read + write 2 B); per reduce-scatter, copy-in of bf16 gradients into the
      fp32 buffer (2 + 4 B). Two gathers and one reduce-scatter per layer and
      microbatch under ZeRO-3; once per step under PP's ZeRO-1.
    - Norms (RMSNorm, KDA's L2 / gated norm, MLA's q / kv norms): 4 B per
      element forward, 6 backward: 14 B over forward + remat + backward.
    - Other: two residual adds (16 B x D each), the output gate (22 B x 12,288)
      and the shared expert's SwiGLU (22 B x 6,144 / TP).

    split_weights: also report the weights' share of MXFP8 quantization as
    "MXFP8 quantize (weights)" (still included in "MXFP8 quantize").
    rows: sequences of ctx tokens packed into one microbatch.
    expert_mxfp8: False for bf16 routed experts (no expert weight quantization)."""
    D, k_routed = 7168, 33.030144e6
    T = CTX_TOKENS[ctx]
    tok = _data()["global_tokens"] / 512  # residual-stream tokens per GPU per step
    mb = tok * cp * tp / (T * rows)  # microbatches per GPU per step (P stages each see P x, on 1/P layers)
    layers = 93
    mixes = cats = 0
    for l in range(layers):
        b, first = l // 12, l % 12 == 0
        stack = b + 1
        n_att = stack + (0 if first else 1)
        mixes += (8 * n_att + 6) + (8 * (stack + 1) + 6)
        if first:
            cats += 2 * stack + 2 * (stack + 1)
    out = {"Attention residuals": (mixes + 3 * cats) * D * tok}

    def lin_q(lins, weights_only=False):
        total = 0.0
        for k, n, lay in lins:
            t = tok * (1 if lay == "rep_local" else tp)  # rows per GPU per step
            kk, nn = (k // tp, n) if lay == "row" else (k, n // tp) if lay == "col" else (k, n)
            total += MX_BYTES * ((0 if weights_only else 3 * t * kk + 2 * t * nn) + nn * kk * w_quant)
        return total
    # Weights: ZeRO-3 re-gathers every microbatch and quantizes them three times
    # (forward, remat, input gradient); PP's ZeRO-1 keeps the gathered stage and
    # both MXFP8 orientations alive, quantizing the stage's 93 / P layers once
    # per step (summed over all layers that is 2 / P per layer).
    w_quant = 3 * mb if P == 1 else 2 / P
    experts_local = 896 // ep * k_routed
    dense_q = (69 * lin_q(_KDA_LIN) + 24 * lin_q(_MLA_LIN)) * DENSE_MXFP8
    out["MXFP8 quantize"] = dense_q + MX_BYTES * experts_local * 92 * w_quant * expert_mxfp8
    if split_weights:
        out["MXFP8 quantize (weights)"] = ((69 * lin_q(_KDA_LIN, True) + 24 * lin_q(_MLA_LIN, True)) * DENSE_MXFP8
                                           + MX_BYTES * experts_local * 92 * w_quant * expert_mxfp8)
    d = {r["key"]: r["numel"] for r in _data()["scopes"]["kda_layer"]["stored"]}
    dm = {r["key"]: r["numel"] for r in _data()["scopes"]["mla_layer"]["stored"]}
    dense = (69 * (d["attention"] + d["shared"] + d["others"]) + 24 * (dm["attention"] + dm["shared"] + dm["others"])) / 93
    gathered = dense / tp + experts_local
    # ZeRO-3: every layer, every microbatch; PP ZeRO-1: the stage's 93 / P layers once per step
    copies = mb * layers if P == 1 else layers / P
    out["FSDP buffer copies"] = (2 * 4 + 6) * gathered * copies
    norm_el = 69 * (2 * D + 12288 / tp) + 24 * (2 * D + 1536 + 512)
    out["Norms (RMSNorm, L2)"] = 14 * norm_el * tok
    out["Other elementwise, casts, reductions"] = (32 * D + 22 * 12288 / tp + 22 * 6144 / tp) * layers * tok
    return out


def nonmatmul_table() -> str:
    """Bytes model vs measured kernel time, FSDP 512 + EP 64 at 8K."""
    model = nonmatmul_bytes("8K")
    meas = _bench("step_graphs_fsdp_ep64_8k.json")["fwd_bwd_by_class_s"]
    head = ["Class", "Estimated bytes per GPU per step", "Ideal time at 8 TB/s", "Measured time",
            "Achieved bandwidth (estimated)"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    tb = tm = 0.0
    for k, b in sorted(model.items(), key=lambda kv: -meas.get(kv[0], 0)):
        m = meas.get(k, 0.0)
        tb, tm = tb + b, tm + m
        out.append(f"<tr><td>{k}</td><td>{b / 1e12:.1f} TB</td><td>{b / HBM_BW:.2f} s</td><td>{m:.2f} s</td>"
                   f"<td>{b / HBM_BW / m:.0%}</td></tr>")
    out.append(f"<tr><td><b>Total</b></td><td>{tb / 1e12:.1f} TB</td><td><b>{tb / HBM_BW:.2f} s</b></td>"
               f"<td><b>{tm:.2f} s</b></td><td>{tb / HBM_BW / tm:.0%}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


# KDA under TP: measured speedup over TP 1 (heads split, but the chunk kernel
# does not scale linearly), per context
KDA_TP_SPEEDUP = {"8K": {1: 1.0, 2: 1.7, 4: 1.7}, "32K": {1: 1.0, 2: 1.8, 4: 3.0}}
KDA_FWD_BWD_US, KDA_FWD_US = 0.40, 0.05  # per token per layer at TP 1, CP 1


def _interp(table: dict, x: float) -> float:
    import math
    pts = sorted((float(k), v) for k, v in table.items())
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            f = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + f * (y1 - y0)


def step_compute(ctx: str = "8K", P: int = 1, cp: int = 1, tp: int = 1, *, shape_aware: bool = True) -> dict:
    """Compute and memory-bound time per GPU per step (no communication, no
    bubble), by op, averaged over the stages. The ideal roofline runs every
    MXFP8 matmul at 60% and every bf16 one at 70%, and memory-bound work at
    8 TB/s. The practical roofline (shape_aware) uses the measured efficiency at
    the config's own shapes and the measured bandwidth of each memory-bound
    class."""
    d = _data()
    S = d["symbols"]
    T = 8192 if ctx == "8K" else 32768
    ep = 512 // P if P > 1 else 64
    tok = d["global_tokens"] / 512
    mb = tok * cp * tp / T  # microbatches per GPU per step, per layer it holds x P / P
    ops = {o[0]: o for o in roofline_ops(ctx)}
    out = {}
    if not shape_aware:
        for name in ("Dense linears", "Routed experts", "LM head", "MLA attention", "KDA attention"):
            out[name] = ops[name][3]
        if tp > 1:  # KDA's measured time still scales with heads per GPU at the uniform rate
            out["KDA attention"] = ops["KDA attention"][3]
        nm = nonmatmul_bytes(ctx, cp=cp, tp=tp, ep=ep, P=P)
        out["Non-matmul (memory-bound)"] = sum(nm.values()) / HBM_BW
        return out
    # dense GEMMs: measured time per layer per microbatch at (T / CP rows, TP split);
    # forward runs twice (remat)
    dense = _bench("shapes_GB300.json")[DENSE_SHAPES]["data"][f"m{T // cp}_tp{tp}"]
    per_layer = {sc: sum(2 * g["fwd"] + g["dgrad"] + g["wgrad"] for g in dense[sc]["gemms"]) * 1e-3
                 for sc in ("kda_layer", "mla_layer")}
    layer0_ffn = 8 * tok * 3 * S["D"] * S["F"] / (dense["kda_layer"]["eff"] * DENSE_PEAK)
    out["Dense linears"] = (69 * per_layer["kda_layer"] + 24 * per_layer["mla_layer"]) * mb + layer0_ffn
    # routed experts: DistMoE's kernels at the config's local tokens
    out["Routed experts"] = ops["Routed experts"][1] / (distmoe_eff(T // (cp * tp), ep) * MXFP8_PEAK)
    # LM head (bf16): measured at T / CP rows
    lm = _bench("kernels_GB300_summary.json")["linears"]["bf16"]["lm_head"]
    out["LM head"] = ops["LM head"][1] / (_interp(lm, T // cp) * 1e12)
    # MLA: forward twice (remat) + backward, at the heads per GPU
    fa = _bench("mla_fa4_GB300.json")["mla_fa4"]["data"]["heads"][str(T)][str(96 // (cp * tp))]
    f = ops["MLA attention"][1] / 4.5  # one forward
    out["MLA attention"] = f / (fa["fwd_tflops"] * 1e12) + 3.5 * f / (fa["fwd_bwd_tflops"] * 1e12)
    # KDA: measured time per token, x the measured CP slowdown, / the TP speedup
    cpf = {"fwd_bwd": 1.0, "fwd": 1.0}
    if cp > 1:
        rows = {r["seq_len"]: r for r in _bench(f"kda_cp_GB300_cp{cp}.json")["results"]}
        r = rows[T]
        cpf = {k: r[f"cp_{k}_ms"] / r[f"local_{k}_ms"] for k in cpf}
    kda_tok = tok * tp  # tokens per GPU per step through its (1 / TP of the) heads
    out["KDA attention"] = (69 * kda_tok * (KDA_FWD_BWD_US * cpf["fwd_bwd"] + KDA_FWD_US * cpf["fwd"]) * 1e-6
                            / KDA_TP_SPEEDUP[ctx][tp])
    # memory-bound: bytes at each class's measured bandwidth (FSDP 512 + EP 64, 8K run)
    meas = _bench("step_graphs_fsdp_ep64_8k.json")["fwd_bwd_by_class_s"]
    ref = nonmatmul_bytes("8K")
    nm = nonmatmul_bytes(ctx, cp=cp, tp=tp, ep=ep, P=P)
    out["Non-matmul (memory-bound)"] = sum(b / HBM_BW / (ref[k] / HBM_BW / meas[k]) for k, b in nm.items())
    return out


_HATCH = (' style="color: var(--muted); background: repeating-linear-gradient(135deg, '
          'rgba(137, 135, 129, 0.10) 0 6px, rgba(137, 135, 129, 0.22) 6px 8px)"')


def compute_table(ctx: str = "8K", P: int = 1) -> str:
    """Compute + memory-bound time per GPU per step (no communication) for one
    family of configs: the ideal and practical rooflines and the practical one's
    parts. Slowdowns against the family's no-split row are red (darker for
    larger), on the total and on the parts that drive it; configs that do not
    fit in memory are hatched."""
    if P == 1:
        label, mem = "FSDP 512, EP 64", lambda cp, tp: fsdp_ep_parts(ctx, cp, tp)
    else:
        label, mem = f"PP {P}, EP {512 // P}", lambda cp, tp: pp_parts(ctx, P, 1, cp, tp)
    head = ["Config", "Ideal roofline", "Practical roofline", "Dense linears", "Routed experts",
            "Attention (MLA + KDA)", "Memory-bound"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    base = base_parts = None
    for cp, tp in SPLITS:
        fits = mem(cp, tp)["total"] <= GB200_BYTES
        u = sum(step_compute(ctx, P, cp, tp, shape_aware=False).values())
        r = step_compute(ctx, P, cp, tp)
        tot = sum(r.values())
        parts = [r["Dense linears"], r["Routed experts"], r["MLA attention"] + r["KDA attention"],
                 r["Non-matmul (memory-bound)"]]
        if base is None:
            base, base_parts = tot, parts
        delta = tot - base

        def red(d):
            # red only for slowdowns, darker with the slowdown's share of the
            # no-split total
            return "" if d < 0.05 else f' style="background: rgba(227, 73, 72, {min(0.45, 0.06 + 0.8 * d / base):.2f})"'

        sh = red(delta)
        diff = "" if abs(delta) < 0.05 else f'<span class="sub">{delta:+.1f} s vs CP 1, TP 1</span>'
        row = "" if fits else _HATCH
        sub = "" if fits else '<span class="sub">does not fit in memory</span>'
        cells = ""
        for x, x0 in zip(parts, base_parts):
            # color the parts that drive the total's growth: at least 0.2 s and a
            # quarter of it
            d = x - x0
            main = d >= max(0.2, 0.25 * delta)
            cells += f"<td{red(d) if main and fits else ''}>{x:.1f} s" + (
                f'<span class="sub">{d:+.1f} s</span>' if main else "") + "</td>"
        out.append(f"<tr{row}><td>{label}, CP {cp}, TP {tp}{sub}</td><td>{u:.1f} s</td>"
                   f"<td{sh if fits else ''}><b>{tot:.1f} s</b>{diff}</td>{cells}</tr>")
    out.append("</table>")
    return "\n".join(out)


def _v(name: str, sub: str) -> str:
    """A math variable with a subscript, e.g. t_comm."""
    return f'<i>{name}</i><sub>{sub}</sub>'


def overlap_formulas(compute_sd: float = 1.3, comm_sd: float = 1.6) -> str:
    """The overlap model as formulas: the shared overlap time, then the two
    penalties, at the given compute and collective slowdowns."""
    tc, tp, to = _v("t", "comm"), _v("t", "compute"), _v("T", "overlap")
    a, k = f"{compute_sd:g}", f"{comm_sd:g}"
    rows = [
        ("def", to, f"min({k} &times; {tc}, {a} &times; {tp})",
         "wall time the two run together, both slowed down"),
        ("pen", "Compute penalty", f"{to} &minus; {to} / {a}",
         "the extra time the overlapped compute took"),
        ("pen", "Collective penalty", f"max(0, {tc} &minus; {to} / {k}) &minus; max(0, {tc} &minus; {tp})",
         "what is left of the collective after the compute ends, beyond what would be exposed anyway"),
    ]
    out = ['<div class="formulas">']
    for kind, lhs, rhs, note in rows:
        out.append(f'<div class="f-row f-{kind}"><span class="f-lhs">{lhs}</span><span class="f-eq">=</span>'
                   f'<span class="f-rhs">{rhs}</span><span class="f-note">{note}</span></div>')
    out.append("</div>")
    return "".join(out)


def hop_formulas() -> str:
    """A PP hop across InfiniBand as formulas: always exposed, plus the compute
    slowdown on both GPUs while it runs (the overlap formulas at 1.1x)."""
    import kimi_k3_step_time as st

    th, tp, to = _v("t", "hop"), _v("t", "compute"), _v("T", "overlap")
    nic, eff = st.IB_GBPS, st.IB_P2P_EFF
    rows = [
        ("def", th, f"bytes / ({eff:g} &times; {nic:.0f} GB/s) = bytes / {eff * nic:.0f} GB/s",
         "a hop's time: one GB200 NIC at 90% of its line rate, as measured"),
        ("def", to, f"min({th}, 1.1 &times; {tp})", "the hop's time, whenever the GPU has compute to run"),
        ("pen", "Compute penalty", f"{to} &minus; {to} / 1.1", "on the sending and the receiving GPU alike"),
    ]
    out = ['<div class="formulas">']
    for kind, lhs, rhs, note in rows:
        out.append(f'<div class="f-row f-{kind}"><span class="f-lhs">{lhs}</span><span class="f-eq">=</span>'
                   f'<span class="f-rhs">{rhs}</span><span class="f-note">{note}</span></div>')
    out.append("</div>")
    return "".join(out)


def fsdp_ib_formula() -> str:
    """Time of an FSDP collective across NVLink domains: a ring per channel, one
    NIC per ring at each domain boundary, plus the ring's sequential steps."""
    import kimi_k3_step_time as st

    tc, n, k = _v("t", "comm"), "<i>n</i>", "<i>k</i>"
    rows = [
        ("def", n, "number of GPUs in the FSDP group", "512 / TP for dense weights, 8 for an expert shard"),
        ("def", k, "number of GPUs of the FSDP group in each NVL domain", "so the NICs it has there: 64 / TP for dense weights, 1 for an expert shard"),
        ("def", "<i>eff</i>", f"{st.IB_COLL_EFF:g} for <i>k</i> &gt; 1, {st.IB_COLL_EFF_ONE_NIC:g} for <i>k</i> = 1",
         "the fraction of a NIC's line rate the ring reaches, as measured"),
        ("def", _v("t", "setup"), f"({n} &minus; 1) &times; {st.RING_STEP_S * 1e6:.0f} &micro;s",
         "setup cost: the ring's n &minus; 1 sequential steps; we measured 2.2 &micro;s per step over NVLink "
         "and 6.5 &micro;s across InfiniBand, and a ring over 8 NVL domains is mostly NVLink steps"),
        ("pen", tc, f"bytes &times; ({n} &minus; 1) / {n} / min(NVLink, {k} &times; {st.IB_GBPS:.0f} GB/s &times; <i>eff</i>) "
                    f"+ {_v('t', 'setup')}", "the transfer, plus the setup cost"),
    ]
    out = ['<div class="formulas">']
    for kind, lhs, rhs, note in rows:
        out.append(f'<div class="f-row f-{kind}"><span class="f-lhs">{lhs}</span><span class="f-eq">=</span>'
                   f'<span class="f-rhs">{rhs}</span><span class="f-note">{note}</span></div>')
    out.append("</div>")
    return "".join(out)


def ib_rule_formula() -> str:
    """The NIC rule: bandwidth of a ring across NVLink domains."""
    return ('<div class="formulas"><div class="f-row f-pen"><span class="f-lhs">Bandwidth</span>'
            '<span class="f-eq">=</span><span class="f-rhs">min(NVLink, <i>k</i> &times; one NIC)</span>'
            '<span class="f-note"><i>k</i>: number of GPUs of the FSDP group in each NVL domain</span></div></div>')


def _pp_schedule(P: int, M: int, F: float, B: float, h: float, v: int = 1):
    """Simulate 1F1B (v = 1) or interleaved 1F1B (v chunks per rank) on P
    ranks, in PyTorch's op order: virtual stage q runs on rank q % P; rank r
    runs (v - 1) P + 2 (P - 1 - r) warmup forwards when interleaved (P - 1 - r
    otherwise), then alternates forward / backward. Receives are posted early,
    so a hop moves as soon as it is sent. Returns
    ops (rank, kind, microbatch, chunk, start, end, wait) and hops (src rank,
    dst rank, kind, microbatch, chunk of the receiver, sent, arrived, used)."""
    Q = P * v

    def nth(k, back):
        mb = (k // Q) * P + k % P if v > 1 else k
        c = (k // P) % v
        return mb, (v - 1 - c if back else c)

    order = []
    for r in range(P):
        n = M * v
        warm = min(n, (v - 1) * P + 2 * (P - 1 - r)) if v > 1 else min(n, P - 1 - r)
        ops = [("F",) + nth(k, False) for k in range(warm)]
        f, b = warm, 0
        while b < n:
            if f < n:
                ops.append(("F",) + nth(f, False)); f += 1
            ops.append(("B",) + nth(b, True)); b += 1
        order.append(ops)
    end, free, idx, ops_out, hops = {}, [0.0] * P, [0] * P, [], []
    done = 0
    while done < P * M * v * 2:
        progressed = False
        for r in range(P):
            if idx[r] >= len(order[r]):
                continue
            kind, mb, c = order[r][idx[r]]
            q = c * P + r
            if kind == "F":
                dep = None if q == 0 else ("F", mb, q - 1)
            else:
                dep = ("F", mb, q) if q == Q - 1 else ("B", mb, q + 1)
            if dep is not None and dep not in end:
                continue
            remote = dep is not None and dep[2] % P != r
            ready = 0.0 if dep is None else end[dep] + (h if remote else 0.0)
            start = max(free[r], ready)
            dur = (F if kind == "F" else B) / v
            end[(kind, mb, q)] = start + dur
            ops_out.append((r, kind, mb, c, start, start + dur, max(0.0, ready - free[r]) if dep else 0.0))
            if remote:
                hops.append((dep[2] % P, r, kind, mb, c, end[dep], end[dep] + h, start))
            free[r] = start + dur
            idx[r] += 1
            done += 1
            progressed = True
        if not progressed:
            raise RuntimeError("schedule deadlock")
    return ops_out, hops


def pp_hop_timeline(v: int = 1) -> str:
    """Two-stage 1F1B timeline, one color per microbatch: forwards solid,
    backwards light, hops as arrows in their microbatch's color, and the time a
    stage waits for a hop in flight as a striped box in that color."""
    P, F, B, h = 4, 1.0, 2.0, 0.5
    M = 8
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#eda100", "#e34948", "#008300"]
    ops, hops = _pp_schedule(P, M, F, B, h, v)
    tmax = max(o[5] for o in ops)
    lab, lane, gap = 80, 30, 34
    top = 78 if v == 1 else 30  # room for the callout above stage 0
    # interleaved: a fixed scale wide enough to read each chunk, scrolled
    w = 960 if v == 1 else int(lab + 20 + 56 * tmax)
    xs = lambda t: lab + (w - lab - 20) * t / tmax
    ys = lambda st: top + st * (lane + gap)
    hgt = top + P * (lane + gap) + 30
    size = 'width="100%"' if v == 1 else f'width="{w}" style="max-width: none"'
    out = [f'<svg viewBox="0 0 {w} {hgt}" {size} role="img" aria-label="pipeline timeline">',
           f'<defs><pattern id="pph-idle-{v}" width="6" height="6" patternUnits="userSpaceOnUse" '
           'patternTransform="rotate(45)"><rect width="6" height="6" fill="#f3f2ee"/>'
           '<rect width="2" height="6" fill="#d9d7cf"/></pattern>']
    for i, c in enumerate(colors):
        out.append(f'<marker id="pph-arr-{v}-{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
                   f'markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
        out.append(f'<pattern id="pph-wait-{v}-{i}" width="5" height="5" patternUnits="userSpaceOnUse" '
                   f'patternTransform="rotate(45)"><rect width="5" height="5" fill="#ffffff"/>'
                   f'<rect width="2.2" height="5" fill="{c}"/></pattern>')
    out.append("</defs>")
    for t in range(0, int(tmax) + 1):
        out.append(f'<line x1="{xs(t):.1f}" x2="{xs(t):.1f}" y1="{top - 8}" y2="{hgt - 26}" stroke="{GRID}"/>')
        out.append(_text(xs(t), hgt - 10, f"{t}", 11, MUTED, "middle"))
    for st in range(P):
        y = ys(st)
        out.append(_text(lab - 10, y + lane / 2 + 4, f"Stage {st}", 13, INK, "end", 600))
        out.append(f'<rect x="{lab}" y="{y}" width="{w - lab - 20}" height="{lane}" fill="url(#pph-idle-{v})" rx="3"/>')
    def tint(c, f=0.35):
        # an opaque pale version of c: f of the color over white
        r, g, b_ = (int(c[k:k + 2], 16) for k in (1, 3, 5))
        return "#" + "".join(f"{round(255 - (255 - v) * f):02x}" for v in (r, g, b_))

    for st, kind, i, ch, a0, b0, _ in ops:
        y, c = ys(st), colors[i % len(colors)]
        fill = c if kind == "F" else tint(c)
        out.append(f'<rect x="{xs(a0):.1f}" y="{y}" width="{xs(b0) - xs(a0):.1f}" height="{lane}" fill="{fill}" '
                   f'stroke="{c}" stroke-width="1.5" rx="3"/>')
        ink = "#ffffff" if kind == "F" else INK
        sub = f'<tspan font-size="8" dy="3">{ch}</tspan>' if v > 1 else ""
        fs = 12 if v == 1 else 11
        out.append(f'<text x="{(xs(a0) + xs(b0)) / 2:.1f}" y="{y + lane / 2 + 4}" font-size="{fs}" fill="{ink}" '
                   f'text-anchor="middle" font-weight="600">{kind}{i}{sub}</text>')
    # the part of a stage's idle time spent waiting for a hop in flight: from
    # when it was free (or the hop left) until the hop arrives
    waits = {(st, kind, i, c): (a0, wt) for st, kind, i, c, a0, _, wt in ops}
    for src, dst, kind, i, c, sent, arr, used in hops:
        a0, wt = waits[(dst, kind, i, c)]
        lo, hi = max(a0 - wt, sent), arr
        if wt > 1e-9 and hi - lo > 1e-9:
            y = ys(dst)
            out.append(f'<rect x="{xs(lo):.1f}" y="{y + 4}" width="{xs(hi) - xs(lo):.1f}" height="{lane - 8}" '
                       f'fill="url(#pph-wait-{v}-{i % len(colors)})" stroke="{colors[i % len(colors)]}" '
                       f'stroke-width="1.2" rx="2"/>')
    for src, dst, kind, i, ch, sent, arr, used in hops:
        c = colors[i % len(colors)]
        y0 = ys(src) + (lane if dst > src else 0)
        y1 = ys(dst) + (0 if dst > src else lane)
        out.append(f'<line x1="{xs(sent):.1f}" y1="{y0}" x2="{xs(arr):.1f}" y2="{y1}" stroke="{c}" '
                   f'stroke-width="2.5" marker-end="url(#pph-arr-{v}-{i % len(colors)})"/>')
    if v == 1:
        # callout on stage 0's gap between F5 and B2: why it cannot run F6 there
        f5 = next(o for o in ops if o[0] == 0 and o[1] == "F" and o[2] == 5)
        b2 = next(o for o in ops if o[0] == 0 and o[1] == "B" and o[2] == 2)
        g0, g1, y0 = xs(f5[5]), xs(b2[4]), ys(0)
        out.append(f'<rect x="{g0 - 1:.1f}" y="{y0 - 3}" width="{g1 - g0 + 2:.1f}" height="{lane + 6}" fill="none" '
                   f'stroke="{INK}" stroke-width="2" rx="3"/>')
        tx = min(g0 - 260, w - 540)
        out.append(f'<line x1="{tx + 250:.1f}" y1="{y0 - 22}" x2="{(g0 + g1) / 2:.1f}" y2="{y0 - 4}" stroke="{INK}" '
                   f'stroke-width="1.5"/>')
        out.append(_text(tx, y0 - 50, "Stage 0 could run F6 here, but it already holds 4 microbatches,", 12, INK,
                         "start", 600))
        out.append(_text(tx, y0 - 34, "1F1B's limit: it must wait for B2, so for B2's gradient to arrive (the hop).",
                         12, INK, "start", 600))
    out.append("</svg>")
    sw = ('<span style="display: inline-block; width: 12px; height: 12px; border-radius: 2px; '
          'vertical-align: -1px; background: {bg}; border: 1.5px solid {bd}"></span>')
    items = [f'<span>{sw.format(bg=c, bd=c)}{sw.format(bg=tint(c), bd=c)} microbatch {i}</span>'
             for i, c in enumerate(colors[:M])]
    leg = ('<div style="display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; '
           'color: var(--text-secondary); margin: 6px 0">' + "".join(items)
           + '<span>solid: forward (1); light: backward (2)</span>'
           '<span><span style="display: inline-block; width: 12px; height: 12px; border-radius: 2px; '
           'background: repeating-linear-gradient(45deg, #f3f2ee 0 4px, #d9d7cf 4px 6px); vertical-align: -1px">'
           '</span> idle: waiting for the other stage\'s compute</span>'
           '<span><span style="display: inline-block; width: 14px; height: 12px; border-radius: 2px; '
           'border: 1.2px solid #52514e; background: repeating-linear-gradient(45deg, #ffffff 0 2px, '
           '#52514e 2px 4px); vertical-align: -1px"></span> waiting for that microbatch\'s hop</span>'
           '<span>arrows: hops (0.5)</span></div>')
    svg = "".join(out)
    if v > 1:
        svg = f'<div style="overflow-x: auto; padding-bottom: 6px">{svg}</div>'
    return leg + svg


def bubble_formulas() -> str:
    """Pipeline step time with even stages, as four parts per schedule (M
    microbatches, t one stage's forward + backward per microbatch, h one hop;
    from the schedule simulation), with each part's time for the timelines."""
    t, h, M, P, v = "<i>t</i>", "<i>h</i>", "<i>M</i>", "<i>P</i>", "<i>v</i>"
    Pc, Mc, tc, hc, vc = 4, 8, 3.0, 0.5, 2  # the timelines
    rows = [
        ("Work", "every microbatch's forward and backward",
         f"{M} {t}", Mc * tc, f"{M} {t}", Mc * tc),
        ("Warmup and cooldown", "stages idle while the pipeline fills and drains",
         f"({P} &minus; 1) {t}", (Pc - 1) * tc, f"({P} &minus; 1) {t} / {v}", (Pc - 1) * tc / vc),
        ("Hops, steady state", "stages waiting for hops between warmup and cooldown",
         f"2 ({P} &minus; 1) / {P} &times; {M} {h}", 2 * (Pc - 1) / Pc * Mc * hc, "0", 0.0),
        ("Hops, fill and drain", "hops while the pipeline fills and drains",
         f"2 ({P} &minus; 2) {h}", 2 * (Pc - 2) * hc, f"2 ({P}{v} &minus; 1) {h}", 2 * (Pc * vc - 1) * hc),
    ]
    head = ["Part of the step", "Plain 1F1B", "Time", "Interleaved 1F1B", "Time"]
    out = ['<table class="dtable bubble-table"><tr>' + "".join(f"<th>{x}</th>" for x in head) + "</tr>"]
    tot1 = tot2 = 0.0
    for name, what, f1, n1, f2, n2 in rows:
        tot1, tot2 = tot1 + n1, tot2 + n2
        out.append(f'<tr><td><b>{name}</b><br><span class="sub">{what}</span></td>'
                   f'<td class="f-rhs">{f1}</td><td>{n1:g}</td><td class="f-rhs">{f2}</td><td>{n2:g}</td></tr>')
    out.append(f'<tr><td><b>Step</b></td><td>the sum</td><td><b>{tot1:g}</b></td><td>the sum</td>'
               f'<td><b>{tot2:g}</b></td></tr>')
    out.append(f'<caption>Time for the timelines above: <i>P</i> = {Pc}, <i>M</i> = {Mc}, <i>t</i> = {tc:g}, '
               f'<i>h</i> = {hc:g}, <i>v</i> = {vc}</caption></table>')
    return "".join(out)


def transport_matrix() -> str:
    """Schedules x transports: which combinations TorchTitan runs today, which
    this study adds, and which drop out."""
    titan = ' style="background: rgba(0, 131, 0, 0.12)"'
    new = ' style="background: rgba(42, 120, 214, 0.12)"'
    cells = {
        ("1F1B", "chain"): (titan, "<b>In TorchTitan</b>", "Ranked"),
        ("1F1B", "cache"): (_HATCH, "<b>Not applicable</b>",
                            "A rank sees each microbatch once: nothing to cache, so it is chain"),
        ("1F1B", "direct"): (new, "<b>New in this study</b>", "Against chain, below"),
        ("Interleaved", "chain"): (_HATCH, "<b>Always worse than cache</b>",
                                   "Resends blocks the rank already holds"),
        ("Interleaved", "cache"): (titan, "<b>In TorchTitan</b>", "Ranked"),
        ("Interleaved", "direct"): (new, "<b>New in this study</b>", "Against cache, below"),
    }
    out = ['<table class="dtable tmatrix"><tr><th></th><th>chain</th><th>cache</th><th>direct</th></tr>']
    for sched in ("1F1B", "Interleaved"):
        row = "".join(f"<td{st_}>{head}<br>{note}</td>"
                      for st_, head, note in (cells[(sched, tr)] for tr in ("chain", "cache", "direct")))
        out.append(f"<tr><th>{sched if sched == '1F1B' else 'Interleaved 1F1B'}</th>{row}</tr>")
    out.append("</table>")
    return "\n".join(out)


def transport_diagram(names: tuple = ("chain", "direct", "cache")) -> str:
    """Hop transports (chain, direct, cache, or a subset) on four stages that
    each finish one attention-residual block (stage 0 also holds the embedding
    E): what every forward hop carries, and for direct the blocks sent straight
    to later stages."""
    col = {"h": INK, "E": "#898781", "A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
    lab, sw, gap, top, rowh = 120, 96, 120, 20, 112
    cache_h = 236  # the cache row: two rows of stages, one column per rank
    w = lab + 4 * sw + 3 * gap + 10
    xs = [lab + i * (sw + gap) for i in range(4)]
    extra = 92  # room for the direct row's arcs
    arr = "trd-arr-" + "-".join(names)  # one marker per variant: two diagrams can share a page
    height = (top + 22 + sum(cache_h if n == "cache" else rowh for n in names)
              + (extra - 12 if "direct" in names else 0) - 20)
    out = [f'<svg viewBox="0 0 {w} {height}" width="100%" role="img" '
           f'aria-label="{", ".join(names)} transports">',
           f'<defs><marker id="{arr}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker></defs>']

    def chips(cx, cy, items):
        n, cw = len(items), 18
        x0 = cx - (n * cw - 2) / 2
        for k, it in enumerate(items):
            x = x0 + k * cw
            if it == "h":
                out.append(f'<rect x="{x:.1f}" y="{cy - 8}" width="16" height="16" rx="3" fill="#ffffff" '
                           f'stroke="{INK}" stroke-width="1.5"/>')
                out.append(_text(x + 8, cy + 4, "h", 11, INK, "middle", 700))
            else:
                out.append(f'<rect x="{x:.1f}" y="{cy - 8}" width="16" height="16" rx="3" fill="{col[it]}"/>')
                out.append(_text(x + 8, cy + 4, it, 10, "#ffffff", "middle", 700))

    rows = [
        ("chain", "plain 1F1B", [None] * 4, [["h", "E", "A"], ["h", "E", "A", "B"], ["h", "E", "A", "B", "C"]], False),
        ("direct", "plain 1F1B", [None] * 4, [["h"], ["h"], ["h"]], True),
        ("cache", "interleaved", ["rank 0", "rank 1", "rank 0", "rank 1"],
         [["h", "E", "A"], ["h", "B"], ["h", "C"]], False),
    ]
    rows = [r for r in rows if r[0] in names]
    made = [["E", "A"], ["B"], ["C"], ["D"]]
    y_next = top + 22
    def stage_box(x, y, i):
        out.append(f'<rect x="{x}" y="{y}" width="{sw}" height="40" rx="6" fill="{SURFACE}" '
                   f'stroke="#52514e" stroke-width="1.2"/>')
        out.append(_text(x + sw / 2, y + 16, f"stage {i}", 11, INK2, "middle", 600))
        if i < 3:
            out.append(_text(x + sw / 2, y + 32, "finishes " + " ".join(made[i]), 10, MUTED, "middle"))

    for name, sched, ranks, hops, arcs in rows:
        if name == "cache":
            # two ranks, two chunks each: stages 0 and 2 on rank 0 (left
            # column), 1 and 3 on rank 1 (right column)
            y = y_next + 16
            y_next += cache_h
            out.append(_text(10, y + 20, name, 15, INK, "start", 700))
            out.append(_text(10, y + 38, sched, 11, MUTED, "start"))
            y2 = y + 128
            for r in range(2):
                out.append(_text(xs[r] + sw / 2, y - 8, f"rank {r}", 11, INK2, "middle", 700))
            for i, (x, yy) in enumerate(((xs[0], y), (xs[1], y), (xs[0], y2), (xs[1], y2))):
                stage_box(x, yy, i)
            # hops: 0 -> 1 across, 1 -> 2 back down to rank 0, 2 -> 3 across
            for (xa, ya, xb, yb), items in (((xs[0] + sw + 4, y + 20, xs[1] - 4, y + 20), hops[0]),
                                            ((xs[1] - 4, y + 36, xs[0] + sw + 4, y2 + 4), hops[1]),
                                            ((xs[0] + sw + 4, y2 + 20, xs[1] - 4, y2 + 20), hops[2])):
                out.append(f'<line x1="{xa}" y1="{ya}" x2="{xb}" y2="{yb}" stroke="#52514e" '
                           f'stroke-width="1.5" marker-end="url(#{arr})"/>')
                cy = (ya + yb) / 2 - (14 if ya == yb else 0)
                chips((xa + xb) / 2 + (0 if ya == yb else 32), cy, items)
            # what each rank keeps from its first chunk, reused by its second
            for r, keep in ((0, ["E", "A"]), (1, ["E", "A", "B"])):
                cx = xs[r] + sw / 2
                bw = sw - 8
                by = y + 58
                out.append(f'<line x1="{cx}" y1="{y + 40}" x2="{cx}" y2="{by}" stroke="#52514e" '
                           f'stroke-width="1.3" stroke-dasharray="4 3"/>')
                out.append(f'<rect x="{cx - bw / 2}" y="{by}" width="{bw}" height="40" rx="5" fill="#ffffff" '
                           f'stroke="#52514e" stroke-width="1.1" stroke-dasharray="4 3"/>')
                out.append(_text(cx, by + 13, "keeps", 10, INK2, "middle", 600))
                chips(cx, by + 27, keep)
                out.append(f'<line x1="{cx}" y1="{by + 40}" x2="{cx}" y2="{y2 - 3}" stroke="#52514e" '
                           f'stroke-width="1.3" stroke-dasharray="4 3" marker-end="url(#{arr})"/>')
            continue
        # the direct row needs room above for its arcs, so it and the rows
        # below it shift down by that much
        y_next += extra - 12 if arcs else 0
        y, y_next = y_next, y_next + rowh
        out.append(_text(10, y + 20, name, 15, INK, "start", 700))
        out.append(_text(10, y + 38, sched, 11, MUTED, "start"))
        for i, x in enumerate(xs):
            out.append(f'<rect x="{x}" y="{y}" width="{sw}" height="40" rx="6" fill="{SURFACE}" '
                       f'stroke="#52514e" stroke-width="1.2"/>')
            out.append(_text(x + sw / 2, y + 16, f"stage {i}", 11, INK2, "middle", 600))
            if i < 3:
                blk = made[i]
                out.append(_text(x + sw / 2, y + 32, "finishes " + " ".join(blk), 10, MUTED, "middle"))
            if ranks[i]:
                out.append(_text(x + sw / 2, y + 54, ranks[i], 10, MUTED, "middle"))
        for i, items in enumerate(hops):
            x0, x1 = xs[i] + sw, xs[i + 1]
            out.append(f'<line x1="{x0 + 4}" y1="{y + 20}" x2="{x1 - 4}" y2="{y + 20}" stroke="#52514e" '
                       f'stroke-width="1.5" marker-end="url(#{arr})"/>')
            chips((x0 + x1) / 2, y + 6, items)
        if arcs:
            # one arc per pair of stages, carrying the blocks the later one uses
            for src, dst, blks in ((0, 3, ["E", "A"]), (0, 2, ["E", "A"]), (1, 3, ["B"]), (0, 1, ["E", "A"]),
                                   (1, 2, ["B"]), (2, 3, ["C"])):
                xa, xb = xs[src] + sw / 2 + 6, xs[dst] + sw / 2 - 6
                hgt = 22 + 26 * (dst - src) - (6 if src == 1 else 0)
                c = col[blks[-1]]
                out.append(f'<path d="M{xa:.1f},{y} C{xa:.1f},{y - hgt * 1.33:.1f} {xb:.1f},{y - hgt * 1.33:.1f} '
                           f'{xb:.1f},{y}" fill="none" stroke="{c}" stroke-width="1.6" marker-end="url(#{arr})"/>')
                chips((xa + xb) / 2, y - hgt, blks)
    out.append("</svg>")
    note = ('<p class="sub" style="color: var(--muted); font-size: 13px; margin: 4px 0 0">Above each '
            'forward hop: what it carries; <b>h</b> is the hidden state, colored squares are attention-residual blocks '
            '(E: the embedding). '
            + ('For direct, the arcs above carry each block straight to every later stage. ' if "direct" in names
               else "")
            + 'The backward mirrors each, with gradients.</p>')
    return "".join(out) + note


def overlap_examples() -> str:
    """Two worked examples of the overlap formulas (NVLink slowdowns)."""
    tc, tp, to = _v("t", "comm"), _v("t", "compute"), _v("T", "overlap")
    head = [tc, tp, to, "Compute penalty", "Collective penalty", "Total"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for c, w in ((1.0, 10.0), (10.0, 10.0)):
        t = min(1.6 * c, 1.3 * w)
        cpen = t - t / 1.3
        qpen = max(0.0, c - t / 1.6) - max(0.0, c - w)
        out.append(f"<tr><td>{c:g} ms</td><td>{w:g} ms</td><td>{t:.2f} ms</td><td>{cpen:.2f} ms</td>"
                   f"<td>{qpen:.2f} ms</td><td><b>{w + cpen + qpen + max(0.0, c - w):.2f} ms</b></td></tr>")
    out.append("</table>")
    return "\n".join(out)


def overlap_table() -> str:
    """1 GB collectives on 16 GPUs in one NVLink domain: bus bandwidth alone,
    NCCL's SMs, the slowdown of the compute they overlap and their own."""
    bw = {r["op"]: r["busbw_GBps"] for r in _bench("nvl_collectives_GB300_4hosts.json")["rows"]
          if r["tag"] == "nvl_4hosts" and r["bytes"] == 1 << 30}
    ov = {(r["compute"], r["collective"]): r for r in _bench("overlap_GB300_nvl_4hosts.json")["rows"]
          if r["size_mb"] == 1024}
    head = ["1 GB collective", "Bus bandwidth alone", "NCCL SMs", "GEMM slowdown", "Memory-bound slowdown",
            "Collective slowdown"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for op, key, label in (("allgather", "all_gather", "All-gather"), ("reducescatter", "reduce_scatter", "Reduce-scatter"),
                           ("alltoall", "all_to_all", "All-to-all")):
        g, m = ov[("bf16_gemm", key)], ov[("memory_bound", key)]
        lo, hi = sorted((g["comm_slowdown"], m["comm_slowdown"]))
        out.append(f"<tr><td>{label}</td><td>{bw[op]:.0f} GB/s</td><td>{g['nccl_blocks']}</td>"
                   f"<td>{g['compute_slowdown']:.2f}x</td><td>{m['compute_slowdown']:.2f}x</td>"
                   f"<td>{f'{lo:.1f}x' if f'{lo:.1f}' == f'{hi:.1f}' else f'{lo:.1f}-{hi:.1f}x'}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


_MATMUL_CLASSES = ("Dense GEMM MXFP8", "Dense GEMM bf16", "DistMoE grouped GEMM (+fused dispatch/combine)", "MLA", "KDA")


def sweep_table() -> str:
    """Practical roofline vs measured fake-PG GPU time for every FSDP 512 + EP 64
    config that runs on one GB300 (8K: all splits; 32K: CP 2 / 4, TP 2 / 4;
    32K without a split needs more than the GPU's memory)."""
    rows = [("8K", 1, 1), ("8K", 2, 1), ("8K", 4, 1), ("8K", 1, 2), ("8K", 1, 4),
            ("32K", 2, 1), ("32K", 4, 1), ("32K", 1, 2), ("32K", 1, 4)]
    head = ["FSDP 512, EP 64", "Practical roofline", "Measured (fake process group)", "Difference"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for ctx, cp, tp in rows:
        path = _Path(__file__).resolve().parent / "bench" / f"step_sweep_{ctx.lower()}_cp{cp}_tp{tp}.json"
        if not path.exists():
            continue
        meas = _json.loads(path.read_text())["fwd_bwd_busy_s"]
        model = sum(step_compute(ctx, 1, cp, tp).values())
        out.append(f"<tr><td>{ctx}, CP {cp}, TP {tp}</td><td>{model:.1f} s</td><td>{meas:.1f} s</td>"
                   f"<td>{model / meas - 1:+.0%}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


NIC_GBPS_GB300 = 100.0  # 800 Gb/s XDR InfiniBand per GPU on the GB300 cluster


def ib_bw_table() -> str:
    """Bus bandwidth across 4 GB300 hosts (16 GPUs) over InfiniBand, per GPU,
    and as a multiple of one GPU's NIC."""
    d = _bench("ib_collectives_GB300_4hosts.json")
    rows = {(r["op"], r["bytes"]): r["busbw_GBps"] for r in d["rows"]}
    names = {"allgather": "All-gather", "reducescatter": "Reduce-scatter", "alltoall": "All-to-all",
             "sendrecv": "Send / recv (rank i to rank i + 8)"}
    sizes = (256 << 20, 1 << 30, 2 << 30)
    head = ["Collective"] + [f"{sz >> 20} MB" for sz in sizes] + ["x one NIC (at 2 GB)"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for op, label in names.items():
        cells = "".join(f"<td>{rows[(op, sz)]:.0f} GB/s</td>" for sz in sizes)
        out.append(f"<tr><td>{label}</td>{cells}<td>{rows[(op, sizes[-1])] / NIC_GBPS_GB300:.1f}x</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def xhost_overlap_table() -> str:
    """A 1 GB all-gather overlapping compute: inside an NVLink domain (16 GPUs)
    and across InfiniBand (8 GPUs)."""
    nv = {(r["compute"], r["collective"]): r for r in _bench("overlap_GB300_nvl_4hosts.json")["rows"] if r["size_mb"] == 1024}
    ib = {(r["compute"], r["collective"]): r for r in _bench("overlap_GB300_ib_2hosts.json")["rows"] if r["size_mb"] == 1024}
    head = ["1 GB collective", "NCCL SMs", "GEMM slowdown", "Memory-bound slowdown", "Collective slowdown"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for label, d in (("All-gather, inside an NVLink domain", nv), ("All-gather, across InfiniBand", ib)):
        g, m = d[("bf16_gemm", "all_gather")], d[("memory_bound", "all_gather")]
        out.append(f"<tr><td>{label}</td><td>{g['nccl_blocks']}</td><td>{g['compute_slowdown']:.2f}x</td>"
                   f"<td>{m['compute_slowdown']:.2f}x</td><td>{min(g['comm_slowdown'], m['comm_slowdown']):.1f}-"
                   f"{max(g['comm_slowdown'], m['comm_slowdown']):.1f}x</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def nvl_domain_table() -> str:
    """Bandwidth and overlap of NVLink collectives within one host (4 GPUs,
    nccl-tests) and across hosts inside one NVL72 domain (16 and 32 GPUs)."""
    import kimi_k3_step_time as st

    links = st.Links()
    ov = {4: "overlap_GB300_chdefault.json", 16: "overlap_GB300_nvl_4hosts.json", 32: "overlap_GB300_nvl_8hosts.json"}
    head = ["Scope", "All-gather, 1 GB", "All-to-all, 256 MB", "NCCL SMs", "GEMM slowdown",
            "Memory-bound slowdown", "Collective slowdown"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for w, label in ((4, "1 host, 4 GPUs"), (16, "4 hosts, 16 GPUs"), (32, "8 hosts, 32 GPUs")):
        ag = _interp(links.by_world[w]["allgather"], 1 << 30) / 1e9
        a2a = _interp(links.by_world[w]["alltoall"], 256 << 20) / 1e9
        d = {(r["compute"], r["collective"]): r for r in _bench(ov[w])["rows"] if r["size_mb"] == 1024}
        g, m = d[("bf16_gemm", "all_gather")], d[("memory_bound", "all_gather")]
        out.append(f"<tr><td>{label}</td><td>{ag:.0f} GB/s</td><td>{a2a:.0f} GB/s</td><td>{g['nccl_blocks']}</td>"
                   f"<td>{g['compute_slowdown']:.2f}x</td><td>{m['compute_slowdown']:.2f}x</td>"
                   f"<td>{min(g['comm_slowdown'], m['comm_slowdown']):.1f}-"
                   f"{max(g['comm_slowdown'], m['comm_slowdown']):.1f}x</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def _ib_scale(hosts: int):
    """The InfiniBand scaling run on `hosts` hosts (4 GPUs each), or None."""
    path = _Path(__file__).resolve().parent / "bench" / f"ib_scale_GB300_{hosts}hosts.json"
    return _json.loads(path.read_text()) if path.exists() else None


def fsdp_ib_table() -> str:
    """FSDP's all-gather and reduce-scatter across InfiniBand over every GPU:
    bandwidth at 1 GB, NCCL's channels and the overlap slowdowns, by island
    size (4 or 8 GPUs on NVLink, each GPU with its own NIC) and job size."""
    runs = [(4, 2, _ib_scale(2), 16), (4, 4, _ib_scale(4), 16), (4, 8, _ib_scale(8), 16)]
    p8 = _Path(__file__).resolve().parent / "bench" / "ib_island8_GB300_4hosts.json"
    if p8.exists():
        runs.append((8, 2, _json.loads(p8.read_text()), 32))
    head = ["Islands", "Channels", "All-gather", "Reduce-scatter", "x one NIC", "GEMM slowdown",
            "Memory-bound slowdown", "Collective slowdown"]
    out = ['<table class="dtable"><caption>1 GB collectives over every GPU; bandwidth alone, and the slowdowns '
           'while overlapping compute</caption><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    rng = lambda xs, f: (f"{min(xs):{f}}-{max(xs):{f}}x" if f"{min(xs):{f}}" != f"{max(xs):{f}}"
                         else f"{xs[0]:{f}}x")
    for size, count, d, ch in runs:
        if not d or not d.get("overlap"):
            continue
        bw = {r["op"]: r["busbw_GBps"] for r in d["rows"] if r["bytes"] == 1 << 30}
        ov = {(r["compute"], r["collective"]): r for r in d["overlap"]["rows"] if r["size_mb"] == 1024}
        g = [ov[("bf16_gemm", k)] for k in ("all_gather", "reduce_scatter")]
        m = [ov[("memory_bound", k)] for k in ("all_gather", "reduce_scatter")]
        out.append(f"<tr><td>{count} of {size} GPUs</td><td>{ch}</td><td>{bw['allgather']:.0f} GB/s</td>"
                   f"<td>{bw['reducescatter']:.0f} GB/s</td><td>{bw['allgather'] / NIC_GBPS_GB300:.1f}x</td>"
                   f"<td>{rng([r['compute_slowdown'] for r in g], '.2f')}</td>"
                   f"<td>{rng([r['compute_slowdown'] for r in m], '.2f')}</td>"
                   f"<td>{rng([r['comm_slowdown'] for r in g + m], '.1f')}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def expert_ib_table() -> str:
    """All-gather / reduce-scatter in groups with one or two GPUs per island (as
    an expert shard's FSDP replicas), all groups at once: bandwidth at 1 GB."""
    here = _Path(__file__).resolve().parent / "bench"
    runs = []
    p1 = here / "ib_rail_groups_GB300_8hosts.json"
    if p1.exists():
        rows = [r for r in _json.loads(p1.read_text())["rows"] if r["tag"] == "pxn_off"]
        runs.append(("1 per island (8 islands of 4)", 1, rows))
    p2 = here / "ib_island8_GB300_4hosts.json"
    if p2.exists():
        runs.append(("2 per island (2 islands of 8)", 2, _json.loads(p2.read_text())["rail"]))
    head = ["Group's GPUs", "All-gather", "Reduce-scatter", "x one NIC"]
    out = ['<table class="dtable"><caption>1 GB collectives in groups spread over the islands, every group at '
           'once</caption><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for label, k, rows in runs:
        bw = {r["op"]: r["busbw_GBps"] for r in rows if r["bytes"] == 1 << 30}
        out.append(f"<tr><td>{label}</td><td>{bw['allgather']:.0f} GB/s</td><td>{bw['reducescatter']:.0f} GB/s</td>"
                   f"<td>{min(bw.values()) / NIC_GBPS_GB300:.1f}-{max(bw.values()) / NIC_GBPS_GB300:.1f}x</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def ib_rule_table() -> str:
    """The NIC rule on the GB300 cluster: all-gather / reduce-scatter bandwidth
    at 1 GB for groups with k GPUs in each NVL domain, in units of one NIC."""
    here = _Path(__file__).resolve().parent / "bench"
    pts = []  # (NVL domain size, k, which GPUs, [bandwidths])
    p1 = here / "ib_rail_groups_GB300_8hosts.json"
    if p1.exists():
        rows = [r for r in _json.loads(p1.read_text())["rows"] if r["tag"] == "pxn_off" and r["bytes"] == 1 << 30]
        pts.append((4, 1, "one GPU from each of 8", [r["busbw_GBps"] for r in rows]))
    p8 = here / "ib_island8_GB300_4hosts.json"
    d8 = _json.loads(p8.read_text()) if p8.exists() else None
    if d8:
        pts.append((8, 2, "two GPUs from each of 2", [r["busbw_GBps"] for r in d8["rail"] if r["bytes"] == 1 << 30]))
    bw4 = [r["busbw_GBps"] for h in (2, 4, 8) if _ib_scale(h) for r in _ib_scale(h)["rows"]
           if r["bytes"] == 1 << 30 and r["op"] in ("allgather", "reducescatter")]
    if bw4:
        pts.append((4, 4, "every GPU of 2, 4 or 8", bw4))
    if d8:
        pts.append((8, 8, "every GPU of 2", [r["busbw_GBps"] for r in d8["rows"]
                                             if r["bytes"] == 1 << 30 and r["op"] in ("allgather", "reducescatter")]))
    head = ["NVL domain size", "FSDP group: GPUs of NVL domains", "<i>k</i> (number of GPUs of the FSDP group per NVL domain)",
            "Measured bandwidth", "NICs' worth"]
    out = ['<table class="dtable"><caption>All-gather and reduce-scatter at 1 GB, every FSDP group at once; one NIC is '
           '100 GB/s here</caption><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for size, k, which, xs in pts:
        out.append(f"<tr><td>{size}</td><td>{which}</td><td><b>{k}</b></td><td>{min(xs):.0f}-{max(xs):.0f} GB/s</td>"
                   f"<td><b>{min(xs) / NIC_GBPS_GB300:.1f}-{max(xs) / NIC_GBPS_GB300:.1f}</b></td></tr>")
    out.append("</table>")
    return "\n".join(out)


def gb200_ib_table() -> str:
    """What the NIC rule means for GB200's 64-GPU domains."""
    import kimi_k3_step_time as st

    nvl = 640  # NVLink all-gather bandwidth at 64 GPUs
    one = st.IB_GBPS * st.IB_COLL_EFF_ONE_NIC
    rows = [
        ("FSDP 512: dense weights", "64 / TP", f"NVLink, about {nvl} GB/s", "InfiniBand is not the limit"),
        ("FSDP 512: an expert shard's 8 replicas, one per NVL domain", "1", f"one NIC, about {one:.0f} GB/s",
         f"{nvl / one:.0f}x slower than the dense weights"),
        ("PP: a hop between two stages", "1 (a pair of GPUs)",
         f"one NIC, about {st.IB_GBPS * st.IB_P2P_EFF:.0f} GB/s", "on the critical path"),
        ("PP: FSDP inside a stage", "-", "NVLink", "never leaves the NVL domain"),
    ]
    head = ["Traffic", "<i>k</i>", "Bandwidth", ""]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for t, k, b, note in rows:
        out.append(f"<tr><td>{t}</td><td>{k}</td><td>{b}</td><td>{note}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def pp_ib_table() -> str:
    """PP's hops across InfiniBand: send/recv bandwidth at 1 GB, and the
    compute slowdown on the sending and the receiving GPU of a one-way 1 GB
    transfer (rank i -> rank i + N / 2), by job size."""
    head = ["GPUs (islands of 4)", "Send / recv", "x one NIC", "NCCL SMs per side", "GEMM: sender", "GEMM: receiver",
            "Memory-bound: sender", "Memory-bound: receiver"]
    out = ['<table class="dtable"><caption>1 GB one-way transfers, each pair on a different island</caption><tr>'
           + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for hosts in (2, 4, 8):
        d = _ib_scale(hosts)
        if not d or not d.get("oneway"):
            continue
        bw = {r["op"]: r["busbw_GBps"] for r in d["rows"] if r["bytes"] == 1 << 30}["sendrecv"]
        rows = [r for r in d["oneway"]["rows"] if r["size_mb"] == 1024]

        def mean(compute, role):
            xs = [r["compute_slowdown"] for r in rows if r["compute"] == compute and r["collective"] == role]
            return sum(xs) / len(xs)

        out.append(f"<tr><td>{4 * hosts} ({hosts})</td><td>{bw:.0f} GB/s</td><td>{bw / NIC_GBPS_GB300:.2f}x</td>"
                   f"<td>{rows[0]['nccl_blocks']}</td><td>{mean('bf16_gemm', 'send_only'):.2f}x</td>"
                   f"<td>{mean('bf16_gemm', 'recv_only'):.2f}x</td><td>{mean('memory_bound', 'send_only'):.2f}x</td>"
                   f"<td>{mean('memory_bound', 'recv_only'):.2f}x</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def one_way_table() -> str:
    """Compute slowdown on the sending and the receiving GPU of a 1 GB one-way
    transfer (a PP hop): over NVLink (2 GPUs of a host) and over InfiniBand (rank
    i on one host to rank i on another), averaged over the ranks."""
    nv = _bench("overlap_GB300_oneway.json")["rows"]
    ib = _bench("overlap_GB300_oneway_ib.json")["rows"]

    def mean(rows, compute, role):
        xs = [r for r in rows if r["compute"] == compute and r["collective"] == role and r["size_mb"] == 1024]
        return sum(r["compute_slowdown"] for r in xs) / len(xs), xs[0]["nccl_blocks"]

    head = ["1 GB one-way transfer", "NCCL SMs per side", "GEMM: sender", "GEMM: receiver",
            "Memory-bound: sender", "Memory-bound: receiver"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for label, rows in (("NVLink", nv), ("InfiniBand", ib)):
        gs, blocks = mean(rows, "bf16_gemm", "send_only")
        gr, _ = mean(rows, "bf16_gemm", "recv_only")
        ms, _ = mean(rows, "memory_bound", "send_only")
        mr, _ = mean(rows, "memory_bound", "recv_only")
        out.append(f"<tr><td>{label}</td><td>{blocks}</td><td>{gs:.2f}x</td><td>{gr:.2f}x</td>"
                   f"<td>{ms:.2f}x</td><td>{mr:.2f}x</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def _ranked(ctx: str, transports: str = "all", overlap_tp_cp: bool = False) -> list:
    """Every config that fits in memory, ranked by step time. PP hop transports:
    chain (TorchTitan's 1F1B), cache (TorchTitan's interleaved) and direct (each
    block straight to its consumers, either schedule); chain under interleaving
    only loses to cache, so it appears in the transport comparison only;
    transports="titan" leaves out direct."""
    import kimi_k3_step_time as st

    m = st.StepModel(overlap_tp_cp=overlap_tp_cp)
    rows = []
    for P in (1, 8, 16):
        scheds = [(1, None)] if P == 1 else [(v, tr) for v in (1, 2, 4)
                                                  for tr in (("chain", "direct") if v == 1 else ("cache", "direct"))]
        if transports == "titan":
            scheds = [x for x in scheds if x[1] != "direct"]
        for v, tr in scheds:
            for cp, tp in SPLITS:
                mem = fsdp_ep_parts(ctx, cp, tp) if P == 1 else pp_parts(ctx, P, 1, cp, tp, v)
                if mem["total"] > GB200_BYTES:
                    continue
                r = m.step(st.Config(ctx, P, cp, tp, v, tr))
                sched = "1F1B" if v == 1 else f"interleaved v = {v}"
                fam = "FSDP 512, EP 64" if P == 1 else f"PP {P}, {sched} ({tr}), EP {512 // P}"
                rows.append({"name": f"{fam}, CP {cp}, TP {tp}", "P": P, "v": v, "transport": tr, "cp": cp, "tp": tp,
                             "mem": mem["total"], "step": r["step"], "split": r["split"], "detail": r["detail"],
                             "M": r["M"]})
    return sorted(rows, key=lambda r: r["step"])


# Ranking chart: three categories, each a hue family with its own ring color;
# segments are (detail key, label, color, style). Within communication, color
# names the parallelism: solid = exposed time, hatched = overlap penalty.
_RANK_GROUPS = [
    ("Compute", "#184f95", [
        ("compute", "GEMMs + attention", "#2a78d6", "solid"),
        ("memory", "Memory-bound kernels", "#86b6ef", "solid"),
        ("weight_prep", "Weight prep: FSDP copies + weight quantization", "#184f95", "solid"),
    ]),
    ("Communication", "#a33c12", [
        ("exposed_fsdp", "FSDP, exposed", "#eb6834", "solid"),
        ("penalty_fsdp", "FSDP, compute slowdown", "#eb6834", "hatch"),
        ("penalty_fsdp_comm", "FSDP, collective slowdown", "#eb6834", "cross"),
        ("exposed_tp", "TP, exposed", "#6250d6", "solid"),
        ("exposed_cp", "CP, exposed", "#eda100", "solid"),
        ("exposed_ep", "EP, exposed", "#e87ba4", "solid"),
    ]),
    ("Pipeline", "#006300", [
        ("bubble", "Bubble (fill and drain)", "#b5b3ab", "solid"),
        ("hops", "Hops, exposed", "#008300", "solid"),
        ("penalty_hop", "Hops, compute slowdown", "#008300", "hatch"),
        ("zero1", "ZeRO-1 gather + reduce-scatter", "#1baf7a", "solid"),
    ]),
    ("Optimizer", "#a3302f", [
        ("optimizer", "DistMuon step", "#e34948", "solid"),
    ]),
]
_RANK_SEGS = [seg for _, _, segs in _RANK_GROUPS for seg in segs]
_RANK_AXIS_S = 32


def _rank_fill(color: str, style: str) -> str:
    """CSS background of a segment: solid, or hatched at 45 degrees."""
    if style == "solid":
        return color
    if style == "cross":
        return (f"repeating-linear-gradient(45deg, {color} 0 1.5px, transparent 1.5px 5px), "
                f"repeating-linear-gradient(135deg, {color} 0 1.5px, {color}40 1.5px 5px)")
    return (f"repeating-linear-gradient(135deg, {color} 0 2.5px, {color}55 2.5px 5px)")


def ranking_chart(ctx: str = "8K") -> str:
    """Every config that fits, ranked by step time, with TorchTitan's hop
    transports (chain for 1F1B, cache for interleaving): one stacked bar per
    config (the step by part, ringed by category) and its memory against one
    GB200."""
    rows = _ranked(ctx, "titan")

    def chip(key, val, label):
        return f'<button class="rk-chip on" data-key="{key}" data-val="{val}" aria-pressed="true">{label}</button>'

    out = ['<div class="rk-filters">'
           '<span>Family</span>' + chip("fam", "1", "FSDP 512") + chip("fam", "8", "PP 8") + chip("fam", "16", "PP 16")
           + '<span>Schedule</span>' + chip("v", "1", "1F1B") + chip("v", "2", "interleaved v = 2")
           + chip("v", "4", "interleaved v = 4") + "</div>"]
    sorts = [("step", "Step"), ("fsdp", "Exposed FSDP"), ("tpcp", "Exposed TP + CP"), ("bubble", "Bubble"),
             ("hops", "Hops"), ("mem", "Memory")]
    out.append('<div class="rk-filters rk-sorts"><span>Sort by</span>' + "".join(
        f'<button class="rk-sort{" on" if k == "step" else ""}" data-sort="{k}">{lab}</button>'
        for k, lab in sorts) + "</div>")
    leg = []
    for gname, ring, segs in _RANK_GROUPS:
        items = "".join(f'<span class="rk-leg-item"><span class="sw" style="background: {_rank_fill(c, st_)}"></span>'
                        f'{html.escape(lab)}</span>' for _, lab, c, st_ in segs)
        leg.append(f'<div class="rk-leg-group" style="border-color: {ring}"><b style="color: {ring}">{gname}</b>'
                   f'{items}</div>')
    out.append('<div class="rk-legend">' + "".join(leg) + "</div>")
    ticks = "".join(f'<span style="left: {100 * t / _RANK_AXIS_S:.2f}%">{t} s</span>'
                    for t in range(0, _RANK_AXIS_S + 1, 5))
    out.append('<div class="rk-row rk-head"><span></span><span>Config</span><span class="rk-axis">' + ticks
               + '</span><span class="r">Step</span><span>Memory of 192 GB</span></div>')
    out.append('<div class="rk-rows">')
    pct = lambda t: 100 * min(t, _RANK_AXIS_S) / _RANK_AXIS_S
    for i, r in enumerate(rows, 1):
        d, x, groups, tip = r["detail"], 0.0, [], []
        tip.append(f"<b>{i}. {html.escape(r['name'])}</b>")
        tip.append(f"Step {r['step']:.2f} s, memory {_dec(r['mem'])}, {r['M']} microbatches per step")
        for gname, ring, segs in _RANK_GROUPS:
            parts = [(lab, d[k], c, st_) for k, lab, c, st_ in segs if d[k] > 0]
            total = sum(v for _, v, _, _ in parts)
            if total <= 0:
                continue
            x0, x1 = x, x + total
            x = x1
            if x0 < _RANK_AXIS_S:
                # the inner strip keeps the group's full length, so a group cut
                # at the axis end keeps its segments' proportions
                inner = 100 * total / (min(x1, _RANK_AXIS_S) - x0)
                segs_html = "".join(f'<span style="width: {100 * v / total:.3f}%; background: {_rank_fill(c, st_)}">'
                                    f"</span>" for _, v, c, st_ in parts)
                groups.append(f'<span class="rk-grp" style="left: {pct(x0):.3f}%; width: {pct(x1) - pct(x0):.3f}%; '
                              f'--ring: {ring}"><span class="rk-inner" style="width: {inner:.2f}%">{segs_html}'
                              f"</span></span>")
            shown = [p for p in parts if p[1] >= 0.005]
            tip.append(f'<i style="color: {ring}">{gname}: {total:.2f} s</i>')
            tip += [f'<span class="sw" style="background: {_rank_fill(c, st_)}"></span>{html.escape(lab)}: {v:.2f} s'
                    for lab, v, c, st_ in shown]
        clipped = x > _RANK_AXIS_S
        fr = r["mem"] / GB200_BYTES
        out.append(
            f'<div class="rk-row" data-fam="{r["P"]}" data-v="{r["v"]}" data-tr="{r["transport"] or "none"}" '
            f'data-step="{r["step"]:.3f}" data-mem="{r["mem"]:.0f}" '
            f'data-fsdp="{d["exposed_fsdp"]:.3f}" data-tpcp="{d["exposed_tp"] + d["exposed_cp"]:.3f}" '
            f'data-bubble="{d["bubble"]:.3f}" data-hops="{d["hops"] + d["penalty_hop"]:.3f}" '
            f'data-tip="{html.escape("<br>".join(tip))}" tabindex="0">'
            f'<span class="rk-n">{i}</span><span class="rk-name">{html.escape(r["name"])}</span>'
            f'<span class="rk-bar{" rk-clipped" if clipped else ""}" role="img" '
            f'aria-label="{html.escape(r["name"])}: {r["step"]:.1f} s">{"".join(groups)}</span>'
            f'<span class="r"><b>{r["step"]:.1f} s</b>{" &raquo;" if clipped else ""}</span>'
            f'<span class="rk-mem"><span class="rk-gauge"><span style="width: {100 * fr:.1f}%"></span></span>'
            f'{_dec(r["mem"])}</span></div>')
    out.append("</div>")
    # table view of the same data
    head = ["#", "Config", "Memory", "Step"] + [lab for _, lab, _, _ in _RANK_SEGS]
    tbl = ['<details class="rk-table"><summary>All configs as a table</summary><table class="dtable"><tr>'
           + "".join(f"<th>{html.escape(h)}</th>" for h in head) + "</tr>"]
    for i, r in enumerate(rows, 1):
        tbl.append(f"<tr><td>{i}</td><td>{html.escape(r['name'])}</td><td>{_dec(r['mem'])}</td>"
                   f"<td><b>{r['step']:.1f}</b></td>"
                   + "".join(f"<td>{r['detail'][k]:.2f}</td>" if r["detail"][k] >= 0.005 else "<td>-</td>"
                             for k, _, _, _ in _RANK_SEGS) + "</tr>")
    tbl.append("</table></details>")
    return '<div class="rk-chart">' + "".join(out) + "".join(tbl) + "</div>"


def overlap_tp_cp_table() -> str:
    """What-if: TP and CP collectives overlapped (with the NVLink overlap
    penalties) instead of exposed, for the configs at the top of each context."""
    rows = []
    for ctx in ("8K", "32K"):
        base = {r["name"]: (i, r) for i, r in enumerate(_ranked(ctx, "titan"), 1)}
        over = {r["name"]: (i, r) for i, r in enumerate(_ranked(ctx, "titan", overlap_tp_cp=True), 1)}
        names = {n for n, (i, _) in base.items() if i <= 3} | {n for n, (i, _) in over.items() if i <= 3}
        best_fsdp = min((r for _, r in base.values() if r["P"] == 1), key=lambda r: r["step"])["name"]
        for n in sorted(names | {best_fsdp}, key=lambda n: over[n][0]):
            rows.append((ctx, n, base[n], over[n]))
    head = ["Context", "Config", "TP / CP exposed (today)", "TP / CP overlapped"]
    out = ['<table class="dtable"><caption>Step time, and in brackets the rank among TorchTitan configs'
           '</caption><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    short = lambda n: n.replace(", EP 64", "").replace(", EP 32", "").replace(" (cache)", "").replace(" (chain)", "")
    for ctx, n, (ib, rb), (io, ro) in rows:
        out.append(f"<tr><td>{ctx}</td><td>{short(n)}</td><td>{rb['step']:.1f} s ({ib})</td>"
                   f"<td><b>{ro['step']:.1f} s</b> ({io})</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def fsdp_postmortem_table() -> str:
    """FSDP 512 + EP 64 without PP: every split that fits, the step by part."""
    import kimi_k3_step_time as st

    m = st.StepModel()
    head = ["Config", "Microbatches per GPU", "Memory", "Compute + memory-bound", "Weight prep",
            "FSDP exposed", "FSDP penalties", "TP / CP exposed", "Optimizer", "Step"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for ctx, splits in (("8K", [(1, 1), (2, 1), (4, 1), (1, 2), (1, 4)]), ("32K", [(4, 1), (1, 4)])):
        for cp, tp in splits:
            c = st.Config(ctx, 1, cp, tp)
            r = m.step(c)
            d = r["detail"]
            mem = fsdp_ep_parts(ctx, cp, tp)["total"]
            out.append(f"<tr><td>{ctx}, CP {cp}, TP {tp}</td><td>{c.M}</td><td>{_dec(mem)}</td>"
                       f"<td>{d['compute'] + d['memory']:.1f} s</td><td>{d['weight_prep']:.1f} s</td>"
                       f"<td>{d['exposed_fsdp']:.1f} s</td><td>{d['penalty_fsdp'] + d['penalty_fsdp_comm']:.1f} s</td>"
                       f"<td>{d['exposed_tp'] + d['exposed_cp']:.1f} s</td><td>{d['optimizer']:.1f} s</td>"
                       f"<td><b>{r['step']:.1f} s</b></td></tr>")
    out.append("</table>")
    return "\n".join(out)


def pp16_vs_pp8_table() -> str:
    """PP 16 against PP 8 at the same split, schedule and transport: the step
    and where the difference comes from."""
    import kimi_k3_step_time as st

    m = st.StepModel()
    cases = [("8K", 1, 2, 1, "direct"), ("8K", 1, 2, 2, "cache"), ("8K", 1, 2, 4, "cache"), ("32K", 1, 4, 2, "cache")]
    head = ["Config", "Memory: PP 8 / PP 16", "PP 8", "PP 16", "Compute", "Bubble", "Hops + their penalty", "Other"]
    out = ['<table class="dtable"><caption>The last four columns: PP 16 minus PP 8</caption><tr>'
           + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for ctx, cp, tp, v, tr in cases:
        r8, r16 = (m.step(st.Config(ctx, P, cp, tp, v, tr)) for P in (8, 16))
        m8, m16 = (pp_parts(ctx, P, 1, cp, tp, v)["total"] for P in (8, 16))
        d8, d16 = r8["detail"], r16["detail"]
        comp = lambda d: d["compute"] + d["memory"] + d["weight_prep"]
        hop = lambda d: d["hops"] + d["penalty_hop"]
        dc, db, dh = comp(d16) - comp(d8), d16["bubble"] - d8["bubble"], hop(d16) - hop(d8)
        other = r16["step"] - r8["step"] - dc - db - dh
        sched = "1F1B" if v == 1 else f"interleaved v = {v}"
        cell = lambda x: f"<td>{x:+.1f} s</td>"
        out.append(f"<tr><td>{ctx}, {sched} ({tr}), CP {cp}, TP {tp}</td><td>{_dec(m8)} / {_dec(m16)}</td>"
                   f"<td>{r8['step']:.1f} s</td><td>{r16['step']:.1f} s</td>{cell(dc)}{cell(db)}{cell(dh)}{cell(other)}</tr>")
    out.append("</table>")
    return "\n".join(out)


def transport_compare_table() -> str:
    """Chain, cache and direct at the same PP config: step, and in brackets what
    the hops add (their exposed time plus the compute penalty they cost)."""
    import kimi_k3_step_time as st

    m = st.StepModel()
    cases = [("8K", 8, 1, 2, 1), ("8K", 16, 1, 2, 1), ("8K", 8, 1, 2, 2), ("8K", 8, 1, 2, 4), ("8K", 16, 1, 1, 2),
             ("8K", 16, 1, 1, 4), ("32K", 8, 4, 1, 1), ("32K", 16, 4, 1, 2), ("32K", 8, 1, 4, 4)]
    head = ["Config", "chain", "cache", "direct"]
    out = ['<table class="dtable"><caption>Step time, and in brackets what the hops cost (exposed time plus '
           'their compute penalty); bold: the fastest transport</caption><tr>'
           + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for ctx, P, cp, tp, v in cases:
        if pp_parts(ctx, P, 1, cp, tp, v)["total"] > GB200_BYTES:
            continue
        res = {}
        for tr in ("chain", "cache", "direct"):
            if v == 1 and tr == "cache":
                continue
            r = m.step(st.Config(ctx, P, cp, tp, v, tr))
            res[tr] = (r["step"], r["detail"]["hops"] + r["detail"]["penalty_hop"])
        best = min(res, key=lambda k: res[k][0])
        sched = "1F1B" if v == 1 else f"interleaved v = {v}"
        cells = ""
        for tr in ("chain", "cache", "direct"):
            if tr not in res:
                cells += "<td>-</td>"
                continue
            t, hp = res[tr]
            txt = f"{t:.1f} s ({hp:.1f} s)"
            cells += f"<td><b>{txt}</b></td>" if tr == best else f"<td>{txt}</td>"
        out.append(f"<tr><td>{ctx}, PP {P}, {sched}, CP {cp}, TP {tp}</td>{cells}</tr>")
    out.append("</table>")
    return "\n".join(out)


def ranking_table(ctx: str = "8K") -> str:
    """The fastest split (CP, TP) of each family, schedule and transport that
    fits, ranked by step time, with the step split into its parts."""
    rows, seen = [], set()
    for r in _ranked(ctx, "titan"):
        key = (r["P"], r["v"], r["transport"])
        if key not in seen:
            seen.add(key); rows.append(r)
    head = ["Rank", "Config", "Memory", "Compute + memory-bound", "Exposed communication",
            "Overlap penalties", "Pipeline: bubble, hops, ZeRO-1", "Optimizer", "Step"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for i, r in enumerate(rows, 1):
        sp = r["split"]
        pipe = f"{sp['pipeline']:.1f} s" if r["P"] > 1 else "-"
        out.append(f"<tr><td>{i}</td><td>{r['name']}</td><td>{_dec(r['mem'])}</td><td>{sp['compute']:.1f} s</td>"
                   f"<td>{sp['exposed']:.1f} s</td><td>{sp['penalty']:.1f} s</td><td>{pipe}</td>"
                   f"<td>{sp['optimizer']:.1f} s</td>"
                   f"<td><b>{r['step']:.1f} s</b></td></tr>")
    out.append("</table>")
    return "\n".join(out)


def held_split(ep: int, tp: int, layers: int) -> tuple[float, float]:
    """Unsharded fp32 gradients + compute weights a GPU holds for `layers`
    layers, split into (routed experts, dense) bytes."""
    d = _data()
    lay = {r["key"]: r["numel"] for r in d["layers"][0]["scope"]["stored"]}
    routed_n = lay["routed"]
    dense_n = sum(lay.values()) - routed_n
    q_blk = dense_quantized_params()
    routed = layers * (MXFP8_WEIGHT_BYTES + 4) * routed_n / ep
    dense = layers * (MXFP8_WEIGHT_BYTES * q_blk + 2 * (dense_n - q_blk) + 4 * dense_n) / tp
    return routed, dense


def tp_dense_table() -> str:
    """How much of what a GPU holds unsharded is dense (TP's target), from no
    parallelism to PP with ZeRO-1. TP rows are shaded against the row above."""
    rows = [
        ("No parallelism (one layer)", "1", 1, 1, 1),
        ("FSDP 512", "2, in flight", 1, 1, 2),
        ("FSDP 512 + EP 64", "2, in flight", 64, 1, 2),
        ("FSDP 512 + EP 64 + TP 2", "2, in flight", 64, 2, 2),
        ("PP 8, ZeRO-1 + EP 64", "12, all step", 64, 1, 12),
        ("PP 8, ZeRO-1 + EP 64 + TP 2", "12, all step", 64, 2, 12),
    ]
    green = ' style="background: rgba(0, 131, 0, 0.12)"'
    head = ["Config", "Layers held unsharded", "Routed experts", "Dense", "Total"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for name, held, ep, tp, layers in rows:
        r, dn = held_split(ep, tp, layers)
        pr, pd = round(100 * r / (r + dn)), round(100 * dn / (r + dn))
        pct = (lambda v: f"<b>({v}%)</b>") if ep > 1 else (lambda v: f"({v}%)")
        sh = green if tp > 1 else ""
        out.append(f"<tr><td>{name}</td><td>{held}</td><td>{_dec(r)} {pct(pr)}</td>"
                   f"<td{sh}>{_dec(dn)} {pct(pd)}</td><td{sh}>{_dec(r + dn)}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def pp_table(ctx: str = "8K", degrees=(8, 16)) -> str:
    """PP 8 / 16 under ZeRO-1 (FSDP's default with PP), with each sequence
    split by CP or TP. Cells are shaded against the same PP degree without a
    split; the total is green if it fits in one GPU, red if not."""
    splits = [(1, 1), (2, 1), (1, 2), (4, 1), (1, 4)]
    head = [f"PP {degrees[0]}, ZeRO-1 (FSDP {512 // degrees[0]}, EP {512 // degrees[0]})" if len(degrees) == 1 else "PP, ZeRO-1", "Activations", "Gradients (stage)",
            "Compute weights (stage, MXFP8)", "Total", "x GPU"]
    green, red = ' style="background: rgba(0, 131, 0, 0.12)"', ' style="background: rgba(227, 73, 72, 0.15)"'
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for P in degrees:
        base = pp_parts(ctx, P, 1)
        for cp, tp in splits:
            r = pp_parts(ctx, P, 1, cp, tp)
            sh = lambda k: ("" if abs(r[k] - base[k]) <= 1e-3 * max(r[k], 1) else green if r[k] < base[k] else red)
            name = f"PP {P}, CP {cp}, TP {tp}"
            fits = r["total"] <= GB200_BYTES
            out.append(f'<tr><td>{name}<span class="sub">{r["microbatches"]:g} microbatches per pipeline</span></td>'
                       f'<td{sh("acts")}>{_dec(r["acts"])}<span class="sub">incl. {_dec(r["meta"])} DistMoE indices</span></td>'
                       f'<td{sh("grads")}>{_dec(r["grads"])}</td>'
                       f'<td{sh("weights")}>{_dec(r["weights"])}</td>'
                       f'<td{green if fits else red}><b>{_dec(r["total"])}</b><span class="sub">{"fits" if fits else "over 192 GB"}</span></td>'
                       f'<td>{r["total"] / GB200_BYTES:.2f}</td></tr>')
    out.append("</table>")
    return "".join(out)


FIT_SPLITS = [(1, 1), (2, 1), (1, 2), (4, 1), (1, 4)]


def pp_vpp_table(ctx: str = "8K") -> str:
    """Rank 0 of PP 8 / 16 under 1F1B and Interleaved 1F1B (v = 2, 4):
    layer-microbatches in flight, activations, total (green if it fits). At 8K
    with no split; at 32K with CP 4, since 32K without a split has only 8
    microbatches per pipeline, all in flight under every schedule."""
    cp = 1 if ctx == "8K" else 4
    head = [f"Schedule (CP {cp})", "Microbatches per pipeline", "Chunk-microbatches in flight (rank 0)",
            "Layers per chunk (rank 0)",
            "Layer-microbatches in flight", "Activations", "Total"]
    out = ['<table class="dtable"><tr>' + "".join(f"<th>{h}</th>" for h in head) + "</tr>"]
    for P in (8, 16):
        for v in (1, 2, 4):
            r = pp_parts(ctx, P, 1, cp, 1, v)
            chunks = pp_inflight(P, v, int(r["microbatches"]))
            fits = r["total"] <= GB200_BYTES
            sched = "1F1B" if v == 1 else f"interleaved, v = {v}"
            out.append(f"<tr><td>PP {P}, {sched}</td><td>{r['microbatches']:.0f}</td><td>{chunks}</td>"
                       f"<td>{rank0_chunk_layers(P, v):g}</td>"
                       f"<td>{r['held']}</td><td>{_dec(r['acts'])}</td>"
                       f"<td{_GREEN if fits else _RED}><b>{_dec(r['total'])}</b></td></tr>")
    out.append("</table>")
    return "\n".join(out)


def fit_map(ctx: str = "8K") -> str:
    """Every config of this section: FSDP + EP 64 and PP 8 / 16 (ZeRO-1), each
    with no split, CP 2 / 4 or TP 2 / 4. Green: fits in 192 GB; red: does not."""
    green, red = ' style="background: rgba(0, 131, 0, 0.12)"', ' style="background: rgba(227, 73, 72, 0.15)"'
    name = lambda cp, tp: f"CP {cp}, TP {tp}"
    families = [("FSDP 512, EP 64", lambda cp, tp: fsdp_ep_parts(ctx, cp, tp))]
    for P in (8, 16):
        for v in (1, 2, 4):
            sched = "1F1B" if v == 1 else f"interleaved, v = {v}"
            families.append((
                f"PP {P}, {sched}<span class='sub'>ZeRO-1, FSDP {512 // P}, EP {512 // P} per stage</span>",
                lambda cp, tp, P=P, v=v: pp_parts(ctx, P, 1, cp, tp, v),
            ))
    out = ['<table class="dtable"><tr><th>Per GPU</th>' + "".join(f"<th>{name(*sp)}</th>" for sp in FIT_SPLITS) + "</tr>"]
    for label, fn in families:
        cells = []
        for cp, tp in FIT_SPLITS:
            total = fn(cp, tp)["total"]
            cells.append(f'<td{green if total <= GB200_BYTES else red}>{_dec(total)}</td>')
        out.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    out.append("</table>")
    return "".join(out)


def fit_count(ctx: str) -> tuple[int, int]:
    n = fits = 0
    for cp, tp in FIT_SPLITS:
        for r in [fsdp_ep_parts(ctx, cp, tp)] + [
            pp_parts(ctx, P, 1, cp, tp, v) for P in (8, 16) for v in (1, 2, 4)
        ]:
            n += 1
            fits += r["total"] <= GB200_BYTES
    return fits, n


def by_context(fn):
    """One SVG per context; the page's context toggle shows one at a time."""
    def render() -> str:
        return "".join(f'<div class="ctx-fig" data-ctx="{c}"{"" if c == "8K" else " hidden"}>{fn(c)}</div>'
                       for c in ("8K", "32K"))
    return render


FIGURES.update({"memory-weights": memory_weights_bf16, "memory-by-kind": by_context(memory_by_kind),
                "memory-remat": by_context(memory_remat), "memory-summary": by_context(memory_summary),
                "memory-distmoe": by_context(memory_distmoe),
                "sharding-steps": by_context(sharding_steps), "zero-table": by_context(zero_table),
                "cp-table": by_context(cp_table), "tp-table": by_context(tp_table),
                "split-steps": by_context(split_steps), "pp-table": by_context(pp_table),
                "pp8-table": by_context(lambda c: pp_table(c, (8,))),
                "pp16-table": by_context(lambda c: pp_table(c, (16,))),
        "tp-dense-table": tp_dense_table,
        "roofline-table": by_context(roofline_table),
        "dense-shapes": by_context(dense_shapes_table),
        "expert-shapes": by_context(expert_shapes_table),
        "mla-shapes": by_context(mla_table),
        "ideal-gap-table": by_context(ideal_gap_table),
        "nonmatmul-table": nonmatmul_table,
        "overlap-table": overlap_table,
        "sweep-table": sweep_table,
        "pp-vpp-table": by_context(pp_vpp_table),
        "one-way-table": one_way_table,
        "ib-bw-table": ib_bw_table,
        "xhost-overlap-table": xhost_overlap_table,
        "ranking-table": by_context(ranking_table),
        "ranking-chart": by_context(ranking_chart),
        "fsdp-postmortem-table": fsdp_postmortem_table,
        "pp16-vs-pp8-table": pp16_vs_pp8_table,
        "transport-compare-table": transport_compare_table,
        "overlap-formulas": overlap_formulas,
        "overlap-formulas-ib": lambda: overlap_formulas(1.3, 1.7),
        "hop-formulas": hop_formulas,
        "transport-diagram": transport_diagram,
        "overlap-tp-cp-table": overlap_tp_cp_table,
        "transport-diagram-titan": lambda: transport_diagram(("chain", "cache")),
        "transport-matrix": transport_matrix,
        "bubble-formulas": bubble_formulas,
        "pp-hop-timeline": pp_hop_timeline,
        "pp-hop-timeline-interleaved": lambda: pp_hop_timeline(2),
        "ib-rule-formula": ib_rule_formula,
        "fsdp-ib-formula": fsdp_ib_formula,
        "fsdp-ib-table": fsdp_ib_table,
        "ib-rule-table": ib_rule_table,
        "gb200-ib-table": gb200_ib_table,
        "expert-ib-table": expert_ib_table,
        "pp-ib-table": pp_ib_table,
        "overlap-examples": overlap_examples,
        "compute-table-fsdp": by_context(lambda c: compute_table(c, 1)),
        "compute-table-pp8": by_context(lambda c: compute_table(c, 8)),
        "compute-table-pp16": by_context(lambda c: compute_table(c, 16)), "fit-map": by_context(fit_map)})
