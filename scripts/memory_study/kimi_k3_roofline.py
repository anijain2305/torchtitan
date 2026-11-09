# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Render the Kimi K3 step-time roofline as a self-contained HTML page.

Reads kimi_k3_data.json (written by kimi_k3_params_viz.py: parameters, layers,
activations and the memory model's inputs) and the measurements in bench/
(kimi_k3_kernel_bench.py and nccl-tests), so it regenerates in about a second
without re-measuring anything.

The page shows:
  1. Roofline inputs (provisional): the kernel-efficiency rule with the
     measured kernel throughput beside it, and NVLink nccl-tests curves.

Usage:
  python scripts/memory_study/kimi_k3_roofline.py [--out scripts/memory_study/kimi_k3_roofline.html]
"""

import argparse
import json
from pathlib import Path

import kimi_k3_step_model as step_model
from page_common import render

HERE = Path(__file__).resolve().parent


def bench_inputs(bench_dir: Path) -> dict | None:
    """Measured kernel throughput (summary of kimi_k3_kernel_bench.py) and
    nccl-tests curves, or None if the bench files are missing."""
    kernels = bench_dir / "kernels_GB300_summary.json"
    if not kernels.exists():
        return None
    nvlink = {}
    for op in ("allgather", "reducescatter", "alltoall", "sendrecv"):
        path = bench_dir / f"nccl_{op}_4gpu.txt"
        if not path.exists():
            continue
        rows = []
        for line in path.read_text().splitlines():
            # size count type redop root | out-of-place: time algbw busbw #wrong | in-place ...
            parts = line.split()
            if len(parts) > 8 and parts[0].isdigit() and parts[2] == "bfloat16":
                rows.append([int(parts[0]), float(parts[5]), float(parts[7])])
        nvlink[op] = rows
    dist_moe = bench_dir / "dist_moe_GB300_ep4.json"
    mla_fa4 = bench_dir / "mla_fa4_GB300.json"
    fake_pg = bench_dir / "fake_pg_GB300.json"
    dist_moe = bench_dir / "dist_moe_GB300_ep4.json"
    mla_fa4 = bench_dir / "mla_fa4_GB300.json"
    fake_pg = bench_dir / "fake_pg_GB300.json"
    shapes = bench_dir / "attention_shapes_GB300.json"
    return {
        "kernels": json.loads(kernels.read_text()),
        "nvlink": nvlink,
        "attention_shapes": json.loads(shapes.read_text()) if shapes.exists() else None,
        "dist_moe": json.loads(dist_moe.read_text()) if dist_moe.exists() else None,
        "mla_fa4": json.loads(mla_fa4.read_text())["mla_fa4"]["data"]
        if mla_fa4.exists()
        else None,
        "fake_pg": json.loads(fake_pg.read_text()) if fake_pg.exists() else None,
        "dist_moe": json.loads(dist_moe.read_text()) if dist_moe.exists() else None,
        "mla_fa4": json.loads(mla_fa4.read_text())["mla_fa4"]["data"]
        if mla_fa4.exists()
        else None,
        "fake_pg": json.loads(fake_pg.read_text()) if fake_pg.exists() else None,
        "kda_cp": {
            cp: json.loads(path.read_text())["results"]
            for cp in (2, 4)
            if (path := bench_dir / f"kda_cp_GB300_cp{cp}.json").exists()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-data",
        type=Path,
        default=HERE / "kimi_k3_data.json",
        help="model data written by kimi_k3_params_viz.py",
    )
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=HERE / "bench",
        help="kernel and nccl-tests results",
    )
    parser.add_argument("--out", type=Path, default=HERE / "kimi_k3_roofline.html")
    # configs and per-GPU memory: kimi_k3_configs.json, from kimi_k3_configs_export.py
    args = parser.parse_args()

    if not args.model_data.exists():
        print(f"{args.model_data} not found; run kimi_k3_params_viz.py first.")
        return 1
    data = {
        "model": json.loads(args.model_data.read_text()),
        data["tp_vs_cp"] = step_model.tp_vs_cp(data["model"], data["bench"])
        "bench": bench_inputs(args.bench_dir),
    }
    configs = HERE / "kimi_k3_configs.json"
    if data["bench"] and configs.exists():
        data["links"] = step_model.link_table(data["bench"])
        data["ranking"] = step_model.rank_configs(data["model"], data["bench"],
                                                  json.loads(configs.read_text())["configs"])
        data["tp_vs_cp"] = step_model.tp_vs_cp(data["model"], data["bench"])
    args.out.write_text(render(HTML_TEMPLATE, data))
    print(f"Wrote {args.out}")
    return 0


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kimi K3 -- step-time roofline</title>
<style>
/*__CSS__*/
.formula { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
           padding: 14px 18px; margin: 12px 0 8px; font-size: 14px; line-height: 2.0; }
.formula .eq { font-family: "Cambria Math", "STIX Two Math", Georgia, serif; font-size: 16px; }
.formula .lbl { display: inline-block; min-width: 250px; color: var(--text-secondary);
                font-family: system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 13px; }
.formula .where { margin-top: 6px; font-size: 13px; color: var(--text-secondary); line-height: 1.6; }
.formula i { font-style: italic; }
</style>
</head>
<body>
<div class="viz-root">
<div class="wrap">
  <div class="topbar">
    <div>
      <h1>Kimi K3: step-time roofline</h1>
      <p class="sub" id="subtitle"></p>
    </div>
    <div class="ctx-float">
      <span class="ctx-label">Context</span>
      <div class="seg" role="radiogroup" aria-label="Context length" id="ctxToggle"></div>
    </div>
  </div>

  <h2>1. Compute time: roofline estimate</h2>
  <p class="sub" id="rfSub"></p>
  <p class="sub" id="rfSymbols"></p>

  <h3 class="subhead">1.1 Dense linears</h3>
  <div class="formula"><div class="eq">
    <i>t</i><sub>dense</sub> = 8 &middot; (<i>T</i>/<i>cp</i>) &middot; (<i>N</i><sub>dense</sub>/<i>tp</i>)
    &nbsp;/&nbsp; ( &eta; &middot; <i>R</i><sub>peak</sub> )</div></div>
  <p class="sub" id="rfDense"></p>

  <h3 class="subhead">1.2 Routed experts</h3>
  <div class="formula">
    <div class="where" style="margin: 0 0 4px"><b>Roofline terms</b></div>
    <div class="eq"><span class="lbl">w13 (gate + up, L &rarr; 2&middot;F<sub>e</sub>)</span>
      <i>t</i><sub>w13</sub> = 8 &middot; (<i>T</i>/(<i>cp</i>&middot;<i>tp</i>)) &middot; (<i>k</i> &middot; <i>N</i><sub>w13</sub>)
      &nbsp;/&nbsp; ( &eta; &middot; <i>R</i><sub>peak</sub> )</div>
    <div class="eq"><span class="lbl">w2 (down, F<sub>e</sub> &rarr; L)</span>
      <i>t</i><sub>w2</sub> = 8 &middot; (<i>T</i>/(<i>cp</i>&middot;<i>tp</i>)) &middot; (<i>k</i> &middot; <i>N</i><sub>w2</sub>)
      &nbsp;/&nbsp; ( &eta; &middot; <i>R</i><sub>peak</sub> )</div>
    <div class="eq"><span class="lbl">Routed experts</span>
      <i>t</i><sub>experts</sub> = <i>t</i><sub>w13</sub> + <i>t</i><sub>w2</sub></div>
  </div>
  <div class="formula" style="background: transparent">
    <div class="where" style="margin: 0 0 4px"><b>Why ep does not appear: one expert, then one GPU</b> (exposition, not
    used directly)</div>
    <div class="eq"><span class="lbl">Tokens per expert</span>
      <i>n</i><sub>e</sub> = (<i>T</i>/(<i>cp</i>&middot;<i>tp</i>)) &middot; <i>ep</i> &middot; <i>k</i> / <i>E</i></div>
    <div class="eq"><span class="lbl">FLOPs per expert</span>
      <i>F</i><sub>expert</sub> = 8 &middot; <i>n</i><sub>e</sub> &middot; (<i>N</i><sub>w13</sub> + <i>N</i><sub>w2</sub>)</div>
    <div class="eq"><span class="lbl">FLOPs per GPU (<i>E</i>/<i>ep</i> experts)</span>
      (<i>E</i>/<i>ep</i>) &middot; <i>F</i><sub>expert</sub> = 8 &middot; (<i>T</i>/(<i>cp</i>&middot;<i>tp</i>)) &middot; <i>k</i>
      &middot; (<i>N</i><sub>w13</sub> + <i>N</i><sub>w2</sub>) &nbsp; &larr; <i>ep</i> cancels</div>
  </div>
  <p class="sub" id="rfExperts"></p>

  <h3 class="subhead">1.3 MLA attention</h3>
  <div class="formula">
    <div class="eq"><span class="lbl">Forward FLOPs per GPU</span>
      <i>F</i> = <i>T</i><sup>2</sup> &middot; (<i>H</i>/(<i>tp</i>&middot;<i>cp</i>)) &middot; (<i>d</i><sub>qk</sub> + <i>d</i><sub>v</sub>)</div>
    <div class="eq"><span class="lbl">MLA attention (roofline term)</span>
      <i>t</i><sub>MLA</sub> = 4.5 &middot; <i>F</i> / ( &eta; &middot; <i>R</i><sub>peak</sub> )</div>
  </div>
  <p class="sub" id="rfMla"></p>

  <h3 class="subhead">1.4 KDA attention (measured model, not a roofline)</h3>
  <div class="formula"><div class="eq">
    <i>t</i><sub>KDA</sub> = (<i>T</i>/<i>cp</i>) &middot; ( 0.40 / <i>s</i>(<i>T</i>, <i>tp</i>) + 0.05 ) &micro;s</div></div>
  <p class="sub" id="rfKda"></p>

  <h3 class="subhead">1.5 Putting it together</h3>
  <div class="formula"><div class="eq">
    <i>t</i><sub>layer</sub> = <i>t</i><sub>dense</sub> + <i>t</i><sub>experts</sub> + ( <i>t</i><sub>MLA</sub> or <i>t</i><sub>KDA</sub> )</div></div>

  <h3 class="subhead">1.6 Per-step compute time per GPU</h3>
  <div class="formula">
    <div class="eq"><span class="lbl">Per step, one GPU</span>
      <i>t</i><sub>step</sub> = <i>n</i><sub>mb</sub> &middot; &Sigma;<sub>stage layers</sub> <i>t</i><sub>layer</sub>
      + <i>t</i><sub>layer 0</sub> + <i>t</i><sub>LM head</sub>,
      &nbsp; <i>n</i><sub>mb</sub> = (tokens per step / <i>T</i>) &middot; <i>pp</i> &middot; <i>cp</i> &middot; <i>tp</i> / 512</div>
    <div class="eq"><span class="lbl">Layer 0 (KDA + dense FFN)</span>
      dense FFN: 8 &middot; <i>D</i><sub>tok</sub> &middot; 3&middot;<i>D</i>&middot;<i>F</i> / ( &eta; &middot; <i>R</i><sub>peak</sub> ), no routed experts</div>
    <div class="eq"><span class="lbl">LM head + chunked loss</span>
      <i>t</i><sub>LM head</sub> = 6 &middot; <i>D</i><sub>tok</sub> &middot; <i>V</i>&middot;<i>D</i> / ( &eta; &middot; <i>R</i><sub>peak</sub> ) &nbsp; (outside the remat blocks: no replay)</div>
  </div>
  <p class="sub" id="rfStepSub"></p>
  <div class="panel"><table id="rfStep"></table></div>

  <h2>2. Measurements: does the roofline hold? (provisional)</h2>
  <p class="sub" id="riSub"></p>
  <div class="callout note" id="riFormulaGap"></div>

  <h3 class="subhead">2.1 Dense GEMMs</h3>
  <p class="sub" id="riLinSub"></p>
  <div class="panel"><table id="riLinears"></table></div>
  <h3 class="subhead">2.2 Routed-expert GEMMs</h3>
  <p class="sub" id="riExSub"></p>
  <div class="panel"><table id="riExperts"></table></div>
  <h3 class="subhead">2.3 Attention</h3>
  <p class="sub" id="riAttnSub"></p>
  <div class="panel"><table id="riAttn"></table></div>
  <p class="sub" id="riKdaSub" style="margin-top: 16px"></p>
  <div class="panel"><table id="riKda"></table></div>
  <div class="panel" style="margin-top: 16px"><table id="riKdaCp"></table></div>
  <h3 class="subhead">2.4 Communication: NVLink</h3>
  <div class="panel"><table id="riNvlink"></table></div>
  <div class="callout note" id="riNote"></div>

  <h3 class="subhead">2.5 Whole step: fake-process-group run</h3>
  <p class="sub" id="fpSub"></p>
  <div class="panel"><table id="fpTable"></table></div>
  <div class="panel" id="fpPies" style="margin-top: 16px"></div>
  <div class="sub" id="fpWhy" style="margin-top: 12px"></div>

  <h2>3. Communication: links and bandwidth</h2>
  <p class="sub" id="cmSub"></p>
  <div class="panel"><table id="cmLinks"></table></div>
  <p class="sub" id="cmPlaceSub" style="margin-top: 16px"></p>
  <div class="panel"><table id="cmPlace"></table></div>

  <h2>4. Step time per parallelism config</h2>
  <p class="sub" id="stSub"></p>
  <div class="formula" id="stFormula"></div>
  <h3 class="subhead">4.1 Which communication is exposed, per layer</h3>
  <p class="sub" id="stLayerSub"></p>
  <div class="panel"><table id="stLayer"></table></div>
  <h3 class="subhead">4.2 Ranking the configs that fit</h3>
  <p class="sub" id="stRankSub"></p>
  <div class="panel" id="stBars"></div>
  <div class="panel" style="margin-top: 16px"><table id="stRank"></table></div>
  <div class="sub" id="stWhy" style="margin-top: 12px"></div>
  <h3 class="subhead">4.3 Deep dive: how the PP step time is computed</h3>
  <div class="sub" id="ppdSub" style="margin-bottom: 16px"></div>
  <div class="panel"><table id="ppdStages"></table></div>
  <div class="panel" style="margin-top: 16px"><table id="ppdSum"></table></div>
  <div class="sub" id="ppdWhy" style="margin-top: 12px"></div>
  <h3 class="subhead">4.4 TP vs CP</h3>
  <p class="sub" id="tcSub"></p>
  <div class="panel"><table id="tcTable"></table></div>
  <div class="sub" id="tcWhy" style="margin-top: 12px"></div>

  <p class="foot" id="foot"></p>
</div>
<div id="tip" role="tooltip"></div>
</div>

<script>
const DATA = /*__DATA__*/null;

/*__HELPERS__*/

// Context-length toggle, shared with the memory page's contexts.
let CTX = Object.keys(DATA.model.contexts)[0];
const C = () => DATA.model.contexts[CTX];

// ---------- section 1: roofline inputs ----------
// Per-GPU dense peaks from the NVIDIA Blackwell datasheet (GB200 column; the
// datasheet lists Tensor Core rates sparse, dense is half). MXFP8 runs at the
// FP8 rate; NVLink 1.8 TB/s is bidirectional, so busbw compares to 900 GB/s.
const DATASHEET = { label: "GB200 datasheet", bf16: 2500, fp8: 5000, nvlink: 900 };

// Efficiency for n tokens under a tiered rule: the highest tier at or below n
// (a count under the first tier takes the first tier).
function ruleEff(rule, dtype, n) {
  let i = 0;
  while (i + 1 < rule.tiers.length && n >= rule.tiers[i + 1]) i++;
  return { eff: rule[dtype][i], tier: rule.tiers[i] };
}

// The roofline's single assumed efficiency for matmul-like kernels (dense,
// experts, MLA), a fraction of DATASHEET peak. bf16: what large GEMMs on this
// GPU reach (dense forward / dgrad 68-73%, DistMoE GEMM kernels 60-73%).
const ROOFLINE_EFF = { bf16: 0.70, mxfp8: 0.60 };

// KDA's measured model (section 1.4), the one kernel the roofline costs from
// measurements rather than FLOPs.
const KDA_MODEL = {
  // KDA, us per token per layer with all 96 heads: forward + backward, and
  // the remat forward (kept flat under TP: the forward barely speeds up with
  // fewer heads). TP t speeds up forward + backward by the measured factor rounded down,
  // by sequence length (TP 1 / 2 / 4 / 8): KDA scales well at 32K but stops
  // improving past TP 2 at 8K, where the kernel no longer fills the GPU (the
  // contended 8K run measured TP 4 / 8 slightly slower than TP 2; held at TP 2's
  // speedup until a clean re-run).
  kda: { fwd_bwd: 0.40, fwd: 0.05, tp: [1, 2, 4, 8],
         tp_speedup: { tiers: [8192, 32768], by_tier: [[1, 1.7, 1.7, 1.7], [1, 1.8, 3.0, 5.0]] } },
};

// A table of measured values with each cell's % of its row's datasheet peak
// (when the row has one); cells under 60% of the row's best are shaded,
// marking where efficiency falls off.
// Diverging shading by distance from the roofline: measured efficiency over
// the assumed one, minus 1. Red (palette #e34948) deepens as measured falls
// short, darkest at -60% or worse; blue (#2a78d6) where it beats the roofline,
// fullest at +20%; neutral within 2%.
function rooflineShade(frac, assumed) {
  const gap = frac / assumed - 1;
  if (gap < -0.02) return `background: rgba(227, 73, 72, ${(0.08 + 0.42 * Math.min(1, -gap / 0.6)).toFixed(3)})`;
  if (gap > 0.02) return `background: rgba(42, 120, 214, ${(0.06 + 0.22 * Math.min(1, gap / 0.2)).toFixed(3)})`;
  return "";
}
const assumedFor = peak => peak === DATASHEET.bf16 ? ROOFLINE_EFF.bf16 : ROOFLINE_EFF.mxfp8;

// Shading within one table, anchored at 90% of the roofline (white): below it
// red (#e34948), darkest at the table's lowest % of peak; above it green
// (#008300), fullest at the table's highest.
const WHITE_OF_ROOFLINE = 0.9;
function rangeShade(frac, white, lo, hi) {
  if (frac < white - 0.005 && white > lo)
    return `background: rgba(227, 73, 72, ${(0.08 + 0.42 * Math.min(1, (white - frac) / (white - lo))).toFixed(3)})`;
  if (frac > white + 0.005 && hi > white)
    return `background: rgba(0, 131, 0, ${(0.06 + 0.26 * Math.min(1, (frac - white) / (hi - white))).toFixed(3)})`;
  return "";
}

function riTable(table, corner, columns, rows, fmtCell, { graded = false } = {}) {
  // graded: shade every cell by its % of peak relative to the table's range
  const fracs = graded ? rows.flatMap(([, values, , peak]) => peak ? values.filter(v => v != null).map(v => v / peak) : []) : [];
  const lo = Math.min(...fracs), hi = Math.max(...fracs);
  table.replaceChildren();
  const head = el("tr");
  head.append(el("th", {}, corner));
  for (const c of columns) head.append(el("th", { class: "r" }, c));
  table.append(head);
  // a row's fifth entry, when true, shades every cell red (a path to avoid)
  for (const [label, values, shadeLow, peak, allRed] of rows) {
    const tr = el("tr");
    tr.append(el("td", {}, label));
    const best = Math.max(...values.filter(v => v != null));
    for (const v of values) {
      const low = !graded && (allRed || (shadeLow && v < 0.6 * best)) && v != null;
      const shade = graded && v != null && peak
        ? rangeShade(v / peak, WHITE_OF_ROOFLINE * assumedFor(peak), lo, hi) : "";
      const td = el("td", { class: `r ${low ? "up" : ""}`, style: shade }, v == null ? "--" : fmtCell(v));
      if (v != null && peak)
        td.append(el("span", { class: "change", style: "color: var(--text-secondary)" },
                     `${(100 * v / peak).toFixed(0)}% of peak`));
      tr.append(td);
    }
    table.append(tr);
  }
}

// Fraction -> whole percent (the shared pct() takes a value already in percent).
const pctOf = v => `${(100 * v).toFixed(0)}%`;

function renderRooflineIntro() {
  const b = t => el("b", {}, t);
  document.getElementById("rfSub").replaceChildren(
    "Compute time from FLOPs and one assumed efficiency, ", b(`${pctOf(ROOFLINE_EFF.bf16)} of peak in bf16 and `),
    b(`${pctOf(ROOFLINE_EFF.mxfp8)} in MXFP8`), ", for every matmul-like kernel, per layer, microbatch and GPU. ",
    "Section 2 checks it against measurements.");
}

function renderFormulaWhere() {
  // b() takes text or nodes, so formulas with subscripts can be bold
  const S = DATA.model.symbols, i = t => el("i", {}, t);
  const b = (...parts) => { const e = el("b"); e.append(...parts); return e; };
  const sub = (base, x) => { const e = el("i", {}, base); e.append(el("sub", {}, x)); return e; };
  document.getElementById("rfSymbols").replaceChildren(
    "Each term is FLOPs divided by achieved throughput, \u03b7 x ", sub("R", "peak"), ", for one transformer ",
    "layer, one microbatch (one sequence of T tokens) and one GPU. ", i("cp"), " and ", i("tp"), " are the CP and ",
    "TP degrees. ", b("\u03b7 x ", sub("R", "peak")), " is the throughput a GPU is assumed to achieve: ",
    sub("R", "peak"), ` is the datasheet's dense peak rate (${DATASHEET.bf16 / 1000} PFLOPS bf16, `,
    `${DATASHEET.fp8 / 1000} PFLOPS FP8/MXFP8) and \u03b7 the fraction real kernels reach, so \u03b7 x `,
    sub("R", "peak"), " = ",
    b(`${exact(ROOFLINE_EFF.bf16 * DATASHEET.bf16)} TFLOPS in bf16`), " and ",
    b(`${exact(ROOFLINE_EFF.mxfp8 * DATASHEET.fp8)} TFLOPS in MXFP8`), ".");
  document.getElementById("rfDense").replaceChildren(
    "This is the familiar ", b("6 N D"), " (2 FLOPs per parameter per token forward, 4 backward) with ",
    b("N = ", sub("N", "dense"), " / tp"), " and ", b("D = T / cp"), ", and ", b("8"), " instead of 6 ",
    "because full remat replays the forward once more. ", b("T / cp"), " is the tokens one GPU runs through the dense linears: CP splits the sequence across cp GPUs, ",
    "and TP gathers it back before the linears (sequence parallelism). ", b(sub("N", "dense"), " / tp"),
    " is the trainable parameters one GPU holds of the layer's dense linears. TP splits the attention projections ",
    "and shared experts; the router and routed down/up projections keep full weights and instead run on the ",
    "routed tokens TP splits (T / (cp x tp)), like the experts -- the same FLOPs, a different GEMM shape. A linear does ", b("2 FLOPs per parameter "),
    b("per token"), " (a multiply and an add) per GEMM, and each runs ", b("4 GEMMs"), ": the forward, the forward ",
    "again when full remat replays the block in backward, and the two backward GEMMs (input gradient and weight ",
    "gradient). Hence ", b("8 = 2 x 4"), ".");
  const L = S.L, Fe = S.Fe, nW13 = 2 * L * Fe, nW2 = Fe * L, M = n => `${(n / 1e6).toFixed(1)}M`;
  document.getElementById("rfExperts").replaceChildren(
    "Each is again ", b("N D"), " times FLOPs per parameter per token, with ", b("D = T / (cp x tp)"), " -- the ",
    "tokens one GPU sends into the routed experts (TorchTitan folds TP into EP: the TP ranks split the routed ",
    "tokens and join the expert-parallel pool, so expert weights are never split by TP) -- and ", b("N = k x"),
    " the GEMM's parameters per expert, since each token visits ", b(`k = ${S.k} experts`), ". ",
    b("ep cancels per GPU"), ": the second box counts one expert's tokens (the EP group's ep GPUs send ",
    "ep x T / (cp x tp) x k visits, spread over E experts), its FLOPs, and the E / ep experts a GPU hosts. So ",
    "a GPU's total expert work does not depend on ep; ep only sets the tokens per expert, and with it the ",
    "GEMM size that decides whether \u03b7 holds. Per expert, ", b("w13"),
    ` fuses the gate and up projections (L = ${exact(L)} -> 2 x Fe = ${exact(2 * Fe)}, `, sub("N", "w13"),
    ` = ${M(nW13)}) and `, b("w2"), ` projects back down (Fe -> L, `, sub("N", "w2"), ` = ${M(nW2)}). `,
    "Full remat does not replay DistMoE, but DistMoE saves only its inputs and in backward re-dispatches them ",
    "and recomputes ", b("both"), " forwards: w13 to rebuild the activation, and w2 because the router-score ",
    "gradient is the output gradient dotted with each expert's output. So ", b("both cost 8"), " (forward, ",
    "recompute, two backward GEMMs), and the routed experts are another 8 N D with N = k x ", sub("N", "e"),
    ".");
  document.getElementById("rfMla").replaceChildren(
    b("F"), " is causal attention's forward FLOPs on one GPU: Q K", el("sup", {}, "T"), " and P V each cost 2 x T x T ",
    "x heads x head dim, halved because each query attends only to earlier keys, giving T", el("sup", {}, "2"),
    ` x heads x (d_qk + d_v) with d_qk = 192, d_v = 128. Per GPU, TP and Ulysses CP split the ${96} heads, `,
    "leaving ", b("H / (tp x cp)"), " heads on all T keys; KV all-gather CP instead keeps all heads on T / cp ",
    "queries against all keys -- the same FLOPs, written here in the head-split form. ", b("4.5"), " = 1 forward + 2.5 for backward (it recomputes ",
    "the scores and runs four matmuls, the usual FlashAttention count) + 1 forward replayed by full remat.");
  document.getElementById("rfKda").replaceChildren(
    b("KDA is hard to roofline"), ": it is a chunked recurrence of small matmuls, triangular solves and state ",
    "updates, roughly 7x slower than its FLOPs at matmul peak. For now it uses a model fitted to measurements: ",
    b("0.40 us per token per layer"), " for forward + backward with all heads, divided by the measured TP speedup ",
    b("s"), " (TP leaves each GPU fewer heads), plus ", b("0.05 us"), " for the remat forward; a GPU runs ",
    b("T / cp"), " tokens. We know this is not a roofline; a FLOP-and-byte model of the chunk kernel is a TODO.");
}

// 1.6: per-step compute time per GPU from the roofline. Every GPU processes
// global tokens / 512 tokens per step whatever the parallelism, so the FLOP
// terms are the same for every config; only attention (T^2) and KDA's TP
// speedup depend on the config.
// Per-step roofline compute terms (seconds per GPU) at one context and dtype.
function stepTerms(ctx, dt) {
  const M = DATA.model, S = M.symbols, T = M.contexts[ctx].seq_len, tok = M.global_tokens / 512;
  const dense = sc => M.scopes[sc].stored.filter(r => r.key !== "routed").reduce((a, r) => a + r.numel, 0);
  const nKda = M.layers[0].count, nMla = M.layers[1].count;
  const nExpert = M.scopes.kda_layer.stored.find(r => r.key === "routed").numel / S.E;
  const kdaAttn = M.scopes.kda_layer.stored.find(r => r.key === "attention").numel;
  const rate = d => ROOFLINE_EFF[d] * (d === "bf16" ? DATASHEET.bf16 : DATASHEET.fp8) * 1e12;
  return {
    dense: 8 * tok * (nKda * dense("kda_layer") + nMla * dense("mla_layer")) / rate(dt),
    experts: 8 * tok * 92 * S.k * nExpert / rate(dt),
    layer0: 8 * tok * (kdaAttn + 3 * S.D * S.F) / rate(dt),
    lmhead: 6 * tok * S.V * S.D / rate(dt),
    mla: 24 * 4.5 * tok * T * S.H * (S.Dn + S.Dr + S.Dv) / rate("bf16"),
    kda: 69 * tok * (KDA_MODEL.kda.fwd_bwd + KDA_MODEL.kda.fwd) * 1e-6,
  };
}

// 2.5: the fake-PG step against the roofline, by op class, with the reasons.
const FP_CLASS = {
  gemm: "GEMMs (dense; non-DistMoE expert GEMMs too)", distmoe: "Routed experts: DistMoE kernels",
  mla: "Attention: MLA (SDPA / flash)", kda: "Attention: KDA",
  fsdp: "FSDP buffer copies (chunk_cat / split copy-out)", copies: "Copies, casts, cat, memset",
  norms: "Norms, softmax (RMSNorm, AttnRes depth mixing)", tokens: "MoE token movement (index / sort / scatter)",
  elem: "Elementwise and other fused Triton", opt: "Optimizer (Muon / AdamW updates)", other: "other",
};
const fpClass = (r, k) => r.breakdown.by_class_s[FP_CLASS[k]] ?? 0;
const fpIdle = r => r.clean_step_s - r.breakdown.gpu_busy_s;
const fpOutside = r => ["fsdp", "copies", "norms", "tokens", "elem", "opt", "other"].reduce((a, k) => a + fpClass(r, k), 0);

// Pie slices: the categorical order of the memory page; idle in neutral gray.
const FP_SLICES = [
  ["Routed experts (DistMoE)", "#2a78d6", r => fpClass(r, "distmoe")],
  ["Dense GEMMs", "#1baf7a", r => fpClass(r, "gemm")],
  ["MLA attention", "#eb6834", r => fpClass(r, "mla")],
  ["KDA attention", "#eda100", r => fpClass(r, "kda")],
  ["FSDP staging copies", "#4a3aa7", r => fpClass(r, "fsdp")],
  ["Other memory-bound ops (copies, casts, norms, elementwise, optimizer)", "#e87ba4",
   r => fpOutside(r) - fpClass(r, "fsdp")],
  ["GPU idle", "#c3c2b7", r => fpIdle(r)],
];

function renderFpPies(runs) {
  const host = document.getElementById("fpPies");
  host.replaceChildren();
  const row = el("div", { style: "display: flex; flex-wrap: wrap; gap: 24px 56px; align-items: flex-start" });
  const R0 = 92, R1 = 54, W = 2 * R0 + 150;
  for (const [title, r] of runs) {
    const total = r.clean_step_s, box = el("div");
    box.append(el("div", { style: "font-weight: 600; font-size: 14px; margin-bottom: 2px" }, title),
               el("div", { style: "color: var(--text-secondary); font-size: 13px" }, `${total.toFixed(1)} s per step`));
    const svg = sel("svg", { width: W, height: 2 * R0 + 24, role: "img", "aria-label": `${title}: time per step by op class` });
    const cx = W / 2, cy = R0 + 12;
    let ang = -Math.PI / 2;
    for (const [label, color, f] of FP_SLICES) {
      const v = f(r), da = 2 * Math.PI * v / total;
      if (v <= 0) continue;
      const a0 = ang, a1 = ang + da, large = da > Math.PI ? 1 : 0;
      const pt = (rad, t) => `${(cx + rad * Math.cos(t)).toFixed(2)} ${(cy + rad * Math.sin(t)).toFixed(2)}`;
      const g = sel("g", { class: "bar", tabindex: 0 });
      g.append(sel("path", {
        d: `M ${pt(R0, a0)} A ${R0} ${R0} 0 ${large} 1 ${pt(R0, a1)} L ${pt(R1, a1)} A ${R1} ${R1} 0 ${large} 0 ${pt(R1, a0)} Z`,
        fill: color, stroke: "var(--surface-1)", "stroke-width": 2 }));
      attachTip(g, () => [{ text: label, color }, { text: `${v.toFixed(2)} s`, cls: "v" },
                          { text: `${(100 * v / total).toFixed(0)}% of the step`, cls: "k" }]);
      svg.append(g);
      // direct label for slices of at least 6%, outside the ring
      if (v / total >= 0.06) {
        const mid = (a0 + a1) / 2, lx = cx + (R0 + 8) * Math.cos(mid), ly = cy + (R0 + 8) * Math.sin(mid);
        svg.append(sel("text", { x: lx.toFixed(1), y: (ly + 4).toFixed(1), class: "vallabel",
                                 "text-anchor": Math.cos(mid) >= 0 ? "start" : "end" }, `${(100 * v / total).toFixed(0)}%`));
      }
      ang = a1;
    }
    svg.append(sel("text", { x: cx, y: cy + 5, class: "rowlabel", "text-anchor": "middle" }, `${total.toFixed(1)} s`));
    box.append(svg);
    row.append(box);
  }
  const legend = el("div", { class: "legend" });
  for (const [label, color] of FP_SLICES) {
    const item = el("span", { style: "display: inline-flex; align-items: center; gap: 6px" });
    item.append(el("span", { style: `width: 10px; height: 10px; border-radius: 2px; background: ${color}; display: inline-block` }), label);
    legend.append(item);
  }
  host.append(legend, row);
}

function renderFakePg() {
  const F = DATA.bench.fake_pg, b = t => el("b", {}, t);
  if (!F) return;
  const sum = o => Object.values(o).reduce((a, x) => a + x, 0);
  const rf8 = stepTerms("8K", "bf16"), rf32 = stepTerms("32K", "bf16");
  const l8 = F.distmoe_longdocs_8k, l32 = F.distmoe_longdocs_32k_cp2;
  const gemms = rf => rf.dense + rf.experts + rf.layer0 + rf.lmhead;
  // [label, roofline 8K, measured fn, roofline 32K]
  const rows = [
    ["Dense GEMMs (incl. layer 0 and LM head)", gemms(rf8) - rf8.experts, r => fpClass(r, "gemm"), gemms(rf32) - rf32.experts],
    ["Routed-expert GEMMs (DistMoE)", rf8.experts, r => fpClass(r, "distmoe"), rf32.experts],
    ["MLA attention", rf8.mla, r => fpClass(r, "mla"), rf32.mla],
    ["KDA attention", rf8.kda, r => fpClass(r, "kda"), rf32.kda],
    ["Work outside the roofline", 0, fpOutside, 0],
    ["  FSDP staging copies (all-gather / reduce-scatter)", 0, r => fpClass(r, "fsdp"), 0],
    ["  other copies, casts, cat, memset", 0, r => fpClass(r, "copies"), 0],
    ["  norms and softmax (RMSNorm, AttnRes depth mixing)", 0, r => fpClass(r, "norms"), 0],
    ["  elementwise, optimizer, MoE token movement, other", 0,
     r => fpClass(r, "elem") + fpClass(r, "opt") + fpClass(r, "tokens") + fpClass(r, "other"), 0],
    ["GPU idle (step minus GPU busy time)", 0, fpIdle, 0],
  ];
  const table = document.getElementById("fpTable");
  table.replaceChildren();
  const head = el("tr");
  for (const [h, r] of [["Seconds per step, FSDP + EP 64, bf16", 0], ["8K roofline (1.6)", 1], ["8K, measured", 1], ["32K roofline (1.6)", 1], ["32K CP 2, measured", 1]])
    head.append(el("th", r ? { class: "r" } : {}, h));
  table.append(head);
  const cell = v => el("td", { class: "r", style: "white-space: nowrap" }, v == null ? "--" : v === 0 ? "-" : `${v.toFixed(2)} s`);
  for (const [label, r8, f, r32] of rows) {
    const tr = el("tr");
    tr.append(el("td", {}, label), cell(r8), cell(f(l8)), cell(r32), cell(f(l32)));
    table.append(tr);
  }
  const tr = el("tr"), lab = el("td");
  lab.append(b("Step"));
  tr.append(lab);
  for (const v of [sum(rf8), l8.clean_step_s, sum(rf32), l32.clean_step_s]) {
    const td = el("td", { class: "r" }); td.append(b(`${v.toFixed(1)} s`)); tr.append(td);
  }
  table.append(tr);

  renderFpPies([["8K", l8], ["32K, CP 2", l32]]);

  document.getElementById("fpSub").replaceChildren(
    b(`Roofline: ${sum(rf8).toFixed(1)} s per step at 8K, ${sum(rf32).toFixed(1)} s at 32K`), b(" (1.6, bf16)."),
    el("br"), b(`Measured: ${l8.clean_step_s.toFixed(1)} s at 8K and ${l32.clean_step_s.toFixed(1)} s at 32K CP 2.`), el("br"),
    "TorchTitan's real training step for one GPU, run in fake-process-group mode: one GB300 stands in for ",
    "logical rank 0 of 512 with FSDP + EP 64 (16M tokens per step, full remat, fp32 master weights with bf16 ",
    "compute, DistMuon), executing its real kernels while collectives move no data. 8K runs 4 microbatches per ",
    "step; 32K runs Ulysses CP 2 (MLA) and KDA's CP path, 2 microbatches. Routing is round-robin balanced, as the ",
    "roofline assumes. DistMoE runs a stand-in model: SwiGLU, since DistMoE does not yet support SiTU-GLU, and ",
    "1,024 routed experts (16 per GPU; same expert FLOPs per token). The data is synthetic documents longer ",
    "than the window, so attention covers the full context as the roofline assumes. ",
    "Step times are steady-state steps; the split by op class comes from profiling one more step.");
  const ul = el("ul", { style: "margin: 6px 0 0 18px; padding: 0" });
  const li = (...parts) => { const e = el("li", { style: "margin-bottom: 4px" }); e.append(...parts); ul.append(e); };
  const kdaCp = fpClass(l32, "kda"), kda8 = fpClass(l8, "kda");
  li(b("The kernels the roofline models are close to it. "),
     `GEMMs take ${(fpClass(l8, "gemm") + fpClass(l8, "distmoe")).toFixed(1)} s at 8K and `,
     `${(fpClass(l32, "gemm") + fpClass(l32, "distmoe")).toFixed(1)} s at 32K against the roofline's `,
     `${gemms(rf8).toFixed(1)} s. The split differs: dense GEMMs run slower than the roofline `,
     `(${fpClass(l8, "gemm").toFixed(1)} s against ${(rf8.dense + rf8.layer0 + rf8.lmhead).toFixed(1)} s; wgrad and the small `,
     "projections sit below 70%, and the class also holds GEMV kernels outside the roofline's FLOPs), while DistMoE's ",
     `GEMM kernels run faster (${fpClass(l8, "distmoe").toFixed(1)} s against ${rf8.experts.toFixed(1)} s, 73% of peak). `,
     "That is a fake-PG effect: with one real rank, DistMoE's dispatch reads no peer memory over NVLink and its ",
     "inputs are 64 aliased copies of the same 128 tokens, which stay in cache; at real EP 4 the same GEMM kernels ",
     "reach 60-64% (2.2). MLA (cuDNN SDPA in these runs) is within about 10% of its roofline term at 32K: ",
     `${fpClass(l32, "mla").toFixed(1)} s against ${rf32.mla.toFixed(1)} s.`);
  li(b(`KDA under CP is far over its roofline term: ${kdaCp.toFixed(1)} s against ${rf32.kda.toFixed(1)} s at 32K CP 2. `),
     `Without CP it takes ${kda8.toFixed(1)} s at 8K, close to the model. Under CP, KDA runs a different kernel `,
     "set (chunked kernels plus a cross-rank summary pass); the CP benchmark in 2.3 measures that at 1.7x the ",
     "time of the same tokens without CP at 32K CP 2, which accounts for about 1.7 s; the remaining ~0.7 s is ",
     "not yet explained (likely the multi-document kernel variants the single-document benchmark does not run).");
  li(b(`About ${fpOutside(l8).toFixed(1)} s (8K) and ${fpOutside(l32).toFixed(1)} s (32K) of GPU work is outside the roofline. `),
     "FSDP's all-gather and reduce-scatter staging copies, dtype casts and concatenations, RMSNorm and AttnRes ",
     "depth-mixing softmax, elementwise ops and the optimizer -- memory-bound work that FLOPs do not count. FSDP's ",
     `staging copies are per microbatch, so they halve at 32K (${fpClass(l8, "fsdp").toFixed(1)} s -> `,
     `${fpClass(l32, "fsdp").toFixed(1)} s, 4 -> 2 microbatches).`);
  li(b(`The GPU sits idle for ${fpIdle(l8).toFixed(1)} s (8K) and ${fpIdle(l32).toFixed(1)} s (32K) of each step. `),
     `The step launches about ${Math.round(l8.breakdown.kernels / 1000)}K kernels at 8K and `,
     `${Math.round(l32.breakdown.kernels / 1000)}K at 32K in eager mode with CUDA graphs off, so the host cannot keep `,
     "the GPU fed; this is likely the largest single fixable item (CUDA graphs or compilation).");
  li(b("Caveats. "), "Collectives move no data, so communication time is absent (FSDP's staging copies are ",
     "real); the DistMoE run is a stand-in model; MLA ran on cuDNN's SDPA flash kernel rather than FA4; one GPU, ",
     "one parallelism layout per context.");
  document.getElementById("fpWhy").replaceChildren(b("Why the measured step is slower than the roofline:"), ul);
}
function renderStepCompute() {
  const M = DATA.model, S = M.symbols, b = t => el("b", {}, t);
  const tok = M.global_tokens / 512;  // tokens per GPU per step
  const dense = sc => M.scopes[sc].stored.filter(r => r.key !== "routed").reduce((a, r) => a + r.numel, 0);
  const nKda = M.layers[0].count, nMla = M.layers[1].count;  // MoE layers by attention type
  const nExpert = M.scopes.kda_layer.stored.find(r => r.key === "routed").numel / S.E;
  const kdaAttn = M.scopes.kda_layer.stored.find(r => r.key === "attention").numel;
  const layer0 = kdaAttn + 3 * S.D * S.F;  // KDA attention + dense FFN, no MoE
  const rate = dt => ROOFLINE_EFF[dt] * (dt === "bf16" ? DATASHEET.bf16 : DATASHEET.fp8) * 1e12;
  const rows = [], cols = [];
  for (const ctx of Object.keys(M.contexts)) for (const dt of ["bf16", "mxfp8"]) cols.push([ctx, dt]);
  const terms = [
    ["Dense linears (68 KDA + 24 MLA layers)", (T, dt) => 8 * tok * (nKda * dense("kda_layer") + nMla * dense("mla_layer")) / rate(dt)],
    ["Routed experts (92 MoE layers)", (T, dt) => 8 * tok * 92 * S.k * nExpert / rate(dt)],
    ["Layer 0 (KDA attention + dense FFN linears)", (T, dt) => 8 * tok * layer0 / rate(dt)],
    ["LM head + chunked loss", (T, dt) => 6 * tok * S.V * S.D / rate(dt)],
    ["MLA attention (24 layers, bf16)", (T, dt) => 24 * 4.5 * tok * T * S.H * (S.Dn + S.Dr + S.Dv) / rate("bf16")],
    ["KDA attention (69 layers, measured model, TP 1)", (T, dt) => 69 * tok * (KDA_MODEL.kda.fwd_bwd + KDA_MODEL.kda.fwd) * 1e-6],
  ];
  const vals = terms.map(([label, f]) => [label, cols.map(([ctx, dt]) => f(M.contexts[ctx].seq_len, dt))]);
  const total = cols.map((_, j) => vals.reduce((a, [, v]) => a + v[j], 0));
  const table = document.getElementById("rfStep");
  table.replaceChildren();
  const head = el("tr");
  head.append(el("th", {}, "Seconds per step per GPU"));
  for (const [ctx, dt] of cols) head.append(el("th", { class: "r" }, `${ctx}, ${dt === "bf16" ? "bf16" : "MXFP8"}`));
  table.append(head);
  const sec = v => v.toFixed(2);
  for (const [label, v] of vals) {
    const tr = el("tr");
    tr.append(el("td", {}, label));
    v.forEach((x, j) => tr.append(el("td", { class: "r" }, `${sec(x)} s`, )));
    table.append(tr);
  }
  const tr = el("tr");
  const totalLabel = el("td");
  totalLabel.append(b("Total compute"));
  tr.append(totalLabel);
  total.forEach(x => { const td = el("td", { class: "r" }); td.append(b(`${sec(x)} s`)); tr.append(td); });
  table.append(tr);
  const kdaTp = t => `${(KDA_MODEL.kda.fwd_bwd * (t / KDA_MODEL.kda.tp_speedup.by_tier[1][KDA_MODEL.kda.tp.indexOf(t)]) / KDA_MODEL.kda.fwd_bwd).toFixed(2)}x`;
  document.getElementById("rfStepSub").replaceChildren(
    b("The same for every config."), ` Each GPU processes ${exact(tok)} tokens per step (${exact(M.global_tokens)} `,
    "over 512 GPUs) whatever the parallelism: PP splits the layers but multiplies the microbatches by the same ",
    "factor, and CP and TP split tokens or weights without adding FLOPs. So the FLOP terms depend only on the ",
    "context length (MLA's T^2) and the dtype; configs differ through communication, the PP bubble and the ",
    "efficiencies of section 2. One exception: KDA's TP speedup is below linear, so TP multiplies the KDA term ",
    `by tp / s (${kdaTp(2)} / ${kdaTp(4)} / ${kdaTp(8)} at TP 2 / 4 / 8 at 32K). MXFP8 speeds up only the dense, `,
    `expert and LM-head GEMMs (${exact(ROOFLINE_EFF.mxfp8 * DATASHEET.fp8)} vs ${exact(ROOFLINE_EFF.bf16 * DATASHEET.bf16)} TFLOPS); attention stays bf16. Non-matmul `,
    "ops (norms, activations, AttnRes depth mixing) and the optimizer step are not included.");
}

// How far the roofline (one assumed efficiency) is from each measured point:
// roofline time / measured time - 1 = measured efficiency / assumed - 1
// (negative = roofline optimistic).
function renderFormulaGap() {
  const K = DATA.bench.kernels, A = DATA.bench.attention_shapes?.attention_shapes?.data;
  const peak = dt => dt === "bf16" ? DATASHEET.bf16 : DATASHEET.fp8, E = ROOFLINE_EFF;
  const gap = (tflops, dt) => tflops / peak(dt) / E[dt] - 1;
  const g = { denseBig: [], denseSmall: [], expBig: [], expSmall: [], mla: [] };
  for (const dt of ["bf16", "mxfp8"]) {
    for (const byM of Object.values(K.linears[dt]))
      for (const [m, v] of Object.entries(byM)) (+m >= 2048 ? g.denseBig : g.denseSmall).push(gap(v, dt));
    for (const r of DATA.bench.dist_moe?.results ?? [])
      if (r.dtype === dt && r.fwd_bwd_gemm_tflops) (r.local_tokens >= 4096 ? g.expBig : g.expSmall).push(gap(r.fwd_bwd_gemm_tflops, dt));
  }
  for (const t of [8192, 32768]) {
    const F4 = DATA.bench.mla_fa4;
    if (F4) for (const h of [96, 48, 24, 12]) g.mla.push(gap(F4.heads[t][h].fwd_bwd_tflops, "bf16"));
  }
  const range = v => {
    const f = x => `${x >= 0 ? "+" : "\u2212"}${Math.abs(100 * x).toFixed(0)}%`;
    return `${f(Math.min(...v))} to ${f(Math.max(...v))}`;
  };
  const b = t => el("b", {}, t);
  document.getElementById("riFormulaGap").replaceChildren(
    el("span", { class: "note-label" }, "How far from the roofline:"),
    " measured kernel time against the roofline's, as roofline / measured - 1 (negative = roofline too ",
    "optimistic). Dense GEMMs: ", b(range(g.denseBig)), " at 2K+ rows, ", b(range(g.denseSmall)), " at 1K. ",
    "Routed experts (DistMoE GEMM kernels, BF16 and MXFP8): ", b(range(g.expBig)), " at 4K+ local tokens, ", b(range(g.expSmall)),
    " below. ",
    "MLA with FA4 (8K and 32K, 12-96 heads): ", b(range(g.mla)), ". The roofline holds for big GEMMs, is ",
    "optimistic for small ones and for attention. How to reconcile the difference is left for later.");
}

// Bold "Roofline: ..." lead for each section-2 subsection.
function rooflineLead(dtypes) {
  const parts = dtypes.map(dt => `${exact(ROOFLINE_EFF[dt] * (dt === "bf16" ? DATASHEET.bf16 : DATASHEET.fp8))} TFLOPS in ` +
    `${dt === "bf16" ? "bf16" : "MXFP8"} (${(100 * ROOFLINE_EFF[dt]).toFixed(0)}% of peak)`);
  return el("b", { style: "display: block" }, `Roofline: ${parts.join(", ")}.`);
}

// Bold "Where it breaks: ..." lead after the roofline number.
// Both leads render as their own lines; the explanation follows as a new paragraph.
const breaksLead = text => el("b", { style: "display: block; margin-bottom: 8px" }, `Where it breaks: ${text}`);

function renderRooflineInputs() {
  renderFakePg();
  renderRooflineIntro();
  renderStepCompute();
  renderFormulaWhere();
  renderFormulaGap();
  const R = DATA.bench, sub = document.getElementById("riSub");
  if (!R) { sub.textContent = "No bench results were found when this page was generated."; return; }
  const K = R.kernels, b = t => el("b", {}, t);
  sub.replaceChildren(
    `Kernel benchmarks on one ${K.gpu.replace("NVIDIA ", "")} of a shared devserver, checking the roofline's `,
    `assumption. Compute cells are white at 90% of the roofline (${(90 * ROOFLINE_EFF.bf16).toFixed(1)}% of peak in bf16, ${(90 * ROOFLINE_EFF.mxfp8).toFixed(0)}% in MXFP8), red `,
    "below it (darkest at the table's lowest value) and green above it (fullest at the table's highest). ",
    "KDA and NVLink are unshaded. Percentages are of the GB200 datasheet's dense peak (2.5 PFLOPS bf16, 5 PFLOPS ",
    "FP8/MXFP8, 900 GB/s NVLink per direction). GEMMs are timed as CUDA-graph replays; runs that shared the ",
    "GPU will be re-measured."
  );

  const tok = n => n >= 1024 ? `${n / 1024}K` : String(n);
  const tf = v => v.toFixed(0);
  const pct = v => `${(100 * v).toFixed(0)}%`;
  const peakOf = dt => dt === "bf16" ? DATASHEET.bf16 : DATASHEET.fp8;
  document.getElementById("riLinSub").replaceChildren(
    rooflineLead(["bf16", "mxfp8"]),
    breaksLead("slightly everywhere in bf16 (64-69% of peak at 2K+ GEMM rows, T / cp: wgrad and the small " +
               "projections sit below 70%), more below 2K (58-64% at 1K; MXFP8 45-58%); TP narrowing the attention " +
               "and shared-expert weights to N / tp is not measured yet."),
    "Measured by the GEMM's rows M = T / CP, covering the forward and both backward GEMMs of the KDA-layer, ",
    "MLA-layer and LM-head linears. MXFP8 dips at 1K, where a big projection finishes in about 60 us and fixed ",
    "per-kernel costs weigh twice as much as in bf16. TP does not shrink M (sequence parallelism ",
    "gathers the tokens first); it narrows the attention and shared-expert weights to N / TP, an effect not ",
    "measured yet.");
  const lin = Object.keys(K.linears.bf16.kda_layer).map(Number).sort((x, y) => x - y);
  const scopes = [["kda_layer", "KDA + MoE layer"], ["mla_layer", "MLA + MoE layer"], ["lm_head", "LM head"]];
  riTable(document.getElementById("riLinears"), "Dense linears, TFLOPS / GEMM rows M = seq len / CP", lin.map(tok),
    ["bf16", "mxfp8"].flatMap(dt => scopes.map(([k, label]) =>
      [`${label}, ${dt === "bf16" ? "bf16" : "MXFP8"}`, lin.map(m => K.linears[dt][k][m]), false, peakOf(dt)])),
    tf, { graded: true });

  // routed experts: DistMoE (the kernel the model uses)
  const S = DATA.model.symbols, DM = R.dist_moe?.results;
  const locals = [1024, 2048, 4096, 8192];
  document.getElementById("riExSub").replaceChildren(
    rooflineLead(["bf16", "mxfp8"]),
    breaksLead("everywhere in MXFP8 (GEMM kernels at most 43% of peak); in bf16 the GEMM kernels reach 60-67% " +
               "at 2K+ local tokens (T / (cp x tp)), 44-56% at 1K."),
    "The routed experts run on ", b("DistMoE"), ": fused dispatch + grouped GEMMs + combine, in its default ",
    "saved-inputs mode, so forward + backward includes the recompute (FLOPs counted as 4x forward). Measured at ",
    "EP 4 over this host's NVLink with Kimi K3's own local expert counts, 14 (EP 64) and 28 (EP 32), balanced ",
    "routing and 25% scratch headroom (tokens x top-k / experts does not divide evenly, and DistMoE faults rather ",
    "than raising an error when an expert overflows its scratch capacity). The table counts ", b("only the fused GEMM "),
    b("kernels' time"), " (dispatch reads and combine writes included), since the roofline counts only GEMM FLOPs. ",
    "DistMoE's helper kernels -- routing metadata, SwiGLU, scale-and-sum, adds, memsets -- are a roughly fixed ",
    "cost per call, shown in the last row: about 2 ms at 4K+ local tokens (10-15% of the call) but 3-7 ms at 1K-2K, ",
    "where they take most of the call; in the whole-step breakdown (2.5) they count as work outside the roofline. ",
    "Kernel times are the minimum over repeats, since other jobs share the GPU.");
  if (DM) {
    const at = (layout, dt, n) => DM.find(r => r.layout === layout && r.dtype === dt && r.local_tokens === n);
    const cell = (layout, dt) => locals.map(n => at(layout, dt, n)?.fwd_bwd_gemm_tflops ?? null);
    riTable(document.getElementById("riExperts"), "DistMoE GEMM kernels, forward + backward, TFLOPS / local seq length", locals.map(tok), [
      ["BF16, 14 local experts (EP 64)", cell("ep64", "bf16"), false, DATASHEET.bf16],
      ["BF16, 28 local experts (EP 32)", cell("ep32", "bf16"), false, DATASHEET.bf16],
      ["MXFP8, 14 local experts (EP 64)", cell("ep64", "mxfp8"), false, DATASHEET.fp8],
      ["MXFP8, 28 local experts (EP 32)", cell("ep32", "mxfp8"), false, DATASHEET.fp8],
    ], tf, { graded: true });
    const helper = document.createElement("tr"), ms = n => at("ep64", "bf16", n)?.fwd_bwd_helper_ms;
    helper.append(el("td", {}, "Helper kernels, ms per call (BF16, 14 local experts; not in the TFLOPS above)"),
                  ...locals.map(n => el("td", { class: "r" }, ms(n) == null ? "--" : `${ms(n).toFixed(1)} ms`)));
    document.getElementById("riExperts").append(helper);
  }
  document.getElementById("riAttnSub").replaceChildren(
    rooflineLead(["bf16"]),
    breaksLead("everywhere for forward + backward (36-52% of peak); worst with few heads at short sequences " +
               "(12 heads: 25% at 8K) and for KV all-gather CP at short sequences (CP 8: 24% at 8K)."),
    b("MLA"), " runs through TorchTitan's varlen path, FlashAttention 4. Forward + backward reaches 36-52% of ",
    "peak, rising with sequence length T; the forward alone 56-66%. The two CP modes give a rank different ",
    "shapes: ", b("Ulysses CP"), " (and TP) leaves the full sequence on 96 / (tp x cp) heads, which holds up at ",
    "16K+ but falls off with few heads at 8K; ", b("KV all-gather CP"), " leaves all heads on the rank's T / cp ",
    "load-balanced queries against all keys, measured here with FA4 (TorchTitan's KV all-gather CP currently ",
    "runs FlexAttention, whose performance on MLA's head dims was very poor). At 32K both modes stay within ",
    "about 10% of each other; at 8K, high CP costs either mode."
  );
  document.getElementById("riKdaSub").replaceChildren(
    el("b", { style: "display: block; margin-bottom: 8px" }, "Roofline: none -- KDA is costed from its measured model (1.4)."),
    b("KDA"), " is a chunked recurrence of small matmuls, not a big GEMM, so it is costed in time per token ",
    `per layer: ${KDA_MODEL.kda.fwd_bwd} us forward + backward with all 96 heads, divided by the measured `,
    "TP speedup rounded down (1.8x / 3x / 5x at 32K for TP 2 / 4 / 8, 1.7x at 8K), plus ",
    `${KDA_MODEL.kda.fwd} us flat for the remat forward. 8K at TP 4 / 8 keeps TP 2's speedup until a clean `,
    "re-run. ", b("Under CP"), " each rank keeps all heads on its T / cp tokens but runs a different kernel set: ",
    "chunked kernels plus a summary pass whose state is exchanged across ranks. The second table measures it ",
    "(Attention Gym's context_parallel_kda, one T-token document split across cp GPUs, GPU kernel time ",
    "including NCCL) against the same T / cp tokens without CP: the forward costs 2.6-3.7x, forward + backward ",
    "1.3-2x, so CP's KDA speedup is well short of cp."
  );

  // MLA: rule and measured, % of bf16 peak (cells show TFLOPS and % of peak)
  // attention_shapes JSON: {attention_shapes: {contended_by, data}}
  const A = R.attention_shapes?.attention_shapes?.data, seq = [4096, 8192, 16384, 32768];
  const at = (obj, t) => obj && obj[t] ? obj[t] : null;
  const F4 = R.mla_fa4, mlaRows = [];
  if (F4) {
    mlaRows.push(["96 heads, forward + backward", seq.map(t => F4.heads[t]?.[96]?.fwd_bwd_tflops ?? null), false, DATASHEET.bf16]);
    mlaRows.push(["96 heads, forward", seq.map(t => F4.heads[t]?.[96]?.fwd_tflops ?? null), false, DATASHEET.bf16]);
    for (const d of [2, 4, 8])
      mlaRows.push([`${96 / d} heads (TP ${d} or Ulysses CP ${d}), fwd + bwd`,
                    seq.map(t => F4.heads[t]?.[96 / d]?.fwd_bwd_tflops ?? null), false, DATASHEET.bf16]);
    for (const c of [2, 4, 8])
      mlaRows.push([`KV all-gather CP ${c} (96 heads, T / ${c} queries), fwd + bwd`,
                    seq.map(t => F4.kv_allgather_cp[t]?.[c]?.fwd_bwd_tflops ?? null), false, DATASHEET.bf16]);
  }
  riTable(document.getElementById("riAttn"), "MLA attention (FA4), TFLOPS / sequence length T", seq.map(tok), mlaRows, tf,
          { graded: true });

  // KDA: us per local token per layer, by TP
  if (A) {
    const tps = [1, 2, 4, 8], us = (t, tp, key) => A.kda_tp[t]?.[tp] ? 1e3 * A.kda_tp[t][tp][key] / t : null;
    const KR = KDA_MODEL.kda, kRule = (t, tp) => {
      const tier = KR.tp_speedup.tiers.filter(x => t >= x).length - 1;
      return KR.fwd_bwd / KR.tp_speedup.by_tier[Math.max(0, tier)][KR.tp.indexOf(tp)];
    };
    riTable(document.getElementById("riKda"), "KDA, us per sequence token per layer (96 / TP heads) / TP degree", tps.map(tp => `TP ${tp}`), [
      ["KDA model, forward + backward, 8K", tps.map(tp => kRule(8192, tp)), false],
      ["Measured, forward + backward, 8K", tps.map(tp => us(8192, tp, "fwd_bwd_ms")), false],
      ["KDA model, forward + backward, 32K", tps.map(tp => kRule(32768, tp)), false],
      ["Measured, forward + backward, 32K", tps.map(tp => us(32768, tp, "fwd_bwd_ms")), false],
      ["KDA model, remat forward, 8K and 32K", tps.map(() => KR.fwd), false],
      ["Measured, forward (remat replay), 8K", tps.map(tp => us(8192, tp, "fwd_ms")), false],
      ["Measured, forward (remat replay), 32K", tps.map(tp => us(32768, tp, "fwd_ms")), false],
    ], v => v.toFixed(3));
  }

  // KDA under CP: time relative to the same local tokens as a standalone document
  if (Object.keys(R.kda_cp ?? {}).length) {
    const seqs = [8192, 16384, 32768], kRows = [];
    for (const cp of [2, 4]) {
      const at = t => R.kda_cp[cp]?.find(r => r.seq_len === t);
      kRows.push([`CP ${cp}, forward + backward`, seqs.map(t => at(t) ? at(t).cp_fwd_bwd_ms / at(t).local_fwd_bwd_ms : null), false]);
      kRows.push([`CP ${cp}, forward (remat replay)`, seqs.map(t => at(t) ? at(t).cp_fwd_ms / at(t).local_fwd_ms : null), false]);
      kRows.push([`CP ${cp}, forward + backward + remat forward`, seqs.map(t => at(t) ?
        (at(t).cp_fwd_bwd_ms + at(t).cp_fwd_ms) / (at(t).local_fwd_bwd_ms + at(t).local_fwd_ms) : null), false]);
    }
    riTable(document.getElementById("riKdaCp"), "KDA under CP, time / same local tokens without CP, by sequence length T",
            seqs.map(tok), kRows, v => `${v.toFixed(2)}x`);
  }

  const sizes = [2 ** 16, 2 ** 20, 2 ** 24, 2 ** 27, 2 ** 30, 2 ** 32];  // ** not <<: 1 << 32 wraps
  const sizeLabel = n => n >= 2 ** 30 ? `${n / 2 ** 30} GiB` : n >= 2 ** 20 ? `${n / 2 ** 20} MiB` : `${n / 1024} KiB`;
  const ops = [["allgather", "All-gather"], ["reducescatter", "Reduce-scatter"], ["alltoall", "All-to-all"], ["sendrecv", "Send/recv"]];
  riTable(document.getElementById("riNvlink"), "4 GPUs, busbw GB/s / message size", sizes.map(sizeLabel),
    ops.filter(([k]) => R.nvlink[k]).map(([k, label]) =>
      [label, sizes.map(n => R.nvlink[k].find(r => r[0] === n)?.[2] ?? null), false, DATASHEET.nvlink]), tf);

  const lat = Math.min(...ops.filter(([k]) => R.nvlink[k]).map(([k]) => R.nvlink[k][0][1]));
  document.getElementById("riNote").replaceChildren(
    el("span", { class: "note-label" }, "Note:"),
    el("b", { style: "display: block; margin-bottom: 6px" }, "Roofline: none yet (communication comes later)."),
    "NVLink reaches about 600 GB/s busbw for large collectives (two thirds of 900 GB/s) with a ",
    b(`${lat.toFixed(0)} us`), " latency floor; send/recv stalls near 170 GB/s at 16-128 MiB, where PP sends ",
    "sit, so it needs a clean re-run. IB was not measured (one host); this host has 100 GB/s per GPU, and the ",
    "target cluster's NIC rate still needs confirming."
  );
}

// ---------- section 3: communication links ----------
const cfgName = r => `${r.label}${r.cp > 1 ? `, CP ${r.cp}` : ""}${r.tp > 1 ? `, TP ${r.tp}` : ""}`;

function renderComms() {
  const Lk = DATA.links, b = t => el("b", {}, t);
  if (!Lk) return;
  document.getElementById("cmSub").replaceChildren(
    b(`Topology: ${Lk.num_gpus} GPUs in ${Lk.num_gpus / Lk.nvlink_domain} NVLink domains of ${Lk.nvlink_domain}. `),
    "Inside a domain GPUs talk over NVLink (900 GB/s per GPU per direction on the datasheet); between domains ",
    `over InfiniBand, taken as ${Lk.ib_gbps} GB/s per GPU per direction. NVLink times use the bus bandwidth `,
    "measured with nccl-tests on this host's 4 GPUs at each message size (small messages are latency-bound); ",
    "the same per-GPU bus bandwidth is assumed for groups of up to 64 GPUs on the NVLink switch. InfiniBand is ",
    "flat at the assumed rate. A collective over n GPUs moves (n - 1) / n of its buffer per GPU; a ring that ",
    "crosses domains runs at the InfiniBand rate.");
  const sizes = Lk.nvlink.allgather.map(r => r[0]);
  const lbl = n => n >= 2 ** 30 ? `${n / 2 ** 30} GiB` : `${n / 2 ** 20} MiB`;
  const ops = [["allgather", "NVLink all-gather"], ["reducescatter", "NVLink reduce-scatter"],
               ["alltoall", "NVLink all-to-all"], ["sendrecv", "NVLink send/recv"]];
  riTable(document.getElementById("cmLinks"), "Bus bandwidth per GPU, GB/s / message size", sizes.map(lbl), [
    ...ops.filter(([k]) => Lk.nvlink[k]).map(([k, l]) => [l, Lk.nvlink[k].map(r => r[1]), false, DATASHEET.nvlink]),
    ["InfiniBand (assumed)", sizes.map(() => Lk.ib_gbps), false],
  ], v => v.toFixed(0));

  document.getElementById("cmPlaceSub").replaceChildren(
    b("Which link each parallelism uses. "), "EP, CP and TP groups are placed inside a domain. A PP stage ",
    "(64 GPUs at PP 8, 32 at PP 16) also sits inside one, so FSDP within a stage is NVLink and the hops between ",
    "stages in different domains are InfiniBand. Without PP, FSDP shards the dense weights over all 512 GPUs and ",
    "replicates each expert shard over the 8 domains (EP 64 x 8), so both its rings cross InfiniBand.");
  const t = document.getElementById("cmPlace");
  t.replaceChildren();
  const head = el("tr");
  for (const [h, r] of [["Collective (group)", 0], ["FSDP + EP 64", 1], ["PP 8", 1], ["PP 16", 1]])
    head.append(el("th", r ? { class: "r" } : {}, h));
  t.append(head);
  for (const row of [
    ["FSDP, dense weights (all-gather bf16, reduce-scatter fp32)", "512 / TP GPUs, InfiniBand", "64 / TP, NVLink", "32 / TP, NVLink"],
    ["FSDP, expert shards", "8 replicas, InfiniBand", "none (EP 64 = stage)", "none (EP 32 = stage)"],
    ["EP dispatch / combine (DistMoE)", "64, NVLink", "64, NVLink", "32, NVLink"],
    ["CP (Ulysses all-to-all for MLA, KDA state)", "CP, NVLink", "CP, NVLink", "CP, NVLink"],
    ["TP (sequence parallel, async)", "TP, NVLink", "TP, NVLink", "TP, NVLink"],
    ["PP hop (hidden state + AttnRes stack)", "--", "InfiniBand", "alternating NVLink / InfiniBand"],
    ["DistMuon gather / scatter of gradients", "InfiniBand", "NVLink", "NVLink"],
  ]) {
    const tr = el("tr");
    row.forEach((c, i) => tr.append(el("td", i ? { class: "r" } : {}, c)));
    t.append(tr);
  }
}

// ---------- section 4: step time per config ----------
function renderStep() {
  const RK = DATA.ranking, b = t => el("b", {}, t);
  if (!RK) return;
  const rows = RK.filter(r => r.ctx === CTX);
  const s1 = v => `${v.toFixed(1)} s`, ms = v => `${(1e3 * v).toFixed(1)} ms`;
  document.getElementById("stSub").replaceChildren(
    `Every config from the memory page that fits in 192 GB at ${CTX} (${rows.length} of them), bf16. For each, `,
    "per layer and microbatch: the compute, each parallelism's communication, the compute it can hide behind, ",
    "and what stays exposed; then the step, with the PP schedule. Two compute numbers: the ", b("roofline"),
    ` (${pctOf(ROOFLINE_EFF.bf16)} of peak for every matmul, KDA from its model) and the `, b("calibrated"),
    " model (section 2's measured efficiencies at the config's shapes -- dense GEMMs at T / cp rows, DistMoE at ",
    "T / (cp x tp) local tokens with its helper kernels, FA4 at 96 / (cp x tp) heads, KDA with its measured CP ",
    "factor -- plus the work outside the roofline measured in 2.5 and FSDP's staging copies). Overlap rules: ",
    b("FSDP"), " gathers the next layer during this layer's forward, and gathers + reduce-scatters during its ",
    "backward (every microbatch without PP or with ZeRO-3; once per step under PP's ZeRO-1 default); ", b("EP"),
    "'s dispatch and combine hide behind the same pass's expert GEMMs (DistMoE fuses them); ", b("TP"),
    " is async, behind its block's GEMMs; ", b("CP"), "'s all-to-alls are ", b("assumed hidden"), " for this study (head-chunked ",
    "Ulysses pipelining, backward all-to-alls behind the weight-gradient GEMMs, or an all-gather of MLA's 576-wide ",
    "latent instead), so they are shown but not added. Eager-mode host idle (2.5) is ",
    "reported apart and not in the step: CUDA graphs or compilation would remove most of it.");
  document.getElementById("stFormula").replaceChildren();
  const F = document.getElementById("stFormula");
  const line = (lbl, eq) => { const d = el("div"); d.append(el("span", { class: "lbl" }, lbl), el("span", { class: "eq" }, eq)); F.append(d); };
  line("Without PP", "t_step = M x sum over layers (t_fwd + t_bwd + exposed comm) + optimizer comm");
  line("With PP (1F1B)", "t_step = (M + P - 1) x (t_stage + 2 x t_hop) + FSDP once per step + optimizer comm");
  line("t_stage, t_hop", "slowest stage's forward + backward for one microbatch; one send between stages");
  line("Exactly", "(t_stage + 2 x t_hop) = max over stages s of (t_s, t_s + 2 x hop_s), the steady-state period");
  const w = el("div", { class: "where" });
  w.append("M: microbatches per pipeline replica (memory page). t_s: stage s's layers (+ LM head on the last) ",
           "with their exposed communication. Hop: one send of the hidden state plus the AttnRes blocks completed so ",
           "far (12 layers per block); each microbatch slot pays it twice, the activation forward and its gradient ",
           "backward, because in 1F1B the receiving stage waits for it. In the table, (M + P - 1) x 2 x t_hop is the ",
           "\"PP hops\" column (every slot, fill and drain included) and (P - 1) x t_stage the bubble (idle stage time).");
  F.append(w);

  // 4.1: per-layer view for the best config of each family
  const fam = r => r.pp === 1 ? "FSDP + EP 64" : `PP ${r.pp}, ZeRO-${r.zero}`;
  const best = {};
  for (const r of rows) if (!best[fam(r)] || r.calibrated.step_s < best[fam(r)].calibrated.step_s) best[fam(r)] = r;
  const picks = Object.values(best).sort((x, y) => x.calibrated.step_s - y.calibrated.step_s);
  document.getElementById("stLayerSub").replaceChildren(
    "One KDA MoE layer, one microbatch, roofline compute, for the fastest config of each family at ", CTX,
    ". Each cell: the collective's time / the compute it hides behind; ", b("red"), " when it does not fit ",
    "(that difference is exposed), with the total it exposes over the whole step in brackets. FSDP's backward ",
    "window holds an all-gather and a reduce-scatter; under PP's ZeRO-1 it runs once per step instead of every ",
    "microbatch, so it is shown in grey. CP is the MLA layer's Ulysses all-to-alls (forward, remat and backward), ",
    "assumed hidden. The last column is the whole step's exposed PP hop time: every microbatch slot, fill and drain ",
    "included, waits for the hop in and its gradient back -- two hops per slot when the stages are equal, less when ",
    "the slowest stage is not the one before the largest hop (PP 16).");
  const t = document.getElementById("stLayer");
  t.replaceChildren();
  const head = el("tr");
  for (const [h, r] of [["Config", 0], ["FSDP forward", 1], ["FSDP backward", 1], ["EP per pass", 1], ["TP per pass", 1],
                        ["CP, MLA layer", 1], ["PP hops, whole step", 1]])
    head.append(el("th", r ? { class: "r" } : {}, h));
  t.append(head);
  const cell = (comm, win, grey, total) => {
    const bad = win != null && comm > win * 1.0001 && comm > 1e-5;
    const txt = comm < 1e-5 ? "-" : win == null ? ms(comm) : `${ms(comm)} / ${ms(win)}`;
    const td = el("td", { class: `r ${bad && !grey ? "up" : ""}`, style: grey ? "color: var(--muted)" : "" }, txt);
    if (bad && !grey && total != null)
      td.append(el("span", { class: "change", style: "display: block" }, `(${s1(total)} exposed per step)`));
    return td;
  };
  for (const r of picks) {
    const L = r.roofline.layer, R0 = r.roofline, tr = el("tr");
    const grey = !L.fsdp_per_mb;
    let hop;
    if (r.pp > 1) {
      hop = el("td", { class: "r up" }, `${s1(R0.exposed.pp)} per step`);
      const slots = r.microbatches + r.pp - 1, perSlot = R0.exposed.pp / slots;
      hop.append(el("span", { class: "change", style: "display: block" },
                    `${ms(perSlot)} per slot x ${exact(slots)} slots (largest hop ${ms(R0.hop_s)})`));
    } else hop = el("td", { class: "r" }, "-");
    tr.append(el("td", {}, cfgName(r)), cell(L.fsdp_ag, L.fwd, grey, R0.fsdp_exposed_fwd_s),
              cell(L.fsdp_ag + L.fsdp_rs, L.bwd, grey, R0.fsdp_exposed_bwd_s),
              cell(L.ep, L.ex_fwd), cell(L.tp, L.tp_gemm), cell(L.cp_mla, null), hop);
    t.append(tr);
  }

  // 4.2: ranking (calibrated), stacked bars and table
  document.getElementById("stRankSub").replaceChildren(
    `Ranked by the calibrated step at ${CTX}. Bars: `, b("compute"), " (M x the slowest stage's compute); ",
    b("PP bubble"), ", stages idle while the pipeline fills and drains, (P - 1) x the slowest stage; ",
    b("PP hops"), ", waiting for the inter-stage transfer (hidden state + AttnRes stack), twice per microbatch slot ",
    "including fill and drain, (M + P - 1) x 2 x hop; ", b("other exposed communication"), ", FSDP, EP, TP or DistMuon ",
    "time that does not fit its overlap window; ", b("FSDP staging copies"), ". The black tick is the roofline step. ",
    "The table adds memory and what each config is bound by (any exposed term over 3% of its step). Eager idle is ",
    "the extra a non-graphed eager run would add.");
  const segs = [["Compute", "#2a78d6", c => c.compute_s], ["PP bubble", "#eda100", c => c.bubble_s],
                ["PP hops (inter-stage transfer)", "#e87ba4", c => c.exposed.pp],
                ["Other exposed communication", "#eb6834", c => Object.entries(c.exposed).filter(([k]) => k !== "pp").reduce((a, [, x]) => a + x, 0)],
                ["FSDP staging copies", "#4a3aa7", c => c.fsdp_copies_s || 0]];
  const top = rows.slice(0, 15), host = document.getElementById("stBars");
  host.replaceChildren();
  const legend = el("div", { class: "legend" });
  for (const [l, c] of segs) {
    const it = el("span", { style: "display: inline-flex; align-items: center; gap: 6px" });
    it.append(el("span", { style: `width: 10px; height: 10px; border-radius: 2px; background: ${c}; display: inline-block` }), l);
    legend.append(it);
  }
  host.append(legend);
  const W = Math.max(640, host.clientWidth - 32), lab = 330, rowH = 24, H = top.length * rowH + 24;
  const maxS = Math.max(...top.map(r => r.calibrated.step_s), ...top.map(r => r.roofline.step_s)) * 1.08;
  const xs = v => lab + (W - lab - 60) * v / maxS;
  const svg = sel("svg", { width: W, height: H, role: "img", "aria-label": "calibrated step time per config" });
  for (let g = 0; g <= maxS; g += 5) {
    svg.append(sel("line", { x1: xs(g), x2: xs(g), y1: 0, y2: H - 18, stroke: "var(--grid)" }));
    svg.append(sel("text", { x: xs(g), y: H - 4, class: "tick", "text-anchor": "middle" }, `${g} s`));
  }
  top.forEach((r, i) => {
    const y = i * rowH + 4, c = r.calibrated;
    svg.append(sel("text", { x: lab - 8, y: y + 13, class: "rowlabel", "text-anchor": "end" }, `${i + 1}. ${cfgName(r)}`));
    let x = 0;
    const g = sel("g", { class: "bar", tabindex: 0 });
    g.append(sel("rect", { x: lab, y, width: W - lab, height: rowH - 6, class: "hit" }));
    for (const [, color, f] of segs) {
      const v = f(c);
      if (v <= 0) continue;
      g.append(sel("rect", { x: xs(x), y, width: Math.max(0.5, xs(x + v) - xs(x) - 1), height: rowH - 8, fill: color }));
      x += v;
    }
    g.append(sel("line", { x1: xs(r.roofline.step_s), x2: xs(r.roofline.step_s), y1: y - 1, y2: y + rowH - 7,
                           stroke: "var(--text-primary)", "stroke-width": 2 }));
    g.append(sel("text", { x: xs(x) + 6, y: y + 12, class: "vallabel" }, s1(c.step_s)));
    attachTip(g, () => [{ text: cfgName(r), cls: "v" },
      ...segs.map(([l, color, f]) => ({ text: `${l}: ${s1(f(c))}`, color })),
      { text: `roofline step ${s1(r.roofline.step_s)} (black tick)`, cls: "k" },
      { text: `eager idle +${s1(c.idle_eager_s)}`, cls: "k" }]);
    svg.append(g);
  });
  host.append(svg);

  const tbl = document.getElementById("stRank");
  tbl.replaceChildren();
  const hd = el("tr");
  for (const [h, rgt] of [["#", 0], ["Config", 0], ["M", 1], ["Memory", 1], ["Roofline step", 1], ["Calibrated step", 1],
                          ["Compute", 1], ["PP bubble", 1], ["Exposed communication", 1], ["FSDP copies", 1],
                          ["Bound by", 0], ["Eager idle", 1]])
    hd.append(el("th", rgt ? { class: "r" } : {}, h));
  tbl.append(hd);
  const NAMES = { fsdp: "FSDP", ep: "EP", tp: "TP", cp: "CP", pp: "PP hops", optimizer: "DistMuon" };
  rows.forEach((r, i) => {
    const c = r.calibrated, tr = el("tr");
    const ex = Object.entries(c.exposed).filter(([, v]) => v >= 0.05).map(([k, v]) => `${NAMES[k]} ${v.toFixed(1)}`).join(", ");
    const stepTd = el("td", { class: "r", style: "white-space: nowrap" });
    stepTd.append(b(s1(c.step_s)));
    tr.append(el("td", {}, String(i + 1)), el("td", {}, cfgName(r)), el("td", { class: "r" }, exact(r.microbatches)),
              el("td", { class: "r", style: "white-space: nowrap" }, `${(r.bytes / 1e9).toFixed(0)} GB`),
              el("td", { class: "r", style: "white-space: nowrap" }, s1(r.roofline.step_s)), stepTd,
              el("td", { class: "r", style: "white-space: nowrap" }, s1(c.compute_s)),
              el("td", { class: "r", style: "white-space: nowrap" }, c.bubble_s ? s1(c.bubble_s) : "-"),
              el("td", { class: "r" }, ex || "-"),
              el("td", { class: "r", style: "white-space: nowrap" }, c.fsdp_copies_s >= 0.05 ? s1(c.fsdp_copies_s) : "-"),
              el("td", {}, c.bound_by.map(k => NAMES[k]).join(", ") || "compute"),
              el("td", { class: "r", style: "white-space: nowrap; color: var(--muted)" }, `+${s1(c.idle_eager_s)}`));
    tbl.append(tr);
  });

  // takeaways
  const ul = el("ul", { style: "margin: 6px 0 0 18px; padding: 0" });
  const li = (...parts) => { const e = el("li", { style: "margin-bottom: 4px" }); e.append(...parts); ul.append(e); };
  const first = rows[0], noPp = rows.filter(r => r.pp === 1), pp = rows.filter(r => r.pp > 1);
  const bestNoPp = noPp[0];
  li(b(`Fastest at ${CTX}: ${cfgName(first)}, ${s1(first.calibrated.step_s)} calibrated `),
     `(${s1(first.roofline.step_s)} roofline, ${(first.bytes / 1e9).toFixed(0)} GB).`);
  if (bestNoPp) {
    const fs = noPp.map(r => r.roofline.exposed.fsdp);
    li(b("Without PP, FSDP is the problem. "), `Its rings span all 512 GPUs, so they run at the InfiniBand rate, `,
       "and it gathers and reduce-scatters every layer every microbatch -- including 8 replicas of each expert ",
       `shard. Exposed FSDP (roofline) ranges ${Math.min(...fs).toFixed(1)}-${Math.max(...fs).toFixed(1)} s per step; `,
       "it is smallest where there are few microbatches (TP or long contexts), which is why ",
       `${cfgName(bestNoPp)} is the best no-PP config (${s1(bestNoPp.calibrated.step_s)}).`);
  }
  if (pp.length) {
    const b0 = pp[0];
    li(b("With PP, FSDP and EP stay on NVLink and hide; the cost moves to the schedule. "),
       `For ${cfgName(b0)}: bubble ${s1(b0.calibrated.bubble_s)}, PP hops ${s1(b0.calibrated.exposed.pp)} `,
       `(a hop carries the hidden state plus up to 8 AttnRes blocks: ${ms(b0.roofline.hop_s)} each way over `,
       "InfiniBand, waited on every microbatch). More microbatches (CP or TP) shrink the bubble but make each ",
       "call smaller, which the calibrated model charges for (DistMoE and dense GEMM efficiency, KDA's CP factor).");
  }
  li(b("TP and EP are never exposed here: "), "async TP's all-gathers and reduce-scatters are a few ms against ",
     "tens of ms of block GEMMs, and DistMoE's dispatch and combine fit within the expert GEMMs at every shape. ",
     b("CP"), "'s Ulysses all-to-alls (0.1-1.4 s per step if exposed) are assumed hidden; what CP still pays is ",
     "KDA's CP compute overhead (1.3-2x) and smaller calls.");
  li(b("Eager idle dwarfs the differences between the top configs: "), "configs with many microbatches launch ",
     "proportionally more kernels; without CUDA graphs or compilation they would pay the idle column on top.");
  li(b("Caveats. "), "Bandwidths are the datasheet, the assumed InfiniBand rate and 4-GPU NVLink measurements; ",
     "nothing multi-node was measured. FSDP rings are flat (HSDP or hierarchical collectives would cut the ",
     "no-PP cost); TP narrowing dense GEMMs is not measured; KDA at CP 8 uses CP 4's factor; memory comes from ",
     "the memory page, whose 32K estimate is below the 32K fake-PG run's peak.");
  document.getElementById("stWhy").replaceChildren(b("What the ranking says:"), ul);
}


// 4.3: the PP step, stage by stage, for the fastest PP config at this context
function renderPpDeepDive() {
  const RK = DATA.ranking, b = t => el("b", {}, t);
  if (!RK) return;
  const r = RK.filter(x => x.ctx === CTX && x.pp > 1)[0];
  if (!r) return;
  const R0 = r.roofline, C0 = r.calibrated, st = R0.stages, P = r.pp, M = r.microbatches;
  const ms = v => `${(1e3 * v).toFixed(1)} ms`, s1 = v => `${v.toFixed(1)} s`;
  const tl = r.seq_len / (r.cp * r.tp);
  const worst = st.reduce((w, x, i) => (x.t_s + 2 * x.hop_s > st[w].t_s + 2 * st[w].hop_s ? i : w), 0);
  const tmax = Math.max(...st.map(x => x.t_s));
  const ol = el("ol", { style: "margin: 6px 0 0 18px; padding: 0" });
  const step = (...parts) => { const e = el("li", { style: "margin-bottom: 4px" }); e.append(...parts); ol.append(e); };
  step(b("Split the 93 layers into stages: "), `ceil(93 / ${P}) = ${st[0].layers} consecutive layers each, the `,
       "remainder on the last (the memory page's split).");
  step(b("Time each stage for one microbatch, t_s: "), "the first stage starts with the token-embedding lookup, ",
       "a memory-bound gather (forward) and scatter-add into the fp32 gradient (backward) with no FLOPs, timed at ",
       "8 TB/s of HBM; its FSDP gather and reduce-scatter (1.17B parameters) hide behind layer 0. Then ", `one ${CTX} sequence split over CP ${r.cp} x TP ${r.tp} `,
       `(${exact(tl)} local tokens). Per layer: dense GEMMs (8 FLOPs / param / token: forward, remat forward, dgrad, `,
       "wgrad), routed experts (8 x local tokens x k x params per expert), MLA (4.5x forward FLOPs) or KDA (measured ",
       "model), plus any communication that does not fit its overlap window. The last stage adds the LM head.");
  step(b("Time the hop after each stage, hop_s: "), "the hidden state plus the AttnRes blocks finished so far, ",
       `(1 + blocks) x ${exact(tl)} tokens x 7168 x 2 bytes, over InfiniBand between domains (NVLink within one).`);
  step(b("Period and step: "), "in 1F1B each microbatch slot on stage s costs t_s plus waiting for the hop twice ",
       "(its output forward, the gradient back), so period = max over s of (t_s + 2 x hop_s); the pipeline runs M ",
       "slots of steady state plus P - 1 of fill and drain: t_step = (M + P - 1) x period.");
  const sub = document.getElementById("ppdSub");
  sub.replaceChildren(b(`Worked example: ${cfgName(r)} at ${CTX}`),
    ` (the fastest PP config here), roofline compute, M = ${exact(M)} microbatches.`, ol);
  const t = document.getElementById("ppdStages");
  t.replaceChildren();
  const head = el("tr");
  for (const [h, rt] of [["Stage", 0], ["Layers", 0], ["Contents", 0], ["t_s", 1], ["Hop after it", 1], ["hop_s", 1],
                         ["t_s + 2 x hop_s", 1], ["Calibrated t_s", 1]])
    head.append(el("th", rt ? { class: "r" } : {}, h));
  t.append(head);
  st.forEach((x, i) => {
    const tr = el("tr"), per = el("td", { class: "r" });
    const v = ms(x.t_s + 2 * x.hop_s);
    per.append(i === worst ? b(`${v} (sets the period)`) : v);
    tr.append(el("td", {}, String(i)), el("td", {}, `${x.first}-${x.last} (${x.layers})`),
              el("td", {}, `${x.embedding ? `Embedding (${(1e3 * x.embedding_s).toFixed(2)} ms) + ` : ""}${x.layers - x.mla} KDA + ${x.mla} MLA${x.lm_head ? " + LM head" : ""}`),
              el("td", { class: "r" }, ms(x.t_s)),
              el("td", { class: "r" }, x.hop_bytes ? `${(x.hop_bytes / 1e6).toFixed(0)} MB` : "-"),
              el("td", { class: "r" }, x.hop_s ? ms(x.hop_s) : "-"), per,
              el("td", { class: "r", style: "color: var(--text-secondary)" }, ms(C0.stages[i].t_s)));
    t.append(tr);
  });
  const sum = document.getElementById("ppdSum");
  sum.replaceChildren();
  const h2 = el("tr");
  for (const [h, rt] of [["Step, as in the ranking table", 0], ["Formula", 0], ["Roofline", 1], ["Calibrated", 1]])
    h2.append(el("th", rt ? { class: "r" } : {}, h));
  sum.append(h2);
  const cmax = Math.max(...C0.stages.map(x => x.t_s));
  for (const [l, f, rv, cv, bold] of [
    ["Compute", "M x slowest stage", R0.compute_s, C0.compute_s],
    ["PP hops", "(M + P - 1) x (period - slowest stage): 2 hops per slot", R0.exposed.pp, C0.exposed.pp],
    ["PP bubble", "(P - 1) x slowest stage: idle fill and drain", R0.bubble_s, C0.bubble_s],
    ["Other (FSDP once per step, DistMuon, FSDP copies)", "", R0.step_s - R0.compute_s - R0.exposed.pp - R0.bubble_s,
     C0.step_s - C0.compute_s - C0.exposed.pp - C0.bubble_s],
    ["Step", `(M + P - 1) x period = (${exact(M)} + ${P - 1}) x ${ms(R0.period_s)} + other`, R0.step_s, C0.step_s, true],
  ]) {
    const tr = el("tr");
    const cell = v => { const td = el("td", { class: "r" }); td.append(bold ? b(s1(v)) : s1(v)); return td; };
    tr.append(el("td", {}, l), el("td", {}, f), cell(rv), cell(cv));
    sum.append(tr);
  }
  const last = st[P - 1], idle = 1 - last.t_s / R0.period_s;
  const ul = el("ul", { style: "margin: 6px 0 0 18px; padding: 0" });
  const li = (...parts) => { const e = el("li", { style: "margin-bottom: 4px" }); e.append(...parts); ul.append(e); };
  li(b(`The last stage is underloaded: ${ms(last.t_s)} against a ${ms(R0.period_s)} period, idle ${(100 * idle).toFixed(0)}% of every slot. `),
     `It has ${last.layers} layers to the others' ${st[0].layers}, and the LM head does not make up the difference. Moving `,
     "layers onto it would shorten the other stages; that needs the memory page's split (and memory) re-checked.");
  li(b(`The hops grow with depth: ${ms(st[0].hop_s)} after stage 0 to ${ms(st[P - 2].hop_s)} into the last stage. `),
     "Each hop carries the AttnRes blocks finished so far, so the pair before the last stage sets the period for the ",
     "whole pipeline: PP pays for the AttnRes stack more than for the hidden state. An interleaved schedule with ",
     "TorchTitan's AttnRes cache caps a hop at P - 1 new blocks, and cuts the bubble by the number of virtual stages.");
  li(b("The calibrated step follows the same procedure; "), "only t_s changes, to measured kernel efficiencies plus ",
     `the work outside the roofline (slowest stage ${ms(cmax)} instead of ${ms(tmax)}).`);
  document.getElementById("ppdWhy").replaceChildren(b("What this shows:"), ul);
}

// 4.4: CP vs TP at the same degree, memory ignored
function renderTpVsCp() {
  const D = DATA.tp_vs_cp, b = t => el("b", {}, t);
  if (!D) return;
  const T = DATA.model.contexts[CTX].seq_len, rows = D.filter(r => r.seq_len === T);
  const s1 = v => `${v.toFixed(1)} s`, s2 = v => `${v.toFixed(2)} s`;
  const name = r => r.tp === 1 ? `CP ${r.cp}` : r.cp === 1 ? `TP ${r.tp}` : `CP ${r.cp} x TP ${r.tp}`;
  document.getElementById("tcSub").replaceChildren(
    "Splitting each microbatch over more GPUs, by sequence (CP) or by heads and weights (TP with sequence ",
    `parallelism), at the same degree: PP 8 with ZeRO-1 at ${CTX}, calibrated, `, b("memory ignored"),
    ", CP communication hidden as in 4.2. Compute is per step per GPU, by component (all layers x microbatches / P). ",
    "At equal degree the routed experts see the same local tokens, MLA the same heads over the full sequence ",
    "(Ulysses CP and TP give the same shape) and the bubble is the same, so the difference is KDA, with a little ",
    "from dense GEMM rows (T / cp under CP).");
  const t = document.getElementById("tcTable");
  t.replaceChildren();
  const head = el("tr");
  for (const [h, rt] of [["Split", 0], ["Degree", 1], ["M", 1], ["Step", 1], ["KDA", 1], ["Dense GEMMs", 1],
                         ["Routed experts", 1], ["MLA", 1], ["PP bubble", 1]])
    head.append(el("th", rt ? { class: "r" } : {}, h));
  t.append(head);
  for (const deg of [2, 4, 8]) {
    const g = rows.filter(r => r.degree === deg), best = Math.min(...g.map(r => r.step_s));
    const kBest = Math.min(...g.map(r => r.parts.kda));
    for (const r of g) {
      const tr = el("tr"), step = el("td", { class: "r" }), kda = el("td", { class: "r" });
      step.append(r.step_s === best ? b(s1(r.step_s)) : s1(r.step_s));
      kda.append(r.parts.kda === kBest ? b(s2(r.parts.kda)) : s2(r.parts.kda));
      tr.append(el("td", {}, name(r)), el("td", { class: "r" }, String(deg)), el("td", { class: "r" }, exact(r.microbatches)),
                step, kda, el("td", { class: "r" }, s2(r.parts.dense)), el("td", { class: "r" }, s2(r.parts.experts)),
                el("td", { class: "r" }, s2(r.parts.mla)), el("td", { class: "r" }, s1(r.bubble_s)));
      if (r === g[g.length - 1] && deg !== 8) tr.style.borderBottom = "2px solid var(--grid)";
      t.append(tr);
    }
  }
  const K = { 8192: [1.7, 1.7, 1.7], 32768: [1.8, 3.0, 5.0] }[T >= 32768 ? 32768 : 8192];
  const ul = el("ul", { style: "margin: 6px 0 0 18px; padding: 0" });
  const li = (...parts) => { const e = el("li", { style: "margin-bottom: 4px" }); e.append(...parts); ul.append(e); };
  li(b("Degree 2: TP is slightly faster. "), "TP splits KDA's heads, so each GPU runs the whole recurrence on half ",
     "the heads with no dependency between ranks; CP splits the sequence, so each GPU needs the previous chunk's ",
     "state and pays a summary pass (measured 1.3-1.6x the same tokens without CP).");
  li(b("Degree 4 and up: CP is faster. "), `KDA's TP speedup flattens as its heads shrink (measured ${K[0]}x / ${K[1]}x / `,
     `${K[2]}x at TP 2 / 4 / 8 at ${T >= 32768 ? "32K" : "8K"}), while CP's summary-pass overhead stays roughly fixed `,
     "(1.3-1.7x), so CP keeps splitting the work by cp.");
  li(b("Mixing pays both overheads. "), "CP 2 x TP 2 and CP 2 x TP 4 take TP's lower KDA efficiency on fewer heads and ",
     "CP's summary pass on top; they match pure CP at best and lose clearly at degree 8.");
  li(b("TP's advantage is memory, not time. "), "Under ZeRO-1 the dense weights stay gathered all step and TP shards ",
     "them (127 GB for TP 2 against 150 GB for CP 2 at 8K), which is why TP 2 ranks first in 4.2 at 8K.");
  document.getElementById("tcWhy").replaceChildren(b("What decides it:"), ul);
  const cav = el("ul", { style: "margin: 6px 0 0 18px; padding: 0" });
  const c = (...parts) => { const e = el("li", { style: "margin-bottom: 4px" }); e.append(...parts); cav.append(e); };
  c(b("TP's narrower GEMMs are not charged. "), "TP divides N of every attention and shared-expert GEMM by tp ",
    "(N = 3,072 for the projections at TP 4); the model runs them at full-width efficiency. Not measured yet; it ",
    "would favour CP, more so at higher tp.");
  c(b("TP has far more communication to hide. "), "An all-gather and a reduce-scatter of the full [T / cp, 7168] ",
    "activation around every attention and shared-expert block, every layer and pass (about 1.4 GB per layer at ",
    "8K), against CP's all-to-alls in the 24 MLA layers only (about 107x less again with an MLA latent all-gather). ",
    "Both are assumed hidden; TP's is the harder assumption, and async TP needs compilation to overlap at all.");
  c(b("KDA at CP 8 is not measured: "), "it takes CP 4's overhead factor. CP 8 could be worse.");
  c(b("High CP at 8K shrinks dense GEMM rows: "), "1K rows at CP 8 run at 58-64% of peak (dense rises to 9.1 s); ",
    "TP keeps T / cp rows. This is in the model.");
  c(b("To settle it: "), "measure KDA under CP 8, dense GEMMs at N / tp, and async-TP overlap on 2-4 GPUs; all fit ",
    "on one host.");
  const cw = el("div", { style: "margin-top: 10px" });
  cw.append(b("Caveats:"), cav);
  document.getElementById("tcWhy").append(cw);
}

function renderAll() {
  renderRooflineInputs();
  renderComms();
  renderStep();
  renderPpDeepDive();
  renderTpVsCp();
}

document.getElementById("subtitle").replaceChildren(
  "Step time for training Kimi K3 on 512 GB200 GPUs, for the configurations that fit in memory on ",
  el("a", { href: "kimi_k3_params.html" }, "the memory page"), ".");
const ctxToggle = document.getElementById("ctxToggle");
for (const key of Object.keys(DATA.model.contexts)) {
  const id = `ctx-${key}`;
  const input = el("input", { type: "radio", name: "ctx", id, value: key });
  if (key === CTX) input.checked = true;
  input.addEventListener("change", () => { CTX = key; renderAll(); });
  ctxToggle.append(input, el("label", { for: id }, key));
}
renderAll();
window.addEventListener("resize", renderAll);
document.getElementById("foot").textContent =
  "Generated by scripts/memory_study/kimi_k3_roofline.py from kimi_k3_data.json and bench/.";
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
