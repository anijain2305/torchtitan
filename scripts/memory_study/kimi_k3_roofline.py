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
    shapes = bench_dir / "attention_shapes_GB300.json"
    return {
        "kernels": json.loads(kernels.read_text()),
        "nvlink": nvlink,
        "attention_shapes": json.loads(shapes.read_text()) if shapes.exists() else None,
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
        "bench": bench_inputs(args.bench_dir),
    }
    configs = HERE / "kimi_k3_configs.json"
    if data["bench"] and configs.exists():
        data["links"] = step_model.link_table(data["bench"])
        data["ranking"] = step_model.rank_configs(
            data["model"], data["bench"], json.loads(configs.read_text())["configs"]
        )
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
        data["tp_vs_cp"] = step_model.tp_vs_cp(data["model"], data["bench"])
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

  <h2>1. Roofline inputs (provisional)</h2>
  <p class="sub" id="riSub"></p>
  <h3 class="subhead">1.1 Compute: dense GEMMs</h3>
  <p class="sub" id="riLinSub"></p>
  <div class="panel"><table id="riLinears"></table></div>
  <h3 class="subhead">1.2 Compute: routed-expert GEMMs and attention</h3>
  <p class="sub" id="riExSub"></p>
  <div class="panel"><table id="riExperts"></table></div>
  <div class="panel" style="margin-top: 16px"><table id="riAttn"></table></div>
  <div class="panel" style="margin-top: 16px"><table id="riKda"></table></div>
  <h3 class="subhead">1.3 Communication: NVLink</h3>
  <div class="panel"><table id="riNvlink"></table></div>
  <div class="callout note" id="riNote"></div>

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

// Roofline efficiency rule: fractions of DATASHEET peak, from the measured
// tables rounded down to 5%. Each tier holds from its token count up to the
// next; a count between tiers takes the lower tier, so small GEMMs are
// penalized and the exact measurements stop mattering.
const ROOFLINE_RULE = {
  // dense linears, by local seq length (GEMM M)
  dense: { tiers: [1024, 2048], bf16: [0.60, 0.65], mxfp8: [0.50, 0.60] },
  // routed-expert GEMMs, by tokens per expert per microbatch
  experts: { tiers: [512, 1024, 2048, 4096], bf16: [0.35, 0.45, 0.55, 0.65], mxfp8: [0.20, 0.30, 0.45, 0.60] },
  // MLA attention (cuDNN / varlen), of bf16 peak, by sequence length (keys): forward
  // + backward, and the forward full remat replays. TP and Ulysses CP only cut
  // heads, which barely changes efficiency; KV all-gather CP needs Flex
  // (~5-10% of peak here), so the roofline models CP as Ulysses + varlen.
  mla: { tiers: [4096, 8192, 16384], fwd_bwd: [0.40, 0.45, 0.50], fwd: [0.55, 0.65, 0.70] },
  // KDA, us per token per layer with all 96 heads: forward + backward, and
  // the remat forward. TP t speeds it up by the measured factor rounded down,
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
function riTable(table, corner, columns, rows, fmtCell) {
  table.replaceChildren();
  const head = el("tr");
  head.append(el("th", {}, corner));
  for (const c of columns) head.append(el("th", { class: "r" }, c));
  table.append(head);
  for (const [label, values, shadeLow, peak] of rows) {
    const tr = el("tr");
    tr.append(el("td", {}, label));
    const best = Math.max(...values.filter(v => v != null));
    for (const v of values) {
      const low = shadeLow && v != null && v < 0.6 * best;
      const td = el("td", { class: `r ${low ? "up" : ""}` }, v == null ? "--" : fmtCell(v));
      if (v != null && peak)
        td.append(el("span", { class: "change", style: "color: var(--text-secondary)" },
                     `${(100 * v / peak).toFixed(0)}% of peak`));
      tr.append(td);
    }
    table.append(tr);
  }
}

function renderRooflineInputs() {
  const R = DATA.bench, sub = document.getElementById("riSub");
  if (!R) { sub.textContent = "No bench results were found when this page was generated."; return; }
  const K = R.kernels, b = t => el("b", {}, t);
  sub.replaceChildren(
    `Measured on one ${K.gpu.replace("NVIDIA ", "")} of a shared devserver, with GEMMs timed as CUDA-graph `,
    "replays (kernel time, as compiled training sees it). The kernel run had the GPU to itself; the ",
    "NVLink sweep may have shared it, so ", b("NVLink will be re-measured"), ". The roofline turns ",
    "FLOPs by kernel class, at the token counts each configuration produces, into compute time, and ",
    "bytes per link into communication time. Values are TFLOPS for forward + backward unless noted; ",
    "under each value is its % of the per-GPU dense peak in NVIDIA's Blackwell datasheet (GB200: ",
    `${DATASHEET.bf16 / 1000} PFLOPS bf16, ${DATASHEET.fp8 / 1000} PFLOPS FP8/MXFP8, ${DATASHEET.nvlink} GB/s `,
    "NVLink per direction; the measurements are on GB300, so treat it as approximate). Shaded cells are ",
    "under 60% of their row's best. Expert GEMMs are timed as dense per-expert GEMMs ",
    "(grouped GEMMs do not run on SM103 yet), and MXFP8 is GEMM-only, without quantization kernels.");

  const tok = n => n >= 1024 ? `${n / 1024}K` : String(n);
  const tf = v => v.toFixed(0);
  // 1.1 and 1.2: the roofline rule, with the measured values beside it
  const pct = v => `${(100 * v).toFixed(0)}%`;
  const ruleTable = (table, corner, rule, measured, peakOf) => {
    table.replaceChildren();
    const head = el("tr");
    head.append(el("th", {}, corner));
    rule.tiers.forEach((t, i) => head.append(el("th", { class: "r" },
      i === rule.tiers.length - 1 ? `>= ${tok(t)}` : i === 0 && rule.tiers.length > 2 ? `<= ${tok(t)}` : tok(t))));
    table.append(head);
    for (const [dt, label] of [["bf16", "bf16"], ["mxfp8", "MXFP8"]]) {
      const tr = el("tr");
      tr.append(el("td", {}, label));
      rule[dt].forEach((eff, i) => {
        const td = el("td", { class: "r" });
        td.append(b(pct(eff)), el("span", { class: "change", style: "color: var(--text-secondary)" },
          `${tf(eff * peakOf(dt))} TFLOPS; measured ${measured(dt, i)}`));
        tr.append(td);
      });
      table.append(tr);
    }
  };
  const peakOf = dt => dt === "bf16" ? DATASHEET.bf16 : DATASHEET.fp8;
  const scopes = ["kda_layer", "mla_layer", "lm_head"];
  const span = vals => {
    const lo = Math.min(...vals), hi = Math.max(...vals);
    return lo === hi ? pct(lo) : `${pct(lo)}-${pct(hi)}`;
  };
  const R_D = ROOFLINE_RULE.dense, R_E = ROOFLINE_RULE.experts;
  // measured efficiencies falling in tier i (token count from tier i up to tier i + 1)
  const inTier = (rule, i) => m => m >= rule.tiers[i] && (i === rule.tiers.length - 1 || m < rule.tiers[i + 1]);
  document.getElementById("riLinSub").replaceChildren(
    "The roofline uses a tiered rule, not the exact measurements: each tier is the measured efficiency ",
    "(forward, input-gradient and weight-gradient GEMMs of the KDA-layer, MLA-layer and LM-head linears) ",
    "rounded down to 5%, and a token count between tiers takes the lower tier. That keeps precise numbers ",
    "from mattering while still ", b("penalizing configs whose local seq length shrinks"), ". 2K already runs ",
    "at full efficiency; only 1K needs a penalty. ", b("MXFP8 does not need M > 2K"), ": an earlier drop at ",
    "1K-2K came from about 75 us of CPU dispatch per eager torchao MXTensor call, which compiled training ",
    "does not pay (GEMMs are now timed as CUDA-graph replays). It still dips at 1K, where a big projection ",
    "finishes in about 60 us and fixed per-kernel costs -- filling the pipeline, the last partial wave of ",
    "tiles -- weigh twice as much as in bf16.");
  ruleTable(document.getElementById("riLinears"), "Dense linears, % of peak / local seq length", R_D,
    (dt, i) => span(scopes.flatMap(sc => Object.entries(K.linears[dt][sc])
      .filter(([m]) => inTier(R_D, i)(Number(m))).map(([, v]) => v / peakOf(dt)))), peakOf);

  // one table: efficiency by local seq length, per EP degree and dtype, with
  // tokens per expert = local x EP x top-k / E in brackets
  const S = DATA.model.symbols, perExpert = (local, ep) => local * ep * S.k / S.E;
  const locals = [1024, 2048, 4096, 8192];  // full efficiency by 8K at EP 32
  const measuredAt = (dt, tier) => K.experts[dt][tier] / peakOf(dt);
  const et = document.getElementById("riExperts");
  et.replaceChildren();
  const eh = el("tr");
  eh.append(el("th", {}, "Expert GEMMs (w13 + w2), % of peak / local seq length"));
  for (const l of locals) eh.append(el("th", { class: "r" }, l === locals[locals.length - 1] ? `>= ${tok(l)}` : tok(l)));
  et.append(eh);
  for (const [ep, used] of [[64, "FSDP + EP 64, PP 8"], [32, "PP 16"]]) {
    for (const [dt, label] of [["bf16", "bf16"], ["mxfp8", "MXFP8"]]) {
      const tr = el("tr");
      tr.append(el("td", {}, `EP ${ep} (${used}), ${label}`));
      for (const l of locals) {
        const n = perExpert(l, ep), r = ruleEff(R_E, dt, n);
        const td = el("td", { class: `r ${r.eff < R_E[dt][R_E[dt].length - 1] ? "up" : ""}` });
        td.append(b(pct(r.eff)), el("span", { class: "change", style: "color: var(--text-secondary)" },
          `(${exact(Math.round(n))} tokens/expert; measured ${pct(measuredAt(dt, r.tier))})`));
        tr.append(td);
      }
      et.append(tr);
    }
  }

  document.getElementById("riExSub").replaceChildren(
    "Expert GEMMs are smaller than the dense projections, so they need more tokens. Each expert receives ",
    "tokens from every GPU in its EP group, so under balanced routing tokens per expert = local seq length ",
    `x EP x top-k / experts = local x EP x ${DATA.model.symbols.k} / ${DATA.model.symbols.E}: each token goes to `,
    `${DATA.model.symbols.k} experts, spread evenly over all ${DATA.model.symbols.E}. That is about 1.14x the `,
    "local seq length at EP 64 and 0.57x at EP 32. The rule tiers tokens per expert like the dense rule ",
    `(${R_E.tiers.map(tok).join(" / ")}: bf16 ${R_E.bf16.map(pct).join(" / ")}, MXFP8 `,
    `${R_E.mxfp8.map(pct).join(" / ")}, a count between tiers taking the lower one); the table applies it `,
    "by local seq length, with tokens per expert and the measured efficiency at that tier in brackets, and ",
    "shades cells below the top tier -- the axis on which high CP, high TP and smaller EP pay. Attention is timed ",
    "eagerly (each call takes milliseconds, so CPU dispatch does not matter). ",
    b("MLA"), " uses the same tiered rule, by sequence length T -- the keys a rank's attention covers, ",
    "which stays the full context length under TP and Ulysses CP (only the heads shrink): ",
    `${ROOFLINE_RULE.mla.tiers.map(tok).join(" / ")} -> forward + backward ${ROOFLINE_RULE.mla.fwd_bwd.map(pct).join(" / ")}`,
    `, remat forward ${ROOFLINE_RULE.mla.fwd.map(pct).join(" / ")} of bf16 peak. TP and Ulysses CP (TorchTitan's `,
    "CP mode that works with varlen) leave each rank the full sequence on fewer heads, which barely costs ",
    "efficiency, so the roofline models CP as Ulysses + varlen. KV all-gather CP, TorchTitan's other mode, ",
    "requires FlexAttention, which runs MLA's 192 / 128 head dims at only 5-10% of peak here, CP or not. ",
    b("KDA"), " is not a big matmul, so it gets no TFLOPS or % of peak: it works in 64-token chunks, each a ",
    "handful of small matmuls (64 x 64 and 64 x 128 blocks per head), a triangular solve for the delta-rule ",
    "correction, a 128 x 128 state update and gating exponentials -- too small to keep the tensor cores busy, ",
    "partly memory- and latency-bound, and chained chunk to chunk by the recurrent state. A rough count puts ",
    "its forward near 7-10% of bf16 peak, but the FLOP count depends on the chunk algorithm, so the rule uses ",
    "measured time per token per layer for the rank's heads: a rank runs T / CP tokens on 96 / TP ",
    `heads (TP gathers the sequence before attention), at ${ROOFLINE_RULE.kda.fwd_bwd} us forward + backward `,
    `plus ${ROOFLINE_RULE.kda.fwd} us for the remat forward per token with all heads, sped up under TP by the `,
    "measured factor rounded down (1.8x / 3x / 5x at 32K for TP 2 / 4 / 8, but only 1.7x at any TP at 8K, where ",
    "the kernel no longer fills the GPU). KDA under ",
    "CP adds state passing between ranks (communication). The shape ",
    "measurements shared the GPU with another process and will be re-measured.");

  // MLA: rule and measured, % of bf16 peak (cells show TFLOPS and % of peak)
  // attention_shapes JSON: {attention_shapes: {contended_by, data}}
  const A = R.attention_shapes?.attention_shapes?.data, seq = [4096, 8192, 16384, 32768];
  const ruleAt = (key, t) => ruleEff({ tiers: ROOFLINE_RULE.mla.tiers, x: ROOFLINE_RULE.mla[key] }, "x", t).eff;
  const at = (obj, t) => obj && obj[t] ? obj[t] : null;
  const mlaRows = [
    ["Roofline uses, forward + backward", seq.map(t => ruleAt("fwd_bwd", t) * DATASHEET.bf16), false, DATASHEET.bf16],
    ["Roofline uses, remat forward", seq.map(t => ruleAt("fwd", t) * DATASHEET.bf16), false, DATASHEET.bf16],
    ["cuDNN, 96 heads, forward + backward", seq.map(t => K.mla_attention[t]?.fwd_bwd_tflops ?? null), false, DATASHEET.bf16],
    ["cuDNN, 96 heads, forward", seq.map(t => K.mla_attention[t]?.fwd_tflops ?? null), false, DATASHEET.bf16],
  ];
  if (A) {
    for (const d of [2, 4, 8])
      mlaRows.push([`cuDNN, ${96 / d} heads (TP ${d} or Ulysses CP ${d}), fwd + bwd`,
                    seq.map(t => at(A.mla_tp_cudnn[t], d)?.fwd_bwd_tflops ?? null), false, DATASHEET.bf16]);
    mlaRows.push(["Flex, 96 heads, no CP, fwd + bwd", seq.map(t => A.mla_flex[t]?.fwd_bwd_tflops ?? null), true, DATASHEET.bf16]);
    for (const c of [2, 4, 8])
      mlaRows.push([`Flex, KV all-gather CP ${c}, fwd + bwd`,
                    seq.map(t => at(A.mla_cp_flex[t], c)?.fwd_bwd_tflops ?? null), true, DATASHEET.bf16]);
  }
  riTable(document.getElementById("riAttn"), "MLA attention, TFLOPS / sequence length T (all keys; TP and Ulysses CP keep it)", seq.map(tok), mlaRows, tf);

  // KDA: us per local token per layer, by TP
  if (A) {
    const tps = [1, 2, 4, 8], us = (t, tp, key) => A.kda_tp[t]?.[tp] ? 1e3 * A.kda_tp[t][tp][key] / t : null;
    const KR = ROOFLINE_RULE.kda, kRule = (t, tp) => {
      const tier = KR.tp_speedup.tiers.filter(x => t >= x).length - 1;
      return KR.fwd_bwd / KR.tp_speedup.by_tier[Math.max(0, tier)][KR.tp.indexOf(tp)];
    };
    riTable(document.getElementById("riKda"), "KDA, us per sequence token per layer (96 / TP heads) / TP degree", tps.map(tp => `TP ${tp}`), [
      ["Roofline uses, forward + backward, 8K", tps.map(tp => kRule(8192, tp)), false],
      ["Measured, forward + backward, 8K", tps.map(tp => us(8192, tp, "fwd_bwd_ms")), false],
      ["Roofline uses, forward + backward, 32K", tps.map(tp => kRule(32768, tp)), false],
      ["Measured, forward + backward, 32K", tps.map(tp => us(32768, tp, "fwd_bwd_ms")), false],
      ["Measured, forward (remat replay), 8K", tps.map(tp => us(8192, tp, "fwd_ms")), false],
      ["Measured, forward (remat replay), 32K", tps.map(tp => us(32768, tp, "fwd_ms")), false],
    ], v => v.toFixed(3));
  }

  const sizes = [2 ** 16, 2 ** 20, 2 ** 24, 2 ** 27, 2 ** 30, 2 ** 32];  // ** not <<: 1 << 32 wraps
  const sizeLabel = n => n >= 2 ** 30 ? `${n / 2 ** 30} GiB` : n >= 2 ** 20 ? `${n / 2 ** 20} MiB` : `${n / 1024} KiB`;
  const ops = [["allgather", "All-gather"], ["reducescatter", "Reduce-scatter"], ["alltoall", "All-to-all"], ["sendrecv", "Send/recv"]];
  riTable(document.getElementById("riNvlink"), "4 GPUs, busbw GB/s / message size", sizes.map(sizeLabel),
    ops.filter(([k]) => R.nvlink[k]).map(([k, label]) =>
      [label, sizes.map(n => R.nvlink[k].find(r => r[0] === n)?.[2] ?? null), true, DATASHEET.nvlink]), tf);

  const lat = Math.min(...ops.filter(([k]) => R.nvlink[k]).map(([k]) => R.nvlink[k][0][1]));
  document.getElementById("riNote").replaceChildren(
    el("span", { class: "note-label" }, "Note:"),
    "NVLink reaches about 600 GB/s busbw for large collectives (about two thirds of the datasheet's 900 GB/s), with a latency floor near ",
    b(`${lat.toFixed(0)} us`), " per call; send/recv stalls near 170 GB/s for 16-128 MiB messages, which ",
    "needs a clean re-run because PP sends sit in that range. IB was not measured (one host): this host ",
    "has 2 x 400 Gb/s RoCE NICs per GPU, 100 GB/s per GPU; the target GB200 cluster's per-GPU NIC rate ",
    "still needs confirming. Dense GEMMs run at about two thirds of peak in bf16 and 62% in MXFP8; expert ",
    "GEMMs fall off below 4K tokens per expert, which high CP or TP produces.");
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
