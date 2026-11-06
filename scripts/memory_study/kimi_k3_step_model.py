# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-config step time for Kimi K3 on 512 GB200s: compute, communication,
overlap and the PP schedule, for the roofline page's sections 3 and 4.

For every parallelism config that fits in memory (kimi_k3_configs.json, from
the memory page), per layer and microbatch on one GPU:

- compute: the roofline (one efficiency for every matmul, KDA from its
  measured model) and a calibrated version (section 2's measured efficiencies
  at the config's shapes, KDA's measured CP factor, and the measured work
  outside the roofline);
- communication per parallelism, its overlap window, and what stays exposed:
    FSDP: next layer's all-gather behind this layer's forward; all-gather +
      reduce-scatter behind this layer's backward;
    EP: DistMoE dispatch + combine behind the same pass's routed-expert GEMMs;
    TP (sequence parallel, async TP): behind its block's GEMMs;
    CP (Ulysses all-to-alls for MLA, KDA state all-gather): assumed hidden
      for this study (CP_HIDDEN), reported but not added;
- the step: without PP, microbatches x layers; with PP, 1F1B,
  (M + P - 1) x period, the period being the slowest stage or adjacent pair
  with its forward and backward hop (hidden state + AttnRes block stack);
- once per step: FSDP's all-gather / reduce-scatter under PP ZeRO-1, and
  DistMuon's gather of sharded gradients into whole matrices and scatter of the
  update (not overlapped).

Topology: 8 NVLink domains of 64 GPUs. EP, CP and TP stay inside a domain; a PP
stage of 64 (PP 8) or 32 (PP 16) GPUs sits inside one, so FSDP within a stage
is NVLink and PP hops between domains are InfiniBand. Without PP, FSDP spans
all 512 GPUs and its ring is limited by InfiniBand. NVLink time uses the
measured 4-GPU NCCL bus bandwidth at the message size; InfiniBand is IB_GBPS
per GPU per direction.
"""

import math

NUM_GPUS = 512
NUM_LAYERS = 93
NVLINK_DOMAIN = 64
IB_GBPS = 50.0  # per GPU per direction
BF16_PEAK = 2500e12
HBM_BYTES_PER_S = 8e12  # GB200 datasheet
ROOFLINE_EFF = 0.70  # bf16, as on the page
ATTN_RES_BLOCK = 12  # layers per AttnRes block
MLA_LAYERS = set(range(3, 92, 4)) | {92}  # full_attention_layers in flavors.py
# KDA measured model (section 1.4): us per sequence token per layer, TP speedups
KDA_FWD_BWD_US, KDA_FWD_US = 0.40, 0.05
KDA_TP_SPEEDUP = {
    8192: {1: 1, 2: 1.7, 4: 1.7, 8: 1.7},
    32768: {1: 1, 2: 1.8, 4: 3.0, 8: 5.0},
}
EXPOSED_KEYS = ("fsdp", "ep", "tp", "cp", "pp", "optimizer")
# Study assumption: CP's communication is hidden (head-chunked Ulysses
# pipelining, backward all-to-alls behind weight-gradient GEMMs, or an MLA
# latent all-gather); its time is still reported, not added to the step.
CP_HIDDEN = True


def interp_log(table: dict, x: float) -> float:
    """Interpolate {n: value} linearly in log(n), clamped at the ends."""
    pts = sorted((float(k), v) for k, v in table.items() if v is not None)
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            f = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + f * (y1 - y0)
    return pts[-1][1]


class Links:
    """Collective time: NVLink from the measured 4-GPU NCCL bus-bandwidth
    curves at the message size, InfiniBand flat at IB_GBPS."""

    def __init__(self, nvlink_rows: dict):
        self.curves = {
            op: {r[0]: r[2] * 1e9 for r in rows} for op, rows in nvlink_rows.items()
        }

    def busbw(self, op: str, size: float, link: str) -> float:
        return (
            IB_GBPS * 1e9
            if link == "ib"
            else interp_log(self.curves[op], max(size, 1.0))
        )

    def time(self, op: str, size: float, n: int, link: str) -> float:
        """size: the whole gathered / reduced tensor (all-gather,
        reduce-scatter) or the per-rank buffer (all-to-all); each rank moves
        size x (n - 1) / n. Send/recv moves size."""
        if n <= 1 or size <= 0:
            return 0.0
        moved = size if op == "sendrecv" else size * (n - 1) / n
        return moved / self.busbw(op, size, link)


class StepModel:
    def __init__(self, data: dict, bench: dict):
        self.S = S = data["symbols"]
        stored = lambda sc: {r["key"]: r["numel"] for r in data["scopes"][sc]["stored"]}
        kda, mla = stored("kda_layer"), stored("mla_layer")
        self.n_expert = kda["routed"] / S["E"]  # params per routed expert
        self.attn = {"kda": kda["attention"], "mla": mla["attention"]}
        self.shared = kda["shared"]
        self.dense = {
            t: self.attn[t] + self.shared + kda["others"] for t in ("kda", "mla")
        }
        self.layer0 = kda["attention"] + 3 * S["D"] * S["F"]
        self.tokens_per_gpu = data["global_tokens"] / NUM_GPUS
        self.links = Links(bench["nvlink"])
        self.lin = bench["kernels"]["linears"]["bf16"]
        self.dm = [
            r
            for r in bench["dist_moe"]["results"]
            if r.get("dtype") == "bf16" and "fwd_bwd_gemm_tflops" in r
        ]
        self.fa4 = bench["mla_fa4"]["heads"]
        self.kda_cp = {int(k): v for k, v in (bench.get("kda_cp") or {}).items()}
        fp = bench["fake_pg"]
        self.fp = {
            8192: fp["distmoe_longdocs_8k"],
            32768: fp["distmoe_longdocs_32k_cp2"],
        }
        self.fp_ref = {
            8192: dict(
                seq_len=8192, cp=1, tp=1, pp=1, ep=64, microbatches=4, zero=None
            ),
            32768: dict(
                seq_len=32768, cp=2, tp=1, pp=1, ep=64, microbatches=2, zero=None
            ),
        }

    # ------------------------------------------------------------ helpers
    @staticmethod
    def kind(i):
        return "mla" if i in MLA_LAYERS else "kda"

    def dense_params(self, i):
        return self.layer0 if i == 0 else self.dense[self.kind(i)]

    def stages(self, P):
        Ls = math.ceil(NUM_LAYERS / P)
        return [list(range(s * Ls, min(NUM_LAYERS, (s + 1) * Ls))) for s in range(P)]

    def expert_cal(self, local_tokens, local_experts):
        """DistMoE GEMM-kernel efficiency and helper seconds per call (fwd+bwd)."""
        layout = "ep64" if local_experts <= 16 else "ep32"
        rows = [r for r in self.dm if r["layout"] == layout]
        eff = interp_log(
            {
                r["local_tokens"]: r["fwd_bwd_gemm_tflops"] * 1e12 / BF16_PEAK
                for r in rows
            },
            local_tokens,
        )
        helper = {r["local_tokens"]: r["fwd_bwd_helper_ms"] * 1e-3 for r in rows}
        top = max(helper)
        h = (
            helper[top] * local_tokens / top
            if local_tokens > top
            else interp_log(helper, local_tokens)
        )
        return eff, h

    def kda_cp_factor(self, T, cp, key):
        """Measured KDA time under CP over the same local tokens without CP
        (cp 8 takes cp 4's factor)."""
        rows = self.kda_cp.get(min(cp, 4))
        if cp == 1 or not rows:
            return 1.0
        return interp_log(
            {r["seq_len"]: r[f"cp_{key}_ms"] / r[f"local_{key}_ms"] for r in rows}, T
        )

    def outside_per_token(self, T):
        """Seconds per local token of work outside the roofline (copies, casts,
        norms, elementwise, optimizer), from the fake-PG run, minus DistMoE's
        helper kernels (added per call instead) and FSDP staging copies
        (modelled from FSDP bytes)."""
        tier = 32768 if T >= 32768 else 8192
        r, ref = self.fp[tier], self.fp_ref[tier]
        by = r["breakdown"]["by_class_s"]
        keys = [
            "Copies, casts, cat, memset",
            "Norms, softmax (RMSNorm, AttnRes depth mixing)",
            "MoE token movement (index / sort / scatter)",
            "Elementwise and other fused Triton",
            "Optimizer (Muon / AdamW updates)",
            "other",
        ]
        total = sum(by.get(k, 0.0) for k in keys)
        tl = ref["seq_len"] / ref["cp"]
        _, h = self.expert_cal(tl, 16)  # the fake-PG stand-in has 16 local experts
        total -= (NUM_LAYERS - 1) * ref["microbatches"] * h
        return max(total, 0.0) / self.tokens_per_gpu

    # ------------------------------------------------------------ compute
    def layer_compute(self, i, cfg, cal):
        """Seconds for one microbatch of layer i on one GPU: forward, backward
        (remat forward + dgrad + wgrad), and the GEMM times the overlap windows
        use (routed experts per pass, TP blocks per pass)."""
        S, T, cp, tp = self.S, cfg["seq_len"], cfg["cp"], cfg["tp"]
        Tm, Tl = T / cp, T / (
            cp * tp
        )  # microbatch tokens on this CP rank; local tokens per GPU
        kind = self.kind(i)
        scope = "kda_layer" if (i == 0 or kind == "kda") else "mla_layer"
        rate = (
            (interp_log(self.lin[scope], Tm) * 1e12)
            if cal
            else ROOFLINE_EFF * BF16_PEAK
        )
        # dense linears: 8 FLOPs / param / token (2 fwd, 2 remat, 4 bwd); TP splits the weights
        n_dense = self.dense_params(i)
        dense = 8 * Tm * n_dense / tp / rate
        tp_block = (
            2 * Tm * (n_dense if i == 0 else self.attn[kind] + self.shared) / tp / rate
        )  # one pass
        # routed experts: 8 x local tokens x k x params per expert (ep cancels)
        ex = helper = 0.0
        if i > 0:
            if cal:
                eff, helper = self.expert_cal(Tl, S["E"] / cfg["ep"])
            else:
                eff = ROOFLINE_EFF
            ex = 8 * Tl * S["k"] * self.n_expert / (eff * BF16_PEAK)
        # attention
        if kind == "mla":
            f_flops = (
                Tl * T * S["H"] * (S["Dn"] + S["Dr"] + S["Dv"])
            )  # forward; total 4.5x
            if cal:
                h = min([96, 48, 24, 12], key=lambda x: abs(x - S["H"] / (cp * tp)))
                e_fb = (
                    interp_log(
                        {
                            int(t): v[str(h)]["fwd_bwd_tflops"]
                            for t, v in self.fa4.items()
                        },
                        T,
                    )
                    * 1e12
                )
                e_f = (
                    interp_log(
                        {int(t): v[str(h)]["fwd_tflops"] for t, v in self.fa4.items()},
                        T,
                    )
                    * 1e12
                )
                # backward = remat forward + backward = (fwd + bwd) - fwd + fwd = 3.5 F at the fwd+bwd rate
                attn_f, attn_b = f_flops / e_f, 3.5 * f_flops / e_fb
            else:
                r = ROOFLINE_EFF * BF16_PEAK
                attn_f, attn_b = f_flops / r, 3.5 * f_flops / r
        else:
            tier = 32768 if T >= 32768 else 8192
            fb_us = KDA_FWD_BWD_US / KDA_TP_SPEEDUP[tier][tp]
            f_us = KDA_FWD_US
            if cal:
                fb_us *= self.kda_cp_factor(T, cp, "fwd_bwd")
                f_us *= self.kda_cp_factor(T, cp, "fwd")
            # forward = one forward; backward = remat forward + backward
            attn_f = Tm * f_us * 1e-6
            attn_b = Tm * fb_us * 1e-6
        fwd = dense / 4 + ex / 4 + attn_f
        bwd = 3 * dense / 4 + 3 * ex / 4 + attn_b
        if cal:
            o = self.outside_per_token(T) * Tl / NUM_LAYERS
            fwd += o / 4 + helper / 4
            bwd += 3 * o / 4 + 3 * helper / 4
        return {
            "fwd": fwd,
            "bwd": bwd,
            "ex_fwd": ex / 4,
            "ex_bwd": ex / 2,
            "tp_fwd": tp_block,
            "tp_bwd": 2 * tp_block,
        }

    def lm_head(self, cfg, cal):
        S, Tm, tp = self.S, cfg["seq_len"] / cfg["cp"], cfg["tp"]
        rate = (
            interp_log(self.lin["lm_head"], Tm) * 1e12
            if cal
            else ROOFLINE_EFF * BF16_PEAK
        )
        t = 6 * Tm * S["V"] * S["D"] / tp / rate  # no remat: forward 2, backward 4
        return t / 3, 2 * t / 3

    def embedding(self, cfg):
        """Token-embedding lookup on the first stage: memory-bound, no FLOPs.
        Forward reads and writes local tokens x D bf16; backward scatter-adds
        the bf16 gradient rows into the fp32 weight gradient (read + write)."""
        Tl, D = cfg["seq_len"] / (cfg["cp"] * cfg["tp"]), self.S["D"]
        return 4 * Tl * D / HBM_BYTES_PER_S, (2 + 8) * Tl * D / HBM_BYTES_PER_S

    def embedding_comm(self, cfg):
        """FSDP all-gather (bf16) and reduce-scatter (fp32) of the embedding
        table, vocab-split by TP and sharded over the stage's FSDP ranks."""
        stage, n_dense, edp, link = self.fsdp_groups(cfg)
        n = self.S["V"] * self.S["D"] / cfg["tp"]
        return (
            self.links.time("allgather", 2 * n, n_dense, link),
            self.links.time("reducescatter", 4 * n, n_dense, link),
        )

    # ------------------------------------------------------------ communication
    def fsdp_groups(self, cfg):
        stage = NUM_GPUS // cfg["pp"]
        return (
            stage,
            stage // cfg["tp"],
            stage // cfg["ep"],
            ("ib" if stage > NVLINK_DOMAIN else "nvlink"),
        )

    def layer_comm(self, i, cfg):
        """Seconds of each collective of layer i for one microbatch (FSDP: one
        all-gather and one reduce-scatter)."""
        L, S = self.links, self.S
        T, cp, tp, ep = cfg["seq_len"], cfg["cp"], cfg["tp"], cfg["ep"]
        Tm, Tl = T / cp, T / (cp * tp)
        kind = self.kind(i)
        stage, n_dense, edp, link = self.fsdp_groups(cfg)
        c = {}
        # FSDP: dense sharded over the stage / tp (CP folds into FSDP); experts
        # sharded EP ways and replicated edp ways (edp > 1 only without PP)
        n_d = self.dense_params(i) / tp
        c["fsdp_ag"] = L.time("allgather", 2 * n_d, n_dense, link)
        c["fsdp_rs"] = L.time("reducescatter", 4 * n_d, n_dense, link)
        c["fsdp_bytes"] = (
            8 * n_d
        )  # 2 x bf16 gather (forward, backward) + fp32 reduce, per microbatch
        if i > 0 and edp > 1:
            shard = S["E"] * self.n_expert / ep
            c["fsdp_ag"] += L.time("allgather", 2 * shard, edp, link)
            c["fsdp_rs"] += L.time("reducescatter", 4 * shard, edp, link)
            c["fsdp_bytes"] += 8 * shard
        # EP (DistMoE): dispatch + combine, local tokens x k x latent width x bf16
        one = (
            L.time("alltoall", Tl * S["k"] * S["L"] * 2, ep, "nvlink") if i > 0 else 0.0
        )
        c["ep_fwd"] = c["ep_bwd"] = 2 * one
        # TP (SP): all-gather + reduce-scatter of [Tm, D] around each block
        if tp > 1:
            x = Tm * S["D"] * 2
            per = L.time("allgather", x, tp, "nvlink") + L.time(
                "reducescatter", x, tp, "nvlink"
            )
            c["tp_fwd"] = c["tp_bwd"] = (1 if i == 0 else 2) * per
        else:
            c["tp_fwd"] = c["tp_bwd"] = 0.0
        # CP: MLA Ulysses (q, k, v in, output out); KDA all-gathers fp32 state
        if cp > 1:
            if kind == "mla":
                per_tok = S["H"] / tp * (2 * (S["Dn"] + S["Dr"]) + 2 * S["Dv"]) * 2
                one = L.time("alltoall", Tl * per_tok, cp, "nvlink")
            else:
                one = L.time(
                    "allgather", cp * S["H"] / tp * S["K"] * S["K"] * 4, cp, "nvlink"
                )
            c["cp_fwd"], c["cp_bwd"] = (
                one,
                2 * one,
            )  # backward: remat forward + backward
        else:
            c["cp_fwd"] = c["cp_bwd"] = 0.0
        return c

    def hop_bytes(self, layers_before, cfg):
        """One PP hop after layers_before layers: hidden state plus the AttnRes
        blocks so far (a partial block counts)."""
        Tl = cfg["seq_len"] / (cfg["cp"] * cfg["tp"])
        return (1 + math.ceil(layers_before / ATTN_RES_BLOCK)) * Tl * self.S["D"] * 2

    def optimizer_comm(self, cfg):
        """DistMuon: each sharded matrix's fp32 gradient gathered whole and the
        update scattered back, as two all-to-alls of the GPU's shard."""
        stage, n_dense, edp, link = self.fsdp_groups(cfg)
        layers = range(NUM_LAYERS) if cfg["pp"] == 1 else self.stages(cfg["pp"])[0]
        dense = sum(self.dense_params(i) for i in layers) / cfg["tp"]
        t = 2 * self.links.time("alltoall", 4 * dense / n_dense, n_dense, link)
        if edp > 1:
            moe = (
                sum(1 for i in layers if i > 0)
                * self.S["E"]
                * self.n_expert
                / cfg["ep"]
            )
            t += 2 * self.links.time("alltoall", 4 * moe / edp, edp, link)
        return t

    # ------------------------------------------------------------ calibrated extras
    def fsdp_step_bytes(self, cfg):
        P, M = cfg["pp"], cfg["microbatches"]
        layers = range(NUM_LAYERS) if P == 1 else self.stages(P)[0]
        per = sum(self.layer_comm(i, cfg)["fsdp_bytes"] for i in layers)
        per += 8 * self.S["V"] * self.S["D"] / cfg["tp"]  # embedding (stage 0)
        return per * (M if (P == 1 or cfg.get("zero") == 3) else 1)

    def fsdp_copies(self, cfg):
        """FSDP staging copies (chunk_cat / split copy-out), at the measured
        seconds per FSDP byte of the 8K fake-PG run."""
        if not hasattr(self, "_copy_rate"):
            meas = self.fp[8192]["breakdown"]["by_class_s"][
                "FSDP buffer copies (chunk_cat / split copy-out)"
            ]
            self._copy_rate = meas / self.fsdp_step_bytes(self.fp_ref[8192])
        return self._copy_rate * self.fsdp_step_bytes(cfg)

    def idle_eager(self, cfg):
        """Eager-mode host idle, scaled from the fake-PG runs by layer calls per
        GPU per step (microbatches x layers on the GPU)."""
        tier = 32768 if cfg["seq_len"] >= 32768 else 8192
        r, ref = self.fp[tier], self.fp_ref[tier]
        idle = r["clean_step_s"] - r["breakdown"]["gpu_busy_s"]
        calls = cfg["microbatches"] * len(self.stages(cfg["pp"])[0])
        return idle * calls / (ref["microbatches"] * NUM_LAYERS)

    # ------------------------------------------------------------ step
    def stage_time(self, layers, cfg, cal, last, first=False):
        """Per microbatch on one GPU of the stage: compute, exposed comm by
        kind, and total (forward and backward)."""
        per_mb_fsdp = cfg["pp"] == 1 or cfg.get("zero") == 3
        comp = 0.0
        exp = {k: 0.0 for k in ("fsdp", "ep", "tp", "cp")}
        comm = {k: 0.0 for k in ("fsdp", "ep", "tp", "cp")}
        once = {"ag": 0.0, "rs": 0.0}
        fsdp_split = {"fwd": 0.0, "bwd": 0.0}  # exposed FSDP by pass, per microbatch
        for i in layers:
            lc, cm = self.layer_compute(i, cfg, cal), self.layer_comm(i, cfg)
            comp += lc["fwd"] + lc["bwd"]
            exp["ep"] += 2 * max(0.0, cm["ep_fwd"] - lc["ex_fwd"]) + max(
                0.0, cm["ep_bwd"] - lc["ex_bwd"]
            )
            exp["tp"] += 2 * max(0.0, cm["tp_fwd"] - lc["tp_fwd"]) + max(
                0.0, cm["tp_bwd"] - lc["tp_bwd"]
            )
            exp["cp"] += 0.0 if CP_HIDDEN else cm["cp_fwd"] + cm["cp_bwd"]
            comm["ep"] += 2 * cm["ep_fwd"] + cm["ep_bwd"]
            comm["tp"] += 2 * cm["tp_fwd"] + cm["tp_bwd"]
            comm["cp"] += cm["cp_fwd"] + cm["cp_bwd"]
            if per_mb_fsdp:
                f_x = max(0.0, cm["fsdp_ag"] - lc["fwd"])
                b_x = max(0.0, cm["fsdp_ag"] + cm["fsdp_rs"] - lc["bwd"])
                exp["fsdp"] += f_x + b_x
                fsdp_split["fwd"] += f_x
                fsdp_split["bwd"] += b_x
                comm["fsdp"] += 2 * cm["fsdp_ag"] + cm["fsdp_rs"]
            else:  # ZeRO-1 under PP: once per step, behind the first forward / last backward
                once["ag"] += max(0.0, cm["fsdp_ag"] - lc["fwd"])
                once["rs"] += max(0.0, cm["fsdp_rs"] - lc["bwd"])
                comm["fsdp"] += (cm["fsdp_ag"] + cm["fsdp_rs"]) / cfg["microbatches"]
        if first:
            # embedding: its FSDP gather / reduce-scatter hide behind layer 0, like a layer's
            f, b = self.embedding(cfg)
            comp += f + b
            ag, rs = self.embedding_comm(cfg)
            l0 = self.layer_compute(layers[0], cfg, cal)
            if per_mb_fsdp:
                exp["fsdp"] += max(0.0, ag - l0["fwd"]) + max(0.0, ag + rs - l0["bwd"])
                fsdp_split["fwd"] += max(0.0, ag - l0["fwd"])
                fsdp_split["bwd"] += max(0.0, ag + rs - l0["bwd"])
                comm["fsdp"] += 2 * ag + rs
            else:
                once["ag"] += max(0.0, ag - l0["fwd"])
                once["rs"] += max(0.0, rs - l0["bwd"])
                comm["fsdp"] += (ag + rs) / cfg["microbatches"]
        if last:
            f, b = self.lm_head(cfg, cal)
            comp += f + b
        return {
            "compute": comp,
            "exposed": exp,
            "comm": comm,
            "once": once,
            "fsdp_split": fsdp_split,
            "total": comp + sum(exp.values()),
        }

    def evaluate(self, cfg, cal):
        P, M = cfg["pp"], cfg["microbatches"]
        stages = self.stages(P)
        st = [
            self.stage_time(layers, cfg, cal, s == P - 1, first=s == 0)
            for s, layers in enumerate(stages)
        ]
        out = {
            "exposed": {k: 0.0 for k in EXPOSED_KEYS},
            "comm": {k: 0.0 for k in EXPOSED_KEYS},
        }
        if P == 1:
            s = st[0]
            out["compute_s"] = M * s["compute"]
            out["fsdp_exposed_fwd_s"] = M * s["fsdp_split"]["fwd"]
            out["fsdp_exposed_bwd_s"] = M * s["fsdp_split"]["bwd"]
            for k, v in s["exposed"].items():
                out["exposed"][k] = M * v
                out["comm"][k] = M * s["comm"][k]
            out["bubble_s"] = 0.0
            step = M * s["total"]
        else:
            # 1F1B: steady-state period = slowest stage, or slowest adjacent pair
            # including the forward and backward hop between them
            per_stage = [s["total"] for s in st]
            period, hops = max(per_stage), []
            for s in range(P - 1):
                before = sum(len(x) for x in stages[: s + 1])
                g = NUM_GPUS // P
                link = (
                    "nvlink"
                    if (s * g) // NVLINK_DOMAIN == ((s + 1) * g) // NVLINK_DOMAIN
                    else "ib"
                )
                c = self.links.time("sendrecv", self.hop_bytes(before, cfg), 2, link)
                hops.append(c)
                period = max(period, per_stage[s] + 2 * c)
            out["stages"] = [
                {
                    "first": x[0],
                    "last": x[-1],
                    "layers": len(x),
                    "mla": sum(1 for i in x if i in MLA_LAYERS),
                    "lm_head": s == P - 1,
                    "embedding": s == 0,
                    "t_s": per_stage[s],
                    "embedding_s": sum(self.embedding(cfg)) if s == 0 else 0.0,
                    "hop_bytes": self.hop_bytes(
                        sum(len(y) for y in stages[: s + 1]), cfg
                    )
                    if s < P - 1
                    else 0.0,
                    "hop_s": hops[s] if s < P - 1 else 0.0,
                }
                for s, x in enumerate(stages)
            ]
            w = per_stage.index(max(per_stage))
            out["compute_s"] = M * st[w]["compute"]
            out["fsdp_exposed_fwd_s"] = M * st[w]["fsdp_split"]["fwd"]
            out["fsdp_exposed_bwd_s"] = M * st[w]["fsdp_split"]["bwd"]
            for k, v in st[w]["exposed"].items():
                out["exposed"][k] = M * v
                out["comm"][k] = M * st[w]["comm"][k]
            # all waiting on hops, fill and drain included; the bubble is the idle stage time
            out["exposed"]["pp"] = (M + P - 1) * (period - per_stage[w])
            out["comm"]["pp"] = M * 2 * max(hops)
            out["bubble_s"] = (P - 1) * per_stage[w]
            out["period_s"], out["hop_s"] = period, max(hops)
            out["stage_imbalance"] = max(per_stage) / (sum(per_stage) / P) - 1
            step = (M + P - 1) * period
            if cfg.get("zero") != 3:
                once = st[0]["once"]["ag"] + st[w]["once"]["rs"]
                out["exposed"]["fsdp"] += once
                step += once
        opt = self.optimizer_comm(cfg)
        out["exposed"]["optimizer"] = out["comm"]["optimizer"] = opt
        step += opt
        if cal:
            out["fsdp_copies_s"] = self.fsdp_copies(cfg)
            step += out["fsdp_copies_s"]
        out["step_s"] = step
        out["idle_eager_s"] = self.idle_eager(cfg)
        # one KDA MoE layer and one MLA layer, per microbatch: each collective
        # against its overlap window (FSDP shown per microbatch even when it runs
        # once per step under ZeRO-1)
        lk, ck = self.layer_compute(1, cfg, cal), self.layer_comm(1, cfg)
        lm, cmm = self.layer_compute(3, cfg, cal), self.layer_comm(3, cfg)
        out["layer"] = {
            "fwd": lk["fwd"],
            "bwd": lk["bwd"],
            "fsdp_ag": ck["fsdp_ag"],
            "fsdp_rs": ck["fsdp_rs"],
            "fsdp_per_mb": P == 1 or cfg.get("zero") == 3,
            "ep": ck["ep_fwd"],
            "ex_fwd": lk["ex_fwd"],
            "ex_bwd": lk["ex_bwd"],
            "tp": ck["tp_fwd"],
            "tp_gemm": lk["tp_fwd"],
            "cp_mla": cmm["cp_fwd"] + cmm["cp_bwd"],
            "cp_kda": ck["cp_fwd"] + ck["cp_bwd"],
            "mla_fwd": lm["fwd"],
            "mla_bwd": lm["bwd"],
        }
        out["bound_by"] = [k for k in EXPOSED_KEYS if out["exposed"][k] > 0.03 * step]
        return out


def tp_vs_cp(data: dict, bench: dict) -> list:
    """Section 4.4: CP, TP and CP 2 x TP at the same degree, PP 8 ZeRO-1,
    memory ignored; calibrated step and the per-step compute by component
    (all layers x microbatches / P, so averaged over stages)."""
    m = StepModel(data, bench)
    S, out = m.S, []
    for T in (8192, 32768):
        seqs = data["global_tokens"] // T
        # TP is swept up to 4 (the memory page's TP columns), so degree 8 mixes CP 2 x TP 4
        for deg, splits in (
            (2, [(2, 1), (1, 2)]),
            (4, [(4, 1), (1, 4), (2, 2)]),
            (8, [(8, 1), (2, 4)]),
        ):
            for cp, tp in splits:
                M = seqs * 8 * cp * tp // NUM_GPUS
                cfg = dict(
                    seq_len=T,
                    cp=cp,
                    tp=tp,
                    pp=8,
                    ep=NUM_GPUS // 8,
                    microbatches=M,
                    zero=1,
                )
                r = m.evaluate(cfg, True)
                Tm, Tl = T / cp, T / (cp * tp)
                parts = {"dense": 0.0, "kda": 0.0, "mla": 0.0, "experts": 0.0}
                for i in range(NUM_LAYERS):
                    lc = m.layer_compute(i, cfg, True)
                    kind = m.kind(i)
                    scope = "kda_layer" if (i == 0 or kind == "kda") else "mla_layer"
                    d = (
                        8
                        * Tm
                        * m.dense_params(i)
                        / tp
                        / (interp_log(m.lin[scope], Tm) * 1e12)
                    )
                    ex = h = 0.0
                    if i > 0:
                        eff, h = m.expert_cal(Tl, S["E"] / cfg["ep"])
                        ex = 8 * Tl * S["k"] * m.n_expert / (eff * BF16_PEAK)
                    o = m.outside_per_token(T) * Tl / NUM_LAYERS
                    parts["dense"] += d
                    parts["experts"] += ex + h
                    parts[kind] += lc["fwd"] + lc["bwd"] - d - ex - h - o
                f = M / 8
                out.append(
                    {
                        "seq_len": T,
                        "degree": deg,
                        "cp": cp,
                        "tp": tp,
                        "microbatches": M,
                        "step_s": r["step_s"],
                        "bubble_s": r["bubble_s"],
                        "parts": {k: v * f for k, v in parts.items()},
                    }
                )
    return out


def rank_configs(data: dict, bench: dict, configs: list) -> list:
    """Every config that fits and is valid, with its roofline and calibrated
    step, sorted by calibrated step time within each context."""
    m = StepModel(data, bench)
    rows = [
        {**c, "roofline": m.evaluate(c, False), "calibrated": m.evaluate(c, True)}
        for c in configs
        if c["fits"] and c["valid"]
    ]
    return sorted(rows, key=lambda r: (r["seq_len"], r["calibrated"]["step_s"]))


def link_table(bench: dict) -> dict:
    """Section 3: the bandwidths the model uses."""
    L = Links(bench["nvlink"])
    sizes = [2**20, 2**24, 2**27, 2**30, 2**32]
    return {
        "ib_gbps": IB_GBPS,
        "nvlink_domain": NVLINK_DOMAIN,
        "num_gpus": NUM_GPUS,
        "nvlink": {
            op: [[s, L.busbw(op, s, "nvlink") / 1e9] for s in sizes] for op in L.curves
        },
    }
