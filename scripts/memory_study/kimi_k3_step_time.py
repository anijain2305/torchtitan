# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Step time per config for Kimi K3 on 512 GB200s: the practical roofline plus
communication, the overlap penalty and the PP schedule.

Per layer and microbatch on one GPU, forward and backward separately:
  - compute: each op at its measured efficiency for the config's shapes, and
    the memory-bound work at its measured bandwidth (blog section 5.2);
  - communication: what each parallelism moves, on its link (NVLink from the
    4-GPU nccl-tests curves; InfiniBand at IB_GBPS per GPU per direction);
  - overlap: each collective runs in the window it hides behind, stretched by
    the measured collective slowdown; the compute it overlaps is charged the
    measured compute slowdown for as long as the collective runs; what does
    not fit in the window is exposed:
      FSDP (ZeRO-3, without PP): the next layer's all-gather behind this
        layer's forward; all-gather + reduce-scatter behind its backward;
      EP: DistMoE's dispatch + combine behind the routed-expert GEMMs (fused,
        no separate kernel, so no extra penalty);
      TP (sequence parallel) and CP: exposed, as TorchTitan runs them with
        MXFP8 today: async TP cannot fuse MXFP8's block-scaled GEMMs, and
        Ulysses' all-to-alls run before and after attention;
  - PP, as TorchTitan runs Kimi K3 today with plain 1F1B: every hop carries the
    hidden state and the whole AttnRes block stack, on the critical path; the
    step is (M + P - 1) x the slowest stage's period, the period including the
    stage's incoming forward and backward hops; sender and receiver are charged
    the measured one-way penalties while a hop overlaps their compute; under
    ZeRO-1 the stage's FSDP all-gather and reduce-scatter run once per step.
"""

import json
import math
from pathlib import Path

import blog_figures as bf

HERE = Path(__file__).resolve().parent
NUM_GPUS, NUM_LAYERS, BLOCK = 512, 93, 12
NVLINK_DOMAIN = 64
IB_GBPS = 50.0  # GB200 NIC line rate per GPU per direction (400 Gb/s)
# Measured on the GB300 cluster (islands of 4 or 8 GPUs on NVLink, 800 Gb/s NIC
# per GPU): point-to-point send/recv reaches 0.90 of line rate. NCCL runs
# all-gather / reduce-scatter as a flat ring per channel, with 4 channels per
# NIC up to 32: NVLink inside an island, a NIC at each island boundary, all
# pipelined. Each ring hands every GPU its share of the island's NICs, so a
# group with k GPUs (so k NICs) in each island gets min(NVLink, k x 0.92 x line
# rate): islands of 4: 3.7 NICs' worth; of 8: about 7; groups of one GPU per
# island (an expert shard's replicas, one per domain): one NIC, 0.82 of it
# (NCCL's PXN does not help); two per island: 1.8 NICs.
IB_P2P_EFF, IB_COLL_EFF = 0.90, 0.92
IB_COLL_EFF_ONE_NIC = 0.82
# A ring over n GPUs takes n - 1 sequential steps; per step about 6.5 us when the
# ring crosses InfiniBand and 2.2 us over NVLink alone (small all-gathers, 32
# GPUs). A ring over 8 domains is mostly NVLink steps: take 4 us.
RING_STEP_S = 4e-6
# DistMuon's optimizer step at the practical roofline: its GPU work, measured in
# fake-process-group steps at 8K (FSDP 512 + EP 64: the optimizer phase's busy
# time; PP 4, stage 0: the step's span, where the GPU is the limit). PP 8 and 16
# hold fewer layers per GPU on the same parameters per GPU; take PP 4's time.
# Run eagerly, as TorchTitan does today, FSDP 512's step takes 3.1 s: launching
# ~700k small ops for all 93 layers' parameters leaves the GPU waiting on the CPU.
OPTIMIZER_S = {"fsdp": 0.82, "pp": 0.50}
# PP hop transport: "chain" (TorchTitan's 1F1B: hidden + whole block stack every
# hop), "cache" (TorchTitan's interleaved: only blocks the receiving rank has not
# seen), "direct" (each block straight to every consumer; only the hidden state
# on the critical path)
MLA_LAYERS = set(range(3, 92, 4)) | {92}

# Overlap penalties with NCCL's default 32 channels over NVLink (4 GPUs of a
# host, and 16 / 32 GPUs of one NVL72 domain alike; blog section 6): compute
# slowdown while a collective runs, and the collective's own slowdown while it
# shares the GPU.
COMPUTE_SLOWDOWN = 1.29
COMM_SLOWDOWN = 1.6
# The same across InfiniBand: NCCL takes one SM per channel, so the slowdown
# follows the island's NICs: islands of 4 (16 channels) 1.18x, of 8 (32
# channels) 1.29x and a collective slowdown of about 1.7x. GB200's 64-GPU
# domains have enough NICs for NCCL's 32 channels. Used for FSDP across NVLink
# domains.
COMPUTE_SLOWDOWN_IB = 1.29
COMM_SLOWDOWN_IB = 1.7
# A one-way transfer (PP hop) across InfiniBand (rank i -> rank i + N / 2, 8
# and 32 GPUs alike): NCCL runs one thread block per side (the NIC moves the
# data), and the compute slows down about the same on the sender and the
# receiver: GEMMs 1.10x, memory-bound kernels 1.01x.
HOP_GEMM_SLOWDOWN, HOP_MEM_SLOWDOWN = 1.10, 1.01


def _interp(pts, x):
    return bf._interp(dict(pts), x)


class Links:
    """Collective time: NVLink from measured bus bandwidth at the message size
    (up to 4 ranks: nccl-tests on one 4-GPU host; more: one multi-host NVL72
    domain on GB300, 16 and 32 GPUs, the 32-GPU curve also standing in for 64);
    InfiniBand flat at IB_GBPS."""

    OPS = ("allgather", "reducescatter", "alltoall", "sendrecv")

    def __init__(self):
        host = {}
        for op in self.OPS:
            rows = {}
            for line in (
                (HERE / "bench" / f"nccl_{op}_4gpu.txt").read_text().splitlines()
            ):
                p = line.split()
                if len(p) > 8 and p[0].isdigit() and p[2] == "bfloat16":
                    rows[int(p[0])] = float(p[7]) * 1e9  # busbw
            host[op] = rows
        self.by_world = {4: host}
        for hosts in (4, 8):
            rows = json.loads(
                (
                    HERE / "bench" / f"nvl_collectives_GB300_{hosts}hosts.json"
                ).read_text()
            )["rows"]
            self.by_world[4 * hosts] = {
                op: {
                    r["bytes"]: r["busbw_GBps"] * 1e9
                    for r in rows
                    if r["op"] == op and r["tag"] == f"nvl_{hosts}hosts"
                }
                for op in self.OPS
            }
        self.curves = host

    def nvlink_curve(self, op, n):
        """The measured curve of the smallest world >= n (the largest if none)."""
        w = min((w for w in self.by_world if w >= n), default=max(self.by_world))
        return self.by_world[w][op]

    def time(self, op, size, n, link):
        """size: the whole gathered / reduced tensor (all-gather,
        reduce-scatter), the per-rank buffer (all-to-all), or the message
        (send/recv). Each rank moves size x (n - 1) / n (size for send/recv)."""
        if n <= 1 or size <= 0:
            return 0.0
        moved = size if op == "sendrecv" else size * (n - 1) / n
        nv_bw = bf._interp(self.nvlink_curve(op, min(n, NVLINK_DOMAIN)), max(size, 1.0))
        if link == "ib_ring":
            # rings through every NIC the group has in a domain (one per GPU of
            # the group there; an expert shard's 8 replicas: one), plus the
            # ring's n - 1 sequential steps
            nics = max(
                1, n // (NUM_GPUS // NVLINK_DOMAIN)
            )  # the group's GPUs per domain
            eff = IB_COLL_EFF if nics > 1 else IB_COLL_EFF_ONE_NIC
            return (
                moved / min(nv_bw, nics * IB_GBPS * 1e9 * eff) + (n - 1) * RING_STEP_S
            )
        if link == "ib":
            if op == "sendrecv":
                return moved / (IB_GBPS * 1e9 * IB_P2P_EFF)
            return moved / (
                IB_GBPS * 1e9
            )  # naive: every byte through the GPU's own NIC
        return moved / nv_bw


class Config:
    def __init__(self, ctx, P=1, cp=1, tp=1, v=1, transport=None):
        d = bf._data()
        self.ctx, self.P, self.cp, self.tp = ctx, P, cp, tp
        self.T = 8192 if ctx == "8K" else 32768
        self.ep = NUM_GPUS // P if P > 1 else 64
        self.Tm = self.T // cp  # GEMM rows per microbatch (after the SP all-gather)
        self.Tl = self.T // (cp * tp)  # local tokens per microbatch
        self.stage_gpus = NUM_GPUS // P
        self.dp = NUM_GPUS // (P * cp * tp)
        self.M = int(
            d["global_tokens"] // self.T // self.dp
        )  # microbatches per pipeline
        self.v = v
        self.transport = transport or ("chain" if v == 1 else "cache")
        # P x v virtual stages, the first NUM_LAYERS % (P v) one layer longer;
        # rank r runs virtual stages r, r + P, ...
        base, extra = divmod(NUM_LAYERS, P * v)
        sizes = [base + (1 if i < extra else 0) for i in range(P * v)]
        starts = [sum(sizes[:i]) for i in range(P * v)]
        self.vstages = [
            list(range(starts[i], starts[i] + sizes[i])) for i in range(P * v)
        ]
        self.stages = [
            [l for i in range(r, P * v, P) for l in self.vstages[i]] for r in range(P)
        ]


class StepModel:
    def __init__(self):
        d = bf._data()
        self.S = d["symbols"]
        sc = {
            k: {r["key"]: r["numel"] for r in v["stored"]}
            for k, v in d["scopes"].items()
        }
        self.kda, self.mla = sc["kda_layer"], sc["mla_layer"]
        self.n_expert = self.kda["routed"] / self.S["E"]
        self.links = Links()
        self.shapes = bf._bench("shapes_GB300.json")
        self.fa4 = bf._bench("mla_fa4_GB300.json")["mla_fa4"]["data"]["heads"]
        self.lm = bf._bench("kernels_GB300_summary.json")["linears"]["bf16"]["lm_head"]

    # ------------------------------------------------------------ compute
    def dense_params(self, i):
        s = self.mla if i in MLA_LAYERS else self.kda
        n = s["attention"] + s["shared"] + s["others"]
        return n + (3 * self.S["D"] * self.S["F"] - s["shared"] if i == 0 else 0)

    def layer_compute(self, c, i):
        """Seconds for one microbatch of layer i on one GPU: forward and
        backward (remat forward + backward), plus the windows comm hides behind."""
        S = self.S
        mla = i in MLA_LAYERS
        scope = "mla_layer" if mla else "kda_layer"
        g = self.shapes["dense"]["data"][f"m{c.Tm}_tp{c.tp}"][scope]["gemms"]
        fwd_d = sum(x["fwd"] for x in g) * 1e-3
        bwd_d = sum(x["fwd"] + x["dgrad"] + x["wgrad"] for x in g) * 1e-3
        if (
            i == 0
        ):  # dense FFN instead of MoE (about 3 D F parameters), at the layer's rate
            eff = self.shapes["dense"]["data"][f"m{c.Tm}_tp{c.tp}"]["kda_layer"]["eff"]
            ffn = 2 * c.Tm * 3 * S["D"] * S["F"] / c.tp / (eff * bf.MXFP8_PEAK)
            fwd_d, bwd_d = fwd_d + ffn, bwd_d + 3 * ffn
        ex_f = ex_b = 0.0
        if i > 0:
            t = (
                8
                * c.Tl
                * S["k"]
                * self.n_expert
                / (bf.distmoe_eff(c.Tl, c.ep) * bf.MXFP8_PEAK)
            )
            ex_f, ex_b = t / 4, 3 * t / 4
        if mla:
            fa = self.fa4[str(c.T)][str(96 // (c.cp * c.tp))]
            F = (
                c.T * c.T * (96 // (c.cp * c.tp)) * (S["Dn"] + S["Dr"] + S["Dv"])
            )  # one forward
            at_f = F / (fa["fwd_tflops"] * 1e12)
            at_b = 3.5 * F / (fa["fwd_bwd_tflops"] * 1e12)  # remat forward + backward
        else:
            cpf = {"fwd_bwd": 1.0, "fwd": 1.0}
            if c.cp > 1:
                r = {
                    x["seq_len"]: x
                    for x in bf._bench(f"kda_cp_GB300_cp{c.cp}.json")["results"]
                }[c.T]
                cpf = {k: r[f"cp_{k}_ms"] / r[f"local_{k}_ms"] for k in cpf}
            sp = bf.KDA_TP_SPEEDUP[c.ctx][c.tp]
            tok = c.Tm  # tokens through the GPU's 1 / TP of the heads
            at_f = tok * bf.KDA_FWD_US * cpf["fwd"] * 1e-6 / sp
            at_b = tok * bf.KDA_FWD_BWD_US * cpf["fwd_bwd"] * 1e-6 / sp
        return {
            "fwd": fwd_d + ex_f + at_f,
            "bwd": bwd_d + ex_b + at_b,
            "dense_fwd": fwd_d,
            "dense_bwd": bwd_d,
            "ex_fwd": ex_f,
            "ex_bwd": ex_b,
        }

    def memory_bound(self, c):
        """Memory-bound seconds per GPU per step at the measured bandwidths:
        (per microbatch per layer, once per step). Under PP's ZeRO-1 the FSDP
        copies happen once per step."""
        t, once, _ = self._memory_bound_parts(c)
        # t covers all 93 layers for one GPU's tokens per step; spread it over the
        # layer calls the GPU makes (its layers x the microbatches it processes)
        calls = NUM_LAYERS / c.P * c.M
        per_layer_mb = sum(t.values()) / calls
        return per_layer_mb, once

    def _memory_bound_parts(self, c):
        """Memory-bound seconds per step by class at the measured bandwidths:
        (per-layer classes, FSDP copies done once per step under PP, the share
        of the per-layer time that prepares weights: FSDP buffer copies under
        ZeRO-3 and the weights' MXFP8 quantization)."""
        meas = bf._bench("step_graphs_fsdp_ep64_8k.json")["fwd_bwd_by_class_s"]
        ref = bf.nonmatmul_bytes("8K")
        nm = bf.nonmatmul_bytes(
            c.ctx, cp=c.cp, tp=c.tp, ep=c.ep, P=c.P, split_weights=True
        )
        wq = nm.pop("MXFP8 quantize (weights)")
        eff = {
            k: ref[k] / bf.HBM_BW / meas[k] for k in ref
        }  # achieved fraction of HBM bandwidth
        t = {k: b / bf.HBM_BW / eff[k] for k, b in nm.items()}
        once = t.pop("FSDP buffer copies") if c.P > 1 else 0.0
        weights = wq / bf.HBM_BW / eff["MXFP8 quantize"] + t.get(
            "FSDP buffer copies", 0.0
        )
        return t, once, weights / sum(t.values())

    # ------------------------------------------------------------ communication
    def layer_comm(self, c, i):
        """Seconds of each collective of layer i for one microbatch."""
        L, S = self.links, self.S
        n_dense = c.stage_gpus // c.tp  # FSDP ranks of the dense weights (CP folds in)
        edp = c.stage_gpus // c.ep  # replicas of each expert, FSDP-sharded across them
        link = "ib_ring" if c.stage_gpus > NVLINK_DOMAIN else "nvlink"
        nd = self.dense_params(i) / c.tp
        ag = L.time("allgather", 2 * nd, n_dense, link)
        rs = L.time("reducescatter", 4 * nd, n_dense, link)
        if i > 0 and edp > 1:
            shard = S["E"] * self.n_expert / c.ep
            ag += L.time("allgather", 2 * shard, edp, link)
            rs += L.time("reducescatter", 4 * shard, edp, link)
        ep = (
            L.time("alltoall", c.Tl * S["k"] * S["L"] * 2, c.ep, "nvlink") * 2
            if i > 0
            else 0.0
        )
        tp = 0.0
        if c.tp > 1:
            x = c.Tm * S["D"] * 2
            tp = 2 * (
                L.time("allgather", x, c.tp, "nvlink")
                + L.time("reducescatter", x, c.tp, "nvlink")
            )
        cpt = 0.0
        if c.cp > 1:
            if i in MLA_LAYERS:
                per_tok = S["H"] / c.tp * (2 * (S["Dn"] + S["Dr"]) + 2 * S["Dv"]) * 2
                cpt = L.time("alltoall", c.Tl * per_tok, c.cp, "nvlink")
            else:
                cpt = L.time(
                    "allgather",
                    c.cp * S["H"] / c.tp * S["K"] * S["K"] * 4,
                    c.cp,
                    "nvlink",
                )
        return {"fsdp_ag": ag, "fsdp_rs": rs, "ep": ep, "tp": tp, "cp": cpt}

    def hop_bytes(self, c, vs):
        """Bytes of the hop into virtual stage vs on the critical path, by
        transport: the hidden state (which carries the current block's partial
        sum) plus AttnRes stack entries (the embedding and finished blocks)."""
        S = c.Tl * self.S["D"] * 2
        l0 = c.vstages[vs][0]
        needed = 1 + l0 // BLOCK
        if c.transport == "direct":
            return S
        if c.transport == "cache" and vs >= c.P:
            known = (
                1 + (c.vstages[vs - c.P][-1] + 1) // BLOCK
            )  # the rank's previous visit
            return S * (1 + needed - known)
        return S * (1 + needed)

    def direct_side_bytes(self, c, r):
        """Direct distribution: per microbatch, the stack bytes rank r sends and
        receives off the critical path, forward + backward. Each block goes once to
        every other rank that runs a later layer (or the output, after the last
        layer), which keeps it for its later chunks; its gradient pieces come back.
        Under interleaving that is nearly every rank."""
        S = c.Tl * self.S["D"] * 2
        owner = {l: q for q in range(c.P) for l in c.stages[q]}
        # (producing layer, producing rank): the embedding, then each finished block
        blocks = [(-1, 0)] + [
            (i, owner[i])
            for i in range(NUM_LAYERS)
            if (i + 1) % BLOCK == 0 or i == NUM_LAYERS - 1
        ]
        last = max(c.stages[q][-1] for q in range(c.P))
        consumers = [
            {owner[l] for l in range(i + 1, NUM_LAYERS)}
            | ({owner[last]} if i < last else set())
            for i, _ in blocks
        ]
        sent = (
            sum(len(cons - {q}) for (_, q), cons in zip(blocks, consumers) if q == r)
            * S
        )
        recv = (
            sum(1 for (_, q), cons in zip(blocks, consumers) if q != r and r in cons)
            * S
        )
        return sent + recv, recv + sent  # forward send+recv, backward mirror

    # ------------------------------------------------------------ overlap
    @staticmethod
    def overlap(comm, window, ib=False):
        """A collective of solo time comm in a compute window: (exposed time,
        compute penalty). ib: the collective crosses NVLink domains."""
        if comm <= 0:
            return 0.0, 0.0
        comp_sd, comm_sd = (
            (COMPUTE_SLOWDOWN_IB, COMM_SLOWDOWN_IB)
            if ib
            else (COMPUTE_SLOWDOWN, COMM_SLOWDOWN)
        )
        stretched = comm * comm_sd
        hidden = min(stretched, window)
        return max(0.0, stretched - window), hidden * (comp_sd - 1)

    def stage_period(self, c, layers):
        """Per microbatch on one GPU of the stage (or the whole model without
        PP): seconds by part."""
        mem_per, _ = self.memory_bound(c)
        out = {
            "compute": 0.0,
            "memory": 0.0,
            "exposed_fsdp": 0.0,
            "exposed_ep": 0.0,
            "exposed_tp": 0.0,
            "exposed_cp": 0.0,
            "penalty_fsdp": 0.0,
        }
        lcs = [self.layer_compute(c, i) for i in layers]
        for j, i in enumerate(layers):
            lc, cm = lcs[j], self.layer_comm(c, i)
            out["compute"] += lc["fwd"] + lc["bwd"]
            out["memory"] += mem_per
            if c.P == 1:  # ZeRO-3: gather + reduce-scatter every microbatch
                nxt = lcs[j + 1]["fwd"] if j + 1 < len(lcs) else lc["fwd"]
                ib = c.stage_gpus > NVLINK_DOMAIN
                e1, p1 = self.overlap(cm["fsdp_ag"], lc["fwd"] + mem_per / 4, ib)
                e2, p2 = self.overlap(
                    cm["fsdp_ag"] + cm["fsdp_rs"], lc["bwd"] + 3 * mem_per / 4, ib
                )
                out["exposed_fsdp"] += e1 + e2
                out["penalty_fsdp"] += p1 + p2
            out["exposed_ep"] += max(0.0, cm["ep"] - lc["ex_fwd"]) + 2 * max(
                0.0, cm["ep"] - lc["ex_bwd"] / 2
            )
            # TP and CP run in sequence with the compute: forward, remat forward
            # and backward each pay their collectives in full
            out["exposed_tp"] += 3 * cm["tp"]
            out["exposed_cp"] += 3 * cm["cp"]
        if layers[0] == 0:  # embedding (memory-bound) on the first stage
            tl, D = c.Tl, self.S["D"]
            out["memory"] += (4 + 10) * tl * D / bf.HBM_BW
        if layers[-1] == NUM_LAYERS - 1:  # LM head (bf16, measured), no remat
            t = (
                6
                * c.Tm
                * self.S["V"]
                * self.S["D"]
                / c.tp
                / (bf._interp(self.lm, c.Tm) * 1e12)
            )
            out["compute"] += t
        return out

    # ------------------------------------------------------------ step
    def step(self, c):
        """Forward + backward (or the pipeline), then the optimizer step."""
        res = self._fwd_bwd(c)
        opt = OPTIMIZER_S["fsdp" if c.P == 1 else "pp"]
        res["detail"]["optimizer"] = opt
        res["split"]["optimizer"] = opt
        res["step"] += opt
        return res

    def _fwd_bwd(self, c):
        res = {
            "config": f"{c.ctx} P{c.P} v{c.v} CP{c.cp} TP{c.tp} {c.transport if c.P > 1 else ''}",
            "M": c.M,
        }
        if c.P == 1:
            per = self.stage_period(c, list(range(NUM_LAYERS)))
            res["parts"] = {k: v * c.M for k, v in per.items()}
            res["step"] = sum(res["parts"].values())
            _, _, wfrac = self._memory_bound_parts(c)
            pa = res["parts"]
            res["detail"] = dict(
                pa,
                memory=pa["memory"] * (1 - wfrac),
                weight_prep=pa["memory"] * wfrac,
                penalty_hop=0.0,
                bubble=0.0,
                hops=0.0,
                zero1=0.0,
            )
            res["split"] = self._split(res["detail"])
            return res
        L = self.links
        ranks = []
        for r in range(c.P):
            t = hops = pen = 0.0
            parts = {}
            for vs in range(r, c.P * c.v, c.P):
                per = self.stage_period(c, c.vstages[vs])
                t += sum(per.values())
                for k, x in per.items():
                    parts[k] = parts.get(k, 0.0) + x
            mm, mem = parts["compute"], parts["memory"]
            # a hop slows whatever the rank computes meanwhile, by its mix of GEMMs
            # and memory-bound work
            hop_pen = (
                mm * (HOP_GEMM_SLOWDOWN - 1) + mem * (HOP_MEM_SLOWDOWN - 1)
            ) / max(mm + mem, 1e-9)
            for vs in range(r, c.P * c.v, c.P):
                # forward hop into the chunk and the backward hop (its gradient)
                # out of it sit on the critical path
                t_in = (
                    L.time("sendrecv", self.hop_bytes(c, vs), 2, "ib")
                    if vs > 0
                    else 0.0
                )
                t_out = (
                    L.time("sendrecv", self.hop_bytes(c, vs + 1), 2, "ib")
                    if vs + 1 < c.P * c.v
                    else 0.0
                )
                hops += t_in + t_out
                pen += (t_in + t_out) * hop_pen
            if c.transport == "direct":
                f, b = self.direct_side_bytes(c, r)
                pen += (f + b) / (IB_GBPS * 1e9 * IB_P2P_EFF) * hop_pen
            ranks.append(
                {
                    "rank": r,
                    "compute+mem+comm": t,
                    "hops": hops,
                    "hop_penalty": pen,
                    "period": t + hops + pen,
                    "parts": parts,
                }
            )
        worst = max(ranks, key=lambda p: p["period"])
        # ZeRO-1: the stage's FSDP gather (first forward) and reduce-scatter (last
        # backward), once per step, plus the once-per-step memory-bound copies
        _, once_mem = self.memory_bound(c)
        layers = c.stages[0]
        ag = sum(self.layer_comm(c, i)["fsdp_ag"] for i in layers)
        rs = sum(self.layer_comm(c, i)["fsdp_rs"] for i in layers)
        res["period"] = worst
        bubble = (c.M + (c.P - 1) / c.v) / c.M
        res["bubble_factor"] = bubble
        res["step"] = c.M * bubble * worst["period"] + ag + rs + once_mem
        res["parts"] = {
            "pipeline": c.M * bubble * worst["period"],
            "zero1_fsdp": ag + rs,
            "once_mem": once_mem,
        }
        # the step by part (the slowest rank, M microbatches): bubble = idle time
        # while the pipeline fills and drains, (P - 1) / v periods; hops = the
        # critical-path transfers of the M steady-state microbatches; zero1 = the
        # stage's gather and reduce-scatter, once per step
        wp, M = worst["parts"], c.M
        res["detail"] = {k: M * x for k, x in wp.items()}
        _, _, wfrac = self._memory_bound_parts(c)
        res["detail"].update(
            memory=M * wp["memory"] * (1 - wfrac),
            weight_prep=M * wp["memory"] * wfrac + once_mem,
            penalty_hop=M * worst["hop_penalty"],
            bubble=(bubble - 1) * M * worst["period"],
            hops=M * worst["hops"],
            zero1=ag + rs,
        )
        res["split"] = self._split(res["detail"])
        return res

    @staticmethod
    def _split(d):
        """The detail summed into the ranking table's columns."""
        return {
            "compute": d["compute"] + d["memory"] + d["weight_prep"],
            "exposed": d["exposed_fsdp"]
            + d["exposed_ep"]
            + d["exposed_tp"]
            + d["exposed_cp"],
            "penalty": d["penalty_fsdp"] + d["penalty_hop"],
            "pipeline": d["bubble"] + d["hops"] + d["zero1"],
        }


if __name__ == "__main__":
    m = StepModel()
    for ctx in ("8K", "32K"):
        for P in (1, 8, 16):
            for v, tr in (
                [(1, None)]
                if P == 1
                else [(1, "chain"), (1, "direct"), (2, "cache"), (4, "cache")]
            ):
                for cp, tp in bf.SPLITS:
                    r = m.step(Config(ctx, P, cp, tp, v, tr))
                    extra = (
                        f" bubble x{r['bubble_factor']:.2f} hops {r['period']['hops']*1e3:.0f} ms"
                        f" pen {r['period']['hop_penalty']*1e3:.0f} ms"
                        if P > 1
                        else ""
                    )
                    print(
                        f"{r['config']:30s} M={r['M']:3d} step {r['step']:6.1f} s{extra}"
                    )
