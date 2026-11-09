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
  - overlap: a collective and the compute it hides behind slow each other
    down by the measured factors for as long as they run together; whatever
    of the collective is left when the compute ends runs alone, exposed:
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
IB_GBPS = 100.0  # GB200 NIC line rate per GPU per direction (2 x 400 Gb/s CX7)
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
# A group with one GPU per NVL domain (an expert shard's FSDP replicas) uses
# one NIC, and NCCL only 1 (all-gather) to 8 (reduce-scatter) SMs: measured with
# one GPU per host (8 hosts, 1 GB), the overlapped compute slows down about
# 1.10x and the collective about 1.2x.
COMPUTE_SLOWDOWN_ONE_NIC = 1.10
COMM_SLOWDOWN_ONE_NIC = 1.2
# A one-way transfer (PP hop) across InfiniBand (rank i -> rank i + N / 2, 8
# and 32 GPUs alike): NCCL runs one thread block per side (the NIC moves the
# data), and the compute slows down about the same on the sender and the
# receiver: GEMMs 1.10x, memory-bound kernels 1.01x; we charge 1.10x on all of
# it. The hop itself slows down 1.0-1.2x meanwhile, which we leave out.
HOP_COMPUTE_SLOWDOWN = 1.10


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
    """ctx: sequence length ("4K", "8K", "32K"). rows: sequences packed into one
    microbatch (attention per sequence). expert_precision: the routed experts'
    GEMMs, "mxfp8" or "bf16". prefetch: FSDP prefetches the next layer's
    all-gather in backward (without PP); off, under CUDA graphs, a layer's
    backward, reduce-scatter and the next all-gather run in sequence."""

    def __init__(
        self,
        ctx,
        P=1,
        cp=1,
        tp=1,
        v=1,
        transport=None,
        *,
        rows=1,
        expert_precision="mxfp8",
        prefetch=True,
    ):
        d = bf._data()
        self.ctx, self.P, self.cp, self.tp = ctx, P, cp, tp
        self.rows, self.expert_precision, self.prefetch = (
            rows,
            expert_precision,
            prefetch,
        )
        self.T = bf.CTX_TOKENS[ctx]
        self.ep = NUM_GPUS // P if P > 1 else 64
        self.Tm = (
            rows * self.T // cp
        )  # GEMM rows per microbatch (after the SP all-gather)
        self.Tl = rows * self.T // (cp * tp)  # local tokens per microbatch
        self.stage_gpus = NUM_GPUS // P
        self.dp = NUM_GPUS // (P * cp * tp)
        self.M = int(
            d["global_tokens"] // (rows * self.T) // self.dp
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


def simulate_pipeline(P, v, M, fwd, bwd, hop, jit_fwd_recv=False):
    """Time of one step of 1F1B (v = 1) or interleaved 1F1B on P ranks, by event
    simulation in PyTorch's op order: virtual stage q runs on rank q % P; rank r
    runs (v - 1) P + 2 (P - 1 - r) warmup forwards when interleaved (P - 1 - r
    otherwise), then alternates forward / backward. fwd[q] / bwd[q]: one
    microbatch's forward / backward time on virtual stage q; hop[q]: a hop
    across boundary q -> q + 1 (either direction). An op starts when its rank
    is free and its input has arrived. Interleaved 1F1B posts its receives
    early, so a hop moves as soon as it is sent; plain 1F1B posts a forward
    receive only when the rank is ready for it (jit_fwd_recv), and the transfer
    starts then."""
    Q = P * v

    def nth(k, back):
        mb = (k // Q) * P + k % P if v > 1 else k
        c = (k // P) % v
        return mb, (v - 1 - c if back else c)

    order = []
    for r in range(P):
        n = M * v
        warm = min(n, (v - 1) * P + 2 * (P - 1 - r)) if v > 1 else min(n, P - 1 - r)
        ops = [(0,) + nth(k, False) for k in range(warm)]
        f, b = warm, 0
        while b < n:
            if f < n:
                ops.append((0,) + nth(f, False))
                f += 1
            ops.append((1,) + nth(b, True))
            b += 1
        order.append(ops)
    end = {}
    free, idx = [0.0] * P, [0] * P
    remaining = P * M * v * 2
    while remaining:
        progressed = False
        for r in range(P):
            while idx[r] < len(order[r]):
                back, mb, c = order[r][idx[r]]
                q = c * P + r
                if not back:
                    dep, h = ((0, mb, q - 1), hop[q - 1]) if q > 0 else (None, 0.0)
                else:
                    dep, h = (
                        ((0, mb, q), 0.0) if q == Q - 1 else ((1, mb, q + 1), hop[q])
                    )
                if dep is not None and dep not in end:
                    break
                if dep is None:
                    ready = 0.0
                elif not back and jit_fwd_recv:
                    ready = max(free[r], end[dep]) + h
                else:
                    ready = end[dep] + h
                start = max(free[r], ready)
                free[r] = end[(back, mb, q)] = start + (bwd[q] if back else fwd[q])
                idx[r] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("pipeline schedule deadlock")
    return max(free)


class StepModel:
    def __init__(self, overlap_tp_cp: bool = False):
        """overlap_tp_cp: a what-if in which TP's collectives hide behind the
        dense GEMMs and CP's behind the rest of the layer, with the NVLink
        overlap penalties, instead of running exposed as in TorchTitan today."""
        self.overlap_tp_cp = overlap_tp_cp
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
        g = self.shapes[bf.DENSE_SHAPES]["data"][f"m{c.Tm}_tp{c.tp}"][scope]["gemms"]
        fwd_d = sum(x["fwd"] for x in g) * 1e-3
        bwd_d = sum(x["fwd"] + x["dgrad"] + x["wgrad"] for x in g) * 1e-3
        if (
            i == 0
        ):  # dense FFN instead of MoE (about 3 D F parameters), at the layer's rate
            eff = self.shapes[bf.DENSE_SHAPES]["data"][f"m{c.Tm}_tp{c.tp}"][
                "kda_layer"
            ]["eff"]
            ffn = 2 * c.Tm * 3 * S["D"] * S["F"] / c.tp / (eff * bf.DENSE_PEAK)
            fwd_d, bwd_d = fwd_d + ffn, bwd_d + 3 * ffn
        ex_f = ex_b = 0.0
        if i > 0:
            peak = bf.MXFP8_PEAK if c.expert_precision == "mxfp8" else bf.BF16_PEAK
            t = (
                8
                * c.Tl
                * S["k"]
                * self.n_expert
                / (bf.distmoe_eff(c.Tl, c.ep, c.expert_precision) * peak)
            )
            ex_f, ex_b = t / 4, 3 * t / 4
        if mla:
            fa = self.fa4[str(c.T)][str(96 // (c.cp * c.tp))]
            F = (
                c.rows
                * c.T
                * c.T
                * (96 // (c.cp * c.tp))
                * (S["Dn"] + S["Dr"] + S["Dv"])
            )  # one forward
            at_f = F / (fa["fwd_tflops"] * 1e12)
            at_b = 3.5 * F / (fa["fwd_bwd_tflops"] * 1e12)  # remat forward + backward
        else:
            cpf = {"fwd_bwd": 1.0, "fwd": 1.0}
            if c.cp > 1:
                # measured at CP 2 and 4; a larger CP takes CP 4's slowdown
                r = {
                    x["seq_len"]: x
                    for x in bf._bench(f"kda_cp_GB300_cp{min(c.cp, 4)}.json")["results"]
                }[c.T]
                cpf = {k: r[f"cp_{k}_ms"] / r[f"local_{k}_ms"] for k in cpf}
            sp = 1.0 if c.tp == 1 else bf.KDA_TP_SPEEDUP[c.ctx][c.tp]
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
            c.ctx,
            cp=c.cp,
            tp=c.tp,
            ep=c.ep,
            P=c.P,
            split_weights=True,
            rows=c.rows,
            expert_mxfp8=c.expert_precision == "mxfp8",
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
        ag_e = rs_e = 0.0
        if i > 0 and edp > 1:
            shard = S["E"] * self.n_expert / c.ep
            ag_e = L.time("allgather", 2 * shard, edp, link)
            rs_e = L.time("reducescatter", 4 * shard, edp, link)
            ag += ag_e
            rs += rs_e
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
                    c.rows * c.cp * S["H"] / c.tp * S["K"] * S["K"] * 4,
                    c.cp,
                    "nvlink",
                )
        return {
            "fsdp_ag": ag,
            "fsdp_rs": rs,
            "fsdp_ag_experts": ag_e,
            "fsdp_rs_experts": rs_e,
            "ep": ep,
            "tp": tp,
            "cp": cpt,
        }

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
    def overlap(comm, window, ib=False, slowdowns=None):
        """A collective of solo time comm overlapping compute of solo time
        window: (exposed time, compute penalty, collective penalty). While they
        run together, the compute runs comp_sd x slower and the collective
        comm_sd x slower, so they share T = min(comm_sd x comm, comp_sd x
        window) of wall time; whichever is left runs alone at full speed after
        that. Exposed is what would not fit even at full speed; the collective
        penalty is the extra exposed time its slowdown causes. ib: the
        collective crosses NVLink domains."""
        if comm <= 0:
            return 0.0, 0.0, 0.0
        comp_sd, comm_sd = slowdowns or (
            (COMPUTE_SLOWDOWN_IB, COMM_SLOWDOWN_IB)
            if ib
            else (COMPUTE_SLOWDOWN, COMM_SLOWDOWN)
        )
        overlapped_compute = min(
            comm * comm_sd / comp_sd, window
        )  # compute run alongside it
        compute_penalty = overlapped_compute * (comp_sd - 1)
        tail = max(
            0.0, comm - overlapped_compute * comp_sd / comm_sd
        )  # the collective left, at full speed
        exposed = max(0.0, comm - window)
        return exposed, compute_penalty, tail - exposed

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
            "penalty_fsdp_comm": 0.0,
            "penalty_tpcp": 0.0,
        }
        lcs = [self.layer_compute(c, i) for i in layers]
        for j, i in enumerate(layers):
            lc, cm = lcs[j], self.layer_comm(c, i)
            out["compute"] += lc["fwd"] + lc["bwd"]
            out["memory"] += mem_per
            if c.P == 1:  # ZeRO-3: gather + reduce-scatter every microbatch
                nxt = lcs[j + 1]["fwd"] if j + 1 < len(lcs) else lc["fwd"]
                ib = c.stage_gpus > NVLINK_DOMAIN

                def mix(total, experts):
                    # the expert part runs on one NIC (its own, smaller
                    # slowdowns); weight both by their share of the time
                    if not ib or total <= 0:
                        return None
                    f = experts / total
                    return (
                        COMPUTE_SLOWDOWN_IB * (1 - f) + COMPUTE_SLOWDOWN_ONE_NIC * f,
                        COMM_SLOWDOWN_IB * (1 - f) + COMM_SLOWDOWN_ONE_NIC * f,
                    )

                e1, p1, q1 = self.overlap(
                    cm["fsdp_ag"],
                    lc["fwd"] + mem_per / 4,
                    ib,
                    mix(cm["fsdp_ag"], cm["fsdp_ag_experts"]),
                )
                if c.prefetch:
                    bwd_c = cm["fsdp_ag"] + cm["fsdp_rs"]
                    e2, p2, q2 = self.overlap(
                        bwd_c,
                        lc["bwd"] + 3 * mem_per / 4,
                        ib,
                        mix(bwd_c, cm["fsdp_ag_experts"] + cm["fsdp_rs_experts"]),
                    )
                else:
                    # the layer's all-gather is issued after the previous
                    # layer's reduce-scatter on the same communicator, and its
                    # backward waits for it: backward, reduce-scatter and
                    # all-gather run one after another
                    e2, p2, q2 = cm["fsdp_ag"] + cm["fsdp_rs"], 0.0, 0.0
                out["exposed_fsdp"] += e1 + e2
                out["penalty_fsdp"] += p1 + p2
                out["penalty_fsdp_comm"] += q1 + q2
            out["exposed_ep"] += max(0.0, cm["ep"] - lc["ex_fwd"]) + 2 * max(
                0.0, cm["ep"] - lc["ex_bwd"] / 2
            )
            # TP and CP run in sequence with the compute: forward, remat forward
            # and backward each pay their collectives in full
            if not self.overlap_tp_cp:
                out["exposed_tp"] += 3 * cm["tp"]
                out["exposed_cp"] += 3 * cm["cp"]
            else:
                dense = lc["dense_fwd"] + lc["dense_bwd"]
                for key, comm, window in (
                    ("exposed_tp", cm["tp"], dense),
                    ("exposed_cp", cm["cp"], lc["fwd"] + lc["bwd"] - dense),
                ):
                    e, p, q = self.overlap(3 * comm, window)
                    out[key] += e
                    out["penalty_tpcp"] += p + q
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
        if c.rows > 1 or c.expert_precision != "mxfp8" or not c.prefetch:
            res[
                "config"
            ] += f" rows{c.rows} {c.expert_precision}{'' if c.prefetch else ' no-prefetch'}"
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
        Q = c.P * c.v
        hop_pen = 1 - 1 / HOP_COMPUTE_SLOWDOWN
        # hop across boundary q -> q + 1 (the transport sets its bytes)
        hop = [
            L.time("sendrecv", self.hop_bytes(c, q + 1), 2, "ib") for q in range(Q - 1)
        ]
        vstage = []  # per virtual stage: (seconds per microbatch, forward share, parts)
        for q in range(Q):
            lay = c.vstages[q]
            per = self.stage_period(c, lay)
            lcs = [self.layer_compute(c, i) for i in lay]
            fwd = sum(x["fwd"] for x in lcs)
            share = fwd / max(fwd + sum(x["bwd"] for x in lcs), 1e-12)
            vstage.append([sum(per.values()), share, per])
        ranks = []
        for r in range(c.P):
            qs = list(range(r, Q, c.P))
            t = sum(vstage[q][0] for q in qs)
            parts = {}
            for q in qs:
                for k, x in vstage[q][2].items():
                    parts[k] = parts.get(k, 0.0) + x
            # a hop slows whatever the rank computes meanwhile: over the hop's
            # time, the compute gets 1 / 1.10 of it done. Per chunk the rank
            # takes part in four hops: forward in and out, backward in and out
            hops_r = sum(
                (hop[q - 1] if q > 0 else 0.0) + (hop[q] if q < Q - 1 else 0.0)
                for q in qs
            )
            pen = 2 * hops_r * hop_pen
            if c.transport == "direct":
                f, b = self.direct_side_bytes(c, r)
                pen += (f + b) / (IB_GBPS * 1e9 * IB_P2P_EFF) * hop_pen
            for q in qs:  # the penalty lengthens the rank's stages in proportion
                vstage[q][0] *= 1 + pen / max(t, 1e-12)
            ranks.append(
                {
                    "rank": r,
                    "compute+mem+comm": t,
                    "hop_penalty": pen,
                    "period": t + pen,
                    "parts": parts,
                }
            )
        worst = max(ranks, key=lambda p: p["period"])
        fwd_t = [x[0] * x[1] for x in vstage]
        bwd_t = [x[0] * (1 - x[1]) for x in vstage]
        # receives posted early in both schedules (a good implementation)
        t_hops = simulate_pipeline(c.P, c.v, c.M, fwd_t, bwd_t, hop)
        t_none = simulate_pipeline(c.P, c.v, c.M, fwd_t, bwd_t, [0.0] * (Q - 1))
        # ZeRO-1: the stage's FSDP gather (first forward) and reduce-scatter (last
        # backward), once per step, plus the once-per-step memory-bound copies
        _, once_mem = self.memory_bound(c)
        layers = c.stages[0]
        ag = sum(self.layer_comm(c, i)["fsdp_ag"] for i in layers)
        rs = sum(self.layer_comm(c, i)["fsdp_rs"] for i in layers)
        res["period"] = worst
        res["step"] = t_hops + ag + rs + once_mem
        res["parts"] = {"pipeline": t_hops, "zero1_fsdp": ag + rs, "once_mem": once_mem}
        # the step by part: the slowest rank's work over M microbatches; bubble =
        # what the schedule adds without hops (fill, drain, uneven stages); hops =
        # what the hops add on top; zero1 = the stage's gather and reduce-scatter
        wp, M = worst["parts"], c.M
        res["detail"] = {k: M * x for k, x in wp.items()}
        _, _, wfrac = self._memory_bound_parts(c)
        res["detail"].update(
            memory=M * wp["memory"] * (1 - wfrac),
            weight_prep=M * wp["memory"] * wfrac + once_mem,
            penalty_hop=M * worst["hop_penalty"],
            bubble=max(0.0, t_none - M * worst["period"]),
            hops=t_hops - t_none,
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
            "penalty": d["penalty_fsdp"]
            + d["penalty_fsdp_comm"]
            + d["penalty_hop"]
            + d["penalty_tpcp"],
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
