# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Greedy memory / recompute-time Pareto frontier over Kimi K3's remat regions.

Start from save_regions=["*"] and, each round, add to recompute_regions the
region that frees the most activation memory per second of extra recompute.
Every candidate policy is measured, not estimated (recomputing a region can
make its producer keep its output, so region bytes do not add up): one
kimi_k3_remat_regions.py run per policy on the 8-layer model, several runs at a
time on the free GPUs.

Scaled to the full model at 8K with FSDP 512 + EP 64 (4 microbatches per step):
  memory = what layer 0 + 68 x a KDA + MoE layer + 24 x an MLA + MoE layer hold
           from forward to backward (allocator growth across the block, mean over
           the 8-layer model's layers of each kind), per microbatch in flight;
  time   = 4 x the recomputed regions' forward times over those 93 layers,
           measured under save_all.

Usage:
  python scripts/memory_study/kimi_k3_remat_pareto.py --gpus 0 2 3 \\
    --out scripts/memory_study/bench/remat_pareto_8k.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KDA_LAYERS, MLA_LAYERS = (1, 2, 4, 5, 6), (3, 7)  # of the 8-layer model
NUM_KDA_MOE, NUM_MLA_MOE = 68, 24  # Kimi K3: layer 0 is KDA + dense FFN
MICROBATCHES = 4  # 16M tokens / 512 GPUs / 8K
GIB = 2**30


def run_policy(recompute: list[str], gpu: int, port: int, workdir: str, policy: str = "save_all") -> dict:
    out = os.path.join(workdir, f"p_{port}.json")
    env = dict(
        os.environ,
        KIMI_K3_INVENTORY_POLICY=policy,
        KIMI_K3_INVENTORY_RECOMPUTE=",".join(recompute),
        KIMI_K3_INVENTORY_OUT=out,
        NGPU="512",
        NCCL_DEBUG="WARN",
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONPATH=f".:scripts/memory_study:{os.environ.get('DIST_MOE_PATH', '/home/anijain/local/dist_moe')}",
    )
    cmd = [
        "torchrun", "--standalone", "--nproc-per-node=1", f"--master-port={port}",
        "-m", "torchtitan.train", "--module", "kimi_k3_remat_regions",
        "--config", "remat_regions_8k", "--comm-backend", "fake",
    ]
    log = open(os.path.join(workdir, f"p_{port}.log"), "w")
    rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800).returncode
    if rc != 0 or not os.path.exists(out):
        raise RuntimeError(f"policy {recompute} failed (rc={rc}), see {log.name}")
    d = json.load(open(out))
    os.remove(out)
    return d


def full_model_bytes(d: dict) -> float:
    g = {int(k.split(".")[1]): v for k, v in d["allocated_growth_per_layer"].items()}
    kda = sum(g[i] for i in KDA_LAYERS) / len(KDA_LAYERS)
    mla = sum(g[i] for i in MLA_LAYERS) / len(MLA_LAYERS)
    return g[0] + NUM_KDA_MOE * kda + NUM_MLA_MOE * mla


def region_times(inventory: dict) -> dict[str, float]:
    """Seconds of forward per step for each region, over the full model."""
    per = {}
    for layer, regions in inventory["layers"].items():
        i = int(layer.split(".")[1])
        weight = 1 if i == 0 else NUM_KDA_MOE / len(KDA_LAYERS) if i in KDA_LAYERS else NUM_MLA_MOE / len(MLA_LAYERS)
        for r in regions:
            per[r["name"]] = per.get(r["name"], 0.0) + weight * r["fwd_ms"] / 1e3 * MICROBATCHES
    return per


def block_time(inventory: dict) -> float:
    """Seconds of whole-block forward per step over the full model (= full remat's replay)."""
    total = 0.0
    for layer, ms in inventory["block_fwd_ms"].items():
        i = int(layer.split(".")[1])
        weight = 1 if i == 0 else NUM_KDA_MOE / len(KDA_LAYERS) if i in KDA_LAYERS else NUM_MLA_MOE / len(MLA_LAYERS)
        total += weight * ms / 1e3 * MICROBATCHES
    return total


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gpus", type=int, nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-gain-gib", type=float, default=0.25, help="stop when no candidate frees this much")
    p.add_argument("--resume", help="a previous --out: continue its frontier instead of starting over")
    a = p.parse_args()
    workdir = tempfile.mkdtemp(prefix="remat_pareto_")
    ports = iter(range(29900, 30900))

    def measure_all(policies: list[list[str]]) -> list[dict]:
        with ThreadPoolExecutor(len(a.gpus)) as pool:
            futures = [
                pool.submit(run_policy, pol, a.gpus[k % len(a.gpus)], next(ports), workdir)
                for k, pol in enumerate(policies)
            ]
            return [f.result() for f in futures]

    base, full = measure_all([[]]) + [run_policy([], a.gpus[0], next(ports), workdir, policy="full")]
    times = region_times(base)
    always = {r["name"] for L in full["layers"].values() for r in L if r["saved"]}
    saved_bytes = {}
    for L in base["layers"].values():
        for r in L:
            saved_bytes[r["name"]] = saved_bytes.get(r["name"], 0) + r["bytes"]
    top_level = {r["name"] for L in base["layers"].values() for r in L if r["parent"] is None}
    # every region the policy controls: one that saves nothing under save_all can
    # still hold memory once its consumer is recomputed (it must keep its output)
    candidates = sorted(n for n in saved_bytes if n not in always)
    print("candidates:", candidates, flush=True)

    recompute, mem, t = [], full_model_bytes(base), 0.0
    frontier = [{"recompute_regions": [], "bytes": mem, "recompute_s": 0.0}]
    trials = []
    if a.resume:
        prev = json.load(open(a.resume))
        frontier, trials = prev["frontier"], prev["trials"]
        recompute, mem, t = list(frontier[-1]["recompute_regions"]), frontier[-1]["bytes"], frontier[-1]["recompute_s"]
    while True:
        left = [c for c in candidates if c not in recompute]
        if not left:
            break
        results = measure_all([recompute + [c] for c in left])
        best = None
        for c, d in zip(left, results):
            m = full_model_bytes(d)
            gain, cost = mem - m, times[c]
            trials.append({"base": list(recompute), "add": c, "bytes": m, "gain": gain, "cost_s": cost})
            print(f"  +{c:45s} frees {gain / GIB:7.2f} GiB for {cost:6.3f} s", flush=True)
            if gain > a.min_gain_gib * GIB and (best is None or gain / cost > best[1] / best[2]):
                best = (c, gain, cost, m)
        if best is None:
            break
        recompute.append(best[0])
        mem, t = best[3], t + best[2]
        frontier.append({"recompute_regions": list(recompute), "bytes": mem, "recompute_s": t})
        print(f"round {len(recompute)}: recompute {best[0]} -> {mem / GIB:.1f} GiB, +{t:.2f} s/step", flush=True)

    out = {
        "setup": "Kimi K3, 8K, FSDP 512 + EP 64, DistMoE, fake PG; 8-layer model scaled to 93 layers; "
        "bytes = activations held per microbatch from forward to backward; recompute_s = extra forward per step",
        "region_forward_s_per_step": times,
        "always_saved": sorted(always),
        "save_all_bytes": full_model_bytes(base),
        "full_remat_bytes": full_model_bytes(full),
        "block_forward_s_per_step": block_time(base),
        # ops outside every region (norms, attention-residual mixing) replay even under save_all
        "save_all_recompute_s": block_time(base) - sum(times[n] for n in top_level),
        "frontier": frontier,
        "trials": trials,
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"Wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
