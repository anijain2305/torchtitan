# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The step-time model against real 512-GB200 Kimi K3 runs.

Each run below matches the model's assumptions: 512 GB200s, 16.78M tokens per
step, FP32 master weights, gradients and optimizer state, bf16 dense GEMMs,
full remat, DistMoE with forced-balanced routing, CUDA graphs. Runs in eager
mode, with learned routing, bf16 master weights or on other clusters are left
out. Per run:
  - measured step: tokens per GPU per step / clean tokens/s/GPU;
  - measured compute: the fake-process-group (FakePG) time of forward +
    backward + optimizer (no communication), where recorded;
    - model step and model compute (compute, memory-bound work, weight
    preparation and the optimizer; for PP, the slowest rank's);
    - the rest (step - compute): communication, overlap penalties and the
    pipeline, measured (step - FakePG) and modelled.

Usage:
  python scripts/memory_study/kimi_k3_model_vs_runs.py
"""

import blog_figures as bf
import kimi_k3_step_time as st

TOKENS_PER_GPU = bf._data()["global_tokens"] / st.NUM_GPUS

# name, model config kwargs, clean tokens/s/GPU, FakePG tokens/s/GPU, FakePG note
RUNS = [
    (
        "V1.51-CG-AR-MX: PP 8, 1F1B, 4K, MXFP8",
        dict(ctx="4K", P=8, v=1, transport="chain"),
        1009.5,
        1499.53,
        "GB300, slowest stage",
    ),
    (
        "V1.51-CG-AR: PP 8, 1F1B, 4K, bf16",
        dict(ctx="4K", P=8, v=1, transport="chain", expert_precision="bf16"),
        871.0,
        1320.42,
        "GB300, one stage",
    ),
    (
        "V2.2-CG CP4 prefetch-MX: 8K x 2 rows, MXFP8",
        dict(ctx="8K", cp=4, rows=2),
        683.0,
        1097.20,
        "",
    ),
    (
        "V2.2-CG CP4 prefetch: 8K x 2 rows, bf16",
        dict(ctx="8K", cp=4, rows=2, expert_precision="bf16"),
        640.0,
        1016.82,
        "",
    ),
    (
        "V2.2-CG CP4-MX: 8K x 2 rows, MXFP8, no prefetch",
        dict(ctx="8K", cp=4, rows=2, prefetch=False),
        457.0,
        1093.64,
        "",
    ),
    (
        "V2.2-CG CP4: 8K x 2 rows, bf16, no prefetch",
        dict(ctx="8K", cp=4, rows=2, expert_precision="bf16", prefetch=False),
        433.0,
        1007.03,
        "",
    ),
    (
        "V2-CG CP8: 32K, bf16, no prefetch (KDA CP 8 ~ CP 4)",
        dict(ctx="32K", cp=8, expert_precision="bf16", prefetch=False),
        419.0,
        974.7,
        "GB300",
    ),
]


def model_run(m, kw):
    kw = dict(kw)
    ctx = kw.pop("ctx")
    r = m.step(st.Config(ctx, **kw))
    return r, r["split"]["compute"] + r["split"]["optimizer"]


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0] * len(xs)
    for k, i in enumerate(order):
        out[i] = k + 1
    return out


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    return 1 - 6 * sum((x - y) ** 2 for x, y in zip(ra, rb)) / (n * (n * n - 1))


def kendall(a, b):
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += 1 if (a[i] - a[j]) * (b[i] - b[j]) > 0 else -1
    return s / (n * (n - 1) / 2)


def main() -> int:
    m = st.StepModel()
    rows = []
    for name, kw, tps, fake_tps, note in RUNS:
        r, comp = model_run(m, kw)
        rows.append(
            dict(
                name=name,
                meas=TOKENS_PER_GPU / tps,
                fake=TOKENS_PER_GPU / fake_tps,
                note=note,
                model=r["step"],
                comp=comp,
                split=r["split"],
                M=r["M"],
            )
        )
    print(
        f"{'run':52s} {'M':>3s} | {'meas':>6s} {'model':>6s} {'err':>5s} | "
        f"{'FakePG':>6s} {'m.comp':>6s} {'err':>5s} | {'rest':>5s} {'m.rest':>6s} | model: exposed penalty pipeline"
    )
    for x in rows:
        s = x["split"]
        print(
            f"{x['name']:52s} {x['M']:3d} | {x['meas']:6.1f} {x['model']:6.1f} {x['model'] / x['meas'] - 1:+5.0%} | "
            f"{x['fake']:6.1f} {x['comp']:6.1f} {x['comp'] / x['fake'] - 1:+5.0%} | "
            f"{x['meas'] - x['fake']:5.1f} {x['model'] - x['comp']:6.1f} | "
            f"{s['exposed']:5.1f} {s['penalty']:5.1f} {s['pipeline']:5.1f}  {x['note']}"
        )
    meas, model = [x["meas"] for x in rows], [x["model"] for x in rows]
    fake, comp = [x["fake"] for x in rows], [x["comp"] for x in rows]
    print(
        f"\nStep ranking:    Spearman {spearman(meas, model):+.2f}, Kendall {kendall(meas, model):+.2f}"
    )
    print(
        f"Compute ranking: Spearman {spearman(fake, comp):+.2f}, Kendall {kendall(fake, comp):+.2f}"
    )
    print("\nMeasured order -> model rank:")
    mr = ranks(model)
    for k, i in enumerate(sorted(range(len(rows)), key=lambda i: meas[i])):
        print(
            f"  {k + 1}. {rows[i]['name']:52s} measured {meas[i]:5.1f} s, model rank {mr[i]} ({model[i]:.1f} s)"
        )
    print("\nPairs (second / first):")
    for a, b, what in (
        (0, 2, "FSDP CP4 / PP 8, MXFP8"),
        (1, 3, "FSDP CP4 / PP 8, bf16"),
        (0, 1, "bf16 / MXFP8, PP 8"),
        (2, 3, "bf16 / MXFP8, FSDP CP4"),
        (2, 4, "no prefetch / prefetch, MXFP8"),
        (3, 5, "no prefetch / prefetch, bf16"),
    ):
        print(
            f"  {what:32s} measured {meas[b] / meas[a]:.2f}x, model {model[b] / model[a]:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
