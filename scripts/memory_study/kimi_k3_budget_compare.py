# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which Kimi K3 configs fit on 512 GPUs at 192 GB (GB200) vs 288 GB (GB300)?

Re-checks the memory study's configs against both budgets (the per-GPU byte
counts are the study's, hardware-independent), and adds the FSDP-free option
the post lacks: ZeRO-1 / ZeRO-2 with EP and TP and no PP.

ZeRO without PP, per GPU:
  - master weights + optimizer state: 8 B/param over all 512 GPUs (as the post);
  - compute weights held all step: routed experts MXFP8 for the GPU's 1/EP of
    them (torchtitan has no expert TP), dense bf16 = TP-sharded / TP + TP-whole;
  - gradients: ZeRO-1 keeps them unsharded (4 B for the params above), ZeRO-2
    reduce-scatters them every microbatch (4 B/param over 512 GPUs);
  - activations as FSDP + EP (one microbatch in flight), routing metadata for
    the EP degree; temporaries = the chunked-loss loop (no FSDP buffers).
"""

import blog_figures as bf

BUDGETS = {"GB200": 192e9, "GB300": 288e9}
GB = 1e9
# From the model's sharding configs with TP 2 (see the TP-split analysis):
ROUTED = 2_722_740_830_208
DENSE_TP_SHARDED = 50.99e9
DENSE_TP_WHOLE = 5.75e9
ALL = ROUTED + DENSE_TP_SHARDED + DENSE_TP_WHOLE
EPS = (64, 128)  # EP degrees dividing both 896 experts and 512 GPUs
SPLITS = [(1, 1), (2, 1), (4, 1), (1, 2), (1, 4), (1, 8), (2, 2), (2, 4)]


def zero_nopp(ctx: str, zero: int, ep: int, cp: int, tp: int) -> dict:
    d = bf._data()
    S = d["symbols"]
    T = d["contexts"][ctx]["seq_len"]
    tl = T / (cp * tp)
    mo = 8 * ALL / 512
    expert_w = bf.MXFP8_WEIGHT_BYTES * ROUTED / ep
    dense_n = DENSE_TP_SHARDED / tp + DENSE_TP_WHOLE
    dense_w = 2 * dense_n
    grads = 4 * (ROUTED / ep + dense_n) if zero == 1 else 4 * ALL / 512
    f = bf.fsdp_ep_parts(ctx, cp, tp) if (cp, tp) in [(1, 1), (2, 1), (4, 1), (1, 2), (1, 4)] else None
    if f is None:  # splits the study never tabulated: scale activations by the token split
        f1 = bf.fsdp_ep_parts(ctx, 1, 1)
        f = dict(f1, acts=f1["acts"] / (cp * tp), meta=f1["meta"] / (cp * tp))
    meta_ep = bf.MOE_LAYERS * bf.routing_metadata(ep, bf.NUM_EXPERTS / ep, tl)
    acts = f["acts"] - f["meta"] + meta_ep if "meta" in f else f["acts"]
    lm = S["V"] * S["D"]
    loop = 6 * lm / tp + tl / d["chunked_loss"] * (442e3 + 4 * S["V"] / tp) + 4 * tl * S["D"]
    total = mo + expert_w + dense_w + grads + acts + loop
    return {"mo": mo, "expert_w": expert_w, "dense_w": dense_w, "grads": grads, "acts": acts, "temp": loop,
            "total": total}


def fits(total: float) -> str:
    return " / ".join(("fits" if total <= b else "over") for b in BUDGETS.values())


def main() -> None:
    print("== Existing fit map (FSDP + EP 64; PP 8 / 16 ZeRO-1 with EP per stage), fits at 192 / 288 GB")
    for ctx in ("8K", "32K"):
        counts = {k: 0 for k in BUDGETS}
        n = 0
        print(f"-- {ctx}: " + "  ".join(f"CP{c}TP{t}" for c, t in bf.FIT_SPLITS))
        fams = [("FSDP 512, EP 64", lambda c, t: bf.fsdp_ep_parts(ctx, c, t))]
        fams += [(f"PP {P} v{v}", lambda c, t, P=P, v=v: bf.pp_parts(ctx, P, 1, c, t, v)) for P in (8, 16) for v in (1, 2, 4)]
        for name, fn in fams:
            cells = []
            for c, t in bf.FIT_SPLITS:
                tot = fn(c, t)["total"]
                n += 1
                for k, b in BUDGETS.items():
                    counts[k] += tot <= b
                mark = "+" if tot <= 192e9 else ("o" if tot <= 288e9 else "x")
                cells.append(f"{tot / GB:6.0f}{mark}")
            print(f"   {name:16s} " + " ".join(cells))
        print(f"   fit: GB200 {counts['GB200']}/{n}, GB300 {counts['GB300']}/{n}   (+ fits 192, o fits only 288, x neither)")

    print("\n== ZeRO without PP (FSDP-free), totals in GB, fits at 192 / 288")
    for ctx in ("8K", "32K"):
        for zero in (1, 2):
            for ep in EPS:
                row = []
                for c, t in SPLITS:
                    r = zero_nopp(ctx, zero, ep, c, t)
                    mark = "+" if r["total"] <= 192e9 else ("o" if r["total"] <= 288e9 else "x")
                    row.append(f"CP{c}TP{t} {r['total'] / GB:5.0f}{mark}")
                print(f"   {ctx} ZeRO-{zero} EP {ep:3d}: " + "  ".join(row))
    r = zero_nopp("8K", 1, 128, 1, 8)
    print("   breakdown 8K ZeRO-1 EP128 TP8: " + ", ".join(f"{k} {v / GB:.0f}" for k, v in r.items()))
    r = zero_nopp("8K", 2, 128, 1, 8)
    print("   breakdown 8K ZeRO-2 EP128 TP8: " + ", ".join(f"{k} {v / GB:.0f}" for k, v in r.items()))

    print("\n== Nugget checks")
    z = {row["label"]: row["total"] for row in bf.zero_parts("8K")}
    for label, tot in z.items():
        print(f"   8K {label:24s} {tot / GB:8.0f} GB  {fits(tot)}")
    for ctx in ("32K",):
        for c, t in [(1, 1), (2, 1), (1, 2), (4, 1), (1, 4)]:
            tot = bf.fsdp_ep_parts(ctx, c, t)["total"]
            print(f"   {ctx} FSDP + EP 64, CP {c}, TP {t}: {tot / GB:5.0f} GB  {fits(tot)}")


if __name__ == "__main__":
    main()
