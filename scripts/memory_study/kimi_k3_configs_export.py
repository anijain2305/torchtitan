# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export the memory page's parallelism configs and per-GPU memory to JSON.

The memory model lives in kimi_k3_params.html's JavaScript (cpConfigs), so this
loads that page in a headless browser and evaluates it for every context, CP
and TP degree, writing kimi_k3_configs.json for the roofline page's ranking.

Usage (needs playwright):
  python scripts/memory_study/kimi_k3_configs_export.py
"""

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent

JS = """() => {
  const out = [];
  for (const ctx of ["8K", "32K"])
    for (const cp of [1, 2, 4, 8])
      for (const tp of [1, 2, 4]) {
        if (cp * tp > 8) continue;
        for (const c of cpConfigs({ ctx, cp, tp })) {
          const P = c.P, g = NUM_GPUS / P;
          out.push({ ctx, seq_len: DATA.contexts[ctx].seq_len, cp, tp, label: c.label, pp: P,
                     zero: c.label.includes("ZeRO-3") ? 3 : (P > 1 ? 1 : null),
                     ep: P > 1 ? g : EP_DEGREE, gpus_per_stage: g, microbatches: c.M,
                     bytes: c.bytes, fits: c.bytes <= GB200_BYTES, valid: c.M >= Math.max(1, P) });
        }
      }
  return out;
}"""


def main() -> int:
    with sync_playwright() as p:
        page = p.chromium.launch().new_page()
        page.goto((HERE / "kimi_k3_params.html").as_uri())
        configs = page.evaluate(JS)
    out = HERE / "kimi_k3_configs.json"
    out.write_text(json.dumps({"gpu_bytes": 192e9, "configs": configs}, indent=1))
    ok = [c for c in configs if c["fits"] and c["valid"]]
    print(f"Wrote {out}: {len(configs)} configs, {len(ok)} fit and are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
