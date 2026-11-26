# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collect the fit sweep's full-depth runs into bench/fit_sweep/fd_runs.json.

Each full-depth job (fd_<ctx>_<split>_<policy>.log, written by
fit_sweep_worker.sh) becomes one entry: peak reserved memory (the last-step
'memory:' value torchtitan logs), throughput per step, the step time (tokens
per GPU per step / median tps of the steps after the first; CP runs stop after
two steps, so they have one timed step) and whether the run ran out of memory.

Usage:
  python scripts/memory_study/fit_sweep_collect.py /tmp/claude-anijain/sweep
"""

import json
import re
import statistics
import sys
from pathlib import Path

GLOBAL_TOKENS_PER_GPU = 16 * 1024 * 1024 // 512  # tokens per GPU per optimizer step


def collect(log_dir: Path) -> dict:
    runs = {}
    for log in sorted(log_dir.glob("fd_*.log")):
        text = log.read_text(errors="replace")
        mem = re.findall(r"memory: ([0-9.]+)GiB", text)
        tps = [int(t.replace(",", "")) for t in re.findall(r"tps: ([0-9,]+)", text)]
        timed = tps[1:] or tps
        runs[log.stem] = {
            "peak_reserved_gib": float(mem[-1]) if mem else None,
            "tps": tps,
            "step_s": GLOBAL_TOKENS_PER_GPU / statistics.median(timed) if timed else None,
            "oom": "OutOfMemoryError" in text,
            "completed": "Training completed" in text,
        }
    return runs


def main() -> None:
    log_dir = Path(sys.argv[1])
    out = Path(__file__).resolve().parent / "bench" / "fit_sweep" / "fd_runs.json"
    runs = collect(log_dir)
    out.write_text(json.dumps({"note": "Fit sweep full-depth runs (eager, fake PG on one GB300, FSDP 512 + "
                                       "EP 64); see fit_sweep_collect.py.", "runs": runs}, indent=1))
    for name, r in runs.items():
        print(f"{name:28s} peak {r['peak_reserved_gib']} GiB  step {r['step_s'] and round(r['step_s'], 2)} s"
              f"{'  OOM' if r['oom'] else ''}{'' if r['completed'] else '  (did not complete)'}")


if __name__ == "__main__":
    main()
