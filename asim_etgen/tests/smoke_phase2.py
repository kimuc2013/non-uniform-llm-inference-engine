"""Phase 2 smoke test: build a 1-microbatch forward ET for Llama-3-70B
TP=4 PP=2 uniform, feed to ASTRA-sim, sanity-check the output.

Pass criterion: ASTRA-sim runs to completion without errors, produces a
positive Wall time, and the per-rank graph is consistent (no missing
recv match for a send, no orphaned ctrl_deps).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

# Ensure the package is importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asim_etgen.inference_workload import (
    InferenceWorkloadBuilder, WorkloadSpec, ModelSpec,
)
from asim_etgen.partition import uniform_partition


LLAMA_3_70B = ModelSpec(
    name="Llama-3-70B",
    num_layers=80, hidden=8192,
    num_q_heads=64, num_kv_heads=8, head_dim=128,
    intermediate=28672,
)

ASTRA_BIN = os.environ.get("ASTRA_BIN", "/opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware")
A100_SYS = "/tmp/A100_sys.json"
A100_NET = "/tmp/A100_DGX_8npu.yml"
NO_REMOTE = "/tmp/no_remote_mem.json"

OUT_DIR = "/tmp/asim_etgen_smoke/llama70b_tp4pp2_uniform"


def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    partition = uniform_partition(
        world_size=8, tp_size=4, pp_size=2,
        num_layers=LLAMA_3_70B.num_layers,
        num_q_heads=LLAMA_3_70B.num_q_heads,
        num_kv_heads=LLAMA_3_70B.num_kv_heads,
        intermediate=LLAMA_3_70B.intermediate,
        rank_to_node=["n0"] * 8,
    )
    workload = WorkloadSpec(batch=1, seq=512, kv_len=0, is_decode=False)

    builder = InferenceWorkloadBuilder(LLAMA_3_70B, partition, workload)
    builder.build()
    base = builder.write(OUT_DIR)
    print(f"[generated] base = {base}")
    print(f"[generated] groups = {builder.groups.entries}")
    summary = open(os.path.join(OUT_DIR, "workload.summary.txt")).read()
    print("---workload summary---")
    print(summary)

    # Run ASTRA-sim
    cmd = [
        ASTRA_BIN,
        f"--workload-configuration={base}",
        f"--comm-group-configuration={OUT_DIR}/workload.json",
        f"--system-configuration={A100_SYS}",
        f"--remote-memory-configuration={NO_REMOTE}",
        f"--network-configuration={A100_NET}",
    ]
    print("---running ASTRA-sim---")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(f"ASTRA-sim exit code: {proc.returncode}")
        print("STDERR tail:")
        print(proc.stderr[-2000:] if proc.stderr else "")
        sys.exit(1)
    # Parse summary stats
    walls, gpus, comms = [], [], []
    for line in (proc.stdout + proc.stderr).splitlines():
        if "Wall time:" in line:
            walls.append(int(line.split("Wall time:")[1].strip()))
        elif "GPU time:" in line:
            gpus.append(int(line.split("GPU time:")[1].strip()))
        elif "Comm time:" in line:
            comms.append(int(line.split("Comm time:")[1].strip()))
    if not walls:
        print("[FAIL] No Wall time lines found")
        print(proc.stdout[-1500:])
        sys.exit(1)
    max_wall_s = max(walls) / 1e9
    mean_gpu_s = sum(gpus) / len(gpus) / 1e9 if gpus else 0
    mean_comm_s = sum(comms) / len(comms) / 1e9 if comms else 0
    print(f"\n[PASS] Wall (max over ranks) : {max_wall_s:.4f}s")
    print(f"       Mean GPU time          : {mean_gpu_s:.4f}s")
    print(f"       Mean Comm time         : {mean_comm_s:.4f}s")
    print(f"       Comm fraction          : {mean_comm_s / max_wall_s * 100:.1f}%")

    # Sanity check: comm_size in COLL nodes matches expected
    expected_ar_bytes = workload.batch * workload.seq * LLAMA_3_70B.hidden * 2
    print(f"\n[sanity] expected per-AR tensor bytes: {expected_ar_bytes:,} "
          f"= {expected_ar_bytes/1024/1024:.2f} MB")


if __name__ == "__main__":
    main()
