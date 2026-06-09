"""Phase 3 smoke test: 1F1B pipeline overlap.

Compares two ETs over the SAME total work (n_microbatches forward passes
through TP=4 PP=2):

  A) Sequential reference: build n_microbatches=1 with B' = n_mb * B
     (i.e., a single big batch that goes through the pipe once)
  B) 1F1B test:           build n_microbatches=n_mb with B
     (n_mb forward passes interleaved through the pipe)

If 1F1B is encoded correctly, (B) should have LOWER wall than (A) by
roughly the pipeline-overlap factor (close to 2× for pp_size=2 at
sufficient microbatches).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asim_etgen.inference_workload import (
    InferenceWorkloadBuilder, WorkloadSpec, ModelSpec,
)
from asim_etgen.partition import uniform_partition


LLAMA_3_70B = ModelSpec(
    name="Llama-3-70B", num_layers=80, hidden=8192,
    num_q_heads=64, num_kv_heads=8, head_dim=128, intermediate=28672,
)

ASTRA_BIN = os.environ.get("ASTRA_BIN", "/opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware")
A100_SYS = "/tmp/A100_sys.json"
A100_NET = "/tmp/A100_DGX_8npu.yml"
NO_REMOTE = "/tmp/no_remote_mem.json"


def run_one(out_dir: str, n_microbatches: int, batch_per_mb: int, seq: int) -> float:
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    p = uniform_partition(world_size=8, tp_size=4, pp_size=2,
                          num_layers=80, num_q_heads=64, num_kv_heads=8,
                          intermediate=28672, rank_to_node=["n0"]*8)
    w = WorkloadSpec(batch=batch_per_mb, seq=seq, kv_len=0,
                     is_decode=False, n_microbatches=n_microbatches)
    builder = InferenceWorkloadBuilder(LLAMA_3_70B, p, w)
    builder.build()
    base = builder.write(out_dir)
    cmd = [
        ASTRA_BIN,
        f"--workload-configuration={base}",
        f"--comm-group-configuration={out_dir}/workload.json",
        f"--system-configuration={A100_SYS}",
        f"--remote-memory-configuration={NO_REMOTE}",
        f"--network-configuration={A100_NET}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    walls = [int(l.split("Wall time:")[1].strip())
             for l in (proc.stdout + proc.stderr).splitlines()
             if "Wall time:" in l]
    if not walls:
        print(f"  FAIL: no walls. stderr tail:\n{proc.stderr[-500:]}")
        return float("nan")
    return max(walls) / 1e9


def main():
    N_MB = 4
    B = 1
    S = 512
    print(f"Comparing N_MB={N_MB}, B={B}, S={S} (Llama-3-70B TP=4 PP=2 on 8xA100 DGX)")
    print()

    # Reference A: 1 microbatch, batch = N_MB * B  (sequential pipeline, single pass with bigger batch)
    # This represents the same total work but with no overlap opportunity.
    print(f"[A] sequential: n_microbatches=1, batch={N_MB * B} (equivalent total work)")
    a_wall = run_one("/tmp/asim_etgen_smoke/phase3_seq", 1, N_MB * B, S)
    print(f"    wall: {a_wall:.4f}s")

    # Test B: 1F1B, n_microbatches = N_MB, batch_per_mb = B
    print(f"[B] 1F1B:       n_microbatches={N_MB}, batch={B}")
    b_wall = run_one("/tmp/asim_etgen_smoke/phase3_1f1b", N_MB, B, S)
    print(f"    wall: {b_wall:.4f}s")

    print()
    print(f"ratio (B / A): {b_wall / a_wall:.3f}")
    # In an ideal world with pp_size=2 and many microbatches, B should be
    # ~half of A (the overlap eliminates one of the two stage-time
    # contributions). With N_MB=4 we expect ratio < 1.0.
    if b_wall < a_wall:
        print(f"[PASS] 1F1B exposed pipeline overlap (B is {(1 - b_wall/a_wall)*100:.1f}% faster)")
    else:
        print(f"[CHECK] B not faster than A — 1F1B encoding likely not exposing overlap")


if __name__ == "__main__":
    main()
