"""Phase 5 smoke: hetero GPU cluster + non-uniform PP layer split.

Simulates 4×Blackwell stage 0 + 4×Ada stage 1 (matching our real cluster
topology) and compares uniform [40,40] PP layer split vs hetero [48,32]:
  expect hetero to be FASTER because Ada is slower per-layer.

Then a stronger hetero: H100 (fast) + V100 (slow). Sweep PP layer splits
from balanced toward H100-heavy. Expect a V-shaped curve.
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
from asim_etgen.partition import PartitionSpec
from asim_etgen.hetero_compute import HeteroSimConfig
from asim_etgen.runtime_model import ComputeProfile


LLAMA_3_70B = ModelSpec(
    name="Llama-3-70B", num_layers=80, hidden=8192,
    num_q_heads=64, num_kv_heads=8, head_dim=128, intermediate=28672,
)


# Achievable factors derived empirically from our Blackwell+Ada calibration.
BLACKWELL = ComputeProfile(
    name="RTX-PRO-Blackwell",
    spec_tflops_bf16=380.0, spec_mem_bw_GBs=1792.0,
)
ADA = ComputeProfile(
    name="RTX6000-Ada",
    spec_tflops_bf16=91.0, spec_mem_bw_GBs=960.0,
)
H100 = ComputeProfile(
    name="H100-SXM5",
    spec_tflops_bf16=989.0, spec_mem_bw_GBs=3350.0,
)
V100 = ComputeProfile(
    name="V100-SXM2",
    spec_tflops_bf16=125.0, spec_mem_bw_GBs=900.0,
)
A100 = ComputeProfile(
    name="A100-SXM4-80GB",
    spec_tflops_bf16=312.0, spec_mem_bw_GBs=2039.0,
)


ASTRA_BIN = os.environ.get("ASTRA_BIN", "/opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware")
NET_FILE = "/tmp/A100_DGX_8npu.yml"
NO_REMOTE = "/tmp/no_remote_mem.json"


def write_sys(hetero: HeteroSimConfig, path: str = "/tmp/hetero_sys.json"):
    """Write the ASTRA-sim system json using the reference GPU's numbers."""
    sys_cfg = hetero.system_config()
    cfg = {
        "scheduling-policy": "LIFO",
        "endpoint-delay": 1,
        "active-chunks-per-dimension": 2,
        "preferred-dataset-splits": 4,
        "all-reduce-implementation": ["ring"],
        "all-gather-implementation": ["ring"],
        "reduce-scatter-implementation": ["ring"],
        "all-to-all-implementation": ["ring"],
        "collective-optimization": "localBWAware",
        "local-mem-bw": sys_cfg["local-mem-bw"],
        "boost-mode": 0,
        "track-local-mem": 0,
        "roofline-enabled": 1,
        "peak-perf": sys_cfg["peak-perf"],
    }
    import json
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def run(out_dir: str, partition: PartitionSpec, workload: WorkloadSpec,
        hetero: HeteroSimConfig) -> float:
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    sys_path = write_sys(hetero)
    builder = InferenceWorkloadBuilder(LLAMA_3_70B, partition, workload, hetero=hetero)
    builder.build()
    base = builder.write(out_dir)
    cmd = [
        ASTRA_BIN,
        f"--workload-configuration={base}",
        f"--comm-group-configuration={out_dir}/workload.json",
        f"--system-configuration={sys_path}",
        f"--remote-memory-configuration={NO_REMOTE}",
        f"--network-configuration={NET_FILE}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    walls = [int(l.split("Wall time:")[1].strip())
             for l in (proc.stdout + proc.stderr).splitlines()
             if "Wall time:" in l]
    if not walls:
        return float("nan")
    return max(walls) / 1e9


def main():
    print("Phase 5: hetero cluster + non-uniform PP layer split")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Test 1: our real cluster topology (4 Blackwell + 4 Ada)
    # ---------------------------------------------------------------
    print("\n[1] 4xBlackwell + 4xAda — replicates our real cluster")
    hetero_real = HeteroSimConfig(
        reference=BLACKWELL,
        gpus_by_rank=[BLACKWELL] * 4 + [ADA] * 4,
    )
    workload = WorkloadSpec(batch=1, seq=512, kv_len=0, is_decode=False,
                            n_microbatches=4)
    splits_to_try = [[40, 40], [44, 36], [48, 32], [52, 28], [56, 24]]
    results = []
    for ls in splits_to_try:
        p = PartitionSpec(
            tp_size=4, pp_size=2,
            layer_splits=ls,
            head_splits=[16]*4, kv_splits=[2]*4, ffn_splits=[7168]*4,
            rank_to_node=["head"]*4 + ["worker"]*4,
        )
        w = run(f"/tmp/asim_etgen_smoke/p5_real_{ls[0]}_{ls[1]}", p, workload, hetero_real)
        results.append((ls, w))
        print(f"   PP layer split {ls}: wall={w:.4f}s")
    best_split, best_wall = min(results, key=lambda r: r[1])
    print(f"   V-bottom (ASTRA-sim): {best_split} at {best_wall:.4f}s")

    # ---------------------------------------------------------------
    # Test 2: 4xH100 + 4xV100 — bigger FLOPS ratio
    # ---------------------------------------------------------------
    print("\n[2] 4xH100 + 4xV100 — extreme hetero (FLOPS ratio ~7.9×)")
    hetero_h100v100 = HeteroSimConfig(
        reference=H100,
        gpus_by_rank=[H100] * 4 + [V100] * 4,
    )
    results_h = []
    for ls in splits_to_try:
        p = PartitionSpec(
            tp_size=4, pp_size=2,
            layer_splits=ls,
            head_splits=[16]*4, kv_splits=[2]*4, ffn_splits=[7168]*4,
            rank_to_node=["n0"]*4 + ["n1"]*4,
        )
        w = run(f"/tmp/asim_etgen_smoke/p5_h100v100_{ls[0]}_{ls[1]}", p, workload, hetero_h100v100)
        results_h.append((ls, w))
        print(f"   PP layer split {ls}: wall={w:.4f}s")
    best_split_h, best_wall_h = min(results_h, key=lambda r: r[1])
    print(f"   V-bottom: {best_split_h} at {best_wall_h:.4f}s")
    # Expect V-bottom even more biased toward H100 (e.g., [56, 24] or [60, 20])

    # ---------------------------------------------------------------
    # Test 3: homogeneous A100 — V-bottom should be at [40, 40]
    # ---------------------------------------------------------------
    print("\n[3] 8xA100 — homogeneous control, expect uniform = best")
    hetero_homo = HeteroSimConfig(
        reference=A100, gpus_by_rank=[A100] * 8,
    )
    results_homo = []
    for ls in [[36, 44], [40, 40], [44, 36]]:
        p = PartitionSpec(
            tp_size=4, pp_size=2,
            layer_splits=ls,
            head_splits=[16]*4, kv_splits=[2]*4, ffn_splits=[7168]*4,
            rank_to_node=["n0"]*8,
        )
        w = run(f"/tmp/asim_etgen_smoke/p5_homo_{ls[0]}_{ls[1]}", p, workload, hetero_homo)
        results_homo.append((ls, w))
        print(f"   PP layer split {ls}: wall={w:.4f}s")
    best_split_homo = min(results_homo, key=lambda r: r[1])[0]
    if best_split_homo == [40, 40]:
        print("   [as expected] uniform [40,40] is V-bottom on homog A100")
    else:
        print(f"   [unexpected]  V-bottom = {best_split_homo} on homog A100")


if __name__ == "__main__":
    main()
