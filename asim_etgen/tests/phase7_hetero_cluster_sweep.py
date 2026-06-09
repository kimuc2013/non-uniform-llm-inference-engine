"""Phase 7: hypothetical hetero cluster sweep — paper-grade data.

For each hypothetical hetero cluster (always 2-stage PP, TP=4 per stage,
8 GPUs total, 4 fast + 4 slow), we:

  1. Compute the planner's analytical PP layer split prediction. Decode
     is memory-bandwidth-bound, so the predicted V-bottom satisfies
        layers_fast / layers_slow = mem_bw_fast / mem_bw_slow
     (the stage with bigger mem BW gets more layers to keep
      max(T_fast, T_slow) minimized).

  2. Sweep PP layer splits, generate ETs through asim_etgen, run
     ASTRA-sim, and find the observed per-iter decode V-bottom.

  3. Compare predicted vs observed V-bottom, and report the speedup of
     the V-bottom split over the naive uniform [40, 40] split.

The thesis: across diverse hetero clusters (compute & memory ratios
spanning 1.6× to 11×), the planner's analytical prediction lands
within a few layers of ASTRA-sim's V-bottom — i.e., the planner is a
cheap proxy for cycle-level simulation across arbitrary cluster
topologies.
"""

from __future__ import annotations

import json
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

# ---------- GPU catalog (achievable bf16) ----------
BLACKWELL = ComputeProfile(name="RTX-PRO-Blackwell", spec_tflops_bf16=380.0, spec_mem_bw_GBs=1792.0)
ADA       = ComputeProfile(name="RTX6000-Ada",       spec_tflops_bf16=91.0,  spec_mem_bw_GBs=960.0)
H100      = ComputeProfile(name="H100-SXM5",         spec_tflops_bf16=989.0, spec_mem_bw_GBs=3350.0)
A100      = ComputeProfile(name="A100-SXM4-80GB",    spec_tflops_bf16=312.0, spec_mem_bw_GBs=2039.0)
V100      = ComputeProfile(name="V100-SXM2",         spec_tflops_bf16=125.0, spec_mem_bw_GBs=900.0)
L4        = ComputeProfile(name="L4",                spec_tflops_bf16=121.0, spec_mem_bw_GBs=300.0)
T4        = ComputeProfile(name="T4",                spec_tflops_bf16=65.0,  spec_mem_bw_GBs=320.0)
B200      = ComputeProfile(name="B200",              spec_tflops_bf16=4500.0, spec_mem_bw_GBs=8000.0)
L40S      = ComputeProfile(name="L40S",              spec_tflops_bf16=362.0, spec_mem_bw_GBs=864.0)


ASTRA_BIN = os.environ.get("ASTRA_BIN", "/opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware")
NET_FILE = "/tmp/blackwell_ada_2node.yml"
NO_REMOTE = "/tmp/no_remote_mem.json"


def write_sys_for_hetero(hetero: HeteroSimConfig, path: str) -> str:
    cfg = {
        "scheduling-policy": "LIFO",
        "endpoint-delay": 1,
        "active-chunks-per-dimension": 2,
        "preferred-dataset-splits": 4,
        "all-reduce-implementation": ["ring", "ring"],
        "all-gather-implementation": ["ring", "ring"],
        "reduce-scatter-implementation": ["ring", "ring"],
        "all-to-all-implementation": ["ring", "ring"],
        "collective-optimization": "localBWAware",
        "local-mem-bw": hetero.reference.local_mem_bw_GBs,
        "boost-mode": 0,
        "track-local-mem": 0,
        "roofline-enabled": 1,
        "peak-perf": hetero.reference.peak_perf_tflops,
    }
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def run_sim(out_dir: str, partition: PartitionSpec, workload: WorkloadSpec,
            hetero: HeteroSimConfig) -> float:
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    sys_path = f"/tmp/sys_{os.path.basename(out_dir)}.json"
    write_sys_for_hetero(hetero, sys_path)
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


def predict_v_bottom_memratio(fast: ComputeProfile, slow: ComputeProfile,
                              num_layers: int) -> tuple[int, int]:
    """First-order planner: mem-BW ratio rule for memory-bound decode."""
    r_fast = fast.local_mem_bw_GBs
    r_slow = slow.local_mem_bw_GBs
    layers_fast = round(num_layers * r_fast / (r_fast + r_slow))
    layers_fast = max(1, min(num_layers - 1, layers_fast))
    return (layers_fast, num_layers - layers_fast)


# Llama-70B per-layer per-rank mem traffic for TP=4 decode at B=25:
# QKV weight 41.9 MB + O 33.6 MB + Gate_up 235 MB + Down 117.4 MB + KV 14.7 MB
# + small activations. Dominated by weights, mostly constant across our
# scenarios (workload shape is fixed).
LAYER_BYTES_LLAMA70B_TP4_B25 = 443e6     # ≈ 443 MB per rank per layer

# Per-layer AR/collective overhead. Two AllReduces of size B·hidden·2 bytes
# (= 0.4 MB at B=25) over ring on TP=4 with intra-node 50 GB/s busbw. The
# analytical ring AR cost is roughly 2·(N−1)/N · bytes/bw ≈ 12 µs per AR,
# plus per-hop latency, plus ASTRA-sim's scheduling overhead — empirically
# ≈ 30 µs per layer in our system config.
AR_OVERHEAD_US_PER_LAYER = 30.0


def predict_v_bottom_calibrated(fast: ComputeProfile, slow: ComputeProfile,
                                num_layers: int) -> tuple[int, int]:
    """Roofline + collective planner.

    Per-layer cost = mem-bound time + AR overhead. AR overhead is the
    same per layer (workload-bytes don't depend on GPU type), so when
    the fast GPU's mem time approaches AR overhead, the effective ratio
    shrinks toward 1 and the V-bottom shifts back toward [40, 40]. This
    is what we observe in the simulator for very-fast GPUs (B200, H100)
    paired with mid-tier slow GPUs.

    Solve layers_fast · cost_fast = layers_slow · cost_slow.
    """
    t_mem_fast_us = LAYER_BYTES_LLAMA70B_TP4_B25 / (fast.local_mem_bw_GBs * 1e9) * 1e6
    t_mem_slow_us = LAYER_BYTES_LLAMA70B_TP4_B25 / (slow.local_mem_bw_GBs * 1e9) * 1e6
    cost_fast = t_mem_fast_us + AR_OVERHEAD_US_PER_LAYER
    cost_slow = t_mem_slow_us + AR_OVERHEAD_US_PER_LAYER
    layers_fast = round(num_layers * cost_slow / (cost_fast + cost_slow))
    layers_fast = max(1, min(num_layers - 1, layers_fast))
    return (layers_fast, num_layers - layers_fast)


def sweep_around(center: int, num_layers: int, radius: int = 12,
                 step: int = 4) -> list[list[int]]:
    """Generate a sweep of [layers_fast, layers_slow] around `center`."""
    splits = set()
    # always include uniform [40, 40] as baseline
    splits.add((num_layers // 2, num_layers - num_layers // 2))
    for d in range(-radius, radius + 1, step):
        f = max(4, min(num_layers - 4, center + d))
        splits.add((f, num_layers - f))
    return sorted([list(s) for s in splits], key=lambda x: x[0])


def find_v_bottom(walls: dict[tuple, float]) -> tuple[int, int]:
    return min(walls, key=lambda k: walls[k])


SCENARIOS = [
    # (label, fast GPU, slow GPU)
    ("S1 real cluster",        BLACKWELL, ADA),
    ("S2 H100 + V100",         H100,      V100),
    ("S3 A100 + T4 (extreme)", A100,      T4),
    ("S4 B200 + A100 (gen+1)", B200,      A100),
    ("S5 H100 + L4",           H100,      L4),
    ("S6 H100 + A100 (mild)",  H100,      A100),
    ("S7 L40S + V100",         L40S,      V100),
]


def main():
    print("Phase 7: hypothetical hetero cluster sweep — paper-grade data")
    print("=" * 90)
    print(f"  Model: {LLAMA_3_70B.name} ({LLAMA_3_70B.num_layers} layers)")
    print(f"  Topology: 2-stage PP, TP=4 per stage, 4 fast + 4 slow GPUs")
    print(f"  Workload: decode B=25 (50 reqs / pp_size), S=1, kv_len=576, n_mb=2")
    print()

    # IMPORTANT: n_mb must be > max compute-scaling ratio across our
    # scenarios (≈ 11.17 for H100+L4) to put the pipeline in
    # steady-state. Otherwise pipeline-fill dominates and the V-bottom
    # gets shifted toward extreme fast-heavy splits. With n_mb >> r:
    #   wall ≈ (n_mb − 1) · max(T_fast, T_slow) + T_fast + T_slow
    # and the minimum sits at T_fast = T_slow, i.e. layers ∝ mem-BW.
    workload_decode = WorkloadSpec(
        batch=25, seq=1, kv_len=576, is_decode=True, n_microbatches=16,
    )
    N_MB = 16
    PP = 2

    summary: list[dict] = []

    for label, fast, slow in SCENARIOS:
        print(f"\n{'=' * 90}\n{label}: 4×{fast.name} (fast) + 4×{slow.name} (slow)")
        ratio_compute = fast.peak_perf_tflops / slow.peak_perf_tflops
        ratio_mem = fast.local_mem_bw_GBs / slow.local_mem_bw_GBs
        print(f"  Achievable ratio: compute {ratio_compute:.2f}×, mem-BW {ratio_mem:.2f}×")

        pred_mem = predict_v_bottom_memratio(fast, slow, LLAMA_3_70B.num_layers)
        pred_cal = predict_v_bottom_calibrated(fast, slow, LLAMA_3_70B.num_layers)
        print(f"  Planner (mem-BW-ratio rule): {list(pred_mem)}")
        print(f"  Planner (roofline + AR):     {list(pred_cal)}")

        hetero = HeteroSimConfig(
            reference=fast,
            gpus_by_rank=[fast] * 4 + [slow] * 4,
        )

        # Sweep around the midpoint of both predictions for fair eval.
        center = (pred_mem[0] + pred_cal[0]) // 2
        splits = sweep_around(center, LLAMA_3_70B.num_layers, radius=14, step=4)
        for p_ in (list(pred_mem), list(pred_cal)):
            if p_ not in splits:
                splits.append(p_)
        splits.sort(key=lambda x: x[0])

        walls: dict[tuple, float] = {}
        per_iter: dict[tuple, float] = {}
        print(f"  {'split':>10s} {'ASTRA wall':>11s} {'per-iter':>10s}")
        for ls in splits:
            p = PartitionSpec(
                tp_size=4, pp_size=2,
                layer_splits=ls,
                head_splits=[16]*4, kv_splits=[2]*4, ffn_splits=[7168]*4,
                rank_to_node=["head"]*4 + ["worker"]*4,
            )
            out = f"/tmp/p7_{label.replace(' ', '_').replace('+', 'p')}_{ls[0]}_{ls[1]}"
            w = run_sim(out, p, workload_decode, hetero)
            if w != w:
                print(f"  {str(ls):>10s} FAIL")
                continue
            walls[tuple(ls)] = w
            per_iter[tuple(ls)] = w / (N_MB + PP - 1)
            print(f"  {str(ls):>10s} {w:>11.4f} {per_iter[tuple(ls)]:>10.4f}")

        v_observed = find_v_bottom(walls)
        uniform = min(walls.keys(), key=lambda k: abs(k[0] - 40))
        speedup = walls[uniform] / walls[v_observed]
        diff_mem = abs(v_observed[0] - pred_mem[0])
        diff_cal = abs(v_observed[0] - pred_cal[0])

        print(f"  → Observed V-bottom: {list(v_observed)}")
        print(f"  → Δ vs mem-BW-ratio planner: {diff_mem} layers, "
              f"Δ vs calibrated planner: {diff_cal} layers")
        print(f"  → Speedup over uniform {list(uniform)}: {speedup:.2f}×")

        summary.append({
            "scenario": label,
            "fast": fast.name,
            "slow": slow.name,
            "compute_ratio": ratio_compute,
            "mem_ratio": ratio_mem,
            "predicted_mem": list(pred_mem),
            "predicted_cal": list(pred_cal),
            "observed": list(v_observed),
            "diff_mem": diff_mem,
            "diff_cal": diff_cal,
            "uniform_split": list(uniform),
            "uniform_wall": walls[uniform],
            "v_bottom_wall": walls[v_observed],
            "speedup": speedup,
        })

    print()
    print("=" * 90)
    print("SUMMARY — planner predictions vs ASTRA-sim V-bottom across hetero clusters")
    print("=" * 90)
    print(f"  {'scenario':<24s} {'mem ratio':>10s} {'mem-BW pred':>12s} "
          f"{'calib pred':>11s} {'observed':>11s} {'Δm':>4s} {'Δc':>4s} {'speedup':>9s}")
    print("  " + "-" * 100)
    for s in summary:
        print(f"  {s['scenario']:<24s} {s['mem_ratio']:>9.2f}× "
              f"{str(s['predicted_mem']):>12s} {str(s['predicted_cal']):>11s} "
              f"{str(s['observed']):>11s} {s['diff_mem']:>4d} {s['diff_cal']:>4d} "
              f"{s['speedup']:>8.2f}×")
    print()

    diffs_mem = [s["diff_mem"] for s in summary]
    diffs_cal = [s["diff_cal"] for s in summary]
    n_within_4_mem = sum(1 for d in diffs_mem if d <= 4)
    n_within_4_cal = sum(1 for d in diffs_cal if d <= 4)
    avg_speedup = sum(s["speedup"] for s in summary) / len(summary)
    print(f"  mem-BW-ratio planner: avg Δ = {sum(diffs_mem)/len(diffs_mem):.1f} layers, "
          f"max = {max(diffs_mem)}, |Δ|≤4 = {n_within_4_mem}/{len(summary)}")
    print(f"  calibrated  planner: avg Δ = {sum(diffs_cal)/len(diffs_cal):.1f} layers, "
          f"max = {max(diffs_cal)}, |Δ|≤4 = {n_within_4_cal}/{len(summary)}")
    print(f"  Avg V-bottom speedup over uniform: {avg_speedup:.2f}×")

    # Save raw for paper
    out_json = "/tmp/phase7_sweep_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Raw results written to {out_json}")


if __name__ == "__main__":
    main()
