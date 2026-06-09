"""Phase 8: planner picks the best plan across TP/PP/Hybrid candidates.

Two entry points:
  - `python phase8_plan_selection.py`         (default: per-iteration matrix)
  - `python phase8_plan_selection.py --serving` (end-to-end serving under
        continuous batching: ServingWorkload of (N, in_len, out_len) →
        prefill_phase + decode_phase composition).



Phase 7 only swept PP layer split with PP=2 TP=4 fixed. That validates
the planner's *partition* output GIVEN a chosen (TP, PP). The actual
paper claim — that the planner picks the right STRATEGY (TP vs PP vs
hybrid + uniform vs non-uniform sharding) given an arbitrary hetero
cluster topology — needs cross-strategy validation.

For each cluster topology, we:

  1. Enumerate feasible candidate plans (varying TP/PP and uniform/
     non-uniform sharding subject to GQA + per-node placement).
  2. Score every plan analytically (per-rank roofline + AR overhead;
     stage wall = max-over-ranks; pipeline wall = (n_mb+pp−1) × max_stage).
  3. Run ASTRA-sim for every plan; observe the real V-bottom plan.
  4. Compare: did the analytical planner pick the simulator's best?

Topology families covered:
  T1. Stage-segregated 2-node:    4×fast on n0 + 4×slow on n1
      → PP=2 natural; cross-node TP penalized; non-uniform layer split wins.
  T2. Single-node mixed TP=8:     4×fast + 4×slow on one node
      → TP=8 PP=1 natural; non-uniform FFN shard wins.
  T3. Per-node mixed TP groups:   2×fast + 2×slow on each of 2 nodes
      → non-uniform TP shards within each stage become necessary.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List

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

H100 = ComputeProfile(name="H100-SXM5", spec_tflops_bf16=989.0, spec_mem_bw_GBs=3350.0)
A100 = ComputeProfile(name="A100-SXM4-80GB", spec_tflops_bf16=312.0, spec_mem_bw_GBs=2039.0)
B200 = ComputeProfile(name="B200", spec_tflops_bf16=4500.0, spec_mem_bw_GBs=8000.0)
L4 = ComputeProfile(name="L4", spec_tflops_bf16=121.0, spec_mem_bw_GBs=300.0)


ASTRA_BIN = os.environ.get("ASTRA_BIN", "/opt/astra-sim/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware")
NO_REMOTE = "/tmp/no_remote_mem.json"
NET_2NODE = "/tmp/2node_2x4_mixed.yml"
NET_1NODE = "/tmp/single_node_8npu.yml"


# ----------------------------------------------------------------------
# Topology & plan dataclasses
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ClusterTopology:
    """A specific hetero cluster: per-rank GPU types + node placement +
    network file."""
    name: str
    rank_to_gpu: tuple[ComputeProfile, ...]
    rank_to_node: tuple[str, ...]
    network_file: str
    is_single_node: bool = False
    n_dims: int = 2     # number of dims in the network YAML

    @property
    def num_ranks(self) -> int:
        return len(self.rank_to_gpu)


@dataclass(frozen=True)
class CandidatePlan:
    """A specific (TP, PP, sharding) plan. ranks are laid out as
    PP-major: stage 0 gets ranks 0..tp_size-1, stage 1 gets next, ..."""
    name: str
    tp_size: int
    pp_size: int
    layer_splits: tuple[int, ...]
    head_splits: tuple[int, ...]
    kv_splits: tuple[int, ...]
    ffn_splits: tuple[int, ...]
    description: str = ""


# ----------------------------------------------------------------------
# Plan enumeration: build feasible plans for a given topology
# ----------------------------------------------------------------------

def gqa_compatible_head_splits(num_q_heads: int, gqa_group: int,
                               tp_size: int) -> list[list[int]]:
    """Enumerate head_splits arrays that sum to num_q_heads, each entry a
    positive multiple of gqa_group. For TP=4 with 64 heads gqa=8 we get
    [16,16,16,16] (uniform) and several biased options like [24,24,8,8].
    For TP=8 only [8]*8 is feasible (every rank needs ≥ gqa_group)."""
    units = num_q_heads // gqa_group       # 64 / 8 = 8 units
    base = units // tp_size
    if base < 1:
        return []
    splits = [[base * gqa_group] * tp_size]  # uniform first
    # Biased: take 1 unit from each of 2 ranks, give to 2 other ranks
    if base >= 2 and tp_size >= 4:
        biased = [base + 1] * (tp_size // 2) + [base - 1] * (tp_size // 2)
        biased = [b * gqa_group for b in biased]
        if all(b > 0 for b in biased) and sum(biased) == num_q_heads:
            splits.append(biased)
    return splits


def ffn_splits_proportional(speeds: list[float], total_ffn: int) -> list[int]:
    """Round each rank's FFN shard ∝ its speed. Adjusts the last to sum
    exactly. Each shard ≥ 64 (a small floor)."""
    raw = [s / sum(speeds) * total_ffn for s in speeds]
    parts = [max(64, int(r)) for r in raw]
    drift = total_ffn - sum(parts)
    parts[-1] += drift
    return parts


def enumerate_plans(topo: ClusterTopology, model: ModelSpec) -> list[CandidatePlan]:
    """Return feasible plans for this topology. Generalized to arbitrary
    world_size (handles 2, 4, 8 GPUs)."""
    plans: list[CandidatePlan] = []
    W = topo.num_ranks
    L = model.num_layers
    NQ = model.num_q_heads
    NKV = model.num_kv_heads
    GQA = model.gqa_group
    FFN = model.intermediate

    nodes = topo.rank_to_node
    speeds = [g.local_mem_bw_GBs for g in topo.rank_to_gpu]

    # ---- helper: enumerate (TP, PP) factorizations of W ----
    factorizations = []
    for tp in range(1, W + 1):
        if W % tp == 0:
            pp = W // tp
            # GQA constraint: TP must divide num_q_heads with a multiple
            # of the GQA group (each rank gets at least one KV head).
            if NQ % tp == 0 and (NQ // tp) % max(1, GQA) == 0 and tp <= NKV * GQA:
                # Also ensure num_kv_heads // tp >= 1
                if NKV % tp == 0 and (NKV // tp) >= 1:
                    factorizations.append((tp, pp))
                elif tp <= NKV:
                    factorizations.append((tp, pp))

    # ---- per-(tp, pp) plan generation ----
    t_layer = lambda bw: 443e6 / (bw * 1e9) * 1e6 + 30.0

    for tp, pp in factorizations:
        # Skip if pp > L (degenerate)
        if pp > L:
            continue
        # ── uniform plan for this (tp, pp) ──
        head_unif = NQ // tp
        kv_unif = max(1, NKV // tp)
        ffn_unif = FFN // tp
        layers_unif = [L // pp + (1 if i < L % pp else 0) for i in range(pp)]

        plans.append(CandidatePlan(
            name=f"TP{tp}_PP{pp}_uniform",
            tp_size=tp, pp_size=pp,
            layer_splits=tuple(layers_unif),
            head_splits=tuple([head_unif] * tp),
            kv_splits=tuple([kv_unif] * tp),
            ffn_splits=tuple([ffn_unif] * tp),
            description=f"TP={tp} PP={pp} uniform",
        ))

        # ── non-uniform layer split for PP > 1, based on per-stage speed ──
        if pp > 1:
            stage_speeds = []
            for s in range(pp):
                stage_ranks = list(range(s * tp, (s + 1) * tp))
                stage_speeds.append(sum(speeds[r] for r in stage_ranks) / tp)
            stage_costs = [t_layer(sp) for sp in stage_speeds]
            inv = [1.0 / c for c in stage_costs]
            raw_layers = [L * x / sum(inv) for x in inv]
            ls = [max(1, round(x)) for x in raw_layers]
            ls[-1] += L - sum(ls)
            if all(1 <= x <= L - (pp - 1) for x in ls) and tuple(ls) != tuple(layers_unif):
                plans.append(CandidatePlan(
                    name=f"TP{tp}_PP{pp}_nonuniform_layer",
                    tp_size=tp, pp_size=pp,
                    layer_splits=tuple(ls),
                    head_splits=tuple([head_unif] * tp),
                    kv_splits=tuple([kv_unif] * tp),
                    ffn_splits=tuple([ffn_unif] * tp),
                    description=f"TP={tp} PP={pp} non-uniform layer {ls}",
                ))

        # ── non-uniform TP shards (FFN ∝ speed) for tp > 1 and mixed stage ──
        if tp > 1:
            stage0_ranks = list(range(0, tp))
            stage0_unique = len(set(topo.rank_to_gpu[r].name for r in stage0_ranks)) > 1
            if stage0_unique:
                ffn_arr = ffn_splits_proportional([speeds[r] for r in stage0_ranks], FFN)
                # Biased head shard (GQA-aware)
                order0 = sorted(range(tp), key=lambda i: -speeds[i])
                head_arr = [head_unif] * tp
                if tp >= 4 and speeds[order0[0]] > speeds[order0[-1]] * 1.5:
                    delta = max(GQA, 1) * 1
                    if head_unif - delta > 0:
                        head_arr[order0[0]] += delta
                        head_arr[order0[-1]] -= delta
                        if tp >= 4 and order0[1] != order0[-1]:
                            head_arr[order0[1]] += delta
                            head_arr[order0[-2]] -= delta
                # Validate sum and positivity
                if sum(head_arr) == NQ and all(h > 0 for h in head_arr):
                    plans.append(CandidatePlan(
                        name=f"TP{tp}_PP{pp}_nonuniform_TP",
                        tp_size=tp, pp_size=pp,
                        layer_splits=tuple(layers_unif),
                        head_splits=tuple(head_arr),
                        kv_splits=tuple(max(1, h // max(1, GQA)) for h in head_arr),
                        ffn_splits=tuple(ffn_arr),
                        description=f"TP={tp} PP={pp} non-uniform TP head={head_arr}",
                    ))

        # ── non-uniform FFN for TP=W case (single TP group) ──
        if tp == W and tp > 1:
            ffn_arr = ffn_splits_proportional(speeds, FFN)
            plans.append(CandidatePlan(
                name=f"TP{tp}_PP{pp}_nonuniform_FFN",
                tp_size=tp, pp_size=pp,
                layer_splits=tuple(layers_unif),
                head_splits=tuple([head_unif] * tp),
                kv_splits=tuple([kv_unif] * tp),
                ffn_splits=tuple(ffn_arr),
                description=f"TP={tp} PP={pp} non-uniform FFN ∝ speed",
            ))

    return plans


def _enumerate_plans_OLD_HARDCODED(topo: ClusterTopology, model: ModelSpec) -> list[CandidatePlan]:
    """[deprecated] Original world=8 hardcoded version. Kept for diffing."""
    plans: list[CandidatePlan] = []
    L = model.num_layers
    NQ = model.num_q_heads
    GQA = model.gqa_group
    FFN = model.intermediate

    nodes = topo.rank_to_node
    speeds = [g.local_mem_bw_GBs for g in topo.rank_to_gpu]

    # ------ PP=2 TP=4 ------
    if topo.num_ranks == 8:
        # uniform layer split
        plans.append(CandidatePlan(
            name="TP4_PP2_uniform",
            tp_size=4, pp_size=2,
            layer_splits=(L // 2, L - L // 2),
            head_splits=tuple([NQ // 4] * 4),
            kv_splits=tuple([model.num_kv_heads // 4] * 4),
            ffn_splits=tuple([FFN // 4] * 4),
            description="TP=4 PP=2 uniform — naive baseline",
        ))
        # non-uniform layer split — proportional to per-stage avg speed
        stage0_speed = sum(speeds[0:4]) / 4
        stage1_speed = sum(speeds[4:8]) / 4
        # Add an AR-overhead correction (Phase 7's calibrated planner).
        # t_layer = bytes/mem_bw + 30µs. bytes = 443 MB.
        t_layer = lambda bw: 443e6 / (bw * 1e9) * 1e6 + 30.0
        c0 = t_layer(stage0_speed); c1 = t_layer(stage1_speed)
        layers_0 = max(1, min(L - 1, round(L * c1 / (c0 + c1))))
        plans.append(CandidatePlan(
            name="TP4_PP2_nonuniform_layer",
            tp_size=4, pp_size=2,
            layer_splits=(layers_0, L - layers_0),
            head_splits=tuple([NQ // 4] * 4),
            kv_splits=tuple([model.num_kv_heads // 4] * 4),
            ffn_splits=tuple([FFN // 4] * 4),
            description=f"TP=4 PP=2 non-uniform layer [{layers_0}, {L - layers_0}]",
        ))
        # Mixed-stage TP — non-uniform TP shards within stage. Only meaningful
        # if a stage's 4 ranks have different speeds.
        stage0_unique = len(set(g.name for g in topo.rank_to_gpu[0:4])) > 1
        if stage0_unique:
            # Pick head split: nearest to GQA-respect for sum=64.
            # Biased shard: heavier on faster ranks of stage 0.
            order0 = sorted(range(4), key=lambda i: -speeds[i])
            head_arr = [16] * 4
            if speeds[order0[0]] > speeds[order0[-1]] * 1.5:
                head_arr[order0[0]] = 24; head_arr[order0[-1]] = 8
                head_arr[order0[1]] = 24; head_arr[order0[2]] = 8
            ffn_arr = ffn_splits_proportional(speeds[0:4], FFN)
            plans.append(CandidatePlan(
                name="TP4_PP2_nonuniform_TP",
                tp_size=4, pp_size=2,
                layer_splits=(L // 2, L - L // 2),
                head_splits=tuple(head_arr),
                kv_splits=tuple([h // GQA for h in head_arr]),
                ffn_splits=tuple(ffn_arr),
                description=f"TP=4 PP=2 non-uniform TP shards stage0 (head {head_arr})",
            ))

        # TP=8 PP=1 — uniform
        plans.append(CandidatePlan(
            name="TP8_PP1_uniform",
            tp_size=8, pp_size=1,
            layer_splits=(L,),
            head_splits=tuple([NQ // 8] * 8),
            kv_splits=tuple([model.num_kv_heads // 8] * 8),
            ffn_splits=tuple([FFN // 8] * 8),
            description="TP=8 PP=1 uniform — single TP group",
        ))
        # TP=8 PP=1 — non-uniform FFN proportional to speed
        ffn_arr = ffn_splits_proportional(speeds, FFN)
        plans.append(CandidatePlan(
            name="TP8_PP1_nonuniform_FFN",
            tp_size=8, pp_size=1,
            layer_splits=(L,),
            head_splits=tuple([NQ // 8] * 8),    # forced uniform by GQA
            kv_splits=tuple([model.num_kv_heads // 8] * 8),
            ffn_splits=tuple(ffn_arr),
            description=f"TP=8 PP=1 non-uniform FFN ∝ speed: "
                        f"{[ffn_arr[i] for i in [0, 4]]}",
        ))
        # TP=2 PP=4 — uniform layers (4 stages of 2 GPUs)
        plans.append(CandidatePlan(
            name="TP2_PP4_uniform",
            tp_size=2, pp_size=4,
            layer_splits=tuple([L // 4] * 4),
            head_splits=tuple([NQ // 2] * 2),
            kv_splits=tuple([model.num_kv_heads // 2] * 2),
            ffn_splits=tuple([FFN // 2] * 2),
            description="TP=2 PP=4 uniform — 4 stages × 2 GPUs",
        ))
        # TP=2 PP=4 — non-uniform layers proportional to per-stage speed
        stage_speeds = [sum(speeds[s*2:(s+1)*2]) / 2 for s in range(4)]
        t_layer = lambda bw: 443e6 / (bw * 1e9) * 1e6 + 30.0
        stage_costs = [t_layer(sp) for sp in stage_speeds]
        inv = [1.0 / c for c in stage_costs]
        raw_layers = [L * x / sum(inv) for x in inv]
        layer_split_4 = [max(1, round(x)) for x in raw_layers]
        layer_split_4[-1] += L - sum(layer_split_4)
        if all(l >= 1 for l in layer_split_4):
            plans.append(CandidatePlan(
                name="TP2_PP4_nonuniform_layer",
                tp_size=2, pp_size=4,
                layer_splits=tuple(layer_split_4),
                head_splits=tuple([NQ // 2] * 2),
                kv_splits=tuple([model.num_kv_heads // 2] * 2),
                ffn_splits=tuple([FFN // 2] * 2),
                description=f"TP=2 PP=4 non-uniform layer {layer_split_4}",
            ))

    return plans


# ----------------------------------------------------------------------
# Analytical planner: score a plan without simulating
# ----------------------------------------------------------------------

INTRA_BUSBW_GBs = 50.0
CROSS_BUSBW_GBs = 1.1
INTRA_AR_LAT_US = 3.0
CROSS_AR_LAT_US = 1000.0


def ar_overhead_us_per_layer(B: int, S: int, hidden: int,
                              tp_size: int, cross_node: bool) -> float:
    """Two ring AllReduces per layer of size (B·S·hidden·2 bytes)."""
    if tp_size <= 1:
        return 0.0
    ar_bytes = B * S * hidden * 2
    busbw = CROSS_BUSBW_GBs if cross_node else INTRA_BUSBW_GBs
    lat = CROSS_AR_LAT_US if cross_node else INTRA_AR_LAT_US
    per_ar_us = 2 * (tp_size - 1) / tp_size * ar_bytes / (busbw * 1e9) * 1e6
    return 2 * (per_ar_us + lat)


def pp_send_us(B: int, S: int, hidden: int, cross_node: bool) -> float:
    bytes_ = B * S * hidden * 2
    bw = CROSS_BUSBW_GBs if cross_node else INTRA_BUSBW_GBs
    lat = CROSS_AR_LAT_US if cross_node else INTRA_AR_LAT_US
    return bytes_ / (bw * 1e9) * 1e6 + lat


def per_rank_layer_bytes(model: ModelSpec, B: int, S: int, kv_len: int,
                          h_q: int, h_kv: int, ffn: int) -> float:
    """Bytes the GPU rank touches per layer (weights + activations + KV).

    Branches FFN bytes by `model.mlp_kind`:
      · swiglu        : gate + up + silu·mul + down  (≈ 3× ffn·hidden weights)
      · relu_2matmul  : fc1 + relu + fc2             (≈ 2× ffn·hidden weights)
    """
    hidden = model.hidden
    head_dim = model.head_dim
    local_qkv_dim = (h_q + 2 * h_kv) * head_dim
    local_q_dim = h_q * head_dim
    bytes_ = 0
    # Layer norms (small)
    bytes_ += B * S * hidden * 2 * 4
    # QKV
    bytes_ += (B * S * hidden + hidden * local_qkv_dim + B * S * local_qkv_dim) * 2
    # Attention KV (S=1 decode) or compute proxy (prefill)
    if S == 1:
        bytes_ += B * h_kv * kv_len * head_dim * 2 * 2     # KV read
    else:
        bytes_ += B * h_q * S * head_dim * 2 * 4           # rough QKVO acts
    # O proj
    bytes_ += (B * S * local_q_dim + local_q_dim * hidden + B * S * hidden) * 2
    # ─── FFN: branch by architecture ─────────────────────────────────
    if model.mlp_kind == "swiglu":
        # gate + up: 2 matmuls (or fused; same bytes)
        bytes_ += 2 * (B * S * hidden + hidden * ffn + B * S * ffn) * 2
        # silu·mul: 3 reads + 1 write of pointwise tensor
        bytes_ += B * S * ffn * 2 * 3
        # down
        bytes_ += (B * S * ffn + ffn * hidden + B * S * hidden) * 2
    elif model.mlp_kind == "relu_2matmul":
        # fc1: hidden → ffn
        bytes_ += (B * S * hidden + hidden * ffn + B * S * ffn) * 2
        # relu pointwise (small)
        bytes_ += B * S * ffn * 2 * 2
        # fc2: ffn → hidden
        bytes_ += (B * S * ffn + ffn * hidden + B * S * hidden) * 2
    else:
        raise ValueError(f"Unknown mlp_kind={model.mlp_kind!r}")
    # Residuals
    bytes_ += 2 * B * S * hidden * 2 * 3
    return bytes_


def predict_wall_us(plan: CandidatePlan, topo: ClusterTopology,
                    model: ModelSpec, workload: WorkloadSpec,
                    n_microbatches: int) -> float:
    """Analytical pipeline wall in microseconds.

    Per-rank per-layer cost = bytes / mem_bw + AR overhead.
    Per-stage cost = n_layers × max_rank_cost.
    Pipeline wall = (n_mb + pp − 1) × max_stage_cost + (pp-1) × PP send cost.
    """
    B, S, kv_len = workload.batch, workload.seq, workload.kv_len
    hidden = model.hidden
    pp_size = plan.pp_size
    tp_size = plan.tp_size
    stage_times = []
    for s in range(pp_size):
        ranks_in_stage = list(range(s * tp_size, (s + 1) * tp_size))
        nodes_in_group = set(topo.rank_to_node[r] for r in ranks_in_stage)
        cross_node_tp = len(nodes_in_group) > 1
        ar_us = ar_overhead_us_per_layer(B, S, hidden, tp_size, cross_node_tp)
        max_rank_us = 0.0
        for tp_rank, r in enumerate(ranks_in_stage):
            h_q = plan.head_splits[tp_rank]
            h_kv = plan.kv_splits[tp_rank]
            ffn = plan.ffn_splits[tp_rank]
            bytes_ = per_rank_layer_bytes(model, B, S, kv_len, h_q, h_kv, ffn)
            mem_bw = topo.rank_to_gpu[r].local_mem_bw_GBs
            t_mem_us = bytes_ / (mem_bw * 1e9) * 1e6
            t_layer_us = t_mem_us + ar_us
            max_rank_us = max(max_rank_us, t_layer_us)
        stage_times.append(plan.layer_splits[s] * max_rank_us)
    if not stage_times:
        return float("inf")
    max_stage = max(stage_times)
    pp_send_total_us = 0.0
    for s in range(pp_size - 1):
        leader_now = s * tp_size
        leader_next = (s + 1) * tp_size
        cross_node_pp = topo.rank_to_node[leader_now] != topo.rank_to_node[leader_next]
        pp_send_total_us += pp_send_us(B, S, hidden, cross_node_pp)
    wall_us = (n_microbatches + pp_size - 1) * max_stage + pp_send_total_us
    return wall_us


# ----------------------------------------------------------------------
# ASTRA-sim runner for a plan
# ----------------------------------------------------------------------

def write_sys(hetero: HeteroSimConfig, path: str, n_dims: int = 2):
    cfg = {
        "scheduling-policy": "LIFO",
        "endpoint-delay": 1,
        "active-chunks-per-dimension": 2,
        "preferred-dataset-splits": 4,
        "all-reduce-implementation": ["ring"] * n_dims,
        "all-gather-implementation": ["ring"] * n_dims,
        "reduce-scatter-implementation": ["ring"] * n_dims,
        "all-to-all-implementation": ["ring"] * n_dims,
        "collective-optimization": "localBWAware",
        "local-mem-bw": hetero.reference.local_mem_bw_GBs,
        "boost-mode": 0,
        "track-local-mem": 0,
        "roofline-enabled": 1,
        "peak-perf": hetero.reference.peak_perf_tflops,
    }
    with open(path, "w") as f:
        json.dump(cfg, f)


def simulator_serving_wall_s(plan: CandidatePlan, topo: ClusterTopology,
                              swl: ServingWorkload, sim_cache: dict) -> dict:
    """Simulator-side composition under continuous batching.

    Runs two ASTRA-sim ETs (prefill iter, decode iter) and composes them
    using the same continuous-batching scheduler model the planner uses:
      prefill phase: N solo prefills pipelined → (N + pp − 1) · T_max_stage_pre
      decode phase: L_out × pp emissions 1F1B → (L_out·pp + pp − 1) · T_max_stage_dec

    sim_cache memoizes ET runs per (plan, iter-spec) so we don't repeat
    the same simulation across SERVING_WORKLOADS that share an iter spec.
    """
    pp = plan.pp_size
    N = swl.n_requests
    L_out = swl.out_len
    avg_kv = swl.in_len + L_out // 2

    pre_iter = WorkloadSpec(batch=1, seq=swl.in_len, kv_len=0,
                             is_decode=False, n_microbatches=1)
    B_decode = max(1, N // pp)
    dec_iter = WorkloadSpec(batch=B_decode, seq=1, kv_len=avg_kv,
                             is_decode=True, n_microbatches=pp)

    cache_key_pre = (plan.name, "pre", pre_iter.batch, pre_iter.seq, pre_iter.kv_len, pre_iter.n_microbatches)
    cache_key_dec = (plan.name, "dec", dec_iter.batch, dec_iter.seq, dec_iter.kv_len, dec_iter.n_microbatches)

    if cache_key_pre not in sim_cache:
        sim_cache[cache_key_pre] = run_astra_for_plan(
            plan, topo, pre_iter,
            f"/tmp/p8serv_{topo.name.split(' ')[0]}_{plan.name}_pre_{pre_iter.seq}",
        )
    if cache_key_dec not in sim_cache:
        sim_cache[cache_key_dec] = run_astra_for_plan(
            plan, topo, dec_iter,
            f"/tmp/p8serv_{topo.name.split(' ')[0]}_{plan.name}_dec_{dec_iter.batch}_{dec_iter.kv_len}",
        )
    pre_iter_wall = sim_cache[cache_key_pre]
    dec_iter_wall = sim_cache[cache_key_dec]

    if pre_iter_wall != pre_iter_wall or dec_iter_wall != dec_iter_wall:
        return {"prefill_wall_s": float("nan"), "decode_wall_s": float("nan"),
                "total_s": float("nan")}

    # pre_iter sim ≈ pp · T_max_stage_pre (n_mb=1, one prefill traversal)
    T_max_stage_pre = pre_iter_wall / pp
    # dec_iter sim ≈ (pp + pp − 1) · T_max_stage_dec for n_mb=pp microbatches
    T_max_stage_dec = dec_iter_wall / (2 * pp - 1)

    prefill_phase = (N + pp - 1) * T_max_stage_pre
    decode_phase = (L_out * pp + pp - 1) * T_max_stage_dec
    total = prefill_phase + decode_phase
    return {
        "prefill_wall_s": prefill_phase,
        "decode_wall_s": decode_phase,
        "total_s": total,
        "T_max_stage_pre_s": T_max_stage_pre,
        "T_max_stage_dec_s": T_max_stage_dec,
    }


def run_astra_for_plan(plan: CandidatePlan, topo: ClusterTopology,
                       workload: WorkloadSpec, out_dir: str) -> float:
    """Returns ASTRA-sim wall (seconds) for this plan on this topology."""
    # Pick reference = fastest GPU.
    fastest = max(topo.rank_to_gpu, key=lambda g: g.local_mem_bw_GBs)
    hetero = HeteroSimConfig(
        reference=fastest,
        gpus_by_rank=list(topo.rank_to_gpu),
    )
    partition = PartitionSpec(
        tp_size=plan.tp_size, pp_size=plan.pp_size,
        layer_splits=list(plan.layer_splits),
        head_splits=list(plan.head_splits),
        kv_splits=list(plan.kv_splits),
        ffn_splits=list(plan.ffn_splits),
        rank_to_node=list(topo.rank_to_node),
    )
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    sys_path = f"/tmp/sys_{os.path.basename(out_dir)}.json"
    write_sys(hetero, sys_path, n_dims=topo.n_dims)
    builder = InferenceWorkloadBuilder(LLAMA_3_70B, partition, workload, hetero=hetero)
    builder.build()
    base = builder.write(out_dir)
    cmd = [
        ASTRA_BIN,
        f"--workload-configuration={base}",
        f"--comm-group-configuration={out_dir}/workload.json",
        f"--system-configuration={sys_path}",
        f"--remote-memory-configuration={NO_REMOTE}",
        f"--network-configuration={topo.network_file}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    walls = [int(l.split("Wall time:")[1].strip())
             for l in (proc.stdout + proc.stderr).splitlines()
             if "Wall time:" in l]
    if not walls:
        return float("nan")
    return max(walls) / 1e9


# ----------------------------------------------------------------------
# Topology definitions
# ----------------------------------------------------------------------

TOPOLOGIES: list[ClusterTopology] = [
    ClusterTopology(
        name="T1 2-node hetero H100|A100",
        rank_to_gpu=tuple([H100] * 4 + [A100] * 4),
        rank_to_node=tuple(["n0"] * 4 + ["n1"] * 4),
        network_file=NET_2NODE,
    ),
    ClusterTopology(
        name="T2 single-node mixed H100+A100",
        rank_to_gpu=tuple([H100] * 4 + [A100] * 4),
        rank_to_node=tuple(["n0"] * 8),
        network_file=NET_1NODE,
        is_single_node=True,
    ),
    ClusterTopology(
        name="T3 per-node mixed 2H+2A each",
        rank_to_gpu=tuple([H100, H100, A100, A100] * 2),
        rank_to_node=tuple(["n0"] * 4 + ["n1"] * 4),
        network_file=NET_2NODE,
    ),
    ClusterTopology(
        name="T4 2-node extreme B200|L4",
        rank_to_gpu=tuple([B200] * 4 + [L4] * 4),
        rank_to_node=tuple(["n0"] * 4 + ["n1"] * 4),
        network_file=NET_2NODE,
    ),
    ClusterTopology(
        name="T5 single-node homog 8×A100",
        rank_to_gpu=tuple([A100] * 8),
        rank_to_node=tuple(["n0"] * 8),
        network_file=NET_1NODE,
        is_single_node=True,
    ),
    ClusterTopology(
        name="T6 4-tier 2-node B200+H100|A100+L4",
        rank_to_gpu=tuple([B200] * 2 + [H100] * 2 + [A100] * 2 + [L4] * 2),
        rank_to_node=tuple(["n0"] * 4 + ["n1"] * 4),
        network_file=NET_2NODE,
    ),
]


@dataclass(frozen=True)
class WorkloadScenario:
    name: str
    workload: WorkloadSpec


WORKLOADS = [
    WorkloadScenario(
        "W1 decode steady-state",
        WorkloadSpec(batch=25, seq=1, kv_len=576, is_decode=True, n_microbatches=16),
    ),
    WorkloadScenario(
        "W2 prefill burst",
        WorkloadSpec(batch=1, seq=512, kv_len=0, is_decode=False, n_microbatches=1),
    ),
    WorkloadScenario(
        "W3 decode large-batch",
        WorkloadSpec(batch=128, seq=1, kv_len=576, is_decode=True, n_microbatches=4),
    ),
]


# ----------------------------------------------------------------------
# End-to-end serving workload (continuous batching)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ServingWorkload:
    """Full serving spec — N requests with the given in_len/out_len. The
    cost model (and simulator-side composer) translate this into a
    (prefill phase) + (decode phase) computation under the engine's
    continuous-batching scheduler:

      Phase 1 — prefill: each of N requests issues a solo prefill
        (B=1, S=in_len). They pipeline through PP stages, so the wall
        is (N + pp − 1) · T_max_stage_prefill.

      Phase 2 — decode: with auto_microbatch on (default), the engine
        splits the active decode pool into pp_size microbatches of
        B=N/pp_size each. In steady state, every cycle of T_max_stage_decode
        emits one microbatch's tokens. Total decode emissions =
        L_out · pp_size; pipeline wall = (L_out·pp + pp − 1) · T_max_stage_decode.

      The engine's scheduler prefers WAITING (prefill) over DECODING, so
      the two phases run sequentially (no overlap) for batched-submission
      workloads. Total wall = prefill_wall + decode_wall.
    """
    n_requests: int
    in_len: int
    out_len: int


SERVING_WORKLOADS: list[tuple[str, ServingWorkload]] = [
    ("S1 in512/out128/N50", ServingWorkload(n_requests=50, in_len=512, out_len=128)),
    ("S2 in1024/out256/N32", ServingWorkload(n_requests=32, in_len=1024, out_len=256)),
    ("S3 in128/out512/N50",  ServingWorkload(n_requests=50, in_len=128, out_len=512)),
]


def predict_serving_wall_us(plan: CandidatePlan, topo: ClusterTopology,
                             model: ModelSpec, swl: ServingWorkload) -> dict:
    """End-to-end serving wall under continuous batching.

    Returns dict with prefill_wall_us, decode_wall_us, total_us, plus the
    per-step T_max_stage values for debugging.
    """
    pp = plan.pp_size
    N = swl.n_requests
    L_out = swl.out_len
    avg_kv = swl.in_len + L_out // 2     # mean KV-cache length during decode

    # Per-step max-stage time for PREFILL (one solo prefill, B=1, S=in_len)
    pre_iter = WorkloadSpec(batch=1, seq=swl.in_len, kv_len=0,
                             is_decode=False, n_microbatches=1)
    pre_step = predict_wall_us(plan, topo, model, pre_iter, n_microbatches=1)
    # predict_wall_us with n_mb=1 returns (1 + pp - 1) * T_max_stage_prefill + pp_send
    # = pp * T_max_stage_prefill + pp_send. Recover T_max_stage:
    pp_send_pre = pp_send_overhead_us(plan, topo, model, pre_iter)
    T_max_stage_prefill = max(0.0, (pre_step - pp_send_pre) / pp) if pp >= 1 else pre_step

    # Per-step max-stage time for DECODE (auto_microbatch ON: B=N/pp)
    B_decode = max(1, N // pp)
    dec_iter = WorkloadSpec(batch=B_decode, seq=1, kv_len=avg_kv,
                             is_decode=True, n_microbatches=pp)
    dec_step = predict_wall_us(plan, topo, model, dec_iter, n_microbatches=pp)
    pp_send_dec = pp_send_overhead_us(plan, topo, model, dec_iter)
    # dec_step ≈ (pp + pp - 1) * T_max_stage_decode + pp_send
    T_max_stage_decode = max(0.0, (dec_step - pp_send_dec) / (2 * pp - 1))

    # ── Phase 1: N solo prefills, pipelined ──
    prefill_wall = (N + pp - 1) * T_max_stage_prefill + (N - 1 + pp) * pp_send_pre / max(N, 1)

    # ── Phase 2: L_out × pp emissions, 1F1B with pp microbatches in flight ──
    n_decode_emissions = L_out * pp
    decode_wall = (n_decode_emissions + pp - 1) * T_max_stage_decode + pp_send_dec * L_out

    total = prefill_wall + decode_wall
    return {
        "prefill_wall_us": prefill_wall,
        "decode_wall_us": decode_wall,
        "total_us": total,
        "T_max_stage_prefill_us": T_max_stage_prefill,
        "T_max_stage_decode_us":  T_max_stage_decode,
    }


def pp_send_overhead_us(plan: CandidatePlan, topo: ClusterTopology,
                          model: ModelSpec, workload: WorkloadSpec) -> float:
    """Sum of PP send costs across stage boundaries for one step."""
    pp = plan.pp_size
    tp = plan.tp_size
    hidden = model.hidden
    B, S = workload.batch, workload.seq
    total = 0.0
    for s in range(pp - 1):
        leader_now = s * tp
        leader_next = (s + 1) * tp
        cross_node = topo.rank_to_node[leader_now] != topo.rank_to_node[leader_next]
        total += pp_send_us(B, S, hidden, cross_node)
    return total


def evaluate_scenario(topo: ClusterTopology, wl_scenario: WorkloadScenario) -> dict:
    """Score and simulate all candidate plans for one (topology, workload)."""
    workload = wl_scenario.workload
    plans = enumerate_plans(topo, LLAMA_3_70B)
    rows = []
    for plan in plans:
        score_us = predict_wall_us(plan, topo, LLAMA_3_70B, workload,
                                    workload.n_microbatches)
        sim_wall_s = run_astra_for_plan(
            plan, topo, workload,
            f"/tmp/p8_{topo.name.split(' ')[0]}_{wl_scenario.name.split(' ')[0]}_{plan.name}",
        )
        rows.append({
            "plan": plan.name, "desc": plan.description,
            "planner_us": score_us, "sim_s": sim_wall_s,
            "skipped_sim": False,
        })

    rows_by_planner = sorted(rows, key=lambda r: r["planner_us"])
    rows_by_sim = sorted(
        [r for r in rows if r["sim_s"] == r["sim_s"]],
        key=lambda r: r["sim_s"]
    )
    planner_pick = rows_by_planner[0]["plan"]
    sim_winner = rows_by_sim[0]["plan"] if rows_by_sim else None
    is_match = planner_pick == sim_winner
    return {
        "topology": topo.name,
        "workload": wl_scenario.name,
        "planner_pick": planner_pick,
        "sim_winner": sim_winner,
        "match": is_match,
        "rows": rows,
        "rows_by_planner": rows_by_planner,
        "rows_by_sim": rows_by_sim,
    }


def evaluate_serving(topo: ClusterTopology, swl_name: str, swl: ServingWorkload,
                     sim_cache: dict) -> dict:
    """End-to-end serving evaluation under continuous batching.

    Each plan is scored by both the planner (predict_serving_wall_us)
    and by the simulator-side composer (simulator_serving_wall_s).
    """
    plans = enumerate_plans(topo, LLAMA_3_70B)
    rows = []
    for plan in plans:
        pred = predict_serving_wall_us(plan, topo, LLAMA_3_70B, swl)
        sim = simulator_serving_wall_s(plan, topo, swl, sim_cache)
        rows.append({
            "plan": plan.name,
            "planner_us": pred["total_us"],
            "planner_prefill_us": pred["prefill_wall_us"],
            "planner_decode_us":  pred["decode_wall_us"],
            "sim_s": sim["total_s"],
            "sim_prefill_s": sim["prefill_wall_s"],
            "sim_decode_s":  sim["decode_wall_s"],
        })
    rows_by_planner = sorted(rows, key=lambda r: r["planner_us"])
    rows_by_sim = sorted(
        [r for r in rows if r["sim_s"] == r["sim_s"]],
        key=lambda r: r["sim_s"],
    )
    planner_pick = rows_by_planner[0]["plan"]
    sim_winner = rows_by_sim[0]["plan"] if rows_by_sim else None
    return {
        "topology": topo.name,
        "serving": swl_name,
        "rows": rows,
        "rows_by_planner": rows_by_planner,
        "rows_by_sim": rows_by_sim,
        "planner_pick": planner_pick,
        "sim_winner": sim_winner,
        "match": planner_pick == sim_winner,
    }


def main_serving():
    """Phase 8b: end-to-end serving validation under continuous batching."""
    print("Phase 8b: end-to-end serving validation (continuous batching)")
    print("=" * 100)

    overall_serving: list[dict] = []
    for topo in TOPOLOGIES:
        sim_cache: dict = {}
        for swl_name, swl in SERVING_WORKLOADS:
            print(f"\n{'=' * 100}")
            print(f"{topo.name}  ×  {swl_name}")
            print(f"  N={swl.n_requests}, in={swl.in_len}, out={swl.out_len}")
            res = evaluate_serving(topo, swl_name, swl, sim_cache)
            print(f"\n  {'plan':<32s} {'planner total(ms)':>17s} {'sim total(s)':>14s} "
                  f"{'pre/dec sim (s)':>20s}")
            print("  " + "-" * 84)
            for r in res["rows_by_planner"]:
                sim_str = (f"{r['sim_s']:.4f}" if r['sim_s'] == r['sim_s']
                           else "FAIL")
                pd_str = (f"{r['sim_prefill_s']:.3f}/{r['sim_decode_s']:.3f}"
                          if r['sim_s'] == r['sim_s'] else "-")
                print(f"  {r['plan']:<32s} {r['planner_us']/1e3:>16.1f} "
                      f"{sim_str:>14s} {pd_str:>20s}")
            slowdown = 1.0
            if not res["match"] and res["sim_winner"]:
                pick_row = next(r for r in res["rows"] if r["plan"] == res["planner_pick"])
                if pick_row["sim_s"] == pick_row["sim_s"]:
                    slowdown = pick_row["sim_s"] / res["rows_by_sim"][0]["sim_s"]
            print(f"\n  Planner pick:   {res['planner_pick']}")
            print(f"  Simulator best: {res['sim_winner']}")
            print(f"  Match: {'YES' if res['match'] else f'NO ({slowdown:.2f}× slow)'}")
            res["slowdown"] = slowdown
            overall_serving.append(res)

    print()
    print("=" * 100)
    print("SUMMARY — end-to-end serving, planner vs simulator under continuous batching")
    print("=" * 100)
    n_match = sum(1 for o in overall_serving if o["match"])
    print(f"  Match rate: {n_match} / {len(overall_serving)} (topology × serving) scenarios")
    print()
    print(f"  {'topology':<34s} {'serving':<22s} {'planner pick':<32s} {'match':>10s}")
    print("  " + "-" * 100)
    for o in overall_serving:
        tag = "✓" if o["match"] else f"× ({o['slowdown']:.2f}×)"
        print(f"  {o['topology']:<34s} {o['serving']:<22s} {o['planner_pick']:<32s} {tag:>10s}")

    families = {}
    for o in overall_serving:
        if o["sim_winner"]:
            fam = "_".join(o["sim_winner"].split("_")[0:2])
            families.setdefault(fam, []).append(f"{o['topology']} / {o['serving']}")
    print()
    print("  Winning-plan diversity:")
    for fam, scens in families.items():
        print(f"    {fam}: {len(scens)} scenarios")
    with open("/tmp/phase8b_serving_results.json", "w") as f:
        json.dump(overall_serving, f, indent=2, default=str)


def main():
    print("Phase 8: planner picks best plan across TP/PP/Hybrid + workloads")
    print("=" * 100)

    overall: list[dict] = []
    for topo in TOPOLOGIES:
        for wl in WORKLOADS:
            print(f"\n{'=' * 100}")
            print(f"{topo.name}  ×  {wl.name}")
            print(f"  rank→GPU:  {[g.name for g in topo.rank_to_gpu]}")
            print(f"  rank→node: {list(topo.rank_to_node)}")
            print(f"  workload:  B={wl.workload.batch} S={wl.workload.seq} "
                  f"kv={wl.workload.kv_len} n_mb={wl.workload.n_microbatches} "
                  f"({'decode' if wl.workload.is_decode else 'prefill'})")

            res = evaluate_scenario(topo, wl)
            print(f"\n  {'plan':<32s} {'planner (µs)':>14s} {'sim wall (s)':>14s}")
            print("  " + "-" * 66)
            for r in res["rows_by_planner"]:
                if r["skipped_sim"]:
                    sim_str = "(sim skip)"
                elif r["sim_s"] != r["sim_s"]:
                    sim_str = "FAIL"
                else:
                    sim_str = f"{r['sim_s']:.4f}"
                print(f"  {r['plan']:<32s} {r['planner_us']:>14.1f} {sim_str:>14s}")

            slowdown = 1.0
            if not res["match"] and res["sim_winner"] is not None:
                pick_row = next(r for r in res["rows"] if r["plan"] == res["planner_pick"])
                best_row = res["rows_by_sim"][0]
                if pick_row["sim_s"] == pick_row["sim_s"]:
                    slowdown = pick_row["sim_s"] / best_row["sim_s"]

            print(f"\n  Planner pick:   {res['planner_pick']}")
            print(f"  Simulator best: {res['sim_winner']}")
            tag = "YES" if res["match"] else f"NO (planner choice {slowdown:.2f}× slower)"
            print(f"  Match: {tag}")
            res["slowdown_if_mismatch"] = slowdown
            overall.append(res)

    print()
    print("=" * 100)
    print("SUMMARY — planner pick vs simulator best across (topology × workload)")
    print("=" * 100)
    n_match = sum(1 for o in overall if o["match"])
    print(f"  Match rate: {n_match} / {len(overall)} scenarios")
    print()
    print(f"  {'topology':<34s} {'workload':<24s} {'planner pick':<32s} {'match':>10s}")
    print("  " + "-" * 100)
    for o in overall:
        tag = "✓" if o["match"] else f"× ({o['slowdown_if_mismatch']:.2f}× slow)"
        print(f"  {o['topology']:<34s} {o['workload']:<24s} {o['planner_pick']:<32s} {tag:>10s}")

    # Family breakdown — which plan family wins where
    print()
    print("  Winning-plan diversity:")
    families = {}
    for o in overall:
        if o["sim_winner"]:
            fam = o["sim_winner"].split("_")[0:2]
            fam = "_".join(fam)
            families.setdefault(fam, []).append(f"{o['topology']} / {o['workload']}")
    for fam, scens in families.items():
        print(f"    {fam}: {len(scens)} scenarios")
        for s in scens:
            print(f"        {s}")

    with open("/tmp/phase8_results.json", "w") as f:
        json.dump(overall, f, indent=2, default=str)


if __name__ == "__main__":
    import sys as _sys
    if "--serving" in _sys.argv:
        main_serving()
    else:
        main()
