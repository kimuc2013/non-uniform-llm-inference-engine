"""Engine-aware, workload-aware re-scorer for planner candidates.

Why this exists separately from `cost_model.wall_time_s`:

  - The cost model emits one number (predicted_wall_s). For paper-quality
    ranking we want to expose the COMPONENTS (prefill wall, decode wall,
    bubble, comm) so we can:
      (a) re-weight by workload class (decode_heavy vs prefill_heavy);
      (b) layer in structural penalties (cross-node TP, slow-rank
          dominance, KV-cache memory pressure, imperfect PP overlap);
      (c) emit a full score breakdown for the JSON output.

  - vLLM-style limitations the cost model under-counts:
      - Imperfect PP overlap: real PP=4 gets ~50-70% overlap not 100%.
        We model this as `pp_bubble_multiplier` that shrinks slowly with PP
        depth past 2.
      - Cross-node TP: each AllReduce hop on slow PCIe/IB adds latency
        floor + bandwidth tax.
      - Slow-rank dominance: if a stage's slowest rank is much slower than
        others, the stage waits — quantified by max/mean ratio.

All penalties are deterministic and reproducible from the partition spec —
no benchmarking required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .cost_model import CostModel, GpuProfile
from .planner import PartitionSpec, Workload
from .workload import WorkloadClass


# ---------------------------------------------------------------------------
# Memory feasibility helpers
# ---------------------------------------------------------------------------
BYTES_PER_BF16 = 2


def _model_param_bytes(model) -> int:
    """Rough param count → bytes. Counts attention + FFN weights, ignores
    norms/embed/lm_head which are <1% noise."""
    n_layers = model.num_layers
    h = model.hidden_size
    # Attention: QKV(combined_dim×h) + O(h×q_proj_dim)
    attn = (model.qkv_combined_dim * h) + (h * model.q_proj_dim)
    # FFN: gate(h×inter) + up(h×inter) + down(inter×h)
    ffn = 3 * h * model.intermediate_size
    # Embed + lm_head
    embed = model.vocab_size * h * 2
    return BYTES_PER_BF16 * (n_layers * (attn + ffn) + embed)


def stage_param_bytes(model, n_layers_in_stage: int, tp_size: int) -> int:
    """Per-rank weight bytes for a stage with n_layers and tp_size shards."""
    h = model.hidden_size
    attn = (model.qkv_combined_dim * h) + (h * model.q_proj_dim)
    ffn = 3 * h * model.intermediate_size
    per_layer = (attn + ffn) // max(1, tp_size)
    embed = (model.vocab_size * h) // max(1, tp_size)
    return BYTES_PER_BF16 * (n_layers_in_stage * per_layer + embed)


def kv_cache_bytes_per_rank(
    model,
    *,
    n_layers_in_stage: int,
    tp_size: int,
    max_concurrent_seqs: int,
    seq_len: int,
) -> int:
    """KV-cache bytes resident on one rank in steady state."""
    kv_heads_local = max(1, model.num_kv_heads // tp_size)
    head_dim = model.head_dim
    # K + V, bf16
    per_token = 2 * kv_heads_local * head_dim * BYTES_PER_BF16
    return n_layers_in_stage * max_concurrent_seqs * seq_len * per_token


# ---------------------------------------------------------------------------
# Score components
# ---------------------------------------------------------------------------
@dataclass
class ScoreBreakdown:
    # Primary: workload-weighted cost (lower is better).
    weighted_cost_s: float = 0.0
    # Components (seconds).
    prefill_wall_s: float = 0.0
    decode_wall_s: float = 0.0
    pp_bubble_s: float = 0.0
    # Penalty multipliers (applied multiplicatively to weighted_cost).
    cross_node_tp_penalty: float = 0.0   # additive seconds
    cross_node_pp_penalty: float = 0.0
    slow_rank_dominance_penalty: float = 0.0
    pp_depth_penalty: float = 0.0
    # Memory check
    peak_weight_GB_per_rank: float = 0.0
    peak_kv_GB_per_rank: float = 0.0
    peak_total_GB_per_rank: float = 0.0
    vram_GB_min_in_cluster: float = 0.0
    memory_feasible: bool = True
    memory_reason: str = ""
    # Predicted PP overlap fraction (0..1) — diagnostic only.
    predicted_pp_overlap: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "weighted_cost_s": round(self.weighted_cost_s, 4),
            "prefill_wall_s": round(self.prefill_wall_s, 4),
            "decode_wall_s": round(self.decode_wall_s, 4),
            "pp_bubble_s": round(self.pp_bubble_s, 4),
            "cross_node_tp_penalty_s": round(self.cross_node_tp_penalty, 4),
            "cross_node_pp_penalty_s": round(self.cross_node_pp_penalty, 4),
            "slow_rank_dominance_penalty_s": round(self.slow_rank_dominance_penalty, 4),
            "pp_depth_penalty_s": round(self.pp_depth_penalty, 4),
            "peak_weight_GB_per_rank": round(self.peak_weight_GB_per_rank, 2),
            "peak_kv_GB_per_rank": round(self.peak_kv_GB_per_rank, 2),
            "peak_total_GB_per_rank": round(self.peak_total_GB_per_rank, 2),
            "vram_GB_min_in_cluster": round(self.vram_GB_min_in_cluster, 2),
            "memory_feasible": self.memory_feasible,
            "memory_reason": self.memory_reason,
            "predicted_pp_overlap": round(self.predicted_pp_overlap, 3),
        }


def predicted_pp_overlap_fraction(pp_size: int) -> float:
    """Best-case PP overlap achievable. Calibrated to measured values:
    PP=1 → 1.0, PP=2 → 0.65, PP=4 → 0.55, deeper → asymptote ~0.45.

    Used as 1/predicted_overlap multiplier on PP-arm wall when scoring."""
    if pp_size <= 1:
        return 1.0
    # Smooth saturating curve.
    return max(0.45, 0.65 - 0.04 * (pp_size - 2))


def score_partition(
    *,
    partition: PartitionSpec,
    workload: Workload,
    cm: CostModel,
    cls: WorkloadClass,
    cluster_min_vram_GB: float,
    fix_micro_overhead_s_per_mb: float = 0.0015,  # 1.5ms per mb from M13d
) -> ScoreBreakdown:
    """Compute structured score for a partition under a workload class."""
    sb = ScoreBreakdown()
    model = cm.model

    # 1) Prefill and decode walls — call cost-model primitives separately so
    #    we can re-weight them.
    t_prefill_step = cm.step_time_s(partition, B=1, S=workload.in_len, kv_len=0)
    t_prefill_step = cm._add_step_overhead(t_prefill_step, partition)
    t_prefill_total = workload.n_requests * t_prefill_step

    avg_kv_len = workload.in_len + workload.out_len // 2
    n_microbatches = max(1, partition.pp_size)
    B_decode = max(1, workload.n_requests // n_microbatches)
    t_decode_step = cm.step_time_s(partition, B=B_decode, S=1, kv_len=avg_kv_len)
    t_decode_step = cm._add_step_overhead(t_decode_step, partition)
    n_decode_emissions = workload.out_len * n_microbatches
    t_decode_total = n_decode_emissions * t_decode_step

    # PP overlap: cost model's wall_time_s assumes ideal pipeline. Real PP
    # only overlaps ~50-70%, so the *effective* decode wall is amplified by
    # 1/overlap_fraction (the slowest stage limits throughput, and the
    # bubble factor for cold-start contributes a one-off cost).
    overlap = predicted_pp_overlap_fraction(partition.pp_size)
    sb.predicted_pp_overlap = overlap
    if partition.pp_size > 1:
        # Decode amplification (real wall ≈ ideal / overlap).
        decode_amplified = t_decode_total / overlap
        sb.pp_bubble_s = decode_amplified - t_decode_total
        t_decode_total = decode_amplified
        # Microbatch fixed-overhead per mb (~1.5ms each).
        # Sub-mbs per timestep ≈ workload.n_requests / max(1, workload.n_requests // pp_size)
        mb_count = partition.pp_size * workload.out_len * partition.pp_size
        sb.pp_bubble_s += mb_count * fix_micro_overhead_s_per_mb * 0.1
        t_decode_total += mb_count * fix_micro_overhead_s_per_mb * 0.1

    sb.prefill_wall_s = t_prefill_total
    sb.decode_wall_s = t_decode_total

    # 2) Workload-weighted base cost.
    base = cls.w_prefill * t_prefill_total + cls.w_decode * t_decode_total

    # 3) Structural penalties.
    # cross-node TP penalty: each cross-node TP stage adds 25% to its
    # decode AllReduce time. Quantify by counting cross-node TP stages and
    # multiplying by per-layer AR time × n_layers in that stage.
    if any(partition.tp_cross_node):
        # Use rate based on biggest stage.
        per_layer_ar = cm.per_layer_allreduce_s(
            B=B_decode, S=1, tp_size=partition.tp_size, cross_node=True
        )
        cross_node_stages = sum(1 for x in partition.tp_cross_node if x)
        sb.cross_node_tp_penalty = (
            cross_node_stages * 0.25 * per_layer_ar
            * (sum(partition.layer_splits) / max(1, partition.pp_size))
            * workload.out_len
        )

    # cross-node PP penalty: each cross-node PP edge adds 50% to its send.
    if any(partition.pp_cross_node):
        per_pp_send = cm.pp_send_s(B_decode, S=1, cross_node=True)
        cross_node_edges = sum(1 for x in partition.pp_cross_node if x)
        sb.cross_node_pp_penalty = (
            cross_node_edges * 0.50 * per_pp_send * workload.out_len
        )

    # slow-rank dominance: compute per-rank per-layer decode time over the
    # stage-0 TP group; if max/mean ratio > 1.5, add (max-mean) × n_layers ×
    # out_len.
    if partition.tp_size > 1:
        stage0_ranks = partition.stage_rank_groups[0]
        per_rank_times = []
        for tp_rank, r in enumerate(stage0_ranks):
            t = cm.per_layer_compute_s(
                r, B_decode, 1,
                partition.tp_head_splits[tp_rank],
                partition.tp_kv_splits[tp_rank],
                partition.tp_ffn_splits[tp_rank],
                kv_len=avg_kv_len,
            )
            per_rank_times.append(t)
        if per_rank_times:
            t_max = max(per_rank_times); t_mean = sum(per_rank_times) / len(per_rank_times)
            if t_mean > 0 and t_max / t_mean > 1.5:
                gap = t_max - t_mean
                n_layers_stage0 = partition.layer_splits[0]
                sb.slow_rank_dominance_penalty = (
                    gap * n_layers_stage0 * (workload.out_len + workload.in_len)
                )

    # PP-depth penalty from workload class (per-stage beyond the first one).
    sb.pp_depth_penalty = (partition.pp_size - 1) * cls.pp_depth_penalty * (
        base + sb.cross_node_tp_penalty + sb.cross_node_pp_penalty
    )

    # 4) Apply TP comm weight (workload-class-specific) — implicit via
    # base reuse of t_decode_total which already includes AR. We mirror
    # tp_comm_weight by adding (tp_comm_weight - 1) × per-layer-AR × n_layers
    # × out_len when tp_size > 1.
    if partition.tp_size > 1 and abs(cls.tp_comm_weight - 1.0) > 1e-6:
        per_layer_ar_intra = cm.per_layer_allreduce_s(
            B=B_decode, S=1, tp_size=partition.tp_size,
            cross_node=any(partition.tp_cross_node),
        )
        total_ar = per_layer_ar_intra * sum(partition.layer_splits) * workload.out_len
        base += (cls.tp_comm_weight - 1.0) * total_ar

    # 5) Final weighted cost = base + penalties.
    sb.weighted_cost_s = (base
                          + sb.cross_node_tp_penalty
                          + sb.cross_node_pp_penalty
                          + sb.slow_rank_dominance_penalty
                          + sb.pp_depth_penalty)

    # 6) Memory feasibility check.
    # Assume max_concurrent_seqs = n_requests, seq_len = in_len + out_len.
    full_seq_len = workload.in_len + workload.out_len
    # Stage 0 is usually the worst (embedding lives there); check each stage.
    weight_max = 0.0
    kv_max = 0.0
    for s, n_layers in enumerate(partition.layer_splits):
        w = stage_param_bytes(model, n_layers, partition.tp_size)
        weight_max = max(weight_max, w)
        kv = kv_cache_bytes_per_rank(
            model,
            n_layers_in_stage=n_layers,
            tp_size=partition.tp_size,
            max_concurrent_seqs=workload.n_requests,
            seq_len=full_seq_len,
        )
        kv_max = max(kv_max, kv)
    sb.peak_weight_GB_per_rank = weight_max / 1e9
    sb.peak_kv_GB_per_rank = kv_max / 1e9
    # Activation working set ≈ 4 × hidden × max(B*S, B) × 2 bytes (rough).
    act_GB = 4 * model.hidden_size * max(workload.n_requests * workload.in_len,
                                         workload.n_requests) * BYTES_PER_BF16 / 1e9
    sb.peak_total_GB_per_rank = sb.peak_weight_GB_per_rank + sb.peak_kv_GB_per_rank + min(2.0, act_GB)
    sb.vram_GB_min_in_cluster = cluster_min_vram_GB
    if sb.peak_total_GB_per_rank > cluster_min_vram_GB:
        sb.memory_feasible = False
        sb.memory_reason = (
            f"per-rank peak {sb.peak_total_GB_per_rank:.1f} GB > "
            f"min VRAM {cluster_min_vram_GB:.1f} GB"
        )

    return sb


__all__ = ["score_partition", "ScoreBreakdown", "predicted_pp_overlap_fraction",
           "stage_param_bytes", "kv_cache_bytes_per_rank"]
