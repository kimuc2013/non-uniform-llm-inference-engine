"""Cost model for hetero hybrid parallelism.

Predicts step time and end-to-end wall time for a given (model, workload,
partition) on a hetero cluster. Inputs come from cluster calibration JSONs.

Per-matmul time uses a two-tier model:
  (1) Per-shape measured lookup — if calib_compute has measured the (K, N)
      pair at multiple M values, interpolate. This captures L2-cache effects
      and kernel-selection behavior that simple roofline misses, especially
      at small M (low concurrency decode).
  (2) Roofline fallback — for shapes not in the calibration table, use
      max(compute, mem-bound) with the GPU's calibrated TFLOPS / mem BW.

Per-layer time = sum of matmuls + attention (KV-read for decode, B*S² compute
for prefill). Per-step time = max over PP stages of (n_layers * per-layer)
+ TP AllReduce + PP send. Wall = N_req * T_prefill + n_emissions * T_decode_step.

Higher-order effects (NCCL kernel-launch latency, Python scheduler, paged-attn
internals) are absorbed by three per-cluster calibration constants:
PER_STAGE_STEP_OVERHEAD_S, PER_LAYER_OVERHEAD_S, the AR latency floor.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .model_spec import ModelSpec


@dataclass(frozen=True)
class GpuProfile:
    """Effective compute, memory, and per-shape matmul cost of one GPU type.

    Populated from `tests/calib_compute.py` JSON. Carries:
      - Aggregate `tflops_*` / `mem_bw_GBs` / `kv_bw_GBs` for roofline fallback
        and attention modeling.
      - `matmul_lookup`: per-(K,N) measured time at multiple M values for
        direct interpolation (preferred over roofline when shape is calibrated).
    """

    name: str
    tflops_prefill: float       # avg TFLOPS at large M (compute-bound regime)
    tflops_decode: float        # avg TFLOPS at small M (memory-bound regime)
    mem_bw_GBs: float           # sustained HBM bandwidth, large copy
    kv_bw_GBs: float            # effective KV-read bandwidth (batched paged-attn pattern)
    # Per-(K, N) sorted-by-M lookup table: dict[(K, N)] = [(M, time_s), ...].
    # Populated from `matmul_by_shape` field of the calib JSON.
    matmul_lookup: dict[tuple[int, int], list[tuple[int, float]]] = field(default_factory=dict)

    @classmethod
    def from_calib_json(cls, path: str | Path) -> "GpuProfile":
        with open(path) as f:
            d = json.load(f)

        # Aggregate TFLOPS from large-M and small-M shapes for fallback.
        large_m_TFLOPS: list[float] = []
        small_m_TFLOPS: list[float] = []
        for shape_key, data in d.get("matmul_by_shape", {}).items():
            for row in data.get("per_m", []):
                m = row["M"]
                f_ = row.get("TFLOPS", 0.0)
                if m >= 1024:
                    large_m_TFLOPS.append(f_)
                elif m <= 50:
                    small_m_TFLOPS.append(f_)
        tflops_prefill = (sum(large_m_TFLOPS) / len(large_m_TFLOPS)
                          if large_m_TFLOPS else 100.0)
        tflops_decode = (sum(small_m_TFLOPS) / len(small_m_TFLOPS)
                         if small_m_TFLOPS else 50.0)

        mem_bw = d["mem_bw"]["large_copy"]["GB_per_s"]
        kv_bw = (d.get("kv_read", {}).get("tp4_b50_s640_kv2", {}).get("effective_GBs")
                 or d.get("kv_read", {}).get("tp4_b50_s640", {}).get("effective_GBs")
                 or mem_bw)

        # Build per-shape lookup.
        lookup: dict[tuple[int, int], list[tuple[int, float]]] = {}
        for _, data in d.get("matmul_by_shape", {}).items():
            K, N = data["K"], data["N"]
            pts = sorted(((row["M"], row["time_ms"] / 1000.0)
                          for row in data["per_m"]), key=lambda x: x[0])
            lookup[(K, N)] = pts

        return cls(
            name=d["gpu_name"],
            tflops_prefill=tflops_prefill,
            tflops_decode=tflops_decode,
            mem_bw_GBs=mem_bw,
            kv_bw_GBs=kv_bw,
            matmul_lookup=lookup,
        )

    def measured_matmul_time(self, M: int, K: int, N: int) -> float | None:
        """Interpolated / extrapolated time (s) from the lookup table.

        Returns None if (K, N) wasn't calibrated. Linear interpolation in M
        between bracketing measured points; linear extrapolation on the two
        nearest points for M outside the calibrated range. Clamps to >= 0.
        """
        pts = self.matmul_lookup.get((K, N))
        if not pts:
            return None
        if M <= pts[0][0]:
            # Extrapolate using the two smallest M points.
            if len(pts) >= 2:
                m0, t0 = pts[0]; m1, t1 = pts[1]
                if m1 != m0:
                    t = t0 + (t1 - t0) * (M - m0) / (m1 - m0)
                    return max(0.0, t)
            return pts[0][1]
        if M >= pts[-1][0]:
            if len(pts) >= 2:
                m0, t0 = pts[-2]; m1, t1 = pts[-1]
                if m1 != m0:
                    t = t0 + (t1 - t0) * (M - m0) / (m1 - m0)
                    return max(0.0, t)
            return pts[-1][1]
        # Bracketing.
        for i in range(len(pts) - 1):
            m0, t0 = pts[i]; m1, t1 = pts[i + 1]
            if m0 <= M <= m1:
                if m1 == m0:
                    return t0
                return t0 + (t1 - t0) * (M - m0) / (m1 - m0)
        return pts[-1][1]  # unreachable


@dataclass(frozen=True)
class NetworkProfile:
    """Cluster network performance.

    AllReduce busbw depends on group size and locality (intra vs cross-node).
    PP send is point-to-point. Numbers can be populated from calibration or
    fall back to typical hardware estimates.
    """

    # GB/s, intra-node AllReduce bus bandwidth at large message size.
    intra_allreduce_busbw_GBs: float = 100.0    # NVLink/PCIe — conservative.
    # GB/s, cross-node AllReduce bus bandwidth at large message size.
    cross_allreduce_busbw_GBs: float = 12.0     # IB or 100GbE typical.
    # ms, AllReduce base latency for small messages.
    intra_allreduce_latency_ms: float = 0.05
    cross_allreduce_latency_ms: float = 0.5
    # GB/s, PP send between nodes.
    inter_node_p2p_GBs: float = 12.0
    intra_node_p2p_GBs: float = 25.0

    @classmethod
    def from_calib_json(cls, path: str | Path) -> "NetworkProfile":
        with open(path) as f:
            d = json.load(f)
        ar = d.get("allreduce", {})
        p2p = d.get("p2p", {})
        # Extract large-msg busbw (use last entry, biggest size).
        intra_busbw = cls._extract_busbw(ar.get("intra_head_4", []))
        cross_busbw = cls._extract_busbw(ar.get("cross_node_8", []))
        intra_lat = cls._extract_latency(ar.get("intra_head_4", []))
        cross_lat = cls._extract_latency(ar.get("cross_node_8", []))
        h2w = cls._extract_p2p_bw(p2p.get("head_to_worker", []))
        return cls(
            intra_allreduce_busbw_GBs=intra_busbw or 100.0,
            cross_allreduce_busbw_GBs=cross_busbw or 12.0,
            intra_allreduce_latency_ms=intra_lat or 0.05,
            cross_allreduce_latency_ms=cross_lat or 0.5,
            inter_node_p2p_GBs=h2w or 12.0,
        )

    @staticmethod
    def _extract_busbw(rows: list[dict]) -> float | None:
        if not rows: return None
        return max(r.get("busbw_GBs", 0) for r in rows)

    @staticmethod
    def _extract_latency(rows: list[dict]) -> float | None:
        if not rows: return None
        # smallest-size row gives latency floor.
        smallest = min(rows, key=lambda r: r["size_bytes"])
        return smallest["time_ms"]

    @staticmethod
    def _extract_p2p_bw(rows: list[dict]) -> float | None:
        if not rows: return None
        return max(r.get("GBs", 0) for r in rows)


@dataclass(frozen=True)
class CostModel:
    """Per-step + end-to-end timing predictor."""

    model: ModelSpec
    network: NetworkProfile
    # Per-rank GPU profile — keyed by rank index (planner assigns these by
    # mapping rank → node → GPU type).
    gpu_per_rank: tuple[GpuProfile, ...]

    # -------- Matmul time: measured lookup + roofline fallback --------
    def matmul_time_s(self, M: int, K: int, N: int, gpu: GpuProfile,
                     regime: str = "auto") -> float:
        """Matmul time (seconds), preferring measured calib lookup.

        Path 1 (preferred): if (K, N) is in `gpu.matmul_lookup`, return the
            measured time interpolated/extrapolated to M. This captures
            L2-cache effects and kernel-selection behavior — crucial for
            small-M decode where Blackwell achieves cache-amplified BW that
            simple HBM-roofline cannot predict.

        Path 2 (roofline fallback): for shapes not in the calibration table
            (e.g., a new model the cluster wasn't calibrated for), use
            max(compute-bound, memory-bound) roofline.
        """
        measured = gpu.measured_matmul_time(M, K, N)
        if measured is not None:
            return measured
        # Roofline fallback.
        flops = 2 * M * K * N
        bytes_moved = (M * K + K * N + M * N) * 2  # bf16
        if regime == "auto":
            regime = "prefill" if M >= 256 else "decode"
        if regime == "prefill":
            t_compute = flops / (gpu.tflops_prefill * 1e12)
            t_mem = bytes_moved / (gpu.mem_bw_GBs * 1e9)
            return max(t_compute, t_mem)
        else:
            return bytes_moved / (gpu.mem_bw_GBs * 1e9)

    # -------- Per-layer compute time per rank --------
    def per_layer_compute_s(self, rank: int, B: int, S: int,
                           tp_split_q: int, tp_split_kv: int, tp_split_ffn: int,
                           kv_len: int) -> float:
        """Sum of one transformer layer's matmuls + attention KV reads, per rank."""
        gpu = self.gpu_per_rank[rank]
        h = self.model.hidden_size
        head_dim = self.model.head_dim
        # M = B * S (token count).
        M = B * S
        # QKV proj: M, hidden → (q_dim + 2*kv_dim) per rank.
        local_qkv_dim = (tp_split_q + 2 * tp_split_kv) * head_dim
        t_qkv = self.matmul_time_s(M, h, local_qkv_dim, gpu)
        # Attention scores+softmax+output (model as KV-read bound for decode).
        # Decode (S=1): KV-cache read per layer ≈ B * kv_len * tp_split_kv *
        # head_dim * 2 (k+v) * 2 (bf16) bytes.
        if S == 1:
            kv_bytes = B * kv_len * tp_split_kv * head_dim * 2 * 2
            t_attn = kv_bytes / (gpu.kv_bw_GBs * 1e9)
        else:
            # Prefill: O(B * S^2 * heads) — model as compute-bound matmul of
            # roughly (B*S, S) @ (S, head_dim) per head.
            attn_flops = 2 * B * tp_split_q * S * S * head_dim
            t_attn = attn_flops / (gpu.tflops_prefill * 1e12)
        # O proj: M, (q_dim per rank) → hidden  [row-parallel, before AllReduce]
        local_q_dim = tp_split_q * head_dim
        t_o = self.matmul_time_s(M, local_q_dim, h, gpu)
        # FFN gate + up: M, hidden → ffn_local each
        t_gate_up = 2 * self.matmul_time_s(M, h, tp_split_ffn, gpu)
        # FFN down: M, ffn_local → hidden  [row-parallel, before AllReduce]
        t_down = self.matmul_time_s(M, tp_split_ffn, h, gpu)
        return t_qkv + t_attn + t_o + t_gate_up + t_down

    # -------- Per-layer AllReduce time (2 per layer: attn-out, ffn-down) --------
    def per_layer_allreduce_s(self, B: int, S: int, tp_size: int, cross_node: bool) -> float:
        """Time for 2 AllReduces of shape (B, S, hidden) bf16 per layer."""
        if tp_size <= 1:
            return 0.0
        tensor_bytes = B * S * self.model.hidden_size * 2
        bus_bytes = 2 * (tp_size - 1) / tp_size * tensor_bytes  # ring AllReduce
        bw = (self.network.cross_allreduce_busbw_GBs if cross_node
              else self.network.intra_allreduce_busbw_GBs)
        lat = (self.network.cross_allreduce_latency_ms if cross_node
               else self.network.intra_allreduce_latency_ms) / 1000.0
        t_one = lat + bus_bytes / (bw * 1e9)
        return 2 * t_one  # 2 AllReduces per layer

    # -------- PP send time (between stages) --------
    def pp_send_s(self, B: int, S: int, cross_node: bool) -> float:
        tensor_bytes = B * S * self.model.hidden_size * 2
        bw = (self.network.inter_node_p2p_GBs if cross_node
              else self.network.intra_node_p2p_GBs)
        return tensor_bytes / (bw * 1e9)

    # -------- Stage time given layer count + tp split --------
    def stage_time_s(self, n_layers: int, stage_ranks: list[int],
                    B: int, S: int, kv_len: int,
                    tp_head_splits: list[int], tp_kv_splits: list[int],
                    tp_ffn_splits: list[int], cross_node_tp: bool) -> float:
        """Stage time = N_layers × (max_rank(per_layer_compute) + AR + overhead).

        Each rank in the TP group gets its OWN tp_*_splits entry (the lists
        are indexed by tp_rank), so a hetero TP group with biased shards
        properly assigns smaller work to slow GPUs and larger to fast ones.
        The stage time is the bottleneck (max-over-ranks) plus comm + overhead.
        """
        tp_size = len(stage_ranks)
        per_rank_times = []
        for tp_rank, r in enumerate(stage_ranks):
            t = self.per_layer_compute_s(
                r, B, S,
                tp_split_q=tp_head_splits[tp_rank],
                tp_split_kv=tp_kv_splits[tp_rank],
                tp_split_ffn=tp_ffn_splits[tp_rank],
                kv_len=kv_len,
            )
            per_rank_times.append(t)
        per_layer_compute = max(per_rank_times)
        per_layer_comm = self.per_layer_allreduce_s(B, S, tp_size, cross_node_tp)
        per_layer_total = per_layer_compute + per_layer_comm + self.PER_LAYER_OVERHEAD_S
        return n_layers * per_layer_total

    # -------- Step time across all PP stages (1F1B steady state) --------
    def step_time_s(self, partition: "PartitionSpec", B: int, S: int,
                   kv_len: int) -> float:
        # Stage time = max compute+comm time for each stage.
        stage_times = []
        for s, (n_layers, stage_ranks) in enumerate(zip(
                partition.layer_splits, partition.stage_rank_groups)):
            # Pass the full per-rank TP split arrays so stage_time_s can index
            # each rank's own shard (proper hetero-TP handling).
            t = self.stage_time_s(
                n_layers=n_layers, stage_ranks=stage_ranks,
                B=B, S=S, kv_len=kv_len,
                tp_head_splits=partition.tp_head_splits[:partition.tp_size],
                tp_kv_splits=partition.tp_kv_splits[:partition.tp_size],
                tp_ffn_splits=partition.tp_ffn_splits[:partition.tp_size],
                cross_node_tp=partition.tp_cross_node[s],
            )
            stage_times.append(t)
        # In 1F1B steady state, slowest stage dominates.
        t_max_stage = max(stage_times)
        # PP send (between stages); we model one send per step in steady state.
        if partition.pp_size > 1:
            t_send = sum(
                self.pp_send_s(B, S, cross_node=partition.pp_cross_node[s])
                for s in range(partition.pp_size - 1)
            )
        else:
            t_send = 0.0
        return t_max_stage + t_send

    # ---------------- Empirical overhead constants ----------------
    #
    # These three constants absorb effects that aren't in the matmul+AR+send
    # roofline: kernel-launch latency, Python scheduler overhead, paged-attn
    # kernel internals, KV cache writes, NCCL launch per-call cost. Calibrated
    # by fitting the V-curve from `sweep_hetero_pp.py` (Llama-3.3-70B).

    # Fixed cost per PP stage step (CPU-side Python loop + control plane +
    # NCCL launch latency × n_collectives_per_layer × n_layers, amortized).
    # Calibrated against measured wall on Llama-3.3-70B hetero PP sweep at
    # the empirical V-bottom — fitted so model wall ≈ measured wall there.
    PER_STAGE_STEP_OVERHEAD_S: float = 0.020   # 20 ms per stage step

    # Per-layer "extra work" inside one stage forward (layer-norm, RoPE,
    # residual, kv-write). Most time is in matmuls + attention which the
    # roofline already counts; this constant absorbs the residual.
    PER_LAYER_OVERHEAD_S: float = 0.00003      # 30 μs per layer

    # Effective per-layer ratio shrinks vs roofline mem-BW ratio because each
    # stage's AR cost is shared regardless of how many layers it holds. We
    # model this by attenuating the GPU-speed gap: when stage_a is slower than
    # stage_b by ratio r_raw, the EFFECTIVE ratio shrinks toward 1 as AR
    # dominates. The attenuation factor below is calibrated to the measured
    # V-bottom (1.5:1 layer ratio empirically vs 1.85 from raw mem_bw).
    HETERO_RATIO_ATTENUATION: float = 0.80     # effective_ratio = 1 + (raw - 1) * α

    # -------- End-to-end wall time --------
    def wall_time_s(self, partition: "PartitionSpec", workload: "Workload") -> float:
        # Prefill: B=1, S=in_len, kv_len=0 (no cache yet)
        t_prefill = self.step_time_s(partition, B=1, S=workload.in_len, kv_len=0)
        t_prefill = self._add_step_overhead(t_prefill, partition)
        # Number of prefills: each request prefills once, batched 1.
        t_prefill_total = workload.n_requests * t_prefill
        # Decode: B=n_requests / pp_size (1F1B microbatches), S=1, kv_len ≈ in_len + out_len/2 (avg).
        avg_kv_len = workload.in_len + workload.out_len // 2
        n_microbatches = max(1, partition.pp_size)
        B_decode = max(1, workload.n_requests // n_microbatches)
        t_decode_step = self.step_time_s(partition, B=B_decode, S=1, kv_len=avg_kv_len)
        t_decode_step = self._add_step_overhead(t_decode_step, partition)
        # In 1F1B steady state, one decode-token completes per max(stage_time).
        # Total token-emissions needed = out_len × n_microbatches (= n_requests
        # × out_len, the same total throughput as pp=1).
        n_decode_emissions = workload.out_len * n_microbatches
        t_decode_total = n_decode_emissions * t_decode_step
        # Pipeline bubble at fill (and drain at the end of each request batch).
        # With n_microbatches ≥ pp_size, bubble fully amortizes; otherwise the
        # remaining stages remain idle.
        bubble_factor = max(0, partition.pp_size - n_microbatches)
        bubble = bubble_factor * t_decode_step
        return t_prefill_total + t_decode_total + bubble

    def _add_step_overhead(self, step_time_s: float, partition: "PartitionSpec") -> float:
        """Add the per-stage fixed overhead (Python loop + NCCL launch).

        Per-layer overhead is already absorbed into each stage's per-layer
        time via `stage_time_s`; here we only add the once-per-step CPU
        scheduler / NCCL kernel-dispatch cost.
        """
        return step_time_s + self.PER_STAGE_STEP_OVERHEAD_S
