"""Enumerate (TP, PP) and pick the best partition for a (model, workload) pair.

Given a cluster as a list of (gpu_type, count, node_id) groups, the planner:
  1. Enumerates feasible (tp_size, pp_size) such that tp * pp == total_gpus
  2. For each (tp, pp), assigns rank → GPU type via a placement heuristic
     (TP groups stay intra-node when possible — cross-node TP is expensive)
  3. Derives optimal non-uniform partition analytically:
     - PP_LAYER_SPLITS ∝ 1/per_layer_time(stage_gpu)
     - TP_FFN_SPLITS  ∝ 1/per_matmul_time(rank_gpu)  (with GQA constraint on heads)
  4. Predicts wall time via CostModel and ranks
  5. Returns top-K plans
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .cost_model import CostModel, GpuProfile, NetworkProfile
from .model_spec import ModelSpec


@dataclass(frozen=True)
class Workload:
    in_len: int
    out_len: int
    n_requests: int


@dataclass(frozen=True)
class GpuGroup:
    """A contiguous block of identical GPUs co-located on one node."""
    profile: GpuProfile
    count: int
    node_id: str


@dataclass
class PartitionSpec:
    """Concrete partition spec consumable by the engine."""
    tp_size: int
    pp_size: int
    layer_splits: List[int]                # length pp_size
    tp_head_splits: List[int]              # length tp_size × pp_size? we use length world (per-rank); same value per TP rank across stages
    tp_kv_splits: List[int]                # same
    tp_ffn_splits: List[int]               # same
    stage_rank_groups: List[List[int]]     # ranks belonging to each stage
    tp_cross_node: List[bool]              # per stage
    pp_cross_node: List[bool]              # length pp_size-1 (between consecutive stages)
    rank_to_node: List[str]                # length world

    def env_vars(self) -> dict:
        """Env-var form consumed by the launcher.

        Launcher's native_runtime reads VLLM_-prefixed keys directly into
        the engine env; plain PP_LAYER_SPLITS is silently dropped. Setting
        AUTO_PP_LAYER_PARTITION=0 also pins the explicit partition (otherwise
        the launcher's auto-mode can override the planner's pick when the
        VRAM probe succeeds).
        """
        out = {
            "TP": str(self.tp_size),
            "PP": str(self.pp_size),
            "VLLM_PP_LAYER_PARTITION": ",".join(str(x) for x in self.layer_splits),
            "AUTO_PP_LAYER_PARTITION": "0",
            # CRITICAL: disable the auto-TP helper so it doesn't overwrite the
            # planner-picked FFN/head splits with its own VRAM-weighted ones.
            "AUTO_TP_SPLIT": "0",
            "AUTOSPLIT": "0",
            "TP_HEAD_SPLITS":  ",".join(str(x) for x in self.tp_head_splits[:self.tp_size]),
            "TP_KV_SPLITS":    ",".join(str(x) for x in self.tp_kv_splits[:self.tp_size]),
            "TP_FFN_SPLITS":   ",".join(str(x) for x in self.tp_ffn_splits[:self.tp_size]),
        }
        return out


@dataclass(frozen=True)
class PlanResult:
    partition: PartitionSpec
    predicted_wall_s: float
    rationale: str


class Planner:
    def __init__(self, model: ModelSpec, network: NetworkProfile,
                 cluster: List[GpuGroup]):
        self.model = model
        self.network = network
        self.cluster = cluster
        self.world_size = sum(g.count for g in cluster)

    def plan(self, workload: Workload, top_k: int = 5) -> List[PlanResult]:
        """Enumerate (TP, PP) candidates × multiple partition variants.

        Variants per (TP, PP):
          - "derived_compute": PP layers ∝ per-layer time; TP FFN/head ∝ TFLOPS
          - "derived_memory" : PP layers ∝ per-layer time; TP FFN/head ∝ mem_BW
          - "uniform"        : even layer split, uniform TP shards

        For each, cost model predicts wall; the global minimum is the
        planner's pick. This is robust to workload regime — for decode-heavy
        workloads the memory-biased variant wins, for prefill-heavy the
        compute-biased one wins, and at low concurrency the uniform variant
        wins (preventing pathological hetero-bias picks).
        """
        candidates: List[PlanResult] = []
        for tp_size, pp_size in self._enumerate_parallelism():
            for variant in ("derived_compute", "derived_memory", "uniform"):
                try:
                    p = self._build_partition(tp_size, pp_size, workload,
                                              variant=variant)
                    cm = CostModel(
                        model=self.model,
                        network=self.network,
                        gpu_per_rank=tuple(self._gpu_for_rank(p.rank_to_node)),
                    )
                    t = cm.wall_time_s(p, workload)
                    candidates.append(PlanResult(
                        partition=p,
                        predicted_wall_s=t,
                        rationale=self._explain(p, t, variant),
                    ))
                except (ValueError, AssertionError):
                    continue
        candidates.sort(key=lambda r: r.predicted_wall_s)
        return candidates[:top_k]

    def _enumerate_parallelism(self) -> List[Tuple[int, int]]:
        """All (tp, pp) such that tp * pp == world_size and tp divides model heads."""
        W = self.world_size
        out = []
        for tp in (1, 2, 4, 8, 16):
            if W % tp != 0:
                continue
            pp = W // tp
            if pp == 0:
                continue
            # tp must divide num_kv_heads (each rank needs ≥1 KV head).
            if self.model.num_kv_heads % tp != 0 and tp > 1:
                continue
            # tp must divide num_q_heads (so uniform is feasible at least).
            if self.model.num_q_heads % tp != 0:
                continue
            # pp must divide num_layers (uniform layer split feasibility).
            if self.model.num_layers % pp != 0 and pp > 1:
                # Non-uniform can still work; require sum(splits)==num_layers.
                pass
            out.append((tp, pp))
        return out

    def _build_partition(self, tp_size: int, pp_size: int,
                        workload: Workload, variant: str = "derived_compute") -> PartitionSpec:
        force_uniform = (variant == "uniform")
        # 1) Assign ranks to nodes via "TP intra-node first" heuristic.
        rank_to_node = self._assign_ranks_to_nodes(tp_size, pp_size)

        # 2) Decide per-stage GPU profile and per-stage TP locality.
        stage_rank_groups: List[List[int]] = []
        tp_cross_node: List[bool] = []
        for s in range(pp_size):
            group = list(range(s * tp_size, (s + 1) * tp_size))
            stage_rank_groups.append(group)
            tp_cross_node.append(len({rank_to_node[r] for r in group}) > 1)

        pp_cross_node = [
            rank_to_node[stage_rank_groups[s][0]] != rank_to_node[stage_rank_groups[s+1][0]]
            for s in range(pp_size - 1)
        ]

        # 3) Derive optimal non-uniform PP layer split (closed form).
        # T_stage(s) ∝ N_layers(s) × c_s.  Minimize max ⇒ N_layers ∝ 1/c_s.
        if force_uniform:
            # Even split with remainder absorbed by front stages.
            base = self.model.num_layers // pp_size
            rem = self.model.num_layers % pp_size
            layer_splits = [base + (1 if i < rem else 0) for i in range(pp_size)]
        else:
            per_layer_compute_s = self._per_layer_compute_cost(
                stage_rank_groups, rank_to_node, tp_size, workload)
            inv = [1.0 / c if c > 0 else 1.0 for c in per_layer_compute_s]
            total_inv = sum(inv)
            ideal_layers = [self.model.num_layers * x / total_inv for x in inv]
            layer_splits = self._round_layer_splits(ideal_layers, self.model.num_layers, pp_size)

        # 4) Derive TP head/KV/FFN splits.  Uniform attention for now (head
        # imbalance requires GQA-multiple-of-group constraint which is tight at
        # high TP).  FFN imbalance: ∝ 1/per_matmul_time(rank_gpu).
        if force_uniform:
            head_per = self.model.num_q_heads // tp_size
            kv_per = self.model.num_kv_heads // tp_size
            ffn_per = self.model.intermediate_size // tp_size
            tp_head_splits = [head_per] * tp_size
            tp_kv_splits = [kv_per] * tp_size
            tp_ffn_splits = [ffn_per] * tp_size
        else:
            bias_axis = "mem_bw" if variant == "derived_memory" else "tflops"
            tp_head_splits, tp_kv_splits, tp_ffn_splits = self._derive_tp_splits(
                tp_size=tp_size, pp_size=pp_size,
                rank_to_node=rank_to_node, workload=workload, bias_axis=bias_axis,
            )

        return PartitionSpec(
            tp_size=tp_size, pp_size=pp_size,
            layer_splits=layer_splits,
            tp_head_splits=tp_head_splits,
            tp_kv_splits=tp_kv_splits,
            tp_ffn_splits=tp_ffn_splits,
            stage_rank_groups=stage_rank_groups,
            tp_cross_node=tp_cross_node,
            pp_cross_node=pp_cross_node,
            rank_to_node=rank_to_node,
        )

    def _assign_ranks_to_nodes(self, tp_size: int, pp_size: int) -> List[str]:
        """Pack ranks into nodes, keeping TP groups intra-node where possible."""
        nodes = []
        for g in self.cluster:
            nodes.extend([g.node_id] * g.count)
        assert len(nodes) >= self.world_size, "not enough GPUs in cluster"
        return nodes[:self.world_size]

    def _per_layer_compute_cost(self, stage_rank_groups, rank_to_node, tp_size,
                                workload: Workload) -> List[float]:
        """Per-layer EFFECTIVE time (compute + AR + per-layer overhead) on each stage.

        Used to derive the closed-form optimal PP layer split. We include
        per-layer AllReduce AND the constant per-layer overhead because both
        are GPU-symmetric — they appear identically on every stage and thus
        dampen the per-stage time ratio toward 1. Excluding them would yield
        partitions biased excessively toward the fast GPU (e.g., raw matmul
        ratio 3.5× → partition [56, 24], but empirically the effective
        per-layer ratio is ~1.5 because comm + overhead are ~half the cost).
        """
        from .cost_model import CostModel
        gpu_per_rank = self._gpu_for_rank(rank_to_node)
        # Decode regime is the typical bottleneck for LLM serving (decode wall
        # >> prefill wall when out_len > 1). Compute per-layer cost at the
        # workload's microbatch B = n_req / pp_size.
        per_rank_q = self.model.num_q_heads // tp_size
        per_rank_kv = self.model.num_kv_heads // tp_size
        per_rank_ffn = self.model.intermediate_size // tp_size
        pp_size = len(stage_rank_groups)
        B = max(1, workload.n_requests // max(1, pp_size))
        S = 1
        kv_len = workload.in_len + workload.out_len // 2
        cm = CostModel(model=self.model, network=self.network,
                       gpu_per_rank=tuple(gpu_per_rank))
        costs = []
        for s, group in enumerate(stage_rank_groups):
            t_compute = max(
                cm.per_layer_compute_s(r, B, S, per_rank_q, per_rank_kv,
                                       per_rank_ffn, kv_len)
                for r in group
            )
            cross_node = len({rank_to_node[r] for r in group}) > 1
            t_ar = cm.per_layer_allreduce_s(B, S, tp_size, cross_node)
            t_per_layer_overhead = cm.PER_LAYER_OVERHEAD_S
            costs.append(t_compute + t_ar + t_per_layer_overhead)
        return costs

    def _round_layer_splits(self, ideal: List[float], total: int,
                           pp_size: int) -> List[int]:
        """Round float layer counts to integers summing to total, each ≥1."""
        rounded = [max(1, int(round(x))) for x in ideal]
        diff = total - sum(rounded)
        # Adjust by walking residuals (largest fractional first).
        residuals = sorted(
            range(pp_size),
            key=lambda i: ideal[i] - int(ideal[i]),
            reverse=(diff > 0),
        )
        i = 0
        while diff != 0 and i < 1000:
            idx = residuals[i % pp_size]
            if diff > 0:
                rounded[idx] += 1; diff -= 1
            else:
                if rounded[idx] > 1:
                    rounded[idx] -= 1; diff += 1
            i += 1
        return rounded

    def _derive_tp_splits(self, tp_size, pp_size, rank_to_node,
                         workload: Workload,
                         bias_axis: str = "tflops") -> tuple[list[int], list[int], list[int]]:
        """Compute per-rank head/KV/FFN splits.

        TP group is "heterogeneous" if its ranks have DIFFERENT GPU profiles
        (different `gpu.name`). This is independent of node placement — the
        GPUs can be in the same chassis but be different SKUs (e.g., a server
        with 2×H100 + 2×A100). Conversely, homogeneous GPUs in different nodes
        still allow uniform TP (just intra-rack vs intra-chassis difference).

        For hetero TP groups:
          - FFN: bias intermediate dim ∝ rank's GPU TFLOPS (no GQA constraint)
          - Head: bias head count ∝ rank's GPU TFLOPS, snapped to GQA multiples
                  (each head_split must be a multiple of num_q_heads/num_kv_heads)
        """
        # Default uniform.
        head_per = self.model.num_q_heads // tp_size
        kv_per = self.model.num_kv_heads // tp_size
        ffn_per = self.model.intermediate_size // tp_size
        tp_head = [head_per] * tp_size
        tp_kv = [kv_per] * tp_size
        tp_ffn = [ffn_per] * tp_size

        # Check first stage's TP group for *GPU-type* heterogeneity.
        stage0_ranks = list(range(tp_size))
        gpu_per_rank = self._gpu_for_rank(rank_to_node)
        gpu_types_in_stage0 = {gpu_per_rank[r].name for r in stage0_ranks}
        if len(gpu_types_in_stage0) <= 1:
            # All ranks in TP group are same GPU type — uniform is best.
            return tp_head, tp_kv, tp_ffn

        # Hetero TP group: bias FFN AND (GQA-permitting) head splits.
        #
        # The "speed" measure depends on whether the workload is compute-
        # bound or memory-bound. Since this varies with workload (and the
        # planner doesn't pre-decide), the outer plan() loop tries both
        # bias_axis values and picks whichever yields lower predicted wall.
        #   - bias_axis="tflops" : rank gets FFN ∝ TFLOPS (good for prefill)
        #   - bias_axis="mem_bw" : rank gets FFN ∝ mem_bw (good for decode)
        if bias_axis == "mem_bw":
            speed_per = [gpu_per_rank[r].mem_bw_GBs for r in stage0_ranks]
        else:
            speed_per = [gpu_per_rank[r].tflops_prefill for r in stage0_ranks]
        total_speed = sum(speed_per)

        # FFN: bias by GPU speed, but snap each shard to a tile-aligned
        # multiple (Tensor Core kernels prefer multiples of 64/128). We use
        # FFN_TILE=128 since SwiGLU's gate/up/down all run as row/col-parallel
        # matmuls whose K or N is the local FFN slice — sub-128 multiples
        # incur padding and wasted FLOPS on Blackwell/H100 class GPUs.
        FFN_TILE = 128
        if self.model.intermediate_size % FFN_TILE != 0:
            # Model's intermediate isn't tile-aligned (rare); fall back to
            # smaller tile that divides it.
            FFN_TILE = 64 if self.model.intermediate_size % 64 == 0 else 8
        ideal_ffn = [self.model.intermediate_size * f / total_speed for f in speed_per]
        n_tiles_total = self.model.intermediate_size // FFN_TILE
        # Hamilton's largest-remainder on tile units, then multiply back.
        raw_tiles = [n_tiles_total * f / total_speed for f in speed_per]
        floor = [int(x) for x in raw_tiles]
        rem = [r - f for r, f in zip(raw_tiles, floor)]
        diff_tiles = n_tiles_total - sum(floor)
        if diff_tiles > 0:
            order = sorted(range(tp_size), key=lambda i: -rem[i])
            for i in order[:diff_tiles]:
                floor[i] += 1
        elif diff_tiles < 0:
            order = sorted(range(tp_size), key=lambda i: rem[i])
            for i in order[:-diff_tiles]:
                if floor[i] > 1:
                    floor[i] -= 1
        # Ensure ≥1 tile per rank
        while any(t < 1 for t in floor):
            lo = floor.index(min(floor)); hi = floor.index(max(floor))
            if floor[hi] <= 1:
                break
            floor[lo] += 1; floor[hi] -= 1
        tp_ffn = [t * FFN_TILE for t in floor]
        # Sanity: sum must match.
        if sum(tp_ffn) != self.model.intermediate_size:
            # Could not satisfy alignment + sum; fall back to uniform FFN.
            tp_ffn = [self.model.intermediate_size // tp_size] * tp_size

        # Head: GQA-aware. Each head_split must be a multiple of
        # gqa_group = num_q_heads / num_kv_heads. We snap each ideal value
        # to the nearest GQA-multiple, then fix-up the sum.
        gqa_group = self.model.gqa_group
        ideal_heads = [self.model.num_q_heads * f / total_speed for f in speed_per]
        snapped = [max(gqa_group, gqa_group * round(x / gqa_group)) for x in ideal_heads]
        # Fix-up so sum matches num_q_heads (each adjustment is one gqa_group).
        diff_heads = self.model.num_q_heads - sum(snapped)
        while diff_heads != 0:
            # Pick the index with the largest discrepancy in the desired direction.
            if diff_heads > 0:
                # Need to ADD heads; prefer fastest rank still below ideal.
                idx = max(range(tp_size),
                          key=lambda i: ideal_heads[i] - snapped[i])
                snapped[idx] += gqa_group
                diff_heads -= gqa_group
            else:
                # Need to REMOVE heads; prefer slowest rank still above ideal,
                # but never drop below gqa_group.
                candidates = [i for i in range(tp_size) if snapped[i] > gqa_group]
                if not candidates:
                    break
                idx = min(candidates, key=lambda i: ideal_heads[i] - snapped[i])
                snapped[idx] -= gqa_group
                diff_heads += gqa_group
        # Strict: if we couldn't satisfy GQA + sum, this variant is invalid;
        # raise so the outer plan() loop skips it instead of silently
        # delivering a "biased" plan that is actually uniform-head.
        if sum(snapped) != self.model.num_q_heads:
            raise ValueError(
                f"GQA-snapped head split couldn't sum to num_q_heads "
                f"({self.model.num_q_heads}) — variant rejected.")
        if any(h <= 0 or h % gqa_group != 0 for h in snapped):
            raise ValueError(
                f"GQA-snapped head split has invalid entries {snapped} — "
                "variant rejected.")
        tp_head = snapped

        # KV: derive from heads (each rank gets head/gqa_group KV heads).
        tp_kv = [h // gqa_group for h in tp_head]
        return tp_head, tp_kv, tp_ffn

    def _gpu_for_rank(self, rank_to_node: List[str]) -> List[GpuProfile]:
        """Sequential rank → GpuProfile mapping.

        Iterates `cluster` in order and expands each group into per-rank
        slots. This preserves multiple GpuGroups on the SAME node_id (e.g.,
        a host containing both H100 and A100 GPUs), which a node-keyed dict
        would collapse to whichever group came last.
        """
        sequential: List[GpuProfile] = []
        for g in self.cluster:
            sequential.extend([g.profile] * g.count)
        # rank_to_node is just an alignment artifact at this point — the
        # i-th rank corresponds to the i-th GPU in the flattened cluster.
        return sequential[:len(rank_to_node)]

    def _explain(self, p: PartitionSpec, t: float, variant: str = "derived") -> str:
        parts = [
            f"tp={p.tp_size} pp={p.pp_size}",
            f"layers={p.layer_splits}",
        ]
        if any(x != p.tp_ffn_splits[0] for x in p.tp_ffn_splits[:p.tp_size]):
            parts.append(f"ffn={p.tp_ffn_splits[:p.tp_size]}")
        if any(p.tp_cross_node):
            parts.append("tp-cross-node")
        if variant == "uniform":
            parts.append("(uniform)")
        return " ".join(parts) + f"  pred={t*1000:.0f}ms"
