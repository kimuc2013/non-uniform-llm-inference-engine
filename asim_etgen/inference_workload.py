"""Inference workload ET generator.

Produces one Chakra Execution Trace (.et) per NPU + a comm-group JSON,
ready to feed `ASTRA_Sim_Analytical_*` directly.

Scope of this file
------------------
* Phase 2: a *single forward pass* over (batch, seq) with uniform partition.
* Phase 3 (later): 1F1B microbatching for multi-microbatch overlap.
* Phase 4 (later): non-uniform partition (already piped through PartitionSpec).

The same workload representation handles BOTH prefill and decode iterations:
prefill is a forward pass at (B=1, S=in_len), decode is a forward pass at
(B=microbatch_size, S=1) with kv_len growing across iterations. The composer
on top (the planner / a separate driver) orchestrates how many prefills vs
decode iterations to chain.

Op layout per transformer layer (per rank)
------------------------------------------
1.  input_norm      (COMP)
2.  qkv_proj        (COMP, shard ∝ head_split[r])
3.  attention       (COMP, depends on KV-cache shape)
4.  o_proj          (COMP, shard ∝ head_split[r])
5.  ar_after_o      (COMM_COLL, tp-group)
6.  residual_1      (COMP)
7.  ffn_norm        (COMP)
8.  gate_up         (COMP, shard ∝ ffn_split[r])
9.  silu_mul        (COMP, shard ∝ ffn_split[r])
10. down            (COMP, shard ∝ ffn_split[r])
11. ar_after_down   (COMM_COLL, tp-group)
12. residual_2      (COMP)

At stage boundaries:
   last op of stage s on tp-rank 0 issues a SEND to next stage tp-rank 0.
   first op of stage s+1 on tp-rank 0 issues a RECV.

For Phase 2 we serialize: stage 0 completes -> stage 1 starts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

from .schema import (
    GlobalMetadata, ChakraNode, NodeType,
    COMP_NODE, COMM_SEND_NODE, COMM_RECV_NODE, COMM_COLL_NODE,
    ALL_REDUCE, encode_message,
    attr_int64, attr_uint64, attr_int32, attr_bool, attr_string,
)
from .partition import PartitionSpec
from . import runtime_model as rt
from .hetero_compute import HeteroSimConfig


@dataclass(frozen=True)
class WorkloadSpec:
    """One inference iteration. Either prefill (S>1, B small) or decode.

    `n_microbatches`: how many microbatches to interleave through the
    pipeline (1F1B). Set to 1 for vanilla sequential PP, or pp_size+ for
    pipeline-fill steady-state behavior. ASTRA-sim's scheduler exposes
    the overlap because stage-s+1 ops are gated on the RECV from stage-s,
    not on stage-s+1's own previous microbatch.
    """
    batch: int
    seq: int
    kv_len: int    # current KV-cache token count when this iteration starts
    is_decode: bool
    n_microbatches: int = 1


@dataclass(frozen=True)
class ModelSpec:
    name: str
    num_layers: int
    hidden: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate: int
    # ──────────────────────────────────────────────────────────────────
    # Architecture flags — default to Llama-style for backwards compat.
    # Set mlp_kind="relu_2matmul" for OPT/GPT-style 2-matmul FFN
    # (fc1 → ReLU → fc2) instead of SwiGLU (gate + up → silu·mul → down).
    # ──────────────────────────────────────────────────────────────────
    mlp_kind: str = "swiglu"      # {"swiglu", "relu_2matmul"}
    norm_kind: str = "rmsnorm"    # {"rmsnorm", "layernorm"}
    has_qkv_bias: bool = False     # Qwen-style q/k/v bias term

    @property
    def gqa_group(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    @property
    def is_mha(self) -> bool:
        return self.num_q_heads == self.num_kv_heads


# -------- group id allocation -----------------------------------------------
#
# pg_name in COMM_COLL_NODE matches keys in workload.json. We mint group ids
# as positive integers starting at 1; ASTRA-sim assigns each its rank
# membership from the JSON.
class _GroupTable:
    def __init__(self):
        self.next_id = 1
        self.entries: dict[str, list[int]] = {}    # pg_name → ranks
        self._lookup: dict[tuple[int, ...], str] = {}

    def get_or_make(self, ranks: List[int]) -> str:
        key = tuple(sorted(ranks))
        if key in self._lookup:
            return self._lookup[key]
        name = str(self.next_id)
        self.next_id += 1
        self.entries[name] = list(key)
        self._lookup[key] = name
        return name

    def to_json(self) -> dict:
        return self.entries


# -------- per-rank node-id allocator ----------------------------------------
class _NodeAlloc:
    def __init__(self):
        self.next_id = 0

    def fresh(self) -> int:
        nid = self.next_id
        self.next_id += 1
        return nid


# -------- core builder ------------------------------------------------------

class InferenceWorkloadBuilder:
    """Builds per-NPU Chakra ETs + a comm-group JSON for one inference
    iteration (Phase 2: single microbatch, no 1F1B overlap)."""

    SCHEMA_VERSION = "0.0.4"

    def __init__(self, model: ModelSpec, partition: PartitionSpec,
                 workload: WorkloadSpec,
                 hetero: HeteroSimConfig | None = None):
        """If `hetero` is None, all ranks are treated as the SAME GPU
        (no work scaling). Pass a HeteroSimConfig to simulate mixed-GPU
        clusters via per-rank work scaling (see `hetero_compute.py`).
        """
        self.model = model
        self.partition = partition
        self.workload = workload
        self.hetero = hetero
        self.groups = _GroupTable()
        # one node-id stream per rank
        self.allocs: list[_NodeAlloc] = [_NodeAlloc() for _ in range(partition.world_size)]
        # accumulator: rank → ordered list of ChakraNode messages
        self.nodes: list[list[ChakraNode]] = [[] for _ in range(partition.world_size)]
        # bookkeeping: rank → the id of its previous op (for ctrl chain)
        self.last_id: list[int | None] = [None] * partition.world_size
        # bookkeeping: tag counter for send/recv pair matching
        self.tag_counter = 0
        # pre-register the TP groups (one per PP stage).
        for s in range(partition.pp_size):
            self.groups.get_or_make(partition.tp_group_ranks(s))

    def _scale_for_rank(self, rank: int, num_ops: int, tensor_size: int) -> tuple[int, int]:
        """Apply hetero-compute scaling to a single COMP node."""
        if self.hetero is None:
            return num_ops, tensor_size
        c = self.hetero.compute_scaling(rank)
        m = self.hetero.memory_scaling(rank)
        return int(num_ops * c), int(tensor_size * m)

    # ----- low-level node emitters -----
    def _emit(self, rank: int, name: str, ntype: NodeType,
              data_deps: List[int] | None = None,
              ctrl_deps: List[int] | None = None,
              attrs: List = None) -> int:
        n = ChakraNode()
        n.id = self.allocs[rank].fresh()
        n.name = name
        n.type = ntype
        if data_deps:
            n.data_deps.extend(data_deps)
        if ctrl_deps:
            n.ctrl_deps.extend(ctrl_deps)
        # Chain via ctrl_deps to previous op on the same rank (sequential
        # by default — Phase 3 will relax this for 1F1B).
        if self.last_id[rank] is not None:
            n.ctrl_deps.append(self.last_id[rank])
        if attrs:
            n.attr.extend(attrs)
        self.nodes[rank].append(n)
        self.last_id[rank] = n.id
        return n.id

    def _emit_comp(self, rank: int, name: str, num_ops: int,
                   tensor_size: int, data_deps: List[int] | None = None) -> int:
        scaled_ops, scaled_bytes = self._scale_for_rank(rank, num_ops, tensor_size)
        attrs = [
            attr_bool("is_cpu_op", False),
            attr_int64("num_ops", scaled_ops),
            attr_uint64("tensor_size", scaled_bytes),
            # LocalMemUsageTracker requires these to be present even when
            # we don't track per-tensor IO. Empty strings are fine.
            attr_string("inputs", ""),
            attr_string("outputs", ""),
        ]
        return self._emit(rank, name, COMP_NODE, data_deps=data_deps, attrs=attrs)

    def _emit_allreduce(self, rank: int, name: str, tp_group_id: str,
                        bytes_: int, data_deps: List[int] | None = None) -> int:
        attrs = [
            attr_bool("is_cpu_op", False),
            attr_int64("comm_type", ALL_REDUCE),
            attr_int64("comm_size", bytes_),
            attr_string("pg_name", tp_group_id),
        ]
        return self._emit(rank, name, COMM_COLL_NODE,
                          data_deps=data_deps, attrs=attrs)

    def _emit_send_recv_pair(self, src_rank: int, dst_rank: int,
                             tag: int, bytes_: int, name_base: str,
                             src_data_deps: List[int]) -> tuple[int, int]:
        send_attrs = [
            attr_bool("is_cpu_op", False),
            attr_int64("comm_size", bytes_),
            attr_int32("comm_tag", tag),
            attr_int32("comm_dst", dst_rank),
        ]
        recv_attrs = [
            attr_bool("is_cpu_op", False),
            attr_int64("comm_size", bytes_),
            attr_int32("comm_tag", tag),
            attr_int32("comm_src", src_rank),
        ]
        send_id = self._emit(src_rank, f"{name_base}_SEND",
                             COMM_SEND_NODE, data_deps=src_data_deps,
                             attrs=send_attrs)
        recv_id = self._emit(dst_rank, f"{name_base}_RECV",
                             COMM_RECV_NODE, data_deps=None, attrs=recv_attrs)
        return send_id, recv_id

    # ----- per-layer compute ops -----
    def _layer_ops(self, rank: int, stage: int, layer_idx_global: int,
                   mb: int):
        """Emit one transformer layer's nodes for this rank, tagged with
        microbatch `mb`. Branches by `model.mlp_kind` to handle SwiGLU
        (Llama/Qwen) vs 2-matmul ReLU (OPT/GPT) FFN."""
        m = self.model
        p = self.partition
        w = self.workload
        tp_rank = p.tp_rank_of_rank(rank)
        h_local = p.head_splits[tp_rank]
        kv_local = p.kv_splits[tp_rank]
        ffn_local = p.ffn_splits[tp_rank]
        head_dim = m.head_dim
        hidden = m.hidden
        B, S = w.batch, w.seq
        prefix = f"mb{mb}.L{layer_idx_global}"

        # 1. input_norm (RMSNorm/LayerNorm bytes are similar; LayerNorm
        #    adds a bias read but negligible vs the rest of the layer)
        ops, size = rt.layernorm_flops_bytes(B, S, hidden)
        self._emit_comp(rank, f"{prefix}.input_norm", ops, size)
        # 2. qkv proj
        local_qkv_dim = (h_local + 2 * kv_local) * head_dim
        ops, size = rt.matmul_flops_bytes(B * S, hidden, local_qkv_dim)
        self._emit_comp(rank, f"{prefix}.qkv", ops, size)
        # 3. attention
        ops, size = rt.attention_flops_bytes(B, S, w.kv_len + S, h_local, head_dim)
        self._emit_comp(rank, f"{prefix}.attn", ops, size)
        # 4. o proj
        local_q_dim = h_local * head_dim
        ops, size = rt.matmul_flops_bytes(B * S, local_q_dim, hidden)
        self._emit_comp(rank, f"{prefix}.o_proj", ops, size)
        # 5. AR after o (skip when TP=1 — no group to reduce)
        ar_bytes = B * S * hidden * 2
        if p.tp_size > 1:
            tp_group = self.groups.get_or_make(p.tp_group_ranks(stage))
            self._emit_allreduce(rank, f"{prefix}.ar_o", tp_group, ar_bytes)
        # 6. residual
        ops, size = rt.residual_add_flops_bytes(B, S, hidden)
        self._emit_comp(rank, f"{prefix}.res1", ops, size)
        # 7. ffn_norm
        ops, size = rt.layernorm_flops_bytes(B, S, hidden)
        self._emit_comp(rank, f"{prefix}.ffn_norm", ops, size)
        # 8-10. FFN — branches by architecture
        if m.mlp_kind == "swiglu":
            # gate + up (fused matmul to 2*ffn), silu·mul, down
            ops, size = rt.matmul_flops_bytes(B * S, hidden, 2 * ffn_local)
            self._emit_comp(rank, f"{prefix}.gate_up", ops, size)
            ops, size = rt.silu_mul_flops_bytes(B, S, ffn_local)
            self._emit_comp(rank, f"{prefix}.silu_mul", ops, size)
            ops, size = rt.matmul_flops_bytes(B * S, ffn_local, hidden)
            self._emit_comp(rank, f"{prefix}.down", ops, size)
        elif m.mlp_kind == "relu_2matmul":
            # OPT/GPT style: fc1 (hidden → ffn_local), ReLU (pointwise),
            # fc2 (ffn_local → hidden). One fewer matmul than SwiGLU and
            # no `gate ⊙ silu(up)` elementwise mul.
            ops, size = rt.matmul_flops_bytes(B * S, hidden, ffn_local)
            self._emit_comp(rank, f"{prefix}.fc1", ops, size)
            # ReLU is single-pass pointwise; reuse silu_mul_flops_bytes
            # bound (it counts a similar pointwise cost — small).
            relu_ops = B * S * ffn_local
            relu_bytes = B * S * ffn_local * 2 * 2
            self._emit_comp(rank, f"{prefix}.relu", relu_ops, relu_bytes)
            ops, size = rt.matmul_flops_bytes(B * S, ffn_local, hidden)
            self._emit_comp(rank, f"{prefix}.fc2", ops, size)
        else:
            raise ValueError(f"Unknown mlp_kind={m.mlp_kind!r}")
        # 11. AR after FFN's row-parallel last matmul (skip if TP=1)
        if p.tp_size > 1:
            self._emit_allreduce(rank, f"{prefix}.ar_down", tp_group, ar_bytes)
        # 12. residual
        ops, size = rt.residual_add_flops_bytes(B, S, hidden)
        self._emit_comp(rank, f"{prefix}.res2", ops, size)

    # ----- top-level builder -----
    def build(self) -> None:
        """Emit `n_microbatches` forward passes through the pipeline.

        Topology per microbatch:
            stage 0 layers → PP_SEND (to stage 1 leader) → stage 1 layers → ...

        1F1B overlap is achieved by issuing each microbatch's ops in
        sequence on each rank (rank can only do one thing at a time)
        while making stage s+1's first op on each microbatch DATA-deps on
        the corresponding PP_RECV. ASTRA-sim's scheduler then naturally
        runs stage-0 microbatch k+1 ops concurrently with stage-1
        microbatch k ops, exposing the pipeline overlap.

        Outer loop is microbatches; inner per-stage; per-layer; per-rank.
        Each rank's stream of ops is ordered: mb0 stage_for_this_rank
        ops, then mb1 stage_for_this_rank ops, then ... (sequential per
        rank; the cross-microbatch overlap happens at the stage-boundary
        send/recv).
        """
        m = self.model
        p = self.partition
        w = self.workload

        # Emit ops microbatch by microbatch. Each rank only does work for
        # stage(s) it belongs to; ranks on other stages remain idle
        # (waiting for recv or having sent already).
        for mb in range(w.n_microbatches):
            layer_start = 0
            for s in range(p.pp_size):
                n_layers_here = p.layer_splits[s]
                tp_group_ranks = p.tp_group_ranks(s)

                if not p.is_first_stage(s):
                    src = p.pp_recv_from(s)
                    act_bytes = w.batch * w.seq * m.hidden * 2
                    tag = self._fresh_pp_tag(mb, s)
                    leader = tp_group_ranks[0]
                    # The recv is what gates this stage's microbatch-mb
                    # computation. ASTRA-sim's scheduler runs this recv
                    # concurrently with the prior microbatch's
                    # computation on this stage.
                    self._emit(
                        leader, f"mb{mb}.PP_RECV_s{s}", COMM_RECV_NODE,
                        data_deps=None,
                        attrs=[
                            attr_bool("is_cpu_op", False),
                            attr_int64("comm_size", act_bytes),
                            attr_int32("comm_tag", tag),
                            attr_int32("comm_src", src),
                        ],
                    )

                for li in range(n_layers_here):
                    layer_global = layer_start + li
                    for r in tp_group_ranks:
                        self._layer_ops(r, s, layer_global, mb)

                if not p.is_last_stage(s):
                    dst = p.pp_send_to(s)
                    act_bytes = w.batch * w.seq * m.hidden * 2
                    tag = self._fresh_pp_tag(mb, s + 1)   # paired w/ next stage's recv
                    leader = tp_group_ranks[0]
                    self._emit(
                        leader, f"mb{mb}.PP_SEND_s{s}", COMM_SEND_NODE,
                        data_deps=None,
                        attrs=[
                            attr_bool("is_cpu_op", False),
                            attr_int64("comm_size", act_bytes),
                            attr_int32("comm_tag", tag),
                            attr_int32("comm_dst", dst),
                        ],
                    )

                layer_start += n_layers_here

    def _fresh_pp_tag(self, mb: int, dst_stage: int) -> int:
        """Stable, collision-free tag for a (microbatch, dst_stage) PP
        send/recv pair. Tag encoding: mb * pp_size + dst_stage."""
        return mb * self.partition.pp_size + dst_stage

    # ----- write to disk -----
    def write(self, out_dir: str, name_template: str = "workload.%d.et") -> str:
        """Encode per-rank ETs + workload.json (comm groups). Returns the
        base path (suitable for `--workload-configuration`)."""
        os.makedirs(out_dir, exist_ok=True)
        for r in range(self.partition.world_size):
            fname = os.path.join(out_dir, name_template % r)
            with open(fname, "wb") as f:
                encode_message(f, GlobalMetadata(version=self.SCHEMA_VERSION))
                for n in self.nodes[r]:
                    encode_message(f, n)
        # workload.json: map group_name -> [ranks]
        gj = {name: ranks for name, ranks in self.groups.entries.items()}
        with open(os.path.join(out_dir, "workload.json"), "w") as f:
            json.dump(gj, f)
        # also a quick summary
        with open(os.path.join(out_dir, "workload.summary.txt"), "w") as f:
            f.write(f"model: {self.model.name}\n")
            f.write(f"tp={self.partition.tp_size} pp={self.partition.pp_size}\n")
            f.write(f"layer_splits={self.partition.layer_splits}\n")
            f.write(f"head_splits={self.partition.head_splits}\n")
            f.write(f"ffn_splits={self.partition.ffn_splits}\n")
            f.write(f"workload: B={self.workload.batch} S={self.workload.seq} "
                    f"kv_len={self.workload.kv_len} "
                    f"is_decode={self.workload.is_decode}\n")
            f.write(f"per-rank node counts: "
                    f"{[len(self.nodes[r]) for r in range(self.partition.world_size)]}\n")
        return os.path.join(out_dir, "workload")
