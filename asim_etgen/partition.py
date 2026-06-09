"""Partition specification for the ET generator.

Supports both uniform AND non-uniform partition along three axes:
  * PP layer split: `layer_splits[s]` = number of layers on stage s
  * TP head split:  `head_splits[r]`  = number of q heads owned by rank r
                                       within the TP group (GQA-aware)
  * TP FFN split:   `ffn_splits[r]`   = intermediate dim owned by rank r
                                       within the TP group

The ET generator reads these to size each rank's COMP nodes appropriately,
so the slow-rank-bottleneck behavior emerges naturally inside ASTRA-sim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class PartitionSpec:
    """Per-cluster partition. tp_size × pp_size == world_size."""
    tp_size: int
    pp_size: int
    layer_splits: List[int]          # length pp_size, sums to num_layers
    head_splits: List[int]           # length tp_size, sums to num_q_heads, GQA-aligned
    kv_splits: List[int]             # length tp_size, sums to num_kv_heads (= head_splits / gqa_group)
    ffn_splits: List[int]            # length tp_size, sums to intermediate
    # Rank → (pp_stage, tp_rank); also rank → node_id for cross-node detection.
    rank_to_node: List[str] = field(default_factory=list)

    @property
    def world_size(self) -> int:
        return self.tp_size * self.pp_size

    def pp_stage_of_rank(self, r: int) -> int:
        return r // self.tp_size

    def tp_rank_of_rank(self, r: int) -> int:
        return r % self.tp_size

    def tp_group_ranks(self, stage: int) -> List[int]:
        """Global rank ids in stage's TP group."""
        return list(range(stage * self.tp_size, (stage + 1) * self.tp_size))

    def is_first_stage(self, stage: int) -> bool:
        return stage == 0

    def is_last_stage(self, stage: int) -> bool:
        return stage == self.pp_size - 1

    def pp_send_to(self, stage: int) -> int | None:
        """Rank to send activation to (tp-rank-0 of next stage)."""
        if self.is_last_stage(stage):
            return None
        return (stage + 1) * self.tp_size  # tp-rank-0 of next stage

    def pp_recv_from(self, stage: int) -> int | None:
        if self.is_first_stage(stage):
            return None
        return (stage - 1) * self.tp_size  # tp-rank-0 of prev stage

    def cross_node_pp(self, stage: int) -> bool:
        if self.is_last_stage(stage) or not self.rank_to_node:
            return False
        my_node = self.rank_to_node[stage * self.tp_size]
        nxt_node = self.rank_to_node[(stage + 1) * self.tp_size]
        return my_node != nxt_node


def uniform_partition(world_size: int, tp_size: int, pp_size: int,
                      num_layers: int, num_q_heads: int, num_kv_heads: int,
                      intermediate: int,
                      rank_to_node: List[str] | None = None) -> PartitionSpec:
    """Trivial uniform split. Useful for sanity checks + baseline."""
    assert tp_size * pp_size == world_size
    base = num_layers // pp_size
    rem = num_layers % pp_size
    layer_splits = [base + (1 if i < rem else 0) for i in range(pp_size)]
    h_per = num_q_heads // tp_size
    kv_per = num_kv_heads // tp_size
    ffn_per = intermediate // tp_size
    return PartitionSpec(
        tp_size=tp_size, pp_size=pp_size,
        layer_splits=layer_splits,
        head_splits=[h_per] * tp_size,
        kv_splits=[kv_per] * tp_size,
        ffn_splits=[ffn_per] * tp_size,
        rank_to_node=rank_to_node or ["node0"] * world_size,
    )
