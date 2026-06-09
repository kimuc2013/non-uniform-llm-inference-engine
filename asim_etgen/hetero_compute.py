"""Hetero-GPU compute support: per-rank scaling of num_ops + tensor_size.

ASTRA-sim's analytical backend uses a SINGLE global peak_perf + local_mem_bw
in the system JSON (one number applies to every NPU). To simulate a cluster
with mixed GPU types, we scale the compute work that each rank is encoded
with: a slow GPU gets the SAME logical work, but with num_ops / tensor_size
multiplied by `global_speed / its_speed`. ASTRA-sim's global roofline then
produces the correct relative time for that rank.

This is mathematically equivalent to using per-NPU peak_perf — and it
preserves ASTRA-sim's cycle-level simulation of compute/comm interleaving
without us monkey-patching the analytical backend.

We define a "reference GPU" whose ops are not scaled (scaling = 1.0). All
other GPU types get a scaling factor based on the *effective* (achievable)
TFLOPS and mem BW for the matmul vs attention regime. Since matmul is the
dominant compute, we use TFLOPS-based scaling for matmul ops and mem-BW
scaling for memory-bound ops (mostly KV-cache reads in attention decode).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .runtime_model import ComputeProfile


@dataclass(frozen=True)
class HeteroSimConfig:
    """Cluster compute spec for ASTRA-sim simulation.

    - `reference`: the GPU whose work is NOT scaled (factor 1.0). The
      system JSON's `peak-perf` and `local-mem-bw` are set to THIS GPU's
      achievable numbers. Convention: pick the fastest GPU in the
      cluster so all scaling factors are ≥ 1.0 (slower GPUs get more
      ops).
    - `gpus_by_rank`: ordered list — gpus_by_rank[r] is the ComputeProfile
      for global rank r.
    """
    reference: ComputeProfile
    gpus_by_rank: list[ComputeProfile]

    def compute_scaling(self, rank: int) -> float:
        """Scale factor applied to num_ops on this rank's COMP nodes.
        Reference GPU → 1.0; slower → > 1.0 (more synthetic work)."""
        ref = self.reference.peak_perf_tflops
        mine = self.gpus_by_rank[rank].peak_perf_tflops
        if mine <= 0:
            return 1.0
        return ref / mine

    def memory_scaling(self, rank: int) -> float:
        """Scale factor applied to tensor_size on this rank's COMP nodes.
        Reference GPU → 1.0; slower → > 1.0 (more bytes-moved synthetic)."""
        ref = self.reference.local_mem_bw_GBs
        mine = self.gpus_by_rank[rank].local_mem_bw_GBs
        if mine <= 0:
            return 1.0
        return ref / mine

    def system_config(self) -> dict:
        """The system JSON values that should be written for ASTRA-sim."""
        return {
            "peak-perf": self.reference.peak_perf_tflops,
            "local-mem-bw": self.reference.local_mem_bw_GBs,
        }
