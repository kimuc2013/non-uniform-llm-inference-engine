"""Custom Chakra ET generator for LLM inference simulation through ASTRA-sim.

Why this module exists
----------------------
STG (Symbolic Tensor Graph), ASTRA-sim's reference workload generator, models
TRAINING workloads (forward + backward + optimizer step, with 1F1B for the
backward sweep). It has no first-class support for LLM INFERENCE (prefill /
decode loops, KV-cache, 1F1B microbatched forward-only). Its
`transformer_inference.py` module is `raise NotImplementedError()`.

To validate non-uniform partition + 1F1B inference on hypothetical clusters,
we generate Chakra Execution Traces from scratch. Each NPU gets its own
per-rank ET that encodes:

  * Per-layer compute ops (`COMP_NODE`) with `num_ops` and `tensor_size`
    appropriate for that rank's TP shard (so ASTRA-sim's roofline runs the
    right amount of work on each rank).
  * Within-stage `COMM_COLL_NODE` AllReduce ops, tagged with `pg_name`
    matching the TP group in the comm-group JSON.
  * Cross-stage `COMM_SEND_NODE` / `COMM_RECV_NODE` with `comm_tag`
    pair-matched by (microbatch_idx, layer_boundary).
  * 1F1B microbatching: multiple microbatches in flight, with `data_deps`
    set so ASTRA-sim's scheduler exposes the pipeline-overlap parallelism
    that uniform sequential PP could not.

ASTRA-sim then performs the cycle-level simulation on top of these ETs +
a network-topology yaml + a system-layer json. Wall time, compute/comm
breakdown, and idle time come from ASTRA-sim, not from us.
"""

from .schema import ChakraNode, ChakraAttr, NodeType, CollectiveCommType
from .runtime_model import ComputeProfile, compute_op_runtime
from .inference_workload import InferenceWorkloadBuilder, WorkloadSpec
from .partition import PartitionSpec

__all__ = [
    "ChakraNode", "ChakraAttr", "NodeType", "CollectiveCommType",
    "ComputeProfile", "compute_op_runtime",
    "InferenceWorkloadBuilder", "WorkloadSpec",
    "PartitionSpec",
]
