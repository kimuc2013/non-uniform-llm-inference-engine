"""Cost-model-based parallelism planner.

Given:
  - cluster spec: per-node GPU profiles (TFLOPS, mem BW) + network BW
  - model spec : Llama-style (num_layers, hidden, q/kv heads, ffn dim)
  - workload   : (input_len, output_len, batch_size) + workload class

Decides:
  - (tp_size, pp_size) parallelism axes
  - PP_LAYER_SPLITS  : per-stage layer counts (non-uniform when stages run
    on different-speed GPUs)
  - TP_HEAD_SPLITS, TP_FFN_SPLITS : per-rank shards (GQA-aware)
  - placement       : which ranks land on which node

Algorithm: enumerate compatible (tp, pp), close-form the optimal partition
inside each, predict wall time via roofline+AllReduce model, re-score with
workload-class weights + structural penalties, pick best.

The planner reads two JSON files written by `tests/calib_compute.py` and
`tests/calib_collective.py`. Both are calibrated once per cluster and reused
across model/workload queries.
"""

from .cost_model import CostModel, GpuProfile, NetworkProfile
from .model_spec import (
    ModelSpec, MODEL_REGISTRY, get_model,
    llama_3_3_70b, llama_3_1_8b, qwen_3_32b, opt_30b,
)
from .planner import Planner, PlanResult, Workload, PartitionSpec, GpuGroup
from .workload import (
    WorkloadClass, WORKLOAD_CLASSES, get_class,
    WorkloadOverride, env_overrides, resolve_workload,
)
from .scorer import (
    score_partition, ScoreBreakdown, predicted_pp_overlap_fraction,
)

__all__ = [
    "CostModel",
    "GpuProfile",
    "NetworkProfile",
    "ModelSpec",
    "MODEL_REGISTRY",
    "get_model",
    "Planner",
    "PlanResult",
    "Workload",
    "PartitionSpec",
    "GpuGroup",
    "WorkloadClass",
    "WORKLOAD_CLASSES",
    "get_class",
    "WorkloadOverride",
    "env_overrides",
    "resolve_workload",
    "score_partition",
    "ScoreBreakdown",
    "predicted_pp_overlap_fraction",
    "llama_3_3_70b",
    "llama_3_1_8b",
    "qwen_3_32b",
    "opt_30b",
]
