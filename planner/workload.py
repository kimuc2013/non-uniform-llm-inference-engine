"""Workload-aware planner enhancement.

Adds three workload *classes* on top of the existing
`Workload(in_len, out_len, n_requests)` dataclass:

  * `decode_heavy`  — short input, long output. Wall dominated by decode
                      steps. Weight decode tput / per-token-latency / TP
                      AllReduce; PP overlap is hard to amortize so PP often
                      hurts.
  * `prefill_heavy` — long input, short output. Wall dominated by prefill
                      GEMMs. Weight stage utilization / large-M efficiency;
                      PP and hybrid are competitive because each prefill
                      microbatch is big enough to overlap.
  * `balanced`      — medium input, medium output. Weight both.

The class influences the cost-model SCORE (not the predicted wall directly):
we score every candidate plan with a weighted combination of:

    cost = w_prefill * t_prefill_total + w_decode * t_decode_total
         + penalty_terms

The cost model already separately computes prefill and decode wall (see
`CostModel.wall_time_s`). We re-implement a thin scorer here that calls the
same primitives but emits the components individually so we can re-weight.

Penalty terms (cheap structural penalties, deterministic, no benchmarking):

  - cross-node TP per stage:   +25% on that stage's per-layer comm time
  - cross-node PP send:        +50% on PP send between those stages
  - slow-rank dominance:       +stage_imbalance * stage_time
  - KV / activation memory:    if peak_mem > vram, FAIL (mark infeasible)
  - per-mb fixed overhead:     workload-class-tunable (decode_heavy
                                 penalizes deeper PP harder; prefill_heavy
                                 amortizes faster)

Env-var overrides (so an external benchmark loop can re-rank without
rebuilding the planner):

  * VLLM_PLANNER_WORKLOAD          = decode_heavy | prefill_heavy | balanced
  * VLLM_PLANNER_INPUT_LEN         = int   (override workload.in_len)
  * VLLM_PLANNER_OUTPUT_LEN        = int   (override workload.out_len)
  * VLLM_PLANNER_NUM_PROMPTS       = int
  * VLLM_PLANNER_CONCURRENT_REQUESTS = int  (alias for n_requests)
  * VLLM_PLANNER_PREFILL_WEIGHT    = float (manual override of w_prefill)
  * VLLM_PLANNER_DECODE_WEIGHT     = float (manual override of w_decode)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Optional

from .planner import Workload


# ---------------------------------------------------------------------------
# Workload class registry. Each class is just a triple
# (default in_len, default out_len, default weights). The weight pair
# influences scoring; in_len/out_len are defaults if the user doesn't pass
# explicit values.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WorkloadClass:
    name: str
    in_len: int                       # representative input tokens
    out_len: int                      # representative output tokens
    n_requests: int                   # representative concurrency
    w_prefill: float                  # multiplicative weight on prefill wall
    w_decode: float                   # multiplicative weight on decode wall
    pp_depth_penalty: float           # extra cost per PP stage beyond 1
    tp_comm_weight: float             # multiplier on per-layer AR cost
    description: str = ""


# Calibrated defaults. The weights aren't meant to predict absolute wall —
# they shift the *ranking* of candidates toward the regime appropriate to
# the workload class. E.g., for decode_heavy, we slightly inflate
# tp_comm_weight (because decode is comm-sensitive) and add a small PP
# depth penalty (each extra PP stage adds a per-mb broadcast cost the
# decode wall amplifies).
WORKLOAD_CLASSES: dict[str, WorkloadClass] = {
    "decode_heavy": WorkloadClass(
        name="decode_heavy",
        in_len=128, out_len=1024, n_requests=64,
        w_prefill=0.5, w_decode=1.5,
        pp_depth_penalty=0.08,
        tp_comm_weight=1.15,
        description="Short input, long output. Decode wall dominates; TP "
                    "preferred; PP overhead amortizes poorly.",
    ),
    "prefill_heavy": WorkloadClass(
        name="prefill_heavy",
        in_len=2048, out_len=128, n_requests=32,
        w_prefill=1.5, w_decode=0.5,
        pp_depth_penalty=0.02,
        tp_comm_weight=0.90,
        description="Long input, short output. Prefill GEMMs dominate; PP "
                    "and hybrid competitive because each mb is large.",
    ),
    "balanced": WorkloadClass(
        name="balanced",
        in_len=512, out_len=256, n_requests=64,
        w_prefill=1.0, w_decode=1.0,
        pp_depth_penalty=0.04,
        tp_comm_weight=1.00,
        description="Medium input, medium output. Both regimes considered.",
    ),
}


def get_class(name: str) -> WorkloadClass:
    if name not in WORKLOAD_CLASSES:
        raise KeyError(
            f"Unknown workload class {name!r}. "
            f"Available: {sorted(WORKLOAD_CLASSES)}"
        )
    return WORKLOAD_CLASSES[name]


# ---------------------------------------------------------------------------
# Env-var resolution. Allows external scripts (run_planner_eval.py, bash) to
# drive the planner without code changes.
# ---------------------------------------------------------------------------
@dataclass
class WorkloadOverride:
    workload_class: Optional[str] = None
    in_len: Optional[int] = None
    out_len: Optional[int] = None
    n_requests: Optional[int] = None
    num_prompts: Optional[int] = None      # alias used in prompts
    w_prefill: Optional[float] = None
    w_decode: Optional[float] = None


def env_overrides(env: dict[str, str] | None = None) -> WorkloadOverride:
    """Collect planner overrides from environment variables. Returns a
    WorkloadOverride with only the set fields populated."""
    e = env if env is not None else os.environ
    ov = WorkloadOverride()
    if v := e.get("VLLM_PLANNER_WORKLOAD"):
        ov.workload_class = v.strip()
    if v := e.get("VLLM_PLANNER_INPUT_LEN"):
        ov.in_len = int(v)
    if v := e.get("VLLM_PLANNER_OUTPUT_LEN"):
        ov.out_len = int(v)
    if v := e.get("VLLM_PLANNER_NUM_PROMPTS"):
        ov.num_prompts = int(v)
    if v := e.get("VLLM_PLANNER_CONCURRENT_REQUESTS"):
        ov.n_requests = int(v)
    if v := e.get("VLLM_PLANNER_PREFILL_WEIGHT"):
        ov.w_prefill = float(v)
    if v := e.get("VLLM_PLANNER_DECODE_WEIGHT"):
        ov.w_decode = float(v)
    return ov


# ---------------------------------------------------------------------------
# Build a Workload + WorkloadClass from (defaults, overrides).
# ---------------------------------------------------------------------------
def resolve_workload(
    *,
    workload_class: str = "balanced",
    in_len: Optional[int] = None,
    out_len: Optional[int] = None,
    n_requests: Optional[int] = None,
    override: Optional[WorkloadOverride] = None,
) -> tuple[Workload, WorkloadClass]:
    """Resolve (class, in_len, out_len, n_requests) from explicit args + env."""
    if override is None:
        override = env_overrides()
    cls_name = override.workload_class or workload_class
    cls = get_class(cls_name)
    final_in = override.in_len if override.in_len is not None else (
        in_len if in_len is not None else cls.in_len)
    final_out = override.out_len if override.out_len is not None else (
        out_len if out_len is not None else cls.out_len)
    final_nreq = override.n_requests if override.n_requests is not None else (
        override.num_prompts if override.num_prompts is not None else (
            n_requests if n_requests is not None else cls.n_requests))
    # Apply weight overrides
    if override.w_prefill is not None or override.w_decode is not None:
        cls = replace(
            cls,
            w_prefill=override.w_prefill if override.w_prefill is not None else cls.w_prefill,
            w_decode=override.w_decode if override.w_decode is not None else cls.w_decode,
        )
    return Workload(in_len=final_in, out_len=final_out, n_requests=final_nreq), cls


__all__ = [
    "WorkloadClass",
    "WORKLOAD_CLASSES",
    "get_class",
    "WorkloadOverride",
    "env_overrides",
    "resolve_workload",
]
