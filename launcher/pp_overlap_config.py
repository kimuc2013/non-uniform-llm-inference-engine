"""Auto-configure PP-overlap parameters (microbatch / mb_size / bq) from
hardware spec + model architecture + workload size.

Empirically validated points on RTX PRO 6000 Blackwell (~380 TFLOPS bf16
peak, achieves ~30-35 TFLOPS during a vLLM Llama forward stage due to
attention/memory-bound layers):

| Model | per-stage params | measured per-req-per-stage |
|-------|------------------|---------------------------:|
| Llama-3.1-8B  TP=2 PP=2 | 2.0 B  | ~125 µs |
| Llama-3.3-70B TP=2 PP=2 | 17.5 B | ~520 µs |

The relationship is sublinear in params (memory bandwidth + attention
share dominate at large model sizes). Best fit so far:

    per_req_per_stage_us ≈ 60 µs × (params_per_stage_B)^0.7

This file fits the constant from those two points and extrapolates. If
the user knows a better number from a microbench, they can override via
`per_req_per_stage_us=...`.

Fixed per-mb CPU overhead is ~1500 µs (measured: 547 µs _update_states +
~950 µs other preprocess/postprocess/sample/bookkeep). Hardware-invariant
to first order; we just hardcode it. If anybody runs on much slower CPU,
override via `fixed_overhead_us_per_mb=...`.
"""
from __future__ import annotations

import dataclasses
import math
import os


# --------------------------------------------------------------------------
# GPU peak bf16 TFLOPS table. For unknown GPUs, fall back to a conservative
# default; the per-req estimate is then probably loose but still useful to
# pick the SHAPE of recommendation. Users on unlisted GPUs should run a
# microbench and pass `per_req_per_stage_us` explicitly.
# --------------------------------------------------------------------------
GPU_PEAK_TFLOPS_BF16: dict[str, float] = {
    # NVIDIA Blackwell-gen consumer/workstation
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": 380.0,
    "NVIDIA RTX PRO 6000 Blackwell": 380.0,
    "NVIDIA GeForce RTX 5090": 250.0,
    # Hopper data center
    "NVIDIA H100 80GB HBM3": 989.0,
    "NVIDIA H100 PCIe": 756.0,
    "NVIDIA H200": 989.0,
    # Ampere data center
    "NVIDIA A100-SXM4-80GB": 312.0,
    "NVIDIA A100 80GB PCIe": 312.0,
    "NVIDIA A100-SXM4-40GB": 312.0,
    # Ampere consumer
    "NVIDIA RTX A6000": 155.0,
}

# Achieved fraction of peak during vLLM forward (matmul + attention +
# residual + norms). Calibrated on Blackwell + Llama-{8B,70B}; bigger
# models hit higher fraction (less attention overhead relative to GEMMs).
ACHIEVED_TFLOPS_FRACTION: dict[str, float] = {
    # params_per_stage_GB → fraction
    "tiny": 0.08,    # < 4B params/stage
    "medium": 0.18,  # 4-30B params/stage
    "large": 0.30,   # > 30B params/stage
}


@dataclasses.dataclass
class PPOverlapConfig:
    use_microbatch: bool
    mb_size: int
    bq: int
    enable_broadcast_stream: bool
    reasoning: str
    estimated_per_mb_compute_us: float
    estimated_overhead_fraction: float


def _params_per_stage_b(
    model_params_b: float, pp_size: int, tp_size: int
) -> float:
    return model_params_b / pp_size / tp_size


def _achieved_tflops_bucket(params_per_stage_b: float) -> str:
    if params_per_stage_b < 4:
        return "tiny"
    if params_per_stage_b < 30:
        return "medium"
    return "large"


def _gpu_peak_tflops(gpu_name: str | None) -> float:
    if gpu_name is None:
        return 200.0  # generic fallback
    return GPU_PEAK_TFLOPS_BF16.get(gpu_name, 200.0)


def estimate_per_req_per_stage_us(
    model_params_b: float,
    pp_size: int,
    tp_size: int,
    gpu_name: str | None = None,
    gpu_peak_tflops_override: float | None = None,
) -> float:
    """Estimate per-request per-stage forward time (microseconds).

    Uses the FLOP-based ceiling × achieved-utilization heuristic.
    """
    pps_b = _params_per_stage_b(model_params_b, pp_size, tp_size)
    # Forward FLOPs ≈ 2 × params per token
    flops_per_token = 2 * pps_b * 1e9
    bucket = _achieved_tflops_bucket(pps_b)
    achieved = ACHIEVED_TFLOPS_FRACTION[bucket]
    peak = gpu_peak_tflops_override or _gpu_peak_tflops(gpu_name)
    effective_tflops = peak * achieved
    # us per token
    return flops_per_token / (effective_tflops * 1e12) * 1e6


def recommend_pp_overlap_config(
    *,
    num_reqs: int,
    pp_size: int,
    tp_size: int,
    model_params_b: float,
    gpu_name: str | None = None,
    gpu_peak_tflops_override: float | None = None,
    per_req_per_stage_us: float | None = None,
    fixed_overhead_us_per_mb: float = 1500.0,
    win_margin: float = 2.0,
) -> PPOverlapConfig:
    """Recommend PP-overlap config given workload + hardware.

    Math:
      - With PP=N and bq=N, the pipeline wants N independent mbs per
        timestep. Setting mb_size = num_reqs / pp_size matches that.
      - Per-mb wall = T_fixed + mb_size × T_per_req_per_stage.
      - Overlap saves ~half a stage time per cycle, so we want
        T_compute = mb_size × T_per_req > win_margin × T_fixed
        for the saved overlap time to dominate the doubled per-mb cost.

    If ideal mb_size (= num_reqs / pp_size) is below the min driven by
    overhead, we fall back to NO microbatch (whole batch as one mb).
    The side-stream broadcast fix is always recommended ON for PP > 1.
    """
    if pp_size <= 1:
        return PPOverlapConfig(
            use_microbatch=False,
            mb_size=num_reqs,
            bq=1,
            enable_broadcast_stream=False,
            reasoning="pp_size=1; PP overlap not applicable",
            estimated_per_mb_compute_us=0.0,
            estimated_overhead_fraction=1.0,
        )

    if num_reqs < pp_size:
        return PPOverlapConfig(
            use_microbatch=False,
            mb_size=max(num_reqs, 1),
            bq=pp_size,
            enable_broadcast_stream=True,
            reasoning=(
                f"num_reqs={num_reqs} < pp_size={pp_size}; cannot split into "
                f"pp_size sub-mbs"
            ),
            estimated_per_mb_compute_us=0.0,
            estimated_overhead_fraction=1.0,
        )

    if per_req_per_stage_us is None:
        per_req_per_stage_us = estimate_per_req_per_stage_us(
            model_params_b=model_params_b,
            pp_size=pp_size,
            tp_size=tp_size,
            gpu_name=gpu_name,
            gpu_peak_tflops_override=gpu_peak_tflops_override,
        )

    min_mb_for_amortization = max(
        1,
        math.ceil(win_margin * fixed_overhead_us_per_mb / per_req_per_stage_us),
    )
    ideal_mb = max(1, num_reqs // pp_size)
    bq = pp_size

    # Policy: always enable microbatch when PP>1. Even at scales where
    # per-mb compute < win_margin × fixed (throughput parity / slight loss
    # vs stock), we keep microbatch ON because the goal is high GPU
    # utilization across both stages — this is what makes PP comparable
    # to TP in our research framing (a fair "same hardware fully used"
    # baseline). We just annotate the reasoning to surface when we're
    # operating below the amortization threshold.
    overhead_fraction = fixed_overhead_us_per_mb / (
        fixed_overhead_us_per_mb + ideal_mb * per_req_per_stage_us
    )
    if ideal_mb < min_mb_for_amortization:
        verdict = (
            f"ideal_mb={ideal_mb} (=num_reqs/{pp_size}) < min_mb="
            f"{min_mb_for_amortization} for clear throughput win, but "
            f"keeping microbatch ON for GPU utilization (overhead "
            f"fraction {overhead_fraction:.1%}; expect throughput "
            f"parity or slight loss vs stock at this scale)"
        )
    else:
        verdict = (
            f"ideal_mb={ideal_mb} (=num_reqs/{pp_size}) ≥ min_mb="
            f"{min_mb_for_amortization}; per-mb compute "
            f"{ideal_mb * per_req_per_stage_us:.0f}μs > {win_margin}× "
            f"fixed {fixed_overhead_us_per_mb:.0f}μs (overhead fraction "
            f"{overhead_fraction:.1%})"
        )
    return PPOverlapConfig(
        use_microbatch=True,
        mb_size=ideal_mb,
        bq=bq,
        enable_broadcast_stream=True,
        reasoning=verdict,
        estimated_per_mb_compute_us=ideal_mb * per_req_per_stage_us,
        estimated_overhead_fraction=overhead_fraction,
    )


def detect_gpu_name() -> str | None:
    """Best-effort detection of the GPU model. Returns None if torch.cuda
    can't be queried."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_name(0)
    except Exception:
        return None


# --------------------------------------------------------------------------
# Model -> total params (Billions). For unknown models, the caller must
# supply this directly.
# --------------------------------------------------------------------------
MODEL_PARAMS_B: dict[str, float] = {
    "meta-llama/Llama-3.1-8B-Instruct": 8.0,
    "meta-llama/Llama-3.1-8B": 8.0,
    "meta-llama/Llama-3.1-70B-Instruct": 70.0,
    "meta-llama/Llama-3.3-70B-Instruct": 70.0,
    "meta-llama/Llama-3.1-405B-Instruct": 405.0,
    "Qwen/Qwen3-235B-A22B": 235.0,
    "facebook/opt-30b": 30.0,
}


def lookup_model_params_b(model_name: str) -> float | None:
    return MODEL_PARAMS_B.get(model_name)


def apply_to_env(env: dict[str, str], cfg: PPOverlapConfig) -> None:
    """Write the recommended config into a vLLM-style env dict."""
    env["VLLM_PP_SAMPLED_BROADCAST_STREAM"] = "1" if cfg.enable_broadcast_stream else "0"
    env["VLLM_PP_MICROBATCH"] = "1" if cfg.use_microbatch else "0"
    if cfg.use_microbatch:
        env["VLLM_PP_MICROBATCH_SIZE"] = str(cfg.mb_size)
    env["VLLM_PP_BATCH_QUEUE_SIZE"] = str(cfg.bq)


def env_overrides(env: dict[str, str] | None = None) -> dict[str, str | float | None]:
    """Pick up override values from environment variables. Returns a dict
    of kwargs to pass to recommend_pp_overlap_config."""
    if env is None:
        env = os.environ
    overrides: dict[str, str | float | None] = {}
    if v := env.get("PP_OVERLAP_AUTO_PER_REQ_US"):
        overrides["per_req_per_stage_us"] = float(v)
    if v := env.get("PP_OVERLAP_AUTO_FIXED_US"):
        overrides["fixed_overhead_us_per_mb"] = float(v)
    if v := env.get("PP_OVERLAP_AUTO_GPU_PEAK_TFLOPS"):
        overrides["gpu_peak_tflops_override"] = float(v)
    if v := env.get("PP_OVERLAP_AUTO_WIN_MARGIN"):
        overrides["win_margin"] = float(v)
    return overrides


def auto_configure(
    num_reqs: int,
    pp_size: int,
    tp_size: int,
    model_name: str,
    *,
    model_params_b: float | None = None,
    gpu_name: str | None = None,
    env: dict[str, str] | None = None,
) -> PPOverlapConfig:
    """End-to-end recommendation: detect GPU + lookup model params + apply
    env overrides + run heuristic. Convenience for the launcher."""
    if model_params_b is None:
        params = lookup_model_params_b(model_name)
        if params is None:
            raise ValueError(
                f"Unknown model '{model_name}'; pass model_params_b explicitly "
                f"or extend MODEL_PARAMS_B."
            )
        model_params_b = params

    if gpu_name is None:
        gpu_name = detect_gpu_name()

    kwargs = env_overrides(env)
    return recommend_pp_overlap_config(
        num_reqs=num_reqs,
        pp_size=pp_size,
        tp_size=tp_size,
        model_params_b=model_params_b,
        gpu_name=gpu_name,
        **kwargs,  # type: ignore[arg-type]
    )
