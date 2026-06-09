"""Common defaults for the new vLLM 0.21-based launcher.

Env var conventions for this project:

* `VLLM_*` keys are forwarded verbatim to the vllm server (and propagated
  to Ray workers via the vllm `Env var prefixes to copy` list).
* `TP_HEAD_SPLITS`, `TP_FFN_SPLITS`, `TP_KV_SPLITS` are the launcher-side
  names; at compose time they are also exported as `VLLM_TP_HEAD_SPLITS`
  etc. so the patched vllm in vllm_new env picks them up.
* `AUTO_TP_SPLIT`, `AUTOSPLIT`, `AUTO_PP_SPLIT` are launcher-side flags
  consumed by the planner (M5). For M2 they are passthrough toggles.
"""
from __future__ import annotations

from collections.abc import Mapping


# Cluster-shape and serving defaults that are NOT topology-derived.
# Topology (HEAD_IP, WORKER_IP, HEAD_GPUS, WORKER_GPUS) comes from the
# Cluster object (cluster.local.env), not from this file.
COMMON_RUN_DEFAULTS: dict[str, str] = {
    # Parallelism strategy (placeholders; each preset overrides)
    "TP": "1",
    "PP": "1",
    "TP_HEAD_SPLITS": "",
    "TP_FFN_SPLITS": "",
    "TP_KV_SPLITS": "",

    # Launcher behavior
    "AUTO_TP_SPLIT": "0",
    "AUTOSPLIT": "0",
    "AUTO_PP_SPLIT": "0",
    "AUTO_TP_FAIL_POLICY": "abort",   # do NOT silently fall back to uniform

    # Executor and scheduling
    "EXECUTOR_BACKEND": "ray",        # ray | mp
    "ASYNC_SCHEDULING": "0",

    # vLLM 0.21-specific runtime env vars
    "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "1",
    "VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY": "FLASHINFER_",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "NCCL_DEBUG": "WARN",

    # Default OpenAI server binding
    "HOST": "0.0.0.0",
    "PORT": "28000",
}


COMMON_MODEL_DEFAULTS: dict[str, str] = {
    # Serving/runtime limits — each preset can override
    "MODEL": "",
    "DTYPE": "bfloat16",
    "MAX_MODEL_LEN": "4096",
    "MAX_NUM_SEQS": "256",
    "MAX_NUM_BATCHED_TOKENS": "2048",
    "ENABLE_CHUNKED_PREFILL": "1",
    "GPU_MEMORY_UTILIZATION": "0.9",
    "QUANTIZATION": "",
}


# Architectural hints used by the planner. Presets fill these in; they
# are mirrored into env so downstream tools can see them. Not consumed
# by vllm directly.
COMMON_ARCH_KEYS: tuple[str, ...] = (
    "MODEL_FAMILY",          # llama | qwen | opt
    "MODEL_SIZE_LABEL",      # 8b | 32b | 30b | 70b
    "NUM_LAYERS",
    "HIDDEN_SIZE",
    "NUM_Q_HEADS",
    "NUM_KV_HEADS",
    "HEAD_DIM",
    "INTERMEDIATE_SIZE",
    "VOCAB_SIZE",
)


def apply_defaults(env: dict[str, str], defaults: Mapping[str, str]) -> None:
    for k, v in defaults.items():
        env.setdefault(k, v)


def apply_common_defaults(env: dict[str, str]) -> None:
    """Apply common run + model defaults via setdefault (CLI / preset wins)."""
    apply_defaults(env, COMMON_RUN_DEFAULTS)
    apply_defaults(env, COMMON_MODEL_DEFAULTS)
