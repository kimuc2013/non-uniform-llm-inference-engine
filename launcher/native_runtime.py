"""Skeleton runtime. M2 implements env composition and dry-run reporting only.

Real server launch + watchdog will be added in M3.
"""
from __future__ import annotations

import shlex
from collections.abc import Iterable

from .config import LaunchConfig


def dry_run_report(cfg: LaunchConfig) -> str:
    """Human-readable summary of a resolved LaunchConfig for the CLI."""
    lines: list[str] = []
    lines.append(f"target               : {cfg.target}")
    lines.append(f"model                : {cfg.model}")
    lines.append(f"tp                   : {cfg.tp}")
    lines.append(f"pp                   : {cfg.pp}")
    lines.append(f"expected world size  : {cfg.expected_world_size}")
    lines.append(f"cluster source       : {cfg.cluster.source_file}")
    lines.append(f"  head               : {cfg.cluster.head_ip}  "
                 f"({cfg.cluster.head_gpus}x {cfg.cluster.head_accel})")
    lines.append(f"  worker             : {cfg.cluster.worker_ip}  "
                 f"({cfg.cluster.worker_gpus}x {cfg.cluster.worker_accel})")
    lines.append(f"  ray address        : {cfg.cluster.ray_address}")

    lines.append("non-uniform TP splits:")
    for k in ("TP_HEAD_SPLITS", "TP_FFN_SPLITS", "TP_KV_SPLITS"):
        v = cfg.env.get(k, "")
        lines.append(f"  {k:18s} = {v}")

    lines.append("vllm-specific env:")
    for k in sorted(cfg.env):
        if k.startswith("VLLM_") or k.startswith("FLASHINFER_"):
            lines.append(f"  {k:36s} = {cfg.env[k]}")

    if cfg.overrides:
        lines.append("CLI overrides:")
        for k in sorted(cfg.overrides):
            lines.append(f"  {k:36s} = {cfg.overrides[k]}")

    return "\n".join(lines)


def build_vllm_argv(cfg: LaunchConfig) -> list[str]:
    """Build the argv that would be passed to python -m vllm.entrypoints.openai.api_server.

    Implemented in M2 so the dry-run can show the exact command. M3 will
    actually invoke it.
    """
    env = cfg.env
    argv: list[str] = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", env["MODEL"],
        "--tensor-parallel-size", str(cfg.tp),
        "--pipeline-parallel-size", str(cfg.pp),
        "--distributed-executor-backend", env.get("EXECUTOR_BACKEND", "ray"),
        "--max-model-len", env.get("MAX_MODEL_LEN", "4096"),
        "--max-num-batched-tokens", env.get("MAX_NUM_BATCHED_TOKENS", "2048"),
        "--max-num-seqs", env.get("MAX_NUM_SEQS", "256"),
        "--gpu-memory-utilization", env.get("GPU_MEMORY_UTILIZATION", "0.9"),
        "--port", env.get("PORT", "28000"),
        "--host", env.get("HOST", "0.0.0.0"),
        "--dtype", env.get("DTYPE", "bfloat16"),
    ]
    if env.get("ENABLE_CHUNKED_PREFILL", "1") == "1":
        argv.append("--enable-chunked-prefill")
    quant = env.get("QUANTIZATION", "").strip()
    if quant:
        argv.extend(["--quantization", quant])
    return argv


def vllm_command_line(cfg: LaunchConfig) -> str:
    """Quoted single-line command for human display."""
    return " ".join(shlex.quote(a) for a in build_vllm_argv(cfg))


def export_block(cfg: LaunchConfig, *, only_vllm_prefix: bool = False) -> str:
    """Render env as bash `export K=V` lines. Useful for reproducing a run."""
    lines: list[str] = []
    keys: Iterable[str] = sorted(cfg.env)
    for k in keys:
        if only_vllm_prefix and not (k.startswith("VLLM_")
                                     or k.startswith("FLASHINFER_")
                                     or k.startswith("NCCL_")):
            continue
        lines.append(f"export {k}={shlex.quote(cfg.env[k])}")
    return "\n".join(lines)
