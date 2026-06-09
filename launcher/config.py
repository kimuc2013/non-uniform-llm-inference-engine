"""Resolve a (preset, cluster, CLI overrides) tuple into a LaunchConfig.

Composition order (later wins for keys that ARE set; earlier wins for
keys still unset because preset functions use setdefault):

    1. common defaults                                  (presets/common.py)
    2. preset                                           (presets/<name>.py)
    3. cluster-derived defaults (HEAD_IP, ...)          (Cluster object)
    4. CLI --set KEY=VALUE overrides                    (explicit user wins)

Each preset uses env.setdefault(...), so a higher-priority layer can
also be implemented by inserting BEFORE calling the preset (CLI
overrides applied to env BEFORE preset run => preset cannot clobber
explicit user choice).

LaunchConfig.env is the final dict that downstream tooling consumes.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from .cluster import Cluster
from .presets import REGISTRY, ALL_TARGETS, apply_common_defaults


@dataclass
class LaunchConfig:
    target: str
    cluster: Cluster
    env: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, str] = field(default_factory=dict)
    cli_invocation: list[str] = field(default_factory=list)

    @property
    def model(self) -> str:
        return self.env.get("MODEL", "")

    @property
    def tp(self) -> int:
        return int(self.env.get("TP", "1"))

    @property
    def pp(self) -> int:
        return int(self.env.get("PP", "1"))

    @property
    def expected_world_size(self) -> int:
        return self.tp * self.pp

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "model": self.model,
            "tp": self.tp,
            "pp": self.pp,
            "expected_world_size": self.expected_world_size,
            "cluster": self.cluster.to_dict(),
            "env": dict(sorted(self.env.items())),
            "overrides": dict(sorted(self.overrides.items())),
            "cli_invocation": self.cli_invocation,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def _apply_cluster_to_env(env: dict[str, str], cluster: Cluster) -> None:
    """Inject cluster topology as env vars. setdefault — explicit overrides win."""
    env.setdefault("HEAD_IP", cluster.head_ip)
    env.setdefault("WORKER_IP", cluster.worker_ip)
    env.setdefault("HEAD_GPUS", str(cluster.head_gpus))
    env.setdefault("WORKER_GPUS", str(cluster.worker_gpus))
    env.setdefault("HEAD_ACCEL", cluster.head_accel)
    env.setdefault("WORKER_ACCEL", cluster.worker_accel)
    env.setdefault("GCS_PORT", str(cluster.gcs_port))
    env.setdefault("RAY_ADDRESS", cluster.ray_address)
    env.setdefault("WORKER_SSH_PORT", str(cluster.worker_ssh_port))
    if cluster.head_py:
        env.setdefault("HEAD_PY", cluster.head_py)
    if cluster.worker_py:
        env.setdefault("WORKER_PY", cluster.worker_py)
    env.setdefault("TEMPDIR", cluster.tempdir)


def _mirror_vllm_prefix(env: dict[str, str]) -> None:
    """Mirror TP_*_SPLITS -> VLLM_TP_*_SPLITS so the patched vllm reads them.

    The patched vllm (M3) reads VLLM_TP_HEAD_SPLITS / FFN / KV. The launcher
    uses the un-prefixed names internally and mirrors them so users can set
    either form via --set.
    """
    for src in ("TP_HEAD_SPLITS", "TP_FFN_SPLITS", "TP_KV_SPLITS"):
        val = env.get(src, "")
        if val:
            env.setdefault(f"VLLM_{src}", val)


def _validate(cfg: LaunchConfig) -> list[str]:
    """Return a list of error strings; empty if config is valid."""
    errs: list[str] = []
    env = cfg.env

    # World size matches cluster GPUs?
    expected = cfg.expected_world_size
    available = cfg.cluster.total_gpus()
    if expected > available:
        errs.append(
            f"TP×PP = {expected} exceeds cluster GPUs {available} "
            f"(head={cfg.cluster.head_gpus} + worker={cfg.cluster.worker_gpus})"
        )

    # TP splits length matches TP?
    tp = cfg.tp
    for k in ("TP_HEAD_SPLITS", "TP_FFN_SPLITS", "TP_KV_SPLITS"):
        v = env.get(k, "").strip()
        if not v:
            continue
        try:
            parts = [int(x) for x in v.split(",")]
        except ValueError:
            errs.append(f"{k}={v!r} is not a CSV of ints")
            continue
        # For PP>1, splits should cover tp per stage, not tp*pp.
        # Old vllm_hetero convention: len == TP. Keep that for now.
        if len(parts) != tp:
            errs.append(
                f"{k} has {len(parts)} entries but TP={tp} "
                f"(expected one entry per TP rank)"
            )
        if any(p <= 0 for p in parts):
            errs.append(f"{k} has non-positive entries: {parts}")

    # GQA consistency: per-rank Q/KV ratio must match global ratio
    head_splits = env.get("TP_HEAD_SPLITS", "").strip()
    kv_splits = env.get("TP_KV_SPLITS", "").strip()
    q_total = int(env.get("NUM_Q_HEADS", "0"))
    kv_total = int(env.get("NUM_KV_HEADS", "0"))
    if head_splits and kv_splits and q_total and kv_total:
        try:
            hs = [int(x) for x in head_splits.split(",")]
            ks = [int(x) for x in kv_splits.split(",")]
            if len(hs) == len(ks):
                for i, (q_i, k_i) in enumerate(zip(hs, ks)):
                    if q_i * kv_total != k_i * q_total:
                        errs.append(
                            f"GQA broken at rank {i}: "
                            f"q_i={q_i}, kv_i={k_i}, global q:kv={q_total}:{kv_total} "
                            f"(need q_i*{kv_total} == kv_i*{q_total}, "
                            f"got {q_i*kv_total} != {k_i*q_total})"
                        )
        except ValueError:
            pass  # CSV error already reported

    # Sums match totals
    ffn_splits = env.get("TP_FFN_SPLITS", "").strip()
    inter = int(env.get("INTERMEDIATE_SIZE", "0"))
    if ffn_splits and inter:
        try:
            parts = [int(x) for x in ffn_splits.split(",")]
            if sum(parts) != inter:
                errs.append(
                    f"TP_FFN_SPLITS sum {sum(parts)} != INTERMEDIATE_SIZE {inter}"
                )
        except ValueError:
            pass

    if head_splits and q_total:
        try:
            parts = [int(x) for x in head_splits.split(",")]
            if sum(parts) != q_total:
                errs.append(
                    f"TP_HEAD_SPLITS sum {sum(parts)} != NUM_Q_HEADS {q_total}"
                )
        except ValueError:
            pass

    if kv_splits and kv_total:
        try:
            parts = [int(x) for x in kv_splits.split(",")]
            if sum(parts) != kv_total:
                errs.append(
                    f"TP_KV_SPLITS sum {sum(parts)} != NUM_KV_HEADS {kv_total}"
                )
        except ValueError:
            pass

    return errs


def resolve_launch_config(
    target: str,
    cluster: Cluster,
    overrides: dict[str, str] | None = None,
    cli_invocation: list[str] | None = None,
) -> tuple[LaunchConfig, list[str]]:
    """Build a LaunchConfig and return (config, validation_errors).

    The caller decides what to do with errors (CLI prints them and exits;
    a unit test might inspect them).
    """
    if target not in REGISTRY:
        raise KeyError(
            f"Unknown target {target!r}. Available: {ALL_TARGETS}"
        )

    env: dict[str, str] = {}

    # 1. CLI overrides go in FIRST so subsequent setdefault calls in
    #    common/preset cannot clobber them.
    if overrides:
        env.update(overrides)

    # 2. preset (each call uses setdefault)
    REGISTRY[target](env)

    # 3. common defaults (setdefault)
    apply_common_defaults(env)

    # 4. cluster topology (setdefault)
    _apply_cluster_to_env(env, cluster)

    # 5. mirror TP_*_SPLITS -> VLLM_TP_*_SPLITS
    _mirror_vllm_prefix(env)

    cfg = LaunchConfig(
        target=target,
        cluster=cluster,
        env=env,
        overrides=dict(overrides or {}),
        cli_invocation=list(cli_invocation or []),
    )
    return cfg, _validate(cfg)
