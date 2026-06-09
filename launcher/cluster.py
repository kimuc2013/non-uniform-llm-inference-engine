"""Cluster topology loader.

Reads a shell-style cluster.local.env file (vllm_custom convention) and
returns a `Cluster` describing the head and worker nodes. The file is
sourced via bash so `$HEAD_IP` style interpolation works.

Required keys:
    HEAD_IP, WORKER_IP, HEAD_GPUS, WORKER_GPUS, HEAD_ACCEL, WORKER_ACCEL

Optional keys:
    GCS_PORT, RAY_ADDRESS, WORKER_SSH_PORT, HEAD_PY, WORKER_PY, TEMPDIR
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path


REQUIRED_KEYS = ("HEAD_IP", "WORKER_IP", "HEAD_GPUS", "WORKER_GPUS",
                 "HEAD_ACCEL", "WORKER_ACCEL")


class ClusterEnvError(RuntimeError):
    pass


@dataclass
class Cluster:
    head_ip: str
    worker_ip: str
    head_gpus: int
    worker_gpus: int
    head_accel: str       # planner profile key (e.g., "blackwell", "ada")
    worker_accel: str
    gcs_port: int = 6380
    ray_address: str = ""
    worker_ssh_port: int = 22
    head_py: str = ""
    worker_py: str = ""
    tempdir: str = "/tmp/vllm_main_ray"
    source_file: str = ""

    def is_single_node(self) -> bool:
        return self.head_ip == self.worker_ip and self.worker_gpus == 0

    def total_gpus(self) -> int:
        return self.head_gpus + (0 if self.is_single_node() else self.worker_gpus)

    def to_dict(self) -> dict:
        return asdict(self)


def _source_env_file(path: Path) -> dict[str, str]:
    """Source a shell env file in bash, return resulting environment dict.

    Uses `bash -c 'set -a; source FILE; env'` and parses NAME=VALUE lines.
    Lines without `=` or starting with `_=` are skipped.
    """
    if not path.is_file():
        raise ClusterEnvError(f"Cluster env file not found: {path}")
    cmd = ["bash", "-c", f"set -a; source {shlex.quote(str(path))}; env"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=10, check=True)
    except subprocess.CalledProcessError as exc:
        raise ClusterEnvError(
            f"Failed to source {path}: {exc.stderr.strip() or exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ClusterEnvError(f"Timed out sourcing {path}") from exc

    out: dict[str, str] = {}
    for line in res.stdout.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k and not k.startswith("_"):
            out[k] = v
    return out


def load_cluster_env(path: str | os.PathLike) -> Cluster:
    """Load a cluster.local.env file and return a Cluster.

    Raises ClusterEnvError if required keys are missing or malformed.
    """
    p = Path(path)
    env = _source_env_file(p)

    missing = [k for k in REQUIRED_KEYS if not env.get(k, "").strip()]
    if missing:
        raise ClusterEnvError(
            f"Cluster file {p} missing required keys: {missing}"
        )

    def _int(k: str, default: int | None = None) -> int:
        raw = env.get(k, "").strip()
        if not raw:
            if default is None:
                raise ClusterEnvError(f"Missing required int key: {k}")
            return default
        try:
            return int(raw)
        except ValueError as exc:
            raise ClusterEnvError(f"Invalid int for {k}={raw!r}") from exc

    gcs_port = _int("GCS_PORT", 6380)
    ray_address = env.get("RAY_ADDRESS", "").strip() or f"{env['HEAD_IP']}:{gcs_port}"

    return Cluster(
        head_ip=env["HEAD_IP"].strip(),
        worker_ip=env["WORKER_IP"].strip(),
        head_gpus=_int("HEAD_GPUS"),
        worker_gpus=_int("WORKER_GPUS"),
        head_accel=env["HEAD_ACCEL"].strip().lower(),
        worker_accel=env["WORKER_ACCEL"].strip().lower(),
        gcs_port=gcs_port,
        ray_address=ray_address,
        worker_ssh_port=_int("WORKER_SSH_PORT", 22),
        head_py=env.get("HEAD_PY", "").strip(),
        worker_py=env.get("WORKER_PY", "").strip(),
        tempdir=env.get("TEMPDIR", "").strip() or "/tmp/vllm_main_ray",
        source_file=str(p),
    )


def dump_cluster(cluster: Cluster, path: str | os.PathLike) -> None:
    Path(path).write_text(json.dumps(cluster.to_dict(), indent=2))
