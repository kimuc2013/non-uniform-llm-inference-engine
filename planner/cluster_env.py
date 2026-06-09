"""Load cluster.local.env (server-specific, gitignored) and expose as a typed config.

All scripts under planner/ (sweep runners, cluster setup, auto wrapper) should
read cluster info via this module — never hard-code IPs / interface names / paths.

Usage:
    from planner.cluster_env import CFG
    print(CFG.head_fabric_ip, CFG.worker_ssh_host)
"""
from __future__ import annotations
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCAL_ENV = REPO / "cluster.local.env"
EXAMPLE_ENV = REPO / "cluster.example.env"


def _source(path: Path) -> dict[str, str]:
    """Run bash to source an env file and capture the resulting exported vars."""
    if not path.exists():
        return {}
    out = subprocess.check_output(
        ["bash", "-c", f"set -a; source {path}; env"],
        text=True,
    )
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


@dataclass(frozen=True)
class ClusterConfig:
    head_ip: str
    worker_ip: str
    head_gpus: int
    worker_gpus: int
    head_accel: str
    worker_accel: str
    head_fabric_ip: str
    worker_fabric_ip: str
    head_fabric_iface: str
    worker_fabric_iface: str
    gcs_port: str
    ray_address: str
    worker_ssh_user: str
    worker_ssh_host: str
    worker_ssh_port: str
    head_py: str
    worker_py: str
    head_ray: str
    worker_ray: str
    worker_cuda_visible_devices: str
    nccl_ib_hca: str
    nccl_net_gdr_level: str
    tempdir: str

    @property
    def ssh_target(self) -> str:
        return f"{self.worker_ssh_user}@{self.worker_ssh_host}"


def load(path: Path | None = None) -> ClusterConfig:
    src = path or LOCAL_ENV
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Copy {EXAMPLE_ENV.name} to cluster.local.env "
            f"and fill in your cluster's IPs/paths."
        )
    e = _source(src)
    def g(k, d=""): return e.get(k, d)
    def gi(k, d=0):
        try: return int(e.get(k, d))
        except: return d
    return ClusterConfig(
        head_ip=g("HEAD_IP"),
        worker_ip=g("WORKER_IP"),
        head_gpus=gi("HEAD_GPUS"),
        worker_gpus=gi("WORKER_GPUS"),
        head_accel=g("HEAD_ACCEL"),
        worker_accel=g("WORKER_ACCEL"),
        head_fabric_ip=g("HEAD_FABRIC_IP", g("HEAD_IP")),
        worker_fabric_ip=g("WORKER_FABRIC_IP", g("WORKER_IP")),
        head_fabric_iface=g("HEAD_FABRIC_IFACE", "lo"),
        worker_fabric_iface=g("WORKER_FABRIC_IFACE", "lo"),
        gcs_port=g("GCS_PORT", "6379"),
        ray_address=g("RAY_ADDRESS", f"{g('HEAD_FABRIC_IP', g('HEAD_IP'))}:{g('GCS_PORT', '6379')}"),
        worker_ssh_user=g("WORKER_SSH_USER", os.environ.get("USER", "root")),
        worker_ssh_host=g("WORKER_SSH_HOST", g("WORKER_IP")),
        worker_ssh_port=g("WORKER_SSH_PORT", "22"),
        head_py=g("HEAD_PY"),
        worker_py=g("WORKER_PY", g("HEAD_PY")),
        head_ray=g("HEAD_RAY", g("HEAD_PY").replace("/python", "/ray")),
        worker_ray=g("WORKER_RAY", g("HEAD_RAY", g("HEAD_PY").replace("/python", "/ray"))),
        worker_cuda_visible_devices=g("WORKER_CUDA_VISIBLE_DEVICES", ""),
        nccl_ib_hca=g("NCCL_IB_HCA", "mlx5"),
        nccl_net_gdr_level=g("NCCL_NET_GDR_LEVEL", "2"),
        tempdir=g("TEMPDIR", "/tmp/vllm_ray"),
    )


# Module-level singleton — most call sites just `from planner.cluster_env import CFG`
CFG = load()


if __name__ == "__main__":
    import dataclasses, json
    print(json.dumps(dataclasses.asdict(CFG), indent=2))
