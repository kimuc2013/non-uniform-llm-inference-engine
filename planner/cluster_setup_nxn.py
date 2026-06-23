"""Parameterized hetero cluster (re)configuration for any N_head + N_worker
layout (1+1, 2+2, 3+3, 4+4). Generalizes cluster_setup_4x4.py: same
worker-via-detached-ray-actor restart, but the per-node GPU count is an
argument so the planner can be validated across the 1+1..4+4 target range.

USAGE:
  python planner/cluster_setup_nxn.py --head-gpus 2 --worker-gpus 2
  from planner.cluster_setup_nxn import ensure_cluster; ensure_cluster(2, 2)

Within a node every GPU is the same type (head=Blackwell, worker=Ada), so any
`--num-gpus N` subset is homogeneous and rank order [0,head)=Blackwell,
[head,world)=Ada is preserved — exactly what hetero_sweep assumes.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from planner.cluster_env import CFG

HEAD_IP = CFG.head_fabric_ip
WORKER_IP = CFG.worker_fabric_ip
HEAD_IB = CFG.head_fabric_iface
WORKER_IB = CFG.worker_fabric_iface
RAY_PORT = CFG.gcs_port
RAY_ADDR = CFG.ray_address


def _check_cluster(hg: int, wg: int) -> tuple[bool, str]:
    try:
        import ray
        if not ray.is_initialized():
            ray.init(address=RAY_ADDR, ignore_reinit_error=True)
        by_ip: dict[str, float] = {}
        for n in ray.nodes():
            if not n.get('alive'):
                continue
            ip = n['NodeManagerAddress']
            g = n.get('Resources', {}).get('GPU', 0)
            by_ip[ip] = max(by_ip.get(ip, 0), g)
        head = by_ip.get(HEAD_IP, 0)
        worker = by_ip.get(WORKER_IP, 0)
        return (head == hg and worker == wg), f"head={head} worker={worker} (want {hg}+{wg})"
    except Exception as e:
        return False, f"ray.init/nodes failed: {e}"


def _restart_worker_via_ray(wg: int) -> None:
    import ray
    if not ray.is_initialized():
        ray.init(address=RAY_ADDR, ignore_reinit_error=True)
    worker_node = None
    for n in ray.nodes():
        if n.get('alive') and n['NodeManagerAddress'] == WORKER_IP:
            worker_node = n['NodeID']; break
    if worker_node is None:
        print("[cluster_setup] worker node not alive — cannot send restart cmd")
        return

    @ray.remote(num_cpus=0.1, scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=worker_node, soft=False))
    def detach_restart():
        import subprocess
        worker_ray = CFG.worker_ray
        cvd = CFG.worker_cuda_visible_devices  # full device list; ray takes first wg
        head_host, head_port = RAY_ADDR.split(':')
        cmd = (
            "sleep 3 && "
            f"{worker_ray} stop --force 2>&1 | tail -5 && "
            "sleep 5 && "
            f"export CUDA_VISIBLE_DEVICES={cvd} && "
            f"export VLLM_HOST_IP={WORKER_IP} && "
            f"export NCCL_SOCKET_IFNAME={WORKER_IB} && "
            f"export NCCL_IB_HCA={CFG.nccl_ib_hca} && "
            "for i in $(seq 1 12); do "
            f"  if timeout 3 bash -c 'cat </dev/tcp/{head_host}/{head_port}' 2>/dev/null; then "
            f"    {worker_ray} start --address={RAY_ADDR} "
            f"--node-ip-address={WORKER_IP} --num-gpus={wg} 2>&1 | tail -10 && break; "
            "  fi; "
            "  echo \"[worker_restart] head GCS not reachable yet, retry $i/12\"; "
            "  sleep 10; "
            "done"
        )
        subprocess.Popen(['nohup', 'bash', '-c', cmd],
                         stdout=open('/tmp/worker_ray_restart.log', 'w'),
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, close_fds=True)
        return "detached"
    try:
        print(f"[cluster_setup] worker detach: {ray.get(detach_restart.remote(), timeout=20)}")
    except Exception as e:
        print(f"[cluster_setup] worker detach error (may still have launched): {e}")


def _restart_head(hg: int) -> None:
    head_ray = CFG.head_ray
    subprocess.run([head_ray, "stop", "--force"], capture_output=True, timeout=30)
    time.sleep(6)
    env = os.environ.copy()
    env["VLLM_HOST_IP"] = HEAD_IP
    env["NCCL_SOCKET_IFNAME"] = HEAD_IB
    env["NCCL_IB_HCA"] = CFG.nccl_ib_hca
    p = subprocess.run(
        [head_ray, "start", "--head", "--node-ip-address", HEAD_IP,
         "--port", RAY_PORT, "--num-gpus", str(hg)],
        env=env, capture_output=True, text=True, timeout=60)
    print(f"[cluster_setup] head ray start rc={p.returncode}")
    if p.returncode != 0:
        print(p.stderr[:2000])
    time.sleep(8)


def ensure_cluster(head_gpus: int, worker_gpus: int, force_restart: bool = False) -> None:
    if not force_restart:
        ok, msg = _check_cluster(head_gpus, worker_gpus)
        if ok:
            print(f"[cluster_setup] already {head_gpus}+{worker_gpus} ({msg}) — skip")
            return
        print(f"[cluster_setup] need {head_gpus}+{worker_gpus} ({msg}) — restarting both")
    _restart_worker_via_ray(worker_gpus)
    try:
        import ray
        ray.shutdown()
    except Exception:
        pass
    _restart_head(head_gpus)
    print(f"[cluster_setup] waiting for worker rejoin ({head_gpus}+{worker_gpus})")
    deadline = time.time() + 180
    msg = "?"
    while time.time() < deadline:
        ok, msg = _check_cluster(head_gpus, worker_gpus)
        print(f"[cluster_setup]   {msg}")
        if ok:
            print(f"[cluster_setup] cluster ready ({head_gpus}+{worker_gpus})")
            return
        time.sleep(8)
    raise RuntimeError(f"cluster_setup timed out for {head_gpus}+{worker_gpus}. "
                       f"Final: {msg}. Check /tmp/worker_ray_restart.log on worker.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-gpus", type=int, required=True)
    ap.add_argument("--worker-gpus", type=int, required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ensure_cluster(a.head_gpus, a.worker_gpus, force_restart=a.force)
