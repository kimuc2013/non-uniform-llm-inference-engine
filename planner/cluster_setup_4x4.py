"""Automated hetero cluster setup. Idempotent: if cluster already matches
HEAD_GPUS + WORKER_GPUS from cluster.local.env, returns fast.
Otherwise restarts ray on both nodes (worker via ray-remote detached process).

USAGE:
  from planner.cluster_setup_4x4 import ensure_4x4_cluster
  ensure_4x4_cluster()
"""
from __future__ import annotations
import os
import subprocess
import time
from planner.cluster_env import CFG

HEAD_IP = CFG.head_fabric_ip
WORKER_IP = CFG.worker_fabric_ip
HEAD_IB = CFG.head_fabric_iface
WORKER_IB = CFG.worker_fabric_iface
RAY_PORT = CFG.gcs_port
RAY_ADDR = CFG.ray_address
PY = CFG.head_py
WORKER_PY = CFG.worker_py


def _check_cluster() -> tuple[bool, str]:
    """Return (is_4x4, message). Does not raise."""
    try:
        import ray
        if not ray.is_initialized():
            ray.init(address=RAY_ADDR, ignore_reinit_error=True)
        nodes = [n for n in ray.nodes() if n.get('alive')]
        by_ip = {n['NodeManagerAddress']: n.get('Resources', {}).get('GPU', 0) for n in nodes}
        head = by_ip.get(HEAD_IP, 0)
        worker = by_ip.get(WORKER_IP, 0)
        total = sum(by_ip.values())
        ok = (head == 4 and worker == 4 and total == 8)
        return ok, f"total={total} head={head} worker={worker}"
    except Exception as e:
        return False, f"ray.init/nodes failed: {e}"


def _restart_worker_via_ray() -> None:
    """Send detached restart command to worker via Ray remote actor.
    Actor self-terminates as part of `ray stop`; the detached subprocess
    survives and runs the new `ray start`."""
    import ray
    if not ray.is_initialized():
        ray.init(address=RAY_ADDR, ignore_reinit_error=True)
    worker_node = None
    for n in ray.nodes():
        if n.get('alive') and n['NodeManagerAddress'] == WORKER_IP:
            worker_node = n['NodeID']; break
    if worker_node is None:
        print(f"[cluster_setup] worker node not alive in cluster — cannot send restart cmd")
        return

    @ray.remote(num_cpus=0.1, scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=worker_node, soft=False))
    def detach_restart():
        import subprocess
        # Worker uses vllm_main env (symlink to vllm_new on worker); its ray binary:
        worker_ray = CFG.worker_ray
        cvd = CFG.worker_cuda_visible_devices
        cmd = (
            "sleep 3 && "
            f"{worker_ray} stop --force 2>&1 | tail -5 && "
            "sleep 10 && "
            f"export CUDA_VISIBLE_DEVICES={cvd} && "
            f"export VLLM_HOST_IP={WORKER_IP} && "
            f"export NCCL_SOCKET_IFNAME={WORKER_IB} && "
            f"export NCCL_IB_HCA={CFG.nccl_ib_hca} && "
            f"{worker_ray} start --address={RAY_ADDR} "
            f"--node-ip-address={WORKER_IP} --num-gpus={CFG.worker_gpus} 2>&1 | tail -10"
        )
        # Launch fully detached so it survives this actor dying
        subprocess.Popen(
            ['nohup', 'bash', '-c', cmd],
            stdout=open('/tmp/worker_ray_restart.log', 'w'),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        return "detached"
    try:
        result = ray.get(detach_restart.remote(), timeout=20)
        print(f"[cluster_setup] worker detach: {result}")
    except Exception as e:
        print(f"[cluster_setup] worker detach error (may still have launched): {e}")


def _restart_head() -> None:
    """Restart head ray directly (we have local subprocess access)."""
    head_ray = CFG.head_ray
    subprocess.run([head_ray, "stop", "--force"], capture_output=True, timeout=30)
    time.sleep(6)
    env = os.environ.copy()
    env["VLLM_HOST_IP"] = HEAD_IP
    env["NCCL_SOCKET_IFNAME"] = HEAD_IB
    env["NCCL_IB_HCA"] = "mlx5"
    p = subprocess.run(
        [head_ray, "start", "--head", "--node-ip-address", HEAD_IP, "--port", RAY_PORT],
        env=env, capture_output=True, text=True, timeout=60,
    )
    print(f"[cluster_setup] head ray start rc={p.returncode}")
    if p.returncode != 0:
        print(p.stderr[:2000])
    time.sleep(8)


def ensure_4x4_cluster(force_restart: bool = False) -> None:
    """Idempotent: if cluster is already 4+4 and force_restart is False, return.
    Otherwise restart both head and worker, then poll for 4+4 readiness."""
    if not force_restart:
        ok, msg = _check_cluster()
        if ok:
            print(f"[cluster_setup] already 4+4 ({msg}) — skip restart")
            return
        print(f"[cluster_setup] not 4+4 ({msg}) — restarting both nodes")

    # 1) Trigger worker restart via current ray (must happen BEFORE we kill head)
    print(f"[cluster_setup] sending worker restart cmd")
    _restart_worker_via_ray()

    # 2) Disconnect our driver from ray (so we can restart head without conflicts)
    try:
        import ray
        ray.shutdown()
    except Exception:
        pass

    # 3) Restart head ray
    print(f"[cluster_setup] restarting head ray")
    _restart_head()

    # 4) Poll for cluster 4+4 ready (worker rejoin)
    print(f"[cluster_setup] waiting for worker to rejoin (4+4)")
    deadline = time.time() + 180
    while time.time() < deadline:
        ok, msg = _check_cluster()
        print(f"[cluster_setup]   {msg}")
        if ok:
            print(f"[cluster_setup] cluster ready (4+4): {msg}")
            return
        time.sleep(8)
    raise RuntimeError(
        f"cluster_setup_4x4 timed out waiting for 4+4 ready. Final: {msg}. "
        f"Check /tmp/worker_ray_restart.log on worker node (if accessible)."
    )


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    ensure_4x4_cluster(force_restart=force)
