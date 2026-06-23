"""Hang diagnostic: TP=8 cross-node + FFN bias with CUDA graph (default).
Server start with VLLM_DIAG_HANG instrumentation already in our worker patches,
+ NCCL_DEBUG=INFO so we capture which collective is hanging.

Goal: get the server stuck so we can py-spy worker stacks and read graph-capture
progress markers.
"""
from __future__ import annotations
import os, signal, socket as sock, subprocess, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from planner.cluster_env import CFG

REPO = _REPO
PY = CFG.head_py
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def main():
    out_dir = REPO / "results" / f"hang_diag_tp_ffn_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "vllm.log"
    port = 30500

    env = os.environ.copy()
    conda = str(Path(PY).parent)
    env["PATH"] = f"{conda}:/usr/local/cuda-12.9/bin:" + env.get("PATH", "")
    env["CC"] = "gcc-12"; env["CXX"] = "g++-12"; env["NVCC_CCBIN"] = "g++-12"
    env["CUDA_HOME"] = "/usr/local/cuda-12.9"
    env["VLLM_HOST_IP"] = CFG.head_fabric_ip
    env["RAY_ADDRESS"] = CFG.ray_address
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    env["NCCL_DEBUG"] = "INFO"
    env["NCCL_DEBUG_SUBSYS"] = "INIT,COLL"   # show collective info
    env["NCCL_SOCKET_IFNAME"] = f"{CFG.head_fabric_iface},{CFG.worker_fabric_iface}"
    env["NCCL_IB_HCA"] = CFG.nccl_ib_hca
    env["NCCL_NET_GDR_LEVEL"] = CFG.nccl_net_gdr_level
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_USE_FLASHINFER_MOE"] = "0"
    env["VLLM_DIAG_HANG"] = "1"   # our instrumentation in worker_busy_loop + compile_or_warm_up_model

    # FFN+50 on 8B
    env["VLLM_TP_FFN_SPLITS"]  = "2688,2688,2688,2688,896,896,896,896"
    env["VLLM_TP_HEAD_SPLITS"] = "4,4,4,4,4,4,4,4"
    env["VLLM_TP_KV_SPLITS"]   = "1,1,1,1,1,1,1,1"

    cmd = [PY, "-m", "vllm.entrypoints.openai.api_server",
           "--model", MODEL,
           "--tensor-parallel-size", "8",
           "--pipeline-parallel-size", "1",
           "--distributed-executor-backend", "ray",
           "--max-model-len", "4096", "--max-num-seqs", "32",
           "--gpu-memory-utilization", "0.85",
           "--dtype", "bfloat16",
           "--port", str(port), "--host", "0.0.0.0",
           "--enable-chunked-prefill",
           "--attention-backend", "FLASH_ATTN"]
    fout = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fout, stderr=subprocess.STDOUT,
                            cwd=str(REPO), preexec_fn=os.setsid)
    print(f"pid={proc.pid} port={port} log={log_path}", flush=True)
    print(f"Will run for 8 min then dump worker stacks via py-spy.", flush=True)

    # Wait until we hit a likely hang (no progress for ~3 min after weights load)
    deadline = time.time() + 480
    last_size = 0; stall_start = None
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"crash rc={proc.returncode}", flush=True); return
        try: txt = log_path.read_text(errors="ignore")
        except: txt = ""
        if "Application startup complete" in txt:
            print("READY (not expected to hang) — proceeding to test", flush=True)
            break
        sz = log_path.stat().st_size if log_path.exists() else 0
        if "Loading weights took" in txt:
            if stall_start is None and sz == last_size:
                stall_start = time.time()
            if sz != last_size:
                stall_start = None
                last_size = sz
            if stall_start and (time.time() - stall_start) > 180:
                print(f"STALL detected — log frozen 3min after weights load. Dumping stacks.", flush=True)
                break
        time.sleep(10)

    # Find worker processes + py-spy dump
    print("=== process tree ===", flush=True)
    subprocess.run(["pstree", "-p", str(proc.pid)], capture_output=False)
    print("\n=== running VLLM workers (head) ===", flush=True)
    subprocess.run("ps aux | grep -E 'VLLM::|RayWorkerProc\\.run' | grep -v grep",
                   shell=True)
    # py-spy each VLLM worker
    print("\n=== py-spy dumps ===", flush=True)
    result = subprocess.run(
        "ps -ef | grep -E 'VLLM::|RayWorkerProc' | grep -v grep | awk '{print $2}' | head -5",
        shell=True, capture_output=True, text=True,
    )
    for pid in result.stdout.strip().split("\n"):
        if not pid.strip(): continue
        print(f"\n--- pid {pid} ---", flush=True)
        subprocess.run(["/data/esca/uckim/miniconda3/envs/vllm_main/bin/py-spy", "dump", "--pid", pid.strip()],
                       capture_output=False, timeout=15)
    # worker node via ssh
    print("\n=== py-spy worker node ===", flush=True)
    ssh_cmd = f"""ssh {CFG.ssh_target} '
pids=$(ps -ef | grep -E "VLLM::|RayWorkerProc" | grep -v grep | awk "{{print \\$2}}" | head -4)
for p in $pids; do echo "--- worker pid $p ---"; /data/esca/uckim/miniconda3/envs/vllm_main/bin/py-spy dump --pid $p 2>&1 | head -40; done
'"""
    subprocess.run(ssh_cmd, shell=True, timeout=120)

    if proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGINT); proc.wait(timeout=20)
        except: os.killpg(proc.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
