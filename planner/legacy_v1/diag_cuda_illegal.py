"""Find the exact kernel triggering CUDA illegal memory access.
CUDA_LAUNCH_BLOCKING=1 makes the stack-trace point at the real failing kernel.
TORCH_USE_CUDA_DSA + TORCH_SHOW_CPP_STACKTRACES surface device-side asserts.
"""
from __future__ import annotations
import os, signal, subprocess, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from planner.cluster_env import CFG

REPO = _REPO
PY = CFG.head_py
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def run(label, ffn_splits, port):
    out_dir = REPO / "results" / f"cudadbg_{label}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "vllm.log"
    env = os.environ.copy()
    conda = str(Path(PY).parent)
    env["PATH"] = f"{conda}:/usr/local/cuda-12.9/bin:" + env.get("PATH", "")
    env["CC"] = "gcc-12"; env["CXX"] = "g++-12"; env["NVCC_CCBIN"] = "g++-12"
    env["CUDA_HOME"] = "/usr/local/cuda-12.9"
    env["VLLM_HOST_IP"] = CFG.head_fabric_ip
    env["RAY_ADDRESS"] = CFG.ray_address
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    env["NCCL_DEBUG"] = "WARN"
    env["NCCL_SOCKET_IFNAME"] = f"{CFG.head_fabric_iface},{CFG.worker_fabric_iface}"
    env["NCCL_IB_HCA"] = CFG.nccl_ib_hca
    env["NCCL_NET_GDR_LEVEL"] = CFG.nccl_net_gdr_level
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_USE_FLASHINFER_MOE"] = "0"
    env["VLLM_TP_FFN_SPLITS"]  = ffn_splits
    env["VLLM_TP_HEAD_SPLITS"] = "4,4,4,4,4,4,4,4"
    env["VLLM_TP_KV_SPLITS"]   = "1,1,1,1,1,1,1,1"
    # CUDA debug flags — synchronous kernel launches so the failing one shows up.
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["TORCH_USE_CUDA_DSA"] = "1"
    env["TORCH_SHOW_CPP_STACKTRACES"] = "1"

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
    print(f"[{label}] pid={proc.pid} port={port} log={log_path}", flush=True)

    deadline = time.time() + 720
    state = "timeout"
    while time.time() < deadline:
        if proc.poll() is not None:
            state = f"crash rc={proc.returncode}"; break
        try: txt = log_path.read_text(errors="ignore")
        except: txt = ""
        if "Application startup complete" in txt:
            state = "ready"; break
        if "illegal memory access" in txt or "CUDA error" in txt:
            state = "cuda_illegal"; break
        if "RuntimeError" in txt:
            state = "runtime_error"; break
        time.sleep(8)
    print(f"[{label}] state: {state}", flush=True)

    # Dump synchronous-kernel stack
    if log_path.exists():
        txt = log_path.read_text(errors="ignore")
        print(f"\n=== [{label}] CUDA error context ===", flush=True)
        lines = txt.splitlines()
        for i, ln in enumerate(lines):
            if "illegal memory" in ln or "CUDA error" in ln or "device-side assert" in ln:
                print(f"-- around line {i} --", flush=True)
                for x in lines[max(0,i-3):min(len(lines), i+40)]:
                    print(x, flush=True)
                break

    if proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGINT); proc.wait(timeout=15)
        except: os.killpg(proc.pid, signal.SIGKILL)
    time.sleep(10)


def main():
    print("=== Test 1: uniform values (should work — sanity) ===", flush=True)
    run("uniform", "1792,1792,1792,1792,1792,1792,1792,1792", 30700)

    # PG cleanup
    import ray
    try:
        ray.init(address=CFG.ray_address, ignore_reinit_error=True, logging_level="ERROR")
        for k in list(ray.util.placement_group_table().keys()):
            try: ray._private.worker.global_worker.core_worker.remove_placement_group(ray.PlacementGroupID(bytes.fromhex(k)))
            except: pass
        ray.shutdown()
    except: pass
    time.sleep(10)

    print("\n=== Test 2: FFN+50 (the actual broken case) ===", flush=True)
    run("ffn50", "2688,2688,2688,2688,896,896,896,896", 30701)


if __name__ == "__main__":
    main()
