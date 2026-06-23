"""Memory profile diagnostic — compare uniform TP=8 vs FFN bias TP=8.
INFO logging captures KV cache budget per rank.
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
MODEL = "meta-llama/Llama-3.1-8B-Instruct"   # smaller, faster

VARIANTS = [
    ("uniform",  "1792,1792,1792,1792,1792,1792,1792,1792"),
    ("ffn50",    "2688,2688,2688,2688,896,896,896,896"),
]

def run(label, ffn_splits):
    out_dir = REPO / "results" / f"memdiag_{label}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "vllm.log"
    port = 30600 + (0 if label == "uniform" else 1)

    env = os.environ.copy()
    conda = str(Path(PY).parent)
    env["PATH"] = f"{conda}:/usr/local/cuda-12.9/bin:" + env.get("PATH", "")
    env["CC"] = "gcc-12"; env["CXX"] = "g++-12"; env["NVCC_CCBIN"] = "g++-12"
    env["CUDA_HOME"] = "/usr/local/cuda-12.9"
    env["VLLM_HOST_IP"] = CFG.head_fabric_ip
    env["RAY_ADDRESS"] = CFG.ray_address
    env["VLLM_LOGGING_LEVEL"] = "INFO"   # important for memory logs
    env["NCCL_DEBUG"] = "WARN"
    env["NCCL_SOCKET_IFNAME"] = f"{CFG.head_fabric_iface},{CFG.worker_fabric_iface}"
    env["NCCL_IB_HCA"] = CFG.nccl_ib_hca
    env["NCCL_NET_GDR_LEVEL"] = CFG.nccl_net_gdr_level
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_USE_FLASHINFER_MOE"] = "0"
    env["VLLM_TP_FFN_SPLITS"]  = ffn_splits
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
    print(f"[{label}] pid={proc.pid} port={port}", flush=True)

    # Wait until ready or timeout
    deadline = time.time() + 480
    state = "timeout"
    while time.time() < deadline:
        if proc.poll() is not None:
            state = f"crash rc={proc.returncode}"; break
        try: txt = log_path.read_text(errors="ignore")
        except: txt = ""
        if "Application startup complete" in txt:
            state = "ready"; break
        if any(k in txt for k in ("RuntimeError: ", "Failed core proc", "out of memory")):
            state = "init_error"; break
        time.sleep(10)
    print(f"[{label}] state: {state}", flush=True)

    # Print memory markers
    if log_path.exists():
        txt = log_path.read_text(errors="ignore")
        print(f"\n=== [{label}] memory markers ===", flush=True)
        for line in txt.splitlines():
            if any(k in line for k in [
                "Available KV cache",
                "GPU KV cache size",
                "Memory profiling",
                "model_memory_usage",
                "Model loading took",
                "Initial free memory",
                "peak_activation_memory",
                "cudagraph_memory",
                "CUDA graph pool memory",
                "Maximum concurrency",
                "non_torch_memory",
                "weights_memory",
            ]):
                print(line, flush=True)

    # Stop
    if proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGINT); proc.wait(timeout=20)
        except: os.killpg(proc.pid, signal.SIGKILL)
    # cleanup PG
    import time as _t; _t.sleep(15)


def main():
    for label, ffn in VARIANTS:
        run(label, ffn)
        print(f"\n{'='*60}\n", flush=True)
        # cleanup between cells
        import ray
        ray.init(address=CFG.ray_address, ignore_reinit_error=True, logging_level="ERROR")
        for k in list(ray.util.placement_group_table().keys()):
            try: ray._private.worker.global_worker.core_worker.remove_placement_group(ray.PlacementGroupID(bytes.fromhex(k)))
            except: pass
        ray.shutdown()
        time.sleep(15)


if __name__ == "__main__":
    main()
