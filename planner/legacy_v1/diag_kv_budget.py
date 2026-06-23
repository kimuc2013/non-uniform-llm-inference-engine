"""Show KV cache budget per rank for: 70B FFN+50 + 8B uniform retry + 8B FFN+50."""
from __future__ import annotations
import os, signal, subprocess, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from planner.cluster_env import CFG

REPO = _REPO
PY = CFG.head_py

CASES = [
    ("70b_ffn50",  "meta-llama/Llama-3.3-70B-Instruct", "5376,5376,5376,5376,1792,1792,1792,1792", 30810),
    ("8b_uniform", "meta-llama/Llama-3.1-8B-Instruct",  "1792,1792,1792,1792,1792,1792,1792,1792", 30811),
    ("8b_ffn50",   "meta-llama/Llama-3.1-8B-Instruct",  "2688,2688,2688,2688,896,896,896,896",   30812),
]


def run(label, model, ffn, port):
    out_dir = REPO / "results" / f"diag_kv_{label}_{time.strftime('%Y%m%d_%H%M%S')}"
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
    env["VLLM_TP_FFN_SPLITS"]  = ffn
    if "70b" in label:
        env["VLLM_TP_HEAD_SPLITS"] = "8,8,8,8,8,8,8,8"
    else:
        env["VLLM_TP_HEAD_SPLITS"] = "4,4,4,4,4,4,4,4"
    env["VLLM_TP_KV_SPLITS"]   = "1,1,1,1,1,1,1,1"

    cmd = [PY, "-m", "vllm.entrypoints.openai.api_server",
           "--model", model,
           "--tensor-parallel-size", "8",
           "--pipeline-parallel-size", "1",
           "--distributed-executor-backend", "ray",
           "--max-model-len", "4096", "--max-num-seqs", "64",
           "--gpu-memory-utilization", "0.85",
           "--dtype", "bfloat16",
           "--port", str(port), "--host", "0.0.0.0",
           "--enable-chunked-prefill",
           "--attention-backend", "FLASH_ATTN"]
    fout = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fout, stderr=subprocess.STDOUT,
                            cwd=str(REPO), preexec_fn=os.setsid)
    print(f"\n[{label}] pid={proc.pid} port={port}", flush=True)
    deadline = time.time() + 900
    state = "timeout"
    while time.time() < deadline:
        if proc.poll() is not None:
            state = f"crash rc={proc.returncode}"; break
        try: txt = log_path.read_text(errors="ignore")
        except: txt = ""
        if "Application startup complete" in txt:
            state = "ready"; break
        if any(k in txt for k in ("CUBLAS_STATUS", "illegal memory", "Failed core proc")):
            state = "cuda_error"; break
        time.sleep(8)
    print(f"[{label}] state: {state}", flush=True)
    if log_path.exists():
        txt = log_path.read_text(errors="ignore")
        for ln in txt.splitlines():
            if any(k in ln for k in ["Model loading took", "Available KV cache memory",
                                      "GPU KV cache size", "CUBLAS_STATUS", "Application startup complete"]):
                print(ln, flush=True)
    if proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGINT); proc.wait(timeout=15)
        except: os.killpg(proc.pid, signal.SIGKILL)
    import ray
    try:
        ray.init(address=CFG.ray_address, ignore_reinit_error=True, logging_level="ERROR")
        for k in list(ray.util.placement_group_table().keys()):
            try: ray._private.worker.global_worker.core_worker.remove_placement_group(ray.PlacementGroupID(bytes.fromhex(k)))
            except: pass
        ray.shutdown()
    except: pass
    time.sleep(10)


def main():
    for label, model, ffn, port in CASES:
        run(label, model, ffn, port)


if __name__ == "__main__":
    main()
