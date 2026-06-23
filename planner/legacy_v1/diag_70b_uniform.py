"""Quick 70B TP=8 uniform diag — is cluster CUDA state actually broken?"""
from __future__ import annotations
import os, signal, subprocess, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from planner.cluster_env import CFG

REPO = _REPO
PY = CFG.head_py
MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def main():
    out_dir = REPO / "results" / f"diag_70b_uniform_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "vllm.log"
    port = 30800

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
    # Uniform 70B
    env["VLLM_TP_FFN_SPLITS"]  = "3584,3584,3584,3584,3584,3584,3584,3584"
    env["VLLM_TP_HEAD_SPLITS"] = "8,8,8,8,8,8,8,8"
    env["VLLM_TP_KV_SPLITS"]   = "1,1,1,1,1,1,1,1"

    cmd = [PY, "-m", "vllm.entrypoints.openai.api_server",
           "--model", MODEL,
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
    print(f"pid={proc.pid} port={port} log={log_path}", flush=True)

    deadline = time.time() + 900
    state = "timeout"
    while time.time() < deadline:
        if proc.poll() is not None:
            state = f"crash rc={proc.returncode}"; break
        try: txt = log_path.read_text(errors="ignore")
        except: txt = ""
        if "Application startup complete" in txt:
            state = "ready"; break
        if any(k in txt for k in ("CUBLAS_STATUS", "illegal memory", "RuntimeError: ", "Failed core proc")):
            state = "cuda_or_init_error"; break
        time.sleep(10)
    print(f"state: {state}", flush=True)

    if log_path.exists():
        txt = log_path.read_text(errors="ignore")
        print("\n=== first error/marker ===", flush=True)
        for ln in txt.splitlines():
            if any(k in ln for k in ["CUBLAS_STATUS", "illegal memory",
                                      "Application startup complete",
                                      "Available KV cache memory",
                                      "Model loading took"]):
                print(ln, flush=True)

    if proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGINT); proc.wait(timeout=20)
        except: os.killpg(proc.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
