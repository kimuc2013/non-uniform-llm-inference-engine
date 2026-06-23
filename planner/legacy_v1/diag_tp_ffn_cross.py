"""Diagnostic: TP=8 cross-node with FFN bias.
- NCCL_DEBUG=INFO to capture collective sizes
- --enforce-eager to disable CUDA graph capture (non-uniform shape suspect)
- 8B model (faster iteration)
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
PERF = REPO / "perf" / "performance.py"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def main():
    out_dir = REPO / "results" / f"diag_tp_ffn_cross_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "vllm.log"
    port = 30400

    env = os.environ.copy()
    conda = str(Path(PY).parent)
    env["PATH"] = f"{conda}:/usr/local/cuda-12.9/bin:" + env.get("PATH", "")
    env["CC"] = "gcc-12"; env["CXX"] = "g++-12"; env["NVCC_CCBIN"] = "g++-12"
    env["CUDA_HOME"] = "/usr/local/cuda-12.9"
    env["VLLM_HOST_IP"] = CFG.head_fabric_ip
    env["RAY_ADDRESS"] = CFG.ray_address
    env["VLLM_LOGGING_LEVEL"] = "INFO"   # more verbose
    env["NCCL_DEBUG"] = "INFO"           # collective size info
    env["NCCL_SOCKET_IFNAME"] = f"{CFG.head_fabric_iface},{CFG.worker_fabric_iface}"
    env["NCCL_IB_HCA"] = CFG.nccl_ib_hca
    env["NCCL_NET_GDR_LEVEL"] = CFG.nccl_net_gdr_level
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_USE_FLASHINFER_MOE"] = "0"

    # FFN bias on 8B: [2688]*4 + [896]*4 = 14336 (FFN+50)
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
           "--enforce-eager",                              # NO CUDA graph
           "--enable-chunked-prefill",
           "--attention-backend", "FLASH_ATTN"]
    fout = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fout, stderr=subprocess.STDOUT,
                            cwd=str(REPO), preexec_fn=os.setsid)
    print(f"pid={proc.pid} port={port} log={log_path}", flush=True)

    deadline = time.time() + 600
    state = "timeout"
    while time.time() < deadline:
        if proc.poll() is not None:
            state = f"crash rc={proc.returncode}"; break
        try: txt = log_path.read_text(errors="ignore")
        except: txt = ""
        if "Application startup complete" in txt:
            state = "ready"; break
        if any(k in txt for k in ("out of memory", "RuntimeError: ", "Failed core proc")):
            state = "init_error"; break
        time.sleep(10)
    print(f"wait_ready: {state}", flush=True)

    if state == "ready":
        prompt_path = out_dir / "prompt.txt"
        prompt_path.write_text("Hello " * 200)
        cmd_perf = [PY, str(PERF),
                    "--base-url", f"http://127.0.0.1:{port}/v1",
                    "--model", MODEL, "--prompt-file", str(prompt_path),
                    "--requests", "16", "--runs", "1",
                    "--max-tokens", "64", "--ignore-eos",
                    "--output-summary-csv", str(out_dir / "perf_summary.csv")]
        try:
            p = subprocess.run(cmd_perf, env=env, cwd=str(REPO),
                               capture_output=True, text=True, timeout=180)
            print(p.stdout[-1500:], flush=True)
        except Exception as e:
            print(f"perf error: {e}", flush=True)

    if proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGINT); proc.wait(timeout=20)
        except:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except: pass


if __name__ == "__main__":
    main()
