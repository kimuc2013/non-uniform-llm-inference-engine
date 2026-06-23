"""Single-node TP=4 with FFN bias — isolate cross-node from non-uniform-TP issues."""
from __future__ import annotations
import os, signal, socket as sock, subprocess, sys, time, json
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from planner.cluster_env import CFG

REPO = _REPO
PY = CFG.head_py
PERF = REPO / "perf" / "performance.py"

# Llama 8B for fast iteration
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def main():
    out_dir = REPO / "results" / f"diag_ffn_bias_single_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "vllm.log"
    port = 30300

    env = os.environ.copy()
    conda = str(Path(PY).parent)
    env["PATH"] = f"{conda}:/usr/local/cuda-12.9/bin:" + env.get("PATH", "")
    env["CC"] = "gcc-12"; env["CXX"] = "g++-12"; env["NVCC_CCBIN"] = "g++-12"
    env["CUDA_HOME"] = "/usr/local/cuda-12.9"
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"   # head 4 Blackwell only
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_USE_FLASHINFER_MOE"] = "0"
    # TP=4 with FFN bias: 8B FFN 14336 total
    # bias +50%: rank 0,1 get more (2688*2 + 1792*2 = 8960; total 14336 / 4 = 3584 uniform)
    # Actually 4 ranks: 2 'Blackwell-ish' get +50%, 2 'Ada-ish' -50%
    env["VLLM_TP_FFN_SPLITS"] = "5376,5376,1792,1792"   # 2*5376 + 2*1792 = 14336
    env["VLLM_TP_HEAD_SPLITS"] = "8,8,8,8"              # uniform
    env["VLLM_TP_KV_SPLITS"]  = "2,2,2,2"               # uniform

    cmd = [PY, "-m", "vllm.entrypoints.openai.api_server",
           "--model", MODEL,
           "--tensor-parallel-size", "4",
           "--pipeline-parallel-size", "1",
           "--distributed-executor-backend", "mp",   # single-node mp avoids ray
           "--max-model-len", "4096", "--max-num-seqs", "64",
           "--gpu-memory-utilization", "0.55",
           "--dtype", "bfloat16",
           "--port", str(port), "--host", "0.0.0.0",
           "--enable-chunked-prefill",
           "--attention-backend", "FLASH_ATTN"]
    fout = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fout, stderr=subprocess.STDOUT,
                            cwd=str(REPO), preexec_fn=os.setsid)
    print(f"pid={proc.pid} port={port}", flush=True)

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
        time.sleep(8)
    print(f"wait_ready: {state}", flush=True)

    if state == "ready":
        prompt_path = out_dir / "prompt.txt"
        prompt_path.write_text("Hello " * 200)
        cmd_perf = [PY, str(PERF),
                    "--base-url", f"http://127.0.0.1:{port}/v1",
                    "--model", MODEL, "--prompt-file", str(prompt_path),
                    "--requests", "32", "--runs", "1",
                    "--max-tokens", "128", "--ignore-eos",
                    "--output-summary-csv", str(out_dir / "perf_summary.csv")]
        env_perf = os.environ.copy()
        env_perf["PATH"] = f"{conda}:" + env_perf.get("PATH", "")
        try:
            p = subprocess.run(cmd_perf, env=env_perf, cwd=str(REPO),
                               capture_output=True, text=True, timeout=300)
            print(p.stdout[-1500:], flush=True)
            sp = out_dir / "perf_summary.csv"
            if sp.exists():
                for line in sp.read_text().splitlines():
                    if "wall_throughput" in line or "request_time" in line:
                        print(f"  {line}", flush=True)
        except Exception as e:
            print(f"perf error: {e}", flush=True)

    if proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGINT); proc.wait(timeout=20)
        except:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except: pass


if __name__ == "__main__":
    main()
