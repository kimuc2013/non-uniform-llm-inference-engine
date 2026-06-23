"""nsys verify PP overlap — v2: launch vLLM api_server WRAPPED with nsys (head side).

Captures:
  - Engine NVTX ranges (execute_model dispatches)
  - CUDA kernels for head ranks (stage 0, Blackwell)
  - NCCL P2P send/recv kernels (cross-stage communication)
  - Side-stream broadcast events (M13 fix)

Worker-side (stage 1, Ada) is harder to profile without root. Head-side trace
already reveals overlap pattern: time between consecutive stage-0 forward kernels
should equal Ada stage time (~63 ms) if overlap works; should equal ~80ms+ if
sequential.

We use --capture-range=cudaProfilerApi + cudaProfilerStart/Stop hooks via vllm
internal — OR just wrap with --delay/--duration to catch steady state.
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
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
NSYS = "/usr/local/cuda-12.9/bin/nsys"

TP = 4
PP = 2
LAYER_SPLIT = [44, 36]
N_REQ = 32
IN_LEN = 512
OUT_LEN = 64

# Estimated init time before we want to start capturing (model load + CUDA graphs)
INIT_DELAY_S = 180   # 70B init takes ~2-3 min
CAPTURE_DURATION_S = 30


def _build_env():
    env = os.environ.copy()
    conda = str(Path(PY).parent)
    env["PATH"] = f"{conda}:/usr/local/cuda-12.9/bin:" + env.get("PATH", "")
    env["CC"] = "gcc-12"; env["CXX"] = "g++-12"; env["NVCC_CCBIN"] = "g++-12"
    env["CUDA_HOME"] = "/usr/local/cuda-12.9"
    env["VLLM_HOST_IP"] = CFG.head_fabric_ip
    env["RAY_ADDRESS"] = CFG.ray_address
    env["VLLM_LOGGING_LEVEL"] = "WARNING"
    env["NCCL_DEBUG"] = "WARN"
    env["NCCL_SOCKET_IFNAME"] = f"{CFG.head_fabric_iface},{CFG.worker_fabric_iface}"
    env["NCCL_IB_HCA"] = "mlx5"
    env["NCCL_NET_GDR_LEVEL"] = "2"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_USE_FLASHINFER_MOE"] = "0"
    env["VLLM_PP_LAYER_PARTITION"] = ",".join(str(x) for x in LAYER_SPLIT)
    env["VLLM_TP_FFN_SPLITS"]  = ",".join(str(x) for x in [7168] * 4)
    env["VLLM_TP_HEAD_SPLITS"] = ",".join(str(x) for x in [16] * 4)
    env["VLLM_TP_KV_SPLITS"]   = ",".join(str(x) for x in [2] * 4)
    env["VLLM_PP_SAMPLED_BROADCAST_STREAM"] = "1"
    env["VLLM_PP_MICROBATCH"] = "1"
    env["VLLM_PP_MICROBATCH_SIZE"] = str(N_REQ // PP)
    env["VLLM_PP_BATCH_QUEUE_SIZE"] = str(PP)
    return env


def _free_port(start=29800):
    p = start
    while True:
        s = sock.socket()
        try: s.bind(("127.0.0.1", p)); s.close(); return p
        except OSError: p += 1
        finally:
            try: s.close()
            except: pass


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = REPO / "results" / f"nsys_overlap_v2_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"OUT: {out_root}", flush=True)

    port = _free_port()
    env = _build_env()

    log_path = out_root / "vllm.log"
    head_trace = out_root / "head"

    # nsys profile wraps the api_server
    cmd = [
        NSYS, "profile",
        "-t", "cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=none",
        "--cuda-memory-usage=false",
        "-o", str(head_trace),
        "--force-overwrite=true",
        "--delay", str(INIT_DELAY_S),
        "--duration", str(CAPTURE_DURATION_S),
        PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--tensor-parallel-size", str(TP),
        "--pipeline-parallel-size", str(PP),
        "--distributed-executor-backend", "ray",
        "--max-model-len", "2048",
        "--max-num-seqs", str(N_REQ),
        "--gpu-memory-utilization", "0.85",
        "--dtype", "bfloat16",
        "--port", str(port), "--host", "0.0.0.0",
        "--enable-chunked-prefill",
        "--attention-backend", "FLASH_ATTN",
    ]
    fout = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fout, stderr=subprocess.STDOUT,
                            cwd=str(REPO), preexec_fn=os.setsid)
    print(f"[vllm-nsys] pid={proc.pid} port={port}", flush=True)
    print(f"[capture] will start after {INIT_DELAY_S}s, run for {CAPTURE_DURATION_S}s", flush=True)

    # Wait for ready
    deadline = time.time() + 900
    ready_at = None
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"[vllm] crashed rc={proc.returncode}", flush=True)
            return 1
        try: txt = log_path.read_text(errors="ignore")
        except: txt = ""
        if "Application startup complete" in txt:
            ready_at = time.time(); break
        time.sleep(8)
    if not ready_at:
        print("[vllm] timeout", flush=True); proc.kill(); return 1
    print(f"[vllm] ready after {ready_at - (time.time() - (deadline - time.time())):.0f}s", flush=True)

    # Wait until nsys capture window opens
    elapsed_since_launch = time.time() - (deadline - 900)  # rough
    # Just sleep a few seconds after ready and start sending requests so the
    # capture window catches steady-state decoding
    print("[wait] sending perf 5s after ready (capture window approximate)", flush=True)
    time.sleep(5)

    # Run perf workload (continuously enough to overlap with capture window)
    print("[perf] running workload during nsys capture", flush=True)
    prompt_path = out_root / "prompt.txt"
    template = ("The following is a detailed analysis of LLM serving with PP overlap. ")
    words_needed = int(IN_LEN / 1.3)
    base = template.split()
    out_words = []
    while len(out_words) < words_needed: out_words.extend(base)
    prompt_path.write_text(" ".join(out_words[:words_needed]))

    perf_cmd = [
        PY, str(PERF),
        "--base-url", f"http://127.0.0.1:{port}/v1",
        "--model", MODEL,
        "--prompt-file", str(prompt_path),
        "--requests", str(N_REQ),
        "--runs", "3",  # multiple runs to span capture window
        "--max-tokens", str(OUT_LEN),
        "--ignore-eos",
        "--output-csv", str(out_root / "perf_runs.csv"),
        "--output-summary-csv", str(out_root / "perf_summary.csv"),
    ]
    perf_env = os.environ.copy()
    perf_env["PATH"] = f"{Path(PY).parent}:" + perf_env.get("PATH", "")
    with (out_root / "perf.log").open("w") as f:
        subprocess.run(perf_cmd, env=perf_env, cwd=str(REPO),
                       stdout=f, stderr=subprocess.STDOUT, timeout=600)

    # Wait for nsys to finish (it has its own duration timer)
    # nsys process will end on its own after duration
    print("[wait] for nsys to finish capture and write report", flush=True)
    try:
        proc.wait(timeout=240)
    except subprocess.TimeoutExpired:
        # nsys done, vllm continues — kill vllm
        pass
    try: os.killpg(proc.pid, signal.SIGINT); proc.wait(timeout=20)
    except:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except: pass

    # Generate stats
    head_rep = str(out_root / "head.nsys-rep")
    if not os.path.exists(head_rep):
        print(f"[stats] no head.nsys-rep found", flush=True)
        # Try sqlite or check inside out_root
        for f in os.listdir(out_root):
            print(f"  {f}")
        return 1
    print(f"[stats] head trace: {os.path.getsize(head_rep)/1024/1024:.1f} MB", flush=True)
    for kind in ["nvtx_sum", "cuda_kern_exec_sum", "cuda_api_sum", "nccl_sum"]:
        try:
            r = subprocess.run([NSYS, "stats", "--report", kind, head_rep],
                               capture_output=True, text=True, timeout=180)
            (out_root / f"head_{kind}.txt").write_text(r.stdout + "\n\nSTDERR:\n" + r.stderr)
        except Exception as e:
            print(f"[stats] head {kind}: {e}", flush=True)

    print(f"[done] artifacts in {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
