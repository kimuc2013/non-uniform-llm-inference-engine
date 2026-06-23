"""Launch the 70B TP4PP2 [44,36] balanced overlap champion config wrapped with
nsys on BOTH head and worker, run a short perf workload, then collect traces.

Outputs:
  results/nsys_pp_overlap_<ts>/head.nsys-rep   — stage 0 (Blackwell) trace
  results/nsys_pp_overlap_<ts>/worker.nsys-rep — stage 1 (Ada) trace
  results/nsys_pp_overlap_<ts>/perf.log
  results/nsys_pp_overlap_<ts>/head_kernel_summary.txt
  results/nsys_pp_overlap_<ts>/head_nvtx_summary.txt

We use nsys' built-in attach via PID after vLLM is fully warmed up — that way
we only capture steady-state decode, not the slow init.
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
WORKER_NSYS = "/usr/local/bin/nsys"

# Champion config
TP = 4
PP = 2
LAYER_SPLIT = [44, 36]
FFN_SPLITS = [7168] * 4
HEAD_SPLITS = [16] * 4
KV_SPLITS = [2] * 4

# Smaller workload for faster profiling
N_REQ = 32   # small but still triggers microbatch (n_req >= pp=2)
IN_LEN = 512
OUT_LEN = 64  # short decode — capture a few cycles


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
    env["VLLM_TP_FFN_SPLITS"]  = ",".join(str(x) for x in FFN_SPLITS)
    env["VLLM_TP_HEAD_SPLITS"] = ",".join(str(x) for x in HEAD_SPLITS)
    env["VLLM_TP_KV_SPLITS"]   = ",".join(str(x) for x in KV_SPLITS)
    # Overlap recipe (launcher-validated):
    env["VLLM_PP_SAMPLED_BROADCAST_STREAM"] = "1"
    env["VLLM_PP_MICROBATCH"] = "1"
    env["VLLM_PP_MICROBATCH_SIZE"] = str(N_REQ // PP)
    env["VLLM_PP_BATCH_QUEUE_SIZE"] = str(PP)
    # NVTX annotations on (for nsys trace)
    env["VLLM_RPD_NVTX"] = "1"
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
    out_root = REPO / "results" / f"nsys_pp_overlap_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"OUT: {out_root}", flush=True)

    port = _free_port()
    env = _build_env()

    # Launch vLLM api_server (NO nsys wrap on api_server itself — we attach later
    # to the engine workers via their PIDs after warmup).
    log_path = out_root / "vllm.log"
    cmd = [
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
    print(f"[vllm] pid={proc.pid} port={port}", flush=True)

    # Wait for ready
    print("[wait] for Application startup complete", flush=True)
    deadline = time.time() + 900
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"[vllm] crashed rc={proc.returncode}", flush=True)
            return 1
        try: txt = log_path.read_text(errors="ignore")
        except: txt = ""
        if "Application startup complete" in txt:
            ready = True; break
        time.sleep(8)
    if not ready:
        print("[vllm] timeout", flush=True)
        try: os.killpg(proc.pid, signal.SIGINT); proc.wait(timeout=15)
        except: os.killpg(proc.pid, signal.SIGKILL)
        return 1
    print("[vllm] ready", flush=True)

    # Find Worker PIDs on head and worker
    # Head workers: VLLM::Worker_PP0_TP{0..3} child processes of EngineCore
    head_pids = subprocess.run(
        ["bash", "-c", "ps -eo pid,comm,args | grep 'RayWorkerProc' | grep -v grep | awk '{print $1}'"],
        capture_output=True, text=True).stdout.strip().splitlines()
    print(f"[head] worker PIDs: {head_pids}", flush=True)

    # Worker side PIDs via SSH
    wpids_raw = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "esca@10.20.0.28",
         "ps -eo pid,comm,args | grep 'RayWorkerProc' | grep -v grep | awk '{print $1}'"],
        capture_output=True, text=True).stdout.strip().splitlines()
    print(f"[worker] worker PIDs: {wpids_raw}", flush=True)

    if not head_pids or not wpids_raw:
        print("[err] no worker PIDs found", flush=True)
        try: os.killpg(proc.pid, signal.SIGINT); proc.wait(timeout=15)
        except: os.killpg(proc.pid, signal.SIGKILL)
        return 1

    # Start nsys profiler attached to ONE rank per stage (PP0_TP0 on head, PP1_TP0 on worker)
    # nsys profile --attach <pid> needs root usually; alternative: use --pid for capture
    # Actually nsys supports `--attach <pid>` to attach to a running process.
    head_attach_pid = int(head_pids[0])
    worker_attach_pid = int(wpids_raw[0])

    head_nsys_out = out_root / "head"
    worker_nsys_out = out_root / "worker"

    print(f"[nsys] starting on head attached to pid {head_attach_pid}", flush=True)
    head_nsys = subprocess.Popen(
        [NSYS, "profile",
         "--trace=cuda,nvtx,osrt",
         "--sample=none",
         "--cpuctxsw=none",
         "--cuda-memory-usage=true",
         "-o", str(head_nsys_out),
         "--force-overwrite", "true",
         "--duration", "30",  # 30s capture window
         "--attach", str(head_attach_pid)],
        stdout=open(out_root / "nsys_head.log", "w"), stderr=subprocess.STDOUT,
    )
    # Start worker nsys via SSH
    print(f"[nsys] starting on worker attached to pid {worker_attach_pid}", flush=True)
    worker_nsys = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", "esca@10.20.0.28",
         f"{WORKER_NSYS} profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none "
         f"--cuda-memory-usage=true -o /tmp/worker_nsys --force-overwrite=true "
         f"--duration 30 --attach {worker_attach_pid}"],
        stdout=open(out_root / "nsys_worker.log", "w"), stderr=subprocess.STDOUT,
    )

    time.sleep(3)  # give nsys time to attach

    # Run a small perf workload while nsys captures
    print("[perf] running short workload during nsys capture", flush=True)
    prompt_path = out_root / "prompt.txt"
    template = ("The following is a detailed analysis of large language model "
                "inference systems with a focus on heterogeneous GPU clusters. ")
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
        "--runs", "1",
        "--max-tokens", str(OUT_LEN),
        "--ignore-eos",
        "--output-csv", str(out_root / "perf_runs.csv"),
        "--output-summary-csv", str(out_root / "perf_summary.csv"),
    ]
    perf_env = os.environ.copy()
    perf_env["PATH"] = f"{Path(PY).parent}:" + perf_env.get("PATH", "")
    with (out_root / "perf.log").open("w") as f:
        subprocess.run(perf_cmd, env=perf_env, cwd=str(REPO),
                       stdout=f, stderr=subprocess.STDOUT, timeout=300)

    # Wait for nsys to finish capture
    print("[nsys] waiting for capture to complete (30s)", flush=True)
    head_nsys.wait(timeout=180)
    worker_nsys.wait(timeout=180)
    print(f"[nsys] head rc={head_nsys.returncode} worker rc={worker_nsys.returncode}", flush=True)

    # Pull worker nsys file via SCP
    print("[scp] pulling worker nsys file", flush=True)
    subprocess.run(["scp", "-o", "BatchMode=yes",
                    "esca@10.20.0.28:/tmp/worker_nsys.nsys-rep",
                    str(out_root / "worker.nsys-rep")],
                   timeout=120)

    # Generate stats (kernel + NVTX summaries)
    print("[stats] head kernel summary", flush=True)
    head_rep = str(out_root / "head.nsys-rep")
    for kind in ["kernels", "nvtx", "nccl"]:
        try:
            r = subprocess.run([NSYS, "stats", "--report", kind, head_rep],
                               capture_output=True, text=True, timeout=120)
            (out_root / f"head_{kind}_summary.txt").write_text(r.stdout)
        except Exception as e:
            print(f"[stats] head {kind}: {e}", flush=True)

    print("[stats] worker kernel summary", flush=True)
    worker_rep = str(out_root / "worker.nsys-rep")
    for kind in ["kernels", "nvtx", "nccl"]:
        try:
            r = subprocess.run([NSYS, "stats", "--report", kind, worker_rep],
                               capture_output=True, text=True, timeout=120)
            (out_root / f"worker_{kind}_summary.txt").write_text(r.stdout)
        except Exception as e:
            print(f"[stats] worker {kind}: {e}", flush=True)

    # Cleanup
    print("[cleanup] stopping vLLM", flush=True)
    try: os.killpg(proc.pid, signal.SIGINT); proc.wait(timeout=20)
    except: os.killpg(proc.pid, signal.SIGKILL)
    subprocess.run(["pkill", "-9", "-f", f"api_server.*--port {port}"], capture_output=True)
    print(f"[done] artifacts in {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
