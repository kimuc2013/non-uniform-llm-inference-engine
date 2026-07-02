"""Verify PP overlap by running the champion config under vllm's torch profiler.

Each Ray worker (8 ranks: PP0_TP0..3 on head, PP1_TP0..3 on worker) writes its
own trace JSON to <torch_profiler_dir>. The trace shows:
  - Forward kernels per rank
  - NCCL P2P send/recv
  - Side-stream broadcast (M13)

Stage 0 (head) and stage 1 (worker) traces overlapped in wall-clock time prove
PP overlap is working.

Steps:
  1. Launch vllm with profiler config
  2. Wait for ready
  3. Warm up with 1 dummy request
  4. POST /start_profile
  5. Run perf workload that fills 5–10 microbatch cycles (~3–5 seconds)
  6. POST /stop_profile
  7. SCP worker's trace from /tmp/torch_traces on Ada node
  8. Print summary timing analysis
"""
from __future__ import annotations
import json, os, signal, socket as sock, subprocess, sys, time, urllib.request
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from planner.cluster_env import CFG
from planner.perf_planner import MODELS

REPO = _REPO
PY = CFG.head_py
PERF = REPO / "perf" / "performance.py"

# eta is MODEL-DEPENDENT (bigger model -> more compute to hide the fixed PP bubble ->
# higher eta), so measure it ON THE DEPLOYMENT MODEL (the one being served, loaded here
# anyway). PP_OVERLAP_MODEL selects it; the hardware params stay cluster-cached.
MODEL_KEY = os.environ.get("PP_OVERLAP_MODEL", "70b")
_SPEC = MODELS[MODEL_KEY]
MODEL = _SPEC.name
N_LAYERS = _SPEC.n_layers
HIDDEN = _SPEC.hidden
FFN_DIM = _SPEC.ffn_dim
N_Q = _SPEC.n_q
N_KV = _SPEC.n_kv

# PP_OVERLAP_PP=1 runs the TP-only (TP=8 PP=1) FLOOR-IDENTIFICATION twin: with pp=1 the
# cost model is c1 = t_step(roofline) + floor — NO eta term — so the engine host floor is
# identified per-model WITHOUT degeneracy. The PP=2 run then identifies eta with floor
# known:  eta = 1 - (c2 - floor - b_max)/(b_rest + t_send).  Two dedicated pre-serving
# runs -> two unknowns, algebraically exact (no priors, no magic constants).
PP = int(os.environ.get("PP_OVERLAP_PP", "2"))
# PP_OVERLAP_TP overrides TP (default world=8). TP=4,PP=1 -> world=4 = HEAD-ONLY twin:
# valid for the ENGINE-FLOOR identification F (engine host cost is a node-local
# software property; the roofline t_step accounts for the TP4 compute), and runnable
# while the worker node is busy.
TP = int(os.environ.get("PP_OVERLAP_TP", str(8 // PP)))
LAYER_SPLIT = ([N_LAYERS] if PP == 1 else
               [N_LAYERS // 2, N_LAYERS - N_LAYERS // 2])   # uniform (eta ~ split-insensitive)
N_REQ = int(os.environ.get("PP_OVERLAP_NREQ", "64"))   # env-tunable batch. mb_size=N_REQ/PP.
             # Small N_REQ -> tiny decode kernels -> GPU starves on Python dispatch (CPU-bound,
             # eta artefact ~0); larger N_REQ -> bigger GEMMs -> GPU-bound -> real overlap.
IN_LEN = 8     # DECODE-CLEAN: tiny prefill (64*8=512 tok, 1 chunk, done instantly) so the
               # trace is dominated by PURE decode steps -- no chunked-prefill contamination
               # of the per-stage busy times (IN=256 inflated head-busy ~2x, gave bogus eta~0)
OUT_LEN = 48   # ~48 decode steps -> ~35 steady-state after dropping warmup


def _build_env(trace_dir):
    env = os.environ.copy()
    conda = str(Path(PY).parent)
    env["PATH"] = f"{conda}:/usr/local/cuda-12.9/bin:" + env.get("PATH", "")
    env["CC"] = "gcc-12"; env["CXX"] = "g++-12"; env["NVCC_CCBIN"] = "g++-12"
    env["CUDA_HOME"] = "/usr/local/cuda-12.9"
    env["VLLM_HOST_IP"] = CFG.head_fabric_ip
    env["RAY_ADDRESS"] = CFG.ray_address
    env["HF_HUB_OFFLINE"] = "1"   # gated 70B: avoid the HF-429 "Parse safetensors" 3h stall
    env["VLLM_LOGGING_LEVEL"] = "WARNING"
    env["NCCL_DEBUG"] = "WARN"
    env["NCCL_SOCKET_IFNAME"] = f"{CFG.head_fabric_iface},{CFG.worker_fabric_iface}"
    env["NCCL_IB_HCA"] = "mlx5"
    env["NCCL_NET_GDR_LEVEL"] = "2"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_USE_FLASHINFER_MOE"] = "0"
    env["VLLM_TP_FFN_SPLITS"]  = ",".join(str(FFN_DIM // TP) for _ in range(TP))
    env["VLLM_TP_HEAD_SPLITS"] = ",".join(str(N_Q // TP) for _ in range(TP))
    env["VLLM_TP_KV_SPLITS"]   = ",".join(str(max(1, N_KV // TP)) for _ in range(TP))
    if PP > 1:                                    # PP-overlap machinery only in the PP twin
        env["VLLM_PP_LAYER_PARTITION"] = ",".join(str(x) for x in LAYER_SPLIT)
        env["VLLM_PP_SAMPLED_BROADCAST_STREAM"] = "1"
        env["VLLM_PP_MICROBATCH"] = "1"
        env["VLLM_PP_MICROBATCH_SIZE"] = os.environ.get("PP_OVERLAP_MB_SIZE", str(N_REQ // PP))
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


def post(url, data=b"", timeout=600):
    # stop_profile BLOCKS until all ranks flush their (large) traces -- 120s was too
    # short and crashed the script mid-flush (=> manual scp + a truncated rank). 600s.
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode()


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = REPO / "results" / f"verify_pp_overlap_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    trace_dir = str(out_root / "traces")  # head writes here; worker writes to same path on its node
    os.makedirs(trace_dir, exist_ok=True)
    # Also create same dir on worker (per-node /data)
    subprocess.run(["ssh", "-o", "BatchMode=yes", "esca@10.20.0.28",
                    f"mkdir -p {trace_dir}"], timeout=15)
    print(f"OUT: {out_root}", flush=True)
    print(f"trace_dir: {trace_dir} (per-node)", flush=True)
    (out_root / "run_meta.json").write_text(json.dumps({
        "model_key": MODEL_KEY, "model": MODEL, "n_layers": N_LAYERS, "hidden": HIDDEN,
        "tp": TP, "pp": PP, "layer_split": LAYER_SPLIT,
        "n_req": N_REQ, "in_len": IN_LEN, "out_len": OUT_LEN}))
    print(f"model={MODEL_KEY} ({MODEL}) TP{TP}PP{PP} split={LAYER_SPLIT}", flush=True)

    port = _free_port()
    env = _build_env(trace_dir)

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
        "--profiler-config.profiler=torch",
        f"--profiler-config.torch_profiler_dir={trace_dir}",
    ]
    if MODEL_KEY in ("opt30b",):
        # base LM without a chat template: /v1/chat/completions 400s without one.
        # Same template the sweep infra uses (hetero_sweep.py).
        cmd += ["--chat-template", str(REPO / "planner" / "base_chat_template.jinja")]
    fout = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fout, stderr=subprocess.STDOUT,
                            cwd=str(REPO), preexec_fn=os.setsid)
    print(f"[vllm] pid={proc.pid} port={port}", flush=True)

    # Wait for ready
    deadline = time.time() + 900
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"[vllm] crashed rc={proc.returncode}", flush=True); return 1
        try: txt = log_path.read_text(errors="ignore")
        except: txt = ""
        if "Application startup complete" in txt:
            ready = True; break
        time.sleep(8)
    if not ready:
        print("[vllm] timeout", flush=True); proc.kill(); return 1
    print("[vllm] ready", flush=True)
    time.sleep(5)

    # Warm-up dummy request (1 req, 4 tokens)
    print("[warmup] 1 dummy request", flush=True)
    try:
        post(f"http://127.0.0.1:{port}/v1/completions",
             json.dumps({"model": MODEL, "prompt": "Hello", "max_tokens": 4,
                          "stream": False}).encode())
    except Exception as e:
        print(f"warmup err: {e}", flush=True)
    time.sleep(3)

    # Start profiling
    print("[profile] POST /start_profile", flush=True)
    s, body = post(f"http://127.0.0.1:{port}/start_profile")
    print(f"  -> {s}: {body[:200]}", flush=True)

    # Run perf workload
    print("[perf] running workload during profile capture", flush=True)
    prompt_path = out_root / "prompt.txt"
    template = ("Analyze PP overlap in heterogeneous inference. ")
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
    perf_env["HF_HUB_OFFLINE"] = "1"   # client tokenizer load: use cached gated 70B, no HF auth
    with (out_root / "perf.log").open("w") as f:
        subprocess.run(perf_cmd, env=perf_env, cwd=str(REPO),
                       stdout=f, stderr=subprocess.STDOUT, timeout=300)

    # Stop profiling
    print("[profile] POST /stop_profile", flush=True)
    s, body = post(f"http://127.0.0.1:{port}/stop_profile")
    print(f"  -> {s}: {body[:200]}", flush=True)
    print("[wait] for trace flush (30s)", flush=True)
    time.sleep(30)

    # Pull worker trace files
    print("[scp] pulling worker traces from /data/esca/.../traces", flush=True)
    subprocess.run(["scp", "-o", "BatchMode=yes", "-r",
                    f"esca@10.20.0.28:{trace_dir}/",
                    str(out_root / "traces_worker")],
                   timeout=600)

    # Cleanup
    try: os.killpg(proc.pid, signal.SIGINT); proc.wait(timeout=20)
    except:
        try: os.killpg(proc.pid, signal.SIGKILL)
        except: pass

    # List traces
    print("[trace files]", flush=True)
    for d in [out_root / "traces", out_root / "traces_worker"]:
        if d.exists():
            for f in sorted(d.glob("**/*.json*")):
                print(f"  {f}  ({f.stat().st_size} bytes)", flush=True)
    print(f"[done] artifacts in {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
