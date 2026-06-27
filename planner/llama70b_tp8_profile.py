"""Profile llama70b TP8 decode to find the ~44ms the κ=1 model under-predicts
(measured 88ms vs predicted ~44ms at n=64). κ=1 charges KV at peak BW (measured
correct); the missing cost scales KV-like but isn't pure BW or attention compute.
Torch profiler → which kernels dominate the Ada-worker decode step.

Run with the vllm_main python. Analyze with qwen_pp_profile_analyze.py.
"""
from __future__ import annotations
import json, os, signal, sys, time, urllib.request, subprocess
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from planner.cluster_env import CFG
import planner.hetero_sweep as HS

PY = CFG.head_py
WORKER = "esca@10.20.0.28"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
CHAT = str(_REPO / "planner" / "base_chat_template.jinja")
TP, PP = 8, 1
LAYERS = [80]
FFN = [28672 // 8] * 8     # 3584
HEAD = [64 // 8] * 8       # 8
KV = [8 // 8] * 8          # 1
N_REQ, IN_LEN, OUT_LEN = 64, 256, 64


def post(url, data=b""):
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.status, r.read().decode()


def main():
    out = _REPO / "results" / "llama70b_tp8_profile"
    trace_dir = str(out / "traces")
    os.makedirs(trace_dir, exist_ok=True)
    subprocess.run(["ssh", "-o", "BatchMode=yes", WORKER, f"mkdir -p {trace_dir}"], timeout=20)
    env = HS._build_env(MODEL, LAYERS, FFN, HEAD, KV)
    port = HS._free_port(29950)
    log = out / "vllm.log"
    cmd = [PY, "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL,
           "--tensor-parallel-size", str(TP), "--pipeline-parallel-size", str(PP),
           "--distributed-executor-backend", "ray", "--max-model-len", "2048",
           "--max-num-seqs", str(N_REQ), "--gpu-memory-utilization", "0.85",
           "--dtype", "bfloat16", "--port", str(port), "--host", "0.0.0.0",
           "--enable-chunked-prefill", "--attention-backend", "FLASH_ATTN",
           "--chat-template", CHAT,
           "--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_dir": trace_dir,
                                            "torch_profiler_use_gzip": True,
                                            "torch_profiler_with_stack": False,
                                            "torch_profiler_record_shapes": False,
                                            "warmup_iterations": 5, "active_iterations": 40})]
    print(f"=== llama70b TP8PP1 profile  port={port} ===", flush=True)
    fout = open(log, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fout, stderr=subprocess.STDOUT,
                            cwd=str(_REPO), preexec_fn=os.setsid)
    try:
        if not HS.wait_ready(log, port, timeout=2400):
            print("NOT READY", flush=True); return
        print("ready; warmup", flush=True)
        try:
            post(f"http://127.0.0.1:{port}/v1/completions",
                 json.dumps({"model": MODEL, "prompt": "Hello", "max_tokens": 4}).encode())
        except Exception as e:
            print(f"warmup err: {e}", flush=True)
        time.sleep(3)
        print("/start_profile", flush=True)
        post(f"http://127.0.0.1:{port}/start_profile")
        r = HS.run_perf(MODEL, port, IN_LEN, OUT_LEN, N_REQ, out / "perf")
        print(f"decode done tps={r.get('total_wall_throughput_tok_s',0):.0f} "
              f"itl={r.get('itl_ms_mean',0):.1f}ms", flush=True)
        post(f"http://127.0.0.1:{port}/stop_profile")
        time.sleep(30)
        subprocess.run(["scp", "-o", "BatchMode=yes", "-r",
                        f"{WORKER}:{trace_dir}/", str(out / "traces_worker")], timeout=600)
        for d in (out / "traces", out / "traces_worker"):
            if d.exists():
                for f in sorted(d.glob("**/*.json*")):
                    print(f"  {f.relative_to(out)}  ({f.stat().st_size} bytes)", flush=True)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        time.sleep(10)
    print("[done] analyze: python planner/qwen_pp_profile_analyze.py results/llama70b_tp8_profile", flush=True)


if __name__ == "__main__":
    main()
