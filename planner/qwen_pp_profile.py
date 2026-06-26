"""Root-cause qwen32b's TP4PP2 serving slowdown via vLLM's torch profiler.

Profiles qwen32b at TP4PP2 (the anomaly: ITL grows with batch, doesn't scale) vs
qwen32b at TP8 (normal: scales like other models) — SAME model/arch (QK-norm, 64
layers), only the parallelism differs. The decisive metric is the worker (Ada)
rank's GPU-busy fraction per decode step:
  - busy ~100% with bigger kernels  -> compute-bound (real qwen3 cost; arch gap)
  - large GPU-idle gap per step      -> CPU/dispatch/sync bound (serving overhead)

Per-rank traces (head + worker via scp) land under results/qwen_pp_profile/<name>/.
Run with the vllm_main python.
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
CHAT_TEMPLATE = str(_REPO / "planner" / "base_chat_template.jinja")
N_REQ, IN_LEN, OUT_LEN = 96, 512, 64   # saturating: where qwen TP4PP2 caps (~1465) and opt scales (~2607)

# Clean control: qwen TP4PP2 (anomaly) vs opt30b TP4PP2 (scales well) under the
# SAME low-overhead profiler settings. If opt's worker stays busy/overlapped while
# qwen's starves in SendRecv, the qwen-specific PP-overlap failure is confirmed.
CONFIGS = {
    "qwen_tp4pp2":   dict(model="Qwen/Qwen3-32B", tp=4, pp=2, layers=[32, 32],
                          ffn=[6400] * 4, head=[16] * 4, kv=[2] * 4, max_len=2048, chat=None),
    "opt30b_tp4pp2": dict(model="facebook/opt-30b", tp=4, pp=2, layers=[24, 24],
                          ffn=[7168] * 4, head=[14] * 4, kv=[14] * 4, max_len=2048, chat=CHAT_TEMPLATE),
}


def post(url, data=b""):
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.status, r.read().decode()


def profile_one(name, cfg, out_root):
    model = cfg["model"]
    cdir = out_root / name
    trace_dir = str(cdir / "traces")
    os.makedirs(trace_dir, exist_ok=True)
    subprocess.run(["ssh", "-o", "BatchMode=yes", WORKER, f"mkdir -p {trace_dir}"], timeout=20)
    env = HS._build_env(model, cfg["layers"], cfg["ffn"], cfg["head"], cfg["kv"])
    HS._apply_pp_overlap_env(env, cfg["tp"], cfg["pp"], N_REQ, model)   # same auto-tuner recipe as the sweep
    port = HS._free_port(29900)
    log = cdir / "vllm.log"
    cmd = [PY, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
           "--tensor-parallel-size", str(cfg["tp"]), "--pipeline-parallel-size", str(cfg["pp"]),
           "--distributed-executor-backend", "ray", "--max-model-len", str(cfg["max_len"]),
           "--max-num-seqs", str(N_REQ), "--gpu-memory-utilization", "0.85",
           "--dtype", "bfloat16", "--port", str(port), "--host", "0.0.0.0",
           "--enable-chunked-prefill", "--attention-backend", "FLASH_ATTN",
           "--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_dir": trace_dir,
                                            "torch_profiler_use_gzip": True,
                                            "torch_profiler_with_stack": False,
                                            "torch_profiler_record_shapes": False,
                                            "warmup_iterations": 5, "active_iterations": 40})]
    if cfg.get("chat"):
        cmd += ["--chat-template", cfg["chat"]]
    print(f"\n=== {name}: TP{cfg['tp']}PP{cfg['pp']} layers={cfg['layers']} port={port} ===", flush=True)
    fout = open(log, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fout, stderr=subprocess.STDOUT,
                            cwd=str(_REPO), preexec_fn=os.setsid)
    try:
        if not HS.wait_ready(log, port, timeout=2400):
            print(f"  [{name}] NOT READY", flush=True)
            return False
        print(f"  [{name}] ready; warmup", flush=True)
        try:
            post(f"http://127.0.0.1:{port}/v1/completions",
                 json.dumps({"model": model, "prompt": "Hello", "max_tokens": 4}).encode())
        except Exception as e:
            print(f"  warmup err: {e}", flush=True)
        time.sleep(3)
        print(f"  [{name}] /start_profile", flush=True)
        post(f"http://127.0.0.1:{port}/start_profile")
        r = HS.run_perf(model, port, IN_LEN, OUT_LEN, N_REQ, cdir / "perf")
        print(f"  [{name}] decode done tps={r.get('total_wall_throughput_tok_s',0):.0f} "
              f"itl={r.get('itl_ms_mean',0):.1f}ms", flush=True)
        print(f"  [{name}] /stop_profile", flush=True)
        post(f"http://127.0.0.1:{port}/stop_profile")
        time.sleep(30)   # trace flush
        print(f"  [{name}] scp worker traces", flush=True)
        subprocess.run(["scp", "-o", "BatchMode=yes", "-r",
                        f"{WORKER}:{trace_dir}/", str(cdir / "traces_worker")], timeout=600)
        for d in (cdir / "traces", cdir / "traces_worker"):
            if d.exists():
                for f in sorted(d.glob("**/*.json*")):
                    print(f"    {f.relative_to(cdir)}  ({f.stat().st_size} bytes)", flush=True)
        return True
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        time.sleep(12)


def main():
    out_root = _REPO / "results" / "qwen_pp_profile"
    out_root.mkdir(parents=True, exist_ok=True)
    for name, cfg in CONFIGS.items():
        profile_one(name, cfg, out_root)
    print("\n[done] analyze with: python planner/qwen_pp_profile_analyze.py", flush=True)


if __name__ == "__main__":
    main()
