"""A/B: does a deeper PP pipeline (smaller mb_size) rescue qwen32b TP4PP2?
qwen got the SAME auto-tuner config (mb_size=reqs/2, bq=2) as opt30b/70b which
scale perfectly, yet qwen's ITL never flattens. Test alternate mb_size at n=96.
Run from repo root with the vllm_main python.
"""
import json, os, signal, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import planner.hetero_sweep as HS
import planner.perf_planner as PP

MODEL = PP.MODELS["qwen32b"].name            # Qwen/Qwen3-32B
TP, PPSZ = 4, 2
LAYERS = [32, 32]
FFN = [25600 // 4] * 4
HEAD = [64 // 4] * 4
KV = [max(1, 8 // 4)] * 4
N_REQ = 96
IN_LEN, OUT_LEN = 512, 512                    # balanced workload shape

# (label, mb_size, bq)  broadcast_stream always on
TRIALS = [
    ("auto_mb48_bq2",   48, 2),               # the recipe qwen actually ran (n_mb=2)
    ("deep_mb24_bq2",   24, 2),               # n_mb=4, deeper pipe
    ("deep_mb24_bq4",   24, 4),               # n_mb=4, deeper bq
    ("deepest_mb12_bq2", 12, 2),              # n_mb=8
]
OUT = REPO / "results" / "qwen_pp_ab"
OUT.mkdir(parents=True, exist_ok=True)
results = []

for label, mb, bq in TRIALS:
    env = HS._build_env(MODEL, LAYERS, FFN, HEAD, KV)
    env["VLLM_PP_SAMPLED_BROADCAST_STREAM"] = "1"
    env["VLLM_PP_MICROBATCH"] = "1"
    env["VLLM_PP_MICROBATCH_SIZE"] = str(mb)
    env["VLLM_PP_BATCH_QUEUE_SIZE"] = str(bq)
    print(f"\n=== {label}: mb_size={mb} bq={bq} (n_mb={-(-N_REQ//mb)}) ===", flush=True)
    port = HS._free_port(29400)
    cdir = OUT / label
    log = cdir / "vllm.log"
    cdir.mkdir(parents=True, exist_ok=True)
    proc = HS.launch_vllm(MODEL, port, TP, PPSZ, env, log,
                          max_num_seqs=max(N_REQ, 16), max_model_len=4096)
    try:
        if not HS.wait_ready(log, port, timeout=2400):
            print(f"  [{label}] NOT READY", flush=True)
            results.append({"label": label, "mb": mb, "bq": bq, "ok": False, "reason": "not_ready"})
            continue
        r = HS.run_perf(MODEL, port, IN_LEN, OUT_LEN, N_REQ, cdir / f"n{N_REQ}")
        tps = r.get("total_wall_throughput_tok_s", 0.0)
        itl = r.get("itl_ms_mean", 0.0)
        print(f"  [{label}] tps={tps:.1f} itl={itl:.1f}ms ok={r.get('perf_ok')}", flush=True)
        results.append({"label": label, "mb": mb, "bq": bq, "ok": r.get("perf_ok"),
                        "tps": tps, "itl": itl})
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception as e:
            print(f"  kill warn: {e}", flush=True)
        time.sleep(12)   # detached script: foreground sleep is fine here, lets GPUs free

(OUT / "ab_results.json").write_text(json.dumps(results, indent=2))
print("\n=== SUMMARY (qwen32b TP4PP2 n=96, balanced) ===", flush=True)
print(f"  reference: opt30b same config = 2607 tps / 35.7ms ; qwen auto = 1465 tps / 64.1ms", flush=True)
for r in results:
    print(f"  {r['label']:18s} mb={r['mb']:2d} bq={r['bq']} -> tps={r.get('tps',0):.0f} itl={r.get('itl',0):.1f}ms", flush=True)
print(f"saved {OUT/'ab_results.json'}", flush=True)
