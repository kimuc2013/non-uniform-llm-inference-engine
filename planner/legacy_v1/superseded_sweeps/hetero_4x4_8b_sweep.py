"""8B 4+4 cross-node full topology sweep.

Same topology + workload structure as hetero_4x4_70b_sweep.py.
Llama-3.1-8B-Instruct: 32 layers, FFN 14336, 32 q-heads, 8 kv-heads.
"""
from __future__ import annotations
import argparse, json, os, signal, socket as sock, subprocess, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from planner.cluster_env import CFG

REPO = _REPO
PY = CFG.head_py
PERF = REPO / "perf" / "performance.py"
HEAD_IB = CFG.head_fabric_ip
RAY_ADDR = CFG.ray_address
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# 8B: 32 layers, FFN 14336, 32 q-heads, 8 kv-heads.
CONFIGS = [
    # TP8 PP1 single stage cross-node — uniform baseline + FFN/Head bias variants.
    # 8B: FFN 14336, q=32, kv=8 → uniform 1792 / 4 / 1 per rank.
    ("TP8PP1_uniform", 8, 1, [32],
     [1792] * 8, [4] * 8, [1] * 8),

    # FFN bias toward Blackwell (rank 0-3).
    ("TP8PP1_ffn_bias+25", 8, 1, [32],
     [2240]*4 + [1344]*4, [4] * 8, [1] * 8),       # 8960 + 5376 = 14336
    ("TP8PP1_ffn_bias+50", 8, 1, [32],
     [2688]*4 +  [896]*4, [4] * 8, [1] * 8),       # 10752 + 3584
    ("TP8PP1_ffn_bias+75", 8, 1, [32],
     [3136]*4 +  [448]*4, [4] * 8, [1] * 8),       # 12544 + 1792

    # TP4 PP2 — layer split sweep (each stage 4 GPUs uniform within)
    ("TP4PP2_layer_uniform_16-16", 4, 2, [16, 16],
     [3584] * 4, [8] * 4, [2] * 4),
    ("TP4PP2_layer_skew+2_18-14", 4, 2, [18, 14],
     [3584] * 4, [8] * 4, [2] * 4),
    ("TP4PP2_layer_skew+4_20-12", 4, 2, [20, 12],
     [3584] * 4, [8] * 4, [2] * 4),
    ("TP4PP2_layer_skew+6_22-10", 4, 2, [22, 10],
     [3584] * 4, [8] * 4, [2] * 4),
    ("TP4PP2_layer_skew+8_24-8", 4, 2, [24, 8],
     [3584] * 4, [8] * 4, [2] * 4),

    # TP2 PP4 — 4 stages, 2 per node
    ("TP2PP4_layer_uniform_8-8-8-8", 2, 4, [8, 8, 8, 8],
     [7168] * 2, [16] * 2, [4] * 2),
    ("TP2PP4_layer_blackbias_9-9-7-7", 2, 4, [9, 9, 7, 7],
     [7168] * 2, [16] * 2, [4] * 2),
    ("TP2PP4_layer_blackbias_10-10-6-6", 2, 4, [10, 10, 6, 6],
     [7168] * 2, [16] * 2, [4] * 2),

    # TP1 PP8 — 8 stages (TP=1, whole FFN/heads/kv per rank).
    ("TP1PP8_layer_uniform_4x8", 1, 8, [4]*8,
     [14336], [32], [8]),
    ("TP1PP8_layer_blackbias_5-5-5-5-3-3-3-3", 1, 8, [5,5,5,5,3,3,3,3],
     [14336], [32], [8]),
    ("TP1PP8_layer_blackbias_6-6-6-6-2-2-2-2", 1, 8, [6,6,6,6,2,2,2,2],
     [14336], [32], [8]),
]

WORKLOADS = {
    "balanced":      (512, 256, 128),
    "decode_heavy":  (128, 512, 128),
    "prefill_heavy": (1024, 128, 128),
}


def _free_port(start=29500):
    p = start
    while True:
        s = sock.socket()
        try: s.bind(("127.0.0.1", p)); s.close(); return p
        except OSError: p += 1
        finally:
            try: s.close()
            except: pass


def _build_env(layer_split, ffn_splits, head_splits, kv_splits, cell_label=""):
    env = os.environ.copy()
    conda = str(Path(CFG.head_py).parent)
    env["PATH"] = f"{conda}:/usr/local/cuda-12.9/bin:" + env.get("PATH", "")
    env["CC"] = "gcc-12"
    env["CXX"] = "g++-12"
    env["NVCC_CCBIN"] = "g++-12"
    env["CUDA_HOME"] = "/usr/local/cuda-12.9"
    env["VLLM_HOST_IP"] = HEAD_IB
    env["RAY_ADDRESS"] = RAY_ADDR
    env["VLLM_LOGGING_LEVEL"] = "WARNING"
    env["NCCL_DEBUG"] = "WARN"
    env["NCCL_SOCKET_IFNAME"] = f"{CFG.head_fabric_iface},{CFG.worker_fabric_iface}"
    env["NCCL_IB_HCA"] = "mlx5"
    env["NCCL_NET_GDR_LEVEL"] = "2"
    # NOTE: gloo single-interface only (head and worker can have different iface
    # names). FAST_COMM bypasses gloo metadata entirely — 4-byte signature over
    # NCCL. NCCL_IB_HCA + NCCL_NET_GDR_LEVEL applied via env above.
    env["VLLM_PP_FAST_COMM"] = "1"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_USE_FLASHINFER_MOE"] = "0"
    env["AUTO_TP_SPLIT"] = "0"
    env["AUTOSPLIT"] = "0"
    env["AUTO_PP_LAYER_PARTITION"] = "0"
    env["VLLM_PP_LAYER_PARTITION"] = ",".join(str(x) for x in layer_split)
    env["VLLM_TP_FFN_SPLITS"]  = ",".join(str(x) for x in ffn_splits)
    env["VLLM_TP_HEAD_SPLITS"] = ",".join(str(x) for x in head_splits)
    env["VLLM_TP_KV_SPLITS"]   = ",".join(str(x) for x in kv_splits)
    return env


def _apply_pp_overlap_env(env, tp, pp, n_reqs):
    """Launcher-validated set ONLY (broadcast_stream + microbatch + sizes)."""
    from launcher import pp_overlap_config as _ppc
    for k in ("VLLM_PP_SAMPLED_BROADCAST_STREAM", "VLLM_PP_MICROBATCH",
              "VLLM_PP_MICROBATCH_SIZE", "VLLM_PP_BATCH_QUEUE_SIZE",
              "VLLM_PP_OVERLAP", "VLLM_PP_FAST_COMM"):
        env.pop(k, None)
    if pp <= 1:
        return
    cfg = _ppc.auto_configure(num_reqs=n_reqs, pp_size=pp, tp_size=tp, model_name=MODEL)
    _ppc.apply_to_env(env, cfg)
    print(f"  [pp_overlap] mb={cfg.use_microbatch} mb_size={cfg.mb_size} "
          f"bq={cfg.bq} broadcast_stream={cfg.enable_broadcast_stream}", flush=True)


def launch_vllm(port, tp, pp, env, log_path, max_num_seqs=128):
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--tensor-parallel-size", str(tp),
        "--pipeline-parallel-size", str(pp),
        "--distributed-executor-backend", "ray",
        "--max-model-len", "4096",
        "--max-num-seqs", str(max_num_seqs),
        "--gpu-memory-utilization", "0.85",
        "--dtype", "bfloat16",
        "--port", str(port),
        "--host", "0.0.0.0",
        "--enable-chunked-prefill",
        "--attention-backend", "FLASH_ATTN",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(log_path, "w")
    return subprocess.Popen(cmd, env=env, stdout=fout, stderr=subprocess.STDOUT,
                            cwd=str(REPO), preexec_fn=os.setsid)


def wait_ready(log_path, port, timeout=900):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try: txt = log_path.read_text(errors="ignore")
        except: txt = ""
        if "Application startup complete" in txt:
            return True
        if any(k in txt for k in ("out of memory", "Failed core proc",
                                   "RuntimeError: ", "ValueError",
                                   "WorkerProc hit an exception",
                                   "CUBLAS_STATUS", "illegal memory access")):
            return False
        s = sock.socket()
        try:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        finally: s.close()
        time.sleep(6)
    return False


def run_perf(port, in_len, out_len, n_req, out_dir, timeout=900):
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / "prompt.txt"
    template = ("The following is a detailed analysis of large language model "
                "inference systems with a focus on heterogeneous GPU clusters. ")
    words_needed = int(in_len / 1.3)
    base = template.split()
    out_words = []
    while len(out_words) < words_needed:
        out_words.extend(base)
    prompt_path.write_text(" ".join(out_words[:words_needed]))
    cmd = [
        PY, str(PERF),
        "--base-url", f"http://127.0.0.1:{port}/v1",
        "--model", MODEL,
        "--prompt-file", str(prompt_path),
        "--requests", str(n_req),
        "--runs", "1",
        "--max-tokens", str(out_len),
        "--ignore-eos",
        "--output-csv", str(out_dir / "perf_runs.csv"),
        "--output-summary-csv", str(out_dir / "perf_summary.csv"),
    ]
    env = os.environ.copy()
    env["PATH"] = f"{Path(CFG.head_py).parent}:" + env.get("PATH", "")
    with (out_dir / "perf.log").open("w") as f:
        try:
            p = subprocess.run(cmd, env=env, cwd=str(REPO),
                               stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout, check=False)
            ok = p.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
    res = {"perf_ok": ok}
    sp = out_dir / "perf_summary.csv"
    if sp.exists():
        for line in sp.read_text().splitlines():
            if "," in line and not line.startswith("metric"):
                k, _, v = line.partition(",")
                try: res[k.strip()] = float(v.strip())
                except: pass
    return res


_ray_session = None
def _ensure_ray():
    global _ray_session
    if _ray_session is None:
        import ray
        if not ray.is_initialized():
            ray.init(address=RAY_ADDR, ignore_reinit_error=True)
        _ray_session = ray
    return _ray_session


def cleanup_pg():
    """Single persistent ray session — avoids session ID accumulation."""
    try:
        ray = _ensure_ray()
        for k in list(ray.util.placement_group_table().keys()):
            try:
                ray._private.worker.global_worker.core_worker.remove_placement_group(
                    ray.PlacementGroupID(bytes.fromhex(k))
                )
            except Exception:
                pass
    except Exception as e:
        print(f"cleanup_pg warn: {e}", flush=True)


_worker_node_id_cache = None
def _get_worker_node_id():
    global _worker_node_id_cache
    if _worker_node_id_cache is None:
        ray = _ensure_ray()
        for n in ray.nodes():
            if n.get('alive') and n['NodeManagerAddress'] == CFG.worker_fabric_ip:
                _worker_node_id_cache = n['NodeID']
                break
    return _worker_node_id_cache


def _kill_worker_vllm():
    try:
        ray = _ensure_ray()
        node_id = _get_worker_node_id()
        if not node_id:
            return
        @ray.remote(num_cpus=0.1, scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=node_id, soft=False))
        def kill_remote():
            import subprocess
            subprocess.run(['pkill','-9','-f','VLLM::'], capture_output=True)
            subprocess.run(['pkill','-9','-f','ray::RayWorkerProc'], capture_output=True)
            return "ok"
        ray.get(kill_remote.remote(), timeout=15)
    except Exception as e:
        print(f"_kill_worker_vllm warn: {e}", flush=True)


def _nuke_compile_cache():
    """Clear torch_compile_cache on head + worker between cells.

    AOT inductor cache is shape-keyed in a way that does NOT capture per-rank
    intermediate dim (FFN/head bias), so reusing prior-cell cache crashes Ada
    ranks. Clear forces fresh compile each cell.
    """
    import shutil
    head_cache = Path("/data/esca/.cache/vllm/torch_compile_cache")
    if head_cache.exists():
        for child in head_cache.iterdir():
            try: shutil.rmtree(child)
            except Exception: pass
    try:
        ray = _ensure_ray()
        node_id = _get_worker_node_id()
        if not node_id:
            return
        @ray.remote(num_cpus=0.1, scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=node_id, soft=False))
        def clean_remote():
            import shutil, os
            p = os.path.expanduser("~/.cache/vllm/torch_compile_cache")
            if not os.path.isdir(p): return 0
            n = 0
            for entry in os.listdir(p):
                ep = os.path.join(p, entry)
                try: shutil.rmtree(ep); n += 1
                except Exception: pass
            return n
        ray.get(clean_remote.remote(), timeout=30)
    except Exception as e:
        print(f"_nuke_compile_cache warn: {e}", flush=True)


def stop(proc, port):
    if proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGINT)
        except: pass
        try: proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except: pass
    subprocess.run(["pkill", "-9", "-f", f"api_server.*--port {port}"],
                   capture_output=True, check=False)
    _kill_worker_vllm()
    cleanup_pg()
    _nuke_compile_cache()
    time.sleep(45)  # longer inter-cell sleep — NCCL/PG settle cross-node


def _ensure_4x4_or_setup():
    try:
        verify_4x4()
        return
    except Exception as e:
        print(f"[sweep] cluster not 4+4 ({e}) — auto-setup", flush=True)
    from planner.cluster_setup_4x4 import ensure_4x4_cluster
    global _ray_session
    _ray_session = None
    ensure_4x4_cluster(force_restart=True)
    _ensure_ray()
    verify_4x4()


def verify_4x4():
    """Stale-tolerant: per-IP max alive node only (ignores transient duplicates
    during a worker rejoin)."""
    ray = _ensure_ray()
    by_ip: dict[str, float] = {}
    for n in ray.nodes():
        if not n.get('alive'): continue
        ip = n['NodeManagerAddress']
        g = n.get('Resources', {}).get('GPU', 0)
        by_ip[ip] = max(by_ip.get(ip, 0), g)
    head_gpu = by_ip.get(CFG.head_fabric_ip, 0)
    worker_gpu = by_ip.get(CFG.worker_fabric_ip, 0)
    total = head_gpu + worker_gpu
    print(f"[verify_4x4] (deduped) head={head_gpu}/{CFG.head_gpus} "
          f"worker={worker_gpu}/{CFG.worker_gpus} total={total}", flush=True)
    if head_gpu != CFG.head_gpus or worker_gpu != CFG.worker_gpus:
        raise RuntimeError(
            f"{CFG.head_gpus}+{CFG.worker_gpus} cluster check FAILED — "
            f"head={head_gpu} worker={worker_gpu}. Run cluster_setup_4x4.py --force."
        )


def _conditional_defensive_cleanup():
    """Run aggressive cleanup ONLY if cluster has dirty state."""
    import subprocess
    head_dirty = subprocess.run(
        ["bash", "-c", "ps -ef | grep -E 'VLLM::|ray::RayWorkerProc' | grep -v grep | wc -l"],
        capture_output=True, text=True).stdout.strip()
    head_dirty_count = int(head_dirty) if head_dirty.isdigit() else 0
    worker_dirty_count = 0
    try:
        ray = _ensure_ray()
        node_id = _get_worker_node_id()
        if node_id:
            @ray.remote(num_cpus=0.1, scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id, soft=False))
            def count_worker():
                import subprocess
                r = subprocess.run(["bash", "-c", "ps -ef | grep -E 'VLLM::|ray::RayWorkerProc' | grep -v grep | wc -l"],
                                   capture_output=True, text=True)
                return int(r.stdout.strip() or 0)
            worker_dirty_count = ray.get(count_worker.remote(), timeout=20)
    except Exception as e:
        print(f"[cleanup] worker dirty-check failed: {e}; assuming dirty", flush=True)
        worker_dirty_count = 999
    pg_count = 0
    try:
        ray = _ensure_ray()
        pg_count = len(ray.util.placement_group_table())
    except Exception: pass
    print(f"[cleanup] head_dirty={head_dirty_count} worker_dirty={worker_dirty_count} pgs={pg_count}", flush=True)
    if head_dirty_count == 0 and worker_dirty_count == 0 and pg_count == 0:
        print(f"[cleanup] cluster clean — skip defensive cleanup", flush=True)
        return
    print(f"[cleanup] dirty state — running defensive cleanup", flush=True)
    _kill_worker_vllm()
    cleanup_pg()
    _nuke_compile_cache()
    time.sleep(10)


def main():
    _ensure_4x4_or_setup()
    _conditional_defensive_cleanup()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", default="balanced,decode_heavy,prefill_heavy")
    ap.add_argument("--configs", default="all")
    args = ap.parse_args()
    wls = [w.strip() for w in args.workloads.split(",")]
    if args.configs == "all":
        cfgs = CONFIGS
    else:
        wanted = set(args.configs.split(","))
        cfgs = [c for c in CONFIGS if c[0] in wanted]

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = REPO / "results" / f"hetero_4x4_8b_full_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"OUT: {out_root}", flush=True)
    print(f"CONFIGS: {len(cfgs)}, WORKLOADS: {len(wls)}, "
          f"TOTAL CELLS: {len(cfgs) * len(wls)}", flush=True)

    runs = []
    for label, tp, pp, layer_split, ffn, head, kv in cfgs:
        for wl in wls:
            in_len, out_len, n_req = WORKLOADS[wl]
            cell = f"8b_{label}_{wl}"
            print(f"\n[{cell}] tp={tp} pp={pp} layers={layer_split} "
                  f"in={in_len} out={out_len} n={n_req}", flush=True)
            cell_dir = out_root / cell
            port = _free_port()
            env = _build_env(layer_split, ffn, head, kv, cell_label=label)
            _apply_pp_overlap_env(env, tp, pp, n_req)
            log_path = cell_dir / "vllm.log"
            proc = launch_vllm(port, tp, pp, env, log_path, max_num_seqs=128)
            ready = wait_ready(log_path, port, timeout=900)
            if not ready:
                rec = {"cell": cell, "label": label, "tp": tp, "pp": pp,
                       "layer_split": layer_split, "workload": wl,
                       "success": False, "reason": "server_did_not_become_ready"}
                stop(proc, port)
                (cell_dir / "record.json").write_text(json.dumps(rec, indent=2))
                runs.append(rec)
                print(f"  FAILED: not ready", flush=True)
                continue
            metrics = run_perf(port, in_len, out_len, n_req, cell_dir)
            stop(proc, port)
            rec = {
                "cell": cell, "label": label, "tp": tp, "pp": pp,
                "layer_split": layer_split,
                "ffn_splits": ffn, "head_splits": head, "kv_splits": kv,
                "workload": wl,
                "in_len": in_len, "out_len": out_len, "n_req": n_req,
                "success": bool(metrics.get("perf_ok")),
                "tps": metrics.get("total_wall_throughput_tok_s", 0.0),
                "runtime_s": metrics.get("total_request_time_s", 0.0),
                "ttft_ms": metrics.get("TTFT_ms_mean", 0.0),
                "itl_ms": metrics.get("itl_ms_mean", 0.0),
            }
            (cell_dir / "record.json").write_text(json.dumps(rec, indent=2))
            runs.append(rec)
            print(f"  done: success={rec['success']} tps={rec['tps']:.1f} "
                  f"runtime={rec['runtime_s']:.1f}s ttft={rec['ttft_ms']:.1f}ms",
                  flush=True)

    import csv
    vp = out_root / "all_runs.csv"
    with vp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "label", "tp", "pp", "layer_split", "workload",
                    "tps", "runtime_s", "ttft_ms", "success"])
        for r in runs:
            w.writerow([r["cell"], r["label"], r["tp"], r["pp"],
                        "-".join(str(x) for x in r["layer_split"]),
                        r["workload"],
                        f"{r.get('tps',0):.1f}", f"{r.get('runtime_s',0):.1f}",
                        f"{r.get('ttft_ms',0):.1f}", str(r["success"])])
    print(f"\nWrote {vp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
