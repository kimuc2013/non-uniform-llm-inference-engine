"""Generalized heterogeneous TP/PP measurement sweep.

ONE script that replaces the per-model copies (hetero_4x4_{8b,70b,opt30b,qwen32b,
mistral123b}_sweep.py + hetero_2x2_70b_balanced.py). The engine is identical
across all of them; only (model, GPU layout, config grid, n_req) differ — those
are now parameters.

  python planner/hetero_sweep.py --model 70b                       # 4+4, full grid, 3 workloads
  python planner/hetero_sweep.py --model 70b --head-gpus 2 --worker-gpus 2 --workloads balanced
  python planner/hetero_sweep.py --model mistral123b --configs TP4PP2_skew+12 --dry-run

Model dims come from planner.perf_planner.MODELS (DRY). The config grid is
generated from (dims, GPU layout): TP=world (ffn-bias levels) + TP×PP
factorizations (layer-skew levels), Blackwell(head)-biased. Ranks [0,head_gpus)
are the fast node (Blackwell), [head_gpus, world) the slow node (Ada).

Requires ray already up with the matching layout (head_gpus on head node, etc.).
PP cells use the launcher pp_overlap auto-tuner recipe ONLY. Hard rules baked in:
CUDA graphs ON (no --enforce-eager), gpu_mem 0.85, HF offline (gated-model safe).
"""
from __future__ import annotations
import argparse, json, os, signal, socket as sock, subprocess, sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from planner.cluster_env import CFG
import planner.perf_planner as PP   # model dims (DRY)

REPO = _REPO
PY = CFG.head_py
PERF = REPO / "perf" / "performance.py"
HEAD_IB = CFG.head_fabric_ip
RAY_ADDR = CFG.ray_address

# per-model default concurrency (matches the original per-model sweeps)
DEFAULT_NREQ = {"8b": 128, "70b": 128, "qwen32b": 128, "opt30b": 64, "mistral123b": 96}
WORKLOAD_SHAPE = {"balanced": (512, 256), "decode_heavy": (128, 512), "prefill_heavy": (1024, 128)}


# ----------------------------------------------------------------------------
# Config-grid generator  (replaces the hand-written per-model CONFIGS lists)
# ----------------------------------------------------------------------------
def _r128(x):
    return max(128, int(round(x / 128.0)) * 128)


def gen_configs(model_key, head_gpus, worker_gpus,
                ffn_bias_pct=(25, 50, 75), skew_levels=(4, 8, 12, 16)):
    """Standard topology grid for (model, head+worker GPU layout).
    ranks [0,head_gpus)=Blackwell(fast), [head_gpus,world)=Ada(slow)."""
    m = PP.MODELS[model_key]
    world = head_gpus + worker_gpus
    L, FFN, NQ, NKV, G = m.n_layers, m.ffn_dim, m.n_q, m.n_kv, m.gqa_group
    cfgs = []   # (label, tp, pp, layer_split, ffn[tp], head[tp], kv[tp])

    for pp in (1, 2, 4, 8):
        if world % pp or pp > L:
            continue
        tp = world // pp
        if NQ % tp or (NKV % tp and NKV < tp):
            continue
        head_u = NQ // tp
        kv_u = max(1, NKV // tp)
        ffn_u = FFN // tp
        if pp == 1:
            # TP=world spanning both nodes → FFN bias (Blackwell ranks get more)
            cfgs.append((f"TP{tp}PP1_uniform", tp, 1, [L], [ffn_u]*tp, [head_u]*tp, [kv_u]*tp))
            for b in ffn_bias_pct:
                bw = _r128(ffn_u * (1 + b/100))
                rem = FFN - head_gpus * bw
                if rem <= 0 or worker_gpus == 0:
                    continue
                ada = rem // worker_gpus
                ada = (ada // 128) * 128
                if ada < 128 or head_gpus*bw + worker_gpus*ada != FFN:
                    # fix remainder onto first slow rank if needed
                    pass
                ffn = [bw]*head_gpus + [ada]*worker_gpus
                if sum(ffn) != FFN:
                    ffn[head_gpus] += FFN - sum(ffn)   # absorb remainder on first Ada rank
                if min(ffn) < 128:
                    continue
                cfgs.append((f"TP{tp}PP1_ffn_bias+{b}", tp, 1, [L], ffn, [head_u]*tp, [kv_u]*tp))
        else:
            # TP×PP: stage s on ranks [s*tp,(s+1)*tp); first head_gpus//tp stages
            # are Blackwell. Layer-skew shifts layers Ada→Blackwell stages.
            n_bw = head_gpus // tp
            n_ada = pp - n_bw
            if n_bw < 1 or n_ada < 1:
                continue
            base = L // pp
            rem = L - base*pp
            uni = [base + (1 if s < rem else 0) for s in range(pp)]
            cfgs.append((f"TP{tp}PP{pp}_uniform", tp, pp, uni, [ffn_u]*tp, [head_u]*tp, [kv_u]*tp))
            for s in skew_levels:
                if s * n_ada > base * n_ada:      # can't take more than Ada has
                    continue
                ls = [uni[i] + (s if i < n_bw else -(s*n_bw)//n_ada) for i in range(pp)]
                if sum(ls) != L:
                    ls[0] += L - sum(ls)
                if min(ls) < 1:
                    continue
                cfgs.append((f"TP{tp}PP{pp}_skew+{s}", tp, pp, ls, [ffn_u]*tp, [head_u]*tp, [kv_u]*tp))
    return cfgs


# ----------------------------------------------------------------------------
# Engine  (identical to the per-model sweeps; parameterized by model + layout)
# ----------------------------------------------------------------------------
def _free_port(start=29400):
    p = start
    while True:
        s = sock.socket()
        try:
            s.bind(("127.0.0.1", p)); s.close(); return p
        except OSError:
            p += 1
        finally:
            try: s.close()
            except: pass


def _build_env(model, layer_split, ffn, head, kv):
    env = os.environ.copy()
    conda = str(Path(CFG.head_py).parent)
    env["PATH"] = f"{conda}:/usr/local/cuda-12.9/bin:" + env.get("PATH", "")
    env["CC"] = "gcc-12"; env["CXX"] = "g++-12"; env["NVCC_CCBIN"] = "g++-12"
    env["CUDA_HOME"] = "/usr/local/cuda-12.9"
    env["VLLM_HOST_IP"] = HEAD_IB; env["RAY_ADDRESS"] = RAY_ADDR
    env["VLLM_LOGGING_LEVEL"] = "WARNING"
    env["HF_HUB_OFFLINE"] = "1"; env["TRANSFORMERS_OFFLINE"] = "1"   # gated-model safe, no Hub throttle
    env["NCCL_DEBUG"] = "WARN"
    env["NCCL_SOCKET_IFNAME"] = f"{CFG.head_fabric_iface},{CFG.worker_fabric_iface}"
    env["NCCL_IB_HCA"] = "mlx5"; env["NCCL_NET_GDR_LEVEL"] = "2"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"; env["VLLM_USE_FLASHINFER_MOE"] = "0"
    env["AUTO_TP_SPLIT"] = "0"; env["AUTOSPLIT"] = "0"; env["AUTO_PP_LAYER_PARTITION"] = "0"
    env["VLLM_PP_LAYER_PARTITION"] = ",".join(map(str, layer_split))
    env["VLLM_TP_FFN_SPLITS"] = ",".join(map(str, ffn))
    env["VLLM_TP_HEAD_SPLITS"] = ",".join(map(str, head))
    env["VLLM_TP_KV_SPLITS"] = ",".join(map(str, kv))
    return env


def _apply_pp_overlap_env(env, tp, pp, n_reqs, model):
    from launcher import pp_overlap_config as _ppc
    for k in ("VLLM_PP_SAMPLED_BROADCAST_STREAM", "VLLM_PP_MICROBATCH",
              "VLLM_PP_MICROBATCH_SIZE", "VLLM_PP_BATCH_QUEUE_SIZE",
              "VLLM_PP_OVERLAP", "VLLM_PP_FAST_COMM"):
        env.pop(k, None)
    if pp <= 1:
        return
    cfg = _ppc.auto_configure(num_reqs=n_reqs, pp_size=pp, tp_size=tp, model_name=model)
    _ppc.apply_to_env(env, cfg)
    print(f"  [pp_overlap] mb={cfg.use_microbatch} mb_size={cfg.mb_size} "
          f"bq={cfg.bq} broadcast_stream={cfg.enable_broadcast_stream}", flush=True)


def launch_vllm(model, port, tp, pp, env, log_path, max_num_seqs):
    cmd = [PY, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
           "--tensor-parallel-size", str(tp), "--pipeline-parallel-size", str(pp),
           "--distributed-executor-backend", "ray", "--max-model-len", "4096",
           "--max-num-seqs", str(max_num_seqs), "--gpu-memory-utilization", "0.85",
           "--dtype", "bfloat16", "--port", str(port), "--host", "0.0.0.0",
           "--enable-chunked-prefill", "--attention-backend", "FLASH_ATTN"]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(log_path, "w")
    return subprocess.Popen(cmd, env=env, stdout=fout, stderr=subprocess.STDOUT,
                            cwd=str(REPO), preexec_fn=os.setsid)


def wait_ready(log_path, port, timeout=4200):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try: txt = log_path.read_text(errors="ignore")
        except: txt = ""
        if "Application startup complete" in txt:
            return True
        if any(k in txt for k in ("out of memory", "Failed core proc", "RuntimeError: ",
                                   "ValueError", "WorkerProc hit an exception",
                                   "CUBLAS_STATUS", "illegal memory access",
                                   "LocalEntryNotFoundError", "GatedRepoError")):
            return False
        s = sock.socket()
        try:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        finally: s.close()
        time.sleep(8)
    return False


def run_perf(model, port, in_len, out_len, n_req, out_dir, timeout=900):
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt = out_dir / "prompt.txt"
    base = ("The following is a detailed analysis of large language model inference "
            "systems with a focus on heterogeneous GPU clusters. ").split()
    words = []
    while len(words) < int(in_len/1.3):
        words.extend(base)
    prompt.write_text(" ".join(words[:int(in_len/1.3)]))
    cmd = [PY, str(PERF), "--base-url", f"http://127.0.0.1:{port}/v1", "--model", model,
           "--prompt-file", str(prompt), "--requests", str(n_req), "--runs", "1",
           "--max-tokens", str(out_len), "--ignore-eos",
           "--output-csv", str(out_dir/"perf_runs.csv"),
           "--output-summary-csv", str(out_dir/"perf_summary.csv")]
    env = os.environ.copy()
    env["PATH"] = f"{Path(CFG.head_py).parent}:" + env.get("PATH", "")
    env["HF_HUB_OFFLINE"] = "1"; env["TRANSFORMERS_OFFLINE"] = "1"   # perf tool tokenizer offline
    with (out_dir/"perf.log").open("w") as f:
        try:
            p = subprocess.run(cmd, env=env, cwd=str(REPO), stdout=f,
                               stderr=subprocess.STDOUT, timeout=timeout, check=False)
            ok = p.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
    res = {"perf_ok": ok}
    sp = out_dir/"perf_summary.csv"
    if sp.exists():
        for line in sp.read_text().splitlines():
            if "," in line and not line.startswith("metric"):
                k, _, v = line.partition(",")
                try: res[k.strip()] = float(v.strip())
                except: pass
    return res


_ray = None
def _ensure_ray():
    global _ray
    if _ray is None:
        import ray
        if not ray.is_initialized():
            ray.init(address=RAY_ADDR, ignore_reinit_error=True)
        _ray = ray
    return _ray


def _worker_node_id():
    ray = _ensure_ray()
    for n in ray.nodes():
        if n.get("alive") and n["NodeManagerAddress"] == CFG.worker_fabric_ip:
            return n["NodeID"]
    return None


def _kill_worker_vllm():
    try:
        ray = _ensure_ray(); nid = _worker_node_id()
        if not nid: return
        @ray.remote(num_cpus=0.1, scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=nid, soft=False))
        def k():
            import subprocess
            subprocess.run(["pkill","-9","-f","VLLM::"], capture_output=True)
            subprocess.run(["pkill","-9","-f","ray::RayWorkerProc"], capture_output=True)
            return "ok"
        ray.get(k.remote(), timeout=15)
    except Exception as e:
        print(f"_kill_worker_vllm warn: {e}", flush=True)


def cleanup_pg():
    try:
        ray = _ensure_ray()
        for k in list(ray.util.placement_group_table().keys()):
            try:
                ray._private.worker.global_worker.core_worker.remove_placement_group(
                    ray.PlacementGroupID(bytes.fromhex(k)))
            except Exception: pass
    except Exception as e:
        print(f"cleanup_pg warn: {e}", flush=True)


def _nuke_compile_cache():
    import shutil
    hc = Path("/data/esca/.cache/vllm/torch_compile_cache")
    if hc.exists():
        for c in hc.iterdir():
            try: shutil.rmtree(c)
            except Exception: pass
    try:
        ray = _ensure_ray(); nid = _worker_node_id()
        if not nid: return
        @ray.remote(num_cpus=0.1, scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=nid, soft=False))
        def c():
            import shutil, os
            p = os.path.expanduser("~/.cache/vllm/torch_compile_cache")
            if os.path.isdir(p):
                for e in os.listdir(p):
                    try: shutil.rmtree(os.path.join(p, e))
                    except Exception: pass
            return 0
        ray.get(c.remote(), timeout=30)
    except Exception as e:
        print(f"_nuke_compile_cache warn: {e}", flush=True)


def stop(proc, port):
    if proc.poll() is None:
        try: os.killpg(proc.pid, signal.SIGINT)
        except: pass
        try: proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except: pass
    subprocess.run(["pkill","-9","-f",f"api_server.*--port {port}"], capture_output=True, check=False)
    _kill_worker_vllm(); cleanup_pg(); _nuke_compile_cache()
    time.sleep(45)


def verify_layout(head_gpus, worker_gpus):
    ray = _ensure_ray()
    by_ip = {}
    for n in ray.nodes():
        if n.get("alive"):
            ip = n["NodeManagerAddress"]
            by_ip[ip] = max(by_ip.get(ip, 0), n.get("Resources", {}).get("GPU", 0))
    h = by_ip.get(CFG.head_fabric_ip, 0); w = by_ip.get(CFG.worker_fabric_ip, 0)
    print(f"[verify_layout] head={h}/{head_gpus} worker={w}/{worker_gpus}", flush=True)
    if h != head_gpus or w != worker_gpus:
        raise RuntimeError(f"ray layout mismatch: need head={head_gpus} worker={worker_gpus}, "
                           f"got head={h} worker={w}. Restart ray with matching --num-gpus per node.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(PP.MODELS))
    ap.add_argument("--head-gpus", type=int, default=4)
    ap.add_argument("--worker-gpus", type=int, default=4)
    ap.add_argument("--workloads", default="balanced,decode_heavy,prefill_heavy")
    ap.add_argument("--configs", default="all", help="comma-sep label substrings, or 'all'")
    ap.add_argument("--n-req", type=int, default=0, help="override per-model default")
    ap.add_argument("--dry-run", action="store_true", help="print config grid, no launch")
    args = ap.parse_args()

    model = PP.MODELS[args.model].name
    n_req = args.n_req or DEFAULT_NREQ.get(args.model, 96)
    wls = [w.strip() for w in args.workloads.split(",")]
    grid = gen_configs(args.model, args.head_gpus, args.worker_gpus)
    if args.configs != "all":
        want = args.configs.split(",")
        grid = [c for c in grid if any(w in c[0] for w in want)]

    print(f"MODEL={model}  layout={args.head_gpus}+{args.worker_gpus}  n_req={n_req}", flush=True)
    print(f"CONFIGS={len(grid)}  WORKLOADS={wls}  CELLS={len(grid)*len(wls)}", flush=True)
    if args.dry_run:
        m = PP.MODELS[args.model]
        for lab, tp, pp, ls, ffn, hd, kv in grid:
            ok = (sum(ffn) == m.ffn_dim and sum(ls) == m.n_layers
                  and all(x % 128 == 0 for x in ffn) and len(ffn) == tp and len(ls) == pp)
            print(f"  {'OK ' if ok else 'BAD'} {lab:22s} tp{tp}pp{pp} L={ls} ffn={ffn} "
                  f"(Σffn={sum(ffn)} ΣL={sum(ls)})")
        return 0

    verify_layout(args.head_gpus, args.worker_gpus)
    _conditional_cleanup()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = REPO / "results" / f"hetero_{args.head_gpus}x{args.worker_gpus}_{args.model}_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"OUT: {out_root}", flush=True)
    runs = []
    for label, tp, pp, ls, ffn, head, kv in grid:
        for wl in wls:
            in_len, out_len = WORKLOAD_SHAPE[wl]
            cell = f"{args.model}_{label}_{wl}"
            print(f"\n[{cell}] tp={tp} pp={pp} L={ls} ffn={ffn[0]}:{ffn[-1]} in={in_len} out={out_len} n={n_req}", flush=True)
            cdir = out_root / cell
            port = _free_port()
            env = _build_env(model, ls, ffn, head, kv)
            _apply_pp_overlap_env(env, tp, pp, n_req, model)
            proc = launch_vllm(model, port, tp, pp, env, cdir/"vllm.log", max_num_seqs=max(n_req, 16))
            rec = {"cell": cell, "label": label, "tp": tp, "pp": pp, "layer_split": ls,
                   "ffn_splits": ffn, "head_splits": head, "kv_splits": kv, "workload": wl,
                   "in_len": in_len, "out_len": out_len, "n_req": n_req}
            if not wait_ready(cdir/"vllm.log", port):
                stop(proc, port); rec.update(success=False, reason="not_ready")
                (cdir/"record.json").write_text(json.dumps(rec, indent=2)); runs.append(rec)
                print("  FAILED: not ready", flush=True); continue
            mtr = run_perf(model, port, in_len, out_len, n_req, cdir)
            stop(proc, port)
            rec.update(success=bool(mtr.get("perf_ok")), tps=mtr.get("total_wall_throughput_tok_s", 0.0),
                       ttft_ms=mtr.get("TTFT_ms_mean", 0.0), itl_ms=mtr.get("itl_ms_mean", 0.0),
                       runtime_s=mtr.get("total_request_time_s", 0.0))
            (cdir/"record.json").write_text(json.dumps(rec, indent=2)); runs.append(rec)
            print(f"  done: success={rec['success']} tps={rec.get('tps',0):.1f} itl={rec.get('itl_ms',0):.1f}ms", flush=True)
    import csv
    with (out_root/"all_runs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell","label","tp","pp","layer_split","workload","tps","itl_ms","ttft_ms","success"])
        for r in runs:
            w.writerow([r["cell"], r["label"], r["tp"], r["pp"], "-".join(map(str, r["layer_split"])),
                        r["workload"], f"{r.get('tps',0):.1f}", f"{r.get('itl_ms',0):.1f}",
                        f"{r.get('ttft_ms',0):.0f}", r.get("success")])
    print(f"\nWrote {out_root/'all_runs.csv'}", flush=True)
    return 0


def _conditional_cleanup():
    import subprocess
    hd = subprocess.run(["bash","-c","ps -ef|grep -E 'VLLM::|ray::RayWorkerProc'|grep -v grep|wc -l"],
                        capture_output=True, text=True).stdout.strip()
    wd = 0
    try:
        ray = _ensure_ray(); nid = _worker_node_id()
        if nid:
            @ray.remote(num_cpus=0.1, scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(node_id=nid, soft=False))
            def c():
                import subprocess
                return int(subprocess.run(["bash","-c","ps -ef|grep -E 'VLLM::|ray::RayWorkerProc'|grep -v grep|wc -l"],
                                          capture_output=True, text=True).stdout.strip() or 0)
            wd = ray.get(c.remote(), timeout=20)
    except Exception:
        wd = 999
    pg = 0
    try: pg = len(_ensure_ray().util.placement_group_table())
    except Exception: pass
    print(f"[cleanup] head_dirty={hd} worker_dirty={wd} pgs={pg}", flush=True)
    # NOTE: another user's VLLM:: procs on the worker also count as 'dirty' here,
    # but esca lacks permission to kill them so they are unaffected (pkill no-ops).
    if str(hd) == "0" and wd == 0 and pg == 0:
        print("[cleanup] clean — skip", flush=True); return
    _kill_worker_vllm(); cleanup_pg(); _nuke_compile_cache(); time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
