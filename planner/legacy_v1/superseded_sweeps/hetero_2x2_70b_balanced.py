"""2+2 GPU (2 Blackwell head + 2 Ada worker) Llama-70B, balanced workload only.
4-way comparison: uniform TP / non-uniform TP / uniform PP / non-uniform PP.

Reuses the (validated, HF_HUB_OFFLINE-fixed) helpers from the mistral 4x4 sweep,
overriding MODEL=70B and skipping the 4x4 cluster check (ray is a 4-GPU 2+2 here).

PREREQ: ray must already be up as 2+2 — head CUDA_VISIBLE_DEVICES=0,1 --num-gpus=2,
worker CUDA_VISIBLE_DEVICES=2,3 --num-gpus=2 (so ray sees 2 Blackwell + 2 Ada,
ranks 0,1 = head Blackwell, 2,3 = worker Ada).

Llama-3.3-70B: 80 layers, FFN 28672, 64 q / 8 kv (GQA 8:1), hidden 8192.
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import planner.hetero_4x4_mistral123b_sweep as S

S.MODEL = "meta-llama/Llama-3.3-70B-Instruct"   # override the module-global model

N_REQ = 16          # 70B on 2 Ada (TP4 => 35GB/rank) is memory-tight; conservative & fair
IN_LEN, OUT_LEN = 512, 256

# label, tp, pp, layer_split, ffn_splits(len=tp), head_splits(len=tp), kv_splits(len=tp)
# ranks 0,1 = Blackwell(head, fast); 2,3 = Ada(worker, slow)
CONFIGS = [
    ("uniform_TP4",    4, 1, [80], [7168]*4,            [16]*4, [2]*4),
    # FFN bias ~2:1 (Blackwell:Ada), tile-aligned (x128): 2*9600+2*4736=28672
    ("nonuniform_TP4", 4, 1, [80], [9600, 9600, 4736, 4736], [16]*4, [2]*4),
    ("uniform_PP",     2, 2, [40, 40], [14336]*2,        [32]*2, [4]*2),
    # layer skew +12 (Blackwell stage gets more): 52+28=80
    ("nonuniform_PP",  2, 2, [52, 28], [14336]*2,        [32]*2, [4]*2),
]


def verify_4gpu():
    ray = S._ensure_ray()
    nodes = [n for n in ray.nodes() if n.get("alive")]
    by_ip = {}
    for n in nodes:
        by_ip[n["NodeManagerAddress"]] = max(by_ip.get(n["NodeManagerAddress"], 0),
                                              n.get("Resources", {}).get("GPU", 0))
    h = by_ip.get(S.CFG.head_fabric_ip, 0)
    w = by_ip.get(S.CFG.worker_fabric_ip, 0)
    print(f"[verify_2x2] head={h} worker={w} total={h+w}", flush=True)
    if not (h == 2 and w == 2):
        raise RuntimeError(f"expected 2+2 ray cluster, got head={h} worker={w}. "
                           f"Restart ray with --num-gpus=2 per node (CVD pinned).")


def main():
    verify_4gpu()
    S._conditional_defensive_cleanup()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = _REPO / "results" / f"hetero_2x2_70b_balanced_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"OUT: {out_root}\nMODEL: {S.MODEL}  n_req={N_REQ}  wl=balanced({IN_LEN}/{OUT_LEN})", flush=True)
    runs = []
    for label, tp, pp, ls, ffn, head, kv in CONFIGS:
        cell = f"70b_2x2_{label}_balanced"
        print(f"\n[{cell}] tp={tp} pp={pp} layers={ls} ffn={ffn[0]}:{ffn[-1]} n={N_REQ}", flush=True)
        cdir = out_root / cell
        port = S._free_port()
        env = S._build_env(ls, ffn, head, kv, cell_label=label)
        S._apply_pp_overlap_env(env, tp, pp, N_REQ)
        log = cdir / "vllm.log"
        proc = S.launch_vllm(port, tp, pp, env, log, max_num_seqs=N_REQ)
        rec = {"cell": cell, "label": label, "tp": tp, "pp": pp,
               "layer_split": ls, "ffn_splits": ffn, "head_splits": head, "kv_splits": kv}
        if not S.wait_ready(log, port, timeout=2400):
            S.stop(proc, port)
            rec.update(success=False, reason="not_ready")
            (cdir / "record.json").write_text(json.dumps(rec, indent=2))
            runs.append(rec); print("  FAILED: not ready", flush=True); continue
        m = S.run_perf(port, IN_LEN, OUT_LEN, N_REQ, cdir)
        S.stop(proc, port)
        rec.update(success=bool(m.get("perf_ok")),
                   tps=m.get("total_wall_throughput_tok_s", 0.0),
                   ttft_ms=m.get("TTFT_ms_mean", 0.0),
                   itl_ms=m.get("itl_ms_mean", 0.0),
                   runtime_s=m.get("total_request_time_s", 0.0))
        (cdir / "record.json").write_text(json.dumps(rec, indent=2))
        runs.append(rec)
        print(f"  done: success={rec['success']} tps={rec.get('tps',0):.1f} "
              f"itl={rec.get('itl_ms',0):.1f}ms ttft={rec.get('ttft_ms',0):.0f}ms", flush=True)
    vp = out_root / "all_runs.csv"
    with vp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "tp", "pp", "layer_split", "ffn_B:A", "tps", "itl_ms", "ttft_ms", "success"])
        for r in runs:
            w.writerow([r["label"], r["tp"], r["pp"], "-".join(map(str, r["layer_split"])),
                        f"{r['ffn_splits'][0]}:{r['ffn_splits'][-1]}",
                        f"{r.get('tps',0):.1f}", f"{r.get('itl_ms',0):.1f}",
                        f"{r.get('ttft_ms',0):.0f}", r.get("success")])
    print(f"\nWrote {vp}", flush=True)
    for r in runs:
        print(f"  {r['label']:16s} tps={r.get('tps',0):7.1f} itl={r.get('itl_ms',0):6.1f}ms "
              f"success={r.get('success')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
