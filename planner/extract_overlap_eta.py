"""Extraction of the EFFECTIVE cost-model pipeline-efficiency eta from a DEDICATED
torch-profiler run (NOT serving-result cells / throughput fits).

TWO-RUN IDENTIFICATION (PLANNER_MATH.md §9): pp=1 twin identifies the per-model ENGINE
FLOOR F = c1 − t_step (no eta term, no degeneracy); the pp=2 twin then identifies
eta = 1 − (c2 − F − b_max)/(b_rest + t_send) with F known. Both runs are dedicated
pre-serving calibration runs. With the MEASURED F (8B: 8.23 ms, 70B: 9.99 ms — engine
floor is ~model-size-independent software cost, as expected) the earlier artifacts
resolve: 70B eta = 0.87 (was 0.65 when the fit-descended floor 1.5 under-charged the
step); 8B raw eta = −0.14 → clamp 0 (was −0.97 before F; the small residue is per-mb
engine cost beyond the CUDA-submit c_mb). eta is still the cost model's EFFECTIVE
exposure coefficient, not the direct cross-stage concurrency (that is 56-78% by direct
trace analysis — a different, also-valid instrument).

eta is MODEL-DEPENDENT, so this reads the run's model/split/workload from
<out_dir>/run_meta.json and extracts eta for THAT model — the one being deployed.

Two ingredients, both from the fresh dedicated run (not eval cells):
  c      = the run's OWN measured step period (perf_summary itl; cross-checked below
           against the worker trace's steady inter-step gap).
  b_max, = per-stage decode busy/step from the MEASURED roofline (perf_planner
  b_rest   stage_time_decode = compute + intra-AR). eta must be defined w.r.t. the
           model's b since that is what the cost formula multiplies.
  eta = 1 - (c - b_max)/(b_rest + t_send)      (pp=2)

The kernel TIMELINE (compute-only union, comm excluded) is read as an independent
CORROBORATION of the structure (which stage is the bottleneck, GPU utilisation) — it
is not the eta source (compute-only b omits intra-AR, a different decomposition).

Usage: python extract_overlap_eta.py <verify_pp_overlap_dir> [n_req_override]
Prints: OVERLAP_ETA <eta>  + the full decomposition for audit.
"""
import csv, gzip, json, sys, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf_planner as P

OUT = Path(sys.argv[1])

# comm kernels SPIN-WAIT on an SM (occupied but not computing) -> counting them as
# "busy" inflates every stage to ~100%; a >1ms size filter instead DROPS Ada's sub-ms
# decode GEMVs and under-counts stage-1 ~9x. So: keep ALL kernels, exclude comm by name.
_COMM = ("nccl", "allreduce", "all_reduce", "broadcast", "sendrecv", "send", "recv",
         "reducescatter", "allgather", "all_gather", "reduce_scatter", "p2p")


def all_kernels(path, dur_us=1):
    import ijson
    ev = []; ncomm = 0
    try:
        with gzip.open(path, "rb") as f:
            for e in ijson.items(f, "traceEvents.item"):
                if e.get("ph") != "X" or e.get("cat") not in ("kernel", "gpu_op"):
                    continue
                if e.get("dur", 0) < dur_us:
                    continue
                nm = e.get("name", "").lower()
                if any(c in nm for c in _COMM):
                    ncomm += 1; continue
                ev.append((float(e["ts"]), float(e["dur"])))
    except (ijson.common.IncompleteJSONError, EOFError, OSError) as ex:
        print(f"    (truncated trace, using {len(ev)} kernels before EOF: {type(ex).__name__})")
    ev.sort()
    print(f"    (excluded {ncomm} comm kernels; kept {len(ev)} compute)")
    return ev


def union_busy(ev):
    tot = 0.0; cs = ce = None
    for ts, dur in ev:
        e = ts + dur
        if ce is None:
            cs, ce = ts, e
        elif ts <= ce:
            ce = max(ce, e)
        else:
            tot += ce - cs; cs, ce = ts, e
    if ce is not None:
        tot += ce - cs
    return tot


def busy_per_step(label, path, c_ms, skip_frac=0.25):
    """Trace-measured per-step compute-busy (ms) = compute-only union over the steady
    decode window (skip warmup+prefill) / steps in it. Corroboration only."""
    ev = all_kernels(path)
    if len(ev) < 50:
        print(f"[{label}] only {len(ev)} kernels — skip corroboration"); return None
    t0, t1 = ev[0][0], ev[-1][0] + ev[-1][1]
    win = [(ts, d) for ts, d in ev if ts >= t0 + skip_frac * (t1 - t0)]
    span = win[-1][0] + win[-1][1] - win[0][0]
    busy = union_busy(win)
    b = busy / 1e3 / (span / (c_ms * 1e3))
    print(f"[{label}] steady {span/1e3:.0f}ms, union-busy {busy/1e3:.0f}ms => "
          f"{b:.1f} ms/step ({busy/span:.0%} util)")
    return b


def read_itl_ms(out_dir):
    f = out_dir / "perf_summary.csv"
    if not f.exists():
        return None
    for row in csv.reader(open(f)):
        if row and row[0] in ("itl_ms_mean", "tpot_ms_mean"):
            return float(row[1])
    return None


def main():
    meta = json.loads((OUT / "run_meta.json").read_text()) if (OUT / "run_meta.json").exists() else {}
    mk = meta.get("model_key", "70b")
    tp = meta.get("tp", 4); pp = meta.get("pp", 2)
    split = meta.get("layer_split")
    n_req = int(sys.argv[2]) if len(sys.argv) > 2 else meta.get("n_req", 64)
    in_len, out_len = meta.get("in_len", 8), meta.get("out_len", 48)

    m = P.MODELS[mk]; hw = P.load_hardware()
    if not split:
        split = ([m.n_layers] if pp == 1 else
                 [m.n_layers // 2, m.n_layers - m.n_layers // 2])

    c = read_itl_ms(OUT)
    if not c:
        print("no perf_summary.csv — cannot get step period c"); return 1
    print(f"model={mk} TP{tp}PP{pp} split={split}  |  step period c = {c:.2f} ms "
          f"(fresh run itl)\n")

    if pp == 1:
        # FLOOR-IDENTIFICATION twin: c1 = t_step(roofline) + floor, NO eta term.
        # The engine host floor (scheduler+sampler+dispatch per step) is identified
        # per-model without degeneracy — no prior, no magic constant.
        w = P.Workload(n_req=n_req, in_len=in_len, out_len=out_len)
        ffn = [m.ffn_dim // tp] * tp
        head = [max(1, m.n_q // tp)] * tp
        kv = [max(1, m.n_kv // tp)] * tp
        cfg = P.Config(tp, 1, [m.n_layers], ffn, head, kv)
        t_step = P.stage_time_decode_ms(m, hw, w, cfg, 0, n_req)
        floor = c - t_step
        print(f"--- engine floor (TP-only identification) ---")
        print(f"  t_step(roofline TP{tp}) = {t_step:.2f} ms   floor = c - t_step = {floor:.2f} ms")
        print(f"ENGINE_FLOOR_MS {max(0.0, floor):.3f}  (model={mk}, raw {floor:.3f})")
        return 0

    # optional argv[3] = measured engine floor from the TP-only twin (else hw prior)
    floor_arg = float(sys.argv[3]) if len(sys.argv) > 3 else None

    # --- eta from the MODEL roofline b (what the cost formula multiplies) + measured c ---
    w = P.Workload(n_req=n_req, in_len=in_len, out_len=out_len)
    ffn = [m.ffn_dim // tp] * tp
    head = [max(1, m.n_q // tp)] * tp
    kv = [max(1, m.n_kv // tp)] * tp
    cfg = P.Config(tp, pp, split, ffn, head, kv)
    n_mb = min(pp, n_req); mb = n_req / n_mb
    busy = [n_mb * (P.stage_time_decode_ms(m, hw, w, cfg, s, mb) + hw.c_mb_ms) for s in range(pp)]
    b_max = max(busy); b_rest = sum(busy) - b_max
    t_send = (mb * m.hidden * P.B_A) / (hw.p2p_bw_gbs * 1e9) * 1e3 + hw.p2p_latency_us / 1e3
    # invert the SAME forward formula the planner uses (perf_planner predict, pp=2):
    #   c = b_max + (1-eta)*(b_rest + t_send) + step_floor   -> subtract step_floor.
    # (omitting it biased eta LOW by step_floor/(b_rest+t_send).)
    # step_floor: PREFER the per-model MEASURED engine floor (TP-only twin, argv[3]);
    # hw.step_floor_ms is only the degraded fallback.
    sf = floor_arg if floor_arg is not None else hw.step_floor_ms
    if floor_arg is not None:
        print(f"  (using MEASURED engine floor {sf:.2f} ms from the TP-only twin)")
    denom = b_rest + t_send
    eta = 1 - (c - sf - b_max) / denom if denom > 0 else 0.0

    print(f"--- eta (roofline b, model-consistent) ---")
    print(f"  roofline busy/stage = {[round(x,1) for x in busy]} ms   b_max={b_max:.1f} b_rest={b_rest:.1f}")
    print(f"  step_floor={sf:.1f}  t_send={t_send:.3f}  eta = 1-({c:.1f}-{sf:.1f}-{b_max:.1f})/({b_rest:.1f}+{t_send:.2f}) = {eta:.3f}")
    tc = b_max + (1 - eta) * b_rest + (1 - eta) * t_send * (pp - 1) + sf
    print(f"  check: model t_cycle at this eta = {tc:.1f} ms vs measured c = {c:.1f} ms")

    # --- corroboration: trace compute-only busy (independent of the roofline) ---
    print(f"\n--- trace corroboration (compute-only union) ---")
    try:
        head_tr = sorted((OUT / "traces").glob("rank[0-3].*.pt.trace.json.gz"),
                         key=lambda p: p.stat().st_size)
        wrk_tr = sorted((OUT / "traces_worker").glob("rank[4-7].*.pt.trace.json.gz"),
                        key=lambda p: p.stat().st_size)
        if head_tr and wrk_tr:
            b0 = busy_per_step("stage0 head(Blackwell)", head_tr[0], c)
            b1 = busy_per_step("stage1 worker(Ada)", wrk_tr[0], c)
            if b0 and b1:
                who = "worker(Ada)" if b1 > b0 else "head(BW)"
                agree = (b1 > b0) == (busy[1] > busy[0])
                print(f"  trace bottleneck = {who}; roofline bottleneck = "
                      f"{'worker(Ada)' if busy[1]>busy[0] else 'head(BW)'} -> "
                      f"{'AGREE' if agree else 'DISAGREE'}")
        else:
            print("  (traces unavailable — roofline eta stands on the measured c alone)")
    except Exception as ex:
        print(f"  (corroboration skipped: {type(ex).__name__})")

    print(f"\nOVERLAP_ETA {max(0.0, min(1.0, eta)):.3f}  (model={mk}, raw {eta:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
