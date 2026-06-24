"""Fit the planner's free parameters to calibration_data.csv.

Free parameters (9):
  ar_latency_us       cross-node per-hop AllReduce latency
  ar_bw_gbs           cross-node AR algorithm bandwidth
  intra_ar_latency_us intra-node per-hop latency
  step_floor_ms       per-decode-step CPU floor (all topologies)
  c_mb_ms             per-microbatch dispatch cost
  c_chunk_ms          per-prefill-chunk overhead
  overlap_eta         PP overlap efficiency
  prefill_overlap     ρ: prefill/decode resource-overlap fraction (wall blend)
  kv_bw_scale         KV-read BW / weight BW (decmicro probe pins it ≈0.32: KV
                      read ~3× slower than weight streaming)

Fixed (anchored from physical derivation in hw_params.json, NOT fitted):
  GPU membw / tflops, p2p, mem capacities, prefill_ar_overlap, NVLink intra-AR.

Loss: median-of-relative-errors (robust to outliers like qwen TP8 cells).
Validation: leave-one-model-out (fit on 3 models, test on held-out model).
"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import planner.perf_planner as P


N_REQ_MAX = 100  # hard operating rule: above this the Ada small-partition rank
                 # OOMs → KV preemption/recompute thrashing, an unsupported
                 # regime the planner does not (and should not) model. Old
                 # n=128 sweeps predate this rule; exclude them from the fit.


# 'chat' is the HELD-OUT self-validation workload — never fit on it (keeps the
# generalization test honest). 'decmicro' (in32/out512) is a decode-shape
# calibration probe and IS used.
HELD_OUT_WORKLOADS = {"chat"}


def load_rows():
    rows = []
    for row in csv.DictReader(open(P.CALIB_CSV)):
        if row["model"] not in P.MODELS:
            continue
        if row.get("regime") == "stock":
            continue
        if row.get("workload") in HELD_OUT_WORKLOADS:
            continue
        if float(row["tps"]) <= 0:
            continue
        if int(row["n_req"]) > N_REQ_MAX:
            continue
        rows.append(row)
    return rows


def build_hw(x) -> P.HardwareSpec:
    base = json.loads(P.HW_PARAMS.read_text())
    # prefill TFLOPS level (2x revision) and prefill-AR-overlap are physical
    # constants baked into hw_params.json — NOT fitted (weakly identified, would
    # make the fit degenerate). Only the prefill/decode overlap ρ is fitted.
    bw = P.GpuType("blackwell", base["blackwell"]["eff_tflops_prefill"],
                   base["blackwell"]["eff_membw_decode_gbs"], base["blackwell"]["mem_gb"])
    ada = P.GpuType("ada", base["ada"]["eff_tflops_prefill"],
                    base["ada"]["eff_membw_decode_gbs"], base["ada"]["mem_gb"])
    return P.HardwareSpec(
        nodes=((bw, 4), (ada, 4)),
        ar_latency_us=x[0],
        ar_bw_gbs=x[1],
        intra_ar_latency_us=x[2],
        intra_ar_bw_gbs=60.0,
        p2p_latency_us=200.0,
        p2p_bw_gbs=10.0,
        overlap_eta=x[6],
        step_floor_ms=x[3],
        c_mb_ms=x[4],
        c_chunk_ms=x[5],
        prefill_overlap=x[7],
        prefill_ar_overlap=base.get("prefill_ar_overlap", 0.0),
        kv_bw_scale=x[8],
    )


def eval_params(x, rows, return_detail=False):
    """Phase-decomposed loss: fit decode cycle (itl), prefill (ttft) and TPS
    jointly. itl pins the decode model, ttft pins the prefill model — much
    stronger constraint than TPS alone (where errors can cancel)."""
    hw = build_hw(x)
    errs = []
    detail = []
    for row in rows:
        m = P.MODELS[row["model"]]
        w = P.Workload(int(row["in_len"]), int(row["out_len"]), int(row["n_req"]))
        try:
            cfg = P.parse_calib_config(row)
        except Exception:
            continue
        meas_tps = float(row["tps"])
        meas_itl = float(row.get("itl_ms") or 0)
        meas_ttft = float(row.get("ttft_ms") or 0)
        r = P.predict(m, hw, w, cfg, overlap=True)
        if not r["feasible"]:
            continue
        terms = []
        rel_tps = (r["tps"] - meas_tps) / meas_tps
        terms.append(abs(rel_tps))
        if meas_itl > 1:
            rel_itl = (r["t_cycle_ms"] - meas_itl) / meas_itl
            terms.append(abs(rel_itl))
        if meas_ttft > 1:
            # sweep reports MEAN ttft across n_req: requests finish prefill
            # staggered over [0, T_prefill] ⇒ mean ≈ half the prefill duration.
            # (Now a useful prefill-level constraint since prefill is no longer
            # the old ~10× overestimate.)
            pred_ttft = r["t_prefill_s"] * 1e3 / 2
            rel_ttft = (pred_ttft - meas_ttft) / meas_ttft
            terms.append(0.5 * abs(rel_ttft))
        errs.append(float(np.mean(terms)))
        detail.append((row["model"], row["label"], row["workload"],
                       meas_tps, r["tps"], rel_tps, int(row["n_req"])))
    if return_detail:
        return errs, detail
    if not errs:
        return 10.0
    return float(np.mean(np.clip(errs, 0, 1.0)))


def fit(rows):
    bounds = [
        (5, 500),       # ar_latency_us (inter-node IB; hierarchical AR now
                        #   charges only 2(n_nodes-1) of these, so it can be
                        #   well below the old 50us floor)
        (1, 30),        # ar_bw_gbs (effective cross-node AR bw; decode AR runs
                        #   far below peak IB)
        (5, 200),       # intra_ar_latency_us
        (0, 80),        # step_floor_ms
        (0, 20),        # c_mb_ms
        (0, 20),        # c_chunk_ms (physical per-chunk CPU dispatch, not a knob)
        (0.3, 1.0),     # overlap_eta
        (0.0, 1.0),     # prefill_overlap (ρ): prefill/decode resource overlap
        (0.3, 1.5),     # kv_bw_scale: KV-read BW relative to weight BW. <1 ⇒ KV
                        #   read slower than weights ⇒ steeper decode slope (the
                        #   microbenchmark shows the per-request slope is under-
                        #   charged ~25-40%).
    ]
    res = differential_evolution(
        eval_params, bounds, args=(rows,), maxiter=60, popsize=20,
        seed=42, tol=1e-4, workers=-1, polish=True)
    return res


def main():
    rows = load_rows()
    print(f"calibration rows: {len(rows)}")

    print("\n=== global fit (all 4 models) ===")
    res = fit(rows)
    x = res.x
    names = ["ar_latency_us", "ar_bw_gbs", "intra_ar_latency_us",
             "step_floor_ms", "c_mb_ms", "c_chunk_ms", "overlap_eta",
             "prefill_overlap", "kv_bw_scale"]
    for n, v in zip(names, x):
        print(f"  {n:22s} = {v:8.2f}")
    print(f"  robust loss = {res.fun:.4f}")

    errs, detail = eval_params(x, rows, return_detail=True)
    by_model = {}
    for mkey, label, wl, meas, pred, rel, nr in detail:
        by_model.setdefault(mkey, []).append(abs(rel))
    print("\n  MAPE per model (global fit):")
    for mkey, lst in sorted(by_model.items()):
        print(f"    {mkey:10s} n={len(lst):3d} MAPE={100*np.mean(lst):6.1f}%  "
              f"median={100*np.median(lst):6.1f}%")

    # Champion match + regret. Group by (model, workload, n_req): with the
    # concurrency axis in the data the champion CROSSES OVER with n_req (TP8 at
    # low load → TP4PP2 at high load), so lumping n_req would compare configs
    # that win at different operating points and is meaningless.
    print("\n  champion match + regret per (model, workload, n_req):")
    by_mw = {}
    for mkey, label, wl, meas, pred, rel, nr in detail:
        by_mw.setdefault((mkey, wl, nr), []).append((label, meas, pred))
    match = 0; total = 0; regrets = []; top3 = 0; rhos = []
    reg_by_n = {}
    for (mkey, wl, nr), lst in sorted(by_mw.items()):
        mc, mc_tps, _ = max(lst, key=lambda t: t[1])
        pc, _, _ = max(lst, key=lambda t: t[2])
        pc_meas = next(meas for lab, meas, _ in lst if lab == pc)
        regret = (mc_tps - pc_meas) / mc_tps * 100
        regrets.append(regret)
        ok = mc == pc
        match += ok; total += 1
        # top-3: is the true (measured) champion among the 3 best-predicted?
        pred_top3 = {lab for lab, _, _ in sorted(lst, key=lambda t: -t[2])[:3]}
        in_top3 = mc in pred_top3
        top3 += in_top3
        if len(lst) >= 3:
            rho = spearmanr([t[1] for t in lst], [t[2] for t in lst]).correlation
            if not math.isnan(rho):
                rhos.append(rho)
        reg_by_n.setdefault(nr, []).append((ok, regret))
        if not ok:   # print only misses to keep output readable
            print(f"    ✗ {mkey:10s} {wl:13s} n={nr:>3d} meas={mc[:24]:24s} "
                  f"pred={pc[:24]:24s} regret={regret:5.1f}% top3={'Y' if in_top3 else 'n'}")
    print(f"  champion match: {match}/{total}; top-3 hit: {top3}/{total}; "
          f"mean regret {np.mean(regrets):.1f}% median {np.median(regrets):.1f}% "
          f"max {np.max(regrets):.1f}%; mean Spearman ρ={np.mean(rhos):.2f}")
    print("  by n_req:  " + "  ".join(
        f"n={nr}:{sum(o for o,_ in v)}/{len(v)},reg{np.mean([r for _,r in v]):.0f}%"
        for nr, v in sorted(reg_by_n.items())))

    # Leave-one-model-out
    print("\n=== leave-one-model-out generalization ===")
    for held in sorted(P.MODELS.keys() - {"mistral123b"}):
        train = [r for r in rows if r["model"] != held]
        test = [r for r in rows if r["model"] == held]
        if not test:
            continue
        res_l = fit(train)
        errs_t, detail_t = eval_params(res_l.x, test, return_detail=True)
        tps_errs = [abs(rel) for _, _, _, _, _, rel, _ in detail_t]
        mape = 100 * np.mean(tps_errs) if tps_errs else float("nan")
        med = 100 * np.median(tps_errs) if tps_errs else float("nan")
        # champion match + regret on held-out, per (workload, n_req)
        by_wl = {}
        for mkey, label, wl, meas, pred, rel, nr in detail_t:
            by_wl.setdefault((wl, nr), []).append((label, meas, pred))
        cm = 0; regs = []
        for wl, lst in by_wl.items():
            mc, mc_tps, _ = max(lst, key=lambda t: t[1])
            pc, _, _ = max(lst, key=lambda t: t[2])
            pc_meas = next(meas for lab, meas, _ in lst if lab == pc)
            regs.append((mc_tps - pc_meas) / mc_tps * 100)
            cm += (mc == pc)
        print(f"  held-out {held:10s}: TPS-MAPE={mape:6.1f}% median={med:6.1f}% "
              f"champion {cm}/{len(by_wl)} regret mean={np.mean(regs):.1f}% max={np.max(regs):.1f}%")

    # Persist fitted params
    out = {
        "fitted": {n: float(v) for n, v in zip(names, x)},
        "loss": float(res.fun),
        "note": ("global fit on 4-model calibration + concurrency (n_req≤100), "
                 "hierarchical AR; robust clipped-MAE loss over tps+itl+ttft"),
    }
    (HERE / "fitted_params.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE/'fitted_params.json'}")


if __name__ == "__main__":
    main()
