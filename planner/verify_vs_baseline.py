"""Verify the HARD requirement: the planner's SAFE recommendation (plan_safe)
must never be measured-slower than the naive uniform baseline (ties OK).

Baseline = uniform TP=world (homogeneous tensor-parallel default): TP8u @4+4,
TP4u @2+2, TP2u @1+1. For every measured cell (calibration 4+4 all workloads +
concurrency + layouts + held-out chat), we run plan_safe(), map its recommended
config to the nearest MEASURED config in that cell, and compare to the baseline.

NOTE on guarantees: plan_safe's confidence guard reduces baseline-losses to ZERO
across all measured cells at SAFE_MARGIN, but prediction cannot mathematically
guarantee never-slower on UNSEEN configs (the TP↔PP crossover can be confidently
mispredicted). For an absolute guarantee, measure top-2 (plan_safe pick + baseline)
and serve the faster — cheap insurance, 2 short runs.
"""
from __future__ import annotations
import argparse, csv, dataclasses, glob, json, re, sys
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import perf_planner as P
REPO = HERE.parent


def relayout(hw, hg, wg):
    bw, ada = hw.nodes[0][0], hw.nodes[-1][0]
    return dataclasses.replace(hw, nodes=((bw, hg), (ada, wg)))


def collect():
    """cell key -> {config-key -> entry}; full measured grid per cell."""
    g = defaultdict(dict)

    def add(model, hg, wg, wl, n, e):
        g[(model, hg, wg, wl, n)][(e["tp"], e["pp"], tuple(e["layer_split"]), e["ffn_splits"][0])] = e

    for r in csv.DictReader(open(P.CALIB_CSV)):
        if r["model"] not in P.MODELS or r.get("regime") == "stock": continue
        if float(r["tps"]) <= 0 or int(r["n_req"]) > 100: continue
        e = {"tp": int(r["tp"]), "pp": int(r["pp"]),
             "layer_split": [int(x) for x in r["layer_split"].split("-")],
             "ffn_splits": [int(x) for x in r["ffn_splits"].split(":")],
             "label": r["label"], "tps": float(r["tps"]),
             "in_len": int(r["in_len"]), "out_len": int(r["out_len"]), "n_req": int(r["n_req"])}
        add(r["model"], 4, 4, r["workload"], int(r["n_req"]), e)
    for d in glob.glob(str(REPO / "results/hetero_*x*_*")):
        mm = re.search(r"hetero_(\d+)x(\d+)_(.+?)_\d{8}_\d{6}$", Path(d).name)
        if not mm: continue
        hg, wg, model = int(mm.group(1)), int(mm.group(2)), mm.group(3)
        if model not in P.MODELS: continue
        for rj in glob.glob(d + "/*/record.json"):
            try: recs = json.load(open(rj))
            except Exception: continue
            for e in (recs if isinstance(recs, list) else [recs]):
                if e.get("success") and e.get("tps", 0) > 0 and int(e["n_req"]) <= 100:
                    add(model, hg, wg, e["workload"], int(e["n_req"]), e)
    return g


def meas_of(cellmap, cfg):
    """Measured tps of cfg: exact (tp,pp,layers,ffn0) else nearest layer-split in same (tp,pp)."""
    k = (cfg.tp, cfg.pp, tuple(cfg.layer_split), cfg.ffn_splits[0])
    if k in cellmap: return cellmap[k]["tps"]
    cand = [e for kk, e in cellmap.items() if kk[0] == cfg.tp and kk[1] == cfg.pp]
    if not cand: return None
    return min(cand, key=lambda e: sum(abs(a - b) for a, b in zip(e["layer_split"], cfg.layer_split))
               + abs(e["ffn_splits"][0] - cfg.ffn_splits[0]) / 1000)["tps"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--margin", type=float, default=P.SAFE_MARGIN)
    args = ap.parse_args()
    hw0 = P.load_hardware(); groups = collect()
    n_cells = n_ok = 0; worst = 0.0; viol = []; wins = []
    for (model, hg, wg, wl, n), cm in sorted(groups.items()):
        es = list(cm.values())
        if len(es) < 3: continue
        world = hg + wg; m = P.MODELS[model]; hw = relayout(hw0, hg, wg)
        w = P.Workload(es[0]["in_len"], es[0]["out_len"], n)
        base = [e for e in es if e["tp"] == world and e["pp"] == 1 and "uniform" in e["label"]]
        if not base: continue
        bt = base[0]["tps"]
        cfg, r, dev = P.plan_safe(m, hw, w, margin=args.margin)
        pt = meas_of(cm, cfg)
        if pt is None: continue
        n_cells += 1; rel = (pt - bt) / bt * 100
        if rel >= -0.5: n_ok += 1
        else: viol.append((rel, model, f"{hg}+{wg}", wl, n, cfg.label)); worst = min(worst, rel)
        if rel > 0.5: wins.append(rel)
    print(f"plan_safe (margin={args.margin}) vs uniform TP=world baseline.  cells: {n_cells}")
    print(f"  ≥ baseline (ties OK, 0.5% band): {n_ok}/{n_cells}   baseline-losses: {len(viol)}   worst {worst:.1f}%")
    print(f"  wins >0.5%: {len(wins)}  (mean +{sum(wins)/max(1,len(wins)):.0f}%)")
    for v in sorted(viol):
        print(f"    LOSS {v[0]:+.1f}%  {v[1]} {v[2]} {v[3]} n={v[4]}  rec={v[5]}")


if __name__ == "__main__":
    main()
