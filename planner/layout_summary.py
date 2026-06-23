"""Consolidate the GPU-count (layout) axis for one model across all measured
layouts (1+1, 2+2, 4+4): per (layout, n_req) show the measured champion, the
non-uniform gain (champion vs the same-topology uniform), and the planner's
zero-refit prediction (regret). Evidence for completeness across 1+1..4+4 and
for the thesis that the non-uniform-TP gain grows as the TP degree shrinks.

Usage: python planner/layout_summary.py --model 8b
"""
from __future__ import annotations
import argparse, dataclasses, glob, json, sys
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import perf_planner as P

REPO = HERE.parent
LAYOUTS = [(1, 1), (2, 2), (4, 4)]


def relayout(hw, hg, wg):
    bw, ada = hw.nodes[0][0], hw.nodes[-1][0]
    return dataclasses.replace(hw, nodes=((bw, hg), (ada, wg)))


def conc_dir(model, hg, wg):
    cands = [d for d in glob.glob(str(REPO / f"results/hetero_{hg}x{wg}_{model}_*"))
             if "_full_" not in d]
    return max(cands, key=lambda d: d.rsplit("_", 2)[-2:]) if cands else None


def load_cells(d):
    out = []
    for rj in glob.glob(str(REPO / d / "*/record.json")):
        try:
            recs = json.load(open(rj))
        except Exception:
            continue
        for e in (recs if isinstance(recs, list) else [recs]):
            if e.get("success") and e.get("tps", 0) > 0 and e.get("workload") == "balanced":
                out.append(e)
    return out


def topo(label):
    return label.split("_")[0]  # TP4PP1 / TP2PP2 / ...


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b")
    args = ap.parse_args()
    base_hw = P.load_hardware()
    m = P.MODELS[args.model]

    print(f"=== {args.model} — layout (GPU-count) axis, balanced ===")
    print(f"{'layout':6s} {'n':>4s}  {'meas champion':24s} {'tps':>6s}  "
          f"{'planner pick':22s} {'regret':>7s}")
    print("-" * 80)
    tpbias_by_layout = defaultdict(list)   # FFN/head bias gain within TP=world,pp=1
    ppskew_by_layout = defaultdict(list)   # layer-skew gain within best PP topo
    for hg, wg in LAYOUTS:
        d = conc_dir(args.model, hg, wg)
        if not d:
            print(f"{hg}+{wg}: (no data)")
            continue
        hw = relayout(base_hw, hg, wg)
        cells = load_cells(d)
        byn = defaultdict(list)
        for e in cells:
            byn[e["n_req"]].append(e)
        for n in sorted(byn):
            entries = byn[n]
            w = P.Workload(entries[0]["in_len"], entries[0]["out_len"], n)
            mc = max(entries, key=lambda e: e["tps"])
            # --- separate the two non-uniform mechanisms ---
            # TP-bias: among pp==1 (cross-node TP=world), best vs uniform
            tp1 = [e for e in entries if e["pp"] == 1]
            tp1u = [e for e in tp1 if "uniform" in e["label"]]
            if tp1 and tp1u:
                tpbias_by_layout[(hg, wg)].append(
                    max(e["tps"] for e in tp1) / tp1u[0]["tps"] - 1)
            # PP-skew: pick the best pp>1 topology family, best vs its uniform
            ppc = [e for e in entries if e["pp"] > 1]
            if ppc:
                best_topo = topo(max(ppc, key=lambda e: e["tps"])["label"])
                fam = [e for e in ppc if topo(e["label"]) == best_topo]
                famu = [e for e in fam if "uniform" in e["label"]]
                if famu:
                    ppskew_by_layout[(hg, wg)].append(
                        max(e["tps"] for e in fam) / famu[0]["tps"] - 1)
            # planner prediction (zero-refit, layout hw)
            preds = []
            for e in entries:
                cfg = P.Config(e["tp"], e["pp"], list(e["layer_split"]),
                               list(e["ffn_splits"]), list(e["head_splits"]),
                               list(e["kv_splits"]), e["label"])
                pr = P.predict(m, hw, w, cfg, overlap=(e["pp"] > 1))
                preds.append((e["label"], e["tps"], pr.get("tps", 0)))
            pick = max(preds, key=lambda x: x[2])
            regret = (mc["tps"] - pick[1]) / mc["tps"] * 100
            print(f"{hg}+{wg:<4d} {n:>4d}  {mc['label']:24s} {mc['tps']:6.0f}  "
                  f"{pick[0]:22s} {regret:>6.1f}%")
    print("-" * 80)
    print("Non-uniform gain by layout (max over n), two mechanisms separated:")
    print(f"  {'layout':8s} {'TP=world':>10s}  {'FFN-bias gain':>14s}  {'PP-skew gain':>13s}")
    for hg, wg in LAYOUTS:
        tb = tpbias_by_layout.get((hg, wg)); ps = ppskew_by_layout.get((hg, wg))
        if tb is None and ps is None:
            continue
        world = hg + wg
        tbs = f"{max(tb)*100:+.1f}%" if tb else "  —"
        pss = f"{max(ps)*100:+.1f}%" if ps else "  —"
        print(f"  {f'{hg}+{wg}':8s} {f'TP{world}':>10s}  {tbs:>14s}  {pss:>13s}")
    print("  thesis: FFN-bias gain should GROW as TP=world shrinks (4+4 TP8 → 1+1 TP2)")


if __name__ == "__main__":
    main()
