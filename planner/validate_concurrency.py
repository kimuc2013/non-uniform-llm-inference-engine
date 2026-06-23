"""Validate the planner against concurrency sweeps, for ANY layout (1+1..4+4).

For each (model, workload, n_req): does the planner rank the measured configs
correctly? Reports per-n_req champion match + regret (predicted-champion's
measured TPS vs best measured TPS) + per-config MAPE.

The cost model is layout-parametric (hierarchical AR derives n_nodes/n_local
from the rank set), and the fitted engine params are per-link / layout-
independent. So validating a NON-4+4 layout with the SAME fitted params is a
generalization test: does a planner calibrated on 4+4 predict 2+2 / 1+1 without
re-fitting? Usage:
    python planner/validate_concurrency.py                  # 4+4 (default)
    python planner/validate_concurrency.py --head-gpus 2 --worker-gpus 2
"""
from __future__ import annotations
import argparse, dataclasses, glob, json, sys
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import perf_planner as P

REPO = HERE.parent
ALL_MODELS = ("8b", "70b", "mistral123b", "opt30b", "qwen32b")


def relayout(hw, hg, wg):
    if (hg, wg) == (hw.nodes[0][1], hw.nodes[-1][1]):
        return hw
    bw, ada = hw.nodes[0][0], hw.nodes[-1][0]
    return dataclasses.replace(hw, nodes=((bw, hg), (ada, wg)))


def conc_dir(model: str, hg: int, wg: int):
    """Newest concurrency sweep dir for (model, layout): hetero_<hg>x<wg>_<model>_<ts>."""
    cands = [d for d in glob.glob(str(REPO / f"results/hetero_{hg}x{wg}_{model}_*"))
             if "_full_" not in d]
    return max(cands, key=lambda d: d.rsplit("_", 2)[-2:]) if cands else None


def load_cells(d: str):
    out = []
    for rj in glob.glob(str(REPO / d / "*/record.json")):
        try:
            recs = json.load(open(rj))
        except Exception:
            continue
        if isinstance(recs, dict):
            recs = [recs]
        for e in recs:
            if e.get("success") and e.get("tps", 0) > 0:
                out.append(e)
    return out


def build_cfg(e) -> P.Config:
    return P.Config(
        tp=e["tp"], pp=e["pp"], layer_split=list(e["layer_split"]),
        ffn_splits=list(e["ffn_splits"]), head_splits=list(e["head_splits"]),
        kv_splits=list(e["kv_splits"]), label=e["label"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-gpus", type=int, default=4)
    ap.add_argument("--worker-gpus", type=int, default=4)
    args = ap.parse_args()
    hg, wg = args.head_gpus, args.worker_gpus

    hw = relayout(P.load_hardware(), hg, wg)
    model_dirs = {m: conc_dir(m, hg, wg) for m in ALL_MODELS}
    model_dirs = {m: d for m, d in model_dirs.items() if d}
    if not model_dirs:
        print(f"no hetero_{hg}x{wg}_* concurrency dirs found"); return 1

    print(f"layout {hg}+{wg}  (fitted params unchanged — generalization test)")
    print(f"{'model':12s} {'wl':13s} {'n':>4s}  {'meas champ':22s} {'pred champ':22s} "
          f"{'regret%':>7s} {'cfg-MAPE%':>9s}  match")
    print("-" * 110)
    grand = {"n": 0, "match": 0, "regret": [], "mape": []}
    by_n = defaultdict(lambda: {"n": 0, "match": 0, "regret": []})
    for model, d in model_dirs.items():
        m = P.MODELS[model]
        cells = load_cells(d)
        groups = defaultdict(list)
        for e in cells:
            groups[(e["workload"], e["n_req"])].append(e)
        for (wl, n), entries in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
            w = P.Workload(entries[0]["in_len"], entries[0]["out_len"], n)
            rows = []
            for e in entries:
                cfg = build_cfg(e)
                pr = P.predict(m, hw, w, cfg, overlap=(cfg.pp > 1))
                rows.append({"label": e["label"], "meas": e["tps"],
                             "pred": pr.get("tps", 0.0)})
            meas_champ = max(rows, key=lambda r: r["meas"])
            pred_champ = max(rows, key=lambda r: r["pred"])
            best_meas = meas_champ["meas"]
            regret = (best_meas - pred_champ["meas"]) / best_meas * 100
            mape = sum(abs(r["pred"] - r["meas"]) / r["meas"] for r in rows) / len(rows) * 100
            match = pred_champ["label"] == meas_champ["label"]
            grand["n"] += 1; grand["match"] += int(match)
            grand["regret"].append(regret); grand["mape"].append(mape)
            by_n[n]["n"] += 1; by_n[n]["match"] += int(match); by_n[n]["regret"].append(regret)
            print(f"{model:12s} {wl:13s} {n:>4d}  {meas_champ['label']:22s} "
                  f"{pred_champ['label']:22s} {regret:>7.1f} {mape:>9.0f}  "
                  f"{'OK' if match else 'MISS'}")
    print("-" * 110)
    g = grand
    print(f"OVERALL  champion {g['match']}/{g['n']}  "
          f"mean regret {sum(g['regret'])/len(g['regret']):.1f}%  "
          f"mean cfg-MAPE {sum(g['mape'])/len(g['mape']):.0f}%")
    print("\nBy concurrency (champion match + mean regret):")
    for n in sorted(by_n):
        b = by_n[n]
        print(f"  n={n:>3d}: {b['match']}/{b['n']} match, regret {sum(b['regret'])/len(b['regret']):.1f}%")


if __name__ == "__main__":
    main()
