"""MAPE breakdown of the current planner over the full calibration CSV,
sliced by (model, topology family, n_req) to locate where error concentrates."""
import csv, sys
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
import planner.perf_planner as P
import numpy as np

hw = P.load_hardware()

def topo(label):
    for t in ("TP8PP1", "TP4PP2", "TP2PP4", "TP1PP8"):
        if label.startswith(t): return t
    return "?"

rows = [r for r in csv.DictReader(open(P.CALIB_CSV))
        if r["model"] in P.MODELS and r.get("regime") != "stock" and float(r["tps"]) > 0]

by_topo = defaultdict(list); by_n = defaultdict(list); by_mt = defaultdict(list)
worst = []
for r in rows:
    m = P.MODELS[r["model"]]
    w = P.Workload(int(r["in_len"]), int(r["out_len"]), int(r["n_req"]))
    try: cfg = P.parse_calib_config(r)
    except Exception: continue
    pr = P.predict(m, hw, w, cfg, overlap=True)
    if not pr["feasible"]: continue
    meas = float(r["tps"]); pred = pr["tps"]
    e = abs(pred - meas) / meas
    t = topo(r["label"])
    by_topo[t].append(e); by_n[int(r["n_req"])].append(e)
    by_mt[(r["model"], t)].append(e)
    worst.append((e, r["model"], r["label"], r["n_req"], r["workload"], meas, pred))

print("=== MAPE by topology family ===")
for t, es in sorted(by_topo.items()):
    print(f"  {t:8s} n={len(es):3d}  MAPE={100*np.mean(es):6.1f}%  median={100*np.median(es):6.1f}%")
print("\n=== MAPE by n_req ===")
for n, es in sorted(by_n.items()):
    print(f"  n={n:>3d}  cnt={len(es):3d}  MAPE={100*np.mean(es):6.1f}%  median={100*np.median(es):6.1f}%")
print("\n=== MAPE by model x topology ===")
for (mk, t), es in sorted(by_mt.items()):
    print(f"  {mk:11s} {t:8s} n={len(es):3d}  MAPE={100*np.mean(es):6.1f}%")
print("\n=== worst 15 cells ===")
for e, mk, lab, n, wl, meas, pred in sorted(worst, reverse=True)[:15]:
    print(f"  {e*100:6.0f}%  {mk:11s} {lab:30s} n={n:>3s} {wl:13s} meas={meas:7.0f} pred={pred:7.0f}")
