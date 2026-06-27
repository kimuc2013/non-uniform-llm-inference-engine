"""Zero-refit validation of the modular MoE planner against measured Mixtral-8x7B.

Compares the (dense-frozen) planner's predictions to the measured sweep for every
(config, n_req) cell, reports per-cell error, the champion match, and checks the
PRE-REGISTERED predictions (planner/mixtral8x7b_prereg.json — frozen before any
measurement). Usage: python planner/validate_mixtral.py
"""
import csv, glob, json, sys, dataclasses
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf_planner as P

REPO = Path(__file__).resolve().parents[1]
MODEL = "mixtral8x7b"


def relayout(hg, wg):
    base = P.load_hardware()
    return dataclasses.replace(base, nodes=((base.nodes[0][0], hg), (base.nodes[-1][0], wg)))


def cells(hg, wg):
    out = []
    for d in glob.glob(str(REPO / "results" / f"hetero_{hg}x{wg}_{MODEL}_*")):
        for rj in glob.glob(d + "/*/record.json") + glob.glob(d + "/record.json"):
            try:
                recs = json.load(open(rj))
            except Exception:
                continue
            for e in (recs if isinstance(recs, list) else [recs]):
                if e.get("success") and e.get("tps", 0) > 0 and e.get("workload") == "balanced":
                    out.append(e)
    return out


def main():
    m = P.MODELS[MODEL]
    prereg = json.load(open(REPO / "planner" / "mixtral8x7b_prereg.json"))
    any_data = False
    for hg, wg in [(4, 4), (2, 2), (1, 1)]:
        cs = cells(hg, wg)
        if not cs:
            continue
        any_data = True
        hw = relayout(hg, wg)
        print(f"\n===== Mixtral-8x7B  {hg}+{wg}  (balanced) =====")
        ns = sorted({e["n_req"] for e in cs})
        for n in ns:
            sub = [e for e in cs if e["n_req"] == n]
            best = {}
            for e in sub:
                k = (e["tp"], e["pp"])
                if k not in best or e["tps"] > best[k][0]:
                    best[k] = (e["tps"], e)
            w = P.Workload(sub[0]["in_len"], sub[0]["out_len"], n)
            ranked = P.plan(m, hw, w, top_k=1)
            pick = ranked[0][1] if ranked else None
            measbest = max(best.values(), key=lambda x: x[0])[1]
            pk = f"TP{pick.tp}xPP{pick.pp}" if pick else "-"
            mb = f"TP{measbest['tp']}xPP{measbest['pp']}"
            pr = prereg["layouts"].get(f"{hg}+{wg}", {}).get(str(n), {}).get("champion", "?")
            flag = "OK" if pk == mb else "MISS"
            print(f"  n={n:3d}  planner_pick={pk:9s} meas_best={mb:9s} [{flag}]  prereg={pr}")
            for k in sorted(best, key=lambda k: -best[k][0]):
                mt, e = best[k]
                c = P.Config(e["tp"], e["pp"], e["layer_split"], e["ffn_splits"],
                             e["head_splits"], e["kv_splits"])
                pt = P.predict(m, hw, w, c)["tps"]
                topo = f"TP{e['tp']}xPP{e['pp']}"
                print(f"      {topo:9s} {e.get('label',''):20s} meas={mt:7.0f} pred={pt:7.0f} "
                      f"err={(pt/mt-1)*100:+5.0f}%")
    if not any_data:
        print("No measured Mixtral data yet — run the sweep first:")
        print("  python planner/hetero_sweep.py --model mixtral8x7b --workloads balanced "
              "--configs uniform --n-req-list 16,32,64,96")


if __name__ == "__main__":
    main()
