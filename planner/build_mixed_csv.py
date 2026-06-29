"""Build the mixed-traffic results CSV (layout-aware) directly from the measured
records + the planner, so baseline / planner-pick / oracle / uplift / regret are
always consistent with the data. Emits one row per (layout, model, n_req).
Usage: python planner/build_mixed_csv.py [out.csv]
"""
import csv, dataclasses, glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf_planner as P

REPO = Path(__file__).resolve().parents[1]
MEAN = {"8b": (1080, 483), "opt30b": (597, 540), "70b": (597, 540)}
LAYOUTS = [(4, 4), (2, 2)]


def relayout(hw, hg, wg):
    bw, ada = hw.nodes[0][0], hw.nodes[-1][0]
    return dataclasses.replace(hw, nodes=((bw, hg), (ada, wg)))


def recs(mk, hg, wg):
    R = {}
    for d in sorted(glob.glob(str(REPO / f"results/hetero_{hg}x{wg}_{mk}_*"))):
        for rj in glob.glob(d + "/*mixed*/record.json"):
            for e in ((lambda x: x if isinstance(x, list) else [x])(json.load(open(rj)))):
                if e.get("workload") == "mixed" and e.get("success") and e.get("tps", 0) > 0:
                    R.setdefault(e["n_req"], {})[e["label"]] = (e["tps"], tuple(e["layer_split"]), e["tp"], e["pp"])
    return R


def main(out=REPO / "RESULTS_PACKAGE" / "data" / "mixed_traffic_results.csv"):
    rows = []
    for hg, wg in LAYOUTS:
        world = hg + wg
        hw = relayout(P.load_hardware(), hg, wg)
        for mk in ["8b", "opt30b", "70b"]:
            R = recs(mk, hg, wg)
            if not R:
                continue
            il, ol = MEAN[mk]
            for n in sorted(R):
                r = R[n]
                b = r.get(f"TP{world}PP1_uniform")
                bt = b[0] if b else 0
                pk = P.plan(P.MODELS[mk], hw, P.Workload(il, ol, n), top_k=1)[0][1]
                cand = [(lab, v) for lab, v in r.items() if v[2] == pk.tp and v[3] == pk.pp]
                if cand:
                    mlab, mv = min(cand, key=lambda x: sum(abs(a - c) for a, c in zip(x[1][1], pk.layer_split)))
                    pt = mv[0]
                    plab = f"TP{pk.tp}PP{pk.pp}_L{'-'.join(map(str, pk.layer_split))}"
                else:
                    pt = bt; plab = f"TP{pk.tp}PP{pk.pp}"
                olab, ov = max(r.items(), key=lambda kv: kv[1][0])
                rows.append({
                    "layout": f"{hg}+{wg}", "model": mk, "n_req": n,
                    "baseline_uniformTP%d_tps" % world: round(bt, 1),
                    "baseline_tps": round(bt, 1),
                    "planner_pick": plab, "planner_pick_tps": round(pt, 1),
                    "oracle_config": olab, "oracle_tps": round(ov[0], 1),
                    "uplift_vs_baseline_pct": round((pt / bt - 1) * 100, 1) if bt else 0,
                    "regret_vs_oracle_pct": round((1 - pt / ov[0]) * 100, 1) if ov[0] else 0,
                })
    cols = ["layout", "model", "n_req", "baseline_tps", "planner_pick", "planner_pick_tps",
            "oracle_config", "oracle_tps", "uplift_vs_baseline_pct", "regret_vs_oracle_pct"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"wrote {out}  ({len(rows)} rows)")
    for row in rows:
        print(f"  {row['layout']} {row['model']:8s} n={row['n_req']:3d}  "
              f"base={row['baseline_tps']:7.1f}  pick={row['planner_pick_tps']:7.1f} "
              f"({row['uplift_vs_baseline_pct']:+.0f}%)  oracle={row['oracle_tps']:7.1f} "
              f"(regret {row['regret_vs_oracle_pct']:.0f}%)")


if __name__ == "__main__":
    main(*( [Path(sys.argv[1])] if len(sys.argv) > 1 else [] ))
