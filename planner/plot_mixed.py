"""Mixed-traffic result figure: per model, baseline (uniform TP=world) vs the
planner's actual pick, on a stream of MIXED request shapes (varied (in,out) per
request in ONE concurrent stream). One grouped bar set per model; concurrency on
x; uplift % over the bars; the planner's picked non-uniform config annotated.

Layout-parametric: emits fig_mixed_traffic_{hg}x{wg}.png for every layout that has
mixed records. The 4+4 figure is also copied to fig_mixed_traffic.png (package name).
Usage: python planner/plot_mixed.py
"""
import glob, json, shutil, sys
import dataclasses
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf_planner as P

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures"; OUT.mkdir(exist_ok=True)
# mean (in,out) of the mixed shape mix actually sent at each layout/model. 8b uses
# the full 4096-cap mix; opt30b/70b use the <=1900 cap (shorter mean).
MEAN = {"8b": (1080, 483), "opt30b": (597, 540), "70b": (597, 540)}
TITLE = {"8b": "Llama-8B", "opt30b": "OPT-30B", "70b": "Llama-70B"}


def relayout(hw, hg, wg):
    bw, ada = hw.nodes[0][0], hw.nodes[-1][0]
    return dataclasses.replace(hw, nodes=((bw, hg), (ada, wg)))


def recs(mk, hg, wg):
    R = {}
    for d in sorted(glob.glob(str(REPO / f"results/hetero_{hg}x{wg}_{mk}_*"))):
        for rj in glob.glob(d + "/*mixed*/record.json"):
            for e in ((lambda x: x if isinstance(x, list) else [x])(json.load(open(rj)))):
                if e.get("workload") == "mixed" and e.get("success") and e.get("tps", 0) > 0:
                    R.setdefault(e["n_req"], {})[e["label"]] = (e["tps"], e["layer_split"], e["tp"], e["pp"])
    return R


def plot_layout(hg, wg):
    world = hg + wg
    hw = relayout(P.load_hardware(), hg, wg)
    models = [mk for mk in ["8b", "opt30b", "70b"] if recs(mk, hg, wg)]
    if not models:
        return None
    fig, axes = plt.subplots(1, len(models), figsize=(len(models) * 4.6, 4.6), squeeze=False)
    axes = axes.flatten()
    for i, mk in enumerate(models):
        ax = axes[i]; R = recs(mk, hg, wg); il, ol = MEAN[mk]
        ns = sorted(R)
        base, pick, oracle, picklab = [], [], [], []
        for n in ns:
            r = R[n]; b = r.get(f"TP{world}PP1_uniform")
            pk = P.plan(P.MODELS[mk], hw, P.Workload(il, ol, n), top_k=1)[0][1]
            cand = [(lab, v) for lab, v in r.items() if v[2] == pk.tp and v[3] == pk.pp]
            if not cand:                      # planner topology not measured this layout
                base.append(b[0] if b else 0); pick.append(b[0] if b else 0)
                picklab.append(f"TP{pk.tp}x PP{pk.pp}")
            else:
                mlab, mv = min(cand, key=lambda x: sum(abs(a - c) for a, c in zip(x[1][1], pk.layer_split)))
                base.append(b[0] if b else 0); pick.append(mv[0])
                picklab.append(f"TP{pk.tp}x PP{pk.pp} L={'-'.join(map(str, pk.layer_split))}")
            # oracle = best measured config at this n (what the planner *could* pick).
            # Drawn so the planner's regret is shown honestly, not hidden.
            oracle.append(max(v[0] for v in r.values()))
        x = np.arange(len(ns)); w = 0.27
        ax.bar(x - w, base, w, color="#bdc1c6", edgecolor="#80868b")
        ax.bar(x, pick, w, color="#1a73e8", edgecolor="#174ea6")
        ax.bar(x + w, oracle, w, color="#fbbc04", edgecolor="#ea8600", hatch="..")
        top = max(max(base), max(pick), max(oracle), 1)
        for j in range(len(ns)):
            if base[j] > 0:
                ax.text(j - w, pick[j] + top * 0.02, f"{(pick[j]/base[j]-1)*100:+.0f}%",
                        ha="center", fontsize=10.5, fontweight="bold",
                        color="#137333" if pick[j] >= base[j] else "#c5221f")
            if oracle[j] > pick[j] * 1.03:    # planner left a gap below the oracle
                ax.text(j + w, oracle[j] + top * 0.02, f"−{(1-pick[j]/oracle[j])*100:.0f}%",
                        ha="center", fontsize=9, fontweight="bold", color="#ea8600")
        ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in ns], fontsize=11)
        ax.set_ylim(0, top * 1.24)
        ax.set_title(f"{TITLE[mk]} — mixed traffic", fontweight="bold", fontsize=12)
        ax.set_ylabel("throughput (tok/s)"); ax.grid(axis="y", alpha=0.3)
        ax.text(0.5, 0.99, "planner pick:  " + picklab[-1], transform=ax.transAxes, ha="center",
                va="top", fontsize=9, color="#174ea6", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#e8f0fe", ec="#1a73e8"))
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc="#bdc1c6", ec="#80868b", label=f"baseline — uniform TP{world}"),
                        Patch(fc="#1a73e8", ec="#174ea6", label="planner pick"),
                        Patch(fc="#fbbc04", ec="#ea8600", hatch="..", label="best measured (oracle)")],
               loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Mixed-traffic serving: varied (input,output) shapes per request in one stream — "
                 f"planner pick vs uniform baseline ({hg}+{wg})", fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    p = OUT / f"fig_mixed_traffic_{hg}x{wg}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}  ({len(models)} models)")
    return p


def main():
    p44 = None
    for hg, wg in [(4, 4), (2, 2), (1, 1)]:
        p = plot_layout(hg, wg)
        if p and (hg, wg) == (4, 4):
            p44 = p
    if p44:                                   # keep the package's canonical name
        shutil.copy(p44, OUT / "fig_mixed_traffic.png")
        print(f"copied {p44.name} -> fig_mixed_traffic.png")


if __name__ == "__main__":
    main()
