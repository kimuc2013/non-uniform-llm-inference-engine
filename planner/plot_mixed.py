"""Mixed-traffic result figure: per model, baseline (uniform TP=world) vs the
planner's actual pick, on a stream of MIXED request shapes. Reads /tmp/mixed
analysis dumped by the analysis step (or recomputes). One grouped bar set per
model; concurrency on x; uplift % over the bars; the planner's picked non-uniform
config annotated."""
import glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf_planner as P

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures"; OUT.mkdir(exist_ok=True)
MEAN = {"8b": (1080, 483), "opt30b": (597, 540), "70b": (597, 540)}
TITLE = {"8b": "Llama-8B", "opt30b": "OPT-30B", "70b": "Llama-70B"}


def recs(mk):
    R = {}
    for d in sorted(glob.glob(str(REPO / f"results/hetero_4x4_{mk}_*"))):
        for rj in glob.glob(d + "/*mixed*/record.json"):
            for e in ((lambda x: x if isinstance(x, list) else [x])(json.load(open(rj)))):
                if e.get("workload") == "mixed" and e.get("success") and e.get("tps", 0) > 0:
                    R.setdefault(e["n_req"], {})[e["label"]] = (e["tps"], e["layer_split"], e["tp"], e["pp"])
    return R


def main():
    hw = P.load_hardware()
    models = [mk for mk in ["8b", "opt30b", "70b"] if recs(mk)]
    fig, axes = plt.subplots(1, len(models), figsize=(len(models) * 4.6, 4.6), squeeze=False)
    axes = axes.flatten()
    for i, mk in enumerate(models):
        ax = axes[i]; R = recs(mk); il, ol = MEAN[mk]
        ns = sorted(R)
        base, pick, picklab = [], [], []
        for n in ns:
            r = R[n]; b = r.get("TP8PP1_uniform")
            pk = P.plan(P.MODELS[mk], hw, P.Workload(il, ol, n), top_k=1)[0][1]
            cand = [(lab, v) for lab, v in r.items() if v[2] == pk.tp and v[3] == pk.pp]
            mlab, mv = min(cand, key=lambda x: sum(abs(a - c) for a, c in zip(x[1][1], pk.layer_split)))
            base.append(b[0] if b else 0); pick.append(mv[0])
            picklab.append(f"TP{pk.tp}x PP{pk.pp} L={'-'.join(map(str, pk.layer_split))}")
        x = np.arange(len(ns)); w = 0.38
        ax.bar(x - w / 2, base, w, color="#bdc1c6", edgecolor="#80868b")
        ax.bar(x + w / 2, pick, w, color="#1a73e8", edgecolor="#174ea6")
        top = max(max(base), max(pick))
        for j in range(len(ns)):
            ax.text(j, max(base[j], pick[j]) + top * 0.02, f"+{(pick[j]/base[j]-1)*100:.0f}%",
                    ha="center", fontsize=12, fontweight="bold", color="#137333")
        ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in ns], fontsize=11)
        ax.set_ylim(0, top * 1.22)
        ax.set_title(f"{TITLE[mk]} — mixed traffic", fontweight="bold", fontsize=12)
        ax.set_ylabel("throughput (tok/s)"); ax.grid(axis="y", alpha=0.3)
        # planner pick banner (consistent across n here)
        ax.text(0.5, 0.99, "planner pick:  " + picklab[-1], transform=ax.transAxes, ha="center",
                va="top", fontsize=9, color="#174ea6", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#e8f0fe", ec="#1a73e8"))
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc="#bdc1c6", ec="#80868b", label="baseline — uniform TP8"),
                        Patch(fc="#1a73e8", ec="#174ea6", label="planner pick (non-uniform)")],
               loc="lower center", ncol=2, fontsize=11, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Mixed-traffic serving: varied (input,output) shapes per request in one stream — "
                 "planner pick vs uniform baseline (4+4)", fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    p = OUT / "fig_mixed_traffic.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")


if __name__ == "__main__":
    main()
