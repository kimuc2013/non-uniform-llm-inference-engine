"""Per-model parallelization comparison: for each model (and layout), one line per
configuration (TP8 / TP4PP2 / TP2PP4 / TP1PP8, uniform + non-uniform skew/FFN-bias),
throughput vs concurrency. Shows the TP<->PP crossover and the non-uniform gain
directly. Baseline (uniform TP=world) is thick grey; the planner's top-1 pick is
thick blue + a star at the best n. Usage: python planner/plot_per_model_configs.py
"""
import glob, json, math, sys
from collections import defaultdict
from pathlib import Path
import dataclasses
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf_planner as P

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures" / "per_model_configs"; OUT.mkdir(parents=True, exist_ok=True)
MODELS = [("8b", "Llama-8B"), ("70b", "Llama-70B"), ("opt30b", "OPT-30B"),
          ("mistral123b", "Mistral-123B"), ("mixtral8x7b", "Mixtral-8x7B")]


def label(e):
    """Readable config name incl. the non-uniform split (Blackwell-side : Ada-side)."""
    tp, pp = e["tp"], e["pp"]
    head = f"TP{tp}" + (f"x PP{pp}" if pp > 1 else "")
    ls, ffn = e.get("layer_split", []), e.get("ffn_splits", [])
    if pp > 1 and max(ls) - min(ls) > 1:
        return f"{head}  L={'-'.join(map(str, ls))}"
    if pp == 1 and ffn and max(ffn) != min(ffn):
        return f"{head}  FFN {ffn[0]}:{ffn[-1]}"
    return f"{head} (uniform)"


def relayout(hg, wg):
    b = P.load_hardware(); return dataclasses.replace(b, nodes=((b.nodes[0][0], hg), (b.nodes[-1][0], wg)))


def cells(model, hg, wg, workload="balanced"):
    out = []
    for rj in glob.glob(str(REPO / f"results/hetero_{hg}x{wg}_{model}_*/*/record.json")):
        d = json.load(open(rj))
        for e in (d if isinstance(d, list) else [d]):
            if e.get("success") and e.get("tps", 0) > 0 and e.get("workload") == workload:
                out.append(e)
    return out


def _tp_of(lab):
    return int(lab.split("TP")[1].split("x")[0].split(" ")[0])


def plot_layout(hg, wg, workload="balanced"):
    world = hg + wg
    data = [(mk, t) for mk, t in MODELS if cells(mk, hg, wg, workload)]
    if not data:
        return
    n = len(data); ncols = min(3, n); nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6.2, nrows * 4.8), squeeze=False)
    axes = axes.flatten()
    for i, (mk, title) in enumerate(data):
        ax = axes[i]
        es = cells(mk, hg, wg, workload)
        ns = sorted({e["n_req"] for e in es})
        bycfg = defaultdict(dict)               # label -> {n_req: tps}
        for e in es:
            bycfg[label(e)][e["n_req"]] = e["tps"]
        # order x: by TP degree desc, uniform before non-uniform within a topology
        cfgs = sorted(bycfg, key=lambda l: (-_tp_of(l), ("L=" in l or "FFN" in l), l))
        # planner pick (at the highest concurrency)
        hw = relayout(hg, wg)
        r = P.plan(P.MODELS[mk], hw, P.Workload(es[0]["in_len"], es[0]["out_len"], max(ns)), top_k=1)
        pick = (f"TP{r[0][1].tp}" + (f"x PP{r[0][1].pp}" if r[0][1].pp > 1 else "")) if r else None
        # grouped bars: one group per config, one bar per concurrency
        x = np.arange(len(cfgs)); nb = len(ns); w = 0.82 / nb
        cols = plt.cm.viridis(np.linspace(0.12, 0.82, nb))
        for k, nr in enumerate(ns):
            ys = [bycfg[c].get(nr, 0) for c in cfgs]
            ax.bar(x + (k - (nb - 1) / 2) * w, ys, w, color=cols[k], label=f"n={nr}")
        ax.set_xticks(x)
        ticklabs = ax.set_xticklabels(cfgs, rotation=35, ha="right", fontsize=8)
        for t, c in zip(ticklabs, cfgs):     # highlight baseline (grey) + planner pick (blue)
            if c == f"TP{world} (uniform)":
                t.set_color("#5f6368"); t.set_fontweight("bold")
            elif pick and c.startswith(pick) and ("L=" in c or "FFN" in c):
                t.set_color("#174ea6"); t.set_fontweight("bold")
        ax.set_title(f"{title}  ({hg}+{wg}, {workload})", fontweight="bold", fontsize=12)
        ax.set_ylabel("throughput (tok/s)"); ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, title="concurrency", loc="upper right", framealpha=0.92)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Per-model parallelization — {workload}, {hg}+{wg}   "
                 f"(grouped bars = concurrency; x-label grey-bold = uniform-TP{world} baseline, "
                 f"blue-bold = planner pick)", fontsize=12)
    plt.tight_layout()
    p = OUT / f"per_model_configs_{hg}x{wg}_{workload}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}  ({n} models)")


if __name__ == "__main__":
    for hg, wg in [(4, 4), (2, 2), (1, 1)]:
        for wl in ["balanced", "decode_heavy", "prefill_heavy"]:
            plot_layout(hg, wg, wl)
