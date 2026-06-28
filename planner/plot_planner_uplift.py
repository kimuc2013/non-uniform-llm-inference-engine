"""Planner uplift over the naive baseline, per batch (concurrency), for the 4
models whose serving realizes the predictions (qwen3-32B excluded — profiled fork
PP-overlap gap, see planner_describe.md §8). Two bars per batch — baseline (uniform
TP=world) and the planner's RAW top-1 pick (no safety guard) — uplift % labeled. No 'measured best' bar.

Only n_req >= 32 is shown: small batches do not saturate the cluster and are not
a representative serving regime. Balanced workload (covers prefill+decode).
Output: figures/planner_uplift/. Usage: python planner/plot_planner_uplift.py
"""
from __future__ import annotations
import dataclasses, glob, json, sys
from pathlib import Path
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import perf_planner as P
REPO = HERE.parent
OUT = REPO / "figures" / "planner_uplift"; OUT.mkdir(parents=True, exist_ok=True)

# qwen32b is EXCLUDED: its TP4PP2 PP-overlap does not engage in the current fork
# (profiled serving-stack gap, not a planner error — see planner_describe.md §8 /
# planner/qwen_pp_profile.py). The planner correctly predicts qwen's PP should scale;
# the serving just doesn't realize it, so its cells would distort the planner-quality
# figure. Evaluated on the 4 models whose serving realizes the predictions.
MODELS = [("8b", "Llama-3.1-8B"), ("70b", "Llama-3.3-70B"), ("opt30b", "OPT-30B"),
          ("mistral123b", "Mistral-Large-123B")]
WORKLOAD = "balanced"


def relayout(hw, hg, wg):
    bw, ada = hw.nodes[0][0], hw.nodes[-1][0]
    return dataclasses.replace(hw, nodes=((bw, hg), (ada, wg)))


def balanced_cells(model, hg, wg):
    out = []
    for d in glob.glob(str(REPO / f"results/hetero_{hg}x{wg}_{model}_*")):
        if "_full_" in d: continue
        for rj in glob.glob(d + "/*/record.json"):
            try: recs = json.load(open(rj))
            except Exception: continue
            for e in (recs if isinstance(recs, list) else [recs]):
                if e.get("success") and e.get("tps", 0) > 0 and e.get("workload") == WORKLOAD:
                    out.append(e)
    return out


def pick_label(scfg):
    """Label showing HOW the non-uniform split was done (Blackwell-side first, Ada last):
      'TP4xPP2  L=24-8'     PP layer skew (fast node 24 layers, slow node 8)
      'TP8  FFN 2688:896'   TP FFN-column bias (Blackwell 2688, Ada 896)
      'TP4xPP2' / 'TP8'     uniform."""
    head = f"TP{scfg.tp}" + (f"xPP{scfg.pp}" if scfg.pp > 1 else "")
    if scfg.pp > 1:
        if max(scfg.layer_split) - min(scfg.layer_split) <= 1:
            return head
        return f"{head}  L={'-'.join(map(str, scfg.layer_split))}"
    if max(scfg.ffn_splits) == min(scfg.ffn_splits):
        return head
    return f"{head}  FFN {scfg.ffn_splits[0]}:{scfg.ffn_splits[-1]}"


def collect_model(mk, hg, wg, world, hw):
    """Return (ns, base, plan, uplift, picks) for a model at this layout, or None if no data."""
    cells = balanced_cells(mk, hg, wg)
    if not cells:
        return None
    byn = defaultdict(list)
    for e in cells: byn[e["n_req"]].append(e)
    ns = [n for n in sorted(byn) if n >= 32]   # small batches are not a real cluster regime
    if not ns:
        return None
    base = []; plan = []; uplift = []; picks = []
    for n in ns:
        es = list({e["label"]: e for e in byn[n]}.values())
        w = P.Workload(es[0]["in_len"], es[0]["out_len"], n)
        b = [e for e in es if e["label"] == f"TP{world}PP1_uniform"]
        tp1 = [e for e in es if e["pp"] == 1]
        bt = b[0]["tps"] if b else (max(e["tps"] for e in tp1) if tp1 else max(e["tps"] for e in es))
        # RAW planner top-1 (no safety guard), mapped to nearest measured config
        ranked = P.plan(P.MODELS[mk], hw, w, top_k=1)
        if not ranked:        # no feasible config at this layout (e.g. 123B won't fit on 2 GPUs)
            base.append(bt); plan.append(bt); uplift.append(0.0); picks.append("—"); continue
        scfg = ranked[0][1]
        same = [e for e in es if e["tp"] == scfg.tp and e["pp"] == scfg.pp]
        pt = (min(same, key=lambda e: sum(abs(a - x) for a, x in zip(e["layer_split"], scfg.layer_split))
                  + abs(e["ffn_splits"][0] - scfg.ffn_splits[0]) / 1000)["tps"] if same else bt)
        base.append(bt); plan.append(pt); uplift.append((pt / bt - 1) * 100 if bt else 0)
        picks.append(pick_label(scfg))
    return ns, base, plan, uplift, picks


def main(hg=4, wg=4):
    import math
    hw = relayout(P.load_hardware(), hg, wg)
    world = hg + wg
    # only models with data at THIS layout (123B needs >=6 GPU, so absent at 1+1/2+2)
    data = [(mk, title, *d) for mk, title in MODELS
            if (d := collect_model(mk, hg, wg, world, hw)) is not None]
    if not data:
        print(f"  [{hg}+{wg}] no data"); return
    # grid sized exactly to the model count (no empty cells): 2->1x2, 3->1x3, 4->2x2
    n = len(data)
    ncols = n if n <= 3 else math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.6, nrows * 4.6), squeeze=False)
    axes = axes.flatten()
    for i, (mk, title, ns, base, plan, uplift, picks) in enumerate(data):
        ax = axes[i]
        x = np.arange(len(ns)); w = 0.40
        ax.bar(x - w / 2, base, w, label=f"baseline — uniform TP{world}", color="#bdc1c6",
               edgecolor="#80868b")
        ax.bar(x + w / 2, plan, w, label="planner pick", color="#1a73e8", edgecolor="#174ea6")
        top = max(max(base), max(plan))
        for j, u in enumerate(uplift):
            # uplift % above the taller bar
            ax.text(j, max(base[j], plan[j]) + top * 0.02, f"{u:+.0f}%", ha="center",
                    fontsize=11, fontweight="bold", color="#137333" if u >= 0 else "#c5221f")
        ax.set_xticks(x); ax.set_xticklabels([f"n={n_}" for n_ in ns], fontsize=11)
        ax.set_ylim(0, top * 1.26)
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.set_ylabel("throughput (tok/s)"); ax.grid(axis="y", alpha=0.3)
        # grey vs blue identified by the banner ("planner pick") + suptitle ("vs uniform
        # baseline"); no legend / no in-bar text (the split is in the banner).
        # prominent banner: which non-uniform config the planner recommends
        uniq = list(dict.fromkeys(picks))
        if len(uniq) == 1:
            banner = f"planner pick:  {uniq[0]}"
        else:
            banner = "planner pick:  " + " | ".join(f"n{n}: {p}" for n, p in zip(ns, picks))
        ax.text(0.5, 0.99, banner, transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color="#174ea6", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="#e8f0fe", ec="#1a73e8", lw=1.2))
    for j in range(n, len(axes)):   # hide any trailing unused cell
        axes[j].set_visible(False)
    fig.suptitle(f"Planner (raw top-1 pick) vs naive baseline (uniform TP{world}) — "
                 f"balanced, {hg}+{wg}, n>=32  (qwen3-32B excluded — fork PP-overlap gap, §8)",
                 fontsize=13)
    plt.tight_layout()
    p = OUT / f"planner_vs_baseline_uplift_{hg}x{wg}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}  ({nrows}x{ncols} grid, {n} models)")
    for mk, title, ns, base, plan, uplift, picks in data:
        print(f"  [{hg}+{wg}] {mk:12s} " + " ".join(f"n{n}:{u:+.0f}%[{p}]" for n, u, p in zip(ns, uplift, picks)))


if __name__ == "__main__":
    for hg, wg in [(4, 4), (2, 2), (1, 1)]:
        main(hg, wg)
