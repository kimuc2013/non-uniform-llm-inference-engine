"""Planner uplift over the naive baseline, per batch (concurrency), for all 5
models. Two bars per batch — baseline (uniform TP=world) and the planner's RAW
top-1 pick (no safety guard) — with the uplift % labeled. No 'measured best' bar.

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

MODELS = [("8b", "Llama-3.1-8B"), ("70b", "Llama-3.3-70B"), ("opt30b", "OPT-30B"),
          ("qwen32b", "Qwen3-32B"), ("mistral123b", "Mistral-Large-123B")]
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


def main(hg=4, wg=4):
    hw = relayout(P.load_hardware(), hg, wg)
    world = hg + wg
    fig, axes = plt.subplots(2, 3, figsize=(20, 11)); axes = axes.flatten()
    summary = {}
    for ax, (mk, title) in zip(axes, MODELS):
        m = P.MODELS[mk]; cells = balanced_cells(mk, hg, wg)
        if not cells:
            ax.set_title(f"{title} (no data)"); ax.set_visible(False); continue
        byn = defaultdict(list)
        for e in cells: byn[e["n_req"]].append(e)
        ns = [n for n in sorted(byn) if n >= 32]   # small batches are not a real cluster regime
        base = []; plan = []; uplift = []
        for n in ns:
            es = list({e["label"]: e for e in byn[n]}.values())
            w = P.Workload(es[0]["in_len"], es[0]["out_len"], n)
            b = [e for e in es if e["label"] == f"TP{world}PP1_uniform"]
            tp1 = [e for e in es if e["pp"] == 1]
            bt = b[0]["tps"] if b else (max(e["tps"] for e in tp1) if tp1 else max(e["tps"] for e in es))
            # RAW planner top-1 (no safety guard), mapped to nearest measured config
            ranked = P.plan(m, hw, w, top_k=1)
            if not ranked:        # no feasible config at this layout (e.g. 123B won't fit on 2 GPUs)
                base.append(bt); plan.append(bt); uplift.append(0.0); continue
            scfg = ranked[0][1]
            same = [e for e in es if e["tp"] == scfg.tp and e["pp"] == scfg.pp]
            pt = (min(same, key=lambda e: sum(abs(a - x) for a, x in zip(e["layer_split"], scfg.layer_split))
                      + abs(e["ffn_splits"][0] - scfg.ffn_splits[0]) / 1000)["tps"] if same else bt)
            base.append(bt); plan.append(pt); uplift.append((pt / bt - 1) * 100 if bt else 0)
        x = np.arange(len(ns)); w = 0.38
        ax.bar(x - w / 2, base, w, label="baseline (uniform TP)", color="#9aa0a6")
        ax.bar(x + w / 2, plan, w, label="planner pick", color="#1a73e8")
        for i, u in enumerate(uplift):
            ax.text(i, max(base[i], plan[i]) * 1.02, f"{u:+.0f}%", ha="center",
                    fontsize=10, fontweight="bold", color="#137333" if u >= 0 else "#c5221f")
        ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in ns])
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.set_ylabel("throughput (tok/s)"); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9, loc="upper left")
        summary[mk] = (ns, uplift)
    # 6th panel: uplift % vs batch, all models (lines)
    axu = axes[5]
    for mk, title in MODELS:
        if mk not in summary: continue
        ns, up = summary[mk]
        axu.plot(ns, up, "-o", label=title, linewidth=2)
    axu.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    axu.set_xlabel("concurrency (n_req)"); axu.set_ylabel("planner uplift over baseline (%)")
    axu.set_title("Uplift vs concurrency (all models)", fontweight="bold", fontsize=13)
    axu.grid(alpha=0.3); axu.legend(fontsize=8)
    fig.suptitle(f"Planner (raw top-1 pick) vs naive baseline (uniform TP{world}) — "
                 f"balanced workload, {hg}+{wg}, n>=32, by batch", fontsize=15, y=1.0)
    plt.tight_layout()
    p = OUT / f"planner_vs_baseline_uplift_{hg}x{wg}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}")
    for mk, (ns, up) in summary.items():
        print(f"  [{hg}+{wg}] {mk:12s} uplift by n: " + " ".join(f"n{n}:{u:+.0f}%" for n, u in zip(ns, up)))


if __name__ == "__main__":
    for hg, wg in [(4, 4), (2, 2), (1, 1)]:
        main(hg, wg)
