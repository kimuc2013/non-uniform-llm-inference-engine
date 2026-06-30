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
# mean (in,out) of the mixed-traffic shape mix (for the planner-pick overlay only)
MIXED_MEAN = {"8b": (1080, 483), "opt30b": (597, 540), "70b": (597, 540)}


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


def method(e):
    """Parallelization METHOD (topology + uniform/non-uniform type), collapsing skew
    degrees so methods are comparable: 'TP8', 'TP8 FFN-bias', 'TP4xPP2',
    'TP4xPP2 skew', 'TP2xPP4', 'TP2xPP4 skew', 'TP1xPP8'."""
    tp, pp = e["tp"], e["pp"]
    head = f"TP{tp}" + (f"xPP{pp}" if pp > 1 else "")
    ls, ffn = e.get("layer_split", []), e.get("ffn_splits", [])
    if pp > 1 and ls and max(ls) - min(ls) > 1:
        return head + " skew"
    if pp == 1 and ffn and max(ffn) != min(ffn):
        return head + " FFN-bias"
    return head


# fixed method order (high TP -> low). Colors are PAIRED by topology: the uniform
# method gets a saturated hue, its non-uniform variant the same hue lighter, so
# related methods read together. The uniform-TP=world baseline is drawn grey+hatch.
METHOD_ORDER = ["TP8", "TP8 FFN-bias", "TP4xPP2", "TP4xPP2 skew", "TP2xPP4", "TP2xPP4 skew",
                "TP1xPP8", "TP4", "TP4 FFN-bias", "TP2xPP2", "TP2xPP2 skew", "TP1xPP4",
                "TP2", "TP2 FFN-bias", "TP1xPP2", "TP1xPP2 skew"]
METHOD_COLOR = {
    # red = pure high-TP; blue = TPxPP2; green = deeper PP; purple = full PP
    "TP8": "#ea4335",     "TP8 FFN-bias": "#f7b4ad",
    "TP4xPP2": "#1a73e8", "TP4xPP2 skew": "#9fc3fa",
    "TP2xPP4": "#137333", "TP2xPP4 skew": "#84caa0",
    "TP1xPP8": "#a142f4",
    "TP4": "#ea4335",     "TP4 FFN-bias": "#f7b4ad",
    "TP2xPP2": "#1a73e8", "TP2xPP2 skew": "#9fc3fa",
    "TP1xPP4": "#137333",
    "TP2": "#ea4335",     "TP2 FFN-bias": "#f7b4ad",
    "TP1xPP2": "#1a73e8", "TP1xPP2 skew": "#9fc3fa",
}
BASE_GREY = "#9aa0a6"


def plot_layout(hg, wg, workload="balanced"):
    world = hg + wg
    data = [(mk, t) for mk, t in MODELS if cells(mk, hg, wg, workload)]
    if not data:
        return
    n = len(data); ncols = min(3, n); nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6.0, nrows * 4.7), squeeze=False)
    axes = axes.flatten()
    seen_methods = []
    for i, (mk, title) in enumerate(data):
        ax = axes[i]
        es = cells(mk, hg, wg, workload)
        ns = sorted({e["n_req"] for e in es})
        # best tps per (method, n_req) — collapses skew+8/+12 into the best 'skew'
        bym = defaultdict(dict)
        for e in es:
            m = method(e); bym[m][e["n_req"]] = max(bym[m].get(e["n_req"], 0), e["tps"])
        methods = [m for m in METHOD_ORDER if m in bym]
        for m in methods:
            if m not in seen_methods:
                seen_methods.append(m)
        # planner-pick topology (highest concurrency). Mixed traffic has no fixed
        # (in,out) per cell (es[0]["in_len"]=-1) -> use the mean of the shape mix.
        hw = relayout(hg, wg)
        if workload == "mixed":
            il, ol = MIXED_MEAN.get(mk, (512, 512))
        else:
            il, ol = es[0]["in_len"], es[0]["out_len"]
        r = P.plan(P.MODELS[mk], hw, P.Workload(il, ol, max(ns)), top_k=1)
        pickm = None
        if r:
            c = r[0][1]; pickm = method({"tp": c.tp, "pp": c.pp,
                "layer_split": c.layer_split, "ffn_splits": c.ffn_splits})
        # GROUP BY BATCH: x = concurrency, one bar per method  -> compare methods per batch
        x = np.arange(len(ns)); nm = len(methods); w = 0.86 / nm
        for k, m in enumerate(methods):
            ys = [bym[m].get(nr, 0) for nr in ns]
            base = (m == f"TP{world}")            # uniform TP=world is the baseline
            ax.bar(x + (k - (nm - 1) / 2) * w, ys, w,
                   color=(BASE_GREY if base else METHOD_COLOR.get(m, "#888")),
                   edgecolor=("#202124" if m == pickm else ("#5f6368" if base else "none")),
                   linewidth=(2.0 if m == pickm else (0.8 if base else 0)),
                   hatch=("///" if base else None), label=m)
        ax.set_xticks(x); ax.set_xticklabels([f"n={nr}" for nr in ns], fontsize=11)
        ax.set_title(f"{title}  ({hg}+{wg}, {workload})", fontweight="bold", fontsize=12)
        ax.set_xlabel("concurrent requests (batch)"); ax.set_ylabel("throughput (tok/s)")
        ax.grid(axis="y", alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    # single shared legend at the bottom (methods are consistent across panels)
    from matplotlib.patches import Patch
    handles = []
    for m in [mm for mm in METHOD_ORDER if mm in seen_methods]:
        if m == f"TP{world}":
            handles.append(Patch(fc=BASE_GREY, ec="#5f6368", hatch="///", label=f"{m}  (baseline)"))
        else:
            handles.append(Patch(fc=METHOD_COLOR.get(m, "#888"), label=m))
    handles.append(Patch(fc="white", ec="#202124", lw=2.0, label="black edge = planner pick"))
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Per-model parallelization comparison — {workload}, {hg}+{wg}   "
                 f"(grouped by batch; each bar = a parallelization method)", fontsize=13)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    p = OUT / f"per_model_configs_{hg}x{wg}_{workload}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved {p}  ({n} models)")


if __name__ == "__main__":
    for hg, wg in [(4, 4), (2, 2), (1, 1)]:
        for wl in ["balanced", "decode_heavy", "prefill_heavy", "mixed"]:
            plot_layout(hg, wg, wl)
