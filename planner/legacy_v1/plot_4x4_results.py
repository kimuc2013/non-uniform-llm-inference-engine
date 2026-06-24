"""4+4 cross-node sweep result plots.

Loads record.json files from the auto_sweep_robust v2 runs and produces:
  fig1_<model>_topology_workload.png  — grouped bar (config × workload)
  fig2_<model>_layer_skew_curve.png   — TP4PP2 skew sweep, 1 line per workload
  fig3_topology_overview.png          — TP8 vs TP4PP2-best vs TP2PP4-best, both models
  fig4_<model>_workload_heatmap.png   — config × workload heatmap

Run:
    python planner/plot_4x4_results.py [--out /tmp/plots]
"""
from __future__ import annotations
import argparse, datetime, glob, json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

# auto_sweep_robust v2 started 2026-06-08 20:58
CUTOFF_70B = datetime.datetime(2026, 6, 8, 20, 58).timestamp()
CUTOFF_8B  = datetime.datetime(2026, 6, 8, 22, 47).timestamp()

CONFIGS_70B = [
    ("TP8PP1_uniform",                       "TP8 PP1"),
    ("TP4PP2_layer_uniform_40-40",           "TP4 PP2 [40,40]"),
    ("TP4PP2_layer_skew+4_44-36",            "TP4 PP2 [44,36] +4"),
    ("TP4PP2_layer_skew+8_48-32",            "TP4 PP2 [48,32] +8"),
    ("TP4PP2_layer_skew+12_52-28",           "TP4 PP2 [52,28] +12"),
    ("TP4PP2_layer_skew+16_56-24",           "TP4 PP2 [56,24] +16"),
    ("TP2PP4_layer_uniform_20-20-20-20",     "TP2 PP4 [20]×4"),
    ("TP2PP4_layer_blackbias_22-22-18-18",   "TP2 PP4 [22,22,18,18]"),
    ("TP2PP4_layer_blackbias_24-24-16-16",   "TP2 PP4 [24,24,16,16]"),
]
CONFIGS_8B = [
    ("TP8PP1_uniform",                       "TP8 PP1"),
    ("TP4PP2_layer_uniform_16-16",           "TP4 PP2 [16,16]"),
    ("TP4PP2_layer_skew+2_18-14",            "TP4 PP2 [18,14] +2"),
    ("TP4PP2_layer_skew+4_20-12",            "TP4 PP2 [20,12] +4"),
    ("TP4PP2_layer_skew+6_22-10",            "TP4 PP2 [22,10] +6"),
    ("TP4PP2_layer_skew+8_24-8",             "TP4 PP2 [24,8] +8"),
    ("TP2PP4_layer_uniform_8-8-8-8",         "TP2 PP4 [8]×4"),
    ("TP2PP4_layer_blackbias_9-9-7-7",       "TP2 PP4 [9,9,7,7]"),
    ("TP2PP4_layer_blackbias_10-10-6-6",     "TP2 PP4 [10,10,6,6]"),
]
WORKLOADS = ["balanced", "decode_heavy", "prefill_heavy"]
WL_LABEL  = {"balanced":"balanced\n(in=512 out=256)",
             "decode_heavy":"decode_heavy\n(in=128 out=512)",
             "prefill_heavy":"prefill_heavy\n(in=1024 out=128)"}
WL_COLOR  = {"balanced":"#4c72b0",
             "decode_heavy":"#dd8452",
             "prefill_heavy":"#55a868"}
STOCK_BASELINE_70B = 1280  # measured stock vLLM TP4PP2 [40,40] balanced 70B
STOCK_BASELINE_8B  = None  # not measured (8B different cluster state at time)


def load_recs(pattern: str, after_ts: float) -> list[dict]:
    recs = []
    for f in glob.glob(pattern):
        try:
            if os.path.getmtime(f) < after_ts: continue
            r = json.load(open(f))
            if r.get("success"): recs.append(r)
        except Exception:
            pass
    return recs


def build_grid(recs, configs):
    """Return {(label, workload): tps}. If multiple records, keep highest TPS."""
    g = {}
    for r in recs:
        k = (r["label"], r["workload"])
        if k not in g or r.get("tps", 0) > g[k]:
            g[k] = r.get("tps", 0)
    # ensure all keys present
    out = {}
    for (label, _) in configs:
        for wl in WORKLOADS:
            out[(label, wl)] = g.get((label, wl), 0)
    return out


def plot_topology_workload(grid, configs, model_name, baseline, out_path):
    """Grouped bar chart: x = configs, groups = workloads."""
    fig, ax = plt.subplots(figsize=(14, 6))
    n = len(configs); w = 0.26
    x = np.arange(n)
    for i, wl in enumerate(WORKLOADS):
        vals = [grid.get((c[0], wl), 0) for c in configs]
        ax.bar(x + (i-1)*w, vals, w, label=WL_LABEL[wl], color=WL_COLOR[wl])
    if baseline:
        ax.axhline(baseline, color="red", linestyle="--", linewidth=1, alpha=0.6,
                   label=f"stock vLLM baseline ({baseline} TPS)")
    ax.set_xticks(x)
    ax.set_xticklabels([c[1] for c in configs], rotation=35, ha="right")
    ax.set_ylabel("TPS (output tokens / s, wall)")
    ax.set_title(f"{model_name} — 4+4 cross-node sweep (n_req=128)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_layer_skew_curve(grid, configs, model_name, out_path):
    """Layer skew sweep (TP4PP2 only): x = skew level, lines = workloads."""
    skew_cfgs = [c for c in configs if c[0].startswith("TP4PP2")]
    skew_labels = []
    for c in skew_cfgs:
        label = c[1].split("[")[-1].rstrip("]")
        skew_labels.append(label)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for wl in WORKLOADS:
        ys = [grid.get((c[0], wl), 0) for c in skew_cfgs]
        ax.plot(range(len(skew_cfgs)), ys, marker="o", linewidth=2,
                label=WL_LABEL[wl], color=WL_COLOR[wl])
    ax.set_xticks(range(len(skew_cfgs)))
    ax.set_xticklabels(skew_labels, rotation=15)
    ax.set_xlabel("Layer split (head Blackwell : worker Ada)")
    ax.set_ylabel("TPS")
    ax.set_title(f"{model_name} TP=4 PP=2 layer-skew sweep — workload-dependent peak")
    ax.legend(loc="best", fontsize=10)
    ax.grid(linestyle=":", alpha=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_topology_overview(grid70, configs70, grid8, configs8, out_path):
    """Group by topology family: pick best of each topology for each model & workload."""
    def topo_of(label):
        if "TP8PP1" in label: return "TP=8 PP=1"
        if "TP4PP2" in label: return "TP=4 PP=2"
        if "TP2PP4" in label: return "TP=2 PP=4"
        return label
    def best_per_topo(grid, configs):
        out = {}  # topo -> {wl: best_tps}
        for c, _ in configs:
            topo = topo_of(c)
            for wl in WORKLOADS:
                v = grid.get((c, wl), 0)
                out.setdefault(topo, {}).setdefault(wl, 0)
                if v > out[topo][wl]: out[topo][wl] = v
        return out
    top70 = best_per_topo(grid70, configs70)
    top8  = best_per_topo(grid8,  configs8)
    topos = ["TP=8 PP=1", "TP=4 PP=2", "TP=2 PP=4"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (name, tdict) in zip(axes, [("70B", top70), ("8B", top8)]):
        x = np.arange(len(topos)); w = 0.26
        for i, wl in enumerate(WORKLOADS):
            vals = [tdict.get(t, {}).get(wl, 0) for t in topos]
            ax.bar(x + (i-1)*w, vals, w, label=WL_LABEL[wl], color=WL_COLOR[wl])
        ax.set_xticks(x); ax.set_xticklabels(topos)
        ax.set_ylabel("TPS (best layer split)")
        ax.set_title(f"{name} — topology best-of")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    axes[1].legend(loc="upper right", fontsize=9)
    plt.suptitle("4+4 cross-node: which topology wins (model × workload)", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_heatmap(grid, configs, model_name, out_path):
    """Heatmap: rows = configs, cols = workloads, color = TPS."""
    mat = np.array([[grid.get((c[0], wl), 0) for wl in WORKLOADS] for c in configs])
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(WORKLOADS)))
    ax.set_xticklabels([w.replace("_heavy","") for w in WORKLOADS])
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([c[1] for c in configs])
    for i in range(len(configs)):
        for j in range(len(WORKLOADS)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v < mat.max()*0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="TPS")
    ax.set_title(f"{model_name} — config × workload heatmap")
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "figures"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    recs_70b = load_recs(str(REPO/"results/hetero_4x4_70b_full_*/*/record.json"), CUTOFF_70B)
    recs_8b  = load_recs(str(REPO/"results/hetero_4x4_8b_full_*/*/record.json"),  CUTOFF_8B)
    print(f"loaded 70B={len(recs_70b)} 8B={len(recs_8b)} success cells")

    grid_70b = build_grid(recs_70b, CONFIGS_70B)
    grid_8b  = build_grid(recs_8b,  CONFIGS_8B)

    plot_topology_workload(grid_70b, CONFIGS_70B, "Llama-3.3 70B", STOCK_BASELINE_70B,
                           out/"fig1_70b_topology_workload.png")
    plot_topology_workload(grid_8b,  CONFIGS_8B,  "Llama-3.1 8B",  STOCK_BASELINE_8B,
                           out/"fig1_8b_topology_workload.png")
    plot_layer_skew_curve(grid_70b, CONFIGS_70B, "Llama-3.3 70B",
                          out/"fig2_70b_layer_skew_curve.png")
    plot_layer_skew_curve(grid_8b,  CONFIGS_8B,  "Llama-3.1 8B",
                          out/"fig2_8b_layer_skew_curve.png")
    plot_topology_overview(grid_70b, CONFIGS_70B, grid_8b, CONFIGS_8B,
                           out/"fig3_topology_overview.png")
    plot_heatmap(grid_70b, CONFIGS_70B, "Llama-3.3 70B", out/"fig4_70b_heatmap.png")
    plot_heatmap(grid_8b,  CONFIGS_8B,  "Llama-3.1 8B",  out/"fig4_8b_heatmap.png")
    print(f"\nall plots saved to {out}/")


if __name__ == "__main__":
    main()
