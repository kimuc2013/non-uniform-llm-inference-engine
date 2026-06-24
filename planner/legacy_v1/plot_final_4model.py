"""Aggregate all 4 models (70B, 8B, OPT-30B, Qwen3-32B) into:
  1. fig_4model_topology_grid.png  — 4×4 grid (model × topology) of bar charts
  2. fig_4model_champion_table.md   — champion per (model, workload)
  3. fig_4model_topology_lines.png  — line plot: x = topology, y = TPS, lines = models

Labels use per-node/per-stage ratios.
"""
from __future__ import annotations
import csv, glob, json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results" / "final" / "plots"


def make_split_label(label, ffn_splits, head_splits, kv_splits, layer_split):
    """Convert (label, splits) → human-readable per-node/per-stage string."""
    if label.startswith("TP8PP1"):
        # TP8 PP1 — Blackwell ranks 0-3, Ada ranks 4-7
        ffn_b, ffn_a = ffn_splits[0], ffn_splits[4]
        h_b, h_a = head_splits[0], head_splits[4]
        kv_b, kv_a = kv_splits[0], kv_splits[4]
        parts = []
        if ffn_b != ffn_a:
            parts.append(f"ffn({ffn_b}:{ffn_a})")
        if h_b != h_a:
            if kv_b != kv_a:
                parts.append(f"head({h_b}:{h_a})")
            else:
                parts.append(f"q({h_b}:{h_a})")
        if not parts:
            return "TP8 PP1 uniform"
        return "TP8 PP1 " + " ".join(parts)

    elif label.startswith("TP4PP2"):
        # TP4PP2 — 2 stages, stage 0 Blackwell (ranks 0-3), stage 1 Ada (ranks 4-7)
        return f"PP2 TP4 ({layer_split[0]}:{layer_split[1]})"

    elif label.startswith("TP2PP4"):
        # TP2PP4 — 4 stages: head 0,1 / worker 2,3
        return f"PP4 TP2 ({':'.join(str(l) for l in layer_split)})"

    elif label.startswith("TP1PP8"):
        # TP1PP8 — 8 stages: head 0-3 / worker 4-7
        # Show compact form if pattern (BBBB|AAAA) detected
        if len(layer_split) == 8 and all(l == layer_split[0] for l in layer_split[:4]) and all(l == layer_split[4] for l in layer_split[4:]):
            if layer_split[0] == layer_split[4]:
                return f"PP8 TP1 ({layer_split[0]}×8)"
            return f"PP8 TP1 ({layer_split[0]}×4 : {layer_split[4]}×4)"
        return f"PP8 TP1 ({':'.join(str(l) for l in layer_split)})"
    return label


MODELS = ["8b", "70b", "opt30b", "qwen32b"]
MODEL_TITLE = {
    "8b": "Llama-3.1-8B",
    "70b": "Llama-3.3-70B",
    "opt30b": "OPT-30B (n_req=64)",
    "qwen32b": "Qwen3-32B",
}
TOPOLOGIES = [
    ("TP8 PP1", "TP8PP1"),
    ("TP4 PP2", "TP4PP2"),
    ("TP2 PP4", "TP2PP4"),
    ("TP1 PP8", "TP1PP8"),
]
WLS = ["balanced", "decode_heavy", "prefill_heavy"]
WL_COLOR = {"balanced": "#4c72b0", "decode_heavy": "#dd8452", "prefill_heavy": "#55a868"}


def collect_model(model):
    """Return list[ rec ] for one model. rec has label, workload, tps, splits, layer_split."""
    pat = f"{REPO}/results/hetero_4x4_{model}_full_*"
    dirs = sorted(glob.glob(pat), key=os.path.getmtime)
    rows = []
    for d in dirs:
        for rj in sorted(glob.glob(os.path.join(d, f"{model}_*/record.json"))):
            try:
                r = json.load(open(rj))
                if not r.get("success"): continue
                rows.append(r)
            except Exception as e:
                print(f"skip {rj}: {e}", file=sys.stderr)
    # Latest-wins per (label, workload)
    by_key = {}
    for r in rows:
        by_key[(r["label"], r["workload"])] = r
    return list(by_key.values())


def main():
    all_data = {m: collect_model(m) for m in MODELS}
    for m, recs in all_data.items():
        print(f"{m}: {len(recs)} cells")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ======================================================
    # 1) 4x4 grid: rows = models, cols = topologies
    # ======================================================
    fig, axes = plt.subplots(len(MODELS), len(TOPOLOGIES), figsize=(28, 18))
    for mi, model in enumerate(MODELS):
        recs = all_data[model]
        for ti, (topo_title, topo_prefix) in enumerate(TOPOLOGIES):
            ax = axes[mi][ti]
            cells = [r for r in recs if r["label"].startswith(topo_prefix)]
            if not cells:
                ax.set_visible(False); continue

            # Group by label, sort to put uniform first then bias variants
            labels = sorted({r["label"] for r in cells}, key=lambda L: (
                0 if "uniform" in L else 1, L
            ))
            short_labels = []
            for L in labels:
                # find a sample record to extract splits
                sample = next(r for r in cells if r["label"] == L)
                short_labels.append(make_split_label(
                    L, sample["ffn_splits"], sample["head_splits"],
                    sample["kv_splits"], sample["layer_split"]))

            n = len(labels); x = np.arange(n); w = 0.27
            for wi, wl in enumerate(WLS):
                ys = []
                for L in labels:
                    matches = [r for r in cells if r["label"] == L and r["workload"] == wl]
                    ys.append(matches[0]["tps"] if matches else 0)
                bars = ax.bar(x + (wi-1)*w, ys, w, label=wl, color=WL_COLOR[wl], alpha=0.9)
                for bar, y in zip(bars, ys):
                    if y > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, y + max(ys)*0.01 if max(ys) > 0 else 0,
                                f"{y:.0f}", ha="center", va="bottom", fontsize=6, rotation=0)
            ax.set_xticks(x)
            ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=8)
            if mi == 0:
                ax.set_title(topo_title, fontsize=11, fontweight="bold")
            if ti == 0:
                ax.set_ylabel(MODEL_TITLE[model] + "\nTPS (tok/s)", fontsize=10, fontweight="bold")
            ax.grid(axis="y", linestyle=":", alpha=0.3)
            if mi == 0 and ti == len(TOPOLOGIES) - 1:
                ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("4+4 cross-node hetero serving — 4 models × 4 topologies × non-uniform variants × 3 workloads",
                 fontsize=14, y=0.995)
    plt.tight_layout()
    out1 = OUT_DIR / "fig_4model_topology_grid.png"
    fig.savefig(out1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out1} ({out1.stat().st_size} bytes)")

    # ======================================================
    # 2) Champion table (markdown)
    # ======================================================
    table_lines = ["# 4-model champion per (workload, topology)\n",
                   "Champion = highest TPS config within (model, workload, topology).\n",
                   "Δ% = champion TPS vs the topology's uniform variant.\n"]
    for model in MODELS:
        recs = all_data[model]
        table_lines.append(f"\n## {MODEL_TITLE[model]}\n")
        table_lines.append("| Workload | Topology | Uniform TPS | Champion config | Champion TPS | Δ% vs uniform |")
        table_lines.append("|---|---|---:|---|---:|---:|")
        for wl in WLS:
            for topo_title, topo_prefix in TOPOLOGIES:
                cells = [r for r in recs if r["label"].startswith(topo_prefix) and r["workload"] == wl]
                if not cells: continue
                cells.sort(key=lambda r: r["tps"], reverse=True)
                champ = cells[0]
                uniform = next((r for r in cells if "uniform" in r["label"]), None)
                uni_tps = uniform["tps"] if uniform else 0
                delta = f"{(champ['tps'] - uni_tps)/uni_tps*100:+.1f}%" if uni_tps else "—"
                champ_label = make_split_label(
                    champ["label"], champ["ffn_splits"], champ["head_splits"],
                    champ["kv_splits"], champ["layer_split"])
                table_lines.append(
                    f"| {wl} | {topo_title} | {uni_tps:.0f} | {champ_label} | {champ['tps']:.0f} | {delta} |"
                )

    # Overall champion per workload (across all topologies)
    table_lines.append("\n## Overall champion per workload (across all topologies)\n")
    table_lines.append("| Model | Workload | Champion config | TPS | Δ% vs TP8 PP1 uniform |")
    table_lines.append("|---|---|---|---:|---:|")
    for model in MODELS:
        recs = all_data[model]
        for wl in WLS:
            cells = [r for r in recs if r["workload"] == wl]
            if not cells: continue
            cells.sort(key=lambda r: r["tps"], reverse=True)
            champ = cells[0]
            tp8_uniform = next((r for r in recs if r["label"] == "TP8PP1_uniform" and r["workload"] == wl), None)
            base_tps = tp8_uniform["tps"] if tp8_uniform else 0
            delta = f"{(champ['tps'] - base_tps)/base_tps*100:+.1f}%" if base_tps else "—"
            champ_label = make_split_label(
                champ["label"], champ["ffn_splits"], champ["head_splits"],
                champ["kv_splits"], champ["layer_split"])
            table_lines.append(
                f"| {MODEL_TITLE[model]} | {wl} | {champ_label} | {champ['tps']:.0f} | {delta} |"
            )

    out2 = REPO / "results" / "final" / "champion_4model.md"
    out2.write_text("\n".join(table_lines))
    print(f"saved {out2}")

    # ======================================================
    # 3) Topology comparison lines: x = topology, y = TPS, lines = models
    # Plot per-workload (3 panels). Y = uniform baseline of topology.
    # ======================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    MODEL_COLOR = {"8b": "#1f77b4", "70b": "#ff7f0e", "opt30b": "#2ca02c", "qwen32b": "#d62728"}
    for wi, wl in enumerate(WLS):
        ax = axes[wi]
        x = np.arange(len(TOPOLOGIES))
        for model in MODELS:
            recs = all_data[model]
            ys_uniform = []
            ys_best = []
            for topo_title, topo_prefix in TOPOLOGIES:
                # uniform within topology
                u = next((r for r in recs
                          if r["label"].startswith(topo_prefix) and "uniform" in r["label"]
                          and r["workload"] == wl), None)
                ys_uniform.append(u["tps"] if u else 0)
                # best within topology
                cells = [r for r in recs if r["label"].startswith(topo_prefix) and r["workload"] == wl]
                if cells:
                    cells.sort(key=lambda r: r["tps"], reverse=True)
                    ys_best.append(cells[0]["tps"])
                else:
                    ys_best.append(0)
            ax.plot(x, ys_uniform, ":", marker="o", label=f"{MODEL_TITLE[model]} uniform",
                    color=MODEL_COLOR[model], alpha=0.5)
            ax.plot(x, ys_best, "-", marker="s", label=f"{MODEL_TITLE[model]} champion",
                    color=MODEL_COLOR[model], linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels([t[0] for t in TOPOLOGIES])
        ax.set_title(wl, fontsize=12, fontweight="bold")
        ax.set_xlabel("Topology")
        ax.set_ylabel("TPS (tok/s)")
        ax.grid(linestyle=":", alpha=0.3)
        ax.legend(fontsize=8, loc="best", ncol=2)

    fig.suptitle("Per-model topology comparison — dotted = uniform within topology, solid = champion config",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    out3 = OUT_DIR / "fig_4model_topology_lines.png"
    fig.savefig(out3, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out3} ({out3.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
