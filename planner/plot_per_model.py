"""Per-model 4-topology plot. Pass --model {8b,70b,opt30b,qwen32b}.

Output: results/final/plots/fig_<model>_4topology_nonuniform.png
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results" / "final" / "plots"

MODEL_TOPOS = {
    "70b": [
        ("TP1 PP8", "TP1PP8_layer_uniform_10x8", "uniform [10]×8", [
            ("TP1PP8_layer_blackbias_12-12-12-12-8-8-8-8", "blackbias [12-8] +20%"),
            ("TP1PP8_layer_blackbias_14-14-14-14-6-6-6-6", "blackbias [14-6] +40%"),
        ]),
        ("TP2 PP4", "TP2PP4_layer_uniform_20-20-20-20", "uniform [20]×4", [
            ("TP2PP4_layer_blackbias_22-22-18-18", "blackbias 22-18"),
            ("TP2PP4_layer_blackbias_24-24-16-16", "blackbias 24-16"),
        ]),
        ("TP4 PP2", "TP4PP2_layer_uniform_40-40", "uniform [40,40]", [
            ("TP4PP2_layer_skew+4_44-36", "skew +4"),
            ("TP4PP2_layer_skew+8_48-32", "skew +8"),
            ("TP4PP2_layer_skew+12_52-28", "skew +12"),
            ("TP4PP2_layer_skew+16_56-24", "skew +16"),
        ]),
        ("TP8 PP1", "TP8PP1_uniform", "uniform", [
            ("TP8PP1_ffn_bias+25", "FFN +25"),
            ("TP8PP1_ffn_bias+50", "FFN +50"),
            ("TP8PP1_ffn_bias+75", "FFN +75"),
        ]),
    ],
    "8b": [
        ("TP1 PP8", "TP1PP8_layer_uniform_4x8", "uniform [4]×8", [
            ("TP1PP8_layer_blackbias_5-5-5-5-3-3-3-3", "blackbias [5-3] +25%"),
            ("TP1PP8_layer_blackbias_6-6-6-6-2-2-2-2", "blackbias [6-2] +50%"),
        ]),
        ("TP2 PP4", "TP2PP4_layer_uniform_8-8-8-8", "uniform [8]×4", [
            ("TP2PP4_layer_blackbias_9-9-7-7", "blackbias 9-7"),
            ("TP2PP4_layer_blackbias_10-10-6-6", "blackbias 10-6"),
        ]),
        ("TP4 PP2", "TP4PP2_layer_uniform_16-16", "uniform [16,16]", [
            ("TP4PP2_layer_skew+2_18-14", "skew +2"),
            ("TP4PP2_layer_skew+4_20-12", "skew +4"),
            ("TP4PP2_layer_skew+6_22-10", "skew +6"),
            ("TP4PP2_layer_skew+8_24-8", "skew +8"),
        ]),
        ("TP8 PP1", "TP8PP1_uniform", "uniform", [
            ("TP8PP1_ffn_bias+25", "FFN +25"),
            ("TP8PP1_ffn_bias+50", "FFN +50"),
            ("TP8PP1_ffn_bias+75", "FFN +75"),
        ]),
    ],
    "opt30b": [
        ("TP1 PP8", "TP1PP8_layer_uniform_6x8", "uniform [6]×8", [
            ("TP1PP8_layer_blackbias_7-7-7-7-5-5-5-5", "blackbias [7-5]"),
            ("TP1PP8_layer_blackbias_8-8-8-8-4-4-4-4", "blackbias [8-4]"),
        ]),
        ("TP2 PP4", "TP2PP4_layer_uniform_12-12-12-12", "uniform [12]×4", [
            ("TP2PP4_layer_blackbias_14-14-10-10", "blackbias 14-10"),
            ("TP2PP4_layer_blackbias_16-16-8-8", "blackbias 16-8"),
        ]),
        ("TP4 PP2", "TP4PP2_layer_uniform_24-24", "uniform [24,24]", [
            ("TP4PP2_layer_skew+4_28-20", "skew +4"),
            ("TP4PP2_layer_skew+8_32-16", "skew +8"),
            ("TP4PP2_layer_skew+12_36-12", "skew +12"),
            ("TP4PP2_layer_skew+16_40-8", "skew +16"),
        ]),
        ("TP8 PP1", "TP8PP1_uniform", "uniform", [
            ("TP8PP1_ffn_bias+25", "FFN +25"),
            ("TP8PP1_ffn_bias+50", "FFN +50"),
            ("TP8PP1_ffn_bias+75", "FFN +75"),
            ("TP8PP1_head_bias+2", "Head 9-5"),
            ("TP8PP1_head_bias+3", "Head 10-4"),
            ("TP8PP1_head_bias+4", "Head 11-3"),
            ("TP8PP1_hybrid_ffn50_head3", "Hybrid FFN+50/Head 10-4"),
        ]),
    ],
    "qwen32b": [
        ("TP1 PP8", "TP1PP8_layer_uniform_8x8", "uniform [8]×8", [
            ("TP1PP8_layer_blackbias_10-10-10-10-6-6-6-6", "blackbias [10-6]"),
            ("TP1PP8_layer_blackbias_12-12-12-12-4-4-4-4", "blackbias [12-4]"),
        ]),
        ("TP2 PP4", "TP2PP4_layer_uniform_16-16-16-16", "uniform [16]×4", [
            ("TP2PP4_layer_blackbias_18-18-14-14", "blackbias 18-14"),
            ("TP2PP4_layer_blackbias_20-20-12-12", "blackbias 20-12"),
        ]),
        ("TP4 PP2", "TP4PP2_layer_uniform_32-32", "uniform [32,32]", [
            ("TP4PP2_layer_skew+4_36-28", "skew +4"),
            ("TP4PP2_layer_skew+8_40-24", "skew +8"),
            ("TP4PP2_layer_skew+12_44-20", "skew +12"),
            ("TP4PP2_layer_skew+16_48-16", "skew +16"),
        ]),
        ("TP8 PP1", "TP8PP1_uniform", "uniform", [
            ("TP8PP1_ffn_bias+25", "FFN +25"),
            ("TP8PP1_ffn_bias+50", "FFN +50"),
            ("TP8PP1_ffn_bias+75", "FFN +75"),
        ]),
    ],
}

WLS = ["balanced", "decode_heavy", "prefill_heavy"]
WL_COLOR = {"balanced":"#4c72b0", "decode_heavy":"#dd8452", "prefill_heavy":"#55a868"}


def collect(model: str) -> dict:
    """Walk all sweep dirs for this model, collect tps with overlap preference."""
    pat = f"/data/esca/uckim/vllm_main/results/hetero_4x4_{model}_full_*"
    dirs = sorted(glob.glob(pat), key=os.path.getmtime)
    rows = []
    for d in dirs:
        for rj in sorted(glob.glob(os.path.join(d, f"{model}_*/record.json"))):
            try:
                r = json.load(open(rj))
                if not r.get("success"): continue
                rows.append({
                    "label": r.get("label", ""), "workload": r.get("workload", ""),
                    "tps": float(r.get("tps", 0)), "pp": r.get("pp", 0),
                    "dir": d,
                })
            except Exception as e:
                print(f"skip {rj}: {e}", file=sys.stderr)
    # Latest-wins per (label, workload): files come in mtime order, so dict overwrite works
    by_key = {}
    for r in rows:
        by_key[(r["label"], r["workload"])] = r["tps"]
    return by_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_TOPOS.keys()))
    args = ap.parse_args()

    by_key = collect(args.model)
    print(f"collected {len(by_key)} (label,wl) entries for {args.model}")
    if not by_key:
        print(f"no data for {args.model}", file=sys.stderr); return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    topos = MODEL_TOPOS[args.model]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for ax_i, (topo_name, uniform_label, uniform_short, variants) in enumerate(topos):
        ax = axes[ax_i]
        labels = [uniform_short] + [v[1] for v in variants]
        all_labels = [uniform_label] + [v[0] for v in variants]
        n = len(labels)
        x = np.arange(n)
        w = 0.27
        for i, wl in enumerate(WLS):
            ys = [by_key.get((lab, wl), 0) for lab in all_labels]
            bars = ax.bar(x + (i - 1) * w, ys, w, label=wl, color=WL_COLOR[wl], alpha=0.9)
            for bar, y in zip(bars, ys):
                if y > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, y + max(ys) * 0.01,
                            f"{y:.0f}", ha="center", va="bottom", fontsize=7)
        for wl in WLS:
            base = by_key.get((uniform_label, wl))
            if base:
                ax.axhline(base, color=WL_COLOR[wl], linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(topo_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("TPS (tok/s)")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"{args.model} 4+4 cross-node — uniform baseline vs non-uniform "
                 f"(PP cells use M13 overlap; dotted lines = uniform per workload)",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    out = OUT_DIR / f"fig_{args.model}_4topology_nonuniform.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
