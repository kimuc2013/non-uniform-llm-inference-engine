"""Re-plot 70B 4+4 sweep:
  4 subplots, one per topology (TP1PP8 / TP2PP4 / TP4PP2 / TP8PP1).
  Each subplot: x = (uniform baseline) + non-uniform variants, grouped by workload.

PP cells use overlap regime (M13 broadcast_stream + microbatch path).
TP8PP1 cells are TP-only (regime='na').

Replaces old plots: deletes results/final/plots/* before saving.
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

# Topology definitions: (topology_label, uniform_label, [(non_uniform_label, short)...])
TOPOS = [
    ("TP1 PP8", "TP1PP8_layer_uniform_10x8", "uniform [10]x8", [
        ("TP1PP8_layer_blackbias_12-12-12-12-8-8-8-8", "blackbias 12-8 (+20%)"),
        ("TP1PP8_layer_blackbias_14-14-14-14-6-6-6-6", "blackbias 14-6 (+40%)"),
    ]),
    ("TP2 PP4", "TP2PP4_layer_uniform_20-20-20-20", "uniform [20]x4", [
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
]

WLS = ["balanced", "decode_heavy", "prefill_heavy"]
WL_LABEL = {"balanced":"balanced", "decode_heavy":"decode_heavy", "prefill_heavy":"prefill_heavy"}
WL_COLOR = {"balanced":"#4c72b0", "decode_heavy":"#dd8452", "prefill_heavy":"#55a868"}


def collect_data() -> list[dict]:
    """Walk all hetero_4x4_70b_full_* dirs, prefer overlap regime when both present."""
    rows = []
    # Source preference order: overlap dirs LAST so they take precedence on key overwrite
    sources = [
        ("/data/esca/uckim/vllm_main/results/hetero_4x4_70b_full_20260609_122900", "stock"),
        ("/data/esca/uckim/vllm_main/results/hetero_4x4_70b_full_20260609_170850", "overlap"),
    ]
    # TP1PP8 dir(s) — find latest
    tp1pp8_dirs = sorted(glob.glob("/data/esca/uckim/vllm_main/results/hetero_4x4_70b_full_*"),
                          key=os.path.getmtime, reverse=True)
    for d in tp1pp8_dirs:
        if any(os.path.exists(os.path.join(d, f"70b_TP1PP8_{x[0][len('TP1PP8_'):]}_balanced/record.json"))
               for x in TOPOS[0][3]) or os.path.exists(os.path.join(d, "70b_TP1PP8_layer_uniform_10x8_balanced/record.json")):
            sources.append((d, "overlap"))
            break
    for src, regime in sources:
        for rj in sorted(glob.glob(os.path.join(src, "70b_*/record.json"))):
            try:
                r = json.load(open(rj))
                if not r.get("success"): continue
                pp = r.get("pp", 0)
                if pp == 1 and regime == "overlap":
                    continue
                rows.append({
                    "label": r.get("label", ""),
                    "workload": r.get("workload", ""),
                    "tps": float(r.get("tps", 0)),
                    "regime": regime if pp > 1 else "na",
                })
            except Exception as e:
                print(f"skip {rj}: {e}", file=sys.stderr)
    return rows


def tps_lookup(rows: list[dict]) -> dict:
    """label×workload → tps. PP cells use overlap if present, else stock."""
    by_key: dict[tuple[str,str,str], float] = {}
    for r in rows:
        k = (r["label"], r["workload"], r["regime"])
        by_key[k] = r["tps"]

    def fetch(label, wl):
        # overlap > stock > na
        for reg in ("overlap", "stock", "na"):
            if (label, wl, reg) in by_key:
                return by_key[(label, wl, reg)]
        return None
    return fetch


def main():
    rows = collect_data()
    print(f"collected {len(rows)} rows")
    fetch = tps_lookup(rows)

    # Delete old plots
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.png"):
        old.unlink()
        print(f"deleted {old}")

    # One subplot per topology
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for ax_i, (topo, uniform_label, uniform_short, variants) in enumerate(TOPOS):
        ax = axes[ax_i]
        labels = [uniform_short] + [v[1] for v in variants]
        all_labels = [uniform_label] + [v[0] for v in variants]
        n = len(labels)
        x = np.arange(n)
        w = 0.27

        for i, wl in enumerate(WLS):
            ys = [fetch(lab, wl) or 0 for lab in all_labels]
            bars = ax.bar(x + (i - 1) * w, ys, w, label=WL_LABEL[wl],
                          color=WL_COLOR[wl], alpha=0.9)
            for bar, y in zip(bars, ys):
                if y > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, y + 15,
                            f"{y:.0f}", ha="center", va="bottom", fontsize=7, rotation=0)

        # Uniform baseline horizontal lines per workload
        for wl in WLS:
            base = fetch(uniform_label, wl)
            if base:
                ax.axhline(base, color=WL_COLOR[wl], linestyle=":", linewidth=0.8, alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{topo}", fontsize=12, fontweight="bold")
        ax.set_ylabel("TPS (tok/s)")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Llama-3.3-70B 4+4 cross-node — uniform baseline vs non-uniform variants "
                 "(PP cells use M13 overlap; dotted lines = uniform per workload)",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    out_path = OUT_DIR / "fig_70b_4topology_nonuniform.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
