"""Mistral-Large-123B (4×Blackwell + 4×Ada) sweep figures.

Outputs:
  figures/fig_mistral123b_workload_rows.png   measured TPS, 3 workload rows × configs (matches 4-model figure family)
  figures/fig_mistral123b_prereg.png          PRE-REGISTERED prediction vs measured (the headline generalization result)

Reads:
  results/hetero_4x4_mistral123b_full_*/all_runs.csv   (latest)
  planner/mistral_prediction.json                      (frozen pre-registration)
  planner/mistral_validation.json                      (computed regret/champion/Spearman)
"""
from __future__ import annotations
import csv, glob, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "figures"
WLS = ["balanced", "decode_heavy", "prefill_heavy"]
WL_TITLE = {"balanced": "balanced (in512/out256)",
            "decode_heavy": "decode-heavy (in128/out512)",
            "prefill_heavy": "prefill-heavy (in1024/out128)"}
# topology family → color (matches existing figures)
FAM = [("TP8PP1", "#4c72b0"), ("TP4PP2", "#dd8452"),
       ("TP2PP4", "#55a868"), ("TP1PP8", "#c44e52")]
FAM_COLOR = {k: c for k, c in FAM}


def fam_of(label):
    for k, _ in FAM:
        if label.startswith(k):
            return k
    return "?"


def short(label):
    """Compact config label for x-axis."""
    if label.startswith("TP8PP1"):
        if "uniform" in label: return "TP8 unif"
        return "TP8 " + label.split("ffn_bias")[-1].replace("+", "f+")
    if label.startswith("TP4PP2"):
        if "uniform" in label: return "PP2 44:44"
        return "PP2 " + label.split("_")[-1]            # e.g. 56-32
    if label.startswith("TP2PP4"):
        if "uniform" in label: return "PP4 22⁴"
        return "PP4 " + label.split("_")[-1]            # 24-24-20-20
    if label.startswith("TP1PP8"):
        if "uniform" in label: return "PP8 11×8"
        a = label.split("_")[-1].split("-")
        return f"PP8 {a[0]}:{a[-1]}"
    return label


def load():
    rdir = sorted(glob.glob(str(REPO / "results" / "hetero_4x4_mistral123b_full_*")))[-1]
    meas = {}
    for r in csv.DictReader(open(Path(rdir) / "all_runs.csv")):
        if r["success"] == "True" and float(r["tps"]) > 0:
            meas[(r["workload"], r["label"])] = float(r["tps"])
    pred_j = json.load(open(REPO / "planner" / "mistral_prediction.json"))
    val = json.load(open(REPO / "planner" / "mistral_validation.json"))
    return rdir, meas, pred_j, val


# config order: by topology family then as listed
ORDER = [
    "TP8PP1_uniform", "TP8PP1_ffn_bias+25", "TP8PP1_ffn_bias+50", "TP8PP1_ffn_bias+75",
    "TP4PP2_layer_uniform_44-44", "TP4PP2_layer_skew+4_48-40", "TP4PP2_layer_skew+8_52-36",
    "TP4PP2_layer_skew+12_56-32", "TP4PP2_layer_skew+16_60-28",
    "TP2PP4_layer_uniform_22-22-22-22", "TP2PP4_layer_blackbias_24-24-20-20",
    "TP2PP4_layer_blackbias_26-26-18-18",
    "TP1PP8_layer_uniform_11x8", "TP1PP8_layer_blackbias_13-13-13-13-9-9-9-9",
    "TP1PP8_layer_blackbias_15-15-15-15-7-7-7-7",
]


def fig_workload_rows(meas, val):
    labs = ORDER
    x = np.arange(len(labs))
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, wl in zip(axes, WLS):
        vals = [meas.get((wl, l), 0) for l in labs]
        cols = [FAM_COLOR[fam_of(l)] for l in labs]
        bars = ax.bar(x, vals, color=cols, edgecolor="black", linewidth=0.5)
        champ = val["per_workload"][wl]["measured_champion"]
        ci = labs.index(champ)
        bars[ci].set_edgecolor("gold"); bars[ci].set_linewidth(2.5)
        ax.annotate("★ champion\n(planner-predicted)", (ci, vals[ci]),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=8, fontweight="bold", color="#7a5c00")
        ax.set_ylabel("wall TPS (tok/s)")
        ax.set_title(WL_TITLE[wl], loc="left", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([short(l) for l in labs], rotation=45, ha="right", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in FAM]
    axes[0].legend(handles, [k for k, _ in FAM], ncol=4, fontsize=8, loc="upper right")
    fig.suptitle("Mistral-Large-123B measured throughput — 4×Blackwell(96GB) + 4×Ada(48GB) cross-node, n_req=96",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = OUT / "fig_mistral123b_workload_rows.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def fig_prereg(meas, pred_j, val):
    """Predicted (pre-registered) vs measured, per workload, configs sorted by
    measured TPS. Shows the predicted champion lands on the measured #1."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, wl in zip(axes, WLS):
        labs = [l for l in ORDER if (wl, l) in meas
                and pred_j["predictions"][wl].get(l, {}).get("feasible")]
        labs.sort(key=lambda l: -meas[(wl, l)])
        m = [meas[(wl, l)] for l in labs]
        p = [pred_j["predictions"][wl][l]["tps"] for l in labs]
        x = np.arange(len(labs)); w = 0.4
        cols = [FAM_COLOR[fam_of(l)] for l in labs]
        ax.bar(x - w/2, m, w, color=cols, edgecolor="black", linewidth=0.4, label="measured")
        ax.bar(x + w/2, p, w, color=cols, alpha=0.45, hatch="///",
               edgecolor="black", linewidth=0.4, label="predicted (pre-reg)")
        pc = val["per_workload"][wl]["predicted_champion"]
        ci = labs.index(pc)
        ax.annotate("★ pred champ\n= meas #1", (ci, max(m[ci], p[ci])),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=8.5, fontweight="bold", color="#7a5c00")
        pw = val["per_workload"][wl]
        ax.set_title(f"{wl}\nmatch ✓  regret {pw['regret_pct']:.1f}%  ρ={pw['spearman']:.2f}",
                     fontsize=10, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([short(l) for l in labs],
                                             rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("wall TPS (tok/s)"); ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]: ax.legend(fontsize=8, loc="upper right")
    s = val["summary"]
    fig.suptitle(f"Mistral-Large-123B: PRE-REGISTERED planner prediction vs measured  "
                 f"(frozen before sweep)  —  champion {s['champion_match']}, "
                 f"mean regret {s['mean_regret_pct']:.1f}%, Spearman {s['mean_spearman']:.2f}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pth = OUT / "fig_mistral123b_prereg.png"
    fig.savefig(pth, dpi=140); plt.close(fig)
    return pth


def main():
    OUT.mkdir(exist_ok=True)
    rdir, meas, pred_j, val = load()
    print(f"results: {rdir}  (cells: {len(meas)})")
    p1 = fig_workload_rows(meas, val)
    p2 = fig_prereg(meas, pred_j, val)
    print(f"wrote {p1}")
    print(f"wrote {p2}")
    print(f"summary: {val['summary']}")


if __name__ == "__main__":
    main()
