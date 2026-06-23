"""Simple paper figure: Llama-70B on 2 Blackwell + 2 Ada, balanced (n_req=16).
uniform vs non-uniform, for TP and PP. One grouped bar chart of wall TPS."""
from __future__ import annotations
import csv, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
rdir = sorted(glob.glob(str(REPO / "results" / "hetero_2x2_70b_balanced_*")))[-1]
rows = {r["label"]: r for r in csv.DictReader(open(Path(rdir) / "all_runs.csv"))}
def tps(l): return float(rows[l]["tps"])

groups = [
    ("Tensor Parallel\n(TP4, cross-node)", "uniform_TP4", "nonuniform_TP4", "ffn 7168:7168", "ffn 9600:4736"),
    ("Pipeline Parallel\n(TP2-PP2)",        "uniform_PP",  "nonuniform_PP",  "layers 40:40",  "layers 52:28"),
]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
x, labels, colors = [], [], []
UNI, NON = "#b0b0b0", "#dd8452"   # uniform=grey, non-uniform=orange
pos = 0; ticks, tick_lab = [], []
for gname, ul, nl, usub, nsub in groups:
    for lbl, sub, col in [(ul, usub, UNI), (nl, nsub, NON)]:
        x.append(pos); colors.append(col)
        labels.append(("uniform" if col == UNI else "non-uniform") + f"\n{sub}")
        pos += 1
    ticks.append(pos - 1.5); tick_lab.append(gname)
    pos += 0.8
vals = [tps("uniform_TP4"), tps("nonuniform_TP4"), tps("uniform_PP"), tps("nonuniform_PP")]
bars = ax.bar(x, vals, color=colors, edgecolor="black", width=0.8)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 4, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")
# gain annotations (uniform -> non-uniform within each group)
for i, (gname, ul, nl, *_), in zip([(0, 1), (2, 3)], groups):
    u_i, n_i = i
    gain = (vals[n_i] - vals[u_i]) / vals[u_i] * 100
    xm = (x[u_i] + x[n_i]) / 2
    ytop = max(vals[u_i], vals[n_i]) + 24
    ax.annotate("", xy=(x[n_i], ytop), xytext=(x[u_i], ytop),
                arrowprops=dict(arrowstyle="->", color="#2a7", lw=2))
    ax.text(xm, ytop + 4, f"+{gain:.1f}%", ha="center", color="#1a6", fontsize=11, fontweight="bold")
ax.set_xticks(ticks); ax.set_xticklabels(tick_lab, fontsize=10, fontweight="bold")
ax.set_ylabel("wall throughput (tok/s)", fontsize=11)
ax.set_ylim(0, max(vals) + 55)
ax.set_title("Llama-3.3-70B on 2×Blackwell(96GB) + 2×Ada(48GB), balanced (n_req=16)\n"
             "non-uniform partitioning helps both TP and PP", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
hu = plt.Rectangle((0, 0), 1, 1, color=UNI); hn = plt.Rectangle((0, 0), 1, 1, color=NON)
ax.legend([hu, hn], ["uniform", "non-uniform (Blackwell-biased)"], fontsize=9, loc="upper right")
fig.subplots_adjust(bottom=0.18)
out = REPO / "figures" / "fig_2x2_70b_balanced.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
print("tps:", {k: round(tps(k), 1) for k in rows})
