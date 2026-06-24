"""Per-model figure: 3 stacked workload rows × all configs (TP8/TP4PP2/TP2PP4/TP1PP8).

Labels follow user format:
  TP8 PP1: 'ffn(B:A)' or 'head(B:A)' or 'ffn(B:A) head(B:A)' — per-rank Blackwell:Ada
  TP4 PP2: '(stage0:stage1)' — layer split per stage
  TP2 PP4: '(s0:s1:s2:s3)'
  TP1 PP8: '(N×4 : M×4)' compact or full ':' list

PP cells use the OVERLAP regime when available; for 70B, cells without overlap data are skipped.

Outputs:
  figures/fig_<model>_workload_rows.png   per model (8b, 70b, opt30b, qwen32b)
  figures/fig_4model_workload_rows.png    integrated 12-row plot (4 models × 3 wl)
  figures/fig_70b_stock_vs_overlap.png    PP stock vs overlap for 70B
"""
from __future__ import annotations
import glob, json, os, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "figures"

MODELS = ["8b", "70b", "opt30b", "qwen32b"]
MODEL_TITLE = {
    "8b": "Llama-3.1-8B",
    "70b": "Llama-3.3-70B",
    "opt30b": "OPT-30B (n_req=64)",
    "qwen32b": "Qwen3-32B",
}

TOPOLOGIES = [
    ("TP8 PP1", "TP8PP1", "#4c72b0"),
    ("TP4 PP2", "TP4PP2", "#dd8452"),
    ("TP2 PP4", "TP2PP4", "#55a868"),
    ("TP1 PP8", "TP1PP8", "#c44e52"),
]
WLS = ["balanced", "decode_heavy", "prefill_heavy"]


def tp_split_label(label, ffn, head, kv):
    """Per-rank Blackwell:Ada for TP8 cells (TP-only configs)."""
    ffn_b, ffn_a = ffn[0], ffn[4]
    h_b, h_a = head[0], head[4]
    kv_b, kv_a = kv[0], kv[4]
    parts = []
    if ffn_b != ffn_a:
        parts.append(f"ffn({ffn_b}:{ffn_a})")
    if h_b != h_a:
        parts.append(f"head({h_b}:{h_a})")
    if not parts:
        return f"uniform ffn({ffn_b}:{ffn_a})"
    return " ".join(parts)


def pp_split_label(label, layer_split):
    """Layer split per PP stage."""
    if "uniform" in label:
        return f"uniform ({':'.join(str(l) for l in layer_split)})"
    # Compact form if Blackwell-only-bias (4 ranks same | 4 ranks same)
    if len(layer_split) == 8 and all(l == layer_split[0] for l in layer_split[:4]) and all(l == layer_split[4] for l in layer_split[4:]) and layer_split[0] != layer_split[4]:
        return f"({layer_split[0]}×4 : {layer_split[4]}×4)"
    return f"({':'.join(str(l) for l in layer_split)})"


def make_label(record):
    L = record["label"]
    ls = record["layer_split"]
    ffn = record["ffn_splits"]
    head = record["head_splits"]
    kv = record["kv_splits"]
    if L.startswith("TP8PP1"):
        return tp_split_label(L, ffn, head, kv)
    return pp_split_label(L, ls)


def collect_records(model: str, prefer_regime: str = "overlap") -> dict:
    """Return {(label, workload): record} preferring the requested regime.

    Strategy:
      - For TP8PP1 cells: take latest measurement
      - For PP cells (TP4PP2/TP2PP4/TP1PP8): prefer regime by directory mtime —
        the later sweep directories used the overlap path (after launcher-validated env);
        earlier ones (122900 for 70B) used stock.
    """
    pat = f"/data/esca/uckim/vllm_main/results/hetero_4x4_{model}_full_*"
    dirs = sorted(glob.glob(pat), key=os.path.getmtime)
    # First pass: collect all
    all_recs = []
    for d in dirs:
        for rj in sorted(glob.glob(os.path.join(d, f"{model}_*/record.json"))):
            try:
                r = json.load(open(rj))
                if not r.get("success"): continue
                r["_dir"] = d
                r["_dir_mtime"] = os.path.getmtime(d)
                all_recs.append(r)
            except Exception as e:
                print(f"skip {rj}: {e}", file=sys.stderr)
    # Latest-wins per (label, workload). For PP cells, we want overlap regime which is
    # the later directory. For 70B specifically: stock is in 122900, overlap is in 170850
    # and 185112. Since dirs are sorted by mtime ascending, last-wins gives overlap.
    by_key = {}
    for r in all_recs:
        by_key[(r["label"], r["workload"])] = r
    return by_key


def workload_order_within_topology(model: str, topo_prefix: str, all_records) -> list[str]:
    """Return ordered list of labels for the section, uniform first."""
    labels = sorted({k[0] for k in all_records if k[0].startswith(topo_prefix)})
    # Put uniform first
    uniform = [L for L in labels if "uniform" in L]
    others = [L for L in labels if "uniform" not in L]
    # For ordering others, use the bias level if encoded in label
    def sort_key(L):
        if "ffn_bias" in L:
            try: return (0, int(L.split("+")[-1]))
            except: return (0, 0)
        if "head_bias" in L:
            try: return (1, int(L.split("+")[-1]))
            except: return (1, 0)
        if "hybrid" in L:
            return (2, 0)
        if "skew" in L:
            try: return (3, int(L.split("+")[-1].split("_")[0]))
            except: return (3, 0)
        if "blackbias" in L:
            # extract first number as proxy
            try:
                bits = L.split("_")[-1].split("-")
                return (4, int(bits[0]))
            except: return (4, 0)
        return (5, 0)
    others.sort(key=sort_key)
    return uniform + others


def plot_one_model(model: str, ax_row=None, save: bool = True, suptitle_extra: str = ""):
    """Generate 3-row workload plot for one model. Returns the figure."""
    recs = collect_records(model)
    if not recs:
        print(f"no records for {model}", file=sys.stderr)
        return None

    # Get TP8PP1 uniform baseline per workload
    baselines = {}
    for wl in WLS:
        r = recs.get(("TP8PP1_uniform", wl))
        baselines[wl] = r["tps"] if r else 0

    # For each workload, figure out section layout (configs ordered by topology)
    # Same x positions across all workload rows for visual alignment
    sections = []  # list of (topo_title, topo_color, [labels])
    for topo_title, topo_prefix, topo_color in TOPOLOGIES:
        labels = workload_order_within_topology(model, topo_prefix, recs)
        if labels:
            sections.append((topo_title, topo_color, topo_prefix, labels))

    # Total positions
    all_positions = []
    for topo_title, topo_color, topo_prefix, labels in sections:
        for L in labels:
            all_positions.append((topo_title, topo_color, topo_prefix, L))

    # Resolve short labels using a sample record per (label) — take balanced if present, else any
    short_labels = []
    for topo_title, topo_color, topo_prefix, L in all_positions:
        sample = None
        for wl in WLS:
            if (L, wl) in recs:
                sample = recs[(L, wl)]; break
        if sample is None:
            short_labels.append(L)
        else:
            short_labels.append(make_label(sample))

    nrows = 3
    fig, axes = plt.subplots(nrows, 1, figsize=(max(18, len(all_positions)*0.7), 4*nrows))
    if nrows == 1:
        axes = [axes]

    for ai, wl in enumerate(WLS):
        ax = axes[ai]
        ys = []
        for topo_title, topo_color, topo_prefix, L in all_positions:
            r = recs.get((L, wl))
            ys.append(r["tps"] if r else 0)

        n = len(ys)
        x = np.arange(n)
        # Best position (highest TPS)
        best_idx = int(np.argmax(ys)) if any(ys) else -1
        baseline = baselines[wl]

        bars = []
        for i, (yi, (topo_title, topo_color, topo_prefix, L)) in enumerate(zip(ys, all_positions)):
            is_uniform = "uniform" in L
            edge_color = "#cc9900" if i == best_idx else "white"
            edge_lw = 3 if i == best_idx else 0.5
            bar = ax.bar(x[i], yi, 0.85, color=topo_color, alpha=0.55 if is_uniform else 0.85,
                          hatch="////" if is_uniform else None,
                          edgecolor=edge_color, linewidth=edge_lw, zorder=2)
            bars.append(bar)
            # value label inside (white text)
            if yi > 0:
                ax.text(x[i], yi*0.5, f"{yi:.0f}",
                        ha="center", va="center", fontsize=7, color="white", rotation=90)
            # delta % above
            if yi > 0 and baseline > 0:
                delta = (yi - baseline) / baseline * 100
                color = "#1c7f3a" if delta >= 0 else "#b8341c"
                sign = "+" if delta >= 0 else ""
                ax.text(x[i], yi + max(ys)*0.015, f"{sign}{delta:.0f}%",
                        ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")
            # ★ best
            if i == best_idx and yi > 0:
                ax.text(x[i], yi + max(ys)*0.07,
                        "★ best", ha="center", va="bottom", fontsize=9,
                        color="#cc9900", fontweight="bold")

        # Section labels above + dividers
        if any(ys):
            ymax = max(ys) * 1.15
        else:
            ymax = 100
        ax.set_ylim(0, ymax)
        col_idx = 0
        for topo_title, topo_color, topo_prefix, labels in sections:
            sec_start = col_idx
            sec_end = col_idx + len(labels)
            sec_mid = (sec_start + sec_end - 1) / 2
            ax.text(sec_mid, ymax * 0.97, topo_title,
                    ha="center", va="top", fontsize=11, color=topo_color, fontweight="bold")
            # divider line at the right edge of this section (except last)
            if sec_end < len(all_positions):
                ax.axvline(sec_end - 0.5, color="gray", linestyle="-", linewidth=0.4, alpha=0.5)
            col_idx = sec_end

        # X labels
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=7.5)
        # Baseline dashed line
        if baseline > 0:
            ax.axhline(baseline, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.text(len(all_positions) - 0.5, baseline, f" baseline {baseline:.0f}",
                    ha="right", va="bottom", fontsize=8, color="gray")
        ax.set_ylabel("throughput (tok/s)", fontsize=10)
        ax.set_title(f"{wl}    (baseline = TP8PP1 uniform = {baseline:.0f} tok/s; "
                     f"hatched = each topology's uniform)",
                     fontsize=11, loc="left")
        ax.grid(axis="y", linestyle=":", alpha=0.3, zorder=1)

    fig.suptitle(
        f"{MODEL_TITLE[model]}{suptitle_extra} 4+4 cross-node (4 Blackwell + 4 Ada) — config sweep per workload\n"
        f"left-to-right: uniform baseline then all TP/PP × split variants; PP cells use M13 overlap; "
        f"labels = % vs TP8PP1-uniform baseline; ★ = best per workload",
        fontsize=12, y=0.995)
    plt.tight_layout()

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"fig_{model}_workload_rows.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out} ({out.stat().st_size} bytes)")
        return out
    return fig


def plot_4model_integrated():
    """12-row plot: 4 models × 3 workloads."""
    figs_to_combine = []
    # Compute total config width per model
    model_data = {}
    max_n = 0
    for m in MODELS:
        recs = collect_records(m)
        if not recs: continue
        sections = []
        for topo_title, topo_prefix, topo_color in TOPOLOGIES:
            labels = workload_order_within_topology(m, topo_prefix, recs)
            if labels:
                sections.append((topo_title, topo_color, topo_prefix, labels))
        all_positions = []
        for topo_title, topo_color, topo_prefix, labels in sections:
            for L in labels:
                all_positions.append((topo_title, topo_color, topo_prefix, L))
        model_data[m] = (recs, sections, all_positions)
        max_n = max(max_n, len(all_positions))

    # Build a 12-row figure
    nrows = len(model_data) * 3
    fig, axes = plt.subplots(nrows, 1, figsize=(max(20, max_n * 0.6), 3.2 * nrows))

    row = 0
    for m in MODELS:
        if m not in model_data: continue
        recs, sections, all_positions = model_data[m]
        baselines = {wl: (recs.get(("TP8PP1_uniform", wl), {}).get("tps", 0) or 0) for wl in WLS}
        short_labels = []
        for topo_title, topo_color, topo_prefix, L in all_positions:
            sample = next((recs[(L, wl)] for wl in WLS if (L, wl) in recs), None)
            short_labels.append(make_label(sample) if sample else L)

        for wl in WLS:
            ax = axes[row]
            ys = [recs.get((L, wl), {}).get("tps", 0) or 0 for _, _, _, L in all_positions]
            n = len(ys); x = np.arange(n)
            best_idx = int(np.argmax(ys)) if any(ys) else -1
            baseline = baselines[wl]
            for i, yi in enumerate(ys):
                _, topo_color, _, L = all_positions[i]
                is_uniform = "uniform" in L
                edge_color = "#cc9900" if i == best_idx else "white"
                edge_lw = 2 if i == best_idx else 0.4
                ax.bar(x[i], yi, 0.85, color=topo_color, alpha=0.55 if is_uniform else 0.85,
                       hatch="////" if is_uniform else None,
                       edgecolor=edge_color, linewidth=edge_lw, zorder=2)
                if yi > 0:
                    ax.text(x[i], yi*0.5, f"{yi:.0f}", ha="center", va="center",
                            fontsize=6, color="white", rotation=90)
                if yi > 0 and baseline > 0:
                    delta = (yi - baseline) / baseline * 100
                    color = "#1c7f3a" if delta >= 0 else "#b8341c"
                    sign = "+" if delta >= 0 else ""
                    ax.text(x[i], yi + max(ys)*0.015 if any(ys) else 0,
                            f"{sign}{delta:.0f}%", ha="center", va="bottom",
                            fontsize=7, color=color, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(short_labels if row % 3 == 2 else [""]*n,
                                rotation=35, ha="right", fontsize=7)
            if baseline > 0:
                ax.axhline(baseline, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
            if any(ys):
                ax.set_ylim(0, max(ys) * 1.18)
            # section dividers
            ymax = ax.get_ylim()[1]
            col_idx = 0
            for topo_title, topo_color, topo_prefix, labels in sections:
                sec_start = col_idx
                sec_end = col_idx + len(labels)
                sec_mid = (sec_start + sec_end - 1) / 2
                ax.text(sec_mid, ymax * 0.95, topo_title,
                        ha="center", va="top", fontsize=9, color=topo_color, fontweight="bold")
                if sec_end < len(all_positions):
                    ax.axvline(sec_end - 0.5, color="gray", linestyle="-", linewidth=0.3, alpha=0.5)
                col_idx = sec_end
            ax.set_ylabel(f"{MODEL_TITLE[m]}\n{wl}\nTPS", fontsize=8, fontweight="bold" if row % 3 == 0 else "normal")
            ax.set_title(f"baseline TP8PP1 uniform = {baseline:.0f} tok/s",
                         fontsize=8, loc="right")
            ax.grid(axis="y", linestyle=":", alpha=0.3, zorder=1)
            row += 1

    fig.suptitle("4-model integrated workload sweep — 4+4 cross-node (Blackwell+Ada); "
                 "PP cells use M13 overlap; labels = % vs TP8PP1 uniform baseline",
                 fontsize=13, y=0.998)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig_4model_workload_rows.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} ({out.stat().st_size} bytes)")


def plot_70b_stock_vs_overlap():
    """70B PP cells stock vs overlap."""
    recs_stock_dir = "/data/esca/uckim/vllm_main/results/hetero_4x4_70b_full_20260609_122900"
    recs_overlap_dirs = [
        "/data/esca/uckim/vllm_main/results/hetero_4x4_70b_full_20260609_170850",
        "/data/esca/uckim/vllm_main/results/hetero_4x4_70b_full_20260609_185112",
    ]

    def load_dir(d):
        out = {}
        for rj in sorted(glob.glob(os.path.join(d, "70b_*/record.json"))):
            try:
                r = json.load(open(rj))
                if r.get("success"): out[(r["label"], r["workload"])] = r
            except: pass
        return out

    stock = load_dir(recs_stock_dir)
    overlap = {}
    for d in recs_overlap_dirs:
        overlap.update(load_dir(d))

    # PP-only labels with both stock and overlap
    pp_prefixes = ["TP4PP2", "TP2PP4"]
    paired_labels = sorted({L for (L, _) in stock if any(L.startswith(p) for p in pp_prefixes)} &
                            {L for (L, _) in overlap if any(L.startswith(p) for p in pp_prefixes)})

    if not paired_labels:
        print("no paired stock/overlap PP cells for 70B", file=sys.stderr)
        return

    # build short labels
    def label_short(L):
        sample = stock.get((L, "balanced")) or stock.get((L, "decode_heavy")) or stock.get((L, "prefill_heavy"))
        return pp_split_label(L, sample["layer_split"]) if sample else L

    short_labels = [label_short(L) for L in paired_labels]

    fig, axes = plt.subplots(3, 1, figsize=(max(14, len(paired_labels)*1.0), 12))
    for ai, wl in enumerate(WLS):
        ax = axes[ai]
        ys_stock = [(stock.get((L, wl), {}).get("tps", 0) or 0) for L in paired_labels]
        ys_overlap = [(overlap.get((L, wl), {}).get("tps", 0) or 0) for L in paired_labels]
        x = np.arange(len(paired_labels))
        w = 0.4
        ax.bar(x - w/2, ys_stock, w, label="stock PP", color="#7f7f7f", alpha=0.85)
        ax.bar(x + w/2, ys_overlap, w, label="M13 overlap PP", color="#1f77b4", alpha=0.85)
        for i, (s, o) in enumerate(zip(ys_stock, ys_overlap)):
            if s > 0: ax.text(x[i]-w/2, s+max(max(ys_stock), max(ys_overlap))*0.005,
                              f"{s:.0f}", ha="center", va="bottom", fontsize=8)
            if o > 0: ax.text(x[i]+w/2, o+max(max(ys_stock), max(ys_overlap))*0.005,
                              f"{o:.0f}", ha="center", va="bottom", fontsize=8)
            if s > 0 and o > 0:
                d = (o - s)/s*100
                color = "#1c7f3a" if d >= 0 else "#b8341c"
                sign = "+" if d >= 0 else ""
                ax.text(x[i], max(s,o)*1.05, f"{sign}{d:.1f}%",
                        ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(short_labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("TPS (tok/s)")
        ax.set_title(f"{wl}", fontsize=11, fontweight="bold", loc="left")
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Llama-3.3-70B PP cells — stock vs M13 overlap (n_req=128, 4+4 cross-node)",
                 fontsize=13, y=0.995)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "fig_70b_stock_vs_overlap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["8b","70b","opt30b","qwen32b","4model","stock_vs_overlap","all"], default="all")
    args = ap.parse_args()

    if args.only in ("8b", "all"):  plot_one_model("8b")
    if args.only in ("70b", "all"): plot_one_model("70b")
    if args.only in ("opt30b", "all"): plot_one_model("opt30b")
    if args.only in ("qwen32b", "all"): plot_one_model("qwen32b")
    if args.only in ("4model", "all"): plot_4model_integrated()
    if args.only in ("stock_vs_overlap", "all"): plot_70b_stock_vs_overlap()
