"""Heterogeneous 4+4 cross-node (4 Blackwell + 4 Ada) TP/PP result plots.

Produces three deliverables from results/hetero_4x4_<model>_full_<ts>/<cell>/record.json:

  1. Per-model 2x2 topology figure  -> fig_<model>_4topology_nonuniform.png
       Each subplot = one topology (TP8PP1, TP4PP2, TP2PP4, TP1PP8).
       Paired bars: uniform baseline + non-uniform variants x 3 workloads.
       PP cells (pp>1) use the OVERLAP regime; %-diff vs uniform labelled on each bar.

  2. Cross-model comparison           -> fig_cross_model_tps_per_param.png
       1x3 facets (one per workload). x = topology, y = TPS / param(B) (normalised),
       one line per model. y uses the best (champion) config within each topology.

  3. Champion per (model, workload)   -> fig_champion_table.png + champion_per_model_workload.csv
       The max-TPS config for every (model, workload), with regime / topology / tok-s-per-B.

Regime (stock vs M13-overlap) is NOT in record.json. It is read from each cell's
vllm.log marker `... mb_enabled=True|False` (True => microbatch/overlap engine).
For pp==1 cells there is no PP, so regime is "n/a" (single regime).

Run:  python planner/plot_hetero_4x4.py            # all figures, all models
      python planner/plot_hetero_4x4.py --model 70b
"""
from __future__ import annotations
import argparse, csv, glob, json, os, re, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT_DIR = REPO / "figures"

# ----------------------------------------------------------------------------
# Model parameter counts (billions) for TPS normalisation.
#   llama70b = Llama-3.3-70B-Instruct, llama8b = Llama-3.1-8B-Instruct (confirmed from logs)
#   opt30b   = OPT-30B, qwen32b = Qwen2.5-32B (standard published sizes)
# ----------------------------------------------------------------------------
PARAMS_B = {"70b": 70.55, "8b": 8.03, "opt30b": 30.0, "qwen32b": 32.5}
MODEL_DISPLAY = {"70b": "Llama-3.3-70B", "8b": "Llama-3.1-8B",
                 "opt30b": "OPT-30B", "qwen32b": "Qwen2.5-32B"}

# Topology display order (TP-heavy -> PP-heavy)
TOPO_ORDER = ["TP8 PP1", "TP4 PP2", "TP2 PP4", "TP1 PP8"]

WLS = ["balanced", "decode_heavy", "prefill_heavy"]
WL_COLOR = {"balanced": "#4c72b0", "decode_heavy": "#dd8452", "prefill_heavy": "#55a868"}
# topology cluster colours for the per-workload config sweep (fig 1)
TOPO_COLOR = {"TP8 PP1": "#4c72b0", "TP4 PP2": "#dd8452",
              "TP2 PP4": "#55a868", "TP1 PP8": "#c44e52"}
BASELINE_LABEL = "TP8PP1_uniform"   # homogeneous all-TP, no-bias reference
MODEL_COLOR = {"70b": "#c44e52", "8b": "#4c72b0", "opt30b": "#8172b3", "qwen32b": "#937860"}
MODEL_MARKER = {"70b": "o", "8b": "s", "opt30b": "^", "qwen32b": "D"}

# Per-model topology map: (topo_name, uniform_label, uniform_short, [(variant_label, short)...]).
# Labels match record.json["label"] exactly. Only configs actually present are plotted.
MODEL_TOPOS = {
    "70b": [
        ("TP8 PP1", "TP8PP1_uniform", "uniform", [
            ("TP8PP1_ffn_bias+25", "FFN +25"),
            ("TP8PP1_ffn_bias+50", "FFN +50"),
            ("TP8PP1_ffn_bias+75", "FFN +75"),
        ]),
        ("TP4 PP2", "TP4PP2_layer_uniform_40-40", "uniform [40,40]", [
            ("TP4PP2_layer_skew+4_44-36", "skew +4"),
            ("TP4PP2_layer_skew+8_48-32", "skew +8"),
            ("TP4PP2_layer_skew+12_52-28", "skew +12"),
            ("TP4PP2_layer_skew+16_56-24", "skew +16"),
        ]),
        ("TP2 PP4", "TP2PP4_layer_uniform_20-20-20-20", "uniform [20]x4", [
            ("TP2PP4_layer_blackbias_22-22-18-18", "blackbias 22-18"),
            ("TP2PP4_layer_blackbias_24-24-16-16", "blackbias 24-16"),
        ]),
        ("TP1 PP8", "TP1PP8_layer_uniform_10x8", "uniform [10]x8", [
            ("TP1PP8_layer_blackbias_12-12-12-12-8-8-8-8", "blackbias [12-8]"),
            ("TP1PP8_layer_blackbias_14-14-14-14-6-6-6-6", "blackbias [14-6]"),
        ]),
    ],
    "8b": [
        ("TP8 PP1", "TP8PP1_uniform", "uniform", [
            ("TP8PP1_ffn_bias+25", "FFN +25"),
            ("TP8PP1_ffn_bias+50", "FFN +50"),
            ("TP8PP1_ffn_bias+75", "FFN +75"),
        ]),
        ("TP4 PP2", "TP4PP2_layer_uniform_16-16", "uniform [16,16]", [
            ("TP4PP2_layer_skew+2_18-14", "skew +2"),
            ("TP4PP2_layer_skew+4_20-12", "skew +4"),
            ("TP4PP2_layer_skew+6_22-10", "skew +6"),
            ("TP4PP2_layer_skew+8_24-8", "skew +8"),
        ]),
        ("TP2 PP4", "TP2PP4_layer_uniform_8-8-8-8", "uniform [8]x4", [
            ("TP2PP4_layer_blackbias_9-9-7-7", "blackbias 9-7"),
            ("TP2PP4_layer_blackbias_10-10-6-6", "blackbias 10-6"),
        ]),
        ("TP1 PP8", "TP1PP8_layer_uniform_4x8", "uniform [4]x8", [
            ("TP1PP8_layer_blackbias_5-5-5-5-3-3-3-3", "blackbias [5-3]"),
            ("TP1PP8_layer_blackbias_6-6-6-6-2-2-2-2", "blackbias [6-2]"),
        ]),
    ],
    "opt30b": [
        ("TP8 PP1", "TP8PP1_uniform", "uniform", [
            ("TP8PP1_ffn_bias+25", "FFN +25"),
            ("TP8PP1_ffn_bias+50", "FFN +50"),
            ("TP8PP1_ffn_bias+75", "FFN +75"),
        ]),
        ("TP4 PP2", "TP4PP2_layer_uniform_24-24", "uniform [24,24]", [
            ("TP4PP2_layer_skew+4_28-20", "skew +4"),
            ("TP4PP2_layer_skew+8_32-16", "skew +8"),
            ("TP4PP2_layer_skew+12_36-12", "skew +12"),
            ("TP4PP2_layer_skew+16_40-8", "skew +16"),
        ]),
        ("TP2 PP4", "TP2PP4_layer_uniform_12-12-12-12", "uniform [12]x4", [
            ("TP2PP4_layer_blackbias_14-14-10-10", "blackbias 14-10"),
            ("TP2PP4_layer_blackbias_16-16-8-8", "blackbias 16-8"),
        ]),
        ("TP1 PP8", "TP1PP8_layer_uniform_6x8", "uniform [6]x8", [
            ("TP1PP8_layer_blackbias_7-7-7-7-5-5-5-5", "blackbias [7-5]"),
            ("TP1PP8_layer_blackbias_8-8-8-8-4-4-4-4", "blackbias [8-4]"),
        ]),
    ],
    "qwen32b": [
        ("TP8 PP1", "TP8PP1_uniform", "uniform", [
            ("TP8PP1_ffn_bias+25", "FFN +25"),
            ("TP8PP1_ffn_bias+50", "FFN +50"),
            ("TP8PP1_ffn_bias+75", "FFN +75"),
        ]),
        ("TP4 PP2", "TP4PP2_layer_uniform_32-32", "uniform [32,32]", [
            ("TP4PP2_layer_skew+4_36-28", "skew +4"),
            ("TP4PP2_layer_skew+8_40-24", "skew +8"),
            ("TP4PP2_layer_skew+12_44-20", "skew +12"),
            ("TP4PP2_layer_skew+16_48-16", "skew +16"),
        ]),
        ("TP2 PP4", "TP2PP4_layer_uniform_16-16-16-16", "uniform [16]x4", [
            ("TP2PP4_layer_blackbias_18-18-14-14", "blackbias 18-14"),
            ("TP2PP4_layer_blackbias_20-20-12-12", "blackbias 20-12"),
        ]),
        ("TP1 PP8", "TP1PP8_layer_uniform_8x8", "uniform [8]x8", [
            ("TP1PP8_layer_blackbias_10-10-10-10-6-6-6-6", "blackbias [10-6]"),
            ("TP1PP8_layer_blackbias_12-12-12-12-4-4-4-4", "blackbias [12-4]"),
        ]),
    ],
}

_MB_RE = re.compile(r"mb_enabled=(True|False)")


def _regime(celldir: str, pp: int) -> str:
    """overlap | stock | n/a | unknown  -- read from vllm.log mb_enabled marker."""
    if pp == 1:
        return "n/a"
    log = os.path.join(celldir, "vllm.log")
    if os.path.exists(log):
        try:
            m = _MB_RE.search(open(log, errors="ignore").read())
            if m:
                return "overlap" if m.group(1) == "True" else "stock"
        except Exception:
            pass
    return "unknown"


def collect(model: str) -> dict:
    """Return {(label, workload): {regime: {tps, pp, mtime}}} with latest-wins per regime.

    Files are read in mtime order so a later successful run overwrites an earlier one.
    """
    pat = f"{RESULTS}/hetero_4x4_{model}_full_*"
    dirs = sorted(glob.glob(pat), key=os.path.getmtime)
    data: dict = defaultdict(dict)
    for d in dirs:
        for cd in sorted(glob.glob(os.path.join(d, f"{model}_*"))):
            rj = os.path.join(cd, "record.json")
            if not os.path.isdir(cd) or not os.path.exists(rj):
                continue
            try:
                r = json.load(open(rj))
            except Exception as e:
                print(f"  skip {rj}: {e}", file=sys.stderr)
                continue
            if not r.get("success"):
                continue
            pp = int(r.get("pp", 0))
            reg = _regime(cd, pp)
            key = (r.get("label", ""), r.get("workload", ""))
            mt = os.path.getmtime(rj)
            prev = data[key].get(reg)
            if prev is None or mt >= prev["mtime"]:
                data[key][reg] = {"tps": float(r.get("tps", 0.0)), "pp": pp, "mtime": mt}
    return data


def preferred_tps(cell: dict):
    """Pick the TPS for plotting: overlap for PP cells, else the single regime.

    cell = {regime: {tps, pp, ...}}. Returns (tps, regime_used) or (None, None).
    """
    if not cell:
        return None, None
    if "overlap" in cell:
        return cell["overlap"]["tps"], "overlap"
    if "n/a" in cell:
        return cell["n/a"]["tps"], "n/a"
    if "stock" in cell:                      # PP cell with only stock measured
        return cell["stock"]["tps"], "stock"
    # unknown regime fallback
    reg = next(iter(cell))
    return cell[reg]["tps"], reg


# ============================================================================
# Figure 1: per-model, ONE wide panel per workload.
#   Left = uniform TP8PP1 baseline; to its right every TP/PP x split config laid
#   out in one long row, grouped & coloured by topology, so the best config for
#   each workload is obvious at a glance. Bars labelled with % vs baseline.
# ============================================================================
def ordered_configs(model: str) -> list:
    """Flat config list in display order: per topology -> uniform first, then variants.
    Each item = (full_label, short_label, topo_name, is_uniform)."""
    items = []
    for topo_name, uni_label, uni_short, variants in MODEL_TOPOS[model]:
        items.append((uni_label, uni_short, topo_name, True))
        for vl, vs in variants:
            items.append((vl, vs, topo_name, False))
    return items


def fig_per_model(model: str, data: dict) -> Path | None:
    cfgs = ordered_configs(model)
    # keep configs present in at least one workload (consistent x across all rows)
    present = [c for c in cfgs
               if any(preferred_tps(data.get((c[0], wl), {}))[0] for wl in WLS)]
    if not present:
        print(f"  [{model}] no plottable data for figure 1 — skipped")
        return None

    shorts = [c[1] for c in present]
    topos = [c[2] for c in present]
    bar_cols = [TOPO_COLOR[t] for t in topos]
    n = len(present)
    x = np.arange(n)
    # topology group boundaries (for separators + group labels)
    groups = []  # (topo_name, start, end)
    s = 0
    for i in range(1, n + 1):
        if i == n or topos[i] != topos[s]:
            groups.append((topos[s], s, i - 1)); s = i

    width = max(11.0, 0.62 * n + 3.0)
    fig, axes = plt.subplots(len(WLS), 1, figsize=(width, 4.4 * len(WLS)))
    if len(WLS) == 1:
        axes = [axes]

    for ax, wl in zip(axes, WLS):
        ys = [preferred_tps(data.get((c[0], wl), {}))[0] or 0.0 for c in present]
        base, _ = preferred_tps(data.get((BASELINE_LABEL, wl), {}))
        if not base:                       # fall back to first present config
            base = next((v for v in ys if v > 0), 0.0)
        ymax = max(ys) if ys else 1.0
        champ = int(np.argmax(ys)) if ymax > 0 else -1

        bars = ax.bar(x, ys, 0.74, color=bar_cols, alpha=0.9,
                      edgecolor="white", linewidth=0.6)
        # uniform-of-each-topology bars: hatched so they read as that topology's baseline
        for i, c in enumerate(present):
            if c[3]:
                bars[i].set_hatch("////")
                bars[i].set_edgecolor("0.25")
        # champion highlight
        if champ >= 0 and ys[champ] > 0:
            bars[champ].set_edgecolor("#d4af37"); bars[champ].set_linewidth(3)
            ax.text(champ, ys[champ] + ymax * 0.085, "★ best", ha="center",
                    va="bottom", fontsize=9, color="#b8860b", fontweight="bold")

        # global baseline line (TP8PP1 uniform)
        if base > 0:
            ax.axhline(base, color="0.30", linestyle="--", linewidth=1.1, alpha=0.8)
            ax.text(n - 0.4, base, f" baseline {base:.0f}", va="bottom", ha="right",
                    fontsize=8, color="0.30", fontweight="bold")

        # per-bar labels: %Δ vs baseline (bold, coloured) + abs tps (small)
        for i, (bar, y) in enumerate(zip(bars, ys)):
            if y <= 0:
                continue
            if base > 0:
                pct = (y - base) / base * 100.0
                col = "#1a7d1a" if pct >= 0 else "#b22222"
                ax.text(i, y + ymax * 0.012, f"{pct:+.0f}%", ha="center", va="bottom",
                        fontsize=7.5, color=col, fontweight="bold")
            ax.text(i, y * 0.5, f"{y:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white", rotation=90)

        # topology group separators + labels
        for gi, (tname, gs, ge) in enumerate(groups):
            if gi > 0:
                ax.axvline(gs - 0.5, color="0.8", linewidth=1.0, linestyle="-")
            ax.text((gs + ge) / 2.0, ymax * 1.16, tname, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=TOPO_COLOR[tname])

        ax.set_xticks(x)
        ax.set_xticklabels(shorts, rotation=35, ha="right", fontsize=8)
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_ylim(0, ymax * 1.26)
        ax.set_ylabel("throughput (tok/s)")
        ax.set_title(f"{wl}     (baseline = TP8PP1 uniform = {base:.0f} tok/s; "
                     f"hatched bar = each topology's uniform)",
                     fontsize=11, fontweight="bold", loc="left")
        ax.grid(axis="y", linestyle=":", alpha=0.3)

    fig.suptitle(
        f"{MODEL_DISPLAY[model]} 4+4 cross-node (4 Blackwell + 4 Ada) — config sweep per workload\n"
        f"left-to-right: uniform baseline then all TP/PP x split variants; PP cells use M13 overlap; "
        f"labels = % vs TP8PP1-uniform baseline; ★ = best per workload",
        fontsize=12.5, y=1.0)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT_DIR / f"fig_{model}_config_sweep.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name} ({out.stat().st_size} bytes)")
    return out


# ============================================================================
# Figure 2: cross-model TPS/param vs topology, faceted by workload
# ============================================================================
def topo_champion_tps(model: str, data: dict, topo_name: str, wl: str):
    """Best TPS over all configs of a topology for one workload. (tps, label) or (None, None)."""
    topos = {t[0]: t for t in MODEL_TOPOS[model]}
    if topo_name not in topos:
        return None, None
    _, uni_label, _, variants = topos[topo_name]
    labels = [uni_label] + [v[0] for v in variants]
    best_t, best_l = None, None
    for lab in labels:
        t, _ = preferred_tps(data.get((lab, wl), {}))
        if t and (best_t is None or t > best_t):
            best_t, best_l = t, lab
    return best_t, best_l


def fig_cross_model(all_data: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=False)
    for ax, wl in zip(axes, WLS):
        for model in ["70b", "8b", "opt30b", "qwen32b"]:
            data = all_data.get(model, {})
            xs, ys = [], []
            for ti, topo in enumerate(TOPO_ORDER):
                t, _ = topo_champion_tps(model, data, topo, wl)
                if t:
                    xs.append(ti)
                    ys.append(t / PARAMS_B[model])
            if not xs:
                continue
            style = dict(color=MODEL_COLOR[model], marker=MODEL_MARKER[model],
                         markersize=8, linewidth=2, alpha=0.9)
            if len(xs) == 1:
                ax.scatter(xs, ys, s=90, color=MODEL_COLOR[model],
                           marker=MODEL_MARKER[model], label=f"{model} (TP8PP1 only)", zorder=3)
            else:
                ax.plot(xs, ys, label=model, **style)
        ax.set_xticks(range(len(TOPO_ORDER)))
        ax.set_xticklabels(TOPO_ORDER, fontsize=10)
        ax.set_title(wl, fontsize=12, fontweight="bold")
        ax.set_xlabel("topology")
        # log y: normalised TPS/param spans ~21 (70b/opt30b) to ~484 (8b); a linear
        # axis squashes the large-model curves flat. Log keeps every shape legible.
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.35)
        ax.legend(fontsize=9, loc="best")
    axes[0].set_ylabel("normalised throughput  (tok/s per billion params, log scale)")
    fig.suptitle("Cross-model: best-in-topology throughput per parameter  (log y)\n"
                 "PP cells = overlap regime;  opt30b n_req=64/maxlen=2048 (others 128/4096);  "
                 "opt30b = TP8PP1 only, qwen32b = no usable data",
                 fontsize=12, y=1.04)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "fig_cross_model_tps_per_param.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name} ({out.stat().st_size} bytes)")
    return out


# ============================================================================
# Figure 3 + CSV: champion per (model, workload)
# ============================================================================
def topo_of_label(model: str, label: str) -> str:
    for topo_name, uni_label, _, variants in MODEL_TOPOS[model]:
        if label == uni_label or label in (v[0] for v in variants):
            return topo_name
    return "?"


def build_champions(all_data: dict) -> list:
    """For each (model, workload): the max-TPS config. Returns list of row dicts."""
    rows = []
    for model in ["70b", "8b", "opt30b", "qwen32b"]:
        data = all_data.get(model, {})
        for wl in WLS:
            best = None
            for (label, w), cell in data.items():
                if w != wl:
                    continue
                t, rg = preferred_tps(cell)
                if t and (best is None or t > best["tps"]):
                    best = {"tps": t, "label": label, "regime": rg}
            if best is None:
                rows.append({"model": model, "workload": wl, "label": "(no data)",
                             "topology": "-", "regime": "-", "tps": None,
                             "tps_per_B": None, "delta_vs_tp8_uniform": None})
                continue
            # reference: TP8PP1 uniform of same model+workload (homogeneous all-TP baseline)
            ref, _ = preferred_tps(data.get(("TP8PP1_uniform", wl), {}))
            delta = (best["tps"] - ref) / ref * 100.0 if ref else None
            rows.append({
                "model": model, "workload": wl, "label": best["label"],
                "topology": topo_of_label(model, best["label"]), "regime": best["regime"],
                "tps": best["tps"], "tps_per_B": best["tps"] / PARAMS_B[model],
                "delta_vs_tp8_uniform": delta,
            })
    return rows


def fig_champion_table(rows: list) -> tuple:
    # CSV
    csv_path = OUT_DIR / "champion_per_model_workload.csv"
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["model", "workload", "label", "topology",
                                             "regime", "tps", "tps_per_B",
                                             "delta_vs_tp8_uniform"])
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow(r)

    # Rendered table figure
    headers = ["Model", "Workload", "Champion config", "Topology", "Regime",
               "TPS", "tok/s / B", "Δ% vs TP8PP1 uniform"]
    cells, cell_colors = [], []
    for r in rows:
        tps = f"{r['tps']:.0f}" if r["tps"] is not None else "—"
        tpb = f"{r['tps_per_B']:.1f}" if r["tps_per_B"] is not None else "—"
        if r["delta_vs_tp8_uniform"] is None:
            dlt, dcol = "—", "white"
        else:
            dlt = f"{r['delta_vs_tp8_uniform']:+.1f}%"
            dcol = "#d6f5d6" if r["delta_vs_tp8_uniform"] >= 0 else "#f7d6d6"
        cells.append([r["model"], r["workload"], r["label"], r["topology"],
                      r["regime"], tps, tpb, dlt])
        cell_colors.append(["white", "white", "white", "white", "white", "white", "white", dcol])

    fig, ax = plt.subplots(figsize=(16, 0.5 * len(rows) + 1.5))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=headers, cellColours=cell_colors,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for c in range(len(headers)):
        tbl[0, c].set_facecolor("#333333")
        tbl[0, c].set_text_props(color="white", fontweight="bold")
    ax.set_title("Champion config per (model, workload) — max throughput "
                 "(PP cells = overlap regime)\n"
                 "Δ%: champion vs same-model/workload TP8PP1 uniform (homogeneous all-TP ref)",
                 fontsize=12, fontweight="bold", pad=14)
    out = OUT_DIR / "fig_champion_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name} ({out.stat().st_size} bytes) + {csv_path.name}")
    return out, csv_path


def print_markdown_table(rows: list):
    print("\n### Champion per (model, workload)\n")
    print("| Model | Workload | Champion config | Topology | Regime | TPS | tok/s/B | Δ% vs TP8 uniform |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        tps = f"{r['tps']:.0f}" if r["tps"] is not None else "—"
        tpb = f"{r['tps_per_B']:.1f}" if r["tps_per_B"] is not None else "—"
        dlt = f"{r['delta_vs_tp8_uniform']:+.1f}%" if r["delta_vs_tp8_uniform"] is not None else "—"
        print(f"| {r['model']} | {r['workload']} | `{r['label']}` | {r['topology']} | "
              f"{r['regime']} | {tps} | {tpb} | {dlt} |")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_TOPOS.keys()),
                    help="only render figure 1 for this model (default: all models + figs 2,3)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else ["70b", "8b", "opt30b", "qwen32b"]
    all_data = {}
    for m in models:
        all_data[m] = collect(m)
        n = sum(len(v) for v in all_data[m].values())
        print(f"[{m}] {len(all_data[m])} (label,wl) keys, {n} regime-cells")

    print("\nFigure 1 (per-model 2x2 topology):")
    for m in models:
        fig_per_model(m, all_data[m])

    if not args.model:
        print("\nFigure 2 (cross-model TPS/param):")
        fig_cross_model(all_data)

        print("\nFigure 3 (champion table):")
        rows = build_champions(all_data)
        fig_champion_table(rows)
        print_markdown_table(rows)

    print(f"\nAll outputs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
