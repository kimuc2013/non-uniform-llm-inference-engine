"""Paper figures for the hetero-planner work (2026-06-23 results):
  1. fig_crossover_concurrency.png  — TPS vs concurrency per model, TP→PP crossover
  2. fig_layout_gain.png            — FFN-bias vs PP-skew gain across 1+1/2+2/4+4
  3. fig_planner_validation.png     — predicted vs measured (all layouts) + regret-by-layout
Reads the measured concurrency/layout sweeps + the cost model (zero-refit).
Output: figures/.
"""
from __future__ import annotations
import csv, dataclasses, glob, json, sys
from pathlib import Path
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import perf_planner as P
REPO = HERE.parent
OUT = REPO / "figures"; OUT.mkdir(exist_ok=True)


def relayout(hw, hg, wg):
    bw, ada = hw.nodes[0][0], hw.nodes[-1][0]
    return dataclasses.replace(hw, nodes=((bw, hg), (ada, wg)))


def load_balanced(d):
    """balanced-workload success entries from a concurrency dir."""
    out = []
    for rj in glob.glob(str(d / "*/record.json")):
        try:
            recs = json.load(open(rj))
        except Exception:
            continue
        for e in (recs if isinstance(recs, list) else [recs]):
            if e.get("success") and e.get("tps", 0) > 0 and e.get("workload") == "balanced":
                out.append(e)
    return out


def best_conc_dir(model, hg, wg):
    """Concurrency dir (hetero_HxW_model_ts, no _full_) with the most balanced rows."""
    cands = [Path(d) for d in glob.glob(str(REPO / f"results/hetero_{hg}x{wg}_{model}_*"))
             if "_full_" not in d]
    cands = [(len(load_balanced(d)), d) for d in cands]
    cands = [(n, d) for n, d in cands if n > 0]
    return max(cands, key=lambda x: x[0])[1] if cands else None


# topology family of a label, and a short pretty name
def fam(label):
    return label.split("_")[0]


# ---------- Figure 1: concurrency crossover ----------
def fig_crossover():
    models = [("8b", "Llama-3.1-8B"), ("70b", "Llama-3.3-70B"), ("mistral123b", "Mistral-Large-123B")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (mk, title) in zip(axes, models):
        d = best_conc_dir(mk, 4, 4)
        if not d:
            ax.set_visible(False); continue
        cells = load_balanced(d)
        byfam = defaultdict(lambda: defaultdict(list))  # fam -> n -> [tps]
        for e in cells:
            byfam[fam(e["label"])][e["n_req"]].append(e["tps"])
        ns = sorted({e["n_req"] for e in cells})
        # champion (max over all configs) per n
        champ = {n: max(e["tps"] for e in cells if e["n_req"] == n) for n in ns}
        for f, color in [("TP8PP1", "#d62728"), ("TP4PP2", "#1f77b4"),
                         ("TP4PP1", "#d62728"), ("TP2PP2", "#1f77b4")]:
            if f not in byfam: continue
            ys = [max(byfam[f].get(n, [0])) for n in ns]   # best within family
            ax.plot(ns, ys, "-o", color=color, label=f"{f} (best)", linewidth=2)
        ax.plot(ns, [champ[n] for n in ns], "k--", alpha=0.4, label="champion")
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("concurrency (n_req)")
        ax.set_ylabel("throughput (tok/s)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Load-dependent topology crossover (4+4, balanced): TP-heavy at low load → PP at high load",
                 fontsize=13, y=1.02)
    plt.tight_layout(); p = OUT / "fig_crossover_concurrency.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); print("saved", p)


# ---------- Figure 2: layout-axis non-uniform gain ----------
def fig_layout_gain():
    layouts = [(1, 1), (2, 2), (4, 4)]
    ffn_gain, pp_gain, labels = [], [], []
    for hg, wg in layouts:
        d = best_conc_dir("8b", hg, wg)
        if not d:
            ffn_gain.append(0); pp_gain.append(0); labels.append(f"{hg}+{wg}\n(no data)"); continue
        cells = load_balanced(d)
        byn = defaultdict(list)
        for e in cells: byn[e["n_req"]].append(e)
        fg = pg = 0.0
        for n, es in byn.items():
            tp1 = [e for e in es if e["pp"] == 1]; tp1u = [e for e in tp1 if "uniform" in e["label"]]
            if tp1 and tp1u: fg = max(fg, max(e["tps"] for e in tp1)/tp1u[0]["tps"] - 1)
            ppc = [e for e in es if e["pp"] > 1]
            if ppc:
                bt = fam(max(ppc, key=lambda e: e["tps"])["label"])
                f2 = [e for e in ppc if fam(e["label"]) == bt]; f2u = [e for e in f2 if "uniform" in e["label"]]
                if f2u: pg = max(pg, max(e["tps"] for e in f2)/f2u[0]["tps"] - 1)
        ffn_gain.append(fg*100); pp_gain.append(pg*100)
        labels.append(f"{hg}+{wg}\nTP{hg+wg}")
    x = np.arange(len(layouts)); w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x - w/2, ffn_gain, w, label="FFN-bias gain (non-uniform TP)", color="#ff7f0e")
    ax.bar(x + w/2, pp_gain, w, label="PP-skew gain (non-uniform PP)", color="#2ca02c")
    for i, (a, b) in enumerate(zip(ffn_gain, pp_gain)):
        ax.text(i - w/2, a + 0.4, f"{a:.1f}%", ha="center", fontsize=9)
        ax.text(i + w/2, b + 0.4, f"{b:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("max non-uniform gain over uniform (%)")
    ax.set_title("Llama-8B: which non-uniform knob matters depends on the layout\n"
                 "FFN-bias grows as TP shrinks; PP-skew grows as GPUs increase", fontsize=11)
    ax.grid(axis="y", alpha=0.3); ax.legend()
    plt.tight_layout(); p = OUT / "fig_layout_gain.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); print("saved", p)


# ---------- Figure 3: planner validation ----------
def fig_validation():
    hw0 = P.load_hardware()
    pts = []   # (meas, pred, layout)
    reg_by_layout = {}
    # calibration (4+4, n<=100)
    for layout, rows in [("4+4", None)]:
        pass
    # gather from concurrency dirs across layouts
    for hg, wg, tag in [(4, 4, "4+4"), (2, 2, "2+2"), (1, 1, "1+1")]:
        hw = relayout(hw0, hg, wg)
        regrets = []
        for mk in ("8b", "70b", "mistral123b"):
            d = best_conc_dir(mk, hg, wg)
            if not d: continue
            m = P.MODELS[mk]; cells = load_balanced(d)
            byn = defaultdict(list)
            for e in cells: byn[e["n_req"]].append(e)
            for n, es in byn.items():
                rows = []
                for e in es:
                    cfg = P.Config(e["tp"], e["pp"], list(e["layer_split"]), list(e["ffn_splits"]),
                                   list(e["head_splits"]), list(e["kv_splits"]), e["label"])
                    pr = P.predict(m, hw, P.Workload(e["in_len"], e["out_len"], n), cfg, overlap=(e["pp"] > 1))
                    pts.append((e["tps"], pr.get("tps", 0), tag))
                    rows.append((e["label"], e["tps"], pr.get("tps", 0)))
                mc = max(rows, key=lambda r: r[1]); pc = max(rows, key=lambda r: r[2])
                pcm = next(t for l, t, _ in rows if l == pc[0])
                regrets.append((mc[1] - pcm) / mc[1] * 100)
        if regrets:
            reg_by_layout[tag] = float(np.mean(regrets))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {"4+4": "#1f77b4", "2+2": "#ff7f0e", "1+1": "#2ca02c"}
    for tag in ("4+4", "2+2", "1+1"):
        xs = [m for m, p, t in pts if t == tag]; ys = [p for m, p, t in pts if t == tag]
        ax1.scatter(xs, ys, s=22, alpha=0.7, color=colors[tag], label=f"{tag} (zero-refit)" if tag != "4+4" else "4+4 (fit)")
    lim = max(max(m for m, p, t in pts), max(p for m, p, t in pts)) * 1.05
    ax1.plot([0, lim], [0, lim], "k--", alpha=0.4)
    ax1.set_xlabel("measured TPS"); ax1.set_ylabel("predicted TPS")
    ax1.set_title("Predicted vs measured (balanced, all layouts)"); ax1.grid(alpha=0.3); ax1.legend()
    tags = list(reg_by_layout); vals = [reg_by_layout[t] for t in tags]
    ax2.bar(tags, vals, color=[colors[t] for t in tags])
    for i, v in enumerate(vals): ax2.text(i, v + 0.1, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("mean regret (%)"); ax2.set_title("Planner regret by layout (lower = better)")
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("Planner accuracy: calibrated on 4+4, generalizes zero-refit to 2+2 / 1+1",
                 fontsize=12, y=1.02)
    plt.tight_layout(); p = OUT / "fig_planner_validation.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); print("saved", p)


# ---------- Figure 4: self-validation vs baseline (held-out workload) ----------
def fig_selfval(workload="chat", in_len=768, out_len=256):
    hw = P.load_hardware()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mk, title in zip(axes, ["8b", "70b"], ["Llama-3.1-8B", "Llama-3.3-70B"]):
        m = P.MODELS[mk]
        cells = []
        for d in (Path(x) for x in glob.glob(str(REPO / f"results/hetero_4x4_{mk}_*")) if "_full_" not in x):
            for rj in glob.glob(str(d / "*/record.json")):
                recs = json.load(open(rj))
                for e in (recs if isinstance(recs, list) else [recs]):
                    if e.get("success") and e.get("workload") == workload and e.get("tps", 0) > 0:
                        cells.append(e)
        if not cells:
            ax.set_visible(False); continue
        byn = defaultdict(list)
        for e in cells: byn[e["n_req"]].append(e)
        ns = sorted(byn); pick = []; base = []; best = []
        for n in ns:
            es = byn[n]; w = P.Workload(in_len, out_len, n)
            best.append(max(e["tps"] for e in es))
            b = [e for e in es if e["label"] == "TP8PP1_uniform"]; base.append(b[0]["tps"] if b else 0)
            # RAW planner top-1 pick (no safety guard), mapped to its nearest measured config
            ranked = P.plan(m, hw, w, top_k=1)
            if not ranked:
                mt = base[-1]
            else:
                scfg = ranked[0][1]
                same = [e for e in es if e["tp"] == scfg.tp and e["pp"] == scfg.pp]
                mt = (min(same, key=lambda e: sum(abs(a - x) for a, x in zip(e["layer_split"], scfg.layer_split))
                          + abs(e["ffn_splits"][0] - scfg.ffn_splits[0]) / 1000)["tps"]
                      if same else base[-1])
            pick.append(mt)
        x = np.arange(len(ns)); w = 0.27
        ax.bar(x - w, base, w, label="baseline (TP8 uniform)", color="#999999")
        ax.bar(x, pick, w, label="planner pick", color="#1f77b4")
        ax.bar(x + w, best, w, label="measured best", color="#2ca02c", alpha=0.6)
        for i, (p, b) in enumerate(zip(pick, base)):
            d = (p / b - 1) * 100 if b else 0
            ax.text(i, max(p, b) + max(best) * 0.02, f"{d:+.0f}%", ha="center",
                    fontsize=9, fontweight="bold", color="green" if d >= 0 else "red")
        ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in ns]); ax.set_title(title, fontweight="bold")
        ax.set_ylabel("throughput (tok/s)"); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Self-validation on a HELD-OUT workload ({workload} in={in_len}/out={out_len}, "
                 "not in calibration): raw planner pick vs naive baseline\n"
                 "raw predicted-best config (no safety guard) — wins big at production load",
                 fontsize=10)
    plt.tight_layout(); p = OUT / "fig_selfval_vs_baseline.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig); print("saved", p)


if __name__ == "__main__":
    fig_crossover()
    fig_layout_gain()
    fig_validation()
    fig_selfval()
