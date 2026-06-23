#!/usr/bin/env python
"""Build the calibration dataset for the hetero TP/PP performance planner.

Walks every sweep result directory matching
    /data/esca/uckim/vllm_main/results/hetero_4x4_{model}_full_{YYYYMMDD_HHMMSS}/
and emits one CSV row per successful record.json cell, deduplicated
latest-wins per (model, label, workload, regime).

Regime rules
------------
- pp == 1 cells have no PP regime -> "tp_only" (so latest-wins dedup
  collapses stale early single-cell reruns of the same TP-only cell).
- 70b PP cells: dir ..._122900 = "stock"; all other 70b dirs = "overlap"
  (170850 / 185112 are the canonical overlap sweeps; the few mid-day
  single-cell dirs are overlap debug runs and get superseded by
  latest-wins anyway).
- All other models: every PP cell is "overlap".

Re-run after a new model (e.g. mistral123b) lands: add its spec to
MODEL_SPECS and rerun. Unknown models are still emitted (with empty
spec columns) and a warning is printed so they are not silently lost.

Usage:
    /data/esca/uckim/miniconda3/envs/vllm_main/bin/python \
        /data/esca/uckim/vllm_main/planner/build_calibration.py
"""

import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]      # repo-relative, not hardcoded
# Two dir shapes: the original full-grid sweeps (..._full_<ts>, one n_req per
# cell, record.json = dict) and the concurrency sweeps (..._<ts>, multiple
# n_req per cell, record.json = list of per-n_req entries). Both feed the fit;
# n_req is part of the dedup key so the batch axis is preserved.
RESULTS_GLOB = str(_REPO / "results" / "hetero_4x4_*")
OUT_CSV = str(_REPO / "planner" / "calibration_data.csv")

# 70b dirs whose PP cells were run with stock vLLM PP (no overlap fork).
STOCK_PP_DIRS_70B = {"hetero_4x4_70b_full_20260609_122900"}

# Per-model architecture constants:
# n_layers, hidden, ffn_dim, n_q, n_kv, head_dim, params_b, vocab
MODEL_SPECS = {
    "8b":      dict(n_layers=32, hidden=4096, ffn_dim=14336, n_q=32, n_kv=8,
                    head_dim=128, params_b=8,    vocab=128256),
    "70b":     dict(n_layers=80, hidden=8192, ffn_dim=28672, n_q=64, n_kv=8,
                    head_dim=128, params_b=70,   vocab=128256),
    "opt30b":  dict(n_layers=48, hidden=7168, ffn_dim=28672, n_q=56, n_kv=56,
                    head_dim=128, params_b=30,   vocab=50272),
    "qwen32b": dict(n_layers=64, hidden=5120, ffn_dim=25600, n_q=64, n_kv=8,
                    head_dim=128, params_b=32.8, vocab=151936),
    "mistral123b": dict(n_layers=88, hidden=12288, ffn_dim=28672, n_q=96, n_kv=8,
                    head_dim=128, params_b=123.0, vocab=32768),
}
SPEC_COLS = ["n_layers", "hidden", "ffn_dim", "n_q", "n_kv",
             "head_dim", "params_b", "vocab"]

DIR_RE = re.compile(r"hetero_4x4_(?P<model>.+)_full_(?P<ts>\d{8}_\d{6})$")
# Concurrency sweeps: no "_full_"; model token must not swallow the suffix.
CONC_DIR_RE = re.compile(r"hetero_4x4_(?P<model>[^/]+?)_(?P<ts>\d{8}_\d{6})$")

COLUMNS = [
    "model", "label", "tp", "pp", "layer_split", "ffn_splits",
    "head_splits", "kv_splits", "workload", "in_len", "out_len", "n_req",
    "tps", "ttft_ms", "itl_ms", "regime",
] + SPEC_COLS + ["source_dir"]


def regime_for(model: str, pp: int, dir_base: str) -> str:
    if pp <= 1:
        return "tp_only"
    if model == "70b" and dir_base in STOCK_PP_DIRS_70B:
        return "stock"
    return "overlap"


def main() -> int:
    sweep_dirs = sorted(d for d in glob.glob(RESULTS_GLOB) if os.path.isdir(d))
    if not sweep_dirs:
        print("ERROR: no sweep dirs matched", RESULTS_GLOB, file=sys.stderr)
        return 1

    # key -> (timestamp, row) ; latest timestamp wins
    best = {}
    n_records = n_success = 0
    unknown_models = set()

    # Additive accumulator: seed from the existing CSV so calibration rows whose
    # source sweep dir was archived/removed during cleanup are NOT lost on a
    # rebuild. Seeded rows carry the lowest possible ts, so any live re-scan of
    # the same (model,label,workload,regime,n_req) cell overrides them. This
    # makes the builder idempotent and never-lossy.
    n_seeded = 0
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["model"], row["label"], row["workload"],
                       row["regime"], int(row["n_req"]))
                # normalise to full COLUMNS (older CSVs may lack columns)
                full = {c: row.get(c, "") for c in COLUMNS}
                best[key] = ("00000000_000000", full)
                n_seeded += 1
    print(f"seeded from existing CSV: {n_seeded} rows")

    for d in sweep_dirs:
        base = os.path.basename(d)
        m = DIR_RE.match(base)          # full-grid sweep (one n_req/cell)
        if not m:
            m = CONC_DIR_RE.match(base)  # concurrency sweep (multi n_req/cell)
        if not m:
            print(f"WARN: cannot parse dir name, skipping: {base}",
                  file=sys.stderr)
            continue
        model, ts = m.group("model"), m.group("ts")
        if model not in MODEL_SPECS:
            unknown_models.add(model)

        for rec_path in sorted(
                glob.glob(os.path.join(d, "*", "record.json"))):
            n_records += 1
            try:
                with open(rec_path) as f:
                    rec = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                print(f"WARN: unreadable record {rec_path}: {e}",
                      file=sys.stderr)
                continue
            # concurrency record.json is a list of per-n_req entries; the
            # full-grid one is a single dict. Normalise to a list.
            entries = rec if isinstance(rec, list) else [rec]
            for rec in entries:
                if not rec.get("success"):
                    continue
                n_success += 1

                pp = int(rec["pp"])
                regime = regime_for(model, pp, base)
                spec = MODEL_SPECS.get(model, {})
                row = {
                    "model": model,
                    "label": rec["label"],
                    "tp": rec["tp"],
                    "pp": pp,
                    "layer_split": "-".join(str(x) for x in rec["layer_split"]),
                    # per-TP-rank splits; first tp/2 ranks = Blackwell (B),
                    # last tp/2 ranks = Ada (A)
                    "ffn_splits": ":".join(str(x) for x in rec["ffn_splits"]),
                    "head_splits": ":".join(str(x) for x in rec["head_splits"]),
                    "kv_splits": ":".join(str(x) for x in rec["kv_splits"]),
                    "workload": rec["workload"],
                    "in_len": rec["in_len"],
                    "out_len": rec["out_len"],
                    "n_req": rec["n_req"],
                    "tps": rec["tps"],
                    "ttft_ms": rec["ttft_ms"],
                    "itl_ms": rec["itl_ms"],
                    "regime": regime,
                    "source_dir": base,
                }
                for c in SPEC_COLS:
                    row[c] = spec.get(c, "")

                # n_req in the key so the concurrency (batch) axis is preserved
                # — otherwise the 5 n_req of a concurrency cell collapse to one.
                key = (model, rec["label"], rec["workload"], regime,
                       rec["n_req"])
                if key not in best or ts > best[key][0]:
                    best[key] = (ts, row)

    rows = sorted(
        (r for _, r in best.values()),
        key=lambda r: (r["model"], r["regime"], r["label"], r["workload"],
                       int(r["n_req"])))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # ---- summary ----
    print(f"scanned dirs:      {len(sweep_dirs)}")
    print(f"records found:     {n_records}")
    print(f"success records:   {n_success}")
    print(f"rows after dedup:  {len(rows)}")
    print(f"wrote:             {OUT_CSV}")
    if unknown_models:
        print("WARN: models without specs (empty spec cols): "
              + ", ".join(sorted(unknown_models)), file=sys.stderr)

    per_model = defaultdict(lambda: defaultdict(int))
    for r in rows:
        per_model[r["model"]][r["regime"]] += 1
    print("\nper-model coverage (rows by regime):")
    for mdl in sorted(per_model):
        regs = per_model[mdl]
        tot = sum(regs.values())
        det = ", ".join(f"{k}={v}" for k, v in sorted(regs.items()))
        print(f"  {mdl:8s} total={tot:3d}  ({det})")

    champs = {}
    for r in rows:
        k = (r["model"], r["workload"])
        if k not in champs or float(r["tps"]) > float(champs[k]["tps"]):
            champs[k] = r
    print("\nchampion (max tps) per model x workload:")
    for (mdl, wl) in sorted(champs):
        r = champs[(mdl, wl)]
        print(f"  {mdl:8s} {wl:14s} -> {r['label']:45s} "
              f"[{r['regime']:7s}] tps={float(r['tps']):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
