"""Canonical regret-vs-oracle evaluation, shared by the planner-fix investigation
so every candidate is judged by the IDENTICAL metric. Regret = 1 - pick/oracle,
where oracle = best MEASURED config at that (model, layout, workload, n_req>=32),
and pick = the measured throughput of the planner's top-1 topology (nearest split).

Provides `eval_all(surface=None)` returning per-layout regret stats + the 4+4
"baseline-loss" guard (cells where the planner pick is >0.5% BELOW uniform TP=world),
so a fix can be checked for 2+2 improvement WITHOUT 4+4 regression.

Usage:
  python planner/regret_eval.py                      # current planner
  # programmatic override (a fix candidate):
  from regret_eval import eval_all
  r = eval_all(surface={1:[...],2:[...],4:[...]})    # overrides _ISO_AR_SURFACE
"""
import dataclasses, glob, json, re, sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import perf_planner as P

REPO = Path(__file__).resolve().parents[1]
MIXED_MEAN = {"8b": (1080, 483), "opt30b": (597, 540), "70b": (597, 540)}


def relayout(hw, hg, wg):
    bw, ada = hw.nodes[0][0], hw.nodes[-1][0]
    return dataclasses.replace(hw, nodes=((bw, hg), (ada, wg)))


def _load_cells():
    """All measured cells -> {(mk,hg,wg,wl,n): {label:(tps,ls,tp,pp)}}, plus
    a {(mk,hg,wg,wl,n):(in,out)} shape map recovered from the records."""
    cells = defaultdict(dict); io = {}
    for d in sorted(glob.glob(str(REPO / "results/hetero_*x*_*"))):
        mo = re.search(r"hetero_(\d)x(\d)_([a-z0-9]+)_", d)
        if not mo:
            continue
        hg, wg, mk = int(mo.group(1)), int(mo.group(2)), mo.group(3)
        if mk not in P.MODELS:
            continue
        for rj in glob.glob(d + "/*/record.json"):
            try:
                recs = json.load(open(rj))
            except Exception:
                continue
            for e in (recs if isinstance(recs, list) else [recs]):
                if not (e.get("success") and e.get("tps", 0) > 0):
                    continue
                key = (mk, hg, wg, e.get("workload"), e["n_req"])
                cells[key][e["label"]] = (e["tps"], tuple(e["layer_split"]), e["tp"], e["pp"])
                if e.get("workload") == "mixed":
                    io[key] = MIXED_MEAN.get(mk, (512, 512))
                elif e.get("in_len", -1) >= 0:
                    io[key] = (e["in_len"], e["out_len"])
    return cells, io


_CELLS, _IO = _load_cells()


@contextmanager
def _override_surface(surface):
    old = P._ISO_AR_SURFACE
    old_ref = P._ISO_AR_REF
    if surface is not None:
        P._ISO_AR_SURFACE = surface
        # keep the anchor self-consistent: ref = bandwidth at n_local=4, ~1.05 MB
        P._ISO_AR_REF = P._row_bw(surface[max(surface)], 1.049) if 4 in surface else old_ref
    try:
        yield
    finally:
        P._ISO_AR_SURFACE = old
        P._ISO_AR_REF = old_ref


def eval_all(surface=None, verbose=False):
    """Return dict: per-layout regret mean + cell list, 4+4 baseline-loss count,
    overall mean regret. `surface` overrides _ISO_AR_SURFACE for the eval."""
    out = {"by_layout": {}, "picks": {}}
    with _override_surface(surface):
        bylayout = defaultdict(list)
        base_losses_44 = []
        pick_topo = {}
        for key, r in _CELLS.items():
            mk, hg, wg, wl, n = key
            if n < 32 or key not in _IO:
                continue
            il, ol = _IO[key]
            world = hg + wg
            hw = relayout(P.load_hardware(), hg, wg)
            # Sweep harness sends IDENTICAL prompts; vLLM prefix caching (default ON)
            # prefills the shared prompt ONCE -> unique prefill fraction = 1/n for
            # uniform-shape cells. Mixed streams have varied shapes -> frac=1.0.
            frac = 1.0 if wl == "mixed" else 1.0 / n
            ranked = P.plan(P.MODELS[mk], hw, P.Workload(il, ol, n, prefill_unique_frac=frac), top_k=1)
            if not ranked:
                continue
            pk = ranked[0][1]
            oracle = max(v[0] for v in r.values())
            cand = [v for v in r.values() if v[2] == pk.tp and v[3] == pk.pp]
            if cand:
                pt = min(cand, key=lambda v: sum(abs(a - c) for a, c in zip(v[1], pk.layer_split)))[0]
            else:
                pt = r.get(f"TP{world}PP1_uniform", (0,))[0]
            regret = (1 - pt / oracle) * 100 if oracle else 0
            bylayout[(hg, wg)].append((mk, wl, n, regret, pt, oracle, pk.tp, pk.pp))
            pick_topo[key] = (pk.tp, pk.pp)
            # 4+4 baseline-loss guard
            if (hg, wg) == (4, 4):
                b = r.get(f"TP{world}PP1_uniform")
                if b and pt < b[0] * 0.995:
                    base_losses_44.append((mk, wl, n, (pt / b[0] - 1) * 100))
        for lay, rs in bylayout.items():
            mean = sum(x[3] for x in rs) / len(rs)
            out["by_layout"][lay] = {"mean_regret": mean, "n_cells": len(rs),
                                      "cells": rs, "big": [x for x in rs if x[3] > 10]}
        out["base_losses_44"] = base_losses_44
        out["pick_topo"] = pick_topo
        allr = [x[3] for rs in bylayout.values() for x in rs]
        out["overall_mean_regret"] = sum(allr) / len(allr) if allr else 0
    if verbose:
        for lay in sorted(out["by_layout"]):
            s = out["by_layout"][lay]
            print(f"  {lay[0]}+{lay[1]}: mean regret {s['mean_regret']:5.1f}%  "
                  f"({s['n_cells']} cells, {len(s['big'])} with >10%)")
        print(f"  4+4 baseline-losses (>0.5% below uniform TP8): {len(out['base_losses_44'])}")
        print(f"  OVERALL mean regret: {out['overall_mean_regret']:.1f}%")
    return out


if __name__ == "__main__":
    print("=== CURRENT planner (regret vs measured oracle, n>=32) ===")
    eval_all(verbose=True)
