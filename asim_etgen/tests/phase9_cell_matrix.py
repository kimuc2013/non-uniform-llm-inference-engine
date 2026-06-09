"""Phase 9: cell matrix experiments — (GPU ratio × Model × Workload).

Builds on Phase 8b's continuous-batching cost model + end-to-end serving
validation, but sweeps a paper-grade matrix of cluster sizes, model
architectures, and workloads on the real cluster's hetero topology
(Blackwell + Ada).

Cell matrix (anchor at C13 = Phase 6 measured V-bottom):

  GPU 1:1      Llama-8B × {S1 chat, S3 decode-heavy}      [C1, C2]
  GPU 2:2      Llama-8B × {S1, S2 prefill-heavy}          [C3, C4]
               Qwen-32B × {S1}                            [C5]
               OPT-30B × {S1}  (simulation only — engine is Llama-only) [C6]
  GPU 4:4      Qwen-32B × {S1, S2, S3}                    [C7, C8, C9]
               OPT-30B × {S1, S2, S3}                     [C10, C11, C12]
               Llama-70B × {S1 (Phase 6 anchor), S2, S3}  [C13, C14, C15]

Each cell:
  1) Enumerate (TP, PP) factorizations × uniform/non-uniform shards
  2) Score every plan with the analytical planner (predict_serving_wall_us)
  3) Run ASTRA-sim for each plan and compose into end-to-end serving wall
  4) Report planner-pick vs simulator-best (+ slowdown if mismatch)

Notes:
  - Engine integration of OPT-30B is OUT OF SCOPE for this session.
    OPT cells (C6, C10-C12) are simulator-only and rely on the
    architecture-aware planner & ET generator that branch `mlp_kind`.
  - C13 (4:4 Llama-70B, S1) reproduces the Phase 6 anchor topology so
    its planner prediction can be cross-checked against the real
    measurement [48, 32] @ 17.28s.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asim_etgen.inference_workload import WorkloadSpec, ModelSpec
from asim_etgen.runtime_model import ComputeProfile

from asim_etgen.tests.phase8_plan_selection import (
    ClusterTopology, ServingWorkload, CandidatePlan,
    enumerate_plans, predict_serving_wall_us, simulator_serving_wall_s,
    NET_2NODE, NET_1NODE,
)

# Per-cluster-size network YAMLs (generated separately for npus_count match).
NET_W2_CROSS = "/tmp/net_W2_cross.yml"   # 1:1 cross-node, single-dim [2]
NET_W4_CROSS = "/tmp/net_W4_cross.yml"   # 2:2 cross-node, 2-dim [2, 2]
NET_W8_CROSS = NET_2NODE                  # 4:4 cross-node, 2-dim [4, 2] (existing)


# ----------------------------------------------------------------------
# Model catalog
# ----------------------------------------------------------------------

LLAMA_3_8B = ModelSpec(
    name="Llama-3-8B", num_layers=32, hidden=4096,
    num_q_heads=32, num_kv_heads=8, head_dim=128, intermediate=14336,
    mlp_kind="swiglu", norm_kind="rmsnorm",
)
QWEN_2_5_32B = ModelSpec(
    name="Qwen2.5-32B", num_layers=64, hidden=5120,
    num_q_heads=40, num_kv_heads=8, head_dim=128, intermediate=27648,
    mlp_kind="swiglu", norm_kind="rmsnorm", has_qkv_bias=True,
)
LLAMA_3_70B = ModelSpec(
    name="Llama-3-70B", num_layers=80, hidden=8192,
    num_q_heads=64, num_kv_heads=8, head_dim=128, intermediate=28672,
    mlp_kind="swiglu", norm_kind="rmsnorm",
)
OPT_30B = ModelSpec(
    name="OPT-30B", num_layers=48, hidden=7168,
    num_q_heads=56, num_kv_heads=56, head_dim=128, intermediate=28672,
    mlp_kind="relu_2matmul", norm_kind="layernorm",
)
ALL_MODELS = [LLAMA_3_8B, QWEN_2_5_32B, LLAMA_3_70B, OPT_30B]


# ----------------------------------------------------------------------
# Real cluster GPUs (Blackwell + Ada hetero)
# ----------------------------------------------------------------------

BLACKWELL = ComputeProfile(
    name="RTX-PRO-Blackwell",
    spec_tflops_bf16=380.0, spec_mem_bw_GBs=1792.0,
)
ADA = ComputeProfile(
    name="RTX6000-Ada",
    spec_tflops_bf16=91.0, spec_mem_bw_GBs=960.0,
)


def make_topology(blackwell_count: int, ada_count: int,
                  single_node: bool) -> ClusterTopology:
    gpus = [BLACKWELL] * blackwell_count + [ADA] * ada_count
    world = blackwell_count + ada_count
    if single_node:
        nodes = ["n0"] * world
        n_dims = 1
        if world == 2:
            net = "/tmp/single_node_2npu.yml"
        elif world == 4:
            net = "/tmp/single_node_4npu.yml"
        else:
            net = NET_1NODE
    else:
        nodes = ["n0"] * blackwell_count + ["n1"] * ada_count
        if world == 2:
            net = NET_W2_CROSS
            n_dims = 1
        elif world == 4:
            net = NET_W4_CROSS
            n_dims = 2
        else:
            net = NET_W8_CROSS
            n_dims = 2
    name = (f"{blackwell_count}B+{ada_count}A "
            f"({'single-node' if single_node else 'cross-node'})")
    return ClusterTopology(
        name=name,
        rank_to_gpu=tuple(gpus),
        rank_to_node=tuple(nodes),
        network_file=net,
        is_single_node=single_node,
        n_dims=n_dims,
    )


# 1:1 = 1 Blackwell + 1 Ada (cross-node)
# 2:2 = 2 Blackwell + 2 Ada (cross-node)
# 4:4 = 4 Blackwell + 4 Ada (cross-node, our real cluster)
TOPO_1x1 = make_topology(1, 1, single_node=False)
TOPO_2x2 = make_topology(2, 2, single_node=False)
TOPO_4x4 = make_topology(4, 4, single_node=False)


# ----------------------------------------------------------------------
# Workloads (in / out / N_req)
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class WL:
    label: str
    in_len: int
    out_len: int
    n_requests: int


WL_S1 = WL("S1 chat (in512/out128)", in_len=512, out_len=128, n_requests=25)
WL_S1_BIG = WL("S1 chat large-N (in512/out128, N=50)", in_len=512, out_len=128, n_requests=50)
WL_S2 = WL("S2 prefill-heavy (in1024/out256)", in_len=1024, out_len=256, n_requests=32)
WL_S3 = WL("S3 decode-heavy (in128/out512)", in_len=128, out_len=512, n_requests=50)


# ----------------------------------------------------------------------
# Cell matrix
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class Cell:
    label: str
    topology: ClusterTopology
    model: ModelSpec
    workload: WL


def cells_for_1x1() -> list[Cell]:
    return [
        Cell("C1 1:1 Llama-8B S1",  TOPO_1x1, LLAMA_3_8B,  WL_S1),
        Cell("C2 1:1 Llama-8B S3",  TOPO_1x1, LLAMA_3_8B,  WL_S3),
    ]


def cells_for_2x2() -> list[Cell]:
    return [
        Cell("C3 2:2 Llama-8B S1",  TOPO_2x2, LLAMA_3_8B,  WL_S1_BIG),
        Cell("C4 2:2 Llama-8B S2",  TOPO_2x2, LLAMA_3_8B,  WL_S2),
        Cell("C5 2:2 Qwen-32B S1",  TOPO_2x2, QWEN_2_5_32B, WL_S1),
        Cell("C6 2:2 OPT-30B S1",   TOPO_2x2, OPT_30B,    WL_S1),
    ]


def cells_for_4x4() -> list[Cell]:
    return [
        Cell("C7 4:4 Qwen-32B S1",  TOPO_4x4, QWEN_2_5_32B, WL_S1_BIG),
        Cell("C8 4:4 Qwen-32B S2",  TOPO_4x4, QWEN_2_5_32B, WL_S2),
        Cell("C9 4:4 Qwen-32B S3",  TOPO_4x4, QWEN_2_5_32B, WL_S3),
        Cell("C10 4:4 OPT-30B S1",  TOPO_4x4, OPT_30B,    WL_S1_BIG),
        Cell("C11 4:4 OPT-30B S2",  TOPO_4x4, OPT_30B,    WL_S2),
        Cell("C12 4:4 OPT-30B S3",  TOPO_4x4, OPT_30B,    WL_S3),
        Cell("C13 4:4 Llama-70B S1 [Phase 6 anchor]", TOPO_4x4, LLAMA_3_70B, WL_S1_BIG),
        Cell("C14 4:4 Llama-70B S2", TOPO_4x4, LLAMA_3_70B, WL_S2),
        Cell("C15 4:4 Llama-70B S3", TOPO_4x4, LLAMA_3_70B, WL_S3),
    ]


ALL_CELLS: list[Cell] = cells_for_1x1() + cells_for_2x2() + cells_for_4x4()


# ----------------------------------------------------------------------
# Cell evaluation
# ----------------------------------------------------------------------

def evaluate_cell(cell: Cell, sim_cache: dict, skip_sim: bool = False) -> dict:
    """Returns per-plan planner predictions and simulator walls (under
    end-to-end continuous-batching composition)."""
    swl = ServingWorkload(
        n_requests=cell.workload.n_requests,
        in_len=cell.workload.in_len,
        out_len=cell.workload.out_len,
    )
    plans = enumerate_plans(cell.topology, cell.model)
    rows = []
    for plan in plans:
        # Skip plans where pp_size > num_layers (degenerate)
        if plan.pp_size > cell.model.num_layers:
            continue
        try:
            pred = predict_serving_wall_us(plan, cell.topology, cell.model, swl)
            planner_us = pred["total_us"]
        except Exception as e:
            planner_us = float("inf")
            pred = {"prefill_wall_us": float("nan"), "decode_wall_us": float("nan"),
                    "total_us": float("inf")}
        if skip_sim:
            sim_total_s = float("nan")
            sim_pre, sim_dec = float("nan"), float("nan")
        else:
            try:
                sim = simulator_serving_wall_s(plan, cell.topology, swl, sim_cache)
                sim_total_s = sim["total_s"]
                sim_pre, sim_dec = sim["prefill_wall_s"], sim["decode_wall_s"]
            except Exception:
                sim_total_s = float("nan")
                sim_pre, sim_dec = float("nan"), float("nan")
        rows.append({
            "plan": plan.name,
            "tp": plan.tp_size, "pp": plan.pp_size,
            "layer_splits": list(plan.layer_splits),
            "head_splits": list(plan.head_splits),
            "ffn_splits": list(plan.ffn_splits),
            "planner_us": planner_us,
            "planner_pre_us": pred["prefill_wall_us"],
            "planner_dec_us": pred["decode_wall_us"],
            "sim_s": sim_total_s,
            "sim_pre_s": sim_pre,
            "sim_dec_s": sim_dec,
        })
    rows_by_planner = sorted(rows, key=lambda r: r["planner_us"])
    rows_by_sim = sorted(
        [r for r in rows if r["sim_s"] == r["sim_s"]],
        key=lambda r: r["sim_s"],
    )
    pick = rows_by_planner[0]["plan"] if rows_by_planner else None
    winner = rows_by_sim[0]["plan"] if rows_by_sim else None
    return {
        "cell": cell.label,
        "model": cell.model.name,
        "gpu_ratio": cell.topology.name,
        "workload": cell.workload.label,
        "rows": rows,
        "rows_by_planner": rows_by_planner,
        "rows_by_sim": rows_by_sim,
        "planner_pick": pick,
        "sim_winner": winner,
        "match": pick == winner,
    }


def print_cell(res: dict) -> None:
    print(f"\n{'=' * 100}")
    print(f"{res['cell']}")
    print(f"  model:    {res['model']}")
    print(f"  gpu:      {res['gpu_ratio']}")
    print(f"  workload: {res['workload']}")
    if not res["rows"]:
        print("  (no feasible plans)")
        return
    print(f"\n  {'plan':<32s} {'planner total (ms)':>20s} {'sim total (s)':>14s} "
          f"{'pre/dec sim (s)':>20s}")
    print("  " + "-" * 88)
    for r in res["rows_by_planner"]:
        sim_str = "FAIL" if r["sim_s"] != r["sim_s"] else f"{r['sim_s']:.4f}"
        if r["sim_s"] == r["sim_s"]:
            pd_str = f"{r['sim_pre_s']:.3f}/{r['sim_dec_s']:.3f}"
        else:
            pd_str = "-"
        print(f"  {r['plan']:<32s} {r['planner_us']/1e3:>19.1f} "
              f"{sim_str:>14s} {pd_str:>20s}")
    print(f"\n  Planner pick:   {res['planner_pick']}")
    print(f"  Simulator best: {res['sim_winner']}")
    if res["match"]:
        print("  Match: YES")
    elif res["planner_pick"] and res["sim_winner"]:
        pick_row = next((r for r in res["rows"] if r["plan"] == res["planner_pick"]), None)
        best_row = res["rows_by_sim"][0] if res["rows_by_sim"] else None
        if pick_row and best_row and pick_row["sim_s"] == pick_row["sim_s"]:
            sd = pick_row["sim_s"] / best_row["sim_s"]
            print(f"  Match: NO ({sd:.2f}× slower than sim best)")
        else:
            print("  Match: NO (sim missing for picker plan)")


def main():
    print("Phase 9: cell matrix — (GPU ratio × Model × Workload)")
    print("=" * 100)
    print(f"  Total cells: {len(ALL_CELLS)}")
    print(f"  Models:      {[m.name for m in ALL_MODELS]}")
    print(f"  Topologies:  1:1, 2:2, 4:4 (Blackwell + Ada cross-node)")

    overall: list[dict] = []
    t_start = time.time()
    for cell in ALL_CELLS:
        sim_cache: dict = {}      # cache iter sims within a single cell
        res = evaluate_cell(cell, sim_cache)
        print_cell(res)
        overall.append(res)

    print()
    print("=" * 100)
    print(f"SUMMARY — {len(ALL_CELLS)} cells, "
          f"total wall: {time.time() - t_start:.1f}s")
    print("=" * 100)

    n_match = sum(1 for o in overall if o["match"])
    print(f"\n  Match rate: {n_match} / {len(overall)} cells")
    print()
    print(f"  {'cell':<46s} {'planner pick':<32s} {'match':>10s}")
    print("  " + "-" * 90)
    for o in overall:
        if o["match"]:
            tag = "✓"
        elif o["planner_pick"] and o["sim_winner"]:
            pick_row = next((r for r in o["rows"] if r["plan"] == o["planner_pick"]), None)
            best_row = o["rows_by_sim"][0] if o["rows_by_sim"] else None
            if pick_row and best_row and pick_row["sim_s"] == pick_row["sim_s"]:
                sd = pick_row["sim_s"] / best_row["sim_s"]
                tag = f"× ({sd:.2f}×)"
            else:
                tag = "× (?)"
        else:
            tag = "×"
        print(f"  {o['cell']:<46s} {(o['planner_pick'] or '-'):<32s} {tag:>10s}")

    # Winning plan-family distribution
    families = {}
    for o in overall:
        if o["sim_winner"]:
            fam = "_".join(o["sim_winner"].split("_")[0:2])
            families.setdefault(fam, []).append(o["cell"])
    print()
    print("  Winning-plan diversity (simulator best):")
    for fam, cells in families.items():
        print(f"    {fam}: {len(cells)} cells")

    out_json = "/tmp/phase9_cell_matrix.json"
    with open(out_json, "w") as f:
        json.dump(overall, f, indent=2, default=str)
    print(f"\n  Raw per-cell results written to {out_json}")


if __name__ == "__main__":
    main()
