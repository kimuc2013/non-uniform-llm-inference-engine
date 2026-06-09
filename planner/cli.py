"""Planner CLI: produce a ranked plan list + JSON record for a (cluster,
model, workload) triple, ready to be consumed by either a paper figure or
an experiment runner.

Usage:
    python -m vllm_main.planner.cli \
        --model llama8b \
        --workload-class decode_heavy \
        --cluster head:RTX-PRO-Blackwell:4 \
        --out plan.json

The cluster argument is a CSV of `node_id:gpu_name:count` triples.

Output JSON schema (one record per planner invocation):

    {
        "timestamp": "...",
        "model": {"name": ..., "params_B": ..., "n_layers": ..., ...},
        "cluster": {"groups": [...], "world_size": N, "min_vram_GB": ...},
        "workload": {"class": ..., "in_len": ..., "out_len": ..., "n_requests": ..., "weights": {...}},
        "candidates": [
            {
                "rank": 1,
                "tp": ..., "pp": ...,
                "layer_splits": [...],
                "tp_head_splits": [...], "tp_kv_splits": [...], "tp_ffn_splits": [...],
                "stage_rank_groups": [[...], ...],
                "tp_cross_node": [...],
                "pp_cross_node": [...],
                "rank_to_node": [...],
                "rationale": "...",
                "predicted_wall_s": ...,
                "score": {ScoreBreakdown.to_dict()},
                "env_vars": {VLLM_PP_LAYER_PARTITION, TP_HEAD_SPLITS, ...},
                "feasible": bool,
            }, ...
        ],
        "selected": {<top-ranked feasible candidate>},
        "shell_export": "export TP=...\\nexport PP=...\\n..."
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .cost_model import CostModel, GpuProfile, NetworkProfile
from .gpu_library import get_gpu, GPU_CATALOG
from .model_spec import get_model, MODEL_REGISTRY
from .network_library import build_network, NETWORK_CATALOG, CROSS_NODE_CATALOG
from .planner import Planner, GpuGroup, PartitionSpec
from .scorer import score_partition, ScoreBreakdown
from .workload import resolve_workload, get_class, WORKLOAD_CLASSES, env_overrides


# Datasheet VRAM per GPU type. Used for memory-feasibility checks. Values
# from gpu_library + manual lookup.
VRAM_GB_BY_GPU = {
    "H100-SXM5":          80,
    "H100-PCIe":          80,
    "H200-SXM":           141,
    "A100-SXM4-80GB":     80,
    "A100-PCIe-80GB":     80,
    "A100-SXM4-40GB":     40,
    "V100-SXM2-32GB":     32,
    "V100-PCIe-32GB":     32,
    "L40S":               48,
    "A40":                48,
    "RTX6000-Ada":        48,
    "RTX-PRO-Blackwell":  96,
}


def parse_cluster_spec(spec: str) -> list[GpuGroup]:
    """Parse `node:gpu:count[,node:gpu:count...]` into GpuGroup list."""
    groups: list[GpuGroup] = []
    for chunk in spec.split(","):
        parts = chunk.strip().split(":")
        if len(parts) != 3:
            raise SystemExit(
                f"--cluster: each chunk must be node_id:gpu_name:count, got {chunk!r}"
            )
        node_id, gpu_name, count = parts
        if gpu_name not in GPU_CATALOG:
            raise SystemExit(
                f"--cluster: unknown GPU {gpu_name!r}. "
                f"Available: {sorted(GPU_CATALOG)}"
            )
        groups.append(GpuGroup(
            profile=get_gpu(gpu_name), count=int(count), node_id=node_id,
        ))
    return groups


def min_vram_GB(groups: list[GpuGroup]) -> float:
    """Smallest VRAM in the cluster (bound for any rank)."""
    return float(min(
        VRAM_GB_BY_GPU.get(g.profile.name, 24) for g in groups
    ))


def build_record(
    *,
    model_alias: str,
    workload_class: str,
    cluster_spec_str: str,
    in_len: int | None = None,
    out_len: int | None = None,
    n_requests: int | None = None,
    network: str = "PCIE_GEN5",
    cross_network: str | None = "ETH-10G",
    top_k: int = 10,
) -> dict[str, Any]:
    """Generate the planner JSON record."""
    model = get_model(model_alias)
    groups = parse_cluster_spec(cluster_spec_str)
    net = build_network(network, cross_network)
    workload, cls = resolve_workload(
        workload_class=workload_class,
        in_len=in_len, out_len=out_len, n_requests=n_requests,
    )
    min_vram = min_vram_GB(groups)

    planner = Planner(model=model, network=net, cluster=groups)
    plan_results = planner.plan(workload, top_k=top_k * 3)  # over-fetch; we re-rank

    candidates_with_score: list[dict[str, Any]] = []
    for pr in plan_results:
        p = pr.partition
        # Cost model with this partition for scoring.
        cm = CostModel(
            model=model,
            network=net,
            gpu_per_rank=tuple(planner._gpu_for_rank(p.rank_to_node)),
        )
        sb = score_partition(
            partition=p, workload=workload, cm=cm, cls=cls,
            cluster_min_vram_GB=min_vram,
        )
        env_vars = p.env_vars()
        candidates_with_score.append({
            "tp": p.tp_size,
            "pp": p.pp_size,
            "layer_splits": p.layer_splits,
            "tp_head_splits": p.tp_head_splits,
            "tp_kv_splits": p.tp_kv_splits,
            "tp_ffn_splits": p.tp_ffn_splits,
            "stage_rank_groups": p.stage_rank_groups,
            "tp_cross_node": p.tp_cross_node,
            "pp_cross_node": p.pp_cross_node,
            "rank_to_node": p.rank_to_node,
            "rationale": pr.rationale,
            "predicted_wall_s": pr.predicted_wall_s,
            "score": sb.to_dict(),
            "env_vars": env_vars,
            "feasible": sb.memory_feasible,
        })

    # Final ranking: feasible candidates by weighted_cost_s asc; infeasible
    # at the bottom by original predicted_wall_s.
    feasible = [c for c in candidates_with_score if c["feasible"]]
    infeasible = [c for c in candidates_with_score if not c["feasible"]]
    feasible.sort(key=lambda c: c["score"]["weighted_cost_s"])
    infeasible.sort(key=lambda c: c["predicted_wall_s"])
    final = feasible + infeasible

    # Dedup by (tp, pp, tuple(layer_splits), tuple(ffn_splits))
    seen: set[tuple] = set()
    dedup: list[dict[str, Any]] = []
    for c in final:
        key = (c["tp"], c["pp"],
               tuple(c["layer_splits"]),
               tuple(c["tp_ffn_splits"][:c["tp"]]))
        if key in seen:
            continue
        seen.add(key)
        c["rank"] = len(dedup) + 1
        dedup.append(c)
        if len(dedup) >= top_k:
            break

    selected = dedup[0] if dedup else None

    # Shell export block for the selected config.
    if selected is not None:
        shell_lines = [
            f"export {k}={v}" for k, v in selected["env_vars"].items()
        ]
        shell_lines.insert(0, f"# Planner-selected: tp={selected['tp']} pp={selected['pp']} "
                              f"workload={cls.name} model={model_alias}")
        shell_export = "\n".join(shell_lines)
    else:
        shell_export = "# planner produced no feasible candidates"

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": {
            "alias": model_alias,
            "name": model.name,
            "num_layers": model.num_layers,
            "hidden_size": model.hidden_size,
            "num_q_heads": model.num_q_heads,
            "num_kv_heads": model.num_kv_heads,
            "head_dim": model.head_dim,
            "intermediate_size": model.intermediate_size,
            "vocab_size": model.vocab_size,
            "gqa_group": model.gqa_group,
            "approx_params_B": _approx_params_B(model),
        },
        "cluster": {
            "spec": cluster_spec_str,
            "groups": [
                {"node_id": g.node_id, "gpu": g.profile.name, "count": g.count,
                 "vram_GB": VRAM_GB_BY_GPU.get(g.profile.name, None),
                 "tflops_prefill": g.profile.tflops_prefill,
                 "tflops_decode": g.profile.tflops_decode,
                 "mem_bw_GBs": g.profile.mem_bw_GBs}
                for g in groups
            ],
            "world_size": sum(g.count for g in groups),
            "min_vram_GB": min_vram,
            "network": network,
            "cross_network": cross_network,
        },
        "workload": {
            "class": cls.name,
            "class_description": cls.description,
            "in_len": workload.in_len,
            "out_len": workload.out_len,
            "n_requests": workload.n_requests,
            "weights": {
                "w_prefill": cls.w_prefill,
                "w_decode": cls.w_decode,
                "pp_depth_penalty": cls.pp_depth_penalty,
                "tp_comm_weight": cls.tp_comm_weight,
            },
        },
        "candidates": dedup,
        "selected": selected,
        "shell_export": shell_export,
    }
    return record


def _approx_params_B(model) -> float:
    """Approximate params in billions. Used for paper table headers."""
    h = model.hidden_size
    per_layer = (
        (model.qkv_combined_dim + model.q_proj_dim) * h        # attn
        + 3 * h * model.intermediate_size                       # ffn
    )
    total = model.num_layers * per_layer + 2 * model.vocab_size * h
    return round(total / 1e9, 2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    choices=sorted(MODEL_REGISTRY.keys()),
                    help="Model alias.")
    ap.add_argument("--workload-class", default="balanced",
                    choices=sorted(WORKLOAD_CLASSES.keys()),
                    help="decode_heavy | prefill_heavy | balanced. May be "
                         "overridden by VLLM_PLANNER_WORKLOAD env var.")
    ap.add_argument("--cluster", required=True,
                    help="CSV of node_id:gpu_name:count, e.g. "
                         "head:RTX-PRO-Blackwell:4,worker:RTX6000-Ada:4")
    ap.add_argument("--network", default="PCIE_GEN5",
                    choices=sorted(NETWORK_CATALOG.keys()),
                    help="Intra-node interconnect profile.")
    ap.add_argument("--cross-network", default="ETH-10G",
                    choices=sorted(CROSS_NODE_CATALOG.keys()) + ["NONE"],
                    help="Cross-node interconnect ('NONE' for single-node).")
    ap.add_argument("--in-len", type=int, default=None)
    ap.add_argument("--out-len", type=int, default=None)
    ap.add_argument("--n-requests", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None,
                    help="Write JSON record to this path. If omitted, prints to stdout.")
    ap.add_argument("--quiet", action="store_true",
                    help="Skip the human-readable banner; useful when consuming JSON.")
    args = ap.parse_args(argv)

    cross_net = None if args.cross_network == "NONE" else args.cross_network

    rec = build_record(
        model_alias=args.model,
        workload_class=args.workload_class,
        cluster_spec_str=args.cluster,
        in_len=args.in_len, out_len=args.out_len, n_requests=args.n_requests,
        network=args.network, cross_network=cross_net,
        top_k=args.top_k,
    )

    if not args.quiet:
        _print_human_summary(rec)

    text = json.dumps(rec, indent=2, ensure_ascii=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        if not args.quiet:
            print(f"\nwrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


def _print_human_summary(rec: dict[str, Any]) -> None:
    """Concise human banner for the CLI (printed to stderr so JSON to stdout)."""
    f = sys.stderr
    print("=" * 78, file=f)
    print(f"Planner result @ {rec['timestamp']}", file=f)
    print("-" * 78, file=f)
    print(f"model:   {rec['model']['alias']} "
          f"({rec['model']['approx_params_B']} B params, "
          f"{rec['model']['num_layers']} layers, "
          f"GQA={rec['model']['gqa_group']})", file=f)
    print(f"cluster: {rec['cluster']['spec']}  "
          f"(world={rec['cluster']['world_size']}, "
          f"min_vram={rec['cluster']['min_vram_GB']:.0f}GB)", file=f)
    print(f"workload: class={rec['workload']['class']} "
          f"in={rec['workload']['in_len']} out={rec['workload']['out_len']} "
          f"n_req={rec['workload']['n_requests']}", file=f)
    print(f"  weights: w_p={rec['workload']['weights']['w_prefill']} "
          f"w_d={rec['workload']['weights']['w_decode']} "
          f"tp_comm×{rec['workload']['weights']['tp_comm_weight']}", file=f)
    print("-" * 78, file=f)
    sel = rec["selected"]
    if sel is None:
        print("NO FEASIBLE CANDIDATE FOUND.", file=f)
    else:
        print(f"SELECTED: tp={sel['tp']} pp={sel['pp']} "
              f"layers={sel['layer_splits']} ffn={sel['tp_ffn_splits'][:sel['tp']]}",
              file=f)
        print(f"  predicted_wall_s={sel['predicted_wall_s']:.2f}  "
              f"weighted_cost={sel['score']['weighted_cost_s']:.2f}",
              file=f)
    print("-" * 78, file=f)
    print(f"Top {len(rec['candidates'])} candidates:", file=f)
    print(f"{'rank':>4} {'tp':>3} {'pp':>3} {'wall_s':>8} {'wcost_s':>8} "
          f"{'ovl':>4} {'feas':>5}  layers", file=f)
    for c in rec["candidates"]:
        print(f"{c['rank']:>4} {c['tp']:>3} {c['pp']:>3} "
              f"{c['predicted_wall_s']:>8.2f} "
              f"{c['score']['weighted_cost_s']:>8.2f} "
              f"{c['score']['predicted_pp_overlap']:>4.2f} "
              f"{'Y' if c['feasible'] else 'N':>5}  {c['layer_splits']}",
              file=f)
    print("=" * 78, file=f)


if __name__ == "__main__":
    raise SystemExit(main())
