"""Launcher CLI. M2 supports `--dry-run` only; M3 will add live launch.

Usage:
    python -m vllm_main.launcher \\
        --target llama70b \\
        --cluster-env ./cluster.local.env \\
        --dry-run \\
        --save-resolved-config /tmp/m2_llama70b.json \\
        [--set TP_HEAD_SPLITS=16,16,16,16,16,16,16,16] \\
        [--auto-tp {keep,on,off}]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .cluster import ClusterEnvError, load_cluster_env
from .config import resolve_launch_config
from .native_runtime import dry_run_report, vllm_command_line, export_block
from .presets import ALL_TARGETS


DEFAULT_CLUSTER_ENV = Path(__file__).resolve().parents[1] / "cluster.local.env"


def _parse_set(items: list[str] | None) -> dict[str, str]:
    if not items:
        return {}
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set requires KEY=VALUE form, got: {item!r}")
        k, _, v = item.partition("=")
        k = k.strip()
        if not k:
            raise SystemExit(f"--set has empty key in: {item!r}")
        out[k] = v
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vllm_main.launcher",
        description=(
            "vLLM 0.21-based heterogeneous launcher. "
            "M2 supports dry-run only; live launch lands in M3."
        ),
    )
    p.add_argument("--target", required=True, choices=ALL_TARGETS,
                   help="Model preset to launch.")
    p.add_argument("--cluster-env", type=Path, default=DEFAULT_CLUSTER_ENV,
                   help=f"Path to cluster.local.env (default: {DEFAULT_CLUSTER_ENV}).")
    p.add_argument("--set", dest="set_items", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="Override or add a single env var. Repeatable.")
    p.add_argument("--auto-tp", choices=("keep", "on", "off"), default="keep",
                   help=("Auto-TP planner toggle. With --auto-plan this maps "
                         "to AUTO_TP_SPLIT in env."))
    p.add_argument("--auto-plan", action="store_true",
                   help=("Run the cost-model planner over the (model, cluster, "
                         "workload) tuple and inject the top plan's env vars."))
    p.add_argument("--in-len", type=int, default=512,
                   help="Workload input length for --auto-plan (default 512).")
    p.add_argument("--out-len", type=int, default=256,
                   help="Workload output length for --auto-plan (default 256).")
    p.add_argument("--n-req", type=int, default=64,
                   help="Concurrent request count for --auto-plan (default 64).")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve and print/save the config; do not launch.")
    p.add_argument("--save-resolved-config", type=Path, default=None,
                   metavar="PATH",
                   help="Write the resolved config JSON to this path.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the human-readable report.")
    p.add_argument(
        "--auto-pp-overlap",
        choices=("auto", "on", "off"),
        default="auto",
        help=("Auto-configure PP-overlap settings (microbatch, mb_size, bq, "
              "broadcast stream) from hardware spec + workload size. "
              "'auto' (default) enables when PP>1, 'off' leaves env as-is, "
              "'on' forces regardless of PP."),
    )
    p.add_argument(
        "--workload-n-req",
        type=int,
        default=None,
        help=("Concurrent request count used by --auto-pp-overlap to pick "
              "mb_size. Falls back to MAX_NUM_SEQS env from the preset."),
    )
    return p


def _run_planner(target: str, cluster, in_len: int, out_len: int,
                 n_req: int) -> tuple[dict[str, str], str]:
    """Build cluster spec from preset + Cluster, run planner, return env vars
    from the top candidate and a human-readable rationale string."""
    # Lazy import — planner has heavy deps; only loaded when --auto-plan used.
    from ..planner import Planner, Workload
    from ..planner import llama_3_3_70b, llama_3_1_8b, qwen_3_32b
    from ..planner.planner import GpuGroup
    from ..planner.gpu_library import RTX_PRO_BLW, get_gpu
    from ..planner.network_library import build_network

    # ModelSpec mapping by target name
    spec_map = {
        "llama70b":   llama_3_3_70b,
        "llama8b":    llama_3_1_8b,
        "qwen3_32b":  qwen_3_32b,
    }
    if target not in spec_map:
        raise SystemExit(f"--auto-plan does not have a ModelSpec for {target!r}")
    model_spec = spec_map[target]

    # Map cluster.local.env accel names to planner GPU profiles
    accel_to_profile = {
        "blackwell":   RTX_PRO_BLW,
        "ada":         get_gpu("RTX6000-Ada"),
        "rtx6000-ada": get_gpu("RTX6000-Ada"),
        "h100":        get_gpu("H100-SXM5"),
        "a100":        get_gpu("A100-SXM4-80GB"),
    }
    head_profile = accel_to_profile.get(cluster.head_accel.lower())
    worker_profile = accel_to_profile.get(cluster.worker_accel.lower())
    if head_profile is None or worker_profile is None:
        raise SystemExit(
            f"--auto-plan: unknown accel name in cluster file: "
            f"head_accel={cluster.head_accel!r} / worker_accel={cluster.worker_accel!r}."
            f" Supported: {sorted(accel_to_profile)}"
        )

    groups = [GpuGroup(profile=head_profile, count=cluster.head_gpus, node_id="head")]
    if not cluster.is_single_node() and cluster.worker_gpus > 0:
        groups.append(GpuGroup(profile=worker_profile,
                               count=cluster.worker_gpus, node_id="worker"))

    network = build_network("PCIE_GEN5", "IB-HDR-200")
    workload = Workload(in_len=in_len, out_len=out_len, n_requests=n_req)
    planner = Planner(model=model_spec, network=network, cluster=groups)
    results = planner.plan(workload, top_k=3)
    if not results:
        raise SystemExit("--auto-plan: planner returned no valid candidates.")
    top = results[0]
    env = top.partition.env_vars()
    rationale = (
        f"auto-plan top: TP={top.partition.tp_size} PP={top.partition.pp_size} "
        f"predicted_wall={top.predicted_wall_s:.2f}s — {top.rationale}"
    )
    return env, rationale


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cluster = load_cluster_env(args.cluster_env)
    except ClusterEnvError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    overrides = _parse_set(args.set_items)
    if args.auto_tp == "on":
        overrides.setdefault("AUTO_TP_SPLIT", "1")
        overrides.setdefault("AUTOSPLIT", "1")
    elif args.auto_tp == "off":
        overrides.setdefault("AUTO_TP_SPLIT", "0")
        overrides.setdefault("AUTOSPLIT", "0")

    plan_rationale: str | None = None
    if args.auto_plan:
        plan_env, plan_rationale = _run_planner(
            target=args.target, cluster=cluster,
            in_len=args.in_len, out_len=args.out_len, n_req=args.n_req,
        )
        # CLI --set overrides still win; plan fills the rest.
        for k, v in plan_env.items():
            overrides.setdefault(k, v)

    cli_invocation = ["python", "-m", "vllm_main.launcher"] + (
        argv if argv is not None else sys.argv[1:]
    )

    cfg, errs = resolve_launch_config(
        target=args.target,
        cluster=cluster,
        overrides=overrides,
        cli_invocation=cli_invocation,
    )

    # PP-overlap auto-config (microbatch / mb_size / bq / broadcast stream).
    # Runs AFTER resolve so we have the final PP/TP/MODEL values. Explicit
    # CLI overrides for these keys are still honored (we use setdefault).
    pp_overlap_rationale: str | None = None
    pp = int(cfg.env.get("PP", "1"))
    do_overlap = args.auto_pp_overlap == "on" or (
        args.auto_pp_overlap == "auto" and pp > 1
    )
    if do_overlap:
        try:
            from .pp_overlap_config import auto_configure, apply_to_env

            n_req = args.workload_n_req
            if n_req is None:
                n_req = int(cfg.env.get("MAX_NUM_SEQS", "16"))
            tp = int(cfg.env.get("TP", "1"))
            model_name = cfg.env.get("MODEL", "")

            overlap_cfg = auto_configure(
                num_reqs=n_req,
                pp_size=pp,
                tp_size=tp,
                model_name=model_name,
            )
            # setdefault — explicit --set wins
            tmp: dict[str, str] = {}
            apply_to_env(tmp, overlap_cfg)
            for k, v in tmp.items():
                cfg.env.setdefault(k, v)
            pp_overlap_rationale = (
                f"auto-pp-overlap: use_microbatch={overlap_cfg.use_microbatch} "
                f"mb_size={overlap_cfg.mb_size} bq={overlap_cfg.bq} "
                f"bcast_stream={overlap_cfg.enable_broadcast_stream} — "
                f"{overlap_cfg.reasoning}"
            )
        except Exception as exc:
            pp_overlap_rationale = f"auto-pp-overlap: FAILED ({exc})"

    if not args.quiet:
        print(dry_run_report(cfg))
        if plan_rationale:
            print(f"\nplan: {plan_rationale}")
        if pp_overlap_rationale:
            print(f"\n{pp_overlap_rationale}")
        print()
        print("would-launch command:")
        print(f"  {vllm_command_line(cfg)}")
        print()
        if errs:
            print("VALIDATION ERRORS:")
            for e in errs:
                print(f"  - {e}")
        else:
            print("validation: OK")

    if args.save_resolved_config:
        args.save_resolved_config.parent.mkdir(parents=True, exist_ok=True)
        cfg.save(args.save_resolved_config)
        if not args.quiet:
            print(f"\nresolved config saved -> {args.save_resolved_config}")

    if not args.dry_run:
        print("\n[NOTE] live launch is implemented in M3. Re-run with --dry-run.",
              file=sys.stderr)
        return 64

    return 0 if not errs else 3
