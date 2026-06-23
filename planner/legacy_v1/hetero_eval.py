"""Heterogeneous multi-node planner evaluation.

For each (model, workload, hetero cluster spec) cell:
  1. Run planner → enumerate candidate (uniform, planner_top_k) configs.
  2. Launch vLLM with `--distributed-executor-backend ray` so head+worker
     can both serve.  Stage 0 lives on the head node (Blackwell), stage 1+
     on the worker (Ada).
  3. Run perf/performance.py against the OpenAI endpoint.
  4. Record metrics, write per-cell record.json + validation table.

Assumes a Ray cluster is already up:
    head:   ray start --head --node-ip-address=<head_ip> --port=6379 --num-gpus=4
    worker: ray start --address=<head_ip>:6379 --num-gpus=4
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from planner.cluster_env import CFG

REPO = Path(__file__).resolve().parents[1]
PY_HEAD = CFG.head_py
PERF = REPO / "perf" / "performance.py"
HEAD_IP = CFG.head_fabric_ip
WORKER_IP = CFG.worker_ssh_host
RAY_ADDRESS = CFG.ray_address

# Import shared helpers from the existing run_eval module.
sys.path.insert(0, str(REPO.parent))
from vllm_main.planner.run_eval import (
    ServerHandle, _free_port, wait_for_ready, parse_perf_summary,
    generate_prompt, stop_server, _wait_gpu_clear,
)


# ---------------------------------------------------------------------------
# Hetero candidate generation
# ---------------------------------------------------------------------------

@dataclass
class HeteroCell:
    model_alias: str
    workload_class: str
    head_n: int      # Blackwell GPUs on head node
    worker_n: int    # Ada GPUs on worker node
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = (
                f"{self.model_alias}_{self.workload_class}"
                f"_{self.head_n}+{self.worker_n}"
            )

    @property
    def total_gpus(self) -> int:
        return self.head_n + self.worker_n


@dataclass
class HeteroRun:
    cell: HeteroCell
    config_id: str
    tp: int
    pp: int
    plan_type: str       # uniform / planner_top1 / planner_top2 / ...
    layer_splits: list[int]
    tp_head_splits: list[int]
    tp_ffn_splits: list[int]
    tp_kv_splits: list[int]
    env_vars: dict[str, str]
    in_len: int
    out_len: int
    n_requests: int
    predicted_wall_s: float = -1.0
    weighted_cost_s: float = -1.0
    measured_tps: float = 0.0
    measured_runtime_s: float = 0.0
    measured_ttft_ms: float = 0.0
    measured_itl_ms: float = 0.0
    success: bool = False
    failure_reason: str = ""


def workload_for_hetero(
    workload_class: str, model_alias: str, total_gpus: int,
) -> tuple[int, int, int]:
    """Pick (in_len, out_len, n_req) for a hetero cell.

    Hetero clusters mix slow + fast GPUs so absolute throughput is lower than a
    full-Blackwell pool — keep `n_req` a bit lower than the homogeneous
    `workload_settings`, but still big enough to give ≥10s wall time.
    """
    if model_alias == "opt30b" and workload_class == "prefill_heavy":
        return 512, 64, 32 * max(1, total_gpus // 2)

    base_in_out = {
        "decode_heavy":  (128, 512),
        "prefill_heavy": (1024, 64),
        "balanced":      (512, 256),
    }[workload_class]
    in_len, out_len = base_in_out

    size_label = {
        "llama8b":   "small",
        "llama70b":  "huge",
        "qwen3_32b": "medium",
        "opt30b":    "medium",
    }.get(model_alias, "small")

    base_req = {
        "decode_heavy":  {"small": 96, "medium": 48, "huge": 24},
        "prefill_heavy": {"small": 256, "medium": 128, "huge": 48},
        "balanced":      {"small": 80, "medium": 40, "huge": 20},
    }[workload_class][size_label]
    if size_label == "small":
        n_req = base_req * max(1, total_gpus // 2)
    elif size_label == "medium":
        n_req = base_req * max(1, total_gpus // 4)
    else:
        n_req = base_req
    return in_len, out_len, n_req


def build_planner_record(cell: HeteroCell) -> dict[str, Any]:
    """Run the planner for a hetero cell."""
    from .cli import build_record
    cluster = (
        f"head:RTX-PRO-Blackwell:{cell.head_n},"
        f"worker:RTX6000-Ada:{cell.worker_n}"
    )
    return build_record(
        model_alias=cell.model_alias,
        workload_class=cell.workload_class,
        cluster_spec_str=cluster,
        network="PCIE_GEN5",
        cross_network="ETH-25G",
        top_k=8,
    )


def enumerate_uniform_baselines(total: int) -> list[tuple[int, int]]:
    """Sweep feasible uniform (TP, PP). Cross-node PP=4+ has init time of
    >30 min for 70B which makes the sweep impractical. We restrict to
    PP ≤ 2 and TP ≤ 4 — covers the paper's key TP=4×PP=2 [50, 30] claim
    and a representative TP-only baseline (TP=2 PP=1 / TP=4 PP=1)."""
    out = []
    for tp in (1, 2, 4):
        if total % tp == 0:
            pp = total // tp
            if 1 <= pp <= 2:
                out.append((tp, pp))
    return out


def memory_feasible_hetero(model_alias: str, tp: int, pp: int,
                           in_len: int, out_len: int, n_req: int) -> tuple[bool, str]:
    """Memory feasibility on the SMALLER of {Blackwell 96GB, Ada 48GB}."""
    from .model_spec import get_model
    from .scorer import stage_param_bytes, kv_cache_bytes_per_rank
    m = get_model(model_alias)
    n_per_stage = max(1, m.num_layers // pp)
    w = stage_param_bytes(m, n_per_stage, tp) / 1e9
    kv = kv_cache_bytes_per_rank(
        m, n_layers_in_stage=n_per_stage, tp_size=tp,
        max_concurrent_seqs=n_req, seq_len=in_len + out_len,
    ) / 1e9
    total = w + kv + 4.0
    # Ada is the smaller node.
    if total > 48.0:
        return False, f"need {total:.1f} GB > Ada 48 GB (w={w:.1f}, kv={kv:.1f})"
    return True, f"need {total:.1f} GB ≤ Ada 48 GB"


def runs_from_planner(cell: HeteroCell, rec: dict[str, Any]) -> list[HeteroRun]:
    in_len, out_len, n_req = workload_for_hetero(
        cell.workload_class, cell.model_alias, cell.total_gpus
    )
    candidates = rec["candidates"]
    runs: list[HeteroRun] = []

    from .model_spec import get_model
    m = get_model(cell.model_alias)

    seen_uniform: set[tuple[int, int]] = set()
    # Uniform sweep — explicitly emit uniform splits to keep the patched
    # linear/parameter paths consistent across all candidates.
    for tp, pp in enumerate_uniform_baselines(cell.total_gpus):
        ok, _ = memory_feasible_hetero(cell.model_alias, tp, pp, in_len, out_len, n_req)
        if not ok:
            continue
        # Uniform per-stage layer counts.
        base = m.num_layers // pp
        extra = m.num_layers - base * pp
        layer_splits = [base + (1 if i < extra else 0) for i in range(pp)]
        # Uniform per-rank FFN / heads / KV shards.
        ffn_uniform = [m.intermediate_size // tp] * tp
        head_uniform = [m.num_q_heads // tp] * tp
        kv_uniform = [max(1, m.num_kv_heads // tp)] * tp
        env_uniform = {
            "TP": str(tp), "PP": str(pp),
            "VLLM_PP_LAYER_PARTITION": ",".join(str(x) for x in layer_splits),
            "AUTO_PP_LAYER_PARTITION": "0",
            "AUTO_TP_SPLIT": "0",
            "AUTOSPLIT": "0",
            "TP_HEAD_SPLITS": ",".join(str(x) for x in head_uniform),
            "TP_KV_SPLITS":   ",".join(str(x) for x in kv_uniform),
            "TP_FFN_SPLITS":  ",".join(str(x) for x in ffn_uniform),
        }
        runs.append(HeteroRun(
            cell=cell, config_id=f"TP{tp}PP{pp}_uniform",
            tp=tp, pp=pp, plan_type="uniform",
            layer_splits=layer_splits, tp_head_splits=head_uniform,
            tp_ffn_splits=ffn_uniform, tp_kv_splits=kv_uniform,
            env_vars=env_uniform,
            in_len=in_len, out_len=out_len, n_requests=n_req,
        ))
        seen_uniform.add((tp, pp))

    # Planner picks: top-3 distinct, restricted to (tp ≤ 4, pp ≤ 2) to avoid
    # the deep-PP / wide-TP cross-node init timeouts.
    distinct = []
    seen_keys: set[tuple] = set()
    for c in candidates:
        if c["pp"] > 2 or c["tp"] > 4:
            continue
        key = (c["tp"], c["pp"], tuple(c["layer_splits"]),
               tuple(c["tp_ffn_splits"][:c["tp"]]))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        distinct.append(c)
        if len(distinct) >= 3:
            break

    for i, c in enumerate(distinct):
        ok, _ = memory_feasible_hetero(cell.model_alias, c["tp"], c["pp"],
                                       in_len, out_len, n_req)
        if not ok:
            continue
        label = f"planner_top{i+1}"
        runs.append(HeteroRun(
            cell=cell, config_id=f"TP{c['tp']}PP{c['pp']}_{label}",
            tp=c["tp"], pp=c["pp"], plan_type=label,
            layer_splits=c["layer_splits"],
            tp_head_splits=c["tp_head_splits"],
            tp_ffn_splits=c["tp_ffn_splits"],
            tp_kv_splits=c["tp_kv_splits"],
            env_vars=c["env_vars"],
            predicted_wall_s=float(c["predicted_wall_s"]),
            weighted_cost_s=float(c["score"]["weighted_cost_s"]),
            in_len=in_len, out_len=out_len, n_requests=n_req,
        ))
    return runs


# ---------------------------------------------------------------------------
# Ray-backed vLLM launcher
# ---------------------------------------------------------------------------

def launch_vllm_ray(
    *, model: str, tp: int, pp: int,
    env_overrides: dict[str, str], log_path: Path,
    max_model_len: int, max_num_seqs: int, gpu_mem: float,
) -> ServerHandle:
    """Start vLLM api_server with --distributed-executor-backend ray.

    Ray will place workers across the connected head + worker nodes
    automatically. PP rank 0 → driver/head node by default, which matches
    our paper claim (fast stage on Blackwell)."""
    port = _free_port()
    cmd = [
        PY_HEAD, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--tensor-parallel-size", str(tp),
        "--pipeline-parallel-size", str(pp),
        "--distributed-executor-backend", "ray",
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--max-num-batched-tokens", str(max(2048, max_model_len)),
        "--gpu-memory-utilization", str(gpu_mem),
        "--dtype", "bfloat16",
        "--port", str(port),
        "--host", "0.0.0.0",
        "--async-scheduling",
        "--enable-chunked-prefill",
        "--enforce-eager",  # CUDA graph cross-node capture hangs; eager works.
    ]
    if "opt" in model.lower():
        tmpl = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}{{ message['content'] }}\n{% endif %}"
            "{% if message['role'] == 'user' %}{{ message['content'] }}\n{% endif %}"
            "{% if message['role'] == 'assistant' %}{{ message['content'] }}\n{% endif %}"
            "{% endfor %}"
        )
        cmd.extend(["--chat-template", tmpl])

    env = os.environ.copy()
    conda_bin = str(Path(CFG.head_py).parent)
    env["PATH"] = f"{conda_bin}:/usr/local/cuda-12.9/bin:" + env.get("PATH", "")
    env.setdefault("CC", "/usr/bin/gcc-12")
    env.setdefault("CXX", "/usr/bin/g++-12")
    env.setdefault("CUDAHOSTCXX", "/usr/bin/g++-12")
    env.setdefault("NVCC_CCBIN", "/usr/bin/g++-12")
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    env.setdefault("VLLM_HOST_IP", HEAD_IP)
    env.setdefault("RAY_ADDRESS", RAY_ADDRESS)
    env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    # Worker dl028 lacks g++-12 so flashinfer JIT compilation fails. Force the
    # FLASH_ATTN backend everywhere — it's prebuilt in the env.
    env.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env.setdefault("VLLM_USE_FLASHINFER_MOE", "0")
    # Cross-node NCCL: explicit IB transport. Both nodes have mlx5 HCA but
    # different interface names — list both. Without this NCCL can pick a
    # 1G management Ethernet and the engine hangs at PP group init.
    env.setdefault("NCCL_IB_HCA", "mlx5")
    env.setdefault("NCCL_IB_DISABLE", "0")
    env.setdefault("NCCL_NET_GDR_LEVEL", "2")
    env.setdefault("NCCL_SOCKET_IFNAME", f"{CFG.head_fabric_iface},{CFG.worker_fabric_iface}")
    env.setdefault("NCCL_DEBUG", "WARN")
    for k, v in env_overrides.items():
        env[str(k)] = str(v)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    flog = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=flog, stderr=subprocess.STDOUT,
        cwd=str(REPO), preexec_fn=os.setsid,
    )
    return ServerHandle(process=proc, port=port, log_path=log_path,
                        started_ts=time.time(), env_vars=env_overrides)


def run_benchmark(
    *, h: ServerHandle, model: str,
    n_requests: int, in_len: int, out_len: int,
    out_dir: Path, timeout_s: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / "prompt.txt"
    prompt_path.write_text(generate_prompt(in_len))
    runs_csv = out_dir / "perf_runs.csv"
    summary_csv = out_dir / "perf_summary.csv"
    perf_log = out_dir / "perf.log"

    cmd = [
        PY_HEAD, str(PERF),
        "--base-url", f"http://127.0.0.1:{h.port}/v1",
        "--model", model,
        "--prompt-file", str(prompt_path),
        "--requests", str(n_requests),
        "--runs", "1",
        "--max-tokens", str(out_len),
        "--ignore-eos",
        "--output-csv", str(runs_csv),
        "--output-summary-csv", str(summary_csv),
    ]
    env = os.environ.copy()
    env["PATH"] = f"{Path(CFG.head_py).parent}:" + env.get("PATH", "")
    with perf_log.open("w") as f:
        try:
            p = subprocess.run(cmd, env=env, cwd=str(REPO),
                               stdout=f, stderr=subprocess.STDOUT,
                               timeout=timeout_s, check=False)
            ok = p.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
    return parse_perf_summary(summary_csv) | {
        "perf_ok": ok,
        "perf_cmd": " ".join(shlex.quote(c) for c in cmd),
    }


def execute_run(r: HeteroRun, run_dir: Path, timeout_s: float) -> bool:
    MODEL_MAP = {
        "llama8b":   "meta-llama/Llama-3.1-8B-Instruct",
        "llama70b":  "meta-llama/Llama-3.3-70B-Instruct",
        "qwen3_32b": "Qwen/Qwen3-32B",
        "opt30b":    "facebook/opt-30b",
    }
    model_name = MODEL_MAP[r.cell.model_alias]
    log_path = run_dir / "vllm.log"

    env_overrides = dict(r.env_vars)
    env_overrides.setdefault("AUTO_TP_SPLIT", "0")
    env_overrides.setdefault("AUTOSPLIT", "0")
    env_overrides.setdefault("AUTO_PP_LAYER_PARTITION", "0")
    if r.pp > 1:
        # PP overlap patches verified working cross-node after the NCCL_SOCKET_IFNAME
        # fix; re-enable.
        env_overrides.setdefault("VLLM_PP_SAMPLED_BROADCAST_STREAM", "1")
        env_overrides.setdefault("VLLM_PP_MICROBATCH", "1")
        env_overrides.setdefault("VLLM_PP_MICROBATCH_SIZE",
                                 str(max(1, r.n_requests // r.pp)))
        env_overrides.setdefault("VLLM_PP_BATCH_QUEUE_SIZE", str(r.pp))
        env_overrides.setdefault("VLLM_PP_OVERLAP", "1")

    max_num_seqs = max(64, r.n_requests)
    max_model_len = 2048 if r.cell.model_alias == "opt30b" else 4096
    gpu_mem = 0.85

    try:
        h = launch_vllm_ray(
            model=model_name, tp=r.tp, pp=r.pp,
            env_overrides=env_overrides, log_path=log_path,
            max_model_len=max_model_len, max_num_seqs=max_num_seqs,
            gpu_mem=gpu_mem,
        )
    except Exception as exc:
        r.success = False; r.failure_reason = f"launch_failed:{exc}"
        return False

    try:
        ready = wait_for_ready(h, timeout_s=720.0)
        if not ready:
            r.success = False
            r.failure_reason = "server_did_not_become_ready"
            return False
        m = run_benchmark(
            h=h, model=model_name, n_requests=r.n_requests,
            in_len=r.in_len, out_len=r.out_len,
            out_dir=run_dir, timeout_s=min(timeout_s, 1800.0),
        )
        if not m.get("perf_ok"):
            r.success = False
            r.failure_reason = "perf_benchmark_failed"
            return False
        r.measured_tps = float(m.get("total_wall_throughput_tok_s")
                               or m.get("total_throughput_tok_s") or 0.0)
        r.measured_runtime_s = float(m.get("total_request_time_s") or 0.0)
        r.measured_ttft_ms = float(m.get("TTFT_ms_mean") or 0.0)
        r.measured_itl_ms = float(m.get("itl_ms_mean") or 0.0)
        r.success = True
        return True
    finally:
        # For Ray backend, the api_server holds workers across nodes.
        # SIGINT first; Ray actors should clean up automatically.
        stop_server(h, cuda_visible="0,1,2,3", grace_s=20.0)
        # Also remote-kill any VLLM:: stragglers on worker.
        subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", CFG.ssh_target,
             "pkill -9 -f 'VLLM::' 2>/dev/null; "
             "pkill -9 -f 'ray::RayWorkerWrapper' 2>/dev/null; true"],
            capture_output=True, check=False, timeout=20,
        )
        # Cross-node Ray: each failed launch can leak a placement group
        # that pins GPUs. Clean them all between runs.
        subprocess.run(
            [PY_HEAD, "-c",
             "import ray; ray.init(address='" + RAY_ADDRESS + "');"
             "[ray._private.worker.global_worker.core_worker.remove_placement_group("
             "ray.PlacementGroupID(bytes.fromhex(k))) "
             "for k in list(ray.util.placement_group_table().keys())]"],
            capture_output=True, check=False, timeout=30,
        )
        time.sleep(5)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="llama8b")
    ap.add_argument("--workloads", default="balanced,decode_heavy,prefill_heavy")
    ap.add_argument("--cells", default="1+1,2+2,4+4")
    ap.add_argument("--results-root", default=str(REPO / "results"), type=Path)
    ap.add_argument("--timeout-s", type=float, default=2400.0)
    args = ap.parse_args(argv)

    cells: list[HeteroCell] = []
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    wls = [w.strip() for w in args.workloads.split(",") if w.strip()]
    pairs = []
    for c in args.cells.split(","):
        h, w = c.strip().split("+")
        pairs.append((int(h), int(w)))
    for model in models:
        for wc in wls:
            for h, w in pairs:
                cells.append(HeteroCell(model_alias=model, workload_class=wc,
                                        head_n=h, worker_n=w))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = args.results_root / f"hetero_eval_{ts}"
    (out_root / "plans").mkdir(parents=True, exist_ok=True)
    (out_root / "runs").mkdir(parents=True, exist_ok=True)

    all_runs: list[HeteroRun] = []
    for cell in cells:
        print(f"\n========== CELL: {cell.label} ==========", flush=True)
        rec = build_planner_record(cell)
        (out_root / "plans" / f"{cell.label}.json").write_text(json.dumps(rec, indent=2))
        runs = runs_from_planner(cell, rec)
        print(f"  {len(runs)} configs to measure:")
        for r in runs:
            print(f"    {r.config_id}  pred_wall={r.predicted_wall_s:.2f}s")
        all_runs.extend(runs)

    for idx, r in enumerate(all_runs, start=1):
        print(f"\n[{idx}/{len(all_runs)}] {r.cell.label}  {r.config_id}", flush=True)
        run_dir = out_root / "runs" / f"{r.cell.label}__{r.config_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ok = execute_run(r, run_dir, args.timeout_s)
        rec = asdict(r)
        (run_dir / "record.json").write_text(json.dumps(rec, indent=2, default=str))
        print(f"  done: success={r.success} tps={r.measured_tps:.1f} "
              f"runtime={r.measured_runtime_s:.1f}s "
              f"reason={r.failure_reason!r}", flush=True)

    # Validation table
    import csv
    val_path = out_root / "validation.csv"
    by_cell: dict[str, list[HeteroRun]] = {}
    for r in all_runs:
        by_cell.setdefault(r.cell.label, []).append(r)
    with val_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell","top1_config","top1_tp","top1_pp","top1_tps",
                    "best_config","best_tp","best_pp","best_tps","same_tp_pp","gap_pct"])
        for label, rs in by_cell.items():
            succ = [r for r in rs if r.success]
            if not succ:
                w.writerow([label, "", 0, 0, "0.0", "", 0, 0, "0.0", "False", ""])
                continue
            top1 = next((r for r in rs if r.plan_type == "planner_top1"), None)
            best = max(succ, key=lambda r: r.measured_tps)
            top1_tps = top1.measured_tps if top1 and top1.success else 0.0
            same = (top1 is not None and top1.success
                    and (top1.tp, top1.pp) == (best.tp, best.pp))
            gap = ((best.measured_tps - top1_tps) / best.measured_tps * 100.0
                   if top1_tps > 0 else None)
            w.writerow([label, top1.config_id if top1 else "",
                        top1.tp if top1 else 0, top1.pp if top1 else 0,
                        f"{top1_tps:.2f}",
                        best.config_id, best.tp, best.pp,
                        f"{best.measured_tps:.2f}",
                        str(same), f"{gap:.2f}" if gap is not None else ""])
    print(f"\nResults dir: {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
