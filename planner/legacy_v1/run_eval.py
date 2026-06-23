"""End-to-end planner evaluation harness.

Given a list of (model, workload_class, n_gpus) cells:
  1. Run the planner to produce candidate configs (uniform sweep + planner pick).
  2. For each candidate, spin up vLLM with the right env vars + GPU subset.
  3. Run the existing perf/performance.py benchmark.
  4. Record metrics in CSV/JSON.
  5. Compare planner predicted rank vs measured rank.

This module DELIBERATELY does not try to be a Ray-cluster orchestrator. It
launches vLLM directly via `python -m vllm.entrypoints.openai.api_server`
on the local node (single-node grouping) using CUDA_VISIBLE_DEVICES to pick
the GPU subset. Multi-node measurements are out of scope for one session
but the planner predictions for hetero clusters are still emitted for the
paper table.

Output layout:
    results/planner_eval_<TS>/
        SUMMARY.md
        plans/<cell>.json          # planner record per cell
        runs/<cell>__<config>/     # each measured config
            vllm.log
            perf_summary.csv
            perf_runs.csv
            record.json            # the planner candidate + measured metrics
        validation.csv             # one row per (cell, config) with diff
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
from planner.cluster_env import CFG
PY = CFG.head_py
PERF = REPO / "perf" / "performance.py"

# Synthetic prompts at multiple lengths.
PROMPT_TEMPLATE = (
    "The following is a detailed analysis of large language model "
    "inference systems with a focus on heterogeneous GPU clusters, "
    "non-uniform tensor parallelism, pipeline parallelism, and the "
    "trade-offs between memory bandwidth, compute throughput, and "
    "communication overhead. The discussion covers cost-model-based "
    "planning that selects among TP-only, PP-only, and TP+PP hybrid "
    "configurations under workload classes such as decode-heavy, "
    "prefill-heavy, and balanced. "
)


def generate_prompt(target_tokens: int) -> str:
    """Build a prompt roughly long enough to tokenize to target_tokens.

    We assume ~1.3 tokens per word for Llama tokenizer. Adjust by re-padding
    if needed (cheap)."""
    words_needed = int(target_tokens / 1.3)
    base_words = PROMPT_TEMPLATE.split()
    out: list[str] = []
    while len(out) < words_needed:
        out.extend(base_words)
    return " ".join(out[:words_needed])


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

@dataclass
class ServerHandle:
    process: subprocess.Popen
    port: int
    log_path: Path
    started_ts: float
    env_vars: dict[str, str]


def _port_in_use(port: int) -> bool:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def _free_port(start: int = 28100) -> int:
    p = start
    while _port_in_use(p):
        p += 1
        if p > 29000:
            raise RuntimeError("no free port")
    return p


def _vllm_env(
    *,
    extra: dict[str, str] | None = None,
    cuda_visible: str | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    # Conda env's bin must be in PATH for `ninja` etc.
    conda_bin = str(Path(CFG.head_py).parent)
    env["PATH"] = f"{conda_bin}:/usr/local/cuda-12.9/bin:" + env.get("PATH", "")
    # Sane defaults that all our recent runs used.
    env.setdefault("CC", "/usr/bin/gcc-12")
    env.setdefault("CXX", "/usr/bin/g++-12")
    env.setdefault("CUDAHOSTCXX", "/usr/bin/g++-12")
    env.setdefault("NVCC_CCBIN", "/usr/bin/g++-12")
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
    env.setdefault("CUDA_PATH", "/usr/local/cuda-12.9")
    env.setdefault("CPATH", "/usr/local/cuda-12.9/include")
    env.setdefault("CPLUS_INCLUDE_PATH", "/usr/local/cuda-12.9/include")
    env.setdefault("C_INCLUDE_PATH", "/usr/local/cuda-12.9/include")
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    if cuda_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    if extra:
        for k, v in extra.items():
            env[str(k)] = str(v)
    return env


def launch_vllm(
    *,
    model: str,
    tp: int,
    pp: int,
    env_overrides: dict[str, str],
    cuda_visible: str,
    log_path: Path,
    max_model_len: int = 4096,
    max_num_seqs: int = 256,
    gpu_mem: float = 0.85,
    extra_args: list[str] | None = None,
) -> ServerHandle:
    """Start vllm OpenAI api_server on a free port. Returns handle."""
    port = _free_port()
    cmd = [
        PY, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--tensor-parallel-size", str(tp),
        "--pipeline-parallel-size", str(pp),
        "--distributed-executor-backend", "mp",
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--max-num-batched-tokens", str(max(2048, max_model_len)),
        "--gpu-memory-utilization", str(gpu_mem),
        "--dtype", "bfloat16",
        "--port", str(port),
        "--host", "0.0.0.0",
        "--async-scheduling",
        "--enable-chunked-prefill",
    ]
    if extra_args:
        cmd.extend(extra_args)
    # Some bare LMs (OPT) ship without a chat template; supply a minimal one
    # so /chat/completions works against the bench script.
    if "opt-" in model.lower() or "opt30b" in model.lower():
        tmpl = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}{{ message['content'] }}\n{% endif %}"
            "{% if message['role'] == 'user' %}{{ message['content'] }}\n{% endif %}"
            "{% if message['role'] == 'assistant' %}{{ message['content'] }}\n{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{% endif %}"
        )
        cmd.extend(["--chat-template", tmpl])

    env = _vllm_env(extra=env_overrides, cuda_visible=cuda_visible)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flog = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=flog, stderr=subprocess.STDOUT,
        cwd=str(REPO), preexec_fn=os.setsid,
    )
    return ServerHandle(
        process=proc, port=port, log_path=log_path,
        started_ts=time.time(), env_vars=env_overrides,
    )


def wait_for_ready(h: ServerHandle, timeout_s: float = 360.0) -> bool:
    """Poll the server log for 'Application startup complete' or fail-keywords."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if h.process.poll() is not None:
            return False  # exited
        try:
            txt = h.log_path.read_text(errors="ignore")
        except Exception:
            txt = ""
        if "Application startup complete" in txt:
            return True
        if any(k in txt for k in (
            "Engine core initialization failed",
            "raise RuntimeError",
            "AssertionError",
            "torch.OutOfMemoryError",
            "out of memory",
        )):
            return False
        time.sleep(5)
    return False


def _wait_gpu_clear(cuda_visible: str, max_wait_s: float = 90.0) -> bool:
    """Wait until all GPUs in cuda_visible show <1 GB used. Returns True if clear."""
    if not cuda_visible:
        return True
    ids = [x.strip() for x in cuda_visible.split(",") if x.strip()]
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used",
                 "--format=csv,noheader,nounits"],
                text=True,
            )
            used = {row.split(",")[0].strip(): int(row.split(",")[1].strip())
                    for row in out.strip().splitlines() if row}
            if all(used.get(i, 99999) < 1024 for i in ids):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def stop_server(h: ServerHandle, *, grace_s: float = 5.0,
                cuda_visible: str = "") -> None:
    if h.process.poll() is not None:
        # Process already exited; still pkill orphans + wait for GPU release.
        pass
    else:
        try:
            os.killpg(h.process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            h.process.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(h.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                h.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
    # Belt-and-suspenders: clean any orphaned api_server still using our port.
    subprocess.run(
        ["pkill", "-9", "-f", f"api_server.*--port {h.port}"],
        capture_output=True, check=False,
    )
    # Also nuke any VLLM::Worker processes that are still hogging GPU memory
    # on the visible GPUs (multiproc executor sometimes leaves children behind).
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid",
             "--format=csv,noheader"],
            text=True,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and "VLLM" in parts[1].upper():
                try:
                    os.kill(int(parts[0]), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
    except Exception:
        pass
    # Wait for actual GPU memory release on the GPUs we used.
    cleared = _wait_gpu_clear(cuda_visible, max_wait_s=90.0)
    if not cleared:
        # Final desperate try: pkill anything named VLLM.
        subprocess.run(["pkill", "-9", "-f", "VLLM::"], capture_output=True, check=False)
        time.sleep(5)


# ---------------------------------------------------------------------------
# Benchmark wrap
# ---------------------------------------------------------------------------

def run_benchmark(
    *,
    h: ServerHandle,
    model: str,
    n_requests: int,
    in_len: int,
    out_len: int,
    out_dir: Path,
    timeout_s: float = 1800.0,
) -> dict[str, Any]:
    """Run perf/performance.py against the server. Returns parsed metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = out_dir / "prompt.txt"
    prompt_path.write_text(generate_prompt(in_len))
    runs_csv = out_dir / "perf_runs.csv"
    summary_csv = out_dir / "perf_summary.csv"
    perf_log = out_dir / "perf.log"

    cmd = [
        PY, str(PERF),
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
    with perf_log.open("w") as f:
        try:
            p = subprocess.run(
                cmd, env=_vllm_env(), cwd=str(REPO),
                stdout=f, stderr=subprocess.STDOUT,
                timeout=timeout_s, check=False,
            )
            ok = p.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False

    return parse_perf_summary(summary_csv) | {
        "perf_ok": ok,
        "perf_cmd": " ".join(shlex.quote(c) for c in cmd),
    }


def parse_perf_summary(p: Path) -> dict[str, Any]:
    if not p.exists():
        return {}
    out: dict[str, Any] = {}
    try:
        for line in p.read_text().splitlines():
            if "," not in line:
                continue
            k, _, v = line.partition(",")
            k = k.strip(); v = v.strip()
            if k == "metric":
                continue
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    except Exception:
        return {}
    return out


# ---------------------------------------------------------------------------
# Candidate generator
# ---------------------------------------------------------------------------

def enumerate_uniform_baselines(world: int) -> list[tuple[int, int]]:
    """Sweep feasible uniform (TP, PP) such that TP*PP = world."""
    out = []
    for tp in (1, 2, 4, 8):
        if world % tp == 0:
            pp = world // tp
            if pp >= 1 and pp <= 8:
                out.append((tp, pp))
    return out


# ---------------------------------------------------------------------------
# Memory feasibility quick check before launching
# ---------------------------------------------------------------------------

def quick_memory_check(
    model_alias: str, tp: int, pp: int, in_len: int, out_len: int,
    n_req: int, vram_GB: float = 96.0,
) -> tuple[bool, str]:
    """Coarse: does (weights + KV) fit in min(vram)?"""
    from .model_spec import get_model
    from .scorer import stage_param_bytes, kv_cache_bytes_per_rank
    m = get_model(model_alias)
    n_per_stage = max(1, m.num_layers // pp)
    w = stage_param_bytes(m, n_per_stage, tp) / 1e9
    kv = kv_cache_bytes_per_rank(
        m, n_layers_in_stage=n_per_stage, tp_size=tp,
        max_concurrent_seqs=n_req, seq_len=in_len + out_len,
    ) / 1e9
    total = w + kv + 4.0  # activations + slack
    if total > vram_GB:
        return False, f"need {total:.1f} GB > VRAM {vram_GB:.0f} GB " \
                      f"(w={w:.1f}, kv={kv:.1f})"
    return True, f"need {total:.1f} GB ≤ VRAM {vram_GB:.0f} GB"


# ---------------------------------------------------------------------------
# Workload class -> concrete (in_len, out_len, n_requests) per model size
# ---------------------------------------------------------------------------

def workload_settings(
    workload_class: str, model_alias: str, world: int,
) -> tuple[int, int, int]:
    """Concrete workload params per (workload class, model size, world size).

    The user's directive: avoid runs that finish in <10 seconds. We size
    `n_requests` so even a 4-GPU TP=4 8B run (which can hit ~4000 tok/s
    measured) generates enough work to last 30-60 seconds. That means
    LOTS of requests for small models and decode-heavy workloads.

    Reference: 8B TP=4 ~4000 tok/s → need ~120K tokens / 30s.
              At decode_heavy in=128 out=512=640 per req → ~190 reqs/30s.
              At balanced  in=512 out=256=768 per req → ~155 reqs/30s.
              At prefill_h in=1024 out=64=1088 per req → ~110 reqs/30s.
    """
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

    # OPT-30B is dense MHA (no GQA), so KV cache is ~5× larger per token than
    # GQA models. For prefill_heavy (long input), the default `n_req` blows up
    # KV memory. Shorten input + halve request count to keep the workload class
    # comparable but feasible on 96 GB Blackwell.
    if model_alias == "opt30b" and workload_class == "prefill_heavy":
        in_len, out_len = 512, 64
        n_req = 64 * max(1, world // 2)
        return in_len, out_len, n_req

    # Per-class baseline scaled for ~30-60s runtime at the per-class TP=4 rate.
    # prefill_heavy has tiny output (64 tok) so the throughput-by-output metric
    # is artificially low — but wall time is what matters for comparison. We
    # still bump n_req hard for prefill_heavy because each request finishes
    # very fast at decoding time, so we need a lot of them to get >10s wall.
    base_req_by_size = {
        # (small, medium, huge) for each workload class
        "decode_heavy":  {"small": 192, "medium": 96,  "huge": 48},
        "prefill_heavy": {"small": 512, "medium": 256, "huge": 96},
        "balanced":      {"small": 160, "medium": 80,  "huge": 40},
    }[workload_class]
    base_req = base_req_by_size[size_label]
    # Scale UP with world size for small models (more GPUs = more throughput).
    # Hold for huge (KV cache memory is the binding constraint).
    if size_label == "small":
        n_req = base_req * max(1, world // 2)
    elif size_label == "medium":
        n_req = base_req * max(1, max(1, world) // 2)
    else:
        n_req = base_req
    return in_len, out_len, n_req


# ---------------------------------------------------------------------------
# Eval driver
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    model_alias: str
    workload_class: str
    n_gpus: int
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.model_alias}_{self.workload_class}_{self.n_gpus}gpu"


@dataclass
class Run:
    cell: Cell
    config_id: str        # "TP4PP1_uniform" / "TP2PP2_planner" / ...
    tp: int
    pp: int
    plan_type: str        # "uniform" / "planner_top1" / "planner_topK"
    layer_splits: list[int]
    tp_head_splits: list[int]
    tp_ffn_splits: list[int]
    tp_kv_splits: list[int]
    env_vars: dict[str, str]
    predicted_wall_s: float
    weighted_cost_s: float
    in_len: int
    out_len: int
    n_requests: int
    # Filled after run:
    measured_tps: float = 0.0
    measured_runtime_s: float = 0.0
    measured_ttft_ms: float = 0.0
    measured_itl_ms: float = 0.0
    success: bool = False
    failure_reason: str = ""
    perf_summary_path: str = ""


def build_planner_record_for_cell(cell: Cell) -> dict[str, Any]:
    """Run the planner CLI in-process for a cell. Cluster is N Blackwell on
    a single node (head)."""
    from .cli import build_record
    return build_record(
        model_alias=cell.model_alias,
        workload_class=cell.workload_class,
        cluster_spec_str=f"head:RTX-PRO-Blackwell:{cell.n_gpus}",
        network="PCIE_GEN5",
        cross_network=None,
        top_k=12,
    )


def runs_from_planner(
    cell: Cell, planner_record: dict[str, Any]
) -> list[Run]:
    """For each cell, build the run list:
      - All feasible uniform (TP, PP) at this world size
      - Planner top-1
      - Planner top-2 (if different from top-1 and uniform)
      - Planner top-3 (if different from above)
    """
    in_len, out_len, n_req = workload_settings(
        cell.workload_class, cell.model_alias, cell.n_gpus,
    )
    selected = planner_record["selected"]
    candidates = planner_record["candidates"]
    # Map (tp, pp, tuple(layers), tuple(ffn[:tp])) -> candidate for lookup.
    cand_by_key: dict[tuple, dict[str, Any]] = {}
    for c in candidates:
        key = (c["tp"], c["pp"],
               tuple(c["layer_splits"]),
               tuple(c["tp_ffn_splits"][:c["tp"]]))
        cand_by_key.setdefault(key, c)

    runs: list[Run] = []

    # Uniform sweep at this world.
    for tp, pp in enumerate_uniform_baselines(cell.n_gpus):
        ok, why = quick_memory_check(
            cell.model_alias, tp, pp, in_len, out_len, n_req,
        )
        if not ok:
            continue
        # Build env_vars for uniform plan: no overrides (rely on vLLM defaults).
        uniform_env = {
            "TP": str(tp), "PP": str(pp),
            "AUTO_PP_LAYER_PARTITION": "0",
            "AUTO_TP_SPLIT": "0",
        }
        runs.append(Run(
            cell=cell, config_id=f"TP{tp}PP{pp}_uniform",
            tp=tp, pp=pp, plan_type="uniform",
            layer_splits=[], tp_head_splits=[], tp_ffn_splits=[], tp_kv_splits=[],
            env_vars=uniform_env,
            predicted_wall_s=-1.0, weighted_cost_s=-1.0,
            in_len=in_len, out_len=out_len, n_requests=n_req,
        ))

    # Planner top-3 candidates (deduped against uniform configs).
    seen_uniform: set[tuple[int, int]] = {(r.tp, r.pp) for r in runs}
    planner_picks: list[tuple[str, dict[str, Any]]] = []
    for i, c in enumerate(candidates[:5]):
        plan_label = f"planner_top{i+1}"
        planner_picks.append((plan_label, c))

    for label, c in planner_picks:
        ok, why = quick_memory_check(
            cell.model_alias, c["tp"], c["pp"], in_len, out_len, n_req,
        )
        if not ok:
            continue
        # If this candidate is identical to a uniform pick AND has uniform
        # splits, we can skip (it would just rerun the same config).
        uniform_match = (c["tp"], c["pp"]) in seen_uniform
        ffn_uniform = all(x == c["tp_ffn_splits"][0]
                          for x in c["tp_ffn_splits"][:c["tp"]])
        layers_uniform = (
            len(set(c["layer_splits"])) == 1 or
            (max(c["layer_splits"]) - min(c["layer_splits"])) <= 1
        )
        if uniform_match and ffn_uniform and layers_uniform and label != "planner_top1":
            continue
        runs.append(Run(
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
        if label == "planner_top1":
            break
    # Add the planner's selected if not yet present.
    if selected is not None:
        sel_key = (selected["tp"], selected["pp"], "planner_selected")
        if not any(r.config_id.startswith(f"TP{selected['tp']}PP{selected['pp']}_planner_top1")
                   for r in runs):
            ok, _ = quick_memory_check(
                cell.model_alias, selected["tp"], selected["pp"],
                in_len, out_len, n_req,
            )
            if ok:
                runs.append(Run(
                    cell=cell, config_id=f"TP{selected['tp']}PP{selected['pp']}_planner_selected",
                    tp=selected["tp"], pp=selected["pp"], plan_type="planner_selected",
                    layer_splits=selected["layer_splits"],
                    tp_head_splits=selected["tp_head_splits"],
                    tp_ffn_splits=selected["tp_ffn_splits"],
                    tp_kv_splits=selected["tp_kv_splits"],
                    env_vars=selected["env_vars"],
                    predicted_wall_s=float(selected["predicted_wall_s"]),
                    weighted_cost_s=float(selected["score"]["weighted_cost_s"]),
                    in_len=in_len, out_len=out_len, n_requests=n_req,
                ))
    return runs


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    cells: list[Cell] = field(default_factory=list)
    results_root: Path = Path("results")
    cuda_visible_pool: str = "0,1,2,3"
    skip_run: bool = False
    timeout_s_per_run: float = 1800.0
    only_planner_picks: bool = False


def run_eval(cfg: EvalConfig) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = cfg.results_root / f"planner_eval_{ts}"
    (out_root / "plans").mkdir(parents=True, exist_ok=True)
    (out_root / "runs").mkdir(parents=True, exist_ok=True)
    summary_lines: list[str] = []

    all_runs: list[Run] = []
    for cell in cfg.cells:
        print(f"\n========== CELL: {cell.label} ==========", flush=True)
        plan_record = build_planner_record_for_cell(cell)
        (out_root / "plans" / f"{cell.label}.json").write_text(
            json.dumps(plan_record, indent=2)
        )
        runs = runs_from_planner(cell, plan_record)
        if cfg.only_planner_picks:
            runs = [r for r in runs if r.plan_type != "uniform"]
        print(f"  {len(runs)} configs to measure:")
        for r in runs:
            print(f"    {r.config_id}  pred_wall={r.predicted_wall_s:.2f}s")
        all_runs.extend(runs)

    # Dry-run path: write the plan + intended runs, no measurement.
    if cfg.skip_run:
        _write_plan_only_summary(out_root, cfg.cells, all_runs)
        return out_root

    # Pick GPUs from the pool: first N of the pool.
    gpu_pool = [int(x) for x in cfg.cuda_visible_pool.split(",")]

    for idx, r in enumerate(all_runs, start=1):
        world = r.cell.n_gpus
        gpus = ",".join(str(x) for x in gpu_pool[:world])
        print(f"\n[{idx}/{len(all_runs)}] {r.cell.label}  {r.config_id}  GPUs={gpus}",
              flush=True)
        run_dir = out_root / "runs" / f"{r.cell.label}__{r.config_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ok = _execute_single_run(r, run_dir, gpus, cfg.timeout_s_per_run)
        # Persist record (whether or not measurement succeeded).
        rec = asdict(r)
        # cell is a Cell object; dataclass conversion handles it.
        (run_dir / "record.json").write_text(json.dumps(rec, indent=2,
                                                       default=str))
        print(f"  done: success={r.success} tps={r.measured_tps:.1f} "
              f"runtime={r.measured_runtime_s:.1f}s "
              f"reason={r.failure_reason!r}", flush=True)

    _write_full_summary(out_root, cfg.cells, all_runs)
    return out_root


def _execute_single_run(
    r: Run, run_dir: Path, cuda_visible: str, timeout_s: float,
) -> bool:
    model_name = {
        "llama8b":   "meta-llama/Llama-3.1-8B-Instruct",
        "llama70b":  "meta-llama/Llama-3.3-70B-Instruct",
        "qwen3_32b": "Qwen/Qwen3-32B",
        "opt30b":    "facebook/opt-30b",
    }[r.cell.model_alias]
    log_path = run_dir / "vllm.log"

    # Merge run-specific env. Always enable PP-overlap features when PP>1.
    env_overrides = dict(r.env_vars)
    # Avoid the launcher's auto-helpers stomping on the planner's choice.
    env_overrides.setdefault("AUTO_TP_SPLIT", "0")
    env_overrides.setdefault("AUTOSPLIT", "0")
    env_overrides.setdefault("AUTO_PP_LAYER_PARTITION", "0")
    # Activate the M13 broadcast fix + microbatch when PP > 1.
    if r.pp > 1:
        env_overrides.setdefault("VLLM_PP_SAMPLED_BROADCAST_STREAM", "1")
        env_overrides.setdefault("VLLM_PP_MICROBATCH", "1")
        env_overrides.setdefault("VLLM_PP_MICROBATCH_SIZE",
                                 str(max(1, r.n_requests // r.pp)))
        env_overrides.setdefault("VLLM_PP_BATCH_QUEUE_SIZE", str(r.pp))
        env_overrides.setdefault("VLLM_PP_OVERLAP", "1")

    # Pick max_num_seqs >= n_requests.
    max_num_seqs = max(256, r.n_requests)
    # Llama-3.x supports 4096+ but cap to avoid KV bloat. OPT capped at 2048.
    max_model_len = 2048 if r.cell.model_alias == "opt30b" else 4096

    # quick recheck — should not OOM thanks to memory_check upstream.
    try:
        h = launch_vllm(
            model=model_name, tp=r.tp, pp=r.pp,
            env_overrides=env_overrides, cuda_visible=cuda_visible,
            log_path=log_path,
            max_model_len=max_model_len, max_num_seqs=max_num_seqs,
            gpu_mem=0.85,
        )
    except Exception as exc:
        r.success = False; r.failure_reason = f"launch_failed:{exc}"
        return False

    try:
        ready = wait_for_ready(h, timeout_s=360.0)
        if not ready:
            r.success = False
            r.failure_reason = "server_did_not_become_ready (see vllm.log)"
            return False

        # Run benchmark.
        m = run_benchmark(
            h=h, model=model_name,
            n_requests=r.n_requests,
            in_len=r.in_len, out_len=r.out_len,
            out_dir=run_dir,
            timeout_s=min(timeout_s, 1800.0),
        )
        if not m.get("perf_ok"):
            r.success = False
            r.failure_reason = "perf_benchmark_failed"
            return False

        # Map metrics into record.
        r.measured_tps = float(m.get("total_wall_throughput_tok_s")
                               or m.get("total_throughput_tok_s") or 0.0)
        r.measured_runtime_s = float(m.get("total_request_time_s") or 0.0)
        r.measured_ttft_ms = float(m.get("TTFT_ms_mean") or 0.0)
        r.measured_itl_ms = float(m.get("itl_ms_mean") or 0.0)
        r.success = True
        r.perf_summary_path = str(run_dir / "perf_summary.csv")
        return True
    finally:
        stop_server(h, cuda_visible=cuda_visible)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def _write_plan_only_summary(out_root: Path, cells: list[Cell], runs: list[Run]) -> None:
    lines: list[str] = []
    lines.append(f"# Planner eval (dry-run): {out_root.name}")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## Cells\n")
    for c in cells:
        lines.append(f"- **{c.label}**: model={c.model_alias} "
                     f"workload={c.workload_class} world={c.n_gpus}")
    lines.append("\n## Intended runs\n")
    by_cell: dict[str, list[Run]] = {}
    for r in runs:
        by_cell.setdefault(r.cell.label, []).append(r)
    for label, rs in by_cell.items():
        lines.append(f"\n### {label}\n")
        lines.append("| config | TP | PP | layers | ffn[:tp] | pred_wall_s | wcost_s |")
        lines.append("|---|---:|---:|---|---|---:|---:|")
        for r in rs:
            lines.append(f"| {r.config_id} | {r.tp} | {r.pp} | "
                         f"{r.layer_splits} | {r.tp_ffn_splits[:r.tp]} | "
                         f"{r.predicted_wall_s:.2f} | {r.weighted_cost_s:.2f} |")
    (out_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"\nDry-run summary written to {out_root}/SUMMARY.md", flush=True)


def _write_full_summary(out_root: Path, cells: list[Cell], runs: list[Run]) -> None:
    import csv

    # Validation: planner top-1 vs measured best
    by_cell: dict[str, list[Run]] = {}
    for r in runs:
        by_cell.setdefault(r.cell.label, []).append(r)

    validation_csv = out_root / "validation.csv"
    with validation_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "cell", "n_configs", "n_succeeded",
            "planner_top1_config", "planner_top1_tps",
            "measured_best_config", "measured_best_tps",
            "top1_match", "rel_gap_pct",
        ])
        cell_validation: list[dict[str, Any]] = []
        for label, rs in by_cell.items():
            succ = [r for r in rs if r.success]
            if not succ:
                w.writerow([label, len(rs), 0, "", 0, "", 0, "False", ""])
                continue
            top1 = next((r for r in rs if r.plan_type == "planner_top1"), None)
            best = max(succ, key=lambda r: r.measured_tps)
            top1_tps = top1.measured_tps if top1 and top1.success else 0.0
            match = (top1 is not None and top1.success and top1.config_id == best.config_id)
            gap = (best.measured_tps - top1_tps) / best.measured_tps * 100.0 if (
                top1_tps > 0 and best.measured_tps > 0) else None
            w.writerow([
                label, len(rs), len(succ),
                top1.config_id if top1 else "",
                f"{top1_tps:.2f}",
                best.config_id, f"{best.measured_tps:.2f}",
                str(match), f"{gap:.2f}" if gap is not None else "",
            ])
            cell_validation.append({
                "label": label, "match": match,
                "top1_config": top1.config_id if top1 else None,
                "top1_tps": top1_tps,
                "best_config": best.config_id,
                "best_tps": best.measured_tps,
                "rel_gap_pct": gap,
                "n_configs": len(rs),
                "n_succeeded": len(succ),
            })

    # SUMMARY.md
    lines: list[str] = []
    lines.append(f"# Planner eval: {out_root.name}")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## Implementation\n")
    lines.append("- Workload-aware planner: `vllm_main/planner/workload.py` + "
                 "`vllm_main/planner/scorer.py`")
    lines.append("- CLI: `python -m vllm_main.planner.cli ...`")
    lines.append("- Eval driver: `python -m vllm_main.planner.run_eval ...`")
    lines.append("- Env vars: `VLLM_PLANNER_WORKLOAD`, `VLLM_PLANNER_INPUT_LEN`, "
                 "`VLLM_PLANNER_OUTPUT_LEN`, `VLLM_PLANNER_NUM_PROMPTS`, "
                 "`VLLM_PLANNER_CONCURRENT_REQUESTS`, "
                 "`VLLM_PLANNER_PREFILL_WEIGHT`, `VLLM_PLANNER_DECODE_WEIGHT`")
    lines.append("\n## Cost-model assumptions\n")
    lines.append("- Roofline matmul time = max(compute, memory). Per-(K,N) "
                 "calibrated lookup when available.")
    lines.append("- TP AllReduce = ring busbw model with per-tensor latency floor.")
    lines.append("- PP send = point-to-point.")
    lines.append("- Workload classes weight prefill vs decode + adjust "
                 "TP comm and PP depth penalties.")
    lines.append("- PP overlap fraction predicted as 0.65 at PP=2, slowly "
                 "decaying with depth (calibrated against M13 measurements).")
    lines.append("- Memory feasibility: rejects plans where per-rank "
                 "(weights + KV cache + activations) > min cluster VRAM.")
    lines.append("\n## Cells\n")
    for c in cells:
        lines.append(f"- **{c.label}**: model={c.model_alias} "
                     f"workload={c.workload_class} world={c.n_gpus}")
    lines.append("\n## Results\n")
    for label, rs in by_cell.items():
        lines.append(f"\n### {label}\n")
        lines.append("| config | TP | PP | layers | ffn[:tp] | pred_wall_s | wcost_s | "
                     "TTFT_ms | tok/s | runtime_s | success |")
        lines.append("|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|")
        # Sort by measured tps desc among successes, fail at bottom.
        rs_sorted = sorted(rs, key=lambda r: (-r.success, -r.measured_tps))
        for r in rs_sorted:
            lines.append(f"| {r.config_id} | {r.tp} | {r.pp} | {r.layer_splits} | "
                         f"{r.tp_ffn_splits[:r.tp]} | {r.predicted_wall_s:.2f} | "
                         f"{r.weighted_cost_s:.2f} | "
                         f"{r.measured_ttft_ms:.1f} | "
                         f"{r.measured_tps:.1f} | "
                         f"{r.measured_runtime_s:.1f} | "
                         f"{'Y' if r.success else 'N: '+r.failure_reason} |")
    lines.append("\n## Planner validation\n")
    lines.append("| cell | n_configs | n_succeeded | top1_config | top1_tok/s | "
                 "measured_best | best_tok/s | top1_match | rel_gap_pct |")
    lines.append("|---|---:|---:|---|---:|---|---:|---|---:|")
    for v in cell_validation:
        gap_str = f"{v['rel_gap_pct']:.2f}" if v['rel_gap_pct'] is not None else "—"
        lines.append(
            f"| {v['label']} | {v['n_configs']} | {v['n_succeeded']} | "
            f"{v['top1_config']} | {v['top1_tps']:.1f} | "
            f"{v['best_config']} | {v['best_tps']:.1f} | "
            f"{'Y' if v['match'] else 'N'} | {gap_str} |"
        )
    lines.append("\n## Next steps\n")
    lines.append("- Extend with multi-node Ray launches for true hetero (head+worker) clusters.")
    lines.append("- Add Qwen3 / OPT runs once Llama eval is complete.")
    lines.append("- Re-calibrate `predicted_pp_overlap_fraction` from measured overlap.")
    (out_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_root}/SUMMARY.md and {validation_csv.name}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="llama8b",
                    help="CSV of model aliases (llama8b,llama70b,qwen3_32b,opt30b)")
    ap.add_argument("--workloads", default="balanced,decode_heavy,prefill_heavy",
                    help="CSV of workload classes")
    ap.add_argument("--gpus", default="1,2,4",
                    help="CSV of world sizes to run")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only run the planner and write SUMMARY.md, no measurements.")
    ap.add_argument("--only-planner-picks", action="store_true",
                    help="Skip uniform baselines; only run planner top-K configs.")
    ap.add_argument("--cuda-pool", default="0,1,2,3",
                    help="Comma-separated GPU IDs to consume from.")
    ap.add_argument("--timeout-s", type=float, default=1800.0)
    ap.add_argument("--results-root", default="results", type=Path)
    args = ap.parse_args(argv)

    cells: list[Cell] = []
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    gpu_list = [int(x) for x in args.gpus.split(",") if x.strip()]
    for model in models:
        for wc in workloads:
            for n in gpu_list:
                cells.append(Cell(model_alias=model, workload_class=wc, n_gpus=n))

    cfg = EvalConfig(
        cells=cells,
        results_root=args.results_root,
        cuda_visible_pool=args.cuda_pool,
        skip_run=args.dry_run,
        timeout_s_per_run=args.timeout_s,
        only_planner_picks=args.only_planner_picks,
    )
    out = run_eval(cfg)
    print(f"\nResults dir: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
