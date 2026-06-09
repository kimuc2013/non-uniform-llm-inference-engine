#!/usr/bin/env python3
"""Measure vLLM streaming latency metrics (throughput, TTFT, TPOT, ITL).

Example:
  python measure_token_metrics.py \
    --base-url http://localhost:8000/v1 \
    --api-key EMPTY \
        --prompt "아기다리고기다리던방학" \
        --runs 5

Tip:
    If you set VLLM_BASE_URL and keep a prompt file (default: prompts/prompt.txt),
    you can run with only --requests/--runs/--max-tokens.
    Use --model or --model-preset to switch model profile in one script.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics as stats
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests


# ==========================
# Defaults (edit here)
# ==========================
# Run `python measure_token_metrics.py` with these defaults.
BASE_DIR = Path(__file__).resolve().parent


def _load_json_object(path: str | Path | None) -> Dict[str, object]:
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists() or not json_path.is_file():
        return {}
    try:
        parsed = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _resolve_default_run_meta_file() -> str | None:
    env_path = os.getenv("VLLM_RUN_META_FILE", "").strip()
    if env_path:
        return env_path
    default_path = BASE_DIR / "meta" / "last_run_meta.json"
    if default_path.is_file():
        return str(default_path)
    return None


DEFAULT_RUN_META_FILE = _resolve_default_run_meta_file()


def _resolve_default_base_url() -> str:
    env_url = os.getenv("VLLM_BASE_URL", "").strip()
    if env_url:
        return env_url

    raw_port = os.getenv("PORT", "").strip()
    if raw_port:
        return f"http://127.0.0.1:{raw_port}/v1"

    run_meta = _load_json_object(DEFAULT_RUN_META_FILE)
    raw_host = str(run_meta.get("host", "")).strip()
    raw_port = str(run_meta.get("port", "")).strip()
    host = "127.0.0.1"
    if raw_host and raw_host not in {"0.0.0.0", "::"}:
        host = raw_host
    if raw_port:
        return f"http://{host}:{raw_port}/v1"

    port = "8000"
    return f"http://127.0.0.1:{port}/v1"


def _resolve_default_model() -> str:
    env_model = os.getenv("VLLM_MODEL", os.getenv("MODEL", "")).strip()
    if env_model:
        if Path(env_model).is_absolute() and not Path(env_model).exists():
            env_model = ""
        else:
            return env_model

    run_meta = _load_json_object(DEFAULT_RUN_META_FILE)
    meta_model = str(run_meta.get("model", "")).strip()
    if meta_model:
        if Path(meta_model).is_absolute() and not Path(meta_model).exists():
            meta_model = ""
        else:
            return meta_model

    return "meta-llama/Llama-3.3-70B-Instruct"


_MODEL_SIZE_B_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*[Bb]\b")


def _infer_model_size_b_from_name(model: str) -> float:
    matches = _MODEL_SIZE_B_PATTERN.findall(str(model))
    if not matches:
        return 0.0
    try:
        return max(float(m) for m in matches)
    except Exception:
        return 0.0


def _infer_model_size_b_from_config(model: str) -> float:
    try:
        from transformers import AutoConfig
    except Exception:
        return 0.0

    try:
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=True)
    except Exception:
        return 0.0

    # Some configs expose direct parameter count.
    for attr in ("num_parameters", "total_num_params", "n_params"):
        raw = getattr(cfg, attr, None)
        try:
            value = float(raw)
        except Exception:
            value = 0.0
        if value > 0:
            return value / 1e9

    # Approximate decoder-only parameter count from config dimensions.
    def _int_attr(*names: str) -> int:
        for name in names:
            raw = getattr(cfg, name, None)
            try:
                value = int(raw)
            except Exception:
                value = 0
            if value > 0:
                return value
        return 0

    layers = _int_attr("num_hidden_layers", "n_layer")
    hidden = _int_attr("hidden_size", "n_embd", "d_model")
    inter = _int_attr("intermediate_size", "ffn_dim")
    vocab = _int_attr("vocab_size")
    if not (layers > 0 and hidden > 0 and inter > 0):
        return 0.0

    act = str(getattr(cfg, "hidden_act", "")).lower()
    gated = any(token in act for token in ("swiglu", "geglu", "silu"))
    mlp_mul = 3 if gated else 2
    per_layer = 4 * hidden * hidden + mlp_mul * hidden * inter
    embeddings = vocab * hidden if vocab > 0 else 0
    total_params = per_layer * layers + embeddings
    return float(total_params) / 1e9


def _infer_model_size_b(model: str) -> float:
    by_name = _infer_model_size_b_from_name(model)
    if by_name > 0:
        return by_name
    return _infer_model_size_b_from_config(model)


def _infer_perf_defaults(model: str) -> tuple[int, float]:
    model_b = _infer_model_size_b(model)
    if model_b >= 300:
        return 512, 250.0
    if model_b >= 150:
        return 768, 350.0
    if model_b >= 60:
        return 2048, 1000.0
    if model_b > 0:
        return 2048, 1400.0
    # Unknown model size: pick conservative timeout slope.
    return 1024, 600.0


DEFAULT_BASE_URL = _resolve_default_base_url()
DEFAULT_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
DEFAULT_MODEL = _resolve_default_model()
DEFAULT_MODEL_MAX_TOKENS, DEFAULT_MODEL_ASSUMED_TOK_S = _infer_perf_defaults(DEFAULT_MODEL)
MODEL_PRESETS: dict[str, str] = {
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama70b": "meta-llama/Llama-3.3-70B-Instruct",
    "llama405b": "meta-llama/Llama-3.1-405B-Instruct-FP8",
    "qwen235b": "Qwen/Qwen3-235B-A22B",
}
DEFAULT_PROMPT_FILE = os.getenv("VLLM_PROMPT_FILE", str(BASE_DIR / "prompts" / "prompt.txt"))
DEFAULT_PROMPT = os.getenv(
    "VLLM_PROMPT",
    "아래 내용을 5줄로 요약하고, 핵심 키워드 10개를 뽑아줘.",
)

DEFAULT_REQUESTS = int(os.getenv("VLLM_REQUESTS", "1"))
DEFAULT_RUNS = int(os.getenv("VLLM_RUNS", "1"))
DEFAULT_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", str(DEFAULT_MODEL_MAX_TOKENS)))
DEFAULT_TEMPERATURE = float(os.getenv("VLLM_TEMPERATURE", "0.0"))
DEFAULT_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "1200.0"))
DEFAULT_TOKENIZER = os.getenv("VLLM_TOKENIZER")  # optional
DEFAULT_METRICS_URL = os.getenv("VLLM_METRICS_URL", "auto")
DEFAULT_INCLUDE_USAGE = os.getenv("VLLM_INCLUDE_USAGE", "1") not in ("0", "false", "False")
DEFAULT_MIN_TOKENS = int(os.getenv("VLLM_MIN_TOKENS", "0"))
DEFAULT_TP_HEAD_SPLITS = os.getenv(
    "VLLM_TP_HEAD_SPLITS",
    os.getenv("TP_HEAD_SPLITS", ""),
)
DEFAULT_TP_FFN_SPLITS = os.getenv(
    "VLLM_TP_FFN_SPLITS",
    os.getenv("TP_FFN_SPLITS", ""),
)
DEFAULT_ASSUMED_TOK_S = float(os.getenv("VLLM_ASSUMED_TOK_S", str(DEFAULT_MODEL_ASSUMED_TOK_S)))
DEFAULT_APPEND_SUMMARY_CSV = os.getenv("VLLM_APPEND_SUMMARY_CSV", "").strip() or None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream from OpenAI-compatible endpoint and emit latency metrics")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL to the OpenAI-compatible API, e.g. http://HOST:PORT/v1 (or env VLLM_BASE_URL)",
    )
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key header value (vLLM ignores it, but OpenAI SDK requires it)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to request")
    parser.add_argument(
        "--model-preset",
        choices=sorted(MODEL_PRESETS.keys()),
        default=None,
        help="Convenience preset for --model.",
    )
    parser.add_argument("--prompt", default=None, help="Prompt content to send in every request")
    parser.add_argument(
        "--prompt-file",
        default=DEFAULT_PROMPT_FILE,
        help="Read prompt content from a file (or env VLLM_PROMPT_FILE). Default: prompts/prompt.txt",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=DEFAULT_REQUESTS,
        help="How many requests to send per run (also used as concurrency)",
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="How many times to repeat the request batch")
    parser.add_argument("--max-tokens", type=int, default=256, help="Generation cap per request")
    parser.set_defaults(max_tokens=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature for decoding")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout per request (seconds). Auto-scales if VLLM_TIMEOUT/--timeout is unset.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-request detailed metrics (default: only one-line progress)",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="Tokenizer name/path for prompt token counting (defaults to --model)",
    )
    parser.add_argument(
        "--metrics-url",
        default=DEFAULT_METRICS_URL,
        help=(
            "Prometheus metrics URL for vLLM (e.g. http://localhost:8000/metrics). "
            "Use 'auto' to derive from --base-url. "
            "If set, the script will capture a before/after snapshot and summarize "
            "queue/prefill/decode/inference times."
        ),
    )
    parser.add_argument(
        "--metrics-timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="HTTP timeout for metrics scraping (seconds).",
    )
    parser.add_argument(
        "--include-usage",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_INCLUDE_USAGE,
        help="Include usage stats in streaming responses (vLLM supports this).",
    )
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore EOS and keep generating until max_tokens.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=DEFAULT_MIN_TOKENS,
        help="Minimum output tokens to generate before stopping.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Write per-run metrics to a CSV file.",
    )
    parser.add_argument(
        "--output-summary-csv",
        default=None,
        help="Write summary metrics to a CSV file (key,value).",
    )
    parser.add_argument(
        "--append-summary-csv",
        default=DEFAULT_APPEND_SUMMARY_CSV,
        help=(
            "Append one wide summary row per execution to a CSV file "
            "(env VLLM_APPEND_SUMMARY_CSV)."
        ),
    )
    parser.add_argument(
        "--tp-head-splits",
        default=DEFAULT_TP_HEAD_SPLITS,
        help="TP head shard splits for reporting (comma-separated; env VLLM_TP_HEAD_SPLITS).",
    )
    parser.add_argument(
        "--tp-ffn-splits",
        default=DEFAULT_TP_FFN_SPLITS,
        help="FFN shard splits for reporting (comma-separated; env VLLM_TP_FFN_SPLITS).",
    )
    parser.add_argument(
        "--run-meta-file",
        default=DEFAULT_RUN_META_FILE,
        help="Optional metadata JSON for shard reporting and default server detection (env VLLM_RUN_META_FILE).",
    )
    parser.add_argument(
        "--assumed-tok-s",
        type=float,
        default=None,
        help="Override assumed model throughput for auto-timeout logic.",
    )
    # Optional positional prompt override:
    #   python measure_token_metrics.py "my prompt here"
    parser.add_argument("prompt_positional", nargs="*", help=argparse.SUPPRESS)
    return parser.parse_args()


def _has_timeout_flag(argv: Sequence[str]) -> bool:
    for arg in argv:
        if arg == "--timeout" or arg.startswith("--timeout="):
            return True
    return False


def _has_max_tokens_flag(argv: Sequence[str]) -> bool:
    for arg in argv:
        if arg == "--max-tokens" or arg.startswith("--max-tokens="):
            return True
    return False


def _auto_timeout(args: argparse.Namespace, assumed_tps: float) -> float:
    if assumed_tps <= 0:
        return float(DEFAULT_TIMEOUT)
    per_req_tokens = max(int(args.max_tokens), 0) + max(int(args.prompt_tokens), 0)
    total_tokens = per_req_tokens * max(int(args.requests), 1)
    est_run_s = total_tokens / assumed_tps
    return float(max(DEFAULT_TIMEOUT, est_run_s * 1.25 + 60.0))


def _load_prompt(prompt: str | None, prompt_file: str | None) -> str:
    if prompt is not None:
        return prompt

    if prompt_file:
        path = Path(prompt_file)
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")

    # Fallback: a small built-in default prompt
    return DEFAULT_PROMPT


def _guess_metrics_url(base_url: str) -> str | None:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}/metrics"


def _parse_prometheus_text(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        value_str = parts[1]
        if "{" in name:
            name = name.split("{", 1)[0]
        try:
            value = float(value_str)
        except ValueError:
            continue
        metrics[name] = metrics.get(name, 0.0) + value
    return metrics


def _fetch_metrics_snapshot(url: str, timeout_s: float) -> Dict[str, float]:
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return _parse_prometheus_text(resp.text)


def _diff_metrics(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
    delta: Dict[str, float] = {}
    for key, after_val in after.items():
        before_val = before.get(key, 0.0)
        value = after_val - before_val
        if value < 0:
            value = 0.0
        delta[key] = value
    return delta


def _extract_histogram_stats(
    metrics: Dict[str, float], base_name: str
) -> tuple[float, float, float]:
    total = metrics.get(f"{base_name}_sum", 0.0)
    count = metrics.get(f"{base_name}_count", 0.0)
    avg = total / count if count > 0 else 0.0
    return total, count, avg


def _inject_metrics_summary(summary: Dict[str, float], metrics_delta: Dict[str, float]) -> None:
    targets = {
        "queue": "vllm:request_queue_time_seconds",
        "prefill": "vllm:request_prefill_time_seconds",
        "decode": "vllm:request_decode_time_seconds",
        "inference": "vllm:request_inference_time_seconds",
    }
    for key, base in targets.items():
        total, count, avg = _extract_histogram_stats(metrics_delta, base)
        summary[f"metrics_{key}_time_s_total"] = float(total)
        summary[f"metrics_{key}_time_count"] = float(count)
        summary[f"metrics_{key}_time_s_avg"] = float(avg)


def _stream_request(session: requests.Session, args: argparse.Namespace) -> Dict[str, float | List[float]]:
    url = f"{args.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    payload = {
        "model": args.model,
        "stream": True,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "messages": args.messages,
    }
    if args.include_usage:
        payload["stream_options"] = {"include_usage": True}
    if args.ignore_eos:
        payload["ignore_eos"] = True
    if args.min_tokens > 0:
        payload["min_tokens"] = args.min_tokens

    t_request = time.perf_counter()
    token_timestamps: List[float] = []
    collected_text: List[str] = []
    usage_tokens: int | None = None

    with session.post(url, headers=headers, json=payload, stream=True, timeout=args.timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or line.startswith(b":"):
                continue
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage")
            if usage and "completion_tokens" in usage:
                usage_tokens = int(usage.get("completion_tokens") or 0)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            token_piece = delta.get("content")
            if token_piece is None:
                token_piece = choice.get("text")
            if token_piece:
                ts = time.perf_counter()
                token_timestamps.append(ts)
                collected_text.append(token_piece)

    if not token_timestamps:
        raise RuntimeError("Model did not stream any tokens; check logs")

    num_stream_chunks = len(token_timestamps)
    num_tokens = usage_tokens if usage_tokens and usage_tokens > 0 else num_stream_chunks

    t_first = token_timestamps[0]
    t_last = token_timestamps[-1]

    ttft = (t_first - t_request) * 1000.0  # ms
    total_duration = (t_last - t_request) * 1000.0  # ms
    tail_duration = (t_last - t_first) * 1000.0 if num_tokens > 1 else 0.0
    throughput = (num_tokens / (total_duration / 1000.0)) if total_duration > 0 else 0.0
    prefill_throughput = (args.prompt_tokens / (ttft / 1000.0)) if ttft > 0 else 0.0

    inter_token = [
        (token_timestamps[i] - token_timestamps[i - 1]) * 1000.0
        for i in range(1, len(token_timestamps))
    ]
    tpot = (tail_duration / (num_tokens - 1)) if num_tokens > 1 else 0.0

    return {
        "TTFT_ms": ttft,
        "throughput_tok_s": throughput,
        "tpot_ms": tpot,
        "itl_ms": inter_token,
        "num_tokens": num_tokens,
        "generated_text": "".join(collected_text),
        "prompt_tokens": args.prompt_tokens,
        "prefill_tok_s": prefill_throughput,
        "t_request_s": t_request,
        "t_first_s": t_first,
        "t_last_s": t_last,
    }


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_vals = sorted(values)
    target = (len(sorted_vals) - 1) * (pct / 100.0)
    low = math.floor(target)
    high = math.ceil(target)
    if low == high:
        return float(sorted_vals[int(target)])
    fraction = target - low
    return float(sorted_vals[low] + (sorted_vals[high] - sorted_vals[low]) * fraction)


def _aggregate(results: Sequence[Dict[str, float | List[float]]]) -> Dict[str, float]:
    ttft = [r["TTFT_ms"] for r in results]
    throughput = [r["throughput_tok_s"] for r in results]
    prefill = [r["prefill_tok_s"] for r in results]
    tpot = [r["tpot_ms"] for r in results]
    itl_flat: List[float] = []
    for r in results:
        itl_flat.extend(r["itl_ms"])

    agg = {
        "TTFT_ms_mean": stats.mean(ttft) if ttft else 0.0,
        "TTFT_ms_p50": _percentile(ttft, 50),
        "TTFT_ms_p95": _percentile(ttft, 95),
        "throughput_tok_s_avg": stats.mean(throughput) if throughput else 0.0,
        "prefill_tok_s_avg": stats.mean(prefill) if prefill else 0.0,
        "tpot_ms_mean": stats.mean(tpot) if tpot else 0.0,
        "tpot_ms_p50": _percentile(tpot, 50),
        "tpot_ms_p95": _percentile(tpot, 95),
        "itl_ms_mean": stats.mean(itl_flat) if itl_flat else 0.0,
        "itl_ms_p50": _percentile(itl_flat, 50),
        "itl_ms_p95": _percentile(itl_flat, 95),
    }
    return agg


@dataclass
class RunMetrics:
    idx: int
    num_tokens: int
    prompt_tokens: int
    TTFT_ms: float
    throughput_tok_s: float
    prefill_tok_s: float
    tpot_ms: float
    itl_avg_ms: float
    raw: Dict[str, float | List[float]]


def _print_kv_block(pairs: List[tuple[str, str]], value_col: int, value_w: int) -> None:
    for label, value in pairs:
        left = f"{label}:"
        pad = max(1, value_col - len(left))
        print(left + (" " * pad) + value.rjust(value_w))


def _parse_split_list(raw: str) -> List[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return []
    try:
        return [int(p) for p in parts]
    except ValueError:
        return []


def _parse_node_bundles(raw: str) -> List[tuple[str, int]]:
    bundles: List[tuple[str, int]] = []
    for part in raw.split(","):
        chunk = part.strip()
        if not chunk or "=" not in chunk:
            continue
        node, count_raw = chunk.split("=", 1)
        node = node.strip()
        try:
            count = int(count_raw.strip())
        except ValueError:
            continue
        if node and count > 0:
            bundles.append((node, count))
    return bundles


def _format_node_shards(raw_splits: str, raw_bundles: str) -> str | None:
    splits = _parse_split_list(raw_splits)
    bundles = _parse_node_bundles(raw_bundles)
    if not splits or not bundles:
        return None
    total_ranks = sum(count for _, count in bundles)
    if total_ranks != len(splits):
        return None
    out: List[str] = []
    idx = 0
    for _, count in bundles:
        part = splits[idx : idx + count]
        if len(part) != count:
            return None
        out.append(str(sum(part)))
        idx += count
    if idx != len(splits):
        return None
    return ", ".join(out)


def _node_shard_pairs(raw_splits: str, raw_bundles: str) -> List[tuple[str, int]]:
    splits = _parse_split_list(raw_splits)
    bundles = _parse_node_bundles(raw_bundles)
    if not splits or not bundles:
        return []
    total_ranks = sum(count for _, count in bundles)
    if total_ranks != len(splits):
        return []
    out: List[tuple[str, int]] = []
    idx = 0
    for node, count in bundles:
        part = splits[idx : idx + count]
        if len(part) != count:
            return []
        out.append((node, sum(part)))
        idx += count
    if idx != len(splits):
        return []
    return out


def _format_shard_value(raw: str, mode: str | None, bundles_raw: str | None) -> str:
    value = raw.strip()
    if bundles_raw:
        node_value = _format_node_shards(value, bundles_raw.strip())
        if node_value:
            return node_value
    return value if value else "unspecified"


def _load_run_meta(path: str | None) -> Dict[str, str]:
    if not path:
        return {}
    meta_path = Path(path)
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _print_table(
    runs: Sequence[RunMetrics],
    summary: Dict[str, float],
    metadata: Sequence[tuple[str, str]] | None = None,
) -> None:
    headers = ["Run", "Tokens", "PromptTok", "TTFT(ms)", "Prefill(tok/s)", "Throughput(tok/s)", "TPOT(ms)", "ITL(ms)"]
    rows = [
        [
            str(run.idx),
            f"{run.num_tokens}",
            f"{run.prompt_tokens}",
            f"{run.TTFT_ms:.1f}",
            f"{run.prefill_tok_s:.2f}",
            f"{run.throughput_tok_s:.2f}",
            f"{run.tpot_ms:.2f}",
            f"{run.itl_avg_ms:.2f}",
        ]
        for run in runs
    ]
    col_widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    def _fmt(row: List[str]) -> str:
        return " | ".join(val.rjust(col_widths[i]) for i, val in enumerate(row))

    divider = "-+-".join("-" * w for w in col_widths)
    print(divider)
    print(_fmt(headers))
    print(divider)
    for row in rows:
        print(_fmt(row))
    print(divider)

    if "total_requests" not in summary:
        return

    print()
    print("=============== result summary ===============")
    print()

    total_block: List[tuple[str, str]] = [
        ("total request", f"{int(summary['total_requests'])}"),
        ("total runtime(s)", f"{summary['total_request_time_s']:.3f}"),
        ("input tokens", f"{int(summary['total_input_tokens'])}"),
        ("generated tokens", f"{int(summary['total_output_tokens'])}"),
    ]
    throughput_block: List[tuple[str, str]] = [
        ("prefill (tok/s)", f"{summary['prefill_tok_s_avg']:.2f}"),
        ("decode (tok/s)", f"{summary['throughput_tok_s_avg']:.2f}"),
        ("Total (tok/s)", f"{summary['total_throughput_tok_s']:.2f}"),
    ]
    ttft_block: List[tuple[str, str]] = [
        ("mean", f"{summary['TTFT_ms_mean']:.2f}"),
        ("median(p50)", f"{summary['TTFT_ms_p50']:.2f}"),
        ("p95", f"{summary['TTFT_ms_p95']:.2f}"),
    ]
    tpot_block: List[tuple[str, str]] = [
        ("mean", f"{summary['tpot_ms_mean']:.2f}"),
        ("median(p50)", f"{summary['tpot_ms_p50']:.2f}"),
        ("p95", f"{summary['tpot_ms_p95']:.2f}"),
    ]
    itl_block: List[tuple[str, str]] = [
        ("mean", f"{summary['itl_ms_mean']:.2f}"),
        ("median(p50)", f"{summary['itl_ms_p50']:.2f}"),
        ("p95", f"{summary['itl_ms_p95']:.2f}"),
    ]

    metrics_block: List[tuple[str, str]] = []
    if "metrics_queue_time_s_avg" in summary:
        metrics_block = [
            (
                "queue avg(s)",
                f"{summary['metrics_queue_time_s_avg']:.4f} (n={int(summary['metrics_queue_time_count'])})",
            ),
            ("queue total(s)", f"{summary['metrics_queue_time_s_total']:.4f}"),
            (
                "prefill avg(s)",
                f"{summary['metrics_prefill_time_s_avg']:.4f} (n={int(summary['metrics_prefill_time_count'])})",
            ),
            ("prefill total(s)", f"{summary['metrics_prefill_time_s_total']:.4f}"),
            (
                "decode avg(s)",
                f"{summary['metrics_decode_time_s_avg']:.4f} (n={int(summary['metrics_decode_time_count'])})",
            ),
            ("decode total(s)", f"{summary['metrics_decode_time_s_total']:.4f}"),
            (
                "inference avg(s)",
                f"{summary['metrics_inference_time_s_avg']:.4f} (n={int(summary['metrics_inference_time_count'])})",
            ),
            ("inference total(s)", f"{summary['metrics_inference_time_s_total']:.4f}"),
        ]

    meta_block = list(metadata or [])

    # Use one shared value column for ALL summary metrics (totals/throughput/latency).
    all_pairs = (
        meta_block
        + total_block
        + throughput_block
        + ttft_block
        + tpot_block
        + itl_block
        + metrics_block
    )
    max_left = max((len(f"{label}:") for label, _ in all_pairs), default=0)
    value_col = max(30, max_left + 5)
    value_w = max((len(value) for _, value in all_pairs), default=0)
    if meta_block:
        print("------------------- shards -------------------")
        _print_kv_block(meta_block, value_col=value_col, value_w=value_w)

    print("------------------- result -------------------")
    _print_kv_block(total_block, value_col=value_col, value_w=value_w)

    print("----------------- throughput -----------------")
    _print_kv_block(throughput_block, value_col=value_col, value_w=value_w)

    print("------------------ TTFT(ms) ------------------")
    _print_kv_block(ttft_block, value_col=value_col, value_w=value_w)
    print("------------------ TPOT(ms) ------------------")
    _print_kv_block(tpot_block, value_col=value_col, value_w=value_w)
    print("------------------ ITL (ms) ------------------")
    _print_kv_block(itl_block, value_col=value_col, value_w=value_w)

    #if metrics_block:
        #print("---------------- server timings (s) ----------------")
        #_print_kv_block(metrics_block, value_col=value_col, value_w=value_w)


def _write_runs_csv(path: Path, runs: Sequence[RunMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "tokens",
        "prompt_tokens",
        "ttft_ms",
        "prefill_tok_s",
        "throughput_tok_s",
        "tpot_ms",
        "itl_avg_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "run": run.idx,
                    "tokens": run.num_tokens,
                    "prompt_tokens": run.prompt_tokens,
                    "ttft_ms": f"{run.TTFT_ms:.4f}",
                    "prefill_tok_s": f"{run.prefill_tok_s:.4f}",
                    "throughput_tok_s": f"{run.throughput_tok_s:.4f}",
                    "tpot_ms": f"{run.tpot_ms:.4f}",
                    "itl_avg_ms": f"{run.itl_avg_ms:.4f}",
                }
            )


def _write_summary_csv(
    path: Path,
    summary: Dict[str, float],
    metadata: Sequence[tuple[str, str]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        if metadata:
            for key, value in metadata:
                writer.writerow([key, value])
        for key, value in sorted(summary.items()):
            writer.writerow([key, f"{value:.6f}"])


def _csv_safe_column(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value.strip())
    cleaned = cleaned.strip("_").lower()
    return cleaned or "unknown"


def _summary_row_fieldnames(run_meta: Dict[str, str]) -> List[str]:
    bundles_raw = str(run_meta.get("ray_node_bundles", "")).strip()
    attn_pairs = _node_shard_pairs(
        str(run_meta.get("tp_head_splits", "")).strip(),
        bundles_raw,
    )
    ffn_pairs = _node_shard_pairs(
        str(run_meta.get("tp_ffn_splits", "")).strip(),
        bundles_raw,
    )
    node_names: List[str] = []
    for node, _ in attn_pairs + ffn_pairs:
        if node not in node_names:
            node_names.append(node)

    fieldnames = [
        "tp",
        "pp",
        "llm_model",
    ]
    for node in node_names:
        suffix = _csv_safe_column(node)
        fieldnames.append(f"attention_shard_{suffix}")
    for node in node_names:
        suffix = _csv_safe_column(node)
        fieldnames.append(f"ffn_shard_{suffix}")
    fieldnames.extend(
        [
            "total_request",
            "total_runtime_s",
            "input_tokens",
            "generated_tokens",
            "prefill_tok_s",
            "decode_tok_s",
            "total_tok_s",
            "ttft_ms",
            "tpot_ms",
            "itl_ms",
        ]
    )
    return fieldnames


def _build_summary_row(
    args: argparse.Namespace,
    summary: Dict[str, float],
    run_meta: Dict[str, str],
) -> Dict[str, str]:
    def _fmt(value: float | None, digits: int = 6) -> str:
        if value is None:
            return ""
        return f"{float(value):.{digits}f}"

    bundles_raw = str(run_meta.get("ray_node_bundles", "")).strip()
    attn_pairs = dict(
        _node_shard_pairs(str(run_meta.get("tp_head_splits", "")).strip(), bundles_raw)
    )
    ffn_pairs = dict(
        _node_shard_pairs(str(run_meta.get("tp_ffn_splits", "")).strip(), bundles_raw)
    )
    node_names: List[str] = []
    for node in list(attn_pairs.keys()) + list(ffn_pairs.keys()):
        if node not in node_names:
            node_names.append(node)

    row = {
        "tp": str(run_meta.get("tp", "")),
        "pp": str(run_meta.get("pp", "")),
        "llm_model": str(args.model),
        "total_request": str(int(summary.get("total_requests", 0.0))),
        "total_runtime_s": _fmt(summary.get("total_request_time_s")),
        "input_tokens": str(int(summary.get("total_input_tokens", 0.0))),
        "generated_tokens": str(int(summary.get("total_output_tokens", 0.0))),
        "prefill_tok_s": _fmt(summary.get("prefill_tok_s_avg"), 4),
        "decode_tok_s": _fmt(summary.get("throughput_tok_s_avg"), 4),
        "total_tok_s": _fmt(summary.get("total_throughput_tok_s"), 4),
        "ttft_ms": _fmt(summary.get("TTFT_ms_mean"), 4),
        "tpot_ms": _fmt(summary.get("tpot_ms_mean"), 4),
        "itl_ms": _fmt(summary.get("itl_ms_mean"), 4),
    }
    for node in node_names:
        suffix = _csv_safe_column(node)
        row[f"attention_shard_{suffix}"] = str(attn_pairs.get(node, ""))
    for node in node_names:
        suffix = _csv_safe_column(node)
        row[f"ffn_shard_{suffix}"] = str(ffn_pairs.get(node, ""))
    return row


def _append_summary_csv(
    path: Path,
    args: argparse.Namespace,
    summary: Dict[str, float],
    run_meta: Dict[str, str],
) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _summary_row_fieldnames(run_meta)
    row = _build_summary_row(args, summary, run_meta)
    backup_path: Path | None = None
    if path.exists() and path.stat().st_size > 0:
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                existing_header = next(reader, [])
        except Exception:
            existing_header = []
        if existing_header != fieldnames:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_path = path.with_name(f"{path.stem}.bak_{timestamp}{path.suffix}")
            path.replace(backup_path)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return backup_path


def _run_single(idx: int, args: argparse.Namespace) -> tuple[int, Dict[str, float | List[float]]]:
    with requests.Session() as session:
        metrics = _stream_request(session, args)
    return idx, metrics


def main() -> None:
    args = _parse_args()
    if args.model_preset:
        args.model = MODEL_PRESETS[str(args.model_preset)]

    if not _has_max_tokens_flag(sys.argv) and not os.getenv("VLLM_MAX_TOKENS"):
        inferred_max_tokens, inferred_tps = _infer_perf_defaults(args.model)
        if inferred_max_tokens > 0:
            args.max_tokens = int(inferred_max_tokens)
    else:
        _, inferred_tps = _infer_perf_defaults(args.model)

    if args.assumed_tok_s is not None:
        assumed_tps = float(args.assumed_tok_s)
    elif os.getenv("VLLM_ASSUMED_TOK_S"):
        assumed_tps = DEFAULT_ASSUMED_TOK_S
    else:
        assumed_tps = float(inferred_tps)

    positional_prompt = " ".join(args.prompt_positional).strip() if args.prompt_positional else None
    prompt = _load_prompt(args.prompt or positional_prompt, args.prompt_file)

    args.messages = _build_messages(prompt)
    tokenizer_name = args.tokenizer or args.model
    args.prompt_tokens = _count_prompt_tokens(tokenizer_name, args.messages)
    if not _has_timeout_flag(sys.argv) and not os.getenv("VLLM_TIMEOUT"):
        auto_timeout = _auto_timeout(args, assumed_tps)
        if auto_timeout != float(args.timeout):
            args.timeout = auto_timeout
            print(
                "[INFO] Auto timeout set to {:.1f}s (requests={}, max_tokens={}, prompt_tokens={}, assumed_tok_s={:.1f})".format(
                    args.timeout, args.requests, args.max_tokens, args.prompt_tokens, assumed_tps
                )
            )

    runs = int(args.runs)
    requests_per_run = int(args.requests)
    if runs <= 0:
        print("[ERROR] --runs must be > 0", file=sys.stderr)
        sys.exit(1)
    if requests_per_run <= 0:
        print("[ERROR] --requests must be > 0", file=sys.stderr)
        sys.exit(1)

    metrics_url = args.metrics_url
    if metrics_url == "auto":
        metrics_url = _guess_metrics_url(args.base_url)
    metrics_before: Dict[str, float] | None = None
    if metrics_url:
        try:
            metrics_before = _fetch_metrics_snapshot(metrics_url, args.metrics_timeout)
            print(f"[INFO] Metrics snapshot (before): {metrics_url}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Metrics scrape failed ({metrics_url}): {exc}. Continuing without metrics.")
            metrics_url = None

    total_requests = runs * requests_per_run
    concurrency = requests_per_run

    print(
        f"[INFO] Dispatching total_requests={total_requests} "
        f"(runs={runs} * requests={requests_per_run}), concurrency={concurrency}"
    )

    per_run: List[RunMetrics] = []
    raw_metrics: List[Dict[str, float | List[float]]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        global_idx = 0
        for batch_idx in range(1, runs + 1):
            futures = []
            for _ in range(requests_per_run):
                global_idx += 1
                futures.append(pool.submit(_run_single, global_idx, args))

            for future in as_completed(futures):
                try:
                    idx, metrics = future.result()
                except Exception as exc:  # noqa: BLE001
                    print(f"[ERROR] request failed: {exc}", file=sys.stderr)
                    raise

                run = RunMetrics(
                    idx=idx,
                    num_tokens=int(metrics["num_tokens"]),
                    prompt_tokens=int(metrics["prompt_tokens"]),
                    TTFT_ms=float(metrics["TTFT_ms"]),
                    prefill_tok_s=float(metrics["prefill_tok_s"]),
                    throughput_tok_s=float(metrics["throughput_tok_s"]),
                    tpot_ms=float(metrics["tpot_ms"]),
                    itl_avg_ms=stats.mean(metrics["itl_ms"]) if metrics["itl_ms"] else 0.0,
                    raw=metrics,
                )
                per_run.append(run)
                raw_metrics.append(metrics)

                if args.verbose:
                    print(
                        f"[INFO] Request {run.idx}/{total_requests} complete (batch {batch_idx}/{runs})\n"
                        f"  tokens={run.num_tokens}\n"
                        f"  prompt_tokens={run.prompt_tokens}\n"
                        f"  TTFT={run.TTFT_ms:.1f} ms\n"
                        f"  prefill_throughput={run.prefill_tok_s:.2f} tok/s\n"
                        f"  throughput={run.throughput_tok_s:.2f} tok/s\n"
                        f"  TPOT={run.tpot_ms:.2f} ms\n"
                        f"  ITL(avg)={run.itl_avg_ms:.2f} ms"
                    )
                else:
                    print(f"[INFO] Request {run.idx}/{total_requests} done")

    per_run.sort(key=lambda r: r.idx)
    summary = _aggregate([run.raw for run in per_run])

    # Global throughput based on actual token generation window (first token to last token).
    t0 = min(float(m["t_request_s"]) for m in raw_metrics)
    t1 = max(float(m["t_last_s"]) for m in raw_metrics)
    t_first_any = min(float(m["t_first_s"]) for m in raw_metrics)
    t_last_any = max(float(m["t_last_s"]) for m in raw_metrics)
    wall_s = max(0.0, t1 - t0)
    generation_window_s = max(0.0, t_last_any - t_first_any)
    total_out_tokens = int(sum(int(m["num_tokens"]) for m in raw_metrics))
    total_input_tokens = int(sum(int(m["prompt_tokens"]) for m in raw_metrics))
    summary["total_output_tokens"] = float(total_out_tokens)
    summary["total_input_tokens"] = float(total_input_tokens)
    summary["total_generation_window_s"] = float(generation_window_s)
    summary["total_throughput_tok_s"] = float(total_out_tokens / generation_window_s) if generation_window_s > 0 else 0.0
    summary["total_request_time_s"] = float(wall_s)
    summary["total_wall_time_s"] = float(wall_s)
    summary["total_wall_throughput_tok_s"] = float(total_out_tokens / wall_s) if wall_s > 0 else 0.0
    summary["total_requests"] = float(total_requests)

    if metrics_url and metrics_before is not None:
        try:
            metrics_after = _fetch_metrics_snapshot(metrics_url, args.metrics_timeout)
            metrics_delta = _diff_metrics(metrics_before, metrics_after)
            _inject_metrics_summary(summary, metrics_delta)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Metrics scrape failed ({metrics_url}): {exc}. Skipping metrics summary.")

    run_meta = _load_run_meta(args.run_meta_file) if args.run_meta_file else {}
    tp_head_arg = args.tp_head_splits.strip()
    tp_ffn_arg = args.tp_ffn_splits.strip()
    tp_mode = "manual" if tp_head_arg or tp_ffn_arg else str(run_meta.get("tp_splits_mode", "")).strip() or None
    tp_head = tp_head_arg or str(run_meta.get("tp_head_splits", "")).strip()
    tp_ffn = tp_ffn_arg or str(run_meta.get("tp_ffn_splits", "")).strip()
    bundles_raw = str(run_meta.get("ray_node_bundles", "")).strip()

    metadata = None
    if tp_head or tp_ffn or bundles_raw:
        metadata = [
            ("attention shard (node)", _format_shard_value(tp_head, tp_mode, bundles_raw)),
            ("FFN shard (node)", _format_shard_value(tp_ffn, tp_mode, bundles_raw)),
        ]

    if args.output_csv:
        _write_runs_csv(Path(args.output_csv), per_run)
        print(f"[INFO] Wrote CSV: {args.output_csv}")
    if args.output_summary_csv:
        _write_summary_csv(Path(args.output_summary_csv), summary, metadata)
        print(f"[INFO] Wrote summary CSV: {args.output_summary_csv}")
    if args.append_summary_csv:
        backup_path = _append_summary_csv(
            Path(args.append_summary_csv),
            args,
            summary,
            run_meta,
        )
        if backup_path is not None:
            print(f"[INFO] Existing append-summary CSV schema changed; backed up old file to: {backup_path}")
        print(f"[INFO] Appended summary row CSV: {args.append_summary_csv}")

    _print_table(per_run, summary, metadata)


def _build_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]


def _count_prompt_tokens(tokenizer_name: str, messages: List[Dict[str, str]]) -> int:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers is required for token counting. "
            "Install it in the active environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            return len(token_ids)
        except (ValueError, TypeError):
            pass
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    return len(tokenizer.encode(text, add_special_tokens=True))


if __name__ == "__main__":
    main()
