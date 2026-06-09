#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sqlite3
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
TOP_LAYER_RE = re.compile(r"^(.+\.)?layers\.(\d+)$")


def _add_metric(store: dict[str, dict[str, float]], key: str, *, count: int, total_ms: float) -> None:
    item = store.setdefault(key, {"count": 0.0, "total_ms": 0.0})
    item["count"] += float(count)
    item["total_ms"] += float(total_ms)


def _shorten(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)] + "..."


def _marker_from_nvtx_range(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    if not raw.startswith("{"):
        return None
    try:
        value = ast.literal_eval(raw)
    except Exception:
        return None
    if not isinstance(value, dict):
        return None
    return value


def _module_from_nvtx_range(raw: str) -> str | None:
    value = _marker_from_nvtx_range(raw)
    if value is None:
        return None
    module = value.get("Module")
    if isinstance(module, str) and module:
        return module
    return None


def _clean_marker_payload(value: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("Inputs", "Outputs", "TrainableParams", "StaticParams"):
        if key in value:
            out[key] = value[key]
    return out


def _fetch_nvtx_summary(conn: sqlite3.Connection) -> list[tuple[str, int, float, float]]:
    query = """
        select coalesce(e.text, s.value, '') as range_name,
               count(*) as instances,
               sum(e.end - e.start) / 1000000.0 as total_ms,
               avg(e.end - e.start) / 1000.0 as avg_us
        from NVTX_EVENTS e
        left join StringIds s on e.textId = s.id
        where e.end is not null
        group by range_name
        order by total_ms desc
    """
    rows: list[tuple[str, int, float, float]] = []
    for name, instances, total_ms, avg_us in conn.execute(query):
        rows.append((str(name), int(instances), float(total_ms or 0.0), float(avg_us or 0.0)))
    return rows


def _fetch_kernel_summary(conn: sqlite3.Connection) -> list[tuple[str, int, float]]:
    query = """
        select s.value as name,
               count(*) as instances,
               sum(k.end - k.start) / 1000000.0 as total_ms
        from CUPTI_ACTIVITY_KIND_KERNEL k
        left join StringIds s on k.demangledName = s.id
        group by s.value
        order by total_ms desc
    """
    rows: list[tuple[str, int, float]] = []
    for name, instances, total_ms in conn.execute(query):
        rows.append((str(name), int(instances), float(total_ms or 0.0)))
    return rows


def _fetch_kernel_totals(conn: sqlite3.Connection) -> tuple[int, float]:
    query = """
        select count(*) as instances,
               sum(end - start) / 1000000.0 as total_ms
        from CUPTI_ACTIVITY_KIND_KERNEL
    """
    row = conn.execute(query).fetchone()
    return int(row[0] or 0), float(row[1] or 0.0)


def _kernel_category(name: str) -> str:
    lower = name.lower()
    if "nccl" in lower or "cross_device_reduce" in lower or "allgather" in lower:
        return "collective"
    if "flash_fwd" in lower or "attention" in lower or "attn" in lower:
        return "attention"
    if "cutlass" in lower or "cublas" in lower or "gemm" in lower:
        return "gemm"
    if "rms_norm" in lower or "layernorm" in lower or "norm" in lower:
        return "norm"
    if "rotary" in lower:
        return "rotary"
    if "reshape_and_cache" in lower or "kv_cache" in lower or "cache" in lower:
        return "kv_cache"
    if "act_and_mul" in lower or "silu" in lower or "gelu" in lower:
        return "activation"
    if "argmax" in lower or "topk" in lower or "sampling" in lower or "logits" in lower:
        return "sampling_logits"
    if "memcpy" in lower or "copy" in lower or "fill" in lower or "elementwise" in lower:
        return "memory_elementwise"
    return "other"


def _range_pct(value: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * value / total


def build_summary(sqlite_path: Path, *, limit: int) -> dict[str, Any]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        nvtx_rows = _fetch_nvtx_summary(conn)
        kernel_rows = _fetch_kernel_summary(conn)
        kernel_instances, kernel_total_ms = _fetch_kernel_totals(conn)

        module_metrics: dict[str, dict[str, float]] = {}
        range_metrics: dict[str, dict[str, float]] = {}
        module_payloads: dict[str, dict[str, Any]] = {}
        layer_to_modules: dict[int, set[str]] = defaultdict(set)
        layer_metrics: dict[int, dict[str, float]] = {}

        for raw, count, total_ms, _avg_us in nvtx_rows:
            marker = _marker_from_nvtx_range(raw)
            module = None
            if marker is not None:
                raw_module = marker.get("Module")
                if isinstance(raw_module, str) and raw_module:
                    module = raw_module
                    module_payloads.setdefault(module, _clean_marker_payload(marker))

            if module:
                _add_metric(module_metrics, module, count=count, total_ms=total_ms)
                top_match = TOP_LAYER_RE.match(module)
                if top_match:
                    layer_idx = int(top_match.group(2))
                    item = layer_metrics.setdefault(
                        layer_idx, {"count": 0.0, "total_ms": 0.0}
                    )
                    item["count"] += float(count)
                    item["total_ms"] += total_ms
                match = LAYER_RE.search(module)
                if match:
                    layer_to_modules[int(match.group(1))].add(module)
            else:
                if not raw:
                    continue
                _add_metric(range_metrics, raw, count=count, total_ms=total_ms)

        kernel_categories: dict[str, dict[str, float]] = {}
        for name, instances, total_ms in kernel_rows:
            category = _kernel_category(name)
            _add_metric(kernel_categories, category, count=instances, total_ms=total_ms)

        top_modules = sorted(
            (
                {
                    "module": module,
                    "instances": int(metric["count"]),
                    "approx_calls": metric["count"] / 2.0,
                    "total_ms": metric["total_ms"],
                    "avg_us": metric["total_ms"] * 1000.0 / metric["count"]
                    if metric["count"]
                    else 0.0,
                    "sample": module_payloads.get(module, {}),
                }
                for module, metric in module_metrics.items()
            ),
            key=lambda item: item["total_ms"],
            reverse=True,
        )[:limit]

        top_ranges = sorted(
            (
                {
                    "range": name,
                    "instances": int(metric["count"]),
                    "total_ms": metric["total_ms"],
                    "avg_us": metric["total_ms"] * 1000.0 / metric["count"]
                    if metric["count"]
                    else 0.0,
                }
                for name, metric in range_metrics.items()
            ),
            key=lambda item: item["total_ms"],
            reverse=True,
        )[:limit]

        top_kernels = [
            {
                "name": name,
                "category": _kernel_category(name),
                "instances": instances,
                "total_ms": total_ms,
                "pct_kernel_time": _range_pct(total_ms, kernel_total_ms),
                "avg_us": (total_ms * 1000.0 / instances) if instances else 0.0,
            }
            for name, instances, total_ms in kernel_rows[:limit]
        ]

        category_rows = sorted(
            (
                {
                    "category": category,
                    "instances": int(metric["count"]),
                    "total_ms": metric["total_ms"],
                    "pct_kernel_time": _range_pct(metric["total_ms"], kernel_total_ms),
                    "avg_us": metric["total_ms"] * 1000.0 / metric["count"]
                    if metric["count"]
                    else 0.0,
                }
                for category, metric in kernel_categories.items()
            ),
            key=lambda item: item["total_ms"],
            reverse=True,
        )

        layer_rows = [
            {
                "layer": layer,
                "nvtx_ranges": int(metric["count"]),
                "approx_calls": metric["count"] / 2.0,
                "total_ms": metric["total_ms"],
                "avg_us": metric["total_ms"] * 1000.0 / metric["count"]
                if metric["count"]
                else 0.0,
                "module_names": len(layer_to_modules.get(layer, set())),
            }
            for layer, metric in sorted(layer_metrics.items())
        ]

        layer_indexes = sorted(layer_to_modules)
        missing_layers: list[int] = []
        if layer_indexes:
            expected = set(range(layer_indexes[0], layer_indexes[-1] + 1))
            missing_layers = sorted(expected.difference(layer_indexes))

        return {
            "source": str(sqlite_path),
            "kernel_instances": kernel_instances,
            "kernel_total_ms": kernel_total_ms,
            "nvtx_instances": sum(row[1] for row in nvtx_rows),
            "nvtx_names": len(nvtx_rows),
            "layer_indexes": layer_indexes,
            "missing_layer_indexes": missing_layers,
            "layer_module_index_count": len(layer_indexes),
            "top_layer_rows": sorted(
                layer_rows,
                key=lambda item: item["total_ms"],
                reverse=True,
            )[:limit],
            "layers": layer_rows,
            "kernel_categories": category_rows,
            "top_kernels": top_kernels,
            "top_nvtx_modules": top_modules,
            "top_nvtx_ranges": top_ranges,
        }
    finally:
        conn.close()


def _print_text(summary: dict[str, Any], *, limit: int) -> str:
    lines: list[str] = []
    lines.append(f"SQLite: {summary['source']}")
    lines.append(
        "NVTX ranges: "
        f"{summary['nvtx_instances']} instances, {summary['nvtx_names']} names"
    )
    lines.append(
        "CUDA kernels: "
        f"{summary['kernel_instances']} instances, {summary['kernel_total_ms']:.3f} ms total"
    )
    lines.append(
        f"Layer modules: {summary['layer_module_index_count']} layer indexes observed"
    )
    if summary["layer_indexes"]:
        layer_ids = summary["layer_indexes"]
        preview = ",".join(str(layer) for layer in layer_ids[:32])
        suffix = "..." if len(layer_ids) > 32 else ""
        lines.append(f"Layer indexes: {preview}{suffix}")
    else:
        lines.append("Layer indexes: none found")

    if summary["top_nvtx_modules"]:
        lines.append("")
        lines.append("Top NVTX modules:")
        for item in summary["top_nvtx_modules"][:limit]:
            lines.append(
                f"{item['total_ms']:10.3f} ms  {item['instances']:8d}x  "
                f"{item['avg_us']:9.2f} us avg  {_shorten(item['module'], 100)}"
            )
    elif summary["top_nvtx_ranges"]:
        lines.append("")
        lines.append("Top NVTX ranges:")
        for item in summary["top_nvtx_ranges"][:limit]:
            lines.append(
                f"{item['total_ms']:10.3f} ms  {item['instances']:8d}x  "
                f"{item['avg_us']:9.2f} us avg  {_shorten(item['range'], 100)}"
            )

    if summary["kernel_categories"]:
        lines.append("")
        lines.append("Top CUDA kernel categories:")
        for item in summary["kernel_categories"][:limit]:
            lines.append(
                f"{item['total_ms']:10.3f} ms  {item['pct_kernel_time']:6.2f}%  "
                f"{item['instances']:8d}x  {item['category']}"
            )

    if summary["top_kernels"]:
        lines.append("")
        lines.append("Top CUDA kernels:")
        for item in summary["top_kernels"][:limit]:
            lines.append(
                f"{item['total_ms']:10.3f} ms  {item['instances']:8d}x  "
                f"{item['avg_us']:9.2f} us avg  {item['category']:18s}  "
                f"{_shorten(item['name'], 100)}"
            )
    return "\n".join(lines)


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(_md_escape(str(cell)) for cell in row) + " |")
    return out


def _print_markdown(summary: dict[str, Any], *, limit: int, prompt: bool) -> str:
    lines: list[str] = []
    if prompt:
        lines.extend(
            [
                "# Prompt",
                "You are analyzing an NVIDIA Nsight Systems profile from a vLLM inference run.",
                "Use the structured summary below to identify bottlenecks, missing layer coverage, and likely optimization targets. Distinguish observed data from inference.",
                "",
            ]
        )
    lines.extend(
        [
            "# Nsight vLLM Profile Summary",
            "",
            "## Capture",
            f"- Input: `{summary.get('input', summary['source'])}`",
            f"- SQLite source: `{summary.get('source_sqlite', summary['source'])}`",
            f"- CUDA kernel instances: {summary['kernel_instances']}",
            f"- CUDA kernel total GPU time: {summary['kernel_total_ms']:.3f} ms",
            f"- NVTX ranges: {summary['nvtx_instances']} instances across {summary['nvtx_names']} names",
            f"- Layer indexes observed: {summary['layer_module_index_count']}",
        ]
    )
    if summary["layer_indexes"]:
        layer_ids = summary["layer_indexes"]
        lines.append(
            "- Layer index span: "
            f"{layer_ids[0]}..{layer_ids[-1]} "
            f"({len(layer_ids)} observed, {len(summary['missing_layer_indexes'])} missing)"
        )
        if summary["missing_layer_indexes"]:
            lines.append(
                "- Missing layer indexes: "
                + ", ".join(str(layer) for layer in summary["missing_layer_indexes"])
            )
    else:
        lines.append("- Layer index span: none detected")

    if summary["top_layer_rows"]:
        lines.extend(["", "## Slowest Top-Level Layers"])
        lines.extend(
            _md_table(
                ["layer", "total_ms", "nvtx_ranges", "approx_calls", "avg_us", "module_names"],
                [
                    [
                        item["layer"],
                        f"{item['total_ms']:.3f}",
                        item["nvtx_ranges"],
                        f"{item['approx_calls']:.1f}",
                        f"{item['avg_us']:.2f}",
                        item["module_names"],
                    ]
                    for item in summary["top_layer_rows"][:limit]
                ],
            )
        )

    if summary["kernel_categories"]:
        lines.extend(["", "## CUDA Kernel Categories"])
        lines.extend(
            _md_table(
                ["category", "total_ms", "pct_kernel_time", "instances", "avg_us"],
                [
                    [
                        item["category"],
                        f"{item['total_ms']:.3f}",
                        f"{item['pct_kernel_time']:.2f}",
                        item["instances"],
                        f"{item['avg_us']:.2f}",
                    ]
                    for item in summary["kernel_categories"][:limit]
                ],
            )
        )

    if summary["top_kernels"]:
        lines.extend(["", "## Top CUDA Kernels"])
        lines.extend(
            _md_table(
                ["rank", "category", "total_ms", "pct", "instances", "avg_us", "kernel"],
                [
                    [
                        idx,
                        item["category"],
                        f"{item['total_ms']:.3f}",
                        f"{item['pct_kernel_time']:.2f}",
                        item["instances"],
                        f"{item['avg_us']:.2f}",
                        _shorten(item["name"], 140),
                    ]
                    for idx, item in enumerate(summary["top_kernels"][:limit], start=1)
                ],
            )
        )

    if summary["top_nvtx_modules"]:
        lines.extend(["", "## Top NVTX Modules"])
        lines.extend(
            _md_table(
                ["rank", "total_ms", "instances", "approx_calls", "avg_us", "module"],
                [
                    [
                        idx,
                        f"{item['total_ms']:.3f}",
                        item["instances"],
                        f"{item['approx_calls']:.1f}",
                        f"{item['avg_us']:.2f}",
                        _shorten(item["module"], 140),
                    ]
                    for idx, item in enumerate(summary["top_nvtx_modules"][:limit], start=1)
                ],
            )
        )
    elif summary["top_nvtx_ranges"]:
        lines.extend(["", "## Top NVTX Ranges"])
        lines.extend(
            _md_table(
                ["rank", "total_ms", "instances", "avg_us", "range"],
                [
                    [
                        idx,
                        f"{item['total_ms']:.3f}",
                        item["instances"],
                        f"{item['avg_us']:.2f}",
                        _shorten(item["range"], 140),
                    ]
                    for idx, item in enumerate(summary["top_nvtx_ranges"][:limit], start=1)
                ],
            )
        )

    lines.extend(
        [
            "",
            "## Notes For Analysis",
            "- Top-level layer NVTX durations are wall-time ranges around module forward calls; nested module durations can overlap and should not be summed as exclusive time.",
            "- CUDA kernel category labels are heuristic name-based groupings.",
            "- Use the `.nsys-rep` timeline for exact ordering and overlap.",
        ]
    )
    return "\n".join(lines)


def _resolve_sqlite_input(input_path: Path, nsys_bin: str) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if input_path.suffix == ".sqlite":
        return input_path, None
    if input_path.suffix != ".nsys-rep":
        raise ValueError("input must be a .sqlite or .nsys-rep file")

    tmpdir = tempfile.TemporaryDirectory(prefix="nsight_summary_")
    sqlite_path = Path(tmpdir.name) / f"{input_path.stem}.sqlite"
    subprocess.run(
        [
            nsys_bin,
            "export",
            "-t",
            "sqlite",
            "-f",
            "true",
            "--quiet=true",
            "-o",
            str(sqlite_path),
            str(input_path),
        ],
        check=True,
    )
    return sqlite_path, tmpdir


def summarize(sqlite_path: Path, *, limit: int) -> None:
    print(_print_text(build_summary(sqlite_path, limit=limit), limit=limit))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Nsight Systems exports for vLLM profiling."
    )
    parser.add_argument("input", type=Path, help="Path to a .sqlite or .nsys-rep file.")
    parser.add_argument("--limit", type=int, default=20, help="Rows per section.")
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "json", "prompt"],
        default="text",
        help="Output format. prompt is Markdown with analysis instructions.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Alias for --format prompt.",
    )
    parser.add_argument("--out", type=Path, help="Write output to this file.")
    parser.add_argument(
        "--nsys-bin",
        default=os.getenv("NSYS_BIN", "/usr/local/cuda-12.9/bin/nsys"),
        help="nsys path used when input is .nsys-rep.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"not a file: {args.input}")

    tmpdir: tempfile.TemporaryDirectory[str] | None = None
    try:
        sqlite_path, tmpdir = _resolve_sqlite_input(args.input, args.nsys_bin)
        summary = build_summary(sqlite_path, limit=max(1, args.limit))
        summary["input"] = str(args.input)
        summary["source_sqlite"] = str(sqlite_path)
        fmt = "prompt" if args.llm else args.format
        if fmt == "json":
            output = json.dumps(summary, ensure_ascii=False, indent=2)
        elif fmt == "markdown":
            output = _print_markdown(summary, limit=max(1, args.limit), prompt=False)
        elif fmt == "prompt":
            output = _print_markdown(summary, limit=max(1, args.limit), prompt=True)
        else:
            output = _print_text(summary, limit=max(1, args.limit))

        if args.out:
            args.out.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
