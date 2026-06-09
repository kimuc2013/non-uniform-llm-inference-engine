#!/usr/bin/env python3
"""Per-stage NVTX time breakdown. Picks up `stage_*` ranges emitted by
gpu_model_runner.py when VLLM_PP_STAGE_NVTX=1.
"""
import re
import sqlite3
import sys


RANGE_TYPES = (
    "stage_preprocess",
    "stage_update_states",
    "stage_prep_inputs",
    "stage_forward",
    "stage_postprocess",
    "stage_pp_bcast_recv",
    "stage_sample",
    "stage_update_states_after_exec",
    "stage_pp_bcast_send",
    "stage_bookkeep",
    "stage_async_output_create",
)


def main(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rows = list(cur.execute(
        "SELECT start, end, text FROM NVTX_EVENTS WHERE end IS NOT NULL "
        "AND text IS NOT NULL"
    ))
    con.close()
    by_type: dict[str, list[int]] = {t: [] for t in RANGE_TYPES}
    for s, e, name in rows:
        if name in by_type:
            by_type[name].append(e - s)

    print(f"=== {db_path} ===")
    total_us = 0.0
    print(f"{'stage':32s}  {'n':>5}  {'total(ms)':>10}  "
          f"{'mean(us)':>10}  {'p50(us)':>10}  {'p90(us)':>10}")
    for t in RANGE_TYPES:
        durs = by_type[t]
        if not durs:
            print(f"  {t:30s}  n=0")
            continue
        durs_us = sorted(d / 1000.0 for d in durs)
        n = len(durs_us)
        total = sum(durs_us)
        total_us += total
        print(f"  {t:30s}  n={n:5d}  "
              f"{total/1000:10.1f}  "
              f"{total/n:10.1f}  "
              f"{durs_us[n//2]:10.1f}  "
              f"{durs_us[int(0.9*n)]:10.1f}")
    print(f"\n  TOTAL (sum, NOT exclusive — nested ranges overlap): {total_us/1000:.1f}ms")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: analyze_stage_nvtx.py <sqlite>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
