#!/usr/bin/env python3
"""Analyze engine-side NVTX ranges added by core.py patch.

Ranges:
  engine_schedule
  engine_exec_submit
  engine_sample_submit
  engine_pop_wait
  engine_update_from_output
"""
import re
import sqlite3
import sys
from statistics import median


RANGE_TYPES = (
    "engine_schedule",
    "engine_exec_submit",
    "engine_sample_submit",
    "engine_pop_wait",
    "engine_update_from_output",
)


def main(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    typ_re = re.compile(r"^(\w+)")
    rows = list(cur.execute(
        "SELECT start, end, text FROM NVTX_EVENTS WHERE end IS NOT NULL "
        "AND text IS NOT NULL"
    ))
    con.close()
    by_type = {t: [] for t in RANGE_TYPES}
    for s, e, name in rows:
        m = typ_re.match(name)
        if m and m.group(1) in by_type:
            by_type[m.group(1)].append(e - s)

    print(f"=== {db_path} ===")
    total_wall_us = 0
    for t in RANGE_TYPES:
        durs = by_type[t]
        if not durs:
            print(f"  {t:30s}  n=0")
            continue
        durs_us = [d / 1000.0 for d in durs]
        durs_us.sort()
        n = len(durs_us)
        total_us = sum(durs_us)
        total_wall_us += total_us
        print(f"  {t:30s}  n={n:5d}  "
              f"total={total_us/1000:7.1f}ms  "
              f"mean={total_us/n:7.1f}us  "
              f"p50={durs_us[n//2]:7.1f}us  "
              f"p90={durs_us[int(0.9*n)]:7.1f}us  "
              f"max={durs_us[-1]:8.1f}us")
    print(f"\n  TOTAL engine CPU: {total_wall_us/1000:.1f}ms")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
