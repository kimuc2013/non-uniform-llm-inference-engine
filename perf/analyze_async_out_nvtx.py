#!/usr/bin/env python3
"""Analyze async-output pipeline NVTX ranges added by multiproc_executor.py.

Ranges (per worker process, per mb):
  eo_total                 -- full enqueue_output()
  eo_get_output            -- output.get_output() inside enqueue_output
                             (this is where async_copy_ready_event.synchronize
                             blocks)
  eo_response_mq_enqueue   -- response_mq.enqueue() inside enqueue_output
  ho_async_q_put           -- async_output_queue.put() inside handle_output
                             (called from worker thread, NOT busy_loop)
  aol_q_get                -- async_output_queue.get() inside busy_loop
                             (this is the busy_loop's "idle wait" between
                             outputs)
  aol_enqueue_output       -- enqueue_output call inside busy_loop
"""
import re
import sqlite3
import sys


RANGE_TYPES = (
    "eo_total",
    "eo_get_output",
    "eo_response_mq_enqueue",
    "ho_async_q_put",
    "aol_q_get",
    "aol_enqueue_output",
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
    by_type: dict[str, list[int]] = {t: [] for t in RANGE_TYPES}
    for s, e, name in rows:
        m = typ_re.match(name)
        if m and m.group(1) in by_type:
            by_type[m.group(1)].append(e - s)

    print(f"=== {db_path} ===")
    total_wall_us = 0.0
    for t in RANGE_TYPES:
        durs = by_type[t]
        if not durs:
            print(f"  {t:30s}  n=0")
            continue
        durs_us = sorted(d / 1000.0 for d in durs)
        n = len(durs_us)
        total_us = sum(durs_us)
        total_wall_us += total_us
        print(f"  {t:30s}  n={n:5d}  "
              f"total={total_us/1000:7.1f}ms  "
              f"mean={total_us/n:7.1f}us  "
              f"p50={durs_us[n//2]:7.1f}us  "
              f"p90={durs_us[int(0.9*n)]:7.1f}us  "
              f"max={durs_us[-1]:8.1f}us")
    print(f"\n  TOTAL async-output NVTX wall: {total_wall_us/1000:.1f}ms "
          "(NOT exclusive — eo_get_output and eo_response_mq_enqueue "
          "are children of eo_total/aol_enqueue_output)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
