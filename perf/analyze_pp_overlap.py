#!/usr/bin/env python3
"""Analyse a vllm PP nsight sqlite for stage-level overlap.

Approach: group NVTX ranges by globalTid (= per-rank process), find each rank's
forward-compute windows from the longest model-compute kernels, and report:
  - per-rank "busy" time
  - max time two ranks are simultaneously busy (= PP overlap)
  - whether ranks are sequential (overlap ≈ 0) or pipelined (overlap > 50% of busy)
"""
import sys
import sqlite3
from collections import defaultdict

DB = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PP_NSIGHT_DB", "/tmp/vllm_pp_nsight.sqlite")
con = sqlite3.connect(DB)
cur = con.cursor()

# Use CUPTI kernel activity per device, which is the truth of when each GPU
# is actually computing. Group by (process, gpu).
print(f"=== {DB} ===")
print()
rows = list(cur.execute("""
    SELECT k.start, k.end, k.deviceId, n.value AS kname
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    LEFT JOIN StringIds n ON n.id = k.shortName
    ORDER BY k.start
    LIMIT 50000
"""))
print(f"kernel events: {len(rows)}")
if not rows:
    sys.exit(0)

# Per-device busy windows
by_dev = defaultdict(list)
for start, end, dev, _ in rows:
    by_dev[dev].append((start, end))

# total busy time per device
def merge(intervals):
    if not intervals: return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged

print()
print("device busy (merged):")
trace_start = min(r[0] for r in rows)
trace_end = max(r[1] for r in rows)
wall = (trace_end - trace_start) / 1e9
print(f"  wall time: {wall:.3f}s")

merged_per_dev = {}
for dev, ivs in by_dev.items():
    m = merge(ivs)
    busy_ns = sum(e - s for s, e in m)
    print(f"  device {dev}: {busy_ns/1e9:.3f}s busy ({100*busy_ns/(trace_end-trace_start):.1f}% of wall)")
    merged_per_dev[dev] = m

# Pairwise overlap
devs = sorted(by_dev.keys())
if len(devs) >= 2:
    print()
    print("pairwise overlap (busy time on BOTH devices):")
    for i in range(len(devs)):
        for j in range(i+1, len(devs)):
            a, b = merged_per_dev[devs[i]], merged_per_dev[devs[j]]
            overlap_ns = 0
            ai, bi = 0, 0
            while ai < len(a) and bi < len(b):
                lo = max(a[ai][0], b[bi][0])
                hi = min(a[ai][1], b[bi][1])
                if lo < hi:
                    overlap_ns += hi - lo
                if a[ai][1] < b[bi][1]:
                    ai += 1
                else:
                    bi += 1
            busy_a = sum(e-s for s,e in a)
            busy_b = sum(e-s for s,e in b)
            min_busy = min(busy_a, busy_b)
            print(f"  dev {devs[i]} vs {devs[j]}: overlap {overlap_ns/1e9:.3f}s "
                  f"({100*overlap_ns/min_busy:.1f}% of shorter busy)")

con.close()
