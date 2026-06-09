#!/usr/bin/env python3
"""Compare two nsight sqlite traces' inter-stage PP overlap.

Usage:
  ./compare_pp_overlap.py <baseline.sqlite> <ring.sqlite>

For each trace, computes per-device busy fraction and PP overlap between
the two PP stages (devices 0,1 = stage 0; 2,3 = stage 1, assuming TP=2 PP=2).
Reports the delta.
"""
import sqlite3
import sys
from collections import defaultdict


def analyze(db_path: str) -> dict:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    rows = list(cur.execute("""
        SELECT k.start, k.end, k.deviceId
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        ORDER BY k.start
    """))
    con.close()
    if not rows:
        return {"path": db_path, "error": "no kernel events"}
    by_dev = defaultdict(list)
    for s, e, d in rows:
        by_dev[d].append((s, e))

    def merge(ivs):
        ivs = sorted(ivs)
        merged = [list(ivs[0])]
        for s, e in ivs[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged

    def overlap_ns(a, b):
        out = 0
        ai = bi = 0
        while ai < len(a) and bi < len(b):
            lo = max(a[ai][0], b[bi][0])
            hi = min(a[ai][1], b[bi][1])
            if lo < hi:
                out += hi - lo
            if a[ai][1] < b[bi][1]:
                ai += 1
            else:
                bi += 1
        return out

    merged = {d: merge(by_dev[d]) for d in by_dev}
    trace_start = min(r[0] for r in rows)
    trace_end = max(r[1] for r in rows)
    wall_ns = trace_end - trace_start

    busy = {d: sum(e - s for s, e in m) for d, m in merged.items()}
    busy_pct = {d: 100 * busy[d] / wall_ns for d in busy}

    # PP=2 mapping: stage 0 = lowest 2 devs, stage 1 = highest 2 devs
    devs = sorted(merged.keys())
    if len(devs) != 4:
        # Fallback: compute all pairwise overlaps
        pairs = []
        for i in range(len(devs)):
            for j in range(i + 1, len(devs)):
                ov = overlap_ns(merged[devs[i]], merged[devs[j]])
                pairs.append((devs[i], devs[j], ov))
        return {
            "path": db_path,
            "wall_s": wall_ns / 1e9,
            "busy_pct": busy_pct,
            "pairs": pairs,
        }

    stage0 = devs[:2]
    stage1 = devs[2:]
    # Inter-stage overlap = overlap between any stage0 device and any stage1 device
    inter_overlaps = []
    for a in stage0:
        for b in stage1:
            ov_ns = overlap_ns(merged[a], merged[b])
            min_busy = min(busy[a], busy[b])
            inter_overlaps.append({
                "a": a, "b": b,
                "overlap_ns": ov_ns,
                "overlap_s": ov_ns / 1e9,
                "pct_of_shorter_busy": 100 * ov_ns / max(1, min_busy),
            })

    return {
        "path": db_path,
        "wall_s": wall_ns / 1e9,
        "busy_pct": busy_pct,
        "stage0": stage0,
        "stage1": stage1,
        "inter_overlaps": inter_overlaps,
        "avg_inter_overlap_pct": (
            sum(d["pct_of_shorter_busy"] for d in inter_overlaps)
            / len(inter_overlaps)
        ),
    }


def report(r: dict, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"file: {r['path']}")
    if "error" in r:
        print(f"ERROR: {r['error']}")
        return
    print(f"wall: {r['wall_s']:.3f}s")
    for d, pct in sorted(r["busy_pct"].items()):
        print(f"  device {d}: {pct:.1f}% busy")
    if "inter_overlaps" not in r:
        for a, b, ov in r["pairs"]:
            print(f"  {a} vs {b}: overlap={ov/1e9:.3f}s")
        return
    print(f"stage 0 devices: {r['stage0']}")
    print(f"stage 1 devices: {r['stage1']}")
    print("inter-stage overlap:")
    for d in r["inter_overlaps"]:
        print(f"  dev {d['a']} <-> dev {d['b']}: "
              f"{d['overlap_s']:.3f}s overlap, "
              f"{d['pct_of_shorter_busy']:.1f}% of shorter busy")
    print(f"AVG inter-stage overlap: {r['avg_inter_overlap_pct']:.1f}%")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    rs = [analyze(p) for p in sys.argv[1:]]
    labels = [f"trace {i}" for i in range(len(rs))]
    for r, lbl in zip(rs, labels):
        report(r, lbl)
    if len(rs) == 2 and all("avg_inter_overlap_pct" in r for r in rs):
        delta = rs[1]["avg_inter_overlap_pct"] - rs[0]["avg_inter_overlap_pct"]
        print(f"\n>>> delta inter-stage overlap: {delta:+.1f} percentage points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
