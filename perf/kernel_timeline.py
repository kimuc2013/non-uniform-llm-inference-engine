#!/usr/bin/env python3
"""Per-device 1ms-bucket occupancy across the whole trace, plus the long
kernels (>5ms) that gate the pipeline.
"""
import sqlite3
import sys


def main(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Trace span
    t0, t1 = cur.execute(
        "SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    if t0 is None:
        print("No kernel events found"); return 1
    span_ms = (t1 - t0) / 1e6
    print(f"Trace span: {span_ms:.1f}ms")

    # Long kernels (potential pipeline gates)
    print("\nLong kernels (>5ms):")
    rows = list(cur.execute(
        "SELECT k.start, k.end, k.deviceId, s.value FROM "
        "CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s "
        "ON s.id = k.demangledName "
        "WHERE (k.end - k.start) > 5000000 "
        "ORDER BY k.start LIMIT 50"
    ))
    print(f"  {len(rows)} kernels >5ms:")
    for s, e, dev, name in rows[:20]:
        d_us = (e - s) / 1000.0
        s_ms = (s - t0) / 1e6
        nm = (name or "?").split("(")[0][:70]
        print(f"   start={s_ms:8.1f}ms  dur={d_us:9.1f}us  dev={dev}  {nm}")

    # Aggregate kernel busy time per device per 1ms bucket using
    # SUM(min(end, bucket_end) - max(start, bucket_start)). We do it
    # in Python over a stride of buckets to avoid loading every kernel.
    bucket_ns = 1_000_000  # 1ms
    n_bins = int((t1 - t0) // bucket_ns) + 1
    # Identify devices used.
    devs = [r[0] for r in cur.execute(
        "SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId"
    )]
    busy = {d: [0] * n_bins for d in devs}
    # Stream all kernels and accumulate per bucket.
    nrows = 0
    for s, e, dev in cur.execute(
        "SELECT start, end, deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE end IS NOT NULL"
    ):
        nrows += 1
        s_b = max(0, (s - t0) // bucket_ns)
        e_b = min(n_bins - 1, (e - t0) // bucket_ns)
        for i in range(int(s_b), int(e_b) + 1):
            bs = t0 + i * bucket_ns
            be = bs + bucket_ns
            busy[dev][i] += max(0, min(e, be) - max(s, bs))
    print(f"\nProcessed {nrows} kernels.")

    # Sample 80 buckets from across the trace at evenly spaced intervals,
    # for a quick eyeball look at the pipeline pattern.
    print("\nOccupancy fraction (X if dev >50% busy in that 1ms bucket):")
    print(f"  ({n_bins} buckets total, showing 80 evenly spaced)")
    sample_idx = [i for i in range(0, n_bins, max(1, n_bins // 80))][:80]
    print(f"{'t(ms)':>7}  " + "  ".join(f"dev{d}" for d in devs))
    for i in sample_idx:
        chars = []
        for d in devs:
            frac = busy[d][i] / bucket_ns
            if frac > 0.5:
                chars.append("X")
            elif frac > 0.1:
                chars.append("·")
            else:
                chars.append(" ")
        print(f"{i:>7}  " + "    ".join(chars))

    # Overall busy fraction
    print("\nOverall device busy %:")
    for d in devs:
        tot = sum(busy[d])
        wall = (t1 - t0)
        print(f"  dev{d}: {tot/wall*100:.1f}%")

    con.close()
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: kernel_timeline.py <sqlite>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
