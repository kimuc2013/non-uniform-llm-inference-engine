#!/usr/bin/env python3
"""Walk NVTX ranges from a nsys sqlite to break down per-microbatch PP timeline.

Reads NVTXEvent rows for the named ranges added by our gpu_worker patch:
  exec_model, irecv_issue, recv_wait, model_runner_exec,
  ring_slot_wait, ring_slot_copy, isend_issue, prev_send_wait,
  prev_send_wait_late.

For each PP rank (inferred from the "pp=N" tag in the range name), reports:
  - total wall covered by the range type
  - per-mb mean/p50/p90 duration
  - gap between consecutive exec_model ranges (CPU idle on the worker)
  - recv_wait fraction of exec_model
  - inter-rank overlap (rank0 exec_model vs rank1 exec_model busy time)

Usage:
  ./analyze_nvtx_pp.py <trace.sqlite>
"""
import re
import sqlite3
import sys
from collections import defaultdict
from statistics import median


RANGE_TYPES = (
    "exec_model",
    "irecv_issue",
    "recv_wait",
    "model_runner_exec",
    "ring_slot_wait",
    "ring_slot_copy",
    "isend_issue",
    "prev_send_wait",
    "prev_send_wait_late",
)


def main(db_path: str) -> int:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    # NVTX events are in NVTX_EVENTS (recent nsys) or NSYS_NVTX_RANGES (older).
    tables = {r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    nvtx_table = None
    for cand in ("NVTX_EVENTS", "NVTX_RANGES"):
        if cand in tables:
            nvtx_table = cand
            break
    if nvtx_table is None:
        print(f"[ERR] No NVTX table found in {db_path}", file=sys.stderr)
        print(f"      Available: {sorted(tables)[:20]}", file=sys.stderr)
        return 2

    # Pull all NVTX ranges with a name and an end time
    rows = list(cur.execute(
        f"SELECT start, end, text FROM {nvtx_table} "
        f"WHERE end IS NOT NULL AND text IS NOT NULL"
    ))
    con.close()
    if not rows:
        print(f"[ERR] no NVTX events with names in {db_path}", file=sys.stderr)
        return 2

    # parse name like "exec_model pp=0 mb=42 slot=0"
    pp_re = re.compile(r"\bpp=(\-?\d+)")
    mb_re = re.compile(r"\bmb=(\d+)")
    typ_re = re.compile(r"^(\w+)")

    # by (rank, range_type) -> list of (start, end, mb)
    buckets: dict[tuple[int, str], list[tuple[int, int, int]]] = defaultdict(list)
    # recv_wait does NOT carry pp/mb (it's in AsyncIntermediateTensors); attach later
    naked = []

    for s, e, name in rows:
        m_typ = typ_re.match(name)
        if not m_typ:
            continue
        typ = m_typ.group(1)
        if typ not in RANGE_TYPES:
            continue
        m_pp = pp_re.search(name)
        m_mb = mb_re.search(name)
        if not m_pp or not m_mb:
            # naked range (e.g., bare "recv_wait")
            naked.append((s, e, typ))
            continue
        pp = int(m_pp.group(1))
        mb = int(m_mb.group(1))
        buckets[(pp, typ)].append((s, e, mb))

    # Per-rank sorted ranges
    for k in buckets:
        buckets[k].sort()

    # Attach naked recv_wait events to a rank by enclosing exec_model
    rank_exec: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for (pp, typ), ranges in buckets.items():
        if typ == "exec_model":
            rank_exec[pp] = ranges

    def find_enclosing_rank(t: int) -> int | None:
        for pp, ranges in rank_exec.items():
            # binary search would be faster, but ranges are small
            for s, e, _ in ranges:
                if s <= t <= e:
                    return pp
        return None

    for s, e, typ in naked:
        pp = find_enclosing_rank((s + e) // 2)
        if pp is None:
            continue
        buckets[(pp, typ)].append((s, e, -1))

    # Re-sort after appending
    for k in buckets:
        buckets[k].sort()

    # Per-rank trace window
    if not rank_exec:
        print("[ERR] no exec_model ranges found", file=sys.stderr)
        return 3

    ranks = sorted(rank_exec.keys())
    print(f"=== {db_path} ===")
    print(f"detected ranks: {ranks}")

    # Use the per-rank exec_model wall as the rank's wall.
    rank_window: dict[int, tuple[int, int]] = {}
    for pp in ranks:
        em = rank_exec[pp]
        rank_window[pp] = (em[0][0], em[-1][1])

    # Per-rank, per-type report
    for pp in ranks:
        wall_start, wall_end = rank_window[pp]
        wall = (wall_end - wall_start) / 1e9
        print(f"\n--- rank pp={pp} ---  wall={wall:.3f}s, #mb={len(rank_exec[pp])}")
        # for each type, total duration + mean per range
        for typ in RANGE_TYPES:
            ranges = buckets.get((pp, typ), [])
            if not ranges:
                continue
            durs_us = [(e - s) / 1e3 for s, e, _ in ranges]
            tot_us = sum(durs_us)
            mean = tot_us / len(durs_us)
            med = median(durs_us)
            durs_sorted = sorted(durs_us)
            p90 = durs_sorted[max(0, int(0.9 * len(durs_sorted)) - 1)]
            pct = 100 * tot_us * 1e-6 / wall
            print(f"  {typ:22s}  n={len(durs_us):4d}  "
                  f"total={tot_us/1000:7.1f}ms ({pct:5.1f}% of wall)  "
                  f"mean={mean:6.1f}us  p50={med:6.1f}us  p90={p90:6.1f}us")

        # CPU gap between consecutive exec_model events at this rank
        em = rank_exec[pp]
        gaps_us = []
        for (s1, e1, _), (s2, e2, _) in zip(em, em[1:]):
            gaps_us.append((s2 - e1) / 1e3)
        if gaps_us:
            gaps_us.sort()
            n = len(gaps_us)
            print(f"  exec_model_gap         n={n:4d}  "
                  f"total={sum(gaps_us)/1000:7.1f}ms ({100*sum(gaps_us)*1e-6/wall:5.1f}% of wall)  "
                  f"mean={sum(gaps_us)/n:6.1f}us  "
                  f"p50={gaps_us[n//2]:6.1f}us  "
                  f"p90={gaps_us[int(0.9 * n)]:6.1f}us  "
                  f"max={gaps_us[-1]:6.1f}us")

    # Inter-rank overlap: how much time both ranks have exec_model active
    if len(ranks) >= 2:
        a_ranges = [(s, e) for s, e, _ in rank_exec[ranks[0]]]
        b_ranges = [(s, e) for s, e, _ in rank_exec[ranks[1]]]

        def merge(ivs):
            if not ivs:
                return []
            ivs = sorted(ivs)
            out = [list(ivs[0])]
            for s, e in ivs[1:]:
                if s <= out[-1][1]:
                    out[-1][1] = max(out[-1][1], e)
                else:
                    out.append([s, e])
            return out

        a_m = merge(a_ranges)
        b_m = merge(b_ranges)
        ov_ns = 0
        ai = bi = 0
        while ai < len(a_m) and bi < len(b_m):
            lo = max(a_m[ai][0], b_m[bi][0])
            hi = min(a_m[ai][1], b_m[bi][1])
            if lo < hi:
                ov_ns += hi - lo
            if a_m[ai][1] < b_m[bi][1]:
                ai += 1
            else:
                bi += 1
        busy_a = sum(e - s for s, e in a_m) / 1e9
        busy_b = sum(e - s for s, e in b_m) / 1e9
        ov_s = ov_ns / 1e9
        print(f"\n=== inter-rank exec_model overlap ===")
        print(f"rank {ranks[0]} busy: {busy_a:.3f}s")
        print(f"rank {ranks[1]} busy: {busy_b:.3f}s")
        print(f"overlap (CPU-side, both inside exec_model): "
              f"{ov_s:.3f}s = {100 * ov_s / max(min(busy_a, busy_b), 1e-9):.1f}% of shorter busy")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
