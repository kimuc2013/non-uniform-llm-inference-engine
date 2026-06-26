"""Analyze qwen_pp_profile traces: per-config WORKER (Ada bottleneck) rank, report
GPU-busy fraction, decode cycle time, and the top kernels by total GPU time.

Decisive read:
  - high GPU-busy% + a dominant compute kernel (attention / RMSNorm/QK-norm / GEMM)
    -> qwen TP4PP2 is compute-bound; a real qwen3 cost the roofline misses (arch gap)
  - low GPU-busy% (big idle gaps) -> CPU/dispatch/sync bound; serving-stack overhead

Usage: python planner/qwen_pp_profile_analyze.py [results/qwen_pp_profile]
"""
from __future__ import annotations
import gzip, json, sys, statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/data/esca/uckim/vllm_main/results/qwen_pp_profile")


def load(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def analyze_rank(trace_path):
    tr = load(trace_path)
    ev = tr["traceEvents"]
    x = [e for e in ev if e.get("ph") == "X"]
    ker = [e for e in x if e.get("cat") in ("kernel", "gpu_op")]
    if not ker:
        return None
    ker.sort(key=lambda e: e["ts"])
    ts = [e["ts"] for e in ker]; dur = [e.get("dur", 0) for e in ker]
    span = ts[-1] + dur[-1] - ts[0]
    busy = sum(dur)
    # forward blocks (decode steps): >5ms gaps between kernel clusters
    big = [e for e in ker if e.get("dur", 0) > 1000]
    blocks = []
    if big:
        cur = [big[0]]
        for e in big[1:]:
            if e["ts"] - (cur[-1]["ts"] + cur[-1]["dur"]) > 5000:
                blocks.append(cur); cur = [e]
            else:
                cur.append(e)
        blocks.append(cur)
    cyc = []
    if len(blocks) > 1:
        cyc = [blocks[i + 1][0]["ts"] - blocks[i][0]["ts"] for i in range(len(blocks) - 1)]
    # top kernels by total time
    by_name = defaultdict(lambda: [0, 0.0])
    for e in ker:
        k = by_name[e.get("name", "?")]
        k[0] += 1; k[1] += e.get("dur", 0)
    top = sorted(by_name.items(), key=lambda kv: -kv[1][1])[:14]
    # classify comm vs compute
    def is_comm(n):
        n = n.lower()
        return any(s in n for s in ("nccl", "allreduce", "sendrecv", "send", "recv", "broadcast", "all_reduce"))
    comm_us = sum(t for n, (c, t) in by_name.items() if is_comm(n))
    return dict(span=span, busy=busy, busy_frac=busy / span if span else 0,
                n_blocks=len(blocks), cycle_ms=(statistics.median(cyc) / 1000 if cyc else 0),
                cyc_first8=[round(c / 1000, 1) for c in cyc[:8]],
                top=top, comm_us=comm_us, total_us=busy)


def main():
    for name in sorted(p.name for p in ROOT.iterdir() if p.is_dir()):
        cdir = ROOT / name
        # worker = Ada bottleneck stage; fall back to head if absent
        wtr = sorted((cdir / "traces_worker").glob("**/*.pt.trace.json*")) \
              or sorted((cdir / "traces_worker").glob("**/rank*.json*"))
        htr = sorted((cdir / "traces").glob("**/*.pt.trace.json*")) \
              or sorted((cdir / "traces").glob("**/rank*.json*"))
        print(f"\n{'='*78}\n{name}   (worker traces: {len(wtr)}, head traces: {len(htr)})\n{'='*78}")
        for label, traces in (("WORKER/Ada", wtr), ("HEAD/Blackwell", htr)):
            if not traces:
                continue
            r = analyze_rank(traces[0])
            if not r:
                print(f"  [{label}] no kernels"); continue
            print(f"  [{label}] {traces[0].name}")
            print(f"     GPU busy {r['busy_frac']:.0%}  ({r['busy']/1e3:.0f}ms busy / {r['span']/1e3:.0f}ms span)")
            print(f"     decode blocks {r['n_blocks']}  median cycle {r['cycle_ms']:.1f}ms  first8 {r['cyc_first8']}")
            print(f"     comm kernels total {r['comm_us']/1e3:.1f}ms ({r['comm_us']/max(1,r['busy'])*100:.0f}% of busy)")
            print(f"     top kernels by total GPU time:")
            for nm, (cnt, tot) in r["top"]:
                print(f"        {tot/1e3:7.1f}ms  x{cnt:5d}  {nm[:70]}")
    print()


if __name__ == "__main__":
    main()
