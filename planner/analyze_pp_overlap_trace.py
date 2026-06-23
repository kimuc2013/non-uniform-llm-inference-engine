"""Analyze torch profiler traces to verify PP overlap.

Key questions:
1. Do stage 0 (head, ranks 0-3) and stage 1 (worker, ranks 4-7) execute
   forward kernels CONCURRENTLY in wall-clock?
2. What is the cycle time between consecutive forwards on stage 0?
3. Is the M13 side-stream broadcast on a non-default CUDA stream?
"""
from __future__ import annotations
import gzip, json, os, sys
from pathlib import Path
import statistics

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/data/esca/uckim/vllm_main/results/verify_pp_overlap_20260611_200044")


def load_trace(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def main():
    head_traces = sorted((OUT / "traces").glob("rank*.pt.trace.json.gz"))
    worker_traces = sorted((OUT / "traces_worker").glob("rank*.pt.trace.json.gz"))
    print(f"head ranks: {[p.name.split('.')[0] for p in head_traces]}")
    print(f"worker ranks: {[p.name.split('.')[0] for p in worker_traces]}")

    # We need a way to align head and worker timelines. PyTorch profiler timestamps
    # are CLOCK_MONOTONIC on their respective hosts — different epochs. We can't
    # directly compare wall-clock. But we CAN measure cycle time and ratio.

    # Pick rank 0 (head stage 0) and rank 4 (worker stage 1)
    def get_forward_intervals(trace):
        # Look at GPU kernel events with category 'kernel' or 'cuda_runtime'
        events = trace["traceEvents"]
        # Find big compute kernels — likely matmul. Use category=kernel and dur > 1ms.
        kernel_events = [e for e in events
                         if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_op")
                         and e.get("dur", 0) > 1000]   # > 1ms
        # sort by ts
        kernel_events.sort(key=lambda e: e["ts"])
        return kernel_events

    def summarize_rank(label, trace_path):
        try:
            tr = load_trace(trace_path)
        except Exception as e:
            print(f"failed to load {trace_path}: {e}")
            return None
        events = tr["traceEvents"]
        # Quick stats
        all_x = [e for e in events if e.get("ph") == "X"]
        gpu_x = [e for e in all_x if e.get("cat") in ("kernel", "gpu_op", "cuda_runtime")]
        large = [e for e in gpu_x if e.get("dur", 0) > 1000]
        print(f"\n[{label}]")
        print(f"  total events: {len(events)}, X (complete) events: {len(all_x)}")
        print(f"  GPU kernels: {len(gpu_x)}, large (>1ms): {len(large)}")
        if not large: return None
        # First, last, span
        ts = [e["ts"] for e in large]
        durs = [e["dur"] for e in large]
        span_us = ts[-1] + durs[-1] - ts[0]
        compute_us = sum(durs)
        gpu_busy_frac = compute_us / span_us if span_us > 0 else 0
        print(f"  span: {span_us/1e6:.2f}s, compute: {compute_us/1e6:.2f}s, "
              f"gpu busy: {gpu_busy_frac:.1%}")
        # Find compute "blocks" — consecutive kernels close in time = one forward
        # Use a 5ms gap as boundary
        blocks = []
        cur = [large[0]]
        for e in large[1:]:
            if e["ts"] - (cur[-1]["ts"] + cur[-1]["dur"]) > 5000:
                blocks.append(cur); cur = [e]
            else:
                cur.append(e)
        blocks.append(cur)
        print(f"  forward blocks (gap >5ms): {len(blocks)}")
        block_durs = [b[-1]["ts"] + b[-1]["dur"] - b[0]["ts"] for b in blocks]
        if block_durs:
            print(f"  block duration: mean {statistics.mean(block_durs)/1000:.1f}ms, "
                  f"median {statistics.median(block_durs)/1000:.1f}ms")
        # Inter-block gap = cycle time
        if len(blocks) > 1:
            gaps = [blocks[i+1][0]["ts"] - blocks[i][0]["ts"] for i in range(len(blocks)-1)]
            print(f"  block start interval (cycle time): "
                  f"mean {statistics.mean(gaps)/1000:.1f}ms, "
                  f"median {statistics.median(gaps)/1000:.1f}ms")
            # Show first 5
            print(f"  first 5 cycles (ms): {[f'{g/1000:.1f}' for g in gaps[:5]]}")

        # Look for NCCL kernels
        nccl_kernels = [e for e in all_x if "nccl" in e.get("name", "").lower() or
                                            "AllReduce" in e.get("name", "") or
                                            "send" in e.get("name", "").lower() or
                                            "recv" in e.get("name", "").lower()]
        print(f"  NCCL/P2P kernels: {len(nccl_kernels)}")
        if nccl_kernels:
            sample = nccl_kernels[:3]
            for e in sample:
                print(f"    name={e.get('name','?')[:60]} dur={e.get('dur',0)}us cat={e.get('cat','?')}")

        # Look for streams — is broadcast on a non-default stream?
        streams = {}
        for e in all_x:
            if "stream" in e.get("args", {}):
                s = e["args"]["stream"]
                streams[s] = streams.get(s, 0) + 1
        print(f"  CUDA streams used: {sorted(streams.items(), key=lambda x: -x[1])[:5]}")

        return {
            "blocks": len(blocks),
            "cycle_ms": statistics.mean(gaps)/1000 if len(blocks) > 1 else 0,
            "block_dur_ms": statistics.mean(block_durs)/1000 if block_durs else 0,
            "gpu_busy_frac": gpu_busy_frac,
        }

    head_results = []
    for p in head_traces:
        r = summarize_rank(f"head {p.name.split('.')[0]}", p)
        if r: head_results.append(r)
    worker_results = []
    for p in worker_traces:
        r = summarize_rank(f"worker {p.name.split('.')[0]}", p)
        if r: worker_results.append(r)

    print("\n========================================")
    print("PP OVERLAP VERDICT")
    print("========================================")
    if head_results and worker_results:
        head_cycle = statistics.mean(r["cycle_ms"] for r in head_results)
        worker_cycle = statistics.mean(r["cycle_ms"] for r in worker_results)
        head_block = statistics.mean(r["block_dur_ms"] for r in head_results)
        worker_block = statistics.mean(r["block_dur_ms"] for r in worker_results)
        head_busy = statistics.mean(r["gpu_busy_frac"] for r in head_results)
        worker_busy = statistics.mean(r["gpu_busy_frac"] for r in worker_results)

        print(f"Stage 0 (head Blackwell):  forward {head_block:.1f}ms, "
              f"cycle {head_cycle:.1f}ms, GPU busy {head_busy:.1%}")
        print(f"Stage 1 (worker Ada):      forward {worker_block:.1f}ms, "
              f"cycle {worker_cycle:.1f}ms, GPU busy {worker_busy:.1%}")
        print()
        if abs(head_cycle - worker_cycle) < max(head_cycle, worker_cycle) * 0.15:
            print(f"VERDICT: cycle times match within 15% — pipeline running in lock-step.")
        if head_busy > 0.5 and worker_busy > 0.5:
            print(f"VERDICT: both stages >50% busy in their respective trace windows — "
                  f"strong evidence of OVERLAP (otherwise idle waiting for the other stage).")
        if head_cycle > head_block * 1.5:
            print(f"NOTE: head cycle ({head_cycle:.0f}ms) > head forward ({head_block:.0f}ms) — "
                  f"head waits for worker stage; matches Ada-bottleneck expectation.")

if __name__ == "__main__":
    main()
