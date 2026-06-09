"""Quick test: do two concurrent matmul streams on the same GPU overlap?

Approximate Llama-3.1-8B per-mb workload:
- Each layer does QKV proj (4096->4096*3), attn (16x16 attention),
  out_proj (4096->4096), gate_up (4096->14336*2), down (14336->4096).
- With TP=2, each shard is 4096->2048 (QKV) / 14336->7168 (FFN) / etc.
- 16 layers (PP rank 0 half), bf16, batch=16 tokens.

We synthesize the dominant GEMM shape and run it on two streams.
If concurrent throughput == 2x single-stream throughput, GPU has
headroom — multi-stream PP overlap would help.
If concurrent throughput == 1x single, GPU is saturated — no benefit.
"""
import time

import torch

DEV = torch.device("cuda")
DTYPE = torch.bfloat16

# Approximate decode workload per mb on PP rank 1 with TP=2:
# Hidden dim 4096, intermediate dim 14336, 16 layers, batch=16 tokens.
# After TP=2 shard: out dim halved.
import sys
M = int(sys.argv[1]) if len(sys.argv) > 1 else 16  # batch tokens per mb
MODEL = sys.argv[2] if len(sys.argv) > 2 else "8b"
if MODEL == "8b":
    K = 4096          # hidden
    N = 14336 // 2    # FFN intermediate (TP=2 shard)
    LAYERS = 16       # half of Llama-8B's 32 layers (PP=2 stage)
elif MODEL == "70b":
    K = 8192          # hidden
    N = 28672 // 2    # FFN intermediate (TP=2 shard)
    LAYERS = 40       # half of Llama-70B's 80 layers (PP=2 stage)
else:
    raise ValueError(f"unknown model {MODEL!r}")


def make_workload(stream: torch.cuda.Stream):
    """One layer = one big GEMM on this stream."""
    a = torch.randn((M, K), dtype=DTYPE, device=DEV)
    b = torch.randn((K, N), dtype=DTYPE, device=DEV)
    return a, b


def run_one(stream: torch.cuda.Stream, a, b, iters: int):
    with torch.cuda.stream(stream):
        for _ in range(iters):
            for _ in range(LAYERS):
                a = (a @ b)[:, :K]
                # rebind a to its first K cols so dims stay consistent
    return a


def main():
    # Warmup
    s = torch.cuda.Stream()
    a, b = make_workload(s)
    run_one(s, a, b, iters=2)
    torch.cuda.synchronize()

    # Single-stream baseline
    s1 = torch.cuda.Stream()
    a1, b1 = make_workload(s1)
    torch.cuda.synchronize()
    t = time.perf_counter()
    run_one(s1, a1, b1, iters=200)
    torch.cuda.synchronize()
    single_t = time.perf_counter() - t
    single_thr = 200 / single_t

    # Two-stream concurrent
    s2a = torch.cuda.Stream()
    s2b = torch.cuda.Stream()
    a2a, b2a = make_workload(s2a)
    a2b, b2b = make_workload(s2b)
    torch.cuda.synchronize()
    t = time.perf_counter()
    # Launch on both streams
    with torch.cuda.stream(s2a):
        for _ in range(200):
            for _ in range(LAYERS):
                a2a = (a2a @ b2a)[:, :K]
    with torch.cuda.stream(s2b):
        for _ in range(200):
            for _ in range(LAYERS):
                a2b = (a2b @ b2b)[:, :K]
    torch.cuda.synchronize()
    two_t = time.perf_counter() - t
    two_thr = 400 / two_t  # 2 streams × 200 iters

    print(f"single-stream: {single_t*1000:.1f}ms for 200 iters "
          f"({single_thr:.1f} iter/s)")
    print(f"two-stream:    {two_t*1000:.1f}ms for 400 iters total "
          f"({two_thr:.1f} iter/s)")
    print(f"speedup:       {two_thr / single_thr:.2f}× "
          f"(2.0× = perfect overlap, 1.0× = full saturation)")


if __name__ == "__main__":
    main()
