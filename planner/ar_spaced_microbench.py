"""Decisive test for the OPT-30B in-decode AR regime.

Back-to-back all-reduce (the plain microbench) measures ~1.1 GB/s at 8-rank/1MB.
But the big-3 models' in-decode effective AR is ~2.85 GB/s — the difference is
that real decode SEPARATES consecutive AllReduces with per-layer work. OPT-30B
(90% AR, multi-head attention) stays at the slow ~1.1 GB/s. Question: what kind
of inter-AR work grants the relief? A compute-bound GEMV, or does OPT's heavy
memory-bound KV-attention (MHA, n_kv=56) FAIL to relieve (or even block) it?

This measures the AR kernel time (CUDA events around just the all_reduce) when
each AR is preceded on the same stream by:
  none     : nothing (back-to-back) — the OPT-like regime
  gemm     : a compute-bound matmul (~target ms) — a GQA-GEMV-like spacer
  memread  : a memory-bound HBM read (~target ms) — a MHA-KV-attention-like spacer
If gemm relieves the AR but memread does not, the separator is memory-bandwidth
contention (computable from each model's per-layer KV/weight HBM bytes) — a
non-overfit mechanism. If both relieve equally, the separator is just timing
spacing (and OPT's tiny compute is the cause). Run at world=8 (4+4) via torchrun.
"""
import os, torch, torch.distributed as dist

HIDDEN = 8192
NREQ = 64                       # AR message = 64*8192*2 = 1.05 MB
TARGET_MS = float(os.environ.get("SPACER_MS", "0.25"))


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    x = torch.randn(NREQ, HIDDEN, dtype=torch.bfloat16, device=dev)

    # size a compute-bound matmul to ~TARGET_MS on this GPU
    K = 4096
    A = torch.randn(K, K, dtype=torch.bfloat16, device=dev)
    B = torch.randn(K, K, dtype=torch.bfloat16, device=dev)
    # size a memory-bound buffer (sum-reduce) to ~TARGET_MS: bytes ~ target*BW
    membuf = torch.randn(48 * 1024 * 1024, dtype=torch.bfloat16, device=dev)  # 96MB

    def spacer(mode):
        if mode == "gemm":
            return A @ B
        if mode == "memread":
            return membuf.sum()
        return None

    def time_kernel(fn, iters=100, warm=20):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize(); dist.barrier()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    # calibrate spacer times
    t_gemm = time_kernel(lambda: (A @ B))
    t_mem = time_kernel(lambda: membuf.sum())

    def ar_only():
        dist.all_reduce(x)

    def ar_after(mode):
        spacer(mode)
        dist.all_reduce(x)

    # AR time measured with events around the WHOLE (spacer+AR); subtract spacer
    results = {}
    for mode in ["none", "gemm", "memread"]:
        for _ in range(20):
            ar_after(mode)
        torch.cuda.synchronize(); dist.barrier()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(100):
            ar_after(mode)
        e.record(); torch.cuda.synchronize()
        total = s.elapsed_time(e) / 100
        spacer_t = {"none": 0.0, "gemm": t_gemm, "memread": t_mem}[mode]
        ar_t = total - spacer_t
        results[mode] = (total, ar_t)

    if rank == 0:
        msg = NREQ * HIDDEN * 2
        print(f"# world={world}  msg={msg/1e6:.3f}MB  spacer_target={TARGET_MS}ms")
        print(f"# calibrated spacers: gemm={t_gemm:.3f}ms  memread={t_mem:.3f}ms")
        print(f"# {'mode':8} {'total_ms':>9} {'AR_ms':>8} {'AR_GB/s':>8}")
        for mode in ["none", "gemm", "memread"]:
            total, ar_t = results[mode]
            bw = msg / (ar_t / 1e3) / 1e9 if ar_t > 0 else 0
            print(f"  {mode:8} {total:9.3f} {ar_t:8.3f} {bw:8.2f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
