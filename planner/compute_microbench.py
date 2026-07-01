"""Measure raw per-GPU compute roofline: decode weight-stream HBM bandwidth and
prefill bf16 GEMM TFLOPS. Single GPU, no torchrun/NCCL. Run once per GPU type:
  head  : python planner/compute_microbench.py            (Blackwell cuda:0)
  worker: ssh esca@10.20.0.28 '<vllm_new py> .../compute_microbench.py'   (Ada)

Prints raw measured values next to the planner's current effective constants so the
isolated-vs-effective gap is visible (the planner's membw/tflops are serving-effective,
NOT raw peak — see hw_params.json notes).
"""
import torch

REF = {  # planner's current effective values (hw_params.json), for side-by-side
    "blackwell": {"membw": 1400, "tflops": 578},
    "ada": {"membw": 707, "tflops": 366},
}


def _time(fn, iters, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3   # seconds/iter


def membw_gbs(dev, gb=20.0, iters=50, reps=3):
    n = int(gb * 1e9 / 2)
    x = torch.randn(n, dtype=torch.bfloat16, device=dev)
    best = 0.0
    for _ in range(reps):
        t = _time(lambda: x.sum(), iters)
        best = max(best, (n * 2) / t / 1e9)
    return best


def prefill_tflops(dev, iters=50, reps=8):
    """Roofline bf16-input/fp32-accumulate GEMM TFLOPS (cuBLAS default acc). Sweep a
    range of large compute-bound shapes (square + FFN-like + attention-proj-like) and
    take the BEST achieved — the roofline. Many reps + long warmup so boost clocks are
    reached (we cannot lock clocks without sudo). c reused (out=) to avoid alloc noise."""
    shapes = [
        (8192, 8192, 8192), (16384, 8192, 8192), (16384, 16384, 8192),   # large square
        (8192, 14336, 4096), (8192, 4096, 14336),                        # FFN up/down
        (8192, 8192, 28672), (16384, 8192, 1024),                        # wide / skinny
    ]
    best = 0.0; best_shape = None
    for (M, K, N) in shapes:
        try:
            a = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
            b = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
            c = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
        except RuntimeError:
            continue                                     # skip if OOM on the smaller GPU
        for _ in range(reps):
            t = _time(lambda: torch.matmul(a, b, out=c), iters, warm=15)
            tf = (2 * M * N * K) / t / 1e12
            if tf > best:
                best, best_shape = tf, (M, K, N)
        del a, b, c; torch.cuda.empty_cache()
    return best, best_shape


def main():
    dev = "cuda:0"
    name = torch.cuda.get_device_name(0)
    gtype = "blackwell" if ("PRO 6000" in name or "B" in name and "6000" in name) else "ada"
    bw = membw_gbs(dev)
    tf, shp = prefill_tflops(dev)
    r = REF.get(gtype, {})
    print(f"GPU={name}  type={gtype}")
    print(f"  membw_decode_gbs  raw_measured={bw:7.0f}  planner_effective={r.get('membw','?'):>5}  "
          f"ratio(eff/raw)={r.get('membw',0)/bw:.2f}")
    print(f"  prefill_tflops    raw_measured={tf:7.0f}  planner_effective={r.get('tflops','?'):>5}  "
          f"ratio(eff/raw)={r.get('tflops',0)/tf:.2f}  best_shape={shp}")
    print(f"MEASURED {gtype} membw_gbs={bw:.0f} tflops={tf:.0f}")


if __name__ == "__main__":
    main()
