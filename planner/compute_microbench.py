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


def prefill_tflops(dev, iters=50, reps=3):
    shapes = [(8192, 8192, 8192), (8192, 14336, 4096)]   # square + FFN-like
    best = 0.0
    for (M, K, N) in shapes:
        a = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        b = torch.randn(K, N, dtype=torch.bfloat16, device=dev)
        c = torch.empty(M, N, dtype=torch.bfloat16, device=dev)
        for _ in range(reps):
            t = _time(lambda: torch.matmul(a, b, out=c), iters)
            best = max(best, (2 * M * N * K) / t / 1e12)
    return best


def main():
    dev = "cuda:0"
    name = torch.cuda.get_device_name(0)
    gtype = "blackwell" if ("PRO 6000" in name or "B" in name and "6000" in name) else "ada"
    bw = membw_gbs(dev)
    tf = prefill_tflops(dev)
    r = REF.get(gtype, {})
    print(f"GPU={name}  type={gtype}")
    print(f"  membw_decode_gbs  raw_measured={bw:7.0f}  planner_effective={r.get('membw','?'):>5}  "
          f"ratio(eff/raw)={r.get('membw',0)/bw:.2f}")
    print(f"  prefill_tflops    raw_measured={tf:7.0f}  planner_effective={r.get('tflops','?'):>5}  "
          f"ratio(eff/raw)={r.get('tflops',0)/tf:.2f}")


if __name__ == "__main__":
    main()
