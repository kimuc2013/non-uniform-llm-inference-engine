"""SOLUTION probe for the in-decode effective AR bandwidth (pre-serving, no model, no
serving data). The plain/spaced/pipelined AR benches all measure ~1.1 GB/s, but real
decode runs it ~2x faster. Hypothesis: the speed-up is a CUDA-GRAPH-level effect — the
whole decode step ([per-layer compute -> AllReduce] x L) is captured as ONE graph and
replayed, so consecutive layers' kernels stream with no per-op launch/sync gaps and the
AllReduce's network wait overlaps the graph's queued work. A single-AR or single-spacer
bench can't see this; a captured L-layer chain can.

Measures, at world=8 (4+4) via torchrun:
  t_full = replay of the captured [matmul->AR->matmul->AR] x L chain
  t_comp = same chain with the AllReduces removed (compute only)
  exposed_AR = (t_full - t_comp) / (2L)  -> effective per-AR time IN the graph
  eff_bw = msg / exposed_AR
Compared against the isolated back-to-back AR (2L on one buffer) = the ~1.1 GB/s regime.
If eff_bw >> 1.1, the graph chain is the correct pre-serving source for ar_bw_gbs.

Static buffers throughout (CUDA-graph capture requires fixed addresses).
"""
import os, torch, torch.distributed as dist

H = int(os.environ.get("AR_HIDDEN", "8192"))
B = int(os.environ.get("AR_BATCH", "64"))          # decode batch -> AR msg = B*H*2
L = int(os.environ.get("AR_LAYERS", "80"))         # layers; 2 AR/layer


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local); dev = f"cuda:{local}"

    # per-layer decode compute = weight-streaming projections (memory-bound at B=64).
    # Wa ~ attn out-proj, Wf ~ ffn down-proj. Static weights + ping-pong activations.
    Wa = torch.randn(H, H, dtype=torch.bfloat16, device=dev)
    Wf = torch.randn(H, H, dtype=torch.bfloat16, device=dev)
    a = torch.randn(B, H, dtype=torch.bfloat16, device=dev)
    b = torch.empty(B, H, dtype=torch.bfloat16, device=dev)

    def chain(with_ar):
        for _ in range(L):
            torch.matmul(a, Wa, out=b)
            if with_ar: dist.all_reduce(b)
            torch.matmul(b, Wf, out=a)
            if with_ar: dist.all_reduce(a)

    def capture(with_ar):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): chain(with_ar)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize(); dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): chain(with_ar)
        return g

    def timeit(fn, iters=50, warm=10):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); dist.barrier()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
        for _ in range(iters): fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    g_full = capture(True)
    g_comp = capture(False)
    t_full = timeit(g_full.replay)
    t_comp = timeit(g_comp.replay)

    # isolated reference: 2L back-to-back ARs on one buffer (dependency = serial), graphed
    x = torch.randn(B, H, dtype=torch.bfloat16, device=dev)
    def iso():
        for _ in range(2 * L): dist.all_reduce(x)
    s2 = torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s2):
        for _ in range(3): iso()
    torch.cuda.current_stream().wait_stream(s2); torch.cuda.synchronize(); dist.barrier()
    g_iso = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_iso): iso()
    t_iso = timeit(g_iso.replay)

    if rank == 0:
        n_ar = 2 * L; msg = B * H * 2
        exposed = (t_full - t_comp) / n_ar                 # ms/AR in the chain
        iso_per = t_iso / n_ar
        bw = msg / (exposed / 1e3) / 1e9 if exposed > 0 else 0
        bw_iso = msg / (iso_per / 1e3) / 1e9 if iso_per > 0 else 0
        print(f"# graph-chain L={L} B={B} H={H} msg={msg/1e6:.3f}MB world={world}")
        print(f"# t_full={t_full:.2f}ms  t_comp={t_comp:.2f}ms  t_iso={t_iso:.2f}ms")
        print(f"# ISOLATED   per_AR={iso_per*1e3:7.1f}us  bw={bw_iso:6.2f} GB/s")
        print(f"# GRAPHCHAIN exposedAR={exposed*1e3:7.1f}us  bw={bw:6.2f} GB/s   boost={bw/bw_iso if bw_iso else 0:.2f}x")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
