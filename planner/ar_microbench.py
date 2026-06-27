"""Directly measure NCCL all-reduce cost vs message size — to replace the
degenerate fitted AR constants (ar_latency_us, ar_bw_gbs). Decode's per-layer AR
message is batch*hidden*2 bytes (bf16 hidden-state all-reduce).

Launch with torchrun across the cluster, e.g. (head + worker, 4 GPUs each):
  HEAD (node 0):   NCCL_SOCKET_IFNAME=ibp3s0  torchrun --nnodes=2 --nproc_per_node=4 \
                     --node_rank=0 --master_addr=10.20.0.30 --master_port=29500 ar_microbench.py
  WORKER (node 1): NCCL_SOCKET_IFNAME=ibp34s0 torchrun --nnodes=2 --nproc_per_node=4 \
                     --node_rank=1 --master_addr=10.20.0.30 --master_port=29500 ar_microbench.py
Single-node (intra, TP4): --nnodes=1 --nproc_per_node=4.
"""
import os, torch, torch.distributed as dist

HIDDEN = 8192          # Llama-70B hidden (AR message = batch*hidden*2 bytes)
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]
# AR_HIDDEN_SWEEP=1: fix batch, sweep hidden (4096=8B, 7168=opt30b, 8192=70B, 12288=Mistral
# + neighbours of 7168) — to test whether hidden=7168 is itself NCCL-anomalous vs a smooth
# bandwidth curve (i.e. opt30b's slow AR is the message/alignment, not overlap).
SWEEP_HIDDEN = os.environ.get("AR_HIDDEN_SWEEP", "0") == "1"
SWEEP_BATCH = 64
HIDDENS = [4096, 6144, 7168, 7680, 8192, 10240, 12288]


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    configs = [(SWEEP_BATCH, h) for h in HIDDENS] if SWEEP_HIDDEN else [(b, HIDDEN) for b in BATCHES]
    if rank == 0:
        print(f"# world={world}  mode={'HIDDEN-sweep@batch=%d' % SWEEP_BATCH if SWEEP_HIDDEN else 'BATCH-sweep@hidden=%d' % HIDDEN}")
        print(f"# {'batch':>6} {'hidden':>6} {'msg_MB':>8} {'AR_ms':>8} {'lat_us/AR':>10} {'algbw_GB/s':>11}")
    use_graph = os.environ.get("AR_CUDA_GRAPH", "0") == "1"
    if rank == 0 and use_graph:
        print("# (CUDA-graph capture: launch overhead removed — in-decode-representative)")
    for b, hid in configs:
        x = torch.randn(b, hid, dtype=torch.bfloat16, device=dev)
        for _ in range(15):
            dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier()
        if use_graph:
            # capture one all-reduce in a CUDA graph, replay it (no per-call launch)
            sstream = torch.cuda.Stream()
            sstream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(sstream):
                for _ in range(3):
                    dist.all_reduce(x)
            torch.cuda.current_stream().wait_stream(sstream)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                dist.all_reduce(x)
            torch.cuda.synchronize(); dist.barrier()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(200):
                g.replay()
            e.record(); torch.cuda.synchronize()
            t_ms = s.elapsed_time(e) / 200
        else:
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(80):
                dist.all_reduce(x)
            e.record(); torch.cuda.synchronize()
            t_ms = s.elapsed_time(e) / 80
        if rank == 0:
            msg = b * hid * 2
            algbw = msg / (t_ms / 1e3) / 1e9
            print(f"  {b:>6} {hid:>6} {msg/1e6:8.3f} {t_ms:8.3f} {t_ms*1e3:10.1f} {algbw:11.1f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
