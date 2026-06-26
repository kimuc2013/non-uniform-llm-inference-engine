"""Direct measurement of the KV-read bandwidth penalty (kappa) — independent of the
LLM-throughput fit. Measures, on one GPU:
  - weight-stream BW: peak HBM bandwidth reading a large CONTIGUOUS bf16 tensor.
  - KV-read BW: the REAL vLLM paged_attention_v1 kernel reading a paged KV cache
    (16-token blocks, scattered block table) — read-dominated, so time ~ KV read.
kappa = KV_read_BW / weight_stream_BW.  Run: python kv_bw_microbench.py [cuda:0]
"""
import sys, torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
torch.cuda.set_device(DEV)
DT = torch.bfloat16


def _time(fn, iters):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1e3   # seconds/iter


def weight_stream_bw(gb=6.0, iters=40):
    n = int(gb * 1e9 / 2)
    x = torch.randn(n, dtype=DT, device=DEV)
    t = _time(lambda: x.sum(), iters)
    return (n * 2) / t / 1e9


def kv_read_bw(num_kv_heads, head_size, block_size, batch, kv_len, scattered=True, iters=40):
    bps = (kv_len + block_size - 1) // block_size
    num_blocks = batch * bps
    # FlashAttention paged kv cache: [num_blocks, page_block_size, num_kv_heads, head_size]
    k = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=DT, device=DEV)
    v = torch.randn(num_blocks, block_size, num_kv_heads, head_size, dtype=DT, device=DEV)
    q = torch.randn(batch, num_kv_heads, head_size, dtype=DT, device=DEV)   # 1 query token / seq
    if scattered:
        bt = torch.randperm(num_blocks, device=DEV).to(torch.int32).view(batch, bps).contiguous()
    else:
        bt = torch.arange(num_blocks, device=DEV, dtype=torch.int32).view(batch, bps).contiguous()
    cu_q = torch.arange(batch + 1, device=DEV, dtype=torch.int32)
    seqused_k = torch.full((batch,), kv_len, dtype=torch.int32, device=DEV)
    scale = 1.0 / (head_size ** 0.5)

    def call():
        flash_attn_varlen_func(q, k, v, max_seqlen_q=1, cu_seqlens_q=cu_q,
                               max_seqlen_k=kv_len, seqused_k=seqused_k, block_table=bt,
                               softmax_scale=scale, causal=True)
    t = _time(call, iters)
    kv_bytes = batch * kv_len * num_kv_heads * head_size * 2 * 2   # k+v, bf16
    return kv_bytes / t / 1e9, t * 1e3


def main():
    name = torch.cuda.get_device_name(DEV)
    w = weight_stream_bw()
    print(f"\n=== {DEV}  {name} ===")
    print(f"weight-stream BW (contiguous read): {w:.0f} GB/s")
    print(f"{'config':38s} {'KV_BW':>8s} {'kappa':>7s} {'ms':>7s}")
    # vary head config + context + scattered vs sequential
    cfgs = [
        ("n_kv=8 d=128 ctx=2048 b=512 scat", 8, 128, 16, 512, 2048, True),
        ("n_kv=8 d=128 ctx=2048 b=512 seq ", 8, 128, 16, 512, 2048, False),
        ("n_kv=8 d=128 ctx=1024 b=512 scat", 8, 128, 16, 512, 1024, True),
        ("n_kv=1 d=128 ctx=2048 b=512 scat", 1, 128, 16, 512, 2048, True),
        ("n_kv=56 d=128 ctx=512 b=256 scat", 56, 128, 16, 256, 512, True),  # MHA (opt-like)
    ]
    for label, nkv, hd, bs, b, ctx, scat in cfgs:
        try:
            kv, ms = kv_read_bw(nkv, hd, bs, b, ctx, scattered=scat)
            print(f"{label:38s} {kv:7.0f}  {kv/w:6.3f} {ms:7.2f}")
        except Exception as ex:
            print(f"{label:38s}  ERR {str(ex)[:50]}")
    print(f"\n(model's fitted kappa = 0.316; weight BW ref in model = "
          f"{'1400' if 'Blackwell' in name else '707' if 'Ada' in name or 'L40' in name else '?'} GB/s)")


if __name__ == "__main__":
    main()
