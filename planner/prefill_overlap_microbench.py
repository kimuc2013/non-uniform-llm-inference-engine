"""Pre-serving prefill_overlap probe (single GPU, no serving). prefill (tensor-core)
and decode (HBM) are disjoint resources -> under chunked continuous batching the
smaller phase hides under the larger. We measure the weight-amortization ratio r of a
prefill GEMM (how much of its time is weight-load vs compute): a GEMM that is mostly
compute (r small) leaves little exposed, so prefill_overlap = 1 - r is the fraction of
prefill that overlaps decode. Random weights, CUDA-graph, no model. Prints
'MEASURED prefill_overlap=<v>'."""
import torch, os
H = 8192
def t(fn, it=50, w=15):
    for _ in range(w): fn()
    torch.cuda.synchronize(); s,e=torch.cuda.Event(True),torch.cuda.Event(True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it
dev="cuda:0"
# compute-bound (large M): weight reused across many rows -> compute dominates
Wt=torch.randn(H,H,dtype=torch.bfloat16,device=dev)
big=torch.randn(4096,H,dtype=torch.bfloat16,device=dev); ob=torch.empty(4096,H,dtype=torch.bfloat16,device=dev)
one=torch.randn(1,H,dtype=torch.bfloat16,device=dev); oo=torch.empty(1,H,dtype=torch.bfloat16,device=dev)
t_big=t(lambda: torch.matmul(big,Wt,out=ob))     # amortized (compute-bound)
t_one=t(lambda: torch.matmul(one,Wt,out=oo))     # weight-load-bound (1 row)
# per-row compute time = t_big/4096; weight-load-bound floor = t_one. r = load/total
r = min(1.0, t_one/(t_big))            # fraction of a small-batch GEMM that is weight-load
po = round(max(0.0, 1.0 - r), 2)
print(f"# t_big(4096 rows)={t_big:.3f}ms t_one(1 row)={t_one:.3f}ms r={r:.3f}")
print(f"MEASURED prefill_overlap={po}")
