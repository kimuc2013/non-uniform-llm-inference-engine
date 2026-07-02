"""Pre-serving dispatch-overhead probe (single GPU, no serving). Measures the HOST-side
CPU cost the engine pays per decode step / microbatch / prefill chunk that is NOT GPU
work: CUDA-graph replay submit + a sampling-like op + Python dispatch. step_floor = host
time to submit one captured graph + sample; c_mb ~ per-extra-launch host latency;
c_chunk ~ scheduler+prepare host span (proxied). Prints MEASURED lines."""
import time, torch
dev="cuda:0"; H=8192
x=torch.randn(64,H,dtype=torch.bfloat16,device=dev); w=torch.randn(H,H,dtype=torch.bfloat16,device=dev); o=torch.empty(64,H,dtype=torch.bfloat16,device=dev)
# capture a tiny "decode-step-like" graph (a couple GEMVs + a sample)
s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        torch.matmul(x,w,out=o); torch.argmax(o,dim=-1)
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    torch.matmul(x,w,out=o); torch.argmax(o,dim=-1)
# HOST time to SUBMIT the replay (not GPU exec): submit N replays back-to-back w/o sync,
# measure wall/N minus the async gpu time -> the per-step host dispatch floor.
def host_submit(n):
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(n): g.replay()
    t_submit=(time.perf_counter()-t0)/n*1e3   # ms host to submit one replay
    torch.cuda.synchronize()
    return t_submit
for _ in range(20): g.replay()
torch.cuda.synchronize()
sub=min(host_submit(200) for _ in range(5))
# per-extra-kernel-launch host latency (single tiny launch)
def one_launch():
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(500): torch.argmax(o,dim=-1)
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/500*1e3
lat=min(one_launch() for _ in range(5))
# RAW measurements ONLY (audit 2026-07-02): the old derivation (+0.3ms "python constant",
# lat*3, lat*20, max() floors) was hand magic — removed. The ENGINE host floor F is a
# different physical quantity (scheduler+sampler python per step) and is identified by
# the per-model TP-only twin (extract_overlap_eta pp=1), NOT by this probe. This probe
# measures the CUDA-side host costs: c_mb ~= one extra graph-replay submit per microbatch
# (structural count: 1), and the per-launch latency for structural-count derivations.
print(f"MEASURED_RAW graph_submit_host_ms={sub:.4f} per_launch_host_ms={lat:.4f}")
print(f"MEASURED c_mb_ms={round(sub,4)}")   # = 1 replay submit per extra microbatch (structural)
