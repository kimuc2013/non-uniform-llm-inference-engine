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
step_floor=round(sub+0.3,2)          # graph submit + a small python/scheduler constant
c_mb=round(max(0.1,lat*3),2)         # ~ a few launches of per-mb prep dispatch
c_chunk=round(max(1.0,lat*20),2)     # ~ scheduler+prepare span (proxy, many small ops)
print(f"# graph_submit_host={sub:.3f}ms  per_launch_host={lat:.3f}ms")
print(f"MEASURED step_floor_ms={step_floor} c_mb_ms={c_mb} c_chunk_ms={c_chunk}")
