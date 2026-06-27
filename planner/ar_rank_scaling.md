# Cross-node AllReduce rank-scaling — measured surface

Source: `planner/ar_microbench.py`, CUDA-graph capture (launch overhead removed,
in-decode-representative), cross-node (head Blackwell + worker Ada over IB),
2026-06-27. Launched via torchrun `--nnodes=2 --nproc_per_node={1,2,4}` (1+1, 2+2,
4+4 layouts → 2/4/8 ranks). `algbw = message / AR_time`.

## Algorithm bandwidth (GB/s) vs (ranks-per-node, message size)

| msg (MB) | n_local=1 (1+1) | n_local=2 (2+2) | n_local=4 (4+4) |
|---------:|----------------:|----------------:|----------------:|
| 0.066    | 0.6             | 0.4             | —               |
| 0.131    | 1.0             | 0.8             | —               |
| 0.262    | 1.2             | 1.0             | —               |
| 0.524    | 1.5             | 1.1             | 1.0             |
| 1.049    | **6.0**         | **4.7**         | **1.1**         |
| 1.573    | —               | —               | 1.1             |
| 2.097    | 9.1             | 6.7             | —               |

## Two effects baked in

1. **NCCL LL→Simple protocol transition (~1 MB).** Below ~1 MB every layout is
   latency-bound at ~1 GB/s; above it, bandwidth opens up. This is why an 8 B
   model (TP decode AR ≈ 0.5 MB) sees *no* small-layout speed-up while a 70 B
   model (≈ 1.05 MB) does — message size, not just topology, gates the cost.

2. **n_local (Ada PCIe/NIC funnel) contention.** Above the threshold the single
   per-node IB NIC + shared PCIe does NOT parallelize across local GPUs, so the
   bandwidth collapses as n_local grows: 6.0 / 4.7 / 1.1 GB/s for 1 / 2 / 4
   ranks-per-node. The 4+4 (n_local=4) case is ~5× slower than 1+1.

## Why this matters for the planner

The old `t_allreduce_ms` inter-node term was **independent of n_local** (it assumed
the per-node NIC sharing cancels exactly). Comparing formula vs measured at 1.05 MB:

| n_local | model_ms (old) | meas_ms | error |
|--------:|---------------:|--------:|------:|
| 1       | 0.272          | 0.174   | +56%  |
| 2       | 0.300          | 0.223   | +34%  |
| 4       | 0.328          | 0.982   | −67%  |

→ the model **over-charged** small layouts (under-predicted TP at 1+1/2+2) and
**under-charged** the 8-rank case (over-predicted TP8 at 4+4). The fix encodes the
table above as `_ISO_AR_SURFACE` and scales the AR bandwidth by the measured ratio
to the n_local=4 @ ~1 MB anchor (so the calibrated 4+4 predictions are unchanged).
