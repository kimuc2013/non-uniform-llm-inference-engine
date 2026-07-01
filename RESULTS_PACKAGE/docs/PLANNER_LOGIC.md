# Planner Logic — Non-Uniform TP/PP Serving Planner

Analytical throughput model in `planner/perf_planner.py`. It predicts decode +
prefill throughput for every `TP×PP` factorization (and non-uniform split) **without
running the model**, then returns the argmax. Every hardware quantity is either
**measured** by a microbench or **fit** from serving data — there is no hand-typed
magic constant.

Symbols: `B`=batch (concurrent requests in a step), `L`=layers in a stage,
`N`=#nodes a group spans, `n_local`=ranks-per-node, `msg`=AllReduce message bytes
= `B·hidden·2` (bf16 hidden-state).

---

## 1. Cost-model equations

### 1.1 Decode — per-stage step time  (`stage_time_decode_ms`)

Each rank's compute is the larger of HBM-bound and tensor-core-bound work:

```
t_mem  = W_bytes/membw  +  KV_bytes/(membw·kv_scale)        # weight stream + KV read
t_flop = 2·params·B / tflops
t_comp = max over ranks of  max(t_mem, t_flop)              # the slow rank sets the stage
```
`W_bytes` = per-rank resident weights (for MoE, the *activated* experts `E_active(B)`);
`KV_bytes = B·kv_avg·L·2·kv_heads·head_dim·2`.

Tensor-parallel AllReduce, twice per layer (after attention-out and FFN-out):
```
t_ar = 2·L · t_allreduce(msg = B·hidden·2)
```
Compute (HBM/tensor-core) and AR (network) use **disjoint resources** and partially
overlap:
```
stage = max(t_comp, t_ar) + (1 − ρ)·min(t_comp, t_ar)       # ρ = decode_ar_overlap ∈ [0,1]
```
ρ→1 ⇒ AR fully hidden behind compute; ρ→0 ⇒ serial.

### 1.2 AllReduce — hierarchical 2-tier  (`t_allreduce_ms`)

A cross-node TP group does **not** pay a flat ring over all ranks: NCCL does an
intra-node reduce-scatter/all-gather over NVLink/PCIe, then one inter-node ring over IB.
```
t_intra = 2(n_local−1)/n_local · msg / intra_bw              # NVLink (head) / PCIe (worker)
t_inter = 2(N−1)/N · msg / ar_bw_gbs                         # IB — RADIX-INDEPENDENT
lat     = 2(N−1)·α_ib + 2(n_local−1)·α_intra
t_allreduce = t_intra + t_inter + lat
```
**The inter-node term is radix-independent**: the per-node IB volume is `2(N−1)/N·msg`
*regardless of `n_local`* (the `/n_local` per-GPU traffic cancels against the `n_local`
GPUs sharing the one per-node NIC), and the NIC bandwidth does not change with how many
local GPUs feed it. So a single measured IB bandwidth `ar_bw_gbs` (× a message-size
curve for the NCCL LL→Simple opening near ~1 MB) covers every radix. *Earlier* the model
scaled this by a per-radix isolated-microbench surface that claimed `n_local=2` was ~4×
faster than `n_local=4` — an LL-protocol artifact that does not survive in serving; it
mispriced 2+2 cross-node TP4 and was removed.

### 1.3 Pipeline parallelism  (`predict`, pp>1)

`bq = pp` micro-batches, capped at `B`. Each stage re-streams its weights once per
micro-batch (mem-bound work serializes on the same GPUs even under perfect overlap):
```
n_mb = min(pp, B);   mb = B / n_mb
busy_s = n_mb · ( stage_decode(mb) + c_mb )                  # per stage s
t_send = mb·hidden·2 / p2p_bw + p2p_lat                      # inter-stage activation
cycle  = max_s busy_s + (1−η)·(Σ busy_s − max_s busy_s)      # η = overlap_eta
                       + (1−η)·t_send·(pp−1) + step_floor
t_decode = out_len · cycle
```
(pp=1: `cycle = stage_decode(B) + step_floor`.)

### 1.4 Prefill (chunked) and total

Input is chunked (`T_CHUNK`); first chunk fills the pipe, later chunks cost the
bottleneck stage + exposed bubble:
```
t_prefill = fill + Σ_chunks [ max_s ts + (1−η)(Σ ts − max_s ts) + c_chunk ]
```
Prefill (tensor-core) and decode (HBM) run on disjoint resources under chunked
continuous batching and partially overlap:
```
T_total = max(t_prefill, t_decode) + (1 − prefill_overlap)·min(t_prefill, t_decode)
tput    = B·out_len / T_total
```

### 1.5 Search  (`plan`)

Enumerate **every** divisor factorization of `world = head+worker` (`TP×PP`), and for
each: the uniform split plus the closed-form non-uniform optima with a ±neighborhood —
**FFN-column bias** (TP), **layer-skew** (PP), Blackwell(head)-biased. `predict` each
feasible config (`mem_feasible`: per-rank weights+KV ≤ `gpu_mem·util`), return the
top-k by predicted tput. The reported pick is the raw argmax (no safety guard).

---

## 2. Measured vs. fit — zero hand-typed constants

| | quantity | source |
|---|---|---|
| **measured** (microbench) | per-GPU decode HBM bandwidth, prefill bf16 GEMM TFLOPS (per GPU type) | `compute_microbench.py` |
| **measured** | isolated cross-node AllReduce bandwidth surface + per-hop latency intercept | `ar_microbench.py` |
| **measured** | point-to-point send bw/latency (PP boundary) | microbench |
| **fit** (least-squares on measured serving throughput) | `ar_bw_gbs` (effective in-serving AR bw), `decode_ar_overlap`, `overlap_eta`, `prefill_overlap`, `step_floor`, `c_mb`, `c_chunk`, `kv_bw_scale`, `ar_latency_us` | `fit_planner.py` → `fitted_params.json` |

**Removed constants:** `AR_EFFECTIVE_FACTOR = 3.74` (a hand-typed multiplier that
faked the effective AR bandwidth) and the **per-radix surface shape** for the
inter-node term. The principle: *measure every hardware primitive a pre-serving bench
can reach; everything that only exists in the live decode loop (the AR pipeline-
concurrency boost, the compute/comm overlap efficiencies) is fit from serving data —
not a number someone typed.*

---

## 3. What the microbenches extract

- **`compute_microbench.py`** (per GPU type, Blackwell / Ada):
  - decode HBM bandwidth — effective GB/s of a large bf16 contiguous reduction
  - prefill TFLOPS — effective bf16 GEMM throughput
- **`ar_microbench.py`** (multi-node `torchrun`, `nproc_per_node ∈ {1,2,4}` = `n_local`):
  - **isolated** cross-node AllReduce bandwidth = `msg / kernel_time`, over (`n_local`,
    message size); plus the small-message latency intercept
  - **pipeline mode** (`AR_PIPELINE=K`): K independent in-flight AllReduces → *sustained*
    bandwidth; the gap vs. the serialized number is the in-decode pipeline-concurrency
    boost — **measured**, not assumed
  - launcher: `run_ar_bench.sh` (head local + worker SSH, serving-matched NCCL env)

> **Not pre-serving-measurable:** the *effective* in-serving AR bandwidth — the AR
> overlapping the live decode compute/schedule. No isolated bench reproduces it, so
> `ar_bw_gbs` is fit from serving throughput (`fit_planner.py`). That is the one place
> the number comes from data rather than a standalone measurement, and it is a fit, not
> a hand-typed constant.

---

## 4. Validation

`planner/regret_eval.py` (regret vs. measured oracle), `verify_vs_baseline.py`
(vs. uniform-TP baseline), `check_consistency.py` (invariants). After the radix-
independent AR fix: 4+4 picks byte-identical (mean regret 6.0%), 2+2 regret
17.6%→6.2%, 1+1 45.2%→33.9%; verify_vs_baseline 92/102 ≥ baseline, +84% mean uplift,
17/17 invariants.
