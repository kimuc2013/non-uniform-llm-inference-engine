# Deployment-Time Analytical Planner — Mathematical Specification

This document is the definitive mathematical statement of the planner: every equation,
every symbol, where every number comes from, and why each form is universally valid
rather than tuned to this cluster. It transcribes the code in `perf_planner.py` exactly
(function names cited per section).

---

## 0. First Principles

**P1 — Measurement-only.** Every parameter is obtained at deployment time from
*pre-serving* measurement on the target cluster (microbenchmarks and dedicated
calibration runs), from system structure (byte widths, layer counts), or from a
citable external source. **No value may descend from previously-served sweep data.**
Served results are used *only* as evaluation ground truth (oracle/regret), never to
set a parameter. Choosing between two candidate values by which better reproduces
serving *is fitting* and is forbidden.

**P2 — No fixed constants.** No *free parameter* is a universal scalar: each is a
function of the model (layers, hidden, heads, experts), the hardware (per-GPU
roofline, link bandwidths, topology), or the system software (engine host floor,
overlap efficiency), and is re-measured or re-derived when any of those change.
The hardware fingerprint (GPU names/counts/memory, driver/CUDA/NCCL, topology
hash) keys the calibration cache; model-dependent parameters additionally key on
the deployment model.

*Structural constants are not parameters.* Numbers that are **derivations with
zero degrees of freedom** given the algorithm, model, or dtype — 2 all-reduces per
decoder layer (attention + MLP projections), the ring factor `2(k−1)/k`, 2 FLOPs
per parameter per token (multiply–accumulate), the factor 2 for K and V tensors,
`pp−1` stage boundaries, 2-byte bf16 widths, hop counts from the topology — are
required by correctness and are *not* subject to this principle. The forbidden
class is a **free parameter set by hand**: a scalar whose value answers to nothing
but "it seemed to work" (a `+0.3 ms` "python constant", an unexplained `0.15`).
The test for any number in the code: its value must be justified by (a) structural
derivation, (b) this deployment's measurement, or (c) a citable source — never by
agreement with served results.

**P3 — Universality.** Every formula is an instance of a standard, mechanism-level
model — roofline, α–β communication, max-plus pipeline algebra — whose form does not
depend on this cluster. Cluster-specific behaviour enters *only* through measured
parameter values, never through the functional form.

---

## 1. Notation

| symbol | meaning | source |
|---|---|---|
| `L` | total decoder layers of model *m* | model config (structural) |
| `h` | hidden size | structural |
| `L_s` | layers on PP stage `s` (layer split) | decision variable |
| `tp, pp` | tensor-/pipeline-parallel degrees, `tp·pp = world` | decision variable |
| `B` | decode batch on a stage (requests) | workload |
| `n` | requests in flight (`n_req`) | workload |
| `K̄` | mean KV context length = `in + out/2` | workload |
| `b_W, b_A, b_KV` | bytes/weight, /activation, /KV element (2,2,2 for bf16) | structural |
| `β_g` | HBM bandwidth of GPU *g* (GB/s) | **measured** (`compute_microbench`) |
| `π_g` | dense bf16 GEMM throughput of GPU *g* (TFLOP/s) | **measured** (`compute_microbench`) |
| `M_g` | device memory of GPU *g* | **measured** (nvidia-smi) |
| `α_ib, β_ib` | inter-node AR latency / effective bandwidth | **measured** (`ar_microbench`, `graph_chain`) |
| `𝔅(V)` | measured message-size bandwidth curve (AR surface) | **measured** (`ar_microbench`) |
| `α_x, β_x` | intra-node AR latency / bandwidth per node type | **measured** (`run_intra_calib`) |
| `ρ_dec` | decode AR↔compute overlap fraction | **measured** (`graph_chain`) |
| `ρ_pre` | prefill AR↔GEMM overlap fraction | prior, measurement pending (§11) |
| `ρ_ph` | prefill↔decode phase overlap | **measured** (`prefill_overlap_microbench`) |
| `η` | PP cross-stage overlap efficiency | **measured per model** (two-run identification, §10) |
| `F` | engine host floor per decode step (ms) | **measured per model** (TP-only twin, §10) |
| `c_mb, c_chunk` | per-microbatch / per-chunk host cost | raw dispatch probe × structural counts (§10) |
| `P_r` | parameter count resident on rank *r* | structural (exact count, `params_on_rank`) |

---

## 2. Communication model (`t_allreduce_ms`)

### 2.1 Intra-node ring (single-node TP group, size *k*)

Standard ring all-reduce moves `2(k−1)/k · V` bytes per GPU with `2(k−1)` latency hops:

```
T_intra(V, k) = 2(k−1)/k · V / β_x + 2(k−1)·α_x
```

`(α_x, β_x)` are measured *per node type* (this cluster: PCIe on both nodes — the
topology query shows no NVLink; nothing assumes NVLink).

### 2.2 Cross-node two-tier AR (group spans `n_nodes`, `n_local` ranks/node)

NCCL performs intra-node reduce-scatter/all-gather plus one inter-node ring:

```
T_ar(V) = T_intra-part + T_inter + latency
T_intra-part = 2(n_local−1)/n_local · V / β_x
T_inter      = 2(n_nodes−1)/n_nodes · V / 𝔹(V)
latency      = 2(n_nodes−1)·α_ib + 2(n_local−1)·α_x
```

**Radix-independence (a theorem, not a fit).** Per-GPU inter-node traffic is
`2(n_nodes−1)/(n_nodes·n_local)·V`, but the `n_local` GPUs of a node share ONE NIC,
so the per-NIC volume is the product — `2(n_nodes−1)/n_nodes·V` — *independent of
`n_local`*. Hence a single measured per-node inter-node bandwidth suffices;
the apparent low-radix speedup in the isolated microbench (LL-protocol artifact)
must not be priced in. The message-size dependence is taken from the measured
anchor-radix curve:

```
𝔹(V) = β_ib · 𝔅_anchor(V) / 𝔅_anchor(V_ref)
```

with `β_ib` the graph-chain effective bandwidth at the reference message
(`V_ref ≈ 1 MB`) — an *in-situ* decode-representative measurement, and
`𝔅_anchor` the isolated surface at the largest measured radix.

### 2.3 Monotonicity guard (`_monotonize_surface`)

Physical requirement: AR *time* is non-decreasing in message size
(`d/dV [V/𝔹(V)] ≥ 0` ⟺ 𝔹 grows at most linearly in V). The raw measured surface
violates this across NCCL's LL→Simple switch (~2 MB, a ~5× bandwidth cliff); fed
raw it would make decode *faster at higher batch* (non-physical) and mis-rank
configs. The guard caps each point:

```
𝔅(V_i) ← min(𝔅(V_i), 𝔅(V_{i−1}) · V_i/V_{i−1})
```

This is a *shape constraint from physics*, applied to measured data — not a refit
(raw values are preserved in the calibration file).

---

## 3. Decode stage time (`stage_time_decode_ms`)

Per stage `s` with layers `L_s`, batch `B`, ranks `r ∈ stage`:

```
weights:  W_r = P_r(L_s, splits) · b_W                (exact per-rank count)
KV read:  KV_r = B · K̄ · L_s · 2 · kv_r · d_head · b_KV
memory:   t_mem,r = W_r/β_r + KV_r/β_r
compute:  t_flop,r = 2 · P_r^flop · B / π_r
roofline: t_comp = max_r max(t_mem,r , t_flop,r)      (slowest rank binds)
AR:       t_ar = 2 L_s · T_ar(B·h·b_A)               (2 ARs per layer)
stage:    t_s(B) = max(t_comp, t_ar) + (1 − ρ_dec) · min(t_comp, t_ar)
```

The last line is the standard partial-overlap interpolation: `ρ_dec = 0` → serial
(sum), `ρ_dec = 1` → fully hidden (max). `ρ_dec` is *measured* by the graph-chain
bench as `1 − exposed/isolated` per-AR time inside a captured [GEMV→AR]×L chain.
Non-uniform TP enters through per-rank `splits` (ffn columns, heads, kv heads) in
`P_r` — the roofline then binds on whichever rank the split overloads.

MoE: the weight *stream* charges every activated expert
(`E_act(B) = E(1−(1−k/E)^B)` in expectation), the FLOPs charge only `top-k` —
both structural.

---

## 4. Prefill stage time (`stage_time_prefill_ms`)

Per chunk of `C` tokens (chunked prefill, `C = T_CHUNK` engine constant):

```
t_flop,r = 2 · P_r^flop · C / π_r         (tensor-core bound)
t_mem,r  = W_r / β_r                      (one weight stream per chunk)
t_comp   = max_r max(t_flop,r , t_mem,r)
t_ar     = 2 L_s · T_ar(C·h·b_A)
t_s^pre(C) = t_comp + (1 − ρ_pre) · t_ar
```

`ρ_pre` (async-TP/sequence-parallel hiding of the large prefill ARs under GEMMs):
currently a documented prior pending its own graph-chain measurement at prefill
shape (§11) — the *only* parameter not yet measured on this cluster.

---

## 5. PP decode cycle (`predict`, decode branch)

With `n_mb = min(pp, n)` microbatches of `mb = n/n_mb` requests (affinity: the
stage time is affine in batch, so `n_mb · t_s(mb)` charges exactly `n_mb` weight
streams and `n` requests of linear work):

```
busy_s  = n_mb · ( t_s(mb) + c_mb )
t_send  = mb·h·b_A / β_p2p + α_p2p                    (stage-boundary activation)
cycle   = max_s busy_s                                 (bottleneck stage)
        + (1 − η) · ( Σ_s busy_s − max_s busy_s )      (exposed other-stage work)
        + (1 − η) · (pp − 1) · t_send                  (exposed boundary sends)
        + F                                            (engine host floor)
T_decode = out_len · cycle
```

This is max-plus pipeline algebra: `η = 1` collapses to the bottleneck stage
(perfect pipelining), `η = 0` to the serial sum. For `pp = 1` the cycle is simply
`t_s(n) + F`.

## 6. Prefill pipeline & phase blend (`predict`, prefill branch)

```
chunks: N_c = ⌈ n·in / C ⌉  (last chunk = remainder, no over-charge)
fill (first chunk):    Σ_s t_s^pre(C) + c_chunk
steady (later chunks): max_s t_s^pre + (1 − η)(Σ_s − max_s) + c_chunk
T_prefill = fill + Σ_steady
```

Phases overlap under chunked continuous batching (tensor-core vs HBM-bound —
disjoint resources):

```
T_total = max(T_prefill, T_decode) + (1 − ρ_ph) · min(T_prefill, T_decode)
TPS     = n · out / T_total
```

`ρ_ph` measured by the single-GPU weight-amortization probe.

## 7. Memory feasibility (`mem_feasible`)

Per rank: weights + KV at target batch + activation head-room ≤ `mem_util · M_g`,
all terms exact structural byte counts. Infeasible configs are discarded, not
penalized (feasibility is a constraint, not a cost).

## 8. Search (`plan`)

* TP row split (closed form): per-rank speed
  `v_r = dw·β_r/β_0 + (1−dw)·π_r/π_0`, allocate columns/heads ∝ `v_r`
  (quantized to head/GQA granularity). The decode weight
  `dw = T_decode/(T_decode+T_prefill)` is *predicted by the model itself* at the
  uniform config — no external constant.
* Layer split: full 1-D scan for `pp = 2` (all `L_0 + L_1 = L`), bounded scan for
  deeper pp.
* Candidates: TP-only (uniform + closed-form biased), all `tp×pp`
  factorizations with uniform/biased splits; rank by predicted TPS; ties by
  memory head-room.

---

## 9. What is measured, when, and why it identifies the parameter

| run (pre-serving) | measures | identifies |
|---|---|---|
| `compute_microbench` (per GPU type) | stream GB/s, best-of GEMM TFLOP/s | `β_g, π_g` |
| `ar_microbench` cross-node, n_local ∈ {1,2,4} × msg sweep | isolated AR surface | `𝔅(V)`, `α_ib` (small-msg intercept) |
| `graph_chain_ar_microbench` ([GEMV→AR]×L captured graph) | in-decode effective AR | `β_ib`, `ρ_dec` |
| `run_intra_calib` (single-node, per node type) | intra AR | `α_x, β_x` |
| `prefill_overlap_microbench` | GEMM weight amortization | `ρ_ph` |
| `dispatch_microbench` | raw host submit/launch µs | inputs to `c_mb, c_chunk` |
| **TP-only twin** (dedicated, `pp=1`, decode-clean) | step period `c₁` | **`F = c₁ − t_s(n)`** |
| **PP twin** (dedicated, `pp=2`, decode-clean) | step period `c₂` | **`η = 1 − (c₂ − F − b_max)/(b_rest + t_send)`** |

**Identification is algebraically exact.** The pp=1 model has one unknown (`F`)
given the roofline; the pp=2 model then has one unknown (`η`). Two dedicated runs,
two unknowns — no degeneracy, no prior. Both are per-model (P2): `F` scales with
the engine's per-step work for *that* model and batch; `η` depends on how much of
*that* model's compute can hide the pipeline bubble.

**Validity domain of η (honest caveat).** The inversion attributes the non-floor,
non-bottleneck residue of the step to pipeline exposure. When a model's stage time
is dominated by communication kernels rather than compute (measured directly from
the trace: compute-only union busy ≪ step), the residue conflates comm exposure
with pipeline exposure and the recovered η is an *effective* lower bound, not a
physical overlap fraction (the fork's directly-traced cross-stage overlap is
56–78%). The trace breakdown (comm:compute ratio, per-stage GPU utilisation) is
therefore recorded next to every η and the planner warns when η is transferred
across models.

---

## 10. Provenance ledger (P1 compliance)

Every scalar the cost model consumes, and its origin:

| parameter | origin | class |
|---|---|---|
| `β_g, π_g, M_g` | microbench / nvidia-smi, this deployment | measured |
| `𝔅(V), α_ib` | ar_microbench, this deployment | measured |
| `β_ib, ρ_dec` | graph_chain, this deployment | measured |
| `α_x, β_x` | intra calib, this deployment | measured |
| `ρ_ph` | gemm probe, this deployment | measured |
| `F, η` | two dedicated decode-clean runs, per model | measured |
| `b_W, b_A, b_KV, L, h, …` | model/dtype structure | structural |
| `C (T_CHUNK)` | engine configuration | structural |
| `β_p2p, α_p2p` | nominal IB class; bounded effect (≤0.3 ms/cycle) | documented prior |
| `ρ_pre` | async-TP literature; **measurement pending** | documented prior |
| `mem_util` | engine default (0.85), same as serving flag | structural |

Anything not in this table does not exist in the cost path. The retired serving
fit (`fit_planner.py`, `fitted_params.json`) is deleted from the load path.

## 11. Open measurements (tracked, not hidden)

1. **`ρ_pre`** — graph-chain at prefill shape (B≈2048) will replace the prior.
2. **`c_mb, c_chunk`** — currently raw-probe × structural-count estimates; a
   bq-sweep twin run would identify them exactly (adds equations, same method as §9).
3. **η in comm-bound regimes** — see validity domain above; the direct-trace
   cross-stage concurrency measurement is the correct instrument there.
4. **`ρ_dec` shape dependence** — measured at one (H, B); a small H-sweep of the
   graph-chain bench would make it model-covariant like everything else.


---

## 12. Workload specification: prefix-caching-aware prefill volume

vLLM's automatic prefix caching (default ON) prefills a shared prompt prefix ONCE.
The workload therefore carries a `prefill_unique_frac` ∈ (0,1]: the fraction of
`n·in` prefill tokens that are unique across the batch. The prefill volume in §6
becomes

```
total_in = max(in, n · in · prefill_unique_frac)
```

while the KV context (`K̄`, §3) is unchanged — every sequence still attends over
its full context during decode. This is a *workload-description* field (what the
traffic actually looks like), never a tuning knob.

**Evaluation consequence (found 2026-07-02).** The measurement harness sends the
IDENTICAL prompt to every request, so all uniform-shape sweep cells actually
served `frac = 1/n` traffic — the "prefill-heavy" cells were ~decode-only
(measured: TTFT 0.3–0.8 s, decode fills the wall; the modeled full-prefill AR
alone would exceed the measured wall — arithmetically impossible). Evaluations
now describe those cells with `frac = 1/n` (mixed streams: `frac = 1`, varied
shapes). Correcting the workload DESCRIPTION — not any model parameter — moved
overall regret 9.0%→7.5% and the mean uplift over uniform-TP +78.8%→+92.1%.
Future sweeps must use unique per-request prefixes to measure true prefill.

## 13. Search widening (no new parameters)

The TP-only candidate set includes, besides the closed-form allocation of §8, a
coarse sweep of the fast-node bias axis (uniform → 2× uniform, step 256 columns):
the closed form is linear in the speed blend while the objective bends (AR/KV
floors), and the brute optimum can sit far from the linear seed (mistral123b:
+2.8%). ~14 extra `predict()` calls per cell (≈70 ms); the `tp_split_opt`
invariant now checks the *decision path* (plan's candidate set) against brute
force. Radix-aware inter-node AR level scaling was tried and REVERTED: the
isolated low-radix bandwidth boost is an LL-protocol artifact that does not
survive the serving NIC bottleneck (the in-serving instruments — graph-chain and
serving-trace AR kernels — see no such boost), and with the workload correction
in place it *worsened* 2+2 regret 2.4%→4.5% (evaluation used to validate
instrument transfer, not as a fit target).


## 14. Twin identification: what is exactly identified, and what is a band

The n_mb-sweep twins (§9) revealed a model-form limitation: `η` is itself
`n_mb`-dependent, so the constant-η two-equation system over {n_mb=2, n_mb=4}
is inconsistent (the deeper-microbatch twin beats even η=1 under the
n_mb-weight-stream busy model). The measurements still yield **hard bounds**:
requiring η ∈ [0,1] across the intra twin and both cross-node PP twins with a
*shared* engine per-microbatch cost gives `c_mb ≥ 3.8 ms` — the engine
per-microbatch cost is **milliseconds**, not the 3.4 µs CUDA-submit floor.
The adopted operating point is the **band minimum** (weakest overlap consistent
with all measurements — an a-priori selection rule, not a fit):
`c_mb = 3.83 ms → η(8B) = 0.635, η(70B) = 0.992` (each re-inverted from its own
PP twin with its measured `F`). Reported consequence: the previous operating
point (c_mb≈0, η(8B) clamped to 0 from a raw −1.38) was measurement-INCONSISTENT
and produced degenerate [31,1] layer splits for 8B; the honest point removes the
degeneracy (8B now picks balanced splits) at a ~1pp aggregate regret cost
(7.5%→8.6%) — the price of consistency, reported, not hidden. Exact point
identification needs an n_mb-dependent η model or a third instrument.
