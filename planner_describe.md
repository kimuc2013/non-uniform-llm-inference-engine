# The Heterogeneous TP/PP Planner — a complete walkthrough

A study guide for the analytical planner in `planner/perf_planner.py` and friends.
Every formula here is **derived from first principles** (roofline, ring/hierarchical
all-reduce, pipeline bubble, water-filling) — the only fitted quantities are ~8
*effective* hardware/engine constants, calibrated **once per cluster**. Read this
top to bottom; each section gives the **intuition → the math → where it is in code
(`file:line`)**. The terse derivations live in `planner/PLANNER_SPEC.md`; this doc
is the pedagogical companion.

---

## 0. What problem does it solve?

We serve LLMs on a **heterogeneous, inter-node** GPU cluster:

- **Head node**: 4× Blackwell (96 GB HBM, NVLink between them) — *fast*.
- **Worker node**: 4× Ada (48 GB, **PCIe, no NVLink**) — *slow*.
- The two nodes are joined by **InfiniBand** (cross-node, slower than NVLink).
- We target every symmetric layout from **1+1 to 4+4** (world = 2, 4, 6, 8).

Given a **model**, a **workload** (input/output lengths + concurrency), the planner
picks the parallelization config that **maximizes serving throughput (TPS)** —
*without running the model*:

```
config = (TP, PP,                       # tensor- & pipeline-parallel degrees
          per-rank FFN / head / KV splits,   # NON-uniform TP: fast GPU holds more
          per-stage layer split)             # NON-uniform PP: fast stage holds more layers
```

**Thesis:** "hybrid parallelism is not always best." The optimal topology and the
value of non-uniform sharding are model-, workload-, and hardware-dependent, and a
small calibrated cost model picks within a few % of the measured optimum.

**Why it's not a lookup table or a learned black box:** the cost model is a closed-
form function of (model dims, hardware rates, workload). It predicts **unseen**
models (Mistral-123B pre-registered 3/3), **unseen** layouts (1+1/2+2 zero-refit,
regret 3.4/7.5%), and survives leave-one-model-out — impossible if it memorized.

---

## 1. Inputs — three dataclasses

### 1.1 `ModelSpec` — architecture → parameter *counts*  · `perf_planner.py:36-84`
Static dims (`n_layers, hidden, ffn_dim, n_q, n_kv, head_dim, vocab`) plus derived
**per-layer parameter counts** that are the "work" quantity the roofline divides by
a rate:

- `p_attn` = Wq + Wkv + Wo = `h·n_q·d_h + 2·h·n_kv·d_h + n_q·d_h·h`
  (GQA shrinks the KV projection by the group factor `g = n_q/n_kv`).
- `p_ffn` = `mats · h · ffn_dim` — `mats=3` for gated SwiGLU (gate+up+down), `2` for GELU.
- `p_embed` = `V·h` (tied) or `2·V·h` (untied: input table + lm_head are separate).
- `gqa_group = n_q/n_kv` — the head-bias **quantum** (you can only move whole KV groups).

> Intuition: ~82% of a layer's weights are FFN (`ffn_frac≈0.824` for 70B), ~18%
> attention. That's *why FFN bias is the dominant non-uniform-TP knob*.

### 1.2 `GpuType` + `HardwareSpec` — effective rates + topology · `perf_planner.py:87-186`
- `GpuType(name, tflops_prefill, membw_gbs, mem_gb)` — three **effective** rates per
  GPU class. *Effective*, not datasheet: cross-node decode is AR/overhead-bound, so
  achieved BW < peak. (Blackwell 1400 GB/s · 578 TFLOPS, Ada 707 · 366.)
- `HardwareSpec.nodes` = rank-ordered `((blackwell,4),(ada,4))`; ranks `[0,head)` are
  the fast node, `[head, world)` the slow node.
- Interconnect: cross-node IB (`ar_latency_us, ar_bw_gbs`), intra-node **NVLink**
  (`nvlink_ar_*`, for Blackwell) vs **PCIe** (`intra_ar_*`, for Ada).
- Helpers: `gpu_of_rank` (rank→GPU class, drives heterogeneity), `node_of_rank`,
  `intra_params(gpu)` → picks NVLink-vs-PCIe by GPU name.
- `load_hardware()` (`:156-186`) builds this from `hw_params.json`, then **overlays**
  `fitted_params.json` so the CLI predicts with the actual fit.

### 1.3 `Workload` — `in_len, out_len, n_req` · used throughout
`n_req` = concurrency (number of in-flight requests). `kv_avg = in_len + out_len/2`
is the **time-averaged** KV length over a decode (the cache grows linearly).

---

## 2. The cost model (the heart)

### 2.1 Per-rank roofline — decode vs prefill · `perf_planner.py:297-337`

The **roofline** model (Williams 2009): a kernel's time = `max(memory-time, compute-time)`.

**Decode** (`stage_time_decode_ms`, `:297-316`) is **memory-bound**: every token
streams the *whole* layer weight matrix from HBM, plus reads the KV cache.
```
t_mem  = (weight_bytes + KV_read_bytes) / membw          # ← dominates
t_flop = 2 · params · batch / tflops                     # GEMV, 2 FLOP/param/token
t_rank = max(t_mem, t_flop)
t_stage = max_r t_rank(r)  +  2·L_s · t_allreduce        # slowest rank gates the TP group
```
- `weight_bytes` is **batch-independent** (the matrix is read once per step regardless
  of how many sequences are batched) → this is *why decode TPS rises with batch until
  the KV/flop term catches up* — the **saturation batch**.
- `KV_read = batch·kv_avg·L_s·2·kv_r·d_h·b_kv` (k+v, only this rank's `kv_r` heads), **batch-linear**.
- `n_AR = 2·L_s` all-reduces per step (attention-out + FFN-down per layer).

**Prefill** (`stage_time_prefill_ms`, `:319-337`) is **compute-bound** (process `T_c`
tokens at once → high arithmetic intensity → tensor-core limited):
```
t_flop = 2·params·chunk_tokens / (tflops · prefill_tf_mult)   # ← dominates
t_mem  = weight_bytes / membw
t_stage = max(t_flop, t_mem) + (1 − prefill_ar_overlap)·2·L_s·t_allreduce
```
- The quadratic-attention term `2·head_r·d_h·Σsᵢ²` is **deliberately dropped** (<7% at
  in_len ≤ 1024).
- `prefill_ar_overlap=0.8`: the AR of large prefill chunks is ~80% hidden under the
  GEMMs (async-TP / sequence-parallel), so only 20% is charged.

### 2.2 `params_on_rank` + `embed_on_stage` · `perf_planner.py:226-247`
The single "work" scalar:
```
params_on_rank = L_s·(p_attn·head_r/n_q + p_ffn·ffn_r/ffn_dim) + embed/tp
```
Non-uniform TP lives in `head_r / ffn_r` (a biased fast rank holds more).
`embed_on_stage(m, pp, stage)`: `pp==1` → full `p_embed`; `pp>1` → exactly one `V·h`
on stage 0 (input table) + one `V·h` on the last stage (lm_head), 0 on interior
stages — **even for tied models** (PP can't share a weight across stages → replicate).
*(This was a bug: the old code charged the full `2·V·h` on both end stages, an
anti-PP bias on the bottleneck Ada stage. Fixed; guarded by consistency invariant #13.)*

### 2.3 AllReduce — ring → **hierarchical 2-tier** · `perf_planner.py:261-283`

This is the hetero-cluster crux. The α–β model of a collective: `t = bw_term + lat_term`.

**Single-node group** (`:261-275`): a bandwidth-optimal **ring** all-reduce
(Rabenseifner). Over N GPUs:
```
t = 2(N−1)/N · msg / bw   +   2(N−1) · α
```
- Bandwidth: each rank moves `(N−1)/N·msg` in reduce-scatter and again in all-gather.
- Latency: `2(N−1)` sequential steps. **`bw, α` come from `intra_params(gpu)`** —
  NVLink (800 GB/s, 4 µs) for a Blackwell group, PCIe (60 GB/s) for an Ada group.

**Cross-node group** (`:276-283`): a **2-tier** collective — NVLink/PCIe intra
reduce-scatter+all-gather, then an IB ring across nodes, then intra all-gather:
```
t = 2(n_local−1)/n_local · msg / bw_intra            # intra tier (NVLink RS+AG)
  + 2(n_nodes−1)/n_nodes · msg / bw_ib               # inter tier (IB ring, NIC shared by n_local GPUs)
  + 2(n_nodes−1)·α_ib + 2(n_local−1)·α_intra         # 2-tier latency
```
> **The decisive point:** a 4+4 TP8 group pays `2(n_nodes−1)=2` IB latency hops, **not**
> `2(Nt−1)=14`. The old flat-ring form over-charged cross-node decode AR ~6× (45 ms/token
> vs ~8 ms measured), which made the planner pick PP at *every* load and miss the
> measured low-concurrency TP champion. This hierarchical fix unblocked the crossover.

Two regimes fall out automatically: **decode** (small msg) is *latency-floor* dominated
by the few inter-node hops → cross-node TP is viable at low load; **prefill** (large
msg) is *IB-bandwidth* dominated.

### 2.4 `predict()` — config → TPS · `perf_planner.py:340-418`
```
mem_feasible? no → tps=0 (hard reject)
T_decode = out_len · T_cycle                          # one cycle = one token across all requests
T_prefill = chunked pipeline (fill + steady)
T_total  = max(T_pre, T_dec) + (1−ρ)·min(T_pre, T_dec)   # phase-overlap blend
TPS = n_req · out_len / T_total
```

**Decode cycle `T_cycle`:**
- `pp==1` (`:360`): `T_cycle = stage_time_decode(batch=n_req) + step_floor`.
- `pp>1` (`:363-379`): `n_mb = min(pp, n_req)` microbatches of `mb = n_req/n_mb`; each
  stage is busy `n_mb·(stage_time(mb) + c_mb)`; the steady-state pipeline cycle is
  ```
  T_cycle = max_s busy_s + (1−η)·(Σ busy − max) + (1−η)·t_send·(pp−1) + step_floor
  ```
  - `max_s busy` = bottleneck stage; `(1−η)·Σrest` = the exposed pipeline **bubble**;
    `η = overlap_eta` (our fork's overlap efficiency).
  - `mb` is a **float** but the count is exact: stage time is **affine** in batch, so
    `n_mb·t_s(n_req/n_mb)` charges `n_mb` weight-streams + exactly `n_req` of linear work.
    (Fixes the old `n_req//pp` that dropped the remainder.)

**Prefill pipeline** (`:382-401`): `n_chunks = ceil(total_in / 8192)`; the first chunk
is the **fill** (`Σ_s stage_time`), each later chunk costs `max_s stage_time + bubble`.
The **last chunk uses its true remainder** (not a full `T_CHUNK`) — fixes a sawtooth.

**Wall blend** (`:403-409`): prefill (tensor-core) and decode (HBM) run on disjoint
units and partially overlap under chunked continuous batching. `ρ=prefill_overlap`:
ρ=0 → serial, ρ=1 → smaller phase fully hidden.

### 2.5 `mem_feasible()` — hard reject · `perf_planner.py:425-445`
Per stage, per rank: `weights + KV + activations + overhead ≤ mem_gb·0.85`.
- KV uses the **max** context `(in_len+out_len)` (the cache must fit at peak), unlike
  the timing path which uses the time-average `kv_avg`.
- `act = 6·T_CHUNK·max(h, 2·ffn_r)·b_a`; `overhead = 3.5 GB` (CUDA ctx + NCCL + graph pool).
- This is what makes 70B infeasible at 1+1 and Mistral-123B infeasible at 2+2.

---

## 3. From cost to decision — optimize + search · `perf_planner.py:452-650`

### 3.1 Non-uniform TP split — `optimal_tp_splits` · `:464-517`
Goal: size per-rank FFN/head/KV so the **slow Ada ranks hold less weight**, equalizing
`t_rank` across the TP group. Minimize `max_r t_rank` s.t. `Σ f_r = ffn_dim`.
- **Case A (biasable attention)**: optimum is `frac_r ∝ speed_r`, where `speed` blends
  decode membw and prefill TFLOPS by `decode_weight`. Heads and FFN both allocated ∝ speed.
- **Case B (attention floor, e.g. no-GQA)**: heads stay uniform, **water-fill** the FFN
  columns — `f_r = (Λ − a_r)/c_r` where `a_r` is the fixed per-rank cost (attn weight +
  **KV read**) and `c_r` the per-FFN-column cost; `Λ` is the water level (`:510`).
- `round_quantized` (`:452-461`): Hamilton/largest-remainder apportionment to snap the
  continuous fractions onto hardware quanta (FFN tile 128, head quantum = GQA group).

### 3.2 Per-stage layer split — `optimal_layer_split` · `:520-585`
Allocate `n_layers` across PP stages to balance **stage busy times**. Key trick: it
**probes the cost model itself** by finite difference (build configs with L=1 and L=2,
read `stage_time`) to fit an **affine** model `t_s(L) = c_s·L + f_s`, then water-fills:
```
c_s·L_s + f_s = T*   s.t. ΣL_s = n_layers
⇒ T* = (n_layers + Σ f_s/c_s) / Σ(1/c_s),   L_s = (T* − f_s)/c_s
```
- `c_s` = per-layer slope (weight stream + KV + **per-layer AR**); `f_s` = intercept
  (embedding / lm_head GEMM + AR base). The last stage's lm_head makes `f_s` large there
  → it holds *fewer* layers. *Ignoring `f_s` under-skews; ignoring the AR-in-`c_s` (and
  using the raw membw ratio) over-skews.* Both were bugs; the affine-offset form is the fix.
- A **constrained** water-fill (pin a stage to ≥1 layer and re-solve on the active set)
  prevents the 0-layer-stage bug.

### 3.3 Search — `plan()` · `:595-650`
Enumerate, predict, rank:
1. **TP-only** (pp=1, tp=world): uniform + the closed-form non-uniform bias.
2. **TP×PP**: for **every divisor `pp` of world** (`2..world`, not just powers of 2 — so
   3+3 with pp∈{3,6} is covered), with `tp = world/pp` (require `n_q % tp == 0`,
   `n_layers ≥ pp`): uniform layers + the closed-form layer split and its **±3 neighborhood**.
3. `predict()` each, drop infeasible, sort by TPS, return top-k.

`decode_weight_of(w)` (`:588-592`) = `out_len/(out_len + in_len/12)` — a fixed
heuristic (prefill ~12× more tokens/sec) routing the closed-form split between the
decode- and prefill-balanced regimes.

### 3.4 `plan_safe` — the never-slower-than-baseline guard · `perf_planner.py`
`plan()` returns the predicted-argmax, which at the crossover can be a *confident
misprediction* (e.g. 8B chat n=16: predicts TP4PP2 +28% over TP8, but TP8 actually
wins). Prediction alone therefore cannot guarantee "never slower than baseline."
`plan_safe(m, hw, w, margin=SAFE_MARGIN=0.30)` adds a **risk-aware guard**: it
returns the planner's top pick *only if* it is predicted to beat the uniform
TP=world baseline (`uniform_tp_baseline`) by more than `margin` (≈2× the MAPE band);
otherwise it returns the baseline. Rationale: act only on a signal larger than the
model's noise, so a small/uncertain edge costs a *tie* (recommend baseline), never a
loss. Empirically this gives **0 baseline-losses across all 55 measured cells** while
keeping the large production-load wins (the big wins have huge predicted margins).
- This is an *empirical* never-slower on all measured data, not a mathematical one
  (a >30% confident misprediction on an unseen config could still slip). For an
  **absolute** guarantee, the planner outputs both its pick and the baseline —
  measure the two (2 short runs) and serve the faster (cheap insurance).
- Trade-off (deliberate, matches the project's rule "ties OK, never slower"): the
  conservative margin converts some small wins (+1 to ~+20%) into ties.

---

## 4. Calibration — what is fitted, and why it isn't overfitting

### 4.1 Fixed (physics / datasheet) vs fitted
- **Fixed inputs** (`hw_params.json`, not in the fit): per-GPU effective membw/TFLOPS/mem;
  `prefill_ar_overlap=0.8`; the **NVLink constants** (800 GB/s, 4 µs — physics, *not* fit).
- **Fitted** (`fit_planner.py`, 8 scalars → `fitted_params.json`): `ar_latency_us,
  ar_bw_gbs` (effective IB AR), `intra_ar_latency_us` (effective PCIe AR),
  `step_floor_ms, c_mb_ms, c_chunk_ms` (CPU dispatch floors), `overlap_eta` (PP overlap),
  `prefill_overlap` (ρ). Each is a **physically-meaningful effective constant**, bounded
  to a physical range.

### 4.2 How the fit works · `fit_planner.py`
- `load_rows()`: the measured calibration CSV, **n_req ≤ 100** (above it the Ada
  small-partition rank OOMs into KV thrashing — an unsupported regime), no stock-PP rows.
- Loss = robust (clipped) mean of relative errors on **TPS *and* ITL (decode cycle) *and*
  TTFT (prefill)** jointly — itl pins the decode model, ttft pins prefill, far stronger
  than TPS alone (where errors can cancel).
- `differential_evolution` over the 8 bounded params; **leave-one-model-out** re-fits on
  3 models and tests the 4th.

### 4.3 Why this is not memorization
- **8 scalars vs 372 data points** (4 models × configs × workloads × n_req). 8 numbers
  cannot memorize 372 — the *structure* does the predicting; the constants set levels.
- **Generalization** (the decisive test): Mistral-123B pre-registered **3/3, regret 0%**
  (predicted *before* the sweep existed); zero-refit transfer to **1+1/2+2** layouts
  (regret 3.4/7.5%); LOMO held-out 70B **9/10**.
- **Gaps were fixed by correcting the *structure***, not by adding fudge terms — and
  each structural fix *improved generalization* (hierarchical AR, per-node NVLink/PCIe:
  champion 19→25, Spearman 0.72→0.83). A fudge factor would have helped the fit set but
  not the held-out predictions.

### 4.4 Building the calibration set · `build_calibration.py`
An **additive accumulator**: it seeds from the existing CSV (so sweeps whose result dir
was archived are never lost), scans `results/hetero_*` dirs (full-grid `_full_` dicts and
concurrency list-form `record.json`), and dedups by `(model, label, workload, regime,
n_req)` — **n_req in the key** so the batch axis is preserved.

---

## 5. Measurement + cluster · `hetero_sweep.py`, `cluster_*.py`

- `hetero_sweep.py`: `gen_configs()` generates the config grid from model dims + layout
  (FFN-bias levels for cross-node TP=world; layer-skew levels for TP×PP). It launches
  vLLM via the launcher (CUDA graphs **on**, `gpu_mem 0.85`, HF offline, PP overlap via the
  auto-tuner — the hard rules), runs the perf driver, writes `record.json` / `all_runs.csv`.
  `--n-req-list` = concurrency mode (one model load, many n_req). `--extra-workload
  NAME:IN:OUT` = a held-out workload for validation.
- `cluster_env.py`: typed config from `cluster.local.env` (IPs, ifaces, GPU counts, paths).
- `cluster_setup_nxn.py` / `_4x4.py`: ray (re)configuration for any N+N layout. The worker
  rejoin via a detached ray actor is **flaky during head restart**; restarting the worker
  directly over SSH is the reliable path.

---

## 6. Validation · `check_consistency.py`, `validate_concurrency.py`, `layout_summary.py`

### 6.1 Self-consistency — 18 invariants (`check_consistency.py`)
Properties the cost model *must* satisfy regardless of data — a failure is a **logic**
bug, not a calibration gap. They cover: finite/positive outputs; TPS & decode-cycle
monotone in n_req (incl. non-divisible n); TP8 throughput saturates; **homogeneous-cluster
collapse** (non-uniform → uniform when there is no heterogeneity); **closed-form == brute-
force optimum** (TP split, layer split); AR sanity (cross ≥ intra, monotone, continuous);
feasibility monotone; bias favors the fast node; champion topology monotone in load;
layouts 1+1/2+2/4+4 sane; + regression guards for the audited bugs (embed charge,
no-0-layer, no prefill sawtooth, full PP factorization, remainder conservation) and the
never-slower guard (`plan_safe` never predicts below the uniform-TP baseline).

### 6.2 Accuracy metrics — **regret** is primary
- **Regret** = `(best_measured − measured_TPS_of_predicted_pick) / best_measured`. This is
  what matters: "how much throughput do you lose by trusting the planner?" Reported because
  exact top-1 (champion match) is **noise-limited** on flat curves (configs that tie within
  run-to-run noise). Also Spearman (rank correlation), top-3 hit-rate, MAPE.
- `validate_concurrency.py --head-gpus H --worker-gpus W`: per-(model, workload, n_req)
  champion+regret for any layout; with the **4+4 fit params unchanged**, validating 2+2/1+1
  is a zero-refit *generalization* test.
- `layout_summary.py`: the GPU-count axis, separating **FFN-bias gain** (non-uniform TP)
  from **PP-skew gain** (non-uniform PP).

### 6.3 Current numbers
Calibration fit (production workloads balanced/decode/prefill): champion 23/34,
**mean regret 2.3%** (median 0%), Spearman 0.82; TPS-MAPE 70B 11.5% / 8B 13.1% /
123B 14.7% / opt30b 29.5%. **Layout generalization (zero-refit): 4+4 regret 0.2%
· 2+2 7.5% · 1+1 3.4%.**

**Realistic operating point — the headline validation.** On the *balanced*
workload (covers prefill+decode, not skewed), at *saturating concurrency*
(n∈{32,64,96}, where the GPUs are fully utilized and the choice actually matters),
across 8B/70B/123B × {1+1,2+2,4+4} = 18 cells:
- At **4+4 (the production layout): regret = 0%** for every model and n (the
  planner picks the measured champion exactly).
- **`plan_safe` ≥ baseline in 18/18 cells** (never slower than naive uniform-TP),
  mean **+23%** over baseline, **+40–78%** at n≥64.
- The only soft spots are the *bias-degree near-ties* and one 2+2 crossover, all in
  zero-refit 1+1/2+2 layouts; `plan_safe` ties the baseline there (never loses).
This is the key result: production workloads (input-heavy, saturating load) land
squarely in the planner's accurate regime — the crossover mispredictions discussed
in §8 are confined to low-load / extreme-workload corners that production avoids.

---

## 7. Key findings (the science)

1. **Load-dependent crossover** — champion flips with concurrency: low n_req → TP=world
   (no PP bubble, latency-bound); high n_req → TP×PP+skew (TP hits a throughput wall as the
   per-token AR grows). 8B crosses at n≈8→16, 70B/123B at n≈16→32. Captured at every layout.
2. **Two non-uniform mechanisms with opposite layout trends** (`layout_summary`):
   - **FFN-bias (non-uniform TP)** grows as TP=world *shrinks*: 4+4 TP8 +3.2% → 2+2 TP4
     +11.2% → 1+1 TP2 +15.5% (fewer GPUs → bigger per-rank weight → more room to bias).
   - **PP-skew (non-uniform PP)** grows as GPUs *increase*: 4+4 +38.6% → 2+2 +21.8% → 1+1
     +16.7% (more stages → more imbalance to correct).
   → *which* non-uniform knob matters depends on the layout.
3. **Never slower than the naive baseline (the SAFE recommendation, `plan_safe`).**
   Requirement: the planner must never recommend a config measured-slower than the
   uniform TP=world default (ties OK). The raw argmax planner violated this in 4/55
   cells (worst −26%, all at the TP↔PP crossover where it confidently mis-ranked PP
   over TP). `plan_safe` (§3.4) deviates from the baseline only when its pick is
   predicted to beat it by >30% (~2× the MAPE band). Result across ALL 55 measured
   cells (incl. the held-out chat workload + zero-refit 1+1/2+2 layouts): **0
   baseline-losses, 29 wins (mean +42%)** — at production load (n≥64) it wins big
   (+40–66%), at low load it ties the baseline instead of losing.
   (`planner/verify_vs_baseline.py`, `figures/fig_selfval_vs_baseline.png`.)

---

## 8. Known corrections still needed (honest)

- **Decode-curve shape + the TP↔PP crossover (the hard open correction).** A
  decode microbenchmark (in=32/out=512, ITL vs batch, TP8 vs TP4PP2 on both models)
  isolated the decode step time into intercept (batch-independent: weight-stream +
  AR-floor + dispatch) and slope (batch-linear: KV-read + flops + AR-bandwidth). It
  showed — for *both* configs — the model's intercept is over-charged and its slope
  under-charged (the predicted decode curve is too flat), which under-predicts low
  batch (hurts TP) and over-predicts high batch (flatters PP). Two structural levers
  were tested honestly: (a) a fitted `decode_ar_overlap` (lift TP8) → the global fit
  rejected it (→0; decode AR is fully exposed), so it is not an AR-overcharge; (b)
  making KV-read bandwidth a fitted `kv_bw_scale` → it pinned to **0.32 (KV read ≈3×
  slower than weight streaming — a real paged-attention effect)**, which steepened the
  slope and improved absolute accuracy (opt30b MAPE 42→29.5%, 123B 17→14.7%, loss
  0.34→0.31). **But neither fixed the crossover baseline-losses** (the slope is still
  ~25–30% short at its bound and the intercept is still high), and production
  champion/regret was ~flat (25→23/34, 2.1→2.3% — within noise). Honest conclusion:
  the TP↔PP crossover precision is a genuine analytical-model limit that single-
  parameter structural fixes do not resolve; `kv_bw_scale` is kept (physically correct,
  better absolute accuracy) but **`plan_safe` (§3.4) remains the never-slower safety
  net, not a model fix.** A proper crossover fix is the open research item (a better
  TP-vs-PP decode-balance model, not a fudge factor).
- **opt30b TP8** (74% MAPE): its no-GQA (`n_kv=n_q=56`) + tied-embed arch mis-scales the TP8
  KV term — same TP-degree family. TP8 isn't its champion at n≤100, so regret is bounded.
- **Low-n FFN-bias degree**: at low load the planner picks `ffn_bias+50` where measured
  prefers `+25` (regret <9%, a flat-curve near-tie where the bias barely matters). Membw-
  driven, not addressed by the AR fix.
- **2+2 n=32 crossover point**: the NVLink fix nudged the planner to cross to PP one step
  early there (one zero-refit cell, 18.5%).
- **Workload-shifted crossover** (surfaced by the held-out self-validation): the
  TP→PP crossover *concurrency* moves with the workload — a longer-prefill workload
  (chat in=768) keeps TP=world winning to higher n than the calibrated balanced (512)
  does. The raw `plan()` did not shift the crossover enough for **8B chat n=16**,
  confidently picking TP4PP2 where TP8-uniform won (−26%). **`plan_safe` neutralizes
  this** (it ties the baseline there instead of losing — see §3.4), so the user-facing
  recommendation is never slower; the underlying prediction gap remains and a
  workload-dependent crossover term would also recover the lost upside at that cell.
- **Deliberate approximations** (not bugs): quadratic-attention prefill term dropped; the PP
  bubble / exposed-P2P terms are moot under the fitted η→1; `decode_weight_of`'s `/12` is a
  routing heuristic; non-uniform FFN/head bias is generated only for pp=1 (correct for
  symmetric n+n layouts where every PP stage is intra-node and homogeneous).

---

## 9. File map

| File | Role |
|---|---|
| `planner/perf_planner.py` | the cost model: Specs, roofline, AR, predict, mem, optimal splits, plan, validate, CLI |
| `planner/fit_planner.py` | fit the 8 effective params (robust loss over tps+itl+ttft, LOMO) |
| `planner/build_calibration.py` | (re)build `calibration_data.csv` (additive accumulator) |
| `planner/hetero_sweep.py` | measurement sweep (any model × layout; `--n-req-list`, `--extra-workload`) |
| `planner/cluster_env.py`, `cluster_setup_nxn.py` | typed cluster config + ray (re)configuration |
| `planner/check_consistency.py` | 17 self-consistency invariants |
| `planner/validate_concurrency.py`, `layout_summary.py` | layout-parametric accuracy + GPU-count axis |
| `planner/hw_params.json`, `fitted_params.json` | fixed effective HW params / fitted engine params |
| `planner/calibration_data.csv` | measured calibration set |
| `planner/PLANNER_SPEC.md` | the terse derivation (this doc's companion) |

## 10. How to run

```bash
# rank configs by predicted TPS for a (model, workload)
python planner/perf_planner.py --model 70b --in-len 512 --out-len 256 --n-req 96
python planner/perf_planner.py --validate          # vs calibration (regret + champion + by n_req)
python planner/check_consistency.py                # 17 invariants (logic soundness)
python planner/verify_vs_baseline.py               # plan_safe never slower than baseline (0/55)
python planner/validate_concurrency.py --head-gpus 2 --worker-gpus 2   # layout generalization
python planner/fit_planner.py                       # refit the 8 params
# the CLI also prints the SAFE recommendation (never-slower guard) + the baseline
# new cluster: edit cluster.local.env → run the ≤6-cell probe → fit → validate
```
