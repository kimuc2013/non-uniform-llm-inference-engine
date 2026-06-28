# Non-Uniform Tensor + Pipeline Parallelism Planner — Architecture, Data, and Results

A self-contained explanation of the analytical serving planner, the cost model it
uses, the re-measurement campaign behind the numbers, and the honest findings
(including corrections made during validation). Cluster under study: **4× Blackwell
RTX PRO 6000 (96 GB, NVLink) on the head node + 4× Ada RTX 6000 (48 GB, PCIe, no
NVLink) on the worker node, joined by cross-node InfiniBand.** This is a
*heterogeneous* cluster: the two GPU types differ in compute, memory bandwidth, and
memory capacity, and the inter-node link is much slower than intra-node.

---

## 1. What the planner does and why

**Thesis: on a heterogeneous cluster, "hybrid" (uniform) parallelism is not always
best — the optimal split is non-uniform and depends on the model, the layout, and
the load.**

Given `(model, hardware layout, workload)` the planner **predicts the serving
throughput of every candidate parallelization configuration analytically — without
running the model — and selects the best one.** This matters because the planner has
to choose the layout *before* the model is loaded; you cannot "try and see" because
each trial is a multi-minute model load.

A *configuration* is:
- a **topology**: tensor-parallel degree `TP` × pipeline-parallel degree `PP`
  (e.g. TP8, TP4×PP2, TP2×PP4, TP1×PP8);
- a **non-uniform split**: how each rank's share of the work is sized. Two knobs:
  - **TP FFN/head/KV column bias** — give the fast Blackwell ranks *more* columns of
    each layer's FFN / attention than the slow Ada ranks (e.g. `FFN 2688:896`);
  - **PP layer skew** — give the fast Blackwell pipeline stage *more* layers than the
    slow Ada stage (e.g. `L=53-27` = 53 layers on Blackwell, 27 on Ada).

The planner's job is to pick the topology **and** the non-uniform split.

---

## 2. The cost model (how a prediction is computed)

Everything is a roofline built from real model + hardware quantities. Notation
(spelled out): `L`=layers, `h`=hidden, `F`=FFN intermediate (per-expert for MoE),
`H_q`/`H_kv`=query/KV heads, `d`=head_dim, `B`=batch (=concurrent requests in
decode), `S`=avg KV context (`in+out/2`). Hardware per GPU `g`: `TF_g`=effective
prefill TFLOPS, `BW_g`=effective decode HBM bandwidth. Bytes `b`=2 (bf16).

### 2.1 Per-rank parameters (the non-uniform knob enters here)
```
P_rank(L_s, h_i, f_i, E) = L_s·( P_attn·(h_i/H_q) + E·P_ffn·(f_i/F) ) + P_embed/TP
```
`h_i`, `f_i` are *this rank's* head / FFN columns (different per rank when non-uniform).
`E` is the MoE expert multiplier (=1 for dense; see §2.5).

### 2.2 Decode step time (memory-bound; per PP stage, bottleneck = slowest rank)
```
for each rank i:  compute_i = max( weight_read_i + KV_read_i ,  flop_i )
compute   = max over ranks (the slowest, usually an Ada rank)
AR        = 2·L_s · allreduce(message = B·h·b, ranks)     # 2 all-reduces / layer
step_decode = max(compute, AR) + (1 - overlap)·min(compute, AR)
```
- `weight_read = P_rank·b / BW_g`, `KV_read = B·S·L_s·2·H_kv,i·d·b / BW_g`,
  `flop = 2·P_rank·B / TF_g`.

### 2.3 The cross-node all-reduce (hierarchical, 2-tier)
```
intra-node:  reduce-scatter + all-gather over NVLink (Blackwell) or PCIe (Ada)
inter-node:  one ring over InfiniBand
allreduce = intra_term + inter_term(measured-bandwidth-surface) + latency
```
The inter-node bandwidth is a **measured 2-D surface** over (ranks-per-node on the
slow side, message size) — see `data/ar_rank_scaling.md`. This captures two real
NCCL effects: the LL→Simple protocol transition near 1 MB, and the bandwidth
collapse when 4 Ada GPUs funnel through one IB NIC + shared PCIe.

### 2.4 Prefill, pipeline cycle, throughput
- **Prefill step** = compute-bound (FLOPs dominate), chunked at `T_chunk`=8192,
  cross-node AR ~80 % hidden under the GEMMs.
- **Pipeline (PP)**: GPipe-style, microbatches `n_mb = min(PP, B)`; cycle =
  slowest stage's busy time + (un-overlapped) bubble + inter-stage P2P + a CPU floor.
- **Throughput** = `B·out_len / max(T_prefill, T_decode)` (the two phases run on
  disjoint resources and partially overlap under continuous batching).
- **Memory feasibility**: `weights(all experts) + 1 resident sequence's KV +
  activations + overhead ≤ capacity·0.85` per rank (paged KV — the engine queues
  excess concurrency rather than co-residing all of it).

### 2.5 Mixture-of-Experts (MoE) — added modularly, dense path unchanged
Two `ModelSpec` fields, `n_experts` and `top_k` (both =1 for dense). The FFN cost
splits **three ways** via the expert multiplier `E`:
- **memory residency**: `E = n_experts` (all experts resident);
- **decode weight stream**: `E = active_experts(B) = n_experts·(1−(1−top_k/n_experts)^B)`
  — the *distinct* experts a B-token step touches: `top_k` at B=1, → `n_experts` at
  large B. This is the **batch-dependent regime shift** (AR-bound at small batch,
  weight-bound at large batch);
- **FLOPs**: `E = top_k` (only the routed experts compute).
Experts are TP-sharded (no expert-parallel all-to-all), so the hidden-state
all-reduce is unchanged. (Verified from the run: vLLM used
`MoEPrepareAndFinalizeNoDPEP` = no expert parallelism.)

### 2.6 Calibration — no per-model fudge, no learned weights
The model has **zero per-model constants and no learned weight matrices** — it is
closed-form roofline + α-β communication. A small set of **~10 global
hardware/engine constants** is calibrated once on the cluster:
- **measured** (microbench, not fitted): per-GPU HBM bandwidth and prefill TFLOPS
  (`hw_params_measured.json`), the cross-node AR bandwidth surface, `kv_bw_scale=1.0`;
- **fitted** (once, on 4+4 serving data): a few latency/floor constants;
- **one residual engine constant**: the in-decode AR *pipeline concurrency* (~3),
  which turns the measured AR *kernel* bandwidth (~1.3 GB/s) into the *effective*
  in-decode bandwidth (~4 GB/s) — see §4.

The proof it is **not an overfit curve**: it transfers **zero-refit** to new models
(Mistral-123B, Mixtral-8×7B were *pre-registered* — predicted before measuring) and
to new layouts (2+2, 1+1), and a *pre-serving hardware auto-calibration* using only
measured constants reproduces the offline fit (40/43 vs 42/43 cells ≥ baseline).

---

## 3. The measurement campaign (where the numbers come from)

Every model was re-measured on the current cluster using the planner config set
(uniform baseline + the planner's non-uniform picks + alternatives):
- **4+4** (8 GPU): 8B, 70B, OPT-30B, Mistral-123B, Mixtral-8×7B — full grid (uniform
  TP8/TP4PP2/TP2PP4/TP1PP8 + PP-skew + FFN-bias) × 3 workloads (balanced
  512/256, decode-heavy 128/512, prefill-heavy 1024/128) × concurrency {16,32,64,96};
- **2+2** (4 GPU): 8B, 70B, OPT-30B, Mixtral — lean (planner pick + baseline,
  balanced);
- **1+1** (2 GPU): 8B, OPT-30B, Mixtral — lean (70B/123B do not fit in 2 GPUs).
Qwen3-32B is excluded (it is a *serving* outlier — the fork's PP-overlap does not
engage for it — not a planner error). All raw cells are in
`data/measured_results_all_layouts.csv` (636 successful cells).

Data integrity: two suspicious-looking baselines (70B and 123B TP8 at high load)
were **re-measured and reproduced exactly** — they are real saturation, not
degradation (see §4).

---

## 4. Headline results

**Validation on fresh data (all layouts):**
- `verify_vs_baseline`: **59/67 cells ≥ uniform-TP baseline, mean uplift +79 %**
  (`data/validation_verify_vs_baseline.txt`);
- `check_consistency`: **17/17 logical invariants hold**;
- champion topology matches the measured best (the planner picks TP4×PP2 at 4+4,
  which is the measured winner).

**4+4 balanced — planner pick vs uniform baseline** (figure
`planner_vs_baseline_uplift_4x4.png`):

| model | n=32 | n=64 | n=96 | planner pick (non-uniform) |
|---|---|---|---|---|
| Llama-8B | +118 % | +92 % | +130 % | TP4×PP2  L=21-11 |
| Llama-70B | +59 % | +169 % | +273 % | TP4×PP2  L=53-27 |
| OPT-30B | +28 % | +76 % | +176 % | TP4×PP2  L=32-16 |
| Mistral-123B | +74 % | +195 % | +29 % | TP4×PP2  L=59-29 |

**Why the uplift is large (and real):** the uniform baseline **TP8 saturates at high
load** — for 70B, TP8 throughput is flat at ~390 tok/s (ITL grows to 238 ms) because
the 8-way cross-node all-reduce becomes the bottleneck on the Ada ranks. The
planner's **non-uniform TP4×PP2** (more layers on the fast Blackwell stage) keeps
scaling to ~1514 tok/s. This is the core evidence for the thesis.

**Smaller layouts — which non-uniform mechanism wins flips per model** (figures
`..._2x2.png`, `..._1x1.png`):
- 2+2: 8B/OPT win with **PP layer-skew** (TP2×PP2, e.g. L=22-10); 70B wins with **TP
  FFN-column bias** (TP4, `FFN 9472:4864`). Uplifts +5..+162 %.
- 1+1: gains shrink (±5 %) — with only 2 GPUs there is little room for non-uniformity.
This per-layout flip is itself thesis evidence: *which* non-uniform knob matters is
situation-dependent.

---

## 5. Key findings and honest corrections (made during validation)

These corrections matter for accuracy; the package reflects the **corrected**
understanding.

1. **The cross-node decode all-reduce is plain NCCL on the compute stream — no true
   overlap.** (Confirmed in the vLLM source: custom all-reduce is disabled across
   nodes; the AR runs on the default stream.)

2. **All models are AR-bound at high load in cross-node TP8 — not just OPT.** A
   direct torch profile of 70B TP8 decode shows it is **84 % ncclAllReduce**. An
   earlier claim that "70B is weight-bound and runs the AR 2.6× faster" was an
   artifact of indirect ITL decomposition and is **withdrawn**.

3. **The all-reduce bandwidth IS measurable and consistent (~1.1–1.4 GB/s).** The
   isolated microbench (1.07 GB/s) matches the in-serving profile (70B 1.2, OPT 1.1)
   once serving-matched NCCL settings (Simple protocol) are used. An earlier "we
   cannot measure it" framing was wrong.

4. **The planner's effective `ar_bw≈4 GB/s` = measured kernel bandwidth (~1.3) ×
   in-decode pipeline concurrency (~3).** The ~160 AR kernels per decode step overlap
   on the GPU timeline (an 81 ms ITL cannot hold 160 serial 851 µs kernels = 136 ms),
   which an *isolated* benchmark cannot reproduce. That concurrency factor is the one
   residual engine constant.

5. **MoE (Mixtral-8×7B) generalizes zero-refit:** the planner — with MoE added but
   never refit — predicted the champion at all four concurrencies *before* measuring,
   capturing the batch-dependent TP8→TP4×PP2 crossover (small batch → few experts
   active → AR-bound → TP8; large batch → all experts → weight-bound → PP helps).

6. **Pre-serving hardware auto-calibration works (no model load).** `calibrate.py`
   measures the silicon constants directly (a serving probe is self-defeating — the
   planner must choose the layout *before* loading). Measured-only constants reach
   40/43 cells ≥ baseline vs the offline fit's 42/43.

### Honest limitations (documented, not hidden)
- **OPT-30B TP8 over-prediction**: OPT is AR-bound (multi-head attention, n_kv=56,
  small per-rank compute). With OPT the lone such outlier, fitting a per-model
  correction would be overfitting one sample — so it is left as a documented gap.
  **The ranking is preserved** (TP8 is not OPT's champion; the planner correctly
  picks TP4×PP2), so it does not cause a wrong recommendation at 4+4.
- **Mixtral 4+4 prefill-heavy** is the worst loss cell (−19.6 %): the MoE prefill
  regime was not in the original fit.
- **2+2 / 1+1** are zero-refit generalization (no re-fit) and are less accurate than
  4+4.
- **Expert parallelism (EP / all-to-all)** is out of scope: the MoE model covers the
  TP-sharded mode vLLM uses by default. High-expert MoE (e.g. DeepSeek-256-expert) or
  `--enable-expert-parallel` would need an all-to-all term.

---

## 6. File index (this package)

```
figures/
  planner_vs_baseline_uplift_4x4.png   planner pick vs uniform baseline, 4+4 (4 models)
  planner_vs_baseline_uplift_2x2.png   ... 2+2 (3 models)
  planner_vs_baseline_uplift_1x1.png   ... 1+1 (2 models)
       (blue banner per panel = the non-uniform config the planner picked,
        e.g. "planner pick: TP4xPP2 L=53-27"; grey=baseline, blue=pick, % above)
  fig_crossover_concurrency.png        TP8↔TP4PP2 champion crossover vs load
  fig_layout_gain.png                  non-uniform gain vs layout (1+1→4+4)
  fig_planner_validation.png           predicted vs measured
  fig_selfval_vs_baseline.png          held-out-workload self-validation
  fig_mistral123b_*.png                Mistral-123B pre-registration / per-workload
  per_model_configs/                   PER-MODEL parallelization comparison: throughput
                                       vs concurrency, one line per config (solid=uniform,
                                       dashed=non-uniform, grey=baseline) — shows the
                                       TP<->PP crossover + non-uniform gain per model.
                                       4+4 (balanced/decode/prefill), 2+2, 1+1.
data/
  measured_results_all_layouts.csv     636 measured cells (layout,model,tp,pp,
                                        layer_split,ffn_splits,workload,n_req,tps,itl,ttft)
  validation_verify_vs_baseline.txt    the 59/67 +79% validation output
  hw_params_measured.json              auto-calibrated hardware constants (membw, TFLOPS)
  ar_rank_scaling.md                   measured cross-node AR bandwidth surface
  mixtral8x7b_prereg.json              Mixtral predictions FROZEN before measuring
  mixtral8x7b_validation.txt           Mixtral zero-refit 4/4 champion match
docs/
  SERVING_PLANNER_EXPLAINED.md         this document
  planner_describe.md                  full pedagogical study guide (intuition+math+code refs)
```

Source code (pushed to GitHub `kimuc2013/non-uniform-llm-inference-engine`, branch
`main`): the planner is `planner/perf_planner.py`; calibration `planner/calibrate.py`
+ `compute_microbench.py` + `ar_microbench.py`; campaign runner
`planner/run_campaign.sh`; figures `planner/plot_planner_uplift.py` +
`plot_paper_figures.py`; validation `planner/verify_vs_baseline.py` +
`check_consistency.py`.
