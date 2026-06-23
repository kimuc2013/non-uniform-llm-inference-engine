# PLANNER_SPEC — Analytical model for a universal hetero TP/PP serving planner

Status: design spec (v1, 2026-06-12).
Scope: closed-form throughput prediction + optimal-split derivation + search, for
arbitrary `(model_spec, hardware_spec, workload)` on heterogeneous multi-node
clusters. Calibration from a ≤6-cell probe set makes the model portable to new
clusters. Reference implementation skeleton already exists in this directory
(`cost_model.py`, `planner.py`, `gpu_library.py`, `network_library.py`); this
spec defines the v2 equations that replace/extend it.

---

## 0. Problem statement

```
predict_TPS(model_spec M, hardware_spec H, workload W, config C) -> TPS_wall, TTFT, feasible?
plan(M, H, W) -> argmax_C TPS_wall(C)  subject to  mem_feasible(C)
```

Config space:

```
C = (tp, pp, layer_split[pp], ffn_splits[tp], head_splits[tp], placement)
    tp · pp = world_size,  Σ layer_split = L,  Σ ffn_splits = d_ff,  Σ head_splits = n_q
```

The predicted metric MUST match what `perf/performance.py` reports:
`total_wall_throughput_tok_s = total_out_tokens / wall_s` with `wall_s` =
(first request issued → last token emitted). All calibration targets are runs
with CUDA graphs ON (never `--enforce-eager`, per project rule) and
`n_req ≤ 100` on the current cluster (memory rule, reproduced by §5).

---

## 1. Notation

Model `M`:

| symbol | meaning | example (Llama-70B) |
|---|---|---|
| `L` | num layers | 80 |
| `h` | hidden size | 8192 |
| `n_q, n_kv` | query / KV heads | 64, 8 |
| `g = n_q/n_kv` | GQA group (head quantum) | 8 |
| `d_h` | head dim | 128 |
| `d_ff` | FFN intermediate | 28672 |
| `V` | vocab | 128256 |
| `b_w, b_kv, b_a` | bytes/param, /KV elem, /activation elem (bf16 → 2) | 2 |

Derived per-layer parameter counts (SwiGLU; for GELU-MLP like OPT use `2·h·d_ff`
and add biases — negligible):

```
P_attn = h·n_q·d_h + 2·h·n_kv·d_h + n_q·d_h·h          # Wq + Wkv + Wo
P_ffn  = 3·h·d_ff                                       # gate, up, down
P_layer = P_attn + P_ffn
attn_frac = P_attn / P_layer ;  ffn_frac = P_ffn / P_layer
P_embed = V·h (+ V·h if untied lm_head)
```

(70B: `P_layer = 856M`, `attn_frac = 0.176`, `ffn_frac = 0.824` — the FFN is
~5× the attention weights, which is why FFN bias is the main TP knob.)

Hardware `H`: nodes `n ∈ {0..N_nodes−1}`, each with `G_n` GPUs of type `g(n)`.
Per GPU type, **calibrated** effective rates (§7):

```
TF_g   : effective bf16 TFLOPS in the prefill (large-M) regime   [flops/s]
BW_g   : effective HBM bandwidth, streaming weights              [bytes/s]
KBW_g  : effective KV-read bandwidth (paged-attn pattern; ≥ BW_g due to L2)
VRAM_g : usable bytes = capacity · util  (util ≈ 0.9)
```

Links: for any GPU pair `(i,j)`, `bw(i,j)` and `lat(i,j)`; in practice two
classes per cluster: `bw_intra/lat_intra` (NVLink/PCIe) and
`bw_cross/lat_cross` (IB/Ethernet).

Workload `W = (in_len, out_len, n_req)`. Decode KV length used everywhere:
`kv̄ = in_len + out_len/2` (time-average).

---

## 2. Component 1 — Roofline per (rank, phase)

Every rank `r` holds `params_on_rank(r)` bytes of weights (eq. §3.1) for the
layers of its PP stage, plus a KV-cache slice. One model step on rank `r`:

**Decode** (S=1, batch `B` sequences in flight on this stage) — memory-bound:

```
bytes_r  = params_on_rank(r) · b_w  +  KV_read_r
KV_read_r = B · kv̄ · L_s · 2 · kv_r · d_h · b_kv        # k+v, kv_r heads on rank
flops_r  = 2 · params_on_rank(r) · B                     # GEMV per token
t_rank_dec(r) = max( params_on_rank(r)·b_w / BW_g(r) + KV_read_r / KBW_g(r),
                     flops_r / TF_g(r) )
```

The weight-streaming term is B-independent (whole weight matrix read once per
step regardless of batch — this is what makes decode TPS rise with `B` until
the flops term or KV term catches up; the crossover B is the saturation batch).

**Prefill** (chunk of `T_c` tokens, e.g. `max_num_batched_tokens`) — compute-bound:

```
flops_r = 2 · params_on_rank(r) · T_c  +  2 · head_r · d_h · Σ_seq(s_i²)   # GEMMs + attn quadratic
t_rank_pre(r) = max( flops_r / TF_g(r),  params_on_rank(r)·b_w / BW_g(r) )
```

The quadratic attention term matters only when `in_len ≳ 4k`; for the standard
sweep (`in_len ≤ 1024`) the linear term dominates (< 7% correction).

---

## 3. Component 2 — Non-uniform TP stage

### 3.1 Parameter placement

TP group of size `Nt`; rank `r` holds `head_r` q-heads (and
`kv_r = head_r / g` KV heads when biasable, else `n_kv/Nt`) and `f_r` FFN
columns. For a stage holding `L_s` layers:

```
params_on_rank(r) = L_s · P_layer · ( attn_frac · head_r/n_q  +  ffn_frac · f_r/d_ff )
                    + P_embed_r                       # first/last stage only; vocab-parallel: P_embed/Nt
```

### 3.2 Stage time

```
t_stage = max_r t_rank(r)  +  n_AR · t_AR(msg)  +  L_s · c_layer
n_AR  = 2 · L_s                                   # attn-out + ffn-down per layer
msg   = B · S · h · b_a                           # S=1 decode, S=T_c prefill
```

**AllReduce — hierarchical (2-tier) collective, NOT a flat cross-node ring**
(the hetero-cluster key fact, corrected 2026-06-23):

A TP group that spans `n_nodes` nodes with `n_local` GPUs each does **not** pay a
flat `2(Nt−1)` serial IB-latency ring. NCCL runs an intra-node reduce-scatter +
all-gather over NVLink and only an inter-node ring over IB, so:

```
single node (n_nodes=1):
  t_AR = 2(N−1)/N · msg / bw_intra              +  2(N−1)        · α_intra
cross node (n_nodes>1):
  t_AR = 2(n_local−1)/n_local · msg / bw_intra  (NVLink RS+AG)
       + 2(n_nodes−1)/n_nodes · msg / bw_cross  (IB inter-node ring, NIC shared
                                                 by the n_local local GPUs)
       + 2(n_nodes−1) · α_cross  +  2(n_local−1) · α_intra
```

The **latency** term is the decisive correction: a 4+4 TP8 group pays
`2(n_nodes−1)=2` IB hop-latencies, **not** `2(Nt−1)=14`. The old flat-ring form
over-charged cross-node decode AR ~6× (e.g. 45 ms/token for 8B TP8 vs ~8 ms
measured), which made the planner pick PP over TP8 at *every* load and miss the
measured low-concurrency TP8 champion. Two regimes still fall out:

- decode (`msg ≈ B·h·b_a`): **latency-floor** dominated by the *inter-node* hops
  only → cross-node TP8 is viable at low load (few microseconds × 2L), and only
  loses to TP4-PP2 once the per-token AR *bandwidth* term grows with batch.
  This batch-dependent crossover is now reproduced (validated §8).
- prefill (`msg ≈ T_c·h·b_a`): **bandwidth** dominated; the inter-node IB ring is
  the bottleneck link, `ar_bytes/token = 2·L · 2(n_nodes−1)/n_nodes · h·b_a`.

### 3.3 Optimal split — closed forms

Minimize `max_r t_rank(r)` subject to `Σ head_r = n_q`, `Σ f_r = d_ff`.
Optimum equalizes `t_rank` across ranks. Define rank speed:

```
speed_r = BW_g(r)   (decode-dominated workload)
speed_r = TF_g(r)   (prefill-dominated workload)
```

**Case A — fully biasable attention** (MHA, `g = 1`, or GQA with
`n_kv/Nt ≥ 2` so KV quantum allows bias). Both `head_r` and `f_r` follow the
same fraction and all terms scale together:

```
frac_r* = speed_r / Σ_q speed_q
head_r = round_to(g,  frac_r*·n_q)        # head quantum = GQA group g
f_r    = round_to(Q_ffn, frac_r*·d_ff)    # FFN quantum Q_ffn = 128 (TC tile)
```

with largest-remainder fix-up so sums match exactly (existing
`planner.py:_derive_tp_splits` logic).

**Case B — attention floor** (GQA at full TP: `n_kv/Nt = 1` ⇒ head bias
infeasible; heads uniform). Then per rank:

```
t_rank(r) = a_r + c_r · f_r
decode:  a_r = (L_s·P_attn·b_w/Nt)/BW_r + KV_read_r/KBW_r      # NOT reduced by ffn bias
         c_r = L_s·3·h·b_w / BW_r                              # bytes per FFN column
prefill: a_r = 2·L_s·P_attn·T_c/Nt / TF_r ;  c_r = 2·L_s·3·h·T_c / TF_r
```

For two GPU classes (`n_F` fast ranks share `f_F` each, `n_S` slow share `f_S`),
equalize `a_F + c_F f_F = a_S + c_S f_S` under `n_F f_F + n_S f_S = d_ff`:

```
f_S* = ( d_ff − n_F·(a_S − a_F)/c_F ) / ( n_S + n_F·c_S/c_F )
f_F* = ( d_ff − n_S·f_S* ) / n_F
```

(General K-class version: `f_r* = (Λ − a_r)/c_r` with the water level
`Λ = (d_ff + Σ a_q/c_q) / Σ 1/c_q`, clamped to `f_r ≥ Q_ffn` and re-solved on
the active set.)

**Saturation / sweet spot (sanity check e).** Because `a_S` is a floor that FFN
bias cannot reduce, `max_r t_rank ≥ a_S` always. Bias beyond `f_F*` only
inflates the fast rank: `d(max_r)/d(bias) > 0` past the equalizer. So predicted
optimum = the equalizing bias, and the curve is V-shaped around it.
Plugging current-cluster numbers (`BW = 1461/798 GB/s`, 4+4 ranks, 70B):
`f_F* = 4861 ≈ +36%` over uniform 3584 — inside the measured +25..+50 sweet
spot, with KV (`a_S` ↑) pushing it toward +25.

**Why head bias > FFN bias for MHA decode (sanity check c).** Bytes moved per
unit of bias quantum, decode, OPT-30B at `B=64, kv̄=896`:

```
per q-head:   ΔKV + ΔW = B·kv̄·2·d_h·b_kv·L + 4·h·d_h·b_w·L ≈ 1409 + 352 = 1762 MB
per FFN tile (128 cols): 128·3·h·b_w·L ≈ 264 MB
```

For MHA the KV read scales with `head_r` (`kv_r = head_r`), so the head knob
moves ~7× more decode bytes per quantum than the FFN knob — the model captures
this automatically because `KV_read_r ∝ kv_r` sits in `t_rank_dec`. For GQA
models at high TP the head knob is unavailable (Case B), so FFN bias is the
only lever and its effect is capped by the attention floor.

---

## 4. Component 3 — PP pipeline

Stage `s ∈ {0..pp−1}` holds `L_s` layers, runs on node `n(s)` (TP group per
§3 inside the stage). Per-microbatch stage time, from §3.2 with the stage's
`B = mb_size`:

```
t_s = t_stage(L_s, mb_size, node n(s))  +  c_mb          # c_mb = per-microbatch CPU dispatch
```

Equivalently in proportional form (uniform-TP stages):
`t_s ≈ (L_s/L) · t_model_on_node(n(s)) + c_mb`.

**Steady-state cycle with `n_mb` in-flight microbatches** (our fork: disjoint
request subsets, `bq = pp`, `mb_size = n_req/n_mb`, send-side ring buffer +
side-stream broadcast):

```
ideal (η=1):   T_cycle = max_s t_s + t_p2p_exposed
general:       T_cycle = max_s t_s + (1−η)·Σ_{s≠s*} t_s + (1−η_p2p)·Σ_s t_send(s)
TPS_decode_phase = n_req / T_cycle              # every request advances 1 token per cycle
t_send(s) = mb_size·S·h·b_a / bw(n(s),n(s+1)) + lat(s,s+1)
```

`η` is the measured overlap efficiency (fork: 0.56–0.78 ⇒ use η≈0.65 default;
stock vLLM PP: 0.12–0.24 ⇒ η≈0.15). Requires `n_mb ≥ pp` for full coverage;
if `n_mb < pp`, add bubble `(pp − n_mb)·mean_s(t_s)` per cycle (ASTRA-sim
finding: `n_mb > compute_ratio` needed for steady-state).

**Optimal layer split — closed form.** Minimize `max_s (L_s·τ_s + c_s)` with
`Σ L_s = L`, `τ_s` = per-layer time on stage `s` (decode: `P_layer·b_w/BW_n(s)`
+ per-layer AR + `c_layer`; prefill: `2·P_layer·T_c/TF_n(s)` + AR):

```
L_s* = (Λ − c_s) / τ_s ,   Λ = ( L + Σ_q c_q/τ_q ) / ( Σ_q 1/τ_q )
     ≈ L · (1/τ_s) / Σ_q (1/τ_q)            # when c_s uniform → L_s ∝ node speed
```

Quantize by largest-remainder to integers ≥ 1, then exhaustively evaluate the
`±1`-neighborhood (move one layer between any stage pair; ≤ pp·(pp−1) extra
evals) and keep the best predicted `TPS_wall`. Note the AR + `c_layer` terms
inside `τ_s` are stage-symmetric, which **attenuates** the split ratio toward
uniform vs. raw `BW` ratios (raw 1.83 → effective ≈ 1.5; this is the
`HETERO_RATIO_ATTENUATION` effect, now derived rather than fitted).

**Why deep pipelines lose unless skewed (sanity check d).** Three additive
penalties grow with `pp`: (i) `(1−η)·Σ_{s≠s*} t_s` has `pp−1` terms;
(ii) `c_mb · n_mb` CPU dispatch per cycle; (iii) granularity — with `L/pp`
layers per stage the integer quantum is coarser relative to stage time, and
`max_s` sits further above the mean. With heavy skew the `max_s` itself drops:
70B TP1PP8, uniform Ada stage `10·P_layer·b_w/BW_A = 21.4 ms` vs skew (14×4 :
6×4) `max(14·2·856M/1461G, 6·2·856M/798G) = max(16.4, 12.9) = 16.4 ms` →
predicted +31%, measured +32.8%.

---

## 5. Component 4 — Workload mixing → TPS_wall

Matches `perf/performance.py` (`total_wall_throughput_tok_s = Σ out_tokens / wall`):

```
# Prefill phase: n_req·in_len tokens flushed through in chunks of T_c
n_chunks  = ceil( n_req·in_len / T_c )
T_prefill = Σ_s t_s_pre(first chunk fill)  +  (n_chunks−1)·[ max_s t_s_pre + (1−η)·Σ_{s≠s*} t_s_pre ]
TTFT      ≈ Σ_s t_s_pre + t_first_decode            # pipeline fill + 1 step

# Decode phase: out_len cycles, all n_req in flight
T_decode  = out_len · ( T_cycle + c_cpu )           # T_cycle from §4 (pp=1 ⇒ T_cycle = t_stage)
c_cpu     = per-step scheduler/sampler/detokenize overhead (calibrated, ~ms-scale)

# Phase combination — NOT additive. Prefill is tensor-core (compute) bound and
# decode is HBM (bandwidth) bound; under chunked continuous batching the engine
# interleaves prefill chunks into decode steps, so the two phases run on disjoint
# hardware resources and PARTIALLY OVERLAP. Modeling them as serial blocks
# double-counts (measured: 70B prefill_heavy T_prefill≈2 s, additive model
# predicted ≈18 s → −50% TPS error and a spurious topology flip). Blend with a
# single calibrated overlap factor ρ (fitted ρ≈0.66 on the current cluster):
T_total   = max(T_prefill, T_decode) + (1−ρ)·min(T_prefill, T_decode)
            # ρ=0 → serial (old additive model); ρ=1 → smaller phase fully hidden
TPS_wall  = n_req · out_len / T_total
```

Refinement (second order, optional): decode KV grows linearly over the phase;
using `kv̄ = in_len + out_len/2` in `KV_read_r` integrates this exactly for the
linear KV term.

**Calibration corrections applied (2026-06-23) — batch axis + hierarchical AR:**
- *Hierarchical AllReduce* (§3.2): replaced the flat cross-node ring latency
  `2(Nt−1)·α_cross` with the 2-tier NVLink+IB form. This was the single largest
  correction — it lifted cross-node TP8 from ~2.6× under-predicted to within
  ~12% and unblocked the load-dependent TP8↔TP4PP2 champion crossover.
- *Concurrency (batch) axis folded into calibration.* The fit now spans
  `n_req ∈ {8,16,32,64,96}` (3 model sizes), not just the single high-load point
  per model. `build_calibration.py` is an additive accumulator (seeds from the
  existing CSV so archived sweeps are not lost) and tags `n_req` in the dedup key.
- *n_req ≤ 100 operating rule enforced in the fit/validation.* Old `n=128` sweeps
  sit in a KV-preemption thrashing regime (Ada small-partition rank OOMs) the
  cost model deliberately does not represent; they were the dominant MAPE source
  (129% at n=128 vs 12–17% at n≤96) and are excluded.
- *Logic fixes from the adversarial consistency audit (2026-06-23).* An 8-component
  audit (each finding independently refuted-or-confirmed) caught five real
  derivation bugs, now fixed: (1) **embedding double-count** — `params_on_rank`
  added the full `p_embed` (=2·V·h untied) on BOTH PP end stages; physically
  stage 0 holds only the input table (V·h) and the last stage only lm_head (V·h),
  so `embed_on_stage()` now charges one V·h per end stage (pp>1) — removes an
  anti-PP bias that hit the bottleneck Ada stage. (2) **partial last prefill
  chunk** was charged a full `T_CHUNK` → a sawtooth in T_prefill; now the last
  chunk takes its true remainder and `c_chunk` is per-chunk not per-stage.
  (3) **dropped-remainder decode** — `mb=n_req//pp` modeled `pp·⌊n_req/pp⌋`
  requests while the TPS numerator used `n_req`; now `n_mb=min(pp,n_req)`,
  `mb=n_req/n_mb` (exact, affine-conserving). (4) **0-layer PP stage** from the
  fix-sum loop → `optimal_layer_split` now does a constrained water-fill (pin to
  ≥1, re-solve the active set). (5) **power-of-2-only PP enumeration** — `plan()`
  now enumerates every divisor of `world` so 3+3 (pp∈{3,6}) is covered. Plus the
  Case-B FFN water-fill now includes the KV-read floor, and decode P2P is charged
  over pp−1 (not pp) links.
- *Per-node intra-AR (NVLink vs PCIe), 2026-06-23.* The Blackwell head has NVLink
  but the Ada worker is PCIe; a single `intra_ar` over-charged the Blackwell
  stage's AllReduce and under-skewed PP layer allocation (the 8B-skew residual).
  `HardwareSpec.intra_params(gpu)` now returns NVLink (≈800 GB/s, 4 µs, a fixed
  physics constant — NOT fitted, to avoid overfitting) for Blackwell and PCIe for
  Ada. This fixed the 4+4 8B layer-skew exactly and lifted **champion 19→25/34,
  mean regret 3.5→2.1%, Spearman 0.72→0.83** after re-fit. (The PP layer-split
  search neighborhood was widened ±2→±3 to track the shifted optimum.)
- *Result (final fit):* champion 25/34, **mean regret 2.1%, median 0.0%**, top-3
  32/34, Spearman ρ≈0.83; TPS-MAPE 70B 11.5% / 8B 13.1% / 123B 17.4% / opt30b
  43%. LOMO held-out 70B 9/10 (regret 0.7%), 8B 7/10. **Layout generalization
  (zero-refit, 4+4 fit params, cost model is layout-parametric): 4+4 champion
  27/30 regret 0.2% · 2+2 regret 7.5% · 1+1 regret 3.4%** — a planner calibrated
  on 4+4 transfers to 1+1/2+2 with bounded regret, covering the 1+1→4+4 target.
  Fitted engine params drift to `step_floor→0, c_mb≈0.07, overlap_eta→1.0,
  prefill_overlap→0`: with CUDA graphs per-step dispatch is hidden and the
  well-overlapped fork pipeline is max-stage-bound.
- *Self-consistency suite* `planner/check_consistency.py` (17 invariants, all
  pass): finite/positive outputs; TPS & decode-cycle monotone in n_req (incl.
  non-divisible n); TP8 throughput saturates; **homogeneous-cluster collapse**
  (non-uniform → uniform when there is no heterogeneity); closed-form TP/layer
  splits == brute-force optimum; AR sanity (cross≥intra, monotone, branch
  continuity); feasibility monotone; bias favors the fast node; champion topology
  monotone in load; layouts 1+1/2+2/3+3/4+4 sane; + regression guards for the
  five audited bugs (embed charge, no-0-layer, no prefill sawtooth, full
  factorization, remainder conservation). A failure here is a logic bug, not a
  calibration gap — this is the "logically convincing" leg of the planner.
- *Residual gaps (current):* (a) **opt30b TP8** (74% MAPE) — its no-GQA
  (n_kv=n_q=56) + tied-embed arch makes the TP8 KV term mis-scale; TP8 is not
  opt30b's champion at n≤100 so regret impact is bounded. (b) **8B layer-skew at
  mid-load** — FIXED by the per-node intra-AR (NVLink vs PCIe) above; TP4PP2_skew+8
  now matches at every n at 4+4. (c2) **low-n FFN-bias degree** — at every layout
  the planner picks ffn_bias+50 where measured prefers +25 (regret 3.7–8.7%, a
  flat-curve near-tie at low load where the bias barely matters); membw-driven,
  not addressed by the AR fix. (c3) **2+2 crossover point** — the NVLink fix
  nudged the planner to cross from TP4PP1 to TP2PP2 one step early at 2+2 n=32
  (18.5% on that one zero-refit cell). (c) opt30b prefill_heavy n=64 (28% regret).
- *Deliberate approximations (audit-flagged, NOT bugs, left as-is):* the
  quadratic attention prefill term `2·head_r·d_h·Σsᵢ²` is dropped (small at our
  seqlens); the PP bubble `(1−η)·b_rest` and exposed P2P are moot under the
  fitted η→1; `decode_weight_of` uses a fixed in/12 token-speed ratio (a routing
  heuristic, not a timing term); non-uniform FFN/head bias is generated only for
  pp=1 — correct for symmetric n+n layouts where every PP stage is intra-node and
  homogeneous, but it would miss within-stage heterogeneity in an asymmetric
  placement. These are documented so "tested == the path the planner runs."

**Calibration corrections applied (2026-06-13) and residual gaps:**
- *Prefill TFLOPS doubled* (Blackwell 289→578, Ada 183→366): the originals were
  back-derived under the additive wall model and read ~2× low; an independent
  free-param fit landed at 1.98× (= Ada bf16-acc peak). See hw_params note.
- *Prefill AR overlap* `prefill_ar_overlap=0.8`: cross-node TP AllReduce of
  large prefill chunks is ~80% hidden under the GEMMs (async-TP /
  sequence-parallel): `t_stage_pre = compute + (1−0.8)·t_ar`.
1. *Residual: TP=world prefill still under-predicted (~2.4×).* The above lifted
   70B TP8 prefill from 388→540 tps (measured ~1369), but a single global ρ
   cannot fit both decode/balanced (which now want low ρ≈0.16 since prefill
   compute shrank) AND TP8 prefill_heavy (which behaves as if prefill is almost
   fully hidden, ρ→1). Consequence: the planner still prefers PP2 over TP8 for
   70B prefill_heavy — the one persistent topology miss (15.7% regret on that
   cell). Proper fix needs a *per-regime* overlap (ρ as a function of the
   prefill:decode work ratio, or full async-TP modeling), deferred.
2. *Small-model overhead regime.* For ≲8B at cross-node TP8, step time is
   dispatch/AR-floor-bound (`step_floor≈48 ms` per hw_params notes) and prefill
   carries a per-chunk CPU floor; a single global `step_floor`/`c_chunk` cannot
   serve both this regime and the 70B/123B saturated regime. These cells
   (8B deep-PP prefill, qwen TP8/PP4+) are reported but excluded from the fit as
   overhead-bound outliers, consistent with the hw_params exclusion policy.

Within-topology layer-skew selection is **noise-limited**: measured PP2 TPS-vs-skew
curves are flat near the optimum (champions win by <5%, comparable to run-to-run
measurement noise; cross-model trends even contradict — opt30b decode prefers
uniform while 70b decode prefers skew+16). The planner is therefore evaluated by
**regret / optimality-gap** (measured TPS lost by picking the predicted config vs
the oracle-best) with Spearman rank-correlation and top-3 hit-rate as secondary
metrics — NOT by exact top-1 champion match, which overfits noise.

---

## 6. Component 5 — Memory feasibility (hard reject)

Per rank `r` (stage `s`, TP group `Nt`):

```
mem_r = params_on_rank(r)·b_w                                       # weights
      + n_req · (in_len + out_len) · L_s · 2 · kv_r · d_h · b_kv    # KV at max len
      + c_act · T_c · max(h, 2·f_r) · b_a                           # peak activations (c_act ≈ 4–8)
      + M_cudagraph                                                  # graph pool, ~1–3 GB (graphs mandatory)
      + M_base                                                       # CUDA ctx + NCCL, ~1.5 GB

feasible(C) ⇔ ∀r:  mem_r ≤ VRAM_g(r) · util        (util = 0.9)
```

Non-uniform caveat: bias REDUCES the small-partition rank's weight share but
NOT its KV share (KV ∝ `kv_r`, which is uniform in Case B) — so the Ada-class
rank's headroom is `VRAM − weights − KV`, and KV grows with `n_req`. On the
current cluster this inequality flips infeasible at `n_req ≳ 100` for the 70B
hetero cells — reproducing the empirical `n_req ≤ 100` rule as a *prediction*,
not a hand rule. Configs failing any rank are rejected, never scored.

---

## 7. Component 6 — Search procedure

```
def plan(M, H, W):
    cands = []
    for tp in divisors(world) if n_q % tp == 0 and (n_kv % tp == 0 or tp <= n_kv):
        pp = world // tp
        for placement in placements(tp, pp, H):        # TP intra-node first; then minimal-cut
            for regime in (decode_speed, prefill_speed, blend(W)):
                splits = closed_form(tp, regime)        # §3.3 Case A or B + quantization
                layers = closed_form_pp(pp, regime)     # §4 + largest-remainder
                for C in neighborhood(splits, layers):  # ±1 head-quantum, ±1 FFN tile, ±1 layer
                    if not feasible(C): continue        # §6
                    cands.append((TPS_wall(C), C))      # §5
    return top_k(cands)
```

Cost: `O(#factorizations · #variants · neighborhood)` ≈ a few hundred
closed-form evaluations — milliseconds, no search heuristics needed. The
`blend(W)` regime weights `speed_r` by the predicted phase shares
`(T_prefill, T_decode)/T_total` so balanced workloads interpolate between the
two pure closed forms. Output: ranked `PartitionSpec` (existing dataclass) with
env-var emission (`VLLM_PP_LAYER_PARTITION`, `TP_HEAD/KV/FFN_SPLITS`) unchanged.

Generalization beyond 2 nodes is structural, not special-cased: §3 only needs
`min_link_bw` over the TP group, §4 only needs per-stage `(τ_s, link to s+1)`;
K GPU classes use the water-filling form of §3.3 and §4 directly.

---

## 8. Calibration — free parameters and the ≤6-cell probe set

Free parameters (everything else is datasheet/model arithmetic):

| param | per | role | fitted from |
|---|---|---|---|
| `TF_g` | GPU type | prefill roofline | probe 1, 2 |
| `BW_g` | GPU type | decode roofline | probe 1, 2 |
| `KBW_g` | GPU type | KV read | probe 1, 2 (two B points) |
| `α_AR, bw_AR` intra | cluster | §3.2 | probe 3 |
| `α_AR, bw_AR` cross | cluster | §3.2 | probe 4 |
| `p2p bw/lat` | link class | §4 send | probe 5 (or nccl-tests, free) |
| `η, η_p2p` | engine | §4 overlap | probe 5 vs 6 |
| `c_cpu, c_mb, c_layer` | engine | overheads | residuals of probes 1, 5, 6 |

Probe set (each = one standard perf-tool cell, CUDA graphs ON; ≤ 6 runs total):

```
P1  TP=G_fast, PP=1, intra-node fast GPUs, balanced workload, two batch points
    (B=8 and B=64 within the run via concurrency ramp)
      decode step(B) = W_bytes/BW_fast + B·k_kv/KBW_fast + 2L·t_AR_intra + c_cpu
      → slope in B gives KBW_fast; intercept (minus P3's AR) gives BW_fast and c_cpu
      prefill chunk time → TF_fast
P2  same on the slow node → TF_slow, BW_slow, KBW_slow
P3  TP=G_fast PP=1 vs TP=1 PP=1 small model (or P1 re-used with single-GPU
    microbench): difference isolates intra-node t_AR(msg) at decode and prefill
    msg sizes → (α_intra, bw_intra) from the 2-point latency/bandwidth fit
P4  TP=world PP=1 cross-node, decode-heavy: step time minus max-rank compute
    → 2L·t_AR_cross → (α_cross, bw_cross) (prefill chunk of same run gives the
    bandwidth point)
P5  TP=world/2 PP=2 cross-node, overlap ON, n_mb=pp: measured T_cycle vs
    max_s t_s (now fully determined by P1–P4) → η and exposed p2p
P6  same as P5 with n_mb=2·pp (or PP=4 if world allows): separates c_mb from η
    (two equations, two unknowns)
```

Fitting is direct inversion (each probe isolates 1–2 parameters), no global
least squares needed; remaining residual is folded into `c_cpu`. Acceptance
gate before trusting the planner on a new cluster: predicted vs measured TPS
within ±15% on all six probes, and rank-order correct on any two probes that
differ in topology.

---

## 9. Sanity checks (the equations must reproduce these directionally)

Verified numerically with current-cluster calibration
(`BW = 1461/798 GB/s`, `TF = 378/145 TFLOPS`, `α_cross ≈ 1 ms`,
`α_intra ≈ 0.05 ms`, `c_cpu ≈ 20 ms`):

| # | phenomenon | mechanism in the model | predicted | measured |
|---|---|---|---|---|
| a | 8B: TP4PP2 ≫ TP8 cross | decode AR latency floor: `2·32·α_cross = 64 ms/step` vs intra `3.2 ms`; weight-stream only `2.5 ms` | emission TPS 5308 vs 1480 (3.6×) | 3036 vs 1608 balanced (1.9×) ✓ dir |
| b | 70B prefill: TP8+FFN(5376:1792) > PP2 uniform | bias drops `max_r` prefill ms/tok 0.122 → 0.072; PP2-uniform stays 0.122 (slow stage = same roofline); cross-AR amortized at `msg = T_c·h·b_a` | 0.072 < 0.122 ms/tok | 1369 > 1319 (and >1155 uniform TP8) ✓ |
| c | OPT-30B decode: head bias > FFN bias | MHA ⇒ `kv_r = head_r`; per head quantum moves 1762 MB (KV 1409 + W 352) vs 264 MB per FFN tile | head knob 6.7× leverage | head(10:4) champions decode ✓ |
| d | TP1PP8 loses unless heavily skewed | `(pp−1)` exposed terms + `c_mb·n_mb` + granularity; skew drops `max_s` 21.4 → 16.4 ms | +31% from skew; PP8 still < PP2 | +32.8% (633→840), PP8 ≪ TP4PP2 ✓ |
| e | FFN bias sweet spot +25..50 | Case-B equalizer: `f_F* = 4861 = +36%`; past it `max_r` rises (attention floor `a_S` unsplittable) | V-shape, min at +36% | champions at +25/+50, worse beyond ✓ |

Bonus consistency point for the paper: §3.3-A and §4's closed forms converge to
the same ideal `work_r ∝ speed_r` roofline (biased TP8 0.072 vs skewed PP2
58:22 → 0.068 ms/tok) — the *residual* difference between strategies is purely
the comm structure (AR latency floor vs p2p/η) plus quantization granularity,
which is exactly what the planner trades off.

---

## 10. Mapping to existing code (v1 → v2 deltas)

| spec section | file | change |
|---|---|---|
| §2, §3.2 | `cost_model.py` | replace per-shape matmul lookup path's roofline fallback with params-fraction roofline; AR: replace flat busbw with `min_link_bw` ring + `2(Nt−1)·α` form |
| §3.3 | `planner.py:_derive_tp_splits` | add Case-B water-filling equalizer (currently only `frac ∝ speed`); keep GQA/tile quantization + largest remainder |
| §4 | `cost_model.py:step_time_s` | add `(1−η)Σ` term and `c_mb·n_mb`; drop fitted `HETERO_RATIO_ATTENUATION` (now derived); take `η` from `launcher/pp_overlap_config.py` auto-tuner regime |
| §5 | `cost_model.py:wall_time_s` | replace `n_req · t_prefill(B=1)` with chunked-pipeline prefill; align with perf-tool wall definition |
| §6 | new `feasibility.py` | currently scorer-level penalty only; make hard reject |
| §8 | new `calibrate.py` | probe-set runner + direct inversion; emits the cluster JSON consumed by `GpuProfile`/`NetworkProfile` |
