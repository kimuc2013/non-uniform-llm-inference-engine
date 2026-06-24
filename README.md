# Heterogeneous TP/PP serving on vLLM — non-uniform parallelism + analytical planner

Research framework for **non-uniform tensor + pipeline parallelism** on a
heterogeneous GPU cluster (developed on 4× Blackwell-96GB head + 4× Ada-48GB
worker, cross-node IB). It contains: (1) vLLM source patches that enable
non-uniform TP shards + a fast PP-overlap path, (2) an **analytical planner**
that predicts throughput and picks `(TP, PP, per-rank FFN/head splits,
per-stage layer split)`, and (3) a generalized measurement sweep + plotting.

**Paper thesis:** *hybrid parallelism is not always best* — the optimal topology
and the value of non-uniform partitioning are workload-, model-, and
hardware-dependent, and a small calibrated cost model can pick within a few
percent of the measured optimum.

## Headline results
- **Planner accuracy.** Closed-form cost model (calibrated once, ~9 effective
  params): **mean regret 2.3%** on production workloads (median 0%, Spearman 0.82);
  **18 self-consistency invariants** all hold. At the realistic operating point —
  *balanced workload (covers prefill+decode), saturating concurrency (n≥32), 4+4
  layout* — **regret is 0%** (picks the measured champion for every model).
  → `planner_describe.md`, `planner/check_consistency.py`.
- **Never slower than the naive baseline.** The `plan_safe` guard deviates from
  uniform-TP only when confident; across **all 55 measured cells** (incl. a
  held-out workload + zero-refit 1+1/2+2 layouts) it has **0 baseline-losses**, and
  at saturating load it beats the naive default by **+23% mean (+40–78% at n≥64)**.
  → `planner/verify_vs_baseline.py`.
- **Generalizes without re-fitting.** Calibrated on 4+4, the cost model is
  layout-parametric: zero-refit transfer to **2+2 (regret 7.5%) and 1+1 (3.4%)**,
  covering the 1+1→4+4 target. Pre-registered on **Mistral-Large-123B** (predicted
  *before* any measurement, ~1.8× the largest calibration model): **champion 3/3,
  regret 0.0%, Spearman 0.84**. → `figures/fig_planner_validation.png`, `figures/fig_mistral123b_prereg.png`.
- **PP overlap fork**: cross-node PP reaches **56–78%** overlap (stock vLLM 12–24%);
  70B 100-req **+16%** throughput. (M13 side-stream sampled-token broadcast = 2×.)
- **Non-uniform TP**: helps **+7–13%** depending on regime; gain grows as TP degree
  shrinks (more per-rank weight, smaller AR floor) — e.g. 70B balanced TP8 **+6.3%**
  (8 GPU) vs TP4 **+12.9%** (4 GPU). Non-uniform PP (layer skew) **+23%** on 2+2.
- Across 4 calibration models the champion is almost always **TP4-PP2 (overlap)**,
  but **70B prefill prefers TP8+FFN-bias** — i.e. not always hybrid.

## Repo layout (post-cleanup 2026-06)
```
patches/
  vllm-0.22.0-hetero-tp-pp.patch   ★ all vLLM source changes (11 files, +1707/-90)
  APPLY.md                          how to re-apply on a fresh vLLM 0.22.0
planner_describe.md                 ★ full pedagogical study guide (intuition + math + code refs)
planner/                            ★ analytical planner (v2) + sweep + plots
  perf_planner.py        closed-form TPS predict / plan() / plan_safe() / --validate / CLI
  fit_planner.py         fit the ~9 effective params (robust loss over tps+itl+ttft, LOMO)
  build_calibration.py   (re)build calibration_data.csv from results/ (additive accumulator)
  hetero_sweep.py        ★ generalized measurement sweep (any model × GPU layout; --extra-workload)
  cluster_env.py         typed cluster config (reads cluster.local.env)
  cluster_setup_nxn.py   parameterized Ray (re)config for any N+N layout (1+1..4+4)
  cluster_setup_4x4.py   idempotent Ray restart on both nodes (4+4)
  check_consistency.py   18 self-consistency invariants (logic soundness, not calibration)
  validate_concurrency.py  layout-parametric per-n_req champion/regret (zero-refit generalization)
  verify_vs_baseline.py  plan_safe never-slower-than-baseline check
  layout_summary.py      GPU-count axis: FFN-bias vs PP-skew gain
  plot_*.py / plot_paper_figures.py   matplotlib figures
  *_pp_overlap_*.py       PP-overlap verification (nsys / torch-profiler)
  PLANNER_SPEC.md        the full cost-model derivation (roofline + hierarchical AR + overlap + memory)
  HANDOFF.md             detailed running status / next steps
  hw_params.json         fixed effective HW params (per-GPU TFLOPS/BW, NVLink/IB, prefill-AR-overlap)
  fitted_params.json     fitted engine params (AR, step_floor, overlap, prefill_overlap, kv_bw_scale)
  calibration_data.csv   measured calibration (4 models × config × workload × concurrency; n_req≤100)
  mistral_prediction.json / mistral_validation.json   pre-registration + comparison
  legacy_v1/             archived v1 cost-model planner + one-off diagnostics + old per-model sweeps
launcher/                vLLM launcher wrapper + pp_overlap_config.py (PP-overlap auto-tuner)
perf/                    request driver (TPS/TTFT/ITL) + NVTX trace analyzers
asim_etgen/              ASTRA-sim execution-trace generator for the same workloads
results/final/           committed curated CSVs (per-model sweeps + Mistral validation)
figures/                 committed figures
cluster.example.env      copy to cluster.local.env (gitignored) and fill in
```
vLLM is **not** vendored — install 0.22.0 and apply the patch (keeps this a thin delta).

## Setup on a new server
```bash
# 1. env + vLLM + patch
pip install -r requirements.txt          # vllm 0.22.0 + torch/ray/scipy/matplotlib/transformers (pinned)
#    then apply the source patch onto the installed vLLM — see patches/APPLY.md
# 2. cluster config (no secrets in the repo)
cp cluster.example.env cluster.local.env   # edit: IPs, GPU counts, IB ifaces, python paths
# 3. bring up Ray (cross-node). Per-node NCCL/GLOO ifaces matter:
#    head:   NCCL_SOCKET_IFNAME=<head_iface>  ray start --head --node-ip-address=<head_fabric_ip> --port=6379 [--num-gpus N]
#    worker: NCCL_SOCKET_IFNAME=<worker_iface> ray start --address=<head>:6379 --node-ip-address=<worker_fabric_ip> --num-gpus N
#    (planner/cluster_setup_4x4.py automates this; `ray status` must show head+worker GPUs)
```

### Models / HF access
Model weights are **not** in the repo (re-download). Gated models (Llama-3.x,
Mistral-Large) need a token + accepted license:
```bash
huggingface-cli login                    # token with access to the gated repos
# accept the license on huggingface.co for each model, then it downloads on first use
```
On a node where the model is already cached, set `HF_HUB_OFFLINE=1` (the sweep
does this automatically) to skip Hub metadata fetches — gated repos otherwise
429-throttle the per-shard header calls and stall model load for hours.

> **Site-specific helper scripts.** Core paths are config-driven (cluster.local.env)
> or repo-relative. A few **reproduction utilities** still hardcode the dev cluster
> — `planner/nsys_pp_overlap_*.py` and `planner/verify_pp_overlap_torch_profiler.py`
> (worker SSH target, nsight path) and `planner/plot_4topology_nonuniform.py` /
> `planner/analyze_pp_overlap_trace.py` (specific result-dir timestamps). Use them as
> templates: set the toolchain in cluster.local.env (`CUDA_HOME/CC/CXX/NSYS_BIN`) and
> point them at your own result dirs.

## Run a sweep (generalized)
```bash
python planner/hetero_sweep.py --model 70b                          # 4+4, full topology grid, 3 workloads
python planner/hetero_sweep.py --model 70b --head-gpus 2 --worker-gpus 2 --workloads balanced
python planner/hetero_sweep.py --model mistral123b --dry-run        # print the config grid, no launch
```
Model dims come from `planner.perf_planner.MODELS`; the config grid (TP=world
FFN-bias + TP×PP layer-skew, Blackwell-biased) is generated from dims + layout.
Hard rules are baked in: CUDA graphs ON, `gpu_mem 0.85`, HF offline (gated-model
safe), PP overlap via the launcher auto-tuner only. Outputs land in
`results/hetero_<H>x<W>_<model>_<ts>/` (gitignored; curate into `results/final/`).

## The planner
```bash
python planner/perf_planner.py --model 70b --in-len 512 --out-len 256 --n-req 96   # rank configs by predicted TPS
python planner/perf_planner.py --validate          # vs calibration_data.csv (MAPE + regret + champion)
python planner/fit_planner.py                       # refit free params; reports regret / Spearman / LOMO
```
Reported as **regret / optimality-gap** (not exact top-1, which is noise-limited on
flat curves) + Spearman + MAPE. See `PLANNER_SPEC.md` for the model and the
documented residual gaps (TP8-prefill under-prediction; small-model overhead regime).

## Reproduce / resume
- Curated results: `results/final/*.csv`, `figures/*.png`. Regenerate figures with `planner/plot_*.py`.
- Recalibrate on a new cluster: run the ≤6-cell probe set (PLANNER_SPEC §8) →
  update `hw_params.json` → `fit_planner.py` → `perf_planner.py --validate`.
- **To resume (or hand to a fresh Claude Code session): read `planner/HANDOFF.md` first**
  — running status, done/pending, infra gotchas (HF gated-model load needs
  `HF_HUB_OFFLINE=1`; per-node IB ifaces; Ray restart), and the file map.

## Hard rules (do not violate for reported numbers)
CUDA graphs ON (never `--enforce-eager`); `gpu_mem 0.85`; `n_req ≤ 100` (Ada
small-partition rank OOMs above); PP overlap only via
`launcher/pp_overlap_config.auto_configure`.
