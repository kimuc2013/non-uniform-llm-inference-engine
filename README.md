# vllm_main — Heterogeneous TP/PP serving on vLLM

Research framework for benchmarking **non-uniform tensor + pipeline parallelism**
across a heterogeneous GPU cluster (e.g. 4× Blackwell head + 4× Ada worker).
Includes a cost-model-based planner that picks (TP, PP) and per-stage layer /
FFN / head splits, and an end-to-end sweep harness that compares against stock
vLLM as the SOTA baseline.

Paper claim: hetero clusters benefit from **non-uniform** PP layer splits +
per-stage TP sharding; the right partition is workload-dependent and a small
cost model can pick it within a few percent of measured optimum.

## Layout

```
launcher/           vLLM launcher wrapper — reads cluster.local.env
perf/               request-driver (TPS, TTFT, ITL) + NVTX trace analyzers
planner/            ★ main entry points
  cluster_env.py        load cluster.local.env (single source of truth)
  cluster_setup_4x4.py  idempotent ray restart on both nodes
  hetero_4x4_70b_sweep.py   70B sweep (TP8/PP1, TP4/PP2 with layer skew, TP2/PP4)
  hetero_4x4_8b_sweep.py    same configs, Llama-3.1 8B
  auto_sweep_robust.sh      autonomous runner: PP verify → full sweeps with retry
  plot_4x4_results.py       matplotlib figures from sweep results
  planner.py / cost_model.py / scorer.py / model_spec.py / workload.py / gpu_library.py
  network_library.py / hetero_eval.py / run_eval.py / cli.py
asim_etgen/         ASTRA-sim execution-trace generator for the same workloads
results/final/      committed: aggregated CSVs (no intermediate cells)
figures/            committed: matplotlib figures from the 4+4 sweeps
```

## Setup

1. Install vLLM (this work was developed against the local `vllm_main` fork —
   any vLLM with the necessary hetero patches works).
2. Apply the upstream hetero fix in `vllm/model_executor/warmup/kernel_warmup.py`
   on every node. In a heterogeneous cluster, only ranks with compute
   capability ≥ 9.0 originally entered `flashinfer_autotune`, which deadlocks
   when other ranks skip it. Our patch coordinates via a CPU all-reduce so all
   ranks make the same decision — see the patch file in commit history.
3. Copy `cluster.example.env` to `cluster.local.env` (gitignored) and fill in
   your cluster's IPs, GPU counts, fabric interfaces, and Python paths.

## Running a full 4+4 sweep

```bash
cd vllm_main
# Sanity check: ensure both nodes are 4+4 in Ray
python -m planner.cluster_env             # print loaded config
python planner/cluster_setup_4x4.py       # idempotent — restart if not 4+4

# Run everything (PP verify → 70B 27 cells → 8B 27 cells), auto-retry + restart on failure
./planner/auto_sweep_robust.sh
```

Aggregate + plot:
```bash
python planner/plot_4x4_results.py        # writes figures/*.png
```

## Single sweep cell (for debugging)

```bash
python planner/hetero_4x4_70b_sweep.py \
    --configs TP4PP2_layer_uniform_40-40 \
    --workloads balanced
```

## Reference numbers (4+4 cross-node, n_req=128)

These are from the committed `results/final/sweep_4x4_70b.csv` and
`sweep_4x4_8b.csv`. All cells passed `success=true`; full reproduction
requires the matching cluster.local.env.

Stock vLLM PP=2 baseline (no PP overlap envs): 1280 TPS on TP=4 PP=2 [40,40]
balanced 70B 128-req. All TP=4 PP=2 cells in the sweep beat this.

### 70B Llama-3.3 highlights
- TP=4 PP=2 [44,36] balanced: 1544 TPS (best balanced)
- TP=4 PP=2 [56,24] decode_heavy: 1655 TPS (best decode, +9% over uniform)
- TP=2 PP=4 always loses (PP bubble cost > skew gain on this cluster)

### 8B Llama-3.1 highlights
- TP=4 PP=2 dominates cross-node TP=8 (3.2× on uniform, up to 4× on decode)
- TP=4 PP=2 [22,10] decode_heavy: 7108 TPS / [24,8]: 7194 TPS (peak)
- Aggressive skew helps decode, hurts prefill — sweet spot is workload-specific

## Key system findings baked in

- **Heterogeneous cluster requires `flashinfer_autotune` patch** (see Setup §2).
  Without it, mixed-cap clusters deadlock at init.
- **PP overlap envs** (`VLLM_PP_OVERLAP`, `VLLM_PP_MICROBATCH_*`, `VLLM_PP_FAST_COMM`)
  give a +7–18% bump on stable clusters but were disabled in the public sweep
  because repeated `ray restart` corrupts NCCL state and they hang. Keep them
  for stable measurement runs only.
- **`mb_size = max_num_seqs / pp`** — too small fragments weights into many
  reads per step (32× amplification on 70B); the sweep uses
  `microbatch_size = 64` for `n_req=128, PP=2`.
- **Worker GPU visibility**: if the physical worker has more GPUs than the
  experiment requires, restart Ray on the worker with
  `CUDA_VISIBLE_DEVICES=0,1,2,3 ray start --num-gpus=4` so the scheduler
  doesn't accidentally pack all bundles onto one node.

## License & status

Internal research code; no warranty.
