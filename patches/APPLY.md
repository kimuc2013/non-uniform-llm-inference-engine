# vLLM custom patch — heterogeneous TP/PP

`vllm-0.22.0-hetero-tp-pp.patch` carries all source-level modifications to vLLM
needed by this project. vLLM itself is **not** vendored — install the matching
release and apply this patch on top, so the repo stays a thin delta.

## Base version
- **vLLM 0.22.0** (pip wheel `vllm-0.22.0-cp38-abi3-manylinux_2_28_x86_64.whl`).
- Patch: **11 files, +1707 / −90 lines** (10 modified + 1 new file).
- Verified: `git apply --check` is clean against the pristine 0.22.0 wheel.

## Apply (on a new server)
```bash
pip install vllm==0.22.0
VLLM_DIR=$(python -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))")
cd "$VLLM_DIR"                       # the site-packages dir that contains vllm/
git apply -p1 /path/to/patches/vllm-0.22.0-hetero-tp-pp.patch
# (not a git repo? use:  patch -p1 < .../vllm-0.22.0-hetero-tp-pp.patch )
```
If vLLM has moved past 0.22.0, expect hunks to need rebasing — the changes are
localized (below) and the env-var contract is stable.

## What the patch changes (by feature)

### 1. Non-uniform tensor parallelism (per-rank uneven shards)
Lets a TP group give faster GPUs (Blackwell) larger FFN/head shards than slower
ones (Ada). Driven by env vars (read at model build):
`VLLM_TP_FFN_SPLITS`, `VLLM_TP_HEAD_SPLITS`, `VLLM_TP_KV_SPLITS` (comma-sep,
len == TP size).
- `vllm/utils/tp_split_utils.py` *(new)* — parse/validate per-rank split lists.
- `vllm/model_executor/layers/linear.py` — `Column/MergedColumn/RowParallelLinear`
  accept `partition_sizes` (uneven) instead of assuming `dim // tp`.
- `vllm/model_executor/parameter.py` — weight loading for uneven shards.
- `vllm/model_executor/models/llama.py` — `LlamaMLP`/attention wire the splits
  in (logs `[vllm_main] ffn_splits=...`). (Qwen/OPT reuse Llama paths.)

### 2. Pipeline-parallel overlap (fork's microbatch + side-stream broadcast)
Overlaps PP stages so cross-node PP reaches 56–78% overlap (vs stock 12–24%).
Env (set by `launcher/pp_overlap_config.py` auto-tuner): `VLLM_PP_LAYER_PARTITION`,
`VLLM_PP_MICROBATCH`, `VLLM_PP_MICROBATCH_SIZE`, `VLLM_PP_BATCH_QUEUE_SIZE`,
`VLLM_PP_SAMPLED_BROADCAST_STREAM`.
- `vllm/v1/worker/gpu_model_runner.py` — **the key fix**: move
  `_pp_receive_prev_sampled_token_ids` broadcast onto a side stream + lazy
  wait_event (was spin-waiting 26 ms/mb on the default stream); microbatch the
  in-flight requests into disjoint subsets.
- `vllm/v1/core/sched/scheduler.py` — bq (batch-queue) microbatch scheduling.
- `vllm/v1/executor/multiproc_executor.py`, `vllm/v1/engine/core.py` — plumb the
  microbatch/overlap path.
- `vllm/distributed/device_communicators/shm_broadcast.py` — broadcast transport.
- (See memory/PP-overlap notes; the M13 side-stream broadcast is the 2× win.)

### 3. Heterogeneous kernel-warmup fix
- `vllm/model_executor/warmup/kernel_warmup.py` + `vllm/v1/worker/gpu_worker.py`
  — collective MIN all-reduce gates flashinfer autotune so a mixed
  Blackwell+Ada cluster enters warmup all-or-nothing (else 4+4 cross-node PP=2
  deadlocks on the capability-gated autotune).

## Hard rules (baked into the sweep harness, keep them on re-apply)
CUDA graphs ON (never `--enforce-eager` for reported numbers), `gpu_mem 0.85`,
`n_req ≤ 100` (Ada small-partition rank OOMs above that), PP overlap **only** via
`launcher/pp_overlap_config.auto_configure` (adding `VLLM_PP_OVERLAP=1`/
`VLLM_PP_FAST_COMM=1` on top deadlocks cross-node).
