#!/bin/bash
# Full re-measurement campaign for a given layout. Runs hetero_sweep per model
# with the paper config set (baseline + planner picks + non-uniform variants)
# across all 3 workloads and the concurrency sweep. Usage:
#   bash planner/run_campaign.sh <head_gpus> <worker_gpus> <model1> <model2> ...
cd /data/esca/uckim/vllm_main || exit 1
PY=/data/esca/uckim/miniconda3/envs/vllm_main/bin/python
HG=$1; WG=$2; shift 2
MODELS="$@"
CONFIGS="${CAMPAIGN_CONFIGS:-uniform,skew+8,skew+12,ffn_bias+50}"
WL="${CAMPAIGN_WORKLOADS:-balanced,decode_heavy,prefill_heavy}"
NREQ="${CAMPAIGN_NREQ:-16,32,64,96}"
LOG=/scfs/esca/campaign_${HG}x${WG}.log

echo "================ CAMPAIGN ${HG}+${WG}  models=[$MODELS]  start $(date) ================" | tee -a "$LOG"
for m in $MODELS; do
  echo "" | tee -a "$LOG"
  echo ">>>>>>>> [$m] ${HG}+${WG}  start $(date) <<<<<<<<" | tee -a "$LOG"
  $PY planner/hetero_sweep.py --model "$m" --head-gpus "$HG" --worker-gpus "$WG" \
      --workloads "$WL" --configs "$CONFIGS" --n-req-list "$NREQ" 2>&1 | tee -a "$LOG" \
      | grep -E "MODEL=|CONFIGS=|\[${m}|n=[0-9]+: success|Wrote|FAILED" | tee -a "${LOG}.summary"
  echo ">>>>>>>> [$m] ${HG}+${WG}  done $(date) <<<<<<<<" | tee -a "$LOG"
done
echo "================ CAMPAIGN ${HG}+${WG} COMPLETE $(date) ================" | tee -a "$LOG"
