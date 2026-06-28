#!/bin/bash
# Re-measure 1+1 and 2+2 layouts (balanced workload, n=32/64/96) for the 4 eval
# models at their feasible layouts. Reconfigures ray between layouts, restores 4+4.
set -u
PY=/data/esca/uckim/miniconda3/envs/vllm_main/bin/python
REPO=/data/esca/uckim/vllm_main
cd "$REPO" || exit 1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

reconfig() { # head worker
  log "=== RECONFIG ${1}+${2} ==="
  $PY planner/cluster_setup_nxn.py --head-gpus "$1" --worker-gpus "$2"
  rc=$?
  if [ $rc -ne 0 ]; then log "!! RECONFIG ${1}+${2} FAILED (rc=$rc)"; return 1; fi
  log "reconfig ${1}+${2} OK"
}

sweep() { # model head worker
  log ">> SWEEP $1 @ ${2}+${3} (balanced, n=32,64,96)"
  $PY planner/hetero_sweep.py --model "$1" --head-gpus "$2" --worker-gpus "$3" \
      --workloads balanced --n-req-list 32,64,96
  rc=$?
  log "<< SWEEP $1 @ ${2}+${3} done (rc=$rc)"
}

log "########## LAYOUT RE-MEASURE START ##########"

# ---- 1+1 ----
reconfig 1 1 || { log "abort: 1+1 reconfig failed"; exit 1; }
sweep 8b 1 1
sweep opt30b 1 1

# ---- 2+2 ----
reconfig 2 2 || { log "abort: 2+2 reconfig failed"; $PY planner/cluster_setup_nxn.py --head-gpus 4 --worker-gpus 4; exit 1; }
sweep 8b 2 2
sweep 70b 2 2
sweep opt30b 2 2

# ---- restore 4+4 ----
reconfig 4 4 || log "!! 4+4 restore failed — restore manually"

log "########## LAYOUT RE-MEASURE DONE ##########"
