#!/bin/bash
# Re-measure 1+1 and 2+2 (balanced, n=32/64/96) for the eval models, using a
# RELIABLE reconfig: head ray restart (local) + worker ray restart via SSH with the
# vllm_new env (the stock cluster_setup_nxn worker path fails to import planner on
# the worker). Cluster is assumed to ALREADY be at 1+1 when this starts.
set -u
PY=/data/esca/uckim/miniconda3/envs/vllm_main/bin/python
HEAD_RAY=/data/esca/uckim/miniconda3/envs/vllm_main/bin/ray
WORKER_RAY=/data/esca/uckim/miniconda3/envs/vllm_new/bin/ray
HEAD_IP=10.20.0.30; HEAD_IB=ibp3s0; PORT=6379
WORKER_IP=10.20.0.28; WORKER_IB=ibp34s0; HCA=mlx5
REPO=/data/esca/uckim/vllm_main
cd "$REPO" || exit 1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

reconfig() { # hg wg
  local hg=$1 wg=$2 want=$(( $1 + $2 ))
  log "=== RECONFIG ${hg}+${wg} (head local + worker SSH) ==="
  $HEAD_RAY stop --force >/dev/null 2>&1; sleep 6
  VLLM_HOST_IP=$HEAD_IP NCCL_SOCKET_IFNAME=$HEAD_IB NCCL_IB_HCA=$HCA \
    $HEAD_RAY start --head --node-ip-address $HEAD_IP --port $PORT --num-gpus $hg >/dev/null 2>&1
  sleep 8
  ssh -o BatchMode=yes esca@$WORKER_IP "
    $WORKER_RAY stop --force 2>&1 | tail -1; sleep 3;
    export CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_HOST_IP=$WORKER_IP NCCL_SOCKET_IFNAME=$WORKER_IB NCCL_IB_HCA=$HCA;
    $WORKER_RAY start --address=$HEAD_IP:$PORT --node-ip-address=$WORKER_IP --num-gpus=$wg 2>&1 | tail -3" 2>&1 | tail -4
  sleep 10
  local got=$($HEAD_RAY status 2>/dev/null | grep -oE "[0-9.]+/[0-9.]+ GPU" | head -1)
  log "reconfig ${hg}+${wg}: ray='$got' (want ${want}.0 GPU)"
  case "$got" in *"/${want}.0 GPU"*) return 0;; *) log "!! reconfig ${hg}+${wg} GPU mismatch"; return 1;; esac
}

sweep() { # model hg wg
  log ">> SWEEP $1 @ ${2}+${3}"
  $PY planner/hetero_sweep.py --model "$1" --head-gpus "$2" --worker-gpus "$3" \
      --workloads balanced --n-req-list 32,64,96
  log "<< SWEEP $1 @ ${2}+${3} done (rc=$?)"
}

log "########## RE-MEASURE START (cluster already 1+1) ##########"

# ---- 1+1 (already configured) ----
sweep 8b 1 1
sweep opt30b 1 1

# ---- 2+2 ----
reconfig 2 2 || { log "abort at 2+2"; reconfig 4 4; exit 1; }
sweep 8b 2 2
sweep 70b 2 2
sweep opt30b 2 2

# ---- restore 4+4 ----
reconfig 4 4 || log "!! 4+4 restore failed — restore manually"

log "########## RE-MEASURE DONE ##########"
