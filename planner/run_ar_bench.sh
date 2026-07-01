#!/bin/bash
# Multi-node AR microbench launcher. n_local = NPROC ranks/node, 2 nodes. Robust env
# forwarding via `env`; script lives on the SHARED /scfs so both nodes see it. Extra
# env passed as trailing KEY=VAL args. Usage: run_ar_bench.sh NPROC [KEY=VAL ...]
set -u
NPROC="${1:-4}"; shift || true
EXTRA=("$@")                                    # array so KEY=VAL survive intact
MASTER=10.20.0.30; PORT="${AR_PORT:-$((29500 + RANDOM % 200))}"   # fabric blocks all but 6379
HEAD_PY=/data/esca/uckim/miniconda3/envs/vllm_main/bin/python
WORKER_PY=/data/esca/uckim/miniconda3/envs/vllm_new/bin/python
SRC="${AR_SCRIPT:-ar_microbench.py}"            # which microbench to run
SCRIPT=/scfs/esca/$(basename "$SRC")            # shared path (both nodes)
cp -f "/data/esca/uckim/vllm_main/planner/$SRC" "$SCRIPT"
DBG="${NCCL_DEBUG:-WARN}"
RDZV="--nnodes=2 --nproc_per_node=$NPROC --master_addr=$MASTER --master_port=$PORT"
CG="AR_CUDA_GRAPH=${AR_CUDA_GRAPH:-1}"
echo "# n_local=$NPROC  port=$PORT  extra='${EXTRA[*]}'  dbg=$DBG"
# worker node_rank=1 (background, ssh, worker IFNAME + vllm_new py). env before the cmd.
ssh esca@10.20.0.28 \
  "env NCCL_SOCKET_IFNAME=ibp34s0 NCCL_IB_HCA=mlx5 NCCL_NET_GDR_LEVEL=2 NCCL_DEBUG=$DBG $CG ${EXTRA[*]} \
   $WORKER_PY -m torch.distributed.run $RDZV --node_rank=1 $SCRIPT" \
  >/scfs/esca/arbench_worker.log 2>&1 &
WPID=$!
sleep 2
# head node_rank=0 (foreground)
env NCCL_SOCKET_IFNAME=ibp3s0 NCCL_IB_HCA=mlx5 NCCL_NET_GDR_LEVEL=2 NCCL_DEBUG=$DBG $CG "${EXTRA[@]}" \
  $HEAD_PY -m torch.distributed.run $RDZV --node_rank=0 $SCRIPT
wait $WPID 2>/dev/null
