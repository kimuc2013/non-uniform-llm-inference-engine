#!/bin/bash
# Multi-node AR microbench launcher. Measures cross-node all-reduce bandwidth at
# n_local = NPROC (ranks per node), 2 nodes. Pass extra env (e.g. NCCL_PROTO=Simple
# AR_OVERLAP=3) as trailing KEY=VAL args — forwarded to BOTH nodes.
#   Usage: run_ar_bench.sh NPROC [KEY=VAL ...]
set -u
NPROC="${1:-4}"; shift || true
EXTRA="$*"
MASTER=10.20.0.30; PORT=$((29500 + RANDOM % 200))
HEAD_PY=/data/esca/uckim/miniconda3/envs/vllm_main/bin/python
WORKER_PY=/data/esca/uckim/miniconda3/envs/vllm_new/bin/python
COMMON="NCCL_IB_HCA=mlx5 NCCL_NET_GDR_LEVEL=2 AR_CUDA_GRAPH=${AR_CUDA_GRAPH:-1} $EXTRA"
RDZV="--nnodes=2 --nproc_per_node=$NPROC --master_addr=$MASTER --master_port=$PORT"
echo "# n_local=$NPROC  port=$PORT  extra='$EXTRA'"
# worker node_rank=1 (background, over ssh, worker IFNAME + vllm_new py)
ssh esca@10.20.0.28 "cd /data/esca/uckim/vllm_main && NCCL_SOCKET_IFNAME=ibp34s0 $COMMON \
  $WORKER_PY -m torch.distributed.run $RDZV --node_rank=1 planner/ar_microbench.py" \
  >/scfs/esca/arbench_worker.log 2>&1 &
WPID=$!
sleep 2
# head node_rank=0 (foreground, head IFNAME + vllm_main py)
cd /data/esca/uckim/vllm_main
NCCL_SOCKET_IFNAME=ibp3s0 $COMMON \
  $HEAD_PY -m torch.distributed.run $RDZV --node_rank=0 planner/ar_microbench.py
wait $WPID 2>/dev/null
