#!/bin/bash
# Calibration helper: one graph-chain AR measurement (B=64, n_local=4) for the
# HardwareProfiler. Stops ray (frees 6379), runs the captured [compute->AR]xL decode
# chain, restores ray 4+4. Prints the ISOLATED / GRAPHCHAIN lines the profiler parses.
set -u
HR=/data/esca/uckim/miniconda3/envs/vllm_main/bin/ray
WR=/data/esca/uckim/miniconda3/envs/vllm_new/bin/ray
W=esca@10.20.0.28
$HR stop --force >/dev/null 2>&1; ssh $W "$WR stop --force >/dev/null 2>&1"; sleep 8
AR_PORT=6379 AR_SCRIPT=graph_chain_ar_microbench.py AR_BATCH=64 AR_LAYERS=80 NCCL_DEBUG=WARN \
  timeout 200 bash /data/esca/uckim/vllm_main/planner/run_ar_bench.sh 4 AR_BATCH=64 AR_LAYERS=80 2>&1 \
  | grep -E "ISOLATED|GRAPHCHAIN|graph-chain"
pkill -u esca -f "[g]raph_chain" 2>/dev/null; ssh $W 'pkill -u esca -f "[g]raph_chain"' 2>/dev/null; sleep 3
$HR stop --force >/dev/null 2>&1; sleep 4
VLLM_HOST_IP=10.20.0.30 NCCL_SOCKET_IFNAME=ibp3s0 NCCL_IB_HCA=mlx5 $HR start --head --node-ip-address 10.20.0.30 --port 6379 --num-gpus 4 >/dev/null 2>&1; sleep 8
ssh $W "$WR stop --force >/dev/null 2>&1; sleep 3; export CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_HOST_IP=10.20.0.28 NCCL_SOCKET_IFNAME=ibp34s0 NCCL_IB_HCA=mlx5; $WR start --address=10.20.0.30:6379 --node-ip-address=10.20.0.28 --num-gpus=4 >/dev/null 2>&1"; sleep 8
