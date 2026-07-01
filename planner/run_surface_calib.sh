#!/bin/bash
# MEASURE the cross-node AR surface: ar_microbench at nproc_per_node in {1,2,4} (n_local),
# 2 nodes, serving NCCL env. Prints "SURFACE <n_local> <msg_MB> <bw>" rows.
set -u
HR=/data/esca/uckim/miniconda3/envs/vllm_main/bin/ray
WR=/data/esca/uckim/miniconda3/envs/vllm_new/bin/ray
$HR stop --force >/dev/null 2>&1; ssh esca@10.20.0.28 "$WR stop --force >/dev/null 2>&1"; sleep 6
for NL in 1 2 4; do
  OUT=$(AR_PORT=6379 AR_CUDA_GRAPH=1 timeout 150 bash /data/esca/uckim/vllm_main/planner/run_ar_bench.sh $NL 2>&1)
  echo "$OUT" | grep -E "^ +[0-9]" | grep -vE "WARNING|OMP" | awk -v nl=$NL '{print "SURFACE",nl,$3,$6}'
  pkill -u esca -f "[a]r_microbench" 2>/dev/null; ssh esca@10.20.0.28 'pkill -u esca -f "[a]r_microbench"' 2>/dev/null; sleep 3
done
VLLM_HOST_IP=10.20.0.30 NCCL_SOCKET_IFNAME=ibp3s0 NCCL_IB_HCA=mlx5 $HR start --head --node-ip-address 10.20.0.30 --port 6379 --num-gpus 4 >/dev/null 2>&1; sleep 8
ssh esca@10.20.0.28 "$WR stop --force >/dev/null 2>&1; sleep 3; export CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_HOST_IP=10.20.0.28 NCCL_SOCKET_IFNAME=ibp34s0 NCCL_IB_HCA=mlx5; $WR start --address=10.20.0.30:6379 --node-ip-address=10.20.0.28 --num-gpus=4 >/dev/null 2>&1"; sleep 8
echo "SURFACE CALIB DONE"
