#!/bin/bash
# PRE-SERVING effective-AR measurement. Stops ray (frees the only fabric-open port
# 6379), verifies the port, runs ar_microbench at n_local=4 over InfiniBand:
#   pass 0 = ISOLATED (batch sweep, CUDA-graph) -> reproduces iso_ar_surface (sanity)
#   pass K = PIPELINED (K independent in-flight AllReduces) -> sustained bw
# The pipelined/isolated ratio = the in-decode overlap boost, MEASURED not fit.
# Restores the 4+4 ray cluster afterward. NCCL_DEBUG=WARN surfaces any IB init hang.
set -u
HEAD_RAY=/data/esca/uckim/miniconda3/envs/vllm_main/bin/ray
WORKER_RAY=/data/esca/uckim/miniconda3/envs/vllm_new/bin/ray
W=esca@10.20.0.28

echo "### stop ray (free 6379) ###"
$HEAD_RAY stop --force >/dev/null 2>&1
ssh $W "$WORKER_RAY stop --force >/dev/null 2>&1"
sleep 8
echo "### port 6379 free? -> $(ssh $W 'timeout 3 bash -c "echo>/dev/tcp/10.20.0.30/6379" 2>/dev/null && echo STILL-LISTENING || echo free')"

echo "### PASS isolated (AR_PIPELINE=0, CUDA-graph) ###"
AR_PORT=6379 AR_CUDA_GRAPH=1 NCCL_DEBUG=WARN timeout 200 bash /data/esca/uckim/vllm_main/planner/run_ar_bench.sh 4 2>&1 \
  | grep -E "^#|^ +[0-9]|NCCL WARN|error|Error|Traceback" | grep -vE "Setting|OMP" | head -20
pkill -u esca -f "[a]r_microbench"; ssh $W 'pkill -u esca -f "[a]r_microbench"'; sleep 4

for K in 4 8; do
  echo "### PASS pipelined K=$K ###"
  AR_PORT=6379 AR_CUDA_GRAPH=0 NCCL_DEBUG=WARN timeout 200 bash /data/esca/uckim/vllm_main/planner/run_ar_bench.sh 4 AR_PIPELINE=$K 2>&1 \
    | grep -E "^#|^ +[0-9]|NCCL WARN|error|Error" | grep -vE "Setting|OMP" | head -20
  pkill -u esca -f "[a]r_microbench"; ssh $W 'pkill -u esca -f "[a]r_microbench"'; sleep 4
done

echo "### restore ray 4+4 ###"
VLLM_HOST_IP=10.20.0.30 NCCL_SOCKET_IFNAME=ibp3s0 NCCL_IB_HCA=mlx5 \
  $HEAD_RAY start --head --node-ip-address 10.20.0.30 --port 6379 --num-gpus 4 >/dev/null 2>&1
sleep 8
ssh $W "$WORKER_RAY stop --force >/dev/null 2>&1; sleep 3; \
  export CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_HOST_IP=10.20.0.28 NCCL_SOCKET_IFNAME=ibp34s0 NCCL_IB_HCA=mlx5; \
  $WORKER_RAY start --address=10.20.0.30:6379 --node-ip-address=10.20.0.28 --num-gpus=4 >/dev/null 2>&1"
sleep 8
echo "### ray restored: $($HEAD_RAY status 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ GPU' | head -1) ###"
echo "### AR MEASURE DONE ###"
