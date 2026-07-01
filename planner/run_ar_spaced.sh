#!/bin/bash
# Pre-serving AR-vs-compute overlap probe: measures the EXPOSED AR bandwidth when each
# cross-node all-reduce is separated by per-layer work (gemm=compute / memread=memory),
# vs back-to-back (none). Ratio bw(gemm)/bw(none) = the in-decode overlap boost, the
# physically-correct source for the effective AR term. Frees 6379, restores ray 4+4.
set -u
HEAD_RAY=/data/esca/uckim/miniconda3/envs/vllm_main/bin/ray
WORKER_RAY=/data/esca/uckim/miniconda3/envs/vllm_new/bin/ray
W=esca@10.20.0.28
$HEAD_RAY stop --force >/dev/null 2>&1; ssh $W "$WORKER_RAY stop --force >/dev/null 2>&1"; sleep 8
echo "### 6379 free? $(ssh $W 'timeout 3 bash -c "echo>/dev/tcp/10.20.0.30/6379" 2>/dev/null && echo NO || echo yes')"
for SMS in 0.10 0.25 0.50; do
  echo "### SPACER_MS=$SMS ###"
  AR_PORT=6379 AR_SCRIPT=ar_spaced_microbench.py AR_CUDA_GRAPH=0 SPACER_MS=$SMS NCCL_DEBUG=WARN \
    timeout 220 bash /data/esca/uckim/vllm_main/planner/run_ar_bench.sh 4 SPACER_MS=$SMS 2>&1 \
    | grep -E "^#|^  (none|gemm|memread)|NCCL WARN|Error" | grep -vE "Setting|OMP"
  pkill -u esca -f "[a]r_spaced"; ssh $W 'pkill -u esca -f "[a]r_spaced"'; sleep 4
done
echo "### restore ray 4+4 ###"
VLLM_HOST_IP=10.20.0.30 NCCL_SOCKET_IFNAME=ibp3s0 NCCL_IB_HCA=mlx5 \
  $HEAD_RAY start --head --node-ip-address 10.20.0.30 --port 6379 --num-gpus 4 >/dev/null 2>&1; sleep 8
ssh $W "$WORKER_RAY stop --force >/dev/null 2>&1; sleep 3; export CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_HOST_IP=10.20.0.28 NCCL_SOCKET_IFNAME=ibp34s0 NCCL_IB_HCA=mlx5; $WORKER_RAY start --address=10.20.0.30:6379 --node-ip-address=10.20.0.28 --num-gpus=4 >/dev/null 2>&1"; sleep 8
echo "### ray restored: $($HEAD_RAY status 2>/dev/null | grep -oE '[0-9.]+/[0-9.]+ GPU' | head -1) ###"
echo "### AR SPACED DONE ###"
