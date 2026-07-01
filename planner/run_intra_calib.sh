#!/bin/bash
# Calibration helper: MEASURE intra-node AR on head + worker (single-node graph-chain,
# serving NCCL env, ray stopped). Prints "INTRA head <bw> <alpha>" / "INTRA worker ...".
set -u
HR=/data/esca/uckim/miniconda3/envs/vllm_main/bin/ray
WR=/data/esca/uckim/miniconda3/envs/vllm_new/bin/ray
HP=/data/esca/uckim/miniconda3/envs/vllm_main/bin/python
WP=/data/esca/uckim/miniconda3/envs/vllm_new/bin/python
$HR stop --force >/dev/null 2>&1; ssh esca@10.20.0.28 "$WR stop --force >/dev/null 2>&1"; sleep 6
run() { # ifname pyexec label host
  local IF=$1 PY=$2 L=$3 HOST=$4 SC=/data/esca/uckim/vllm_main/planner/ar_microbench.py
  local CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_IB_HCA=mlx5 NCCL_SOCKET_IFNAME=$IF NCCL_NET_GDR_LEVEL=2 AR_CUDA_GRAPH=1 timeout 120 $PY -m torch.distributed.run --standalone --nproc_per_node=4 $SC"
  local OUT
  if [ -n "$HOST" ]; then scp -q $SC $HOST:/tmp/ar_mb.py; OUT=$(ssh $HOST "${CMD/$SC//tmp/ar_mb.py}" 2>/dev/null); else OUT=$($CMD 2>/dev/null); fi
  # fit bw+alpha from the 1.049MB and 2.097MB rows: T=1.5*msg/bw + 6*alpha
  echo "$OUT" | grep -E "^ +(64|128)" | $HP -c "
import sys
r={}
for ln in sys.stdin:
    p=ln.split()
    if len(p)>=6: r[float(p[2])]=float(p[3])/1000.0  # msg_MB -> AR_ms
if 1.049 in r and 2.097 in r:
    t1,t2=r[1.049],r[2.097]; bw=1.5*(2.097-1.049)*1e6/((t2-t1)/1e3)/1e9; a=(t1-1.5*1.049e6/(bw*1e9)*1e3)/6*1e3
    print(f'INTRA $L {bw:.1f} {max(0.5,a):.1f}')
"
}
run ibp3s0 $HP head ""
run ibp34s0 $WP worker esca@10.20.0.28
# restore ray
VLLM_HOST_IP=10.20.0.30 NCCL_SOCKET_IFNAME=ibp3s0 NCCL_IB_HCA=mlx5 $HR start --head --node-ip-address 10.20.0.30 --port 6379 --num-gpus 4 >/dev/null 2>&1; sleep 8
ssh esca@10.20.0.28 "$WR stop --force >/dev/null 2>&1; sleep 3; export CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_HOST_IP=10.20.0.28 NCCL_SOCKET_IFNAME=ibp34s0 NCCL_IB_HCA=mlx5; $WR start --address=10.20.0.30:6379 --node-ip-address=10.20.0.28 --num-gpus=4 >/dev/null 2>&1"; sleep 8
echo "INTRA CALIB DONE"
