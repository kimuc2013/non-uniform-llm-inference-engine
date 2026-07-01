#!/bin/bash
# Single-node intra AR (nnodes=1, nproc=4): measures intra-node AllReduce bw+latency on
# head (Blackwell/NVLink) and worker (Ada/PCIe). No cross-node -> no rendezvous/ray dance.
set -u
HP=/data/esca/uckim/miniconda3/envs/vllm_main/bin/python
WP=/data/esca/uckim/miniconda3/envs/vllm_new/bin/python
echo "### HEAD (Blackwell intra) ###"
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_LEVEL=NVL AR_CUDA_GRAPH=1 \
  $HP -m torch.distributed.run --standalone --nproc_per_node=4 /data/esca/uckim/vllm_main/planner/ar_microbench.py 2>&1 \
  | grep -E "^ +[0-9]" | grep -vE "WARNING|OMP"
echo "### WORKER (Ada intra) ###"
scp -q planner/ar_microbench.py esca@10.20.0.28:/tmp/ar_microbench.py
ssh esca@10.20.0.28 "CUDA_VISIBLE_DEVICES=0,1,2,3 AR_CUDA_GRAPH=1 $WP -m torch.distributed.run --standalone --nproc_per_node=4 /tmp/ar_microbench.py 2>&1 | grep -E '^ +[0-9]' | grep -vE 'WARNING|OMP'"
