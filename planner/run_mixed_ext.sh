#!/bin/bash
# Extended 4+4 mixed-traffic sweep: the TP/PP combos not yet in mixed (pure PP8 +
# deeper-PP skew). Run when all 8 GPUs are free. Logs to /scfs/esca.
cd /data/esca/uckim/vllm_main
CAMPAIGN_WORKLOADS=mixed \
CAMPAIGN_CONFIGS=TP1PP8_uniform,TP2PP4_skew+8,TP2PP4_skew+12 \
CAMPAIGN_NREQ=64,96 \
  bash planner/run_campaign.sh 4 4 8b opt30b 70b
