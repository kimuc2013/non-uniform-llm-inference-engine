#!/bin/bash
# Fully autonomous hetero-cluster sweep runner.
#   1) PP verify (1 cell) — must beat $PP_VERIFY_FLOOR_TPS to proceed
#   2) Full ours sweep (70B 27 + 8B 27)
#   3) Per-cell auto-retry: cleanup → cluster restart between attempts
#
# All site-specific info comes from cluster.local.env (gitignored).

set -u
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO="$( dirname "$SCRIPT_DIR" )"
if [ ! -f "$REPO/cluster.local.env" ]; then
    echo "[fatal] $REPO/cluster.local.env not found. Copy cluster.example.env and fill it in."
    exit 1
fi
source "$REPO/cluster.local.env"
PY="$HEAD_PY"
PP_VERIFY_FLOOR_TPS="${PP_VERIFY_FLOOR_TPS:-1200}"
LOG="${LOG:-/tmp/auto_sweep.log}"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"

ts() { date +%H:%M:%S; }

cleanup_all() {
    echo "[$(ts)] cleanup: kill all"
    pkill -9 -f "VLLM::|api_server|RayWorkerProc\.run|performance.py" 2>/dev/null || true
    ssh -o ConnectTimeout=5 -p "$WORKER_SSH_PORT" "$WORKER_SSH_USER@$WORKER_SSH_HOST" \
        'pkill -9 -f "VLLM::|api_server|ray::RayWorkerProc"; true' 2>/dev/null || true
    "$PY" -c "
import ray
ray.init(address='$RAY_ADDRESS', logging_level='ERROR')
for k in list(ray.util.placement_group_table().keys()):
    try: ray._private.worker.global_worker.core_worker.remove_placement_group(ray.PlacementGroupID(bytes.fromhex(k)))
    except: pass
" 2>/dev/null || true
    sleep 10
}

verify_cluster() {
    "$PY" -c "
import ray; ray.init(address='$RAY_ADDRESS', logging_level='ERROR')
by = {n['NodeManagerAddress']: n.get('Resources',{}).get('GPU',0) for n in ray.nodes() if n.get('alive')}
ok = by.get('$HEAD_FABRIC_IP',0)==$HEAD_GPUS and by.get('$WORKER_FABRIC_IP',0)==$WORKER_GPUS
print('OK' if ok else f'FAIL {by}')
" 2>/dev/null | tail -1
}

restart_cluster() {
    echo "[$(ts)] restarting cluster"
    cleanup_all
    "$PY" "$REPO/planner/cluster_setup_4x4.py" --force 2>&1 | tail -3
    sleep 5
    # Direct ssh worker restart as fallback (cluster_setup also tries this)
    ssh -o ConnectTimeout=5 -p "$WORKER_SSH_PORT" "$WORKER_SSH_USER@$WORKER_SSH_HOST" "
        $WORKER_RAY stop --force 2>&1 | tail -2
        sleep 5
        export CUDA_VISIBLE_DEVICES=$WORKER_CUDA_VISIBLE_DEVICES
        export VLLM_HOST_IP=$WORKER_FABRIC_IP
        export NCCL_SOCKET_IFNAME=$WORKER_FABRIC_IFACE
        export NCCL_IB_HCA=$NCCL_IB_HCA
        $WORKER_RAY start --address=$RAY_ADDRESS --node-ip-address=$WORKER_FABRIC_IP --num-gpus=$WORKER_GPUS 2>&1 | tail -3
    " 2>&1 | tail -5
    sleep 15
    echo "[$(ts)] cluster state: $(verify_cluster)"
}

# Find latest record.json for a given cell pattern; print "tps ok"
last_cell_result() {
    pattern="$1"
    rec=$(ls -t "$REPO"/results/hetero_4x4_*_full_*/"$pattern"/record.json 2>/dev/null | head -1)
    if [ -z "$rec" ]; then echo "0 false"; return; fi
    "$PY" -c "import json; r=json.load(open('$rec')); print(f\"{r.get('tps',0):.0f} {str(r.get('success',False)).lower()}\")"
}

run_cell() {
    script="$1"; cfg="$2"; wl="$3"
    echo "[$(ts)] RUN $script cfg=$cfg wl=$wl"
    "$PY" "$REPO/planner/$script" --configs "$cfg" --workloads "$wl" 2>&1 | tail -3
    if [[ "$script" == *"70b"* ]]; then prefix="70b"; else prefix="8b"; fi
    result=$(last_cell_result "${prefix}_${cfg}_${wl}")
    tps=$(echo "$result" | awk '{print $1}')
    ok=$(echo "$result" | awk '{print $2}')
    echo "[$(ts)] RESULT cfg=$cfg wl=$wl tps=$tps ok=$ok"
    if [ "$ok" == "true" ] && (( $(echo "$tps > 50" | bc -l) )); then return 0; fi
    return 1
}

run_cell_with_retry() {
    script="$1"; cfg="$2"; wl="$3"
    for attempt in 1 2 3; do
        if run_cell "$script" "$cfg" "$wl"; then return 0; fi
        echo "[$(ts)] cell FAILED (attempt $attempt) — recover"
        if [ $attempt -eq 1 ]; then cleanup_all
        elif [ $attempt -eq 2 ]; then restart_cluster
        fi
    done
    echo "[$(ts)] cell GIVING UP after 3 attempts: $cfg $wl"
    return 1
}

# === 1) PP verify ===
echo "[$(ts)] === STAGE 1: PP verify (TP4PP2 [uniform] balanced 70B, must > $PP_VERIFY_FLOOR_TPS TPS) ==="
if ! verify_cluster | grep -q OK; then restart_cluster; fi

run_cell_with_retry hetero_4x4_70b_sweep.py TP4PP2_layer_uniform_40-40 balanced
result=$(last_cell_result "70b_TP4PP2_layer_uniform_40-40_balanced")
tps=$(echo "$result" | awk '{print $1}')
if (( $(echo "$tps < $PP_VERIFY_FLOOR_TPS" | bc -l) )); then
    echo "[$(ts)] PP verify FAILED ($tps TPS < $PP_VERIFY_FLOOR_TPS) — STOPPING"
    exit 1
fi
echo "[$(ts)] === PP verify PASSED ($tps TPS) ==="

# === 2) Full 70B sweep ===
echo "[$(ts)] === STAGE 2: full 70B sweep ==="
CONFIGS_70B="TP8PP1_uniform TP4PP2_layer_uniform_40-40 TP4PP2_layer_skew+4_44-36 TP4PP2_layer_skew+8_48-32 TP4PP2_layer_skew+12_52-28 TP4PP2_layer_skew+16_56-24 TP2PP4_layer_uniform_20-20-20-20 TP2PP4_layer_blackbias_22-22-18-18 TP2PP4_layer_blackbias_24-24-16-16"
WORKLOADS="balanced decode_heavy prefill_heavy"
for cfg in $CONFIGS_70B; do
    for wl in $WORKLOADS; do
        if [ "$cfg" == "TP4PP2_layer_uniform_40-40" ] && [ "$wl" == "balanced" ]; then continue; fi
        run_cell_with_retry hetero_4x4_70b_sweep.py "$cfg" "$wl" || true
    done
done

# === 3) Full 8B sweep ===
echo "[$(ts)] === STAGE 3: full 8B sweep ==="
CONFIGS_8B="TP8PP1_uniform TP4PP2_layer_uniform_16-16 TP4PP2_layer_skew+2_18-14 TP4PP2_layer_skew+4_20-12 TP4PP2_layer_skew+6_22-10 TP4PP2_layer_skew+8_24-8 TP2PP4_layer_uniform_8-8-8-8 TP2PP4_layer_blackbias_9-9-7-7 TP2PP4_layer_blackbias_10-10-6-6"
for cfg in $CONFIGS_8B; do
    for wl in $WORKLOADS; do
        run_cell_with_retry hetero_4x4_8b_sweep.py "$cfg" "$wl" || true
    done
done

echo "[$(ts)] === ALL DONE ==="
echo "70B records: $(ls $REPO/results/hetero_4x4_70b_full_*/*/record.json 2>/dev/null | wc -l)"
echo "8B  records: $(ls $REPO/results/hetero_4x4_8b_full_*/*/record.json 2>/dev/null | wc -l)"
