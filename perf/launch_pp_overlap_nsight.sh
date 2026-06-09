#!/usr/bin/env bash
# Launch vLLM with TP=2 PP=2 (mp executor, single node) under nsys and run
# a small benchmark. Used to validate send-side ring buffer PP overlap patch.
#
# Knobs (env):
#   PORT, MODEL, MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS
#   VLLM_PP_OVERLAP, VLLM_PP_SEND_RING_SIZE, VLLM_PP_MICROBATCH(_SIZE)
#   VLLM_PP_BATCH_QUEUE_SIZE
#   NSIGHT_LABEL  -> output basename suffix
#   PERF_REQUESTS, PERF_MAX_TOKENS, PERF_RUNS
#
# Usage:
#   VLLM_PP_OVERLAP=1 VLLM_PP_SEND_RING_SIZE=2 \
#       NSIGHT_LABEL=ring2 ./launch_pp_overlap_nsight.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
NSYS_BIN="${NSYS_BIN:-/usr/local/cuda-12.9/bin/nsys}"
[[ -x "${NSYS_BIN}" ]] || NSYS_BIN="$(command -v nsys)"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${PORT:-28100}"
HOST_FOR_CLIENT="${HOST_FOR_CLIENT:-127.0.0.1}"
BASE_URL="http://${HOST_FOR_CLIENT}:${PORT}/v1"
HEALTH_URL="http://${HOST_FOR_CLIENT}:${PORT}/health"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
DTYPE="${DTYPE:-bfloat16}"

VLLM_PP_OVERLAP="${VLLM_PP_OVERLAP:-1}"
VLLM_PP_SEND_RING_SIZE="${VLLM_PP_SEND_RING_SIZE:-0}"
VLLM_PP_MICROBATCH="${VLLM_PP_MICROBATCH:-1}"
VLLM_PP_MICROBATCH_SIZE="${VLLM_PP_MICROBATCH_SIZE:-16}"
VLLM_PP_BATCH_QUEUE_SIZE="${VLLM_PP_BATCH_QUEUE_SIZE:-2}"
VLLM_PP_NVTX="${VLLM_PP_NVTX:-1}"
VLLM_PP_RECV_PREISSUE="${VLLM_PP_RECV_PREISSUE:-0}"
VLLM_PP_SKIP_METADATA="${VLLM_PP_SKIP_METADATA:-0}"
VLLM_PP_FAST_COMM="${VLLM_PP_FAST_COMM:-0}"
VLLM_PP_BG_IRECV="${VLLM_PP_BG_IRECV:-0}"
VLLM_PP_SHAPE_DEBUG="${VLLM_PP_SHAPE_DEBUG:-0}"
VLLM_ENGINE_NVTX="${VLLM_ENGINE_NVTX:-0}"
VLLM_PP_SKIP_PREP_SYNC="${VLLM_PP_SKIP_PREP_SYNC:-0}"
VLLM_PP_SKIP_ASYNC_COPY_SYNC="${VLLM_PP_SKIP_ASYNC_COPY_SYNC:-0}"
VLLM_ASYNC_OUT_NVTX="${VLLM_ASYNC_OUT_NVTX:-0}"
VLLM_USE_DISPATCHER="${VLLM_USE_DISPATCHER:-0}"
VLLM_DISPATCHER_NVTX="${VLLM_DISPATCHER_NVTX:-0}"
VLLM_PP_SAMPLED_BROADCAST_STREAM="${VLLM_PP_SAMPLED_BROADCAST_STREAM:-0}"
VLLM_PP_STAGE_NVTX="${VLLM_PP_STAGE_NVTX:-0}"
VLLM_PP_KEEP_INFLIGHT_IN_BATCH="${VLLM_PP_KEEP_INFLIGHT_IN_BATCH:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

PERF_REQUESTS="${PERF_REQUESTS:-16}"
PERF_MAX_TOKENS="${PERF_MAX_TOKENS:-64}"
PERF_RUNS="${PERF_RUNS:-1}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
WARMUP_MAX_TOKENS="${WARMUP_MAX_TOKENS:-16}"

STAMP="$(date +%Y%m%d_%H%M%S)"
NSIGHT_LABEL="${NSIGHT_LABEL:-tp2pp2_mp}"
NSIGHT_OUT_DIR="${SCRIPT_DIR}/../logs/nsight"
mkdir -p "${NSIGHT_OUT_DIR}"
NSIGHT_OUT_BASE="${NSIGHT_OUT_DIR}/pp_overlap_${NSIGHT_LABEL}_${STAMP}"
SESSION_NAME="vllm_overlap_${STAMP}_$$"

VLLM_LOG="${NSIGHT_OUT_DIR}/${NSIGHT_LABEL}_${STAMP}_vllm.log"

# ---- pre-flight ----
if pgrep -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" >/dev/null; then
  echo "[ERROR] vllm already running on port ${PORT}" >&2
  exit 1
fi

# ---- env for vllm process ----
# CC/CXX -> gcc-12 because nvcc 12.9 rejects gcc-13/14.
# flashinfer JIT compiles sampling kernels on first launch; use a supported host.
ENV_VARS=(
  "VLLM_PP_OVERLAP=${VLLM_PP_OVERLAP}"
  "VLLM_PP_SEND_RING_SIZE=${VLLM_PP_SEND_RING_SIZE}"
  "VLLM_PP_MICROBATCH=${VLLM_PP_MICROBATCH}"
  "VLLM_PP_MICROBATCH_SIZE=${VLLM_PP_MICROBATCH_SIZE}"
  "VLLM_PP_BATCH_QUEUE_SIZE=${VLLM_PP_BATCH_QUEUE_SIZE}"
  "VLLM_PP_NVTX=${VLLM_PP_NVTX}"
  "VLLM_PP_RECV_PREISSUE=${VLLM_PP_RECV_PREISSUE}"
  "VLLM_PP_SKIP_METADATA=${VLLM_PP_SKIP_METADATA}"
  "VLLM_PP_FAST_COMM=${VLLM_PP_FAST_COMM}"
  "VLLM_PP_BG_IRECV=${VLLM_PP_BG_IRECV}"
  "VLLM_PP_SHAPE_DEBUG=${VLLM_PP_SHAPE_DEBUG}"
  "VLLM_ENGINE_NVTX=${VLLM_ENGINE_NVTX}"
  "VLLM_PP_SKIP_PREP_SYNC=${VLLM_PP_SKIP_PREP_SYNC}"
  "VLLM_PP_SKIP_ASYNC_COPY_SYNC=${VLLM_PP_SKIP_ASYNC_COPY_SYNC}"
  "VLLM_ASYNC_OUT_NVTX=${VLLM_ASYNC_OUT_NVTX}"
  "VLLM_USE_DISPATCHER=${VLLM_USE_DISPATCHER}"
  "VLLM_DISPATCHER_NVTX=${VLLM_DISPATCHER_NVTX}"
  "VLLM_PP_SAMPLED_BROADCAST_STREAM=${VLLM_PP_SAMPLED_BROADCAST_STREAM}"
  "VLLM_PP_STAGE_NVTX=${VLLM_PP_STAGE_NVTX}"
  "VLLM_PP_KEEP_INFLIGHT_IN_BATCH=${VLLM_PP_KEEP_INFLIGHT_IN_BATCH}"
  "VLLM_USE_V2_MODEL_RUNNER=0"
  "VLLM_LOGGING_LEVEL=INFO"
  "CC=/usr/bin/gcc-12"
  "CXX=/usr/bin/g++-12"
  "CUDAHOSTCXX=/usr/bin/g++-12"
  "NVCC_CCBIN=/usr/bin/g++-12"
  # Force CUDA 12.9 headers (system /usr/include has stale CUDA 12.0 headers
  # that conflict with nvcc 12.9 / flashinfer libcudacxx).
  "CUDA_HOME=/usr/local/cuda-12.9"
  "CUDA_PATH=/usr/local/cuda-12.9"
  "CPATH=/usr/local/cuda-12.9/include"
  "CPLUS_INCLUDE_PATH=/usr/local/cuda-12.9/include"
  "C_INCLUDE_PATH=/usr/local/cuda-12.9/include"
  "PATH=/usr/local/cuda-12.9/bin:/data/esca/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)

VLLM_ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model "${MODEL}"
  --tensor-parallel-size 2
  --pipeline-parallel-size 2
  --distributed-executor-backend mp
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --dtype "${DTYPE}"
  --port "${PORT}"
  --host 0.0.0.0
  --async-scheduling
  --enable-chunked-prefill
)
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  VLLM_ARGS+=(--enforce-eager)
fi

NSYS_CMD=(
  "${NSYS_BIN}" profile
  -t cuda,nvtx,osrt
  --sample=none
  --cpuctxsw=none
  --cuda-graph-trace=graph
  --trace-fork-before-exec=true
  --force-overwrite=true
  --stop-on-exit=false
  --kill=none
  --wait=all
  --duration=300
  --start-later=true
  --session-new="${SESSION_NAME}"
  --export=sqlite
  -o "${NSIGHT_OUT_BASE}"
)

echo "[INFO] launching vllm under nsys: ${NSIGHT_OUT_BASE}.nsys-rep"
echo "[INFO] vllm log:                  ${VLLM_LOG}"
echo "[INFO] env:                       ${ENV_VARS[*]}"

set -x
env "${ENV_VARS[@]}" "${NSYS_CMD[@]}" \
    "${PYTHON_BIN}" "${VLLM_ARGS[@]}" \
    >"${VLLM_LOG}" 2>&1 &
PROFILE_PID=$!
set +x

cleanup() {
  echo "[INFO] cleanup: stopping vllm + nsys"
  "${NSYS_BIN}" stop --session="${SESSION_NAME}" 2>/dev/null || true
  # Kill any vllm python under our nsys process (vllm api_server + workers).
  pkill -INT -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" 2>/dev/null || true
  pkill -INT -f "EngineCore_DP0\|Worker_PP" 2>/dev/null || true
  sleep 3
  pkill -KILL -f "vllm.entrypoints.openai.api_server.*--port ${PORT}" 2>/dev/null || true
  pkill -KILL -f "EngineCore_DP0\|Worker_PP" 2>/dev/null || true
  if kill -0 "${PROFILE_PID}" 2>/dev/null; then
    kill -INT "${PROFILE_PID}" 2>/dev/null || true
    sleep 5
    kill -KILL "${PROFILE_PID}" 2>/dev/null || true
  fi
  wait "${PROFILE_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---- wait for /health ----
echo "[INFO] waiting for health on ${HEALTH_URL} (timeout 240s)"
deadline=$((SECONDS + 240))
until curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; do
  if ! kill -0 "${PROFILE_PID}" 2>/dev/null; then
    echo "[ERROR] nsys/vllm parent died before /health responded; see ${VLLM_LOG}" >&2
    tail -40 "${VLLM_LOG}" >&2 || true
    exit 1
  fi
  # nsys may stay alive even if python child died -- detect that case.
  if grep -q "Engine core initialization failed\|RuntimeError\|FAILED:" "${VLLM_LOG}" 2>/dev/null; then
    echo "[ERROR] vllm python child failed (engine init); see ${VLLM_LOG}" >&2
    tail -40 "${VLLM_LOG}" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "[ERROR] timeout waiting for /health" >&2
    exit 1
  fi
  sleep 2
done
echo "[INFO] health ok"

# ---- warmup ----
echo "[INFO] warmup: requests=${WARMUP_REQUESTS} max_tokens=${WARMUP_MAX_TOKENS}"
VLLM_BASE_URL="${BASE_URL}" VLLM_MODEL="${MODEL}" MODEL="${MODEL}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/performance.py" \
    --model "${MODEL}" \
    --max-tokens "${WARMUP_MAX_TOKENS}" \
    --requests "${WARMUP_REQUESTS}" \
    --runs 1 \
    --ignore-eos 2>&1 | tail -5 || {
      echo "[WARN] warmup failed; continuing" >&2
    }

# ---- start capture ----
echo "[INFO] start nsys capture"
"${NSYS_BIN}" start --session="${SESSION_NAME}"

# ---- workload ----
echo "[INFO] perf: requests=${PERF_REQUESTS} max_tokens=${PERF_MAX_TOKENS} runs=${PERF_RUNS}"
VLLM_BASE_URL="${BASE_URL}" VLLM_MODEL="${MODEL}" MODEL="${MODEL}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/performance.py" \
    --model "${MODEL}" \
    --max-tokens "${PERF_MAX_TOKENS}" \
    --min-tokens "${PERF_MAX_TOKENS}" \
    --requests "${PERF_REQUESTS}" \
    --runs "${PERF_RUNS}" \
    --ignore-eos

# ---- stop capture ----
echo "[INFO] stop nsys capture"
"${NSYS_BIN}" stop --session="${SESSION_NAME}"

# wait briefly for sqlite export
sleep 5

echo "[INFO] outputs:"
ls -la "${NSIGHT_OUT_BASE}".* 2>/dev/null || true
echo "[INFO] vllm log: ${VLLM_LOG}"
