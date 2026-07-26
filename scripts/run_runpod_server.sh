#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"
GAMEAGENT_BUILD_DIR="${GAMEAGENT_BUILD_DIR:-${GAMEAGENT_ROOT}/buildrelease}"
GAMEAGENT_BINARY="${GAMEAGENT_BINARY:-${GAMEAGENT_BUILD_DIR}/GameAgent}"

read_pid1_environment_variable() {
  local variable_name="$1"
  local entry
  [[ -r /proc/1/environ ]] || return 1
  while IFS= read -r -d '' entry; do
    if [[ "${entry}" == "${variable_name}="* ]]; then
      printf '%s\n' "${entry#*=}"
      return 0
    fi
  done < /proc/1/environ
  return 1
}

export GAMEAGENT_HOST="${GAMEAGENT_HOST:-0.0.0.0}"
export GAMEAGENT_PORT="${GAMEAGENT_PORT:-8080}"
export GAMEAGENT_CAPTURE_DIR="${GAMEAGENT_CAPTURE_DIR:-/workspace/gameagent/captures}"
export GAMEAGENT_METRICS_PATH="${GAMEAGENT_METRICS_PATH:-/workspace/gameagent/metrics.jsonl}"
export GAMEAGENT_FRAME_BUFFER_CAPACITY="${GAMEAGENT_FRAME_BUFFER_CAPACITY:-32}"
export GAMEAGENT_OBSERVATION_FRAMES="${GAMEAGENT_OBSERVATION_FRAMES:-8}"
export GAMEAGENT_ACTION_HISTORY_SIZE="${GAMEAGENT_ACTION_HISTORY_SIZE:-8}"
export GAMEAGENT_PERSISTENCE_QUEUE_CAPACITY="${GAMEAGENT_PERSISTENCE_QUEUE_CAPACITY:-256}"
export GAMEAGENT_DECISION_INTERVAL_MS="${GAMEAGENT_DECISION_INTERVAL_MS:-50}"
export GAMEAGENT_FAILSAFE_TIMEOUT_MS="${GAMEAGENT_FAILSAFE_TIMEOUT_MS:-200}"
export GAMEAGENT_MAX_JPEG_MB="${GAMEAGENT_MAX_JPEG_MB:-8}"
export GAMEAGENT_SAVE_FRAMES="${GAMEAGENT_SAVE_FRAMES:-0}"
export GAMEAGENT_COLLECT_TRAINING_DATA="${GAMEAGENT_COLLECT_TRAINING_DATA:-0}"
export GAMEAGENT_TRAINING_CAPTURE_DIR="${GAMEAGENT_TRAINING_CAPTURE_DIR:-/workspace/gameagent/training_captures}"
export GAMEAGENT_SUBZERO_FACING="${GAMEAGENT_SUBZERO_FACING:-right}"
export GAMEAGENT_TEMPORAL_MODEL="${GAMEAGENT_TEMPORAL_MODEL:-0}"
export GAMEAGENT_RESNET50_CHECKPOINT="${GAMEAGENT_RESNET50_CHECKPOINT:-/workspace/resnet50_best.ckpt}"
export GAMEAGENT_TEMPORAL_CHECKPOINT="${GAMEAGENT_TEMPORAL_CHECKPOINT:-}"
export GAMEAGENT_DEVICE_ID="${GAMEAGENT_DEVICE_ID:-0}"
export GAMEAGENT_LSTM_PROJECTION_SIZE="${GAMEAGENT_LSTM_PROJECTION_SIZE:-256}"
export GAMEAGENT_LSTM_HIDDEN_SIZE="${GAMEAGENT_LSTM_HIDDEN_SIZE:-256}"

PUBLIC_POD_ID="${GAMEAGENT_PUBLIC_POD_ID:-${RUNPOD_POD_ID:-}}"
if [[ -z "${PUBLIC_POD_ID}" ]]; then
  PUBLIC_POD_ID="$(
    read_pid1_environment_variable RUNPOD_POD_ID || true
  )"
fi
if [[ -n "${PUBLIC_POD_ID}" &&
      ! "${PUBLIC_POD_ID}" =~ ^[[:alnum:]-]+$ ]]; then
  echo "Invalid public RunPod pod ID: ${PUBLIC_POD_ID}" >&2
  exit 1
fi

PUBLIC_WEBSOCKET_URL="${GAMEAGENT_PUBLIC_WEBSOCKET_URL:-}"
PUBLIC_HEALTH_URL="${GAMEAGENT_PUBLIC_HEALTH_URL:-}"
if [[ -n "${PUBLIC_POD_ID}" ]]; then
  PUBLIC_WEBSOCKET_URL="${PUBLIC_WEBSOCKET_URL:-wss://${PUBLIC_POD_ID}-${GAMEAGENT_PORT}.proxy.runpod.net/ws}"
  PUBLIC_HEALTH_URL="${PUBLIC_HEALTH_URL:-https://${PUBLIC_POD_ID}-${GAMEAGENT_PORT}.proxy.runpod.net/health}"
fi

if [[ ! -x "${GAMEAGENT_BINARY}" ]]; then
  echo "GameAgent executable not found: ${GAMEAGENT_BINARY}" >&2
  echo "Run: bash ${GAMEAGENT_ROOT}/scripts/build_runpod.sh" >&2
  exit 1
fi

if [[ -z "${GAMEAGENT_API_TOKEN:-}" ]]; then
  if [[ "${GAMEAGENT_ALLOW_UNAUTHENTICATED:-0}" != "1" ]]; then
    echo "GAMEAGENT_API_TOKEN is required for a public RunPod server." >&2
    echo "Generate one with: openssl rand -hex 32" >&2
    echo "For temporary unsafe testing only, set GAMEAGENT_ALLOW_UNAUTHENTICATED=1." >&2
    exit 1
  fi
  echo "[WARNING] Starting a public server without authentication." >&2
else
  export GAMEAGENT_API_TOKEN
fi

mkdir -p "${GAMEAGENT_CAPTURE_DIR}"
if [[ "${GAMEAGENT_COLLECT_TRAINING_DATA}" == "1" ]]; then
  mkdir -p "${GAMEAGENT_TRAINING_CAPTURE_DIR}"
fi
mkdir -p "$(dirname "${GAMEAGENT_METRICS_PATH}")"

echo "Starting GameAgent on ${GAMEAGENT_HOST}:${GAMEAGENT_PORT}"
echo "Bind WebSocket (internal): ws://${GAMEAGENT_HOST}:${GAMEAGENT_PORT}/ws"
if [[ -n "${PUBLIC_WEBSOCKET_URL}" ]]; then
  echo "Public WebSocket URL (copy into client): ${PUBLIC_WEBSOCKET_URL}"
  echo "Public health URL: ${PUBLIC_HEALTH_URL}"
else
  echo "Public WebSocket URL: unavailable" >&2
  echo "Set GAMEAGENT_PUBLIC_POD_ID to the ID shown by RunPod." >&2
fi
echo "Metrics: ${GAMEAGENT_METRICS_PATH}"
echo "Circular buffer: ${GAMEAGENT_FRAME_BUFFER_CAPACITY} frames"
echo "Observation: ${GAMEAGENT_OBSERVATION_FRAMES} frames"
echo "Confirmed action history: ${GAMEAGENT_ACTION_HISTORY_SIZE} states"
echo "Async persistence queue: ${GAMEAGENT_PERSISTENCE_QUEUE_CAPACITY} frames"
echo "Decision interval: ${GAMEAGENT_DECISION_INTERVAL_MS} ms"
echo "Action protocol: v2/state; failsafe ${GAMEAGENT_FAILSAFE_TIMEOUT_MS} ms"
echo "Mock agent: Sub-Zero facing ${GAMEAGENT_SUBZERO_FACING}"
if [[ "${GAMEAGENT_TEMPORAL_MODEL}" == "1" ]]; then
  if [[ ! -f "${GAMEAGENT_RESNET50_CHECKPOINT}" ]]; then
    echo "ResNet50 checkpoint not found: ${GAMEAGENT_RESNET50_CHECKPOINT}" >&2
    exit 1
  fi
  echo "Temporal features: frozen ResNet50 + GPU LSTM"
  echo "ResNet50 checkpoint: ${GAMEAGENT_RESNET50_CHECKPOINT}"
  echo "Temporal shape: projection=${GAMEAGENT_LSTM_PROJECTION_SIZE}, hidden=${GAMEAGENT_LSTM_HIDDEN_SIZE}"
  echo "Temporal CUDA device: ${GAMEAGENT_DEVICE_ID}"
  if [[ -n "${GAMEAGENT_TEMPORAL_CHECKPOINT}" ]]; then
    if [[ ! -f "${GAMEAGENT_TEMPORAL_CHECKPOINT}" ]]; then
      echo "Temporal policy checkpoint not found: ${GAMEAGENT_TEMPORAL_CHECKPOINT}" >&2
      exit 1
    fi
    echo "Temporal policy: ${GAMEAGENT_TEMPORAL_CHECKPOINT} (student shadow)"
  else
    echo "Temporal policy: not loaded (teacher-only)"
  fi
else
  echo "Temporal features: disabled (set GAMEAGENT_TEMPORAL_MODEL=1 to enable)"
fi
if [[ "${GAMEAGENT_SAVE_FRAMES}" == "1" ]]; then
  echo "Frame capture: one debug JPEG per connection"
else
  echo "Frame capture: disabled (in-memory path only)"
fi
if [[ "${GAMEAGENT_COLLECT_TRAINING_DATA}" == "1" ]]; then
  echo "Training capture: async bounded queue -> ${GAMEAGENT_TRAINING_CAPTURE_DIR}"
else
  echo "Training capture: disabled"
fi

cd "${GAMEAGENT_ROOT}"
exec "${GAMEAGENT_BINARY}"
