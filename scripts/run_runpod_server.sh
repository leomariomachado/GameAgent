#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"
GAMEAGENT_BUILD_DIR="${GAMEAGENT_BUILD_DIR:-${GAMEAGENT_ROOT}/buildrelease}"
GAMEAGENT_BINARY="${GAMEAGENT_BINARY:-${GAMEAGENT_BUILD_DIR}/GameAgent}"

export GAMEAGENT_HOST="${GAMEAGENT_HOST:-0.0.0.0}"
export GAMEAGENT_PORT="${GAMEAGENT_PORT:-8080}"
export GAMEAGENT_CAPTURE_DIR="${GAMEAGENT_CAPTURE_DIR:-/workspace/gameagent/captures}"
export GAMEAGENT_METRICS_PATH="${GAMEAGENT_METRICS_PATH:-/workspace/gameagent/metrics.jsonl}"
export GAMEAGENT_FRAME_BUFFER_CAPACITY="${GAMEAGENT_FRAME_BUFFER_CAPACITY:-32}"
export GAMEAGENT_OBSERVATION_FRAMES="${GAMEAGENT_OBSERVATION_FRAMES:-8}"
export GAMEAGENT_ACTION_HISTORY_SIZE="${GAMEAGENT_ACTION_HISTORY_SIZE:-8}"
export GAMEAGENT_DECISION_INTERVAL_MS="${GAMEAGENT_DECISION_INTERVAL_MS:-50}"
export GAMEAGENT_FAILSAFE_TIMEOUT_MS="${GAMEAGENT_FAILSAFE_TIMEOUT_MS:-200}"
export GAMEAGENT_MAX_JPEG_MB="${GAMEAGENT_MAX_JPEG_MB:-8}"
export GAMEAGENT_SAVE_FRAMES="${GAMEAGENT_SAVE_FRAMES:-0}"
export GAMEAGENT_SAVE_EVERY_N_FRAMES="${GAMEAGENT_SAVE_EVERY_N_FRAMES:-1}"
export GAMEAGENT_MAX_SAVED_FRAMES="${GAMEAGENT_MAX_SAVED_FRAMES:-300}"

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
mkdir -p "$(dirname "${GAMEAGENT_METRICS_PATH}")"

echo "Starting GameAgent on ${GAMEAGENT_HOST}:${GAMEAGENT_PORT}"
echo "WebSocket: ws://${GAMEAGENT_HOST}:${GAMEAGENT_PORT}/ws"
echo "Metrics: ${GAMEAGENT_METRICS_PATH}"
echo "Circular buffer: ${GAMEAGENT_FRAME_BUFFER_CAPACITY} frames"
echo "Observation: ${GAMEAGENT_OBSERVATION_FRAMES} frames"
echo "Confirmed action history: ${GAMEAGENT_ACTION_HISTORY_SIZE} states"
echo "Decision interval: ${GAMEAGENT_DECISION_INTERVAL_MS} ms"
echo "Action protocol: v2/state; failsafe ${GAMEAGENT_FAILSAFE_TIMEOUT_MS} ms"
if [[ "${GAMEAGENT_SAVE_FRAMES}" == "1" ]]; then
  echo "Frame capture: every ${GAMEAGENT_SAVE_EVERY_N_FRAMES} frame(s), maximum ${GAMEAGENT_MAX_SAVED_FRAMES} per connection"
else
  echo "Frame capture: disabled (in-memory path only)"
fi
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
  echo "Public health URL: https://${RUNPOD_POD_ID}-${GAMEAGENT_PORT}.proxy.runpod.net/health"
  echo "Public WebSocket URL: wss://${RUNPOD_POD_ID}-${GAMEAGENT_PORT}.proxy.runpod.net/ws"
fi

cd "${GAMEAGENT_ROOT}"
exec "${GAMEAGENT_BINARY}"
