#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"
EVALUATION_ROOT="${GAMEAGENT_EVALUATION_ROOT:-/workspace/gameagent/evaluations}"
TRAINABLE_LSTM="${GAMEAGENT_RL_TRAINABLE_LSTM:-1}"
TRAINABLE_RESNET="${GAMEAGENT_RL_TRAINABLE_RESNET:-1}"
if [[ "${TRAINABLE_LSTM}" == "1" ]]; then
  SOURCE_CHECKPOINT="${GAMEAGENT_EVALUATION_SOURCE_CHECKPOINT:-/workspace/gameagent/rl_checkpoints/shared_recurrent_latest.pt}"
  SOURCE_METADATA="${GAMEAGENT_EVALUATION_SOURCE_METADATA:-/workspace/gameagent/rl_checkpoints/shared_recurrent_metadata_latest.garl}"
else
  SOURCE_CHECKPOINT="${GAMEAGENT_EVALUATION_SOURCE_CHECKPOINT:-/workspace/gameagent/rl_checkpoints/shared_policy_latest.garl}"
  SOURCE_METADATA=""
fi
RUN_LABEL="${GAMEAGENT_EVALUATION_NAME:-evaluation_$(date -u +%Y%m%dT%H%M%SZ)_p$$}"
RUN_DIRECTORY="${EVALUATION_ROOT}/${RUN_LABEL}"
SNAPSHOT_DIRECTORY="${RUN_DIRECTORY}/checkpoint_snapshot"
if [[ "${TRAINABLE_LSTM}" == "1" ]]; then
  SNAPSHOT_CHECKPOINT="${SNAPSHOT_DIRECTORY}/shared_recurrent_latest.pt"
  SNAPSHOT_METADATA="${SNAPSHOT_DIRECTORY}/shared_recurrent_metadata_latest.garl"
else
  SNAPSHOT_CHECKPOINT="${SNAPSHOT_DIRECTORY}/shared_policy_latest.garl"
  SNAPSHOT_METADATA=""
fi

if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
  echo "Shared RL checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -x "${GAMEAGENT_ROOT}/buildrelease/GameAgent" ]]; then
  echo "GameAgent executable not found." >&2
  echo "Run: bash ${GAMEAGENT_ROOT}/scripts/build_runpod.sh" >&2
  exit 1
fi

mkdir -p "${SNAPSHOT_DIRECTORY}"
cp "${SOURCE_CHECKPOINT}" "${SNAPSHOT_CHECKPOINT}"
SNAPSHOT_FILES=("${SNAPSHOT_CHECKPOINT}")
if [[ -n "${SOURCE_METADATA}" && -f "${SOURCE_METADATA}" ]]; then
  cp "${SOURCE_METADATA}" "${SNAPSHOT_METADATA}"
  SNAPSHOT_FILES+=("${SNAPSHOT_METADATA}")
fi
sha256sum "${SOURCE_CHECKPOINT}" > "${RUN_DIRECTORY}/source_checkpoint.sha256"
sha256sum "${SNAPSHOT_FILES[@]}" > "${RUN_DIRECTORY}/snapshot_before.sha256"
if ! cmp -s "${SOURCE_CHECKPOINT}" "${SNAPSHOT_CHECKPOINT}"; then
  echo "Evaluation checkpoint snapshot verification failed." >&2
  exit 1
fi
ln -sfn "${RUN_DIRECTORY}" "${EVALUATION_ROOT}/latest"

export GAMEAGENT_TEMPORAL_MODEL=1
export GAMEAGENT_REINFORCEMENT_LEARNING=1
export GAMEAGENT_RL_TRAINABLE_LSTM="${TRAINABLE_LSTM}"
export GAMEAGENT_RL_TRAINABLE_RESNET="${TRAINABLE_RESNET}"
export GAMEAGENT_RL_TRAINING=0
export GAMEAGENT_RL_RESUME=1
export GAMEAGENT_RL_EPSILON_START=0
export GAMEAGENT_RL_EPSILON_FINAL=0
export GAMEAGENT_RL_SCRIPTED_EXPLORATION=0
export GAMEAGENT_RL_CHECKPOINT_DIR="${SNAPSHOT_DIRECTORY}"
export GAMEAGENT_METRICS_PATH="${RUN_DIRECTORY}/metrics.jsonl"
export GAMEAGENT_SAVE_FRAMES="${GAMEAGENT_SAVE_FRAMES:-0}"
export GAMEAGENT_COLLECT_TRAINING_DATA=0
export GAMEAGENT_TRAINING_RESET_STATE="${GAMEAGENT_TRAINING_RESET_STATE:-1}"
export GAMEAGENT_PORT="${GAMEAGENT_EVALUATION_PORT:-${GAMEAGENT_PORT:-8080}}"
export GAMEAGENT_DECISION_INTERVAL_MS="${GAMEAGENT_DECISION_INTERVAL_MS:-100}"
export GAMEAGENT_FAILSAFE_TIMEOUT_MS="${GAMEAGENT_FAILSAFE_TIMEOUT_MS:-750}"

finish_evaluation() {
  local exit_code=$?
  trap - EXIT
  sha256sum "${SNAPSHOT_FILES[@]}" > "${RUN_DIRECTORY}/snapshot_after.sha256"
  if cmp -s \
      "${RUN_DIRECTORY}/snapshot_before.sha256" \
      "${RUN_DIRECTORY}/snapshot_after.sha256"; then
    echo "[EVALUATION] checkpoint snapshot remained byte-identical"
  else
    echo "[EVALUATION][ERROR] checkpoint snapshot changed" >&2
    exit_code=1
  fi
  echo "[EVALUATION] artifacts: ${RUN_DIRECTORY}"
  exit "${exit_code}"
}
trap finish_evaluation EXIT

server_pid=""
signal_received=0
stop_evaluation() {
  signal_received=1
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM "${server_pid}" 2>/dev/null || true
  fi
}
trap stop_evaluation INT TERM

exec > >(tee -a "${RUN_DIRECTORY}/server.log") 2>&1

echo "[EVALUATION] pure recurrent DQN=${TRAINABLE_LSTM}, trainable ResNet checkpoint=${TRAINABLE_RESNET}, epsilon=0, training=off, checkpoint writes=off"
echo "[EVALUATION] source checkpoint: ${SOURCE_CHECKPOINT}"
echo "[EVALUATION] isolated snapshot: ${SNAPSHOT_CHECKPOINT}"
echo "[EVALUATION] metrics: ${GAMEAGENT_METRICS_PATH}"
echo "[EVALUATION] port: ${GAMEAGENT_PORT}"

bash "${GAMEAGENT_ROOT}/scripts/run_runpod_server.sh" &
server_pid=$!
set +e
wait "${server_pid}"
server_status=$?
if kill -0 "${server_pid}" 2>/dev/null; then
  wait "${server_pid}"
  server_status=$?
fi
set -e
server_pid=""
if [[ "${signal_received}" == "1" ]]; then
  exit 130
fi
exit "${server_status}"
