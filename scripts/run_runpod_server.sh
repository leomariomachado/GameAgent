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
export GAMEAGENT_DEBUG_CAPTURE_INTERVAL_MS="${GAMEAGENT_DEBUG_CAPTURE_INTERVAL_MS:-3000}"
export GAMEAGENT_COLLECT_TRAINING_DATA="${GAMEAGENT_COLLECT_TRAINING_DATA:-0}"
export GAMEAGENT_TRAINING_CAPTURE_DIR="${GAMEAGENT_TRAINING_CAPTURE_DIR:-/workspace/gameagent/training_captures}"
export GAMEAGENT_DEFAULT_FACING="${GAMEAGENT_DEFAULT_FACING:-right}"
export GAMEAGENT_TEMPORAL_MODEL="${GAMEAGENT_TEMPORAL_MODEL:-0}"
export GAMEAGENT_RESNET50_CHECKPOINT="${GAMEAGENT_RESNET50_CHECKPOINT:-/workspace/resnet50_best.ckpt}"
export GAMEAGENT_TEMPORAL_CHECKPOINT="${GAMEAGENT_TEMPORAL_CHECKPOINT:-}"
export GAMEAGENT_DEVICE_ID="${GAMEAGENT_DEVICE_ID:-0}"
export GAMEAGENT_LSTM_PROJECTION_SIZE="${GAMEAGENT_LSTM_PROJECTION_SIZE:-256}"
export GAMEAGENT_LSTM_HIDDEN_SIZE="${GAMEAGENT_LSTM_HIDDEN_SIZE:-256}"
export GAMEAGENT_REINFORCEMENT_LEARNING="${GAMEAGENT_REINFORCEMENT_LEARNING:-0}"
export GAMEAGENT_RL_TRAINING="${GAMEAGENT_RL_TRAINING:-1}"
export GAMEAGENT_RL_RESUME="${GAMEAGENT_RL_RESUME:-1}"
export GAMEAGENT_RL_TRAINABLE_LSTM="${GAMEAGENT_RL_TRAINABLE_LSTM:-1}"
export GAMEAGENT_RL_TRAINABLE_RESNET="${GAMEAGENT_RL_TRAINABLE_RESNET:-0}"
export GAMEAGENT_RL_RECURRENT_SERVICE="${GAMEAGENT_RL_RECURRENT_SERVICE:-${GAMEAGENT_ROOT}/scripts/recurrent_dqn_service.py}"
export GAMEAGENT_PLAYER_SIDE="${GAMEAGENT_PLAYER_SIDE:-left}"
export GAMEAGENT_RL_CHECKPOINT_DIR="${GAMEAGENT_RL_CHECKPOINT_DIR:-/workspace/gameagent/rl_checkpoints}"
export GAMEAGENT_RL_HIDDEN_SIZE="${GAMEAGENT_RL_HIDDEN_SIZE:-128}"
export GAMEAGENT_RL_REPLAY_CAPACITY="${GAMEAGENT_RL_REPLAY_CAPACITY:-20000}"
export GAMEAGENT_RL_MIN_REPLAY_SIZE="${GAMEAGENT_RL_MIN_REPLAY_SIZE:-256}"
export GAMEAGENT_RL_BATCH_SIZE="${GAMEAGENT_RL_BATCH_SIZE:-64}"
export GAMEAGENT_RL_TARGET_UPDATE="${GAMEAGENT_RL_TARGET_UPDATE:-500}"
export GAMEAGENT_RL_GAMMA="${GAMEAGENT_RL_GAMMA:-0.99}"
export GAMEAGENT_RL_LEARNING_RATE="${GAMEAGENT_RL_LEARNING_RATE:-0.0001}"
export GAMEAGENT_RL_RESNET_LEARNING_RATE="${GAMEAGENT_RL_RESNET_LEARNING_RATE:-0.000001}"
export GAMEAGENT_RL_RESNET_BATCH_SIZE="${GAMEAGENT_RL_RESNET_BATCH_SIZE:-4}"
export GAMEAGENT_RL_RESNET_UPDATE_INTERVAL="${GAMEAGENT_RL_RESNET_UPDATE_INTERVAL:-16}"
export GAMEAGENT_RL_RESNET_REPLAY_IMAGES="${GAMEAGENT_RL_RESNET_REPLAY_IMAGES:-4096}"
export GAMEAGENT_RL_EPSILON_START="${GAMEAGENT_RL_EPSILON_START:-1.0}"
export GAMEAGENT_RL_EPSILON_FINAL="${GAMEAGENT_RL_EPSILON_FINAL:-0.05}"
export GAMEAGENT_RL_EPSILON_DECAY="${GAMEAGENT_RL_EPSILON_DECAY:-100000}"
export GAMEAGENT_RL_SCRIPTED_EXPLORATION="${GAMEAGENT_RL_SCRIPTED_EXPLORATION:-0.50}"
export GAMEAGENT_RL_REWARD_LOG_INTERVAL_MS="${GAMEAGENT_RL_REWARD_LOG_INTERVAL_MS:-5000}"
export GAMEAGENT_TRAINING_RESET_STATE="${GAMEAGENT_TRAINING_RESET_STATE:-1}"

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
echo "Character profiles: subzero, scorpion, noob_saibot, sindel"
echo "Legacy/default profile: subzero facing ${GAMEAGENT_DEFAULT_FACING}"
if [[ "${GAMEAGENT_TEMPORAL_MODEL}" == "1" ]]; then
  if [[ ! -f "${GAMEAGENT_RESNET50_CHECKPOINT}" ]]; then
    echo "ResNet50 checkpoint not found: ${GAMEAGENT_RESNET50_CHECKPOINT}" >&2
    exit 1
  fi
  if [[ "${GAMEAGENT_REINFORCEMENT_LEARNING}" == "1" &&
        "${GAMEAGENT_RL_TRAINABLE_LSTM}" == "1" ]]; then
    if [[ "${GAMEAGENT_RL_TRAINABLE_RESNET}" == "1" ]]; then
      echo "Temporal features: trainable ResNet50 stage 4 + trainable GPU LSTM"
    else
      echo "Temporal features: frozen ResNet50 + trainable GPU LSTM"
    fi
  else
    echo "Temporal features: frozen ResNet50 + inference GPU LSTM"
  fi
  echo "ResNet50 checkpoint: ${GAMEAGENT_RESNET50_CHECKPOINT}"
  echo "Temporal shape: projection=${GAMEAGENT_LSTM_PROJECTION_SIZE}, hidden=${GAMEAGENT_LSTM_HIDDEN_SIZE}"
  echo "Temporal CUDA device: ${GAMEAGENT_DEVICE_ID}"
  if [[ -n "${GAMEAGENT_TEMPORAL_CHECKPOINT}" ]]; then
    if [[ ! -f "${GAMEAGENT_TEMPORAL_CHECKPOINT}" ]]; then
      echo "Temporal policy checkpoint not found: ${GAMEAGENT_TEMPORAL_CHECKPOINT}" >&2
      exit 1
    fi
    echo "Optional temporal policy: ${GAMEAGENT_TEMPORAL_CHECKPOINT}"
  else
    if [[ "${GAMEAGENT_REINFORCEMENT_LEARNING}" != "1" ||
          "${GAMEAGENT_RL_TRAINABLE_LSTM}" != "1" ]]; then
      echo "Temporal policy: initialized uniformly from scratch"
    fi
  fi
else
  echo "Temporal features: disabled (set GAMEAGENT_TEMPORAL_MODEL=1 to enable)"
fi
if [[ "${GAMEAGENT_REINFORCEMENT_LEARNING}" == "1" ]]; then
  if [[ "${GAMEAGENT_TEMPORAL_MODEL}" != "1" ]]; then
    echo "RL requires GAMEAGENT_TEMPORAL_MODEL=1." >&2
    exit 1
  fi
  mkdir -p "${GAMEAGENT_RL_CHECKPOINT_DIR}"
  echo "Reinforcement control: shared multi-character recurrent semi-Markov Double DQN"
  echo "RL mode: $([[ "${GAMEAGENT_RL_TRAINING}" == "1" ]] && echo training || echo evaluation)"
  if [[ "${GAMEAGENT_RL_TRAINING}" == "0" ]]; then
    echo "RL persistence: read-only (no replay append, optimizer update or checkpoint write)"
  fi
  echo "Fight gate: health HUD, player ${GAMEAGENT_PLAYER_SIDE}"
  if [[ "${GAMEAGENT_RL_TRAINABLE_LSTM}" == "1" ]]; then
    echo "RL metadata: ${GAMEAGENT_RL_CHECKPOINT_DIR}/shared_recurrent_metadata_latest.garl"
    echo "RL best metadata: ${GAMEAGENT_RL_CHECKPOINT_DIR}/shared_recurrent_metadata_best_<character>.garl"
  else
    echo "RL latest: ${GAMEAGENT_RL_CHECKPOINT_DIR}/shared_policy_latest.garl"
    echo "RL best: ${GAMEAGENT_RL_CHECKPOINT_DIR}/shared_policy_best_<character>.garl"
  fi
  echo "RL transfer: shared trunk + character/opponent embeddings + per-character heads"
  echo "RL trainable LSTM: ${GAMEAGENT_RL_TRAINABLE_LSTM}"
  echo "RL trainable ResNet50 stage 4: ${GAMEAGENT_RL_TRAINABLE_RESNET}"
  if [[ "${GAMEAGENT_RL_TRAINABLE_LSTM}" == "1" ]]; then
    if [[ ! -f "${GAMEAGENT_RL_RECURRENT_SERVICE}" ]]; then
      echo "Recurrent RL service not found: ${GAMEAGENT_RL_RECURRENT_SERVICE}" >&2
      exit 1
    fi
    if ! python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
      echo "Trainable recurrent RL requires PyTorch with CUDA support." >&2
      exit 1
    fi
    echo "RL recurrent checkpoint: ${GAMEAGENT_RL_CHECKPOINT_DIR}/shared_recurrent_latest.pt"
    echo "RL recurrent device: cuda:${GAMEAGENT_DEVICE_ID}"
    if [[ "${GAMEAGENT_RL_TRAINABLE_RESNET}" == "1" ]]; then
      echo "RL ResNet fine-tuning: lr=${GAMEAGENT_RL_RESNET_LEARNING_RATE}, batch=${GAMEAGENT_RL_RESNET_BATCH_SIZE}, every=${GAMEAGENT_RL_RESNET_UPDATE_INTERVAL} recurrent updates"
      echo "RL ResNet compressed replay images: ${GAMEAGENT_RL_RESNET_REPLAY_IMAGES}"
    fi
  fi
  echo "RL resume: ${GAMEAGENT_RL_RESUME}"
  echo "RL replay=${GAMEAGENT_RL_REPLAY_CAPACITY}, warmup=${GAMEAGENT_RL_MIN_REPLAY_SIZE}, batch=${GAMEAGENT_RL_BATCH_SIZE}"
  echo "RL epsilon=${GAMEAGENT_RL_EPSILON_START}->${GAMEAGENT_RL_EPSILON_FINAL} over ${GAMEAGENT_RL_EPSILON_DECAY} options"
  echo "Exploration split: ${GAMEAGENT_RL_SCRIPTED_EXPLORATION} scripted, remainder primitive"
  echo "RL reward summary: every ${GAMEAGENT_RL_REWARD_LOG_INTERVAL_MS} ms"
  echo "Training reset request after round_lost: ${GAMEAGENT_TRAINING_RESET_STATE}"
else
  echo "Reinforcement control: disabled (teacher/mock controls the client)"
fi
if [[ "${GAMEAGENT_SAVE_FRAMES}" == "1" ]]; then
  echo "Frame capture: one debug JPEG every ${GAMEAGENT_DEBUG_CAPTURE_INTERVAL_MS} ms"
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
