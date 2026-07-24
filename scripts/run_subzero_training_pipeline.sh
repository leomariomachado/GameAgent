#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"
TRAINING_CAPTURES="${GAMEAGENT_TRAINING_CAPTURE_DIR:-/workspace/gameagent/training_captures}"
TRAINING_FEATURES="${GAMEAGENT_TRAINING_FEATURE_DIR:-/workspace/gameagent/training_features}"
CHECKPOINT="${GAMEAGENT_RESNET50_CHECKPOINT:-/workspace/resnet50_best.ckpt}"
METRICS="${GAMEAGENT_METRICS_PATH:-/workspace/gameagent/metrics.jsonl}"
CHECKPOINT_DIR="${GAMEAGENT_TEMPORAL_CHECKPOINT_DIR:-/workspace/gameagent/checkpoints}"

if [[ ! -d "${TRAINING_CAPTURES}" ]]; then
  echo "Training captures not found: ${TRAINING_CAPTURES}" >&2
  echo "Collect a gameplay session with GAMEAGENT_COLLECT_TRAINING_DATA=1." >&2
  exit 1
fi

GAMEAGENT_CAPTURE_ROOT="${TRAINING_CAPTURES}" \
GAMEAGENT_FEATURE_OUTPUT_DIR="${TRAINING_FEATURES}" \
GAMEAGENT_RESNET50_CHECKPOINT="${CHECKPOINT}" \
bash "${GAMEAGENT_ROOT}/scripts/run_resnet50_feature_test.sh"

GAMEAGENT_FEATURE_OUTPUT_DIR="${TRAINING_FEATURES}" \
GAMEAGENT_METRICS_PATH="${METRICS}" \
GAMEAGENT_TEMPORAL_CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
bash "${GAMEAGENT_ROOT}/scripts/run_subzero_pretraining.sh"

echo "[PASS] Sub-Zero capture-to-checkpoint pipeline completed"
