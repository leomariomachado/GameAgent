#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"
FEATURE_ROOT="${GAMEAGENT_FEATURE_OUTPUT_DIR:-/workspace/gameagent/resnet50_features}"
METRICS="${GAMEAGENT_METRICS_PATH:-/workspace/gameagent/metrics.jsonl}"
OUTPUT_DIR="${GAMEAGENT_TEMPORAL_CHECKPOINT_DIR:-/workspace/gameagent/checkpoints}"
EPOCHS="${GAMEAGENT_PRETRAIN_EPOCHS:-30}"
BATCH_SIZE="${GAMEAGENT_PRETRAIN_BATCH_SIZE:-64}"
RESUME="${GAMEAGENT_PRETRAIN_RESUME:-0}"

if [[ ! -f "${GAMEAGENT_ROOT}/scripts/train_subzero_temporal.py" ]]; then
  echo "Sub-Zero trainer not found under ${GAMEAGENT_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${FEATURE_ROOT}" ]]; then
  echo "ResNet50 feature root not found: ${FEATURE_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${METRICS}" ]]; then
  echo "GameAgent metrics file not found: ${METRICS}" >&2
  exit 1
fi

arguments=(
  --feature-root "${FEATURE_ROOT}"
  --metrics "${METRICS}"
  --output-dir "${OUTPUT_DIR}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
)
if [[ "${RESUME}" == "1" ]]; then
  arguments+=(--resume)
fi

cd "${GAMEAGENT_ROOT}"
exec python3 scripts/train_subzero_temporal.py "${arguments[@]}"
