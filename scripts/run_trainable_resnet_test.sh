#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"
export GAMEAGENT_RESNET50_CHECKPOINT="${GAMEAGENT_RESNET50_CHECKPOINT:-/workspace/resnet50_best.ckpt}"

if [[ ! -f "${GAMEAGENT_RESNET50_CHECKPOINT}" ]]; then
  echo "ResNet50 checkpoint not found: ${GAMEAGENT_RESNET50_CHECKPOINT}" >&2
  exit 1
fi

cd "${GAMEAGENT_ROOT}"
exec python3 -u tests/test_trainable_resnet_service.py
