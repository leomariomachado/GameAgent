#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"

# v8 never overwrites v7. It imports only the trained projection, ResNet50,
# LSTM, embeddings and shared strategy representation; value heads, replay
# and optimizer state start clean under the corrected target formulation.
export GAMEAGENT_RL_CHECKPOINT_DIR="${GAMEAGENT_RL_CHECKPOINT_DIR:-/workspace/gameagent/rl_combat_state_v8}"
export GAMEAGENT_RL_MIGRATION_CHECKPOINT="${GAMEAGENT_RL_MIGRATION_CHECKPOINT:-/workspace/gameagent/rl_combat_state_v7_calibrated/shared_recurrent_latest.pt}"
export GAMEAGENT_RL_RESUME="${GAMEAGENT_RL_RESUME:-1}"
export GAMEAGENT_RL_BACKEND="${GAMEAGENT_RL_BACKEND:-cpp}"

if [[ "${GAMEAGENT_RL_BACKEND}" == "cpp" ]]; then
  CPP_CHECKPOINT="${GAMEAGENT_RL_CHECKPOINT_DIR}/shared_recurrent_latest.baseai.ckpt"
  MIGRATED_RESNET="${GAMEAGENT_RL_CHECKPOINT_DIR}/resnet50_stage4_migrated.ckpt"
  BASE_RESNET="${GAMEAGENT_RESNET50_CHECKPOINT:-/workspace/resnet50_best.ckpt}"
  if [[ ! -f "${CPP_CHECKPOINT}" &&
        -f "${GAMEAGENT_RL_MIGRATION_CHECKPOINT}" ]]; then
    mkdir -p "${GAMEAGENT_RL_CHECKPOINT_DIR}"
    python3 "${GAMEAGENT_ROOT}/scripts/convert_recurrent_checkpoint_to_baseai.py" \
      --input "${GAMEAGENT_RL_MIGRATION_CHECKPOINT}" \
      --output "${CPP_CHECKPOINT}" \
      --representation-only \
      --base-resnet "${BASE_RESNET}" \
      --output-resnet "${MIGRATED_RESNET}"
  fi
  if [[ -f "${MIGRATED_RESNET}" ]]; then
    export GAMEAGENT_RESNET50_CHECKPOINT="${MIGRATED_RESNET}"
  fi
fi

exec bash "${GAMEAGENT_ROOT}/scripts/run_character_rl.sh"
