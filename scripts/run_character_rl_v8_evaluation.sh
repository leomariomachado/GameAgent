#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"
SOURCE_DIR="${GAMEAGENT_RL_CHECKPOINT_DIR:-/workspace/gameagent/rl_combat_state_v8}"
RL_BACKEND="${GAMEAGENT_RL_BACKEND:-cpp}"
if [[ "${RL_BACKEND}" == "cpp" ]]; then
  RECURRENT_FILENAME="shared_recurrent_latest.baseai.ckpt"
else
  RECURRENT_FILENAME="shared_recurrent_latest.pt"
fi

export GAMEAGENT_RL_BACKEND="${RL_BACKEND}"
export GAMEAGENT_EVALUATION_SOURCE_CHECKPOINT="${GAMEAGENT_EVALUATION_SOURCE_CHECKPOINT:-${SOURCE_DIR}/${RECURRENT_FILENAME}}"
export GAMEAGENT_EVALUATION_SOURCE_METADATA="${GAMEAGENT_EVALUATION_SOURCE_METADATA:-${SOURCE_DIR}/shared_recurrent_metadata_latest.garl}"

exec bash "${GAMEAGENT_ROOT}/scripts/run_character_rl_evaluation.sh"
