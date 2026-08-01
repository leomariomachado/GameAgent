#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"
SOURCE_DIR="${GAMEAGENT_RL_CHECKPOINT_DIR:-/workspace/gameagent/rl_combat_state_v8}"

export GAMEAGENT_EVALUATION_SOURCE_CHECKPOINT="${GAMEAGENT_EVALUATION_SOURCE_CHECKPOINT:-${SOURCE_DIR}/shared_recurrent_latest.pt}"
export GAMEAGENT_EVALUATION_SOURCE_METADATA="${GAMEAGENT_EVALUATION_SOURCE_METADATA:-${SOURCE_DIR}/shared_recurrent_metadata_latest.garl}"

exec bash "${GAMEAGENT_ROOT}/scripts/run_character_rl_evaluation.sh"
