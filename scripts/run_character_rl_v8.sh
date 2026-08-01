#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"

# v8 never overwrites v7. It imports only the trained projection, ResNet50,
# LSTM, embeddings and shared strategy representation; value heads, replay
# and optimizer state start clean under the corrected target formulation.
export GAMEAGENT_RL_CHECKPOINT_DIR="${GAMEAGENT_RL_CHECKPOINT_DIR:-/workspace/gameagent/rl_combat_state_v8}"
export GAMEAGENT_RL_MIGRATION_CHECKPOINT="${GAMEAGENT_RL_MIGRATION_CHECKPOINT:-/workspace/gameagent/rl_combat_state_v7_calibrated/shared_recurrent_latest.pt}"
export GAMEAGENT_RL_RESUME="${GAMEAGENT_RL_RESUME:-1}"

exec bash "${GAMEAGENT_ROOT}/scripts/run_character_rl.sh"
