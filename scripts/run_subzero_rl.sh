#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"

export GAMEAGENT_TEMPORAL_MODEL=1
export GAMEAGENT_REINFORCEMENT_LEARNING=1
export GAMEAGENT_RL_TRAINING="${GAMEAGENT_RL_TRAINING:-1}"
export GAMEAGENT_RL_RESUME="${GAMEAGENT_RL_RESUME:-1}"
export GAMEAGENT_PLAYER_SIDE="${GAMEAGENT_PLAYER_SIDE:-left}"
export GAMEAGENT_SUBZERO_FACING="${GAMEAGENT_SUBZERO_FACING:-right}"
export GAMEAGENT_RESNET50_CHECKPOINT="${GAMEAGENT_RESNET50_CHECKPOINT:-/workspace/resnet50_best.ckpt}"
export GAMEAGENT_TEMPORAL_CHECKPOINT="${GAMEAGENT_TEMPORAL_CHECKPOINT:-/workspace/gameagent/checkpoints_v2/subzero_temporal_latest.gat}"
export GAMEAGENT_RL_CHECKPOINT="${GAMEAGENT_RL_CHECKPOINT:-/workspace/gameagent/rl_checkpoints/subzero_dqn_latest.garl}"
export GAMEAGENT_RL_BEST_CHECKPOINT="${GAMEAGENT_RL_BEST_CHECKPOINT:-/workspace/gameagent/rl_checkpoints/subzero_dqn_best.garl}"
export GAMEAGENT_SAVE_FRAMES="${GAMEAGENT_SAVE_FRAMES:-0}"
export GAMEAGENT_COLLECT_TRAINING_DATA="${GAMEAGENT_COLLECT_TRAINING_DATA:-0}"

exec bash "${GAMEAGENT_ROOT}/scripts/run_runpod_server.sh"
