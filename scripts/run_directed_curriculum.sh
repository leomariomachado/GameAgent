#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"
CURRICULUM_PATH="${GAMEAGENT_RL_DIRECTED_CURRICULUM_PATH:-/workspace/gameagent/action_curriculum.json}"

if [[ ! -f "${CURRICULUM_PATH}" ]]; then
  echo "Directed curriculum JSON not found: ${CURRICULUM_PATH}" >&2
  echo "Copy ${GAMEAGENT_ROOT}/configs/directed_curriculum.example.json first." >&2
  exit 1
fi

export GAMEAGENT_RL_DIRECTED_CURRICULUM_PATH="${CURRICULUM_PATH}"
exec bash "${GAMEAGENT_ROOT}/scripts/run_character_rl.sh"
