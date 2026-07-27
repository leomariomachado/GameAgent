#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"

echo "[INFO] run_subzero_rl.sh is now a compatibility alias."
echo "[INFO] The client session_config selects the character; using the uniform multi-character RL launcher."
exec bash "${GAMEAGENT_ROOT}/scripts/run_character_rl.sh"
