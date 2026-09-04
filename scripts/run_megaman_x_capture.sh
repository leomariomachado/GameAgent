#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"

# Capture-only mode deliberately does not create a temporal policy, replay or
# checkpoint. It validates the JPEG stream and retains sparse native frames
# for HUD/state analysis before Mega Man reward logic is implemented.
export GAMEAGENT_GAME_PROFILE="${GAMEAGENT_GAME_PROFILE:-megaman_x}"
export GAMEAGENT_TEMPORAL_MODEL=0
export GAMEAGENT_REINFORCEMENT_LEARNING=0
export GAMEAGENT_SAVE_FRAMES=1
export GAMEAGENT_CAPTURE_DIR="${GAMEAGENT_CAPTURE_DIR:-/workspace/gameagent/megaman_x_captures}"
export GAMEAGENT_DEBUG_CAPTURE_INTERVAL_MS="${GAMEAGENT_DEBUG_CAPTURE_INTERVAL_MS:-3000}"

exec bash "${GAMEAGENT_ROOT}/scripts/run_runpod_server.sh"
