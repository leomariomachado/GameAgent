#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"
cd "${GAMEAGENT_ROOT}"
exec python3 -u tests/benchmark_joint_training_memory.py
