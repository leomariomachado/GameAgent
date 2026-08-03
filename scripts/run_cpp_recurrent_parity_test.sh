#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"
BUILD_DIR="${GAMEAGENT_BUILD_DIR:-${GAMEAGENT_ROOT}/buildrelease}"
CHECKPOINT="${GAMEAGENT_CPP_PARITY_CHECKPOINT:-/tmp/gameagent_cpp_parity.baseai.ckpt}"
FIXTURE="${GAMEAGENT_CPP_PARITY_FIXTURE:-/tmp/gameagent_cpp_parity.txt}"
EXPECTED_TRAINED="${GAMEAGENT_CPP_PARITY_EXPECTED_TRAINED:-/tmp/gameagent_cpp_parity_expected.baseai.ckpt}"
ACTUAL_TRAINED="${GAMEAGENT_CPP_PARITY_ACTUAL_TRAINED:-/tmp/gameagent_cpp_parity_actual.baseai.ckpt}"

python3 "${GAMEAGENT_ROOT}/tests/generate_cpp_recurrent_parity.py" \
  "${CHECKPOINT}" "${FIXTURE}" "${EXPECTED_TRAINED}"
"${BUILD_DIR}/test_gameagent_cpp_recurrent_parity" \
  "${CHECKPOINT}" "${FIXTURE}"
"${BUILD_DIR}/test_gameagent_cpp_recurrent_training_parity" \
  "${CHECKPOINT}" "${FIXTURE}" "${ACTUAL_TRAINED}"
python3 "${GAMEAGENT_ROOT}/tests/compare_cpp_recurrent_checkpoints.py" \
  "${EXPECTED_TRAINED}" "${ACTUAL_TRAINED}"
