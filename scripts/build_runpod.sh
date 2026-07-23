#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"
BASEAI_ROOT="${BASEAI_ROOT:-/tmp/BaseAI}"
BUILD_DIR="${GAMEAGENT_BUILD_DIR:-${GAMEAGENT_ROOT}/buildrelease}"
BUILD_JOBS="${GAMEAGENT_BUILD_JOBS:-$(nproc)}"

export PATH="/usr/local/cuda/bin:${PATH}"
CUDA_COMPILER="${CMAKE_CUDA_COMPILER:-$(command -v nvcc)}"
CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-}"
if [[ -z "${CUDA_ARCHITECTURES}" ]]; then
  CUDA_ARCHITECTURES="$(
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader |
      head -n 1 |
      tr -d '.'
  )"
fi

cmake -S "${GAMEAGENT_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${CUDA_COMPILER}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  -DGAMEAGENT_BASEAI_SOURCE_DIR="${BASEAI_ROOT}"

cmake --build "${BUILD_DIR}" --target \
  GameAgent \
  test_gameagent_mock_protocol \
  test_gameagent_http_server \
  test_gameagent_stream_protocol \
  test_gameagent_frame_buffer \
  test_gameagent_websocket_server \
  test_gameagent_action_history \
  -j"${BUILD_JOBS}"

ctest --test-dir "${BUILD_DIR}" --output-on-failure

echo "[PASS] GameAgent RunPod build and tests passed"
