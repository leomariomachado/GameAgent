#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"
BASEAI_ROOT="${BASEAI_ROOT:-/tmp/BaseAI}"
BUILD_DIR="${GAMEAGENT_BUILD_DIR:-${GAMEAGENT_ROOT}/buildrelease}"
CHECKPOINT="${GAMEAGENT_RESNET50_CHECKPOINT:-/workspace/resnet50_best.ckpt}"
CAPTURE_ROOT="${GAMEAGENT_CAPTURE_ROOT:-/workspace/gameagent/captures}"
OUTPUT_DIR="${GAMEAGENT_FEATURE_OUTPUT_DIR:-/workspace/gameagent/resnet50_features}"
BUILD_JOBS="${GAMEAGENT_BUILD_JOBS:-$(nproc)}"

if [[ ! -f "${GAMEAGENT_ROOT}/CMakeLists.txt" ]]; then
  echo "GameAgent source tree not found: ${GAMEAGENT_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${BASEAI_ROOT}/CMakeLists.txt" ]]; then
  echo "BaseAI source tree not found: ${BASEAI_ROOT}" >&2
  exit 1
fi

export PATH="/usr/local/cuda/bin:${PATH}"
source "${GAMEAGENT_ROOT}/scripts/ensure_cmake.sh"
gameagent_ensure_cmake

CUDA_COMPILER="${CMAKE_CUDA_COMPILER:-$(command -v nvcc)}"
CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-}"
if [[ -z "${CUDA_ARCHITECTURES}" ]]; then
  CUDA_ARCHITECTURES="$(
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader |
      head -n 1 |
      tr -d '.'
  )"
fi

"${GAMEAGENT_CMAKE_BIN}" -S "${GAMEAGENT_ROOT}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="${CUDA_COMPILER}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
  -DGAMEAGENT_BASEAI_SOURCE_DIR="${BASEAI_ROOT}"

"${GAMEAGENT_CMAKE_BIN}" --build "${BUILD_DIR}" \
  --target test_gameagent_resnet50_feature_extractor \
  -j"${BUILD_JOBS}"

export GAMEAGENT_RESNET50_CHECKPOINT="${CHECKPOINT}"
export GAMEAGENT_CAPTURE_ROOT="${CAPTURE_ROOT}"
export GAMEAGENT_FEATURE_OUTPUT_DIR="${OUTPUT_DIR}"

exec "${BUILD_DIR}/test_gameagent_resnet50_feature_extractor"
