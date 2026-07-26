#!/usr/bin/env bash

# Shared RunPod CMake resolver. This file is meant to be sourced by the build
# helpers so every entry point uses the same CMake/CTest pair.

gameagent_cmake_version_is_supported() {
  local executable="$1"
  local minimum="$2"
  local version
  local oldest
  version="$("${executable}" --version 2>/dev/null | awk 'NR == 1 {print $3}')"
  [[ -n "${version}" ]] || return 1
  oldest="$(
    printf '%s\n%s\n' "${minimum}" "${version}" |
      sort -V |
      head -n 1
  )"
  [[ "${oldest}" == "${minimum}" ]]
}

gameagent_resolve_executable() {
  local candidate="$1"
  if [[ "${candidate}" == */* ]]; then
    [[ -x "${candidate}" ]] && printf '%s\n' "${candidate}"
  else
    command -v "${candidate}" 2>/dev/null || true
  fi
}

gameagent_find_supported_cmake() {
  local minimum="$1"
  local candidate
  local resolved
  for candidate in \
    "${GAMEAGENT_CMAKE_BIN:-}" \
    "${HOME}/.local/bin/cmake" \
    "cmake"; do
    [[ -n "${candidate}" ]] || continue
    resolved="$(gameagent_resolve_executable "${candidate}")"
    if [[ -n "${resolved}" ]] &&
       gameagent_cmake_version_is_supported "${resolved}" "${minimum}"; then
      printf '%s\n' "${resolved}"
      return 0
    fi
  done
  return 1
}

gameagent_ensure_cmake() {
  local minimum="${GAMEAGENT_MINIMUM_CMAKE_VERSION:-3.28}"
  local cmake_bin
  local ctest_bin

  cmake_bin="$(gameagent_find_supported_cmake "${minimum}" || true)"
  if [[ -z "${cmake_bin}" ]]; then
    if ! python3 -m pip --version >/dev/null 2>&1; then
      echo "CMake ${minimum}+ is required and Python pip is unavailable." >&2
      return 1
    fi
    echo "[SETUP] Installing CMake ${minimum}+ in ${HOME}/.local ..." >&2
    python3 -m pip install \
      --user \
      --upgrade \
      --disable-pip-version-check \
      --no-warn-script-location \
      "cmake>=${minimum},<4"
    cmake_bin="$(gameagent_find_supported_cmake "${minimum}" || true)"
  fi
  if [[ -z "${cmake_bin}" ]]; then
    echo "Unable to locate CMake ${minimum}+ after installation." >&2
    return 1
  fi

  ctest_bin="$(dirname "${cmake_bin}")/ctest"
  if [[ ! -x "${ctest_bin}" ]]; then
    ctest_bin="$(command -v ctest || true)"
  fi
  if [[ -z "${ctest_bin}" || ! -x "${ctest_bin}" ]]; then
    echo "Unable to locate ctest next to ${cmake_bin}." >&2
    return 1
  fi

  export GAMEAGENT_CMAKE_BIN="${cmake_bin}"
  export GAMEAGENT_CTEST_BIN="${ctest_bin}"
  echo "[INFO] CMake: ${GAMEAGENT_CMAKE_BIN} ($("${GAMEAGENT_CMAKE_BIN}" --version | head -n 1))"
}
