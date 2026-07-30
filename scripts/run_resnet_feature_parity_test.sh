#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"
cd "${GAMEAGENT_ROOT}"
exec python3 -u tests/test_resnet_feature_parity.py
