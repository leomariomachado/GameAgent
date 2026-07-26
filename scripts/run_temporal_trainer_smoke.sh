#!/usr/bin/env bash
set -euo pipefail

GAMEAGENT_ROOT="${GAMEAGENT_ROOT:-/tmp/GameAgent}"
SMOKE_ROOT="${GAMEAGENT_TEMPORAL_SMOKE_ROOT:-}"
if [[ -z "${SMOKE_ROOT}" ]]; then
  SMOKE_ROOT="$(mktemp -d /tmp/gameagent_temporal_smoke.XXXXXX)"
else
  mkdir -p "${SMOKE_ROOT}"
fi
OUTPUT_DIR="${SMOKE_ROOT}/checkpoints"

python3 "${GAMEAGENT_ROOT}/tests/generate_temporal_smoke_data.py" \
  "${SMOKE_ROOT}/data"

python3 -u "${GAMEAGENT_ROOT}/scripts/train_subzero_temporal.py" \
  --features-manifest "${SMOKE_ROOT}/data/features.jsonl" \
  --metrics "${SMOKE_ROOT}/data/metrics.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 12 \
  --batch-size 64 \
  --feature-load-workers 8 \
  --early-stopping-patience 5

test -s "${OUTPUT_DIR}/subzero_temporal_best.gat"
echo "[PASS] temporal trainer v2 smoke checkpoint: ${OUTPUT_DIR}/subzero_temporal_best.gat"
