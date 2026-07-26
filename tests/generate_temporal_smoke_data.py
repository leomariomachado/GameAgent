#!/usr/bin/env python3
"""Generate a small deterministic decision-clock dataset for trainer tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COMMANDS_BY_PHASE: tuple[tuple[int, ...], ...] = (
    (),
    (3,),
    (9,),
    (2, 7),
)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    root = arguments.output.resolve()
    root.mkdir(parents=True, exist_ok=True)

    feature_records: list[dict[str, object]] = []
    metric_records: list[dict[str, object]] = []
    for connection_id in range(1, 6):
        sequence_id = (
            root
            / f"session_{connection_id}"
            / f"connection_{connection_id}"
        )
        sequence_id.mkdir(parents=True, exist_ok=True)
        start_ms = 1_800_000_000_000 + connection_id * 100_000
        frame_count = 64
        for frame_index in range(frame_count):
            frame_id = frame_index + 1
            frame_time = start_ms + frame_index * 100
            phase = frame_index % len(COMMANDS_BY_PHASE)
            feature = np.zeros(2048, dtype=np.float32)
            feature[phase] = 4.0
            feature[16 + connection_id] = 1.0
            feature[32] = frame_index / frame_count
            feature_path = sequence_id / f"feature_{frame_id:06d}.f32"
            feature.tofile(feature_path)
            feature_records.append({
                "sequence_id": str(sequence_id),
                "frame_id": frame_id,
                "received_timestamp_ms": frame_time,
                "feature_path": str(feature_path),
                "feature_size": 2048,
            })

        decision_count = frame_count * 2 - 1
        for decision_index in range(decision_count):
            decision_id = decision_index + 1
            decision_time = start_ms + decision_index * 50
            frame_index = min(decision_index // 2, frame_count - 1)
            commands = list(
                COMMANDS_BY_PHASE[
                    frame_index % len(COMMANDS_BY_PHASE)
                ]
            )
            metric_records.append({
                "event": "decision",
                "connection_id": connection_id,
                "decision_id": decision_id,
                "timestamp_ms": decision_time,
                "commands": commands,
                "curriculum_source": "synthetic_smoke",
                "mock_move": f"phase_{frame_index % 4}",
            })
            metric_records.append({
                "event": "action_ack",
                "connection_id": connection_id,
                "decision_id": decision_id,
                "acknowledgement_received_timestamp_ms":
                    decision_time + 10,
                "applied_commands": commands,
            })

    write_jsonl(root / "features.jsonl", feature_records)
    write_jsonl(root / "metrics.jsonl", metric_records)
    print(
        f"[PASS] temporal smoke data: frames={len(feature_records)} "
        f"records={len(metric_records)} root={root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
