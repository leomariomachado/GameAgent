#!/usr/bin/env python3
"""Dependency-free end-to-end smoke test for the GameAgent V2.1 server."""

from __future__ import annotations

import argparse
import json
import os
import select
import struct
import sys
import time
from pathlib import Path

from send_websocket_frames import (
    binary_envelope,
    connect_websocket,
    receive_server_frame,
    send_client_frame,
)


def send_json(connection, payload: dict) -> None:
    send_client_frame(
        connection,
        0x1,
        json.dumps(payload, separators=(",", ":")).encode("utf-8"),
    )


def receive_json(connection, timeout: float) -> dict:
    readable, _, _ = select.select([connection], [], [], timeout)
    if not readable:
        raise TimeoutError("timed out waiting for a server message")
    while True:
        opcode, payload = receive_server_frame(connection)
        if opcode == 0x9:
            send_client_frame(connection, 0xA, payload)
            continue
        if opcode == 0x8:
            raise ConnectionError("server closed the WebSocket")
        if opcode != 0x1:
            continue
        return json.loads(payload)


def message(message_type: str, request_id: int, **fields) -> dict:
    return {
        "protocol_version": 2,
        "request_id": request_id,
        "timestamp_ms": time.time_ns() // 1_000_000,
        "type": message_type,
        **fields,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:18080/ws")
    parser.add_argument("--jpeg", type=Path, required=True)
    parser.add_argument("--token", default=os.getenv("GAMEAGENT_API_TOKEN"))
    parser.add_argument("--timeout", type=float, default=20.0)
    arguments = parser.parse_args()

    jpeg = arguments.jpeg.read_bytes()
    if not jpeg.startswith(b"\xff\xd8"):
        parser.error("--jpeg must point to a JPEG image")

    connection = connect_websocket(arguments.url, arguments.token)
    seen: set[str] = set()
    try:
        send_json(
            connection,
            message(
                "configure_session",
                1,
                player_character="noob_saibot",
                player_side="left",
                initial_facing="right",
                encounter_type="endurance",
                opponent_count=2,
                current_stage=0,
                opponents=["jax", "nightwolf"],
            ),
        )
        configured = receive_json(connection, arguments.timeout)
        if configured.get("type") != "configure_session_ack":
            raise RuntimeError(f"configuration failed: {configured}")
        if configured.get("architecture_version") != 21:
            raise RuntimeError("server did not advertise architecture V2.1")

        now_ms = time.time_ns() // 1_000_000
        send_client_frame(connection, 0x2, binary_envelope(1, now_ms, jpeg))
        send_json(
            connection,
            message(
                "set_agent_mode",
                2,
                mode="evaluation",
                reset_evaluation_session=True,
                apply_on_next_episode=False,
            ),
        )
        send_json(
            connection,
            message(
                "set_exploration",
                3,
                mode="automatic_from_value",
                value=0.35,
                reset_matchup_counter=True,
            ),
        )

        deadline = time.monotonic() + arguments.timeout
        next_frame = time.monotonic()
        frame_id = 2
        while time.monotonic() < deadline:
            if time.monotonic() >= next_frame:
                now_ms = time.time_ns() // 1_000_000
                send_client_frame(
                    connection,
                    0x2,
                    binary_envelope(frame_id, now_ms, jpeg),
                )
                frame_id += 1
                next_frame += 1.0 / 15.0
            try:
                response = receive_json(connection, 0.02)
            except TimeoutError:
                continue
            response_type = response.get("type")
            if isinstance(response_type, str):
                seen.add(response_type)
            if response_type == "action":
                send_json(
                    connection,
                    {
                        "protocol_version": 2,
                        "type": "action_ack",
                        "decision_id": response["decision_id"],
                        "applied_commands": response.get("commands", []),
                        "applied_timestamp_ms": time.time_ns() // 1_000_000,
                    },
                )
            elif response_type == "agent_mode_changed":
                if response.get("effective_mode") != "evaluation":
                    raise RuntimeError("evaluation mode was not applied")
                if response.get("effective_epsilon") != 0.0:
                    raise RuntimeError("evaluation epsilon is not zero")
            elif response_type == "telemetry":
                if response.get("agent_mode") != "evaluation":
                    continue
                if response.get("epsilon") != 0.0:
                    raise RuntimeError("telemetry evaluation epsilon is not zero")
                if not all(
                    response.get(field) is True
                    for field in (
                        "weights_frozen",
                        "replay_frozen",
                        "scheduler_frozen",
                    )
                ):
                    raise RuntimeError("evaluation learner is not fully frozen")

            required = {
                "agent_mode_changed",
                "exploration_changed",
                "action",
                "telemetry",
                "telemetry_sample",
                "telemetry_marker",
            }
            if required.issubset(seen):
                print(
                    "[PASS] GameAgent V2.1 WebSocket smoke passed "
                    f"messages={','.join(sorted(seen))} frames={frame_id - 1}"
                )
                return 0
        raise TimeoutError(f"missing V2.1 messages: {sorted(required - seen)}")
    finally:
        try:
            send_client_frame(connection, 0x8, struct.pack("!H", 1000))
        except OSError:
            pass
        connection.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        raise
