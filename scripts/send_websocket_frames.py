#!/usr/bin/env python3
"""Dependency-free diagnostic client for the GameAgent WebSocket stream."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import select
import socket
import ssl
import struct
import time
from pathlib import Path
from urllib.parse import urlparse


WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("WebSocket connection closed")
        chunks.extend(chunk)
    return bytes(chunks)


def connect_websocket(url: str, token: str | None) -> socket.socket:
    parsed = urlparse(url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
        raise ValueError("URL must start with ws:// or wss://")
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    connection = socket.create_connection((parsed.hostname, port), timeout=10)
    if parsed.scheme == "wss":
        context = ssl.create_default_context()
        connection = context.wrap_socket(connection, server_hostname=parsed.hostname)

    key = base64.b64encode(os.urandom(16)).decode("ascii")
    path = parsed.path or "/ws"
    if parsed.query:
        path += "?" + parsed.query
    headers = [
        f"GET {path} HTTP/1.1",
        f"Host: {parsed.netloc}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
    ]
    if token:
        headers.append(f"Authorization: Bearer {token}")
    connection.sendall(("\r\n".join(headers) + "\r\n\r\n").encode("ascii"))

    response = bytearray()
    while b"\r\n\r\n" not in response:
        response.extend(connection.recv(4096))
        if len(response) > 65536:
            raise ConnectionError("oversized HTTP upgrade response")
    header_text = response.decode("latin-1")
    if not header_text.startswith("HTTP/1.1 101 "):
        raise ConnectionError(f"WebSocket upgrade rejected:\n{header_text}")
    expected = base64.b64encode(
        hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
    ).decode("ascii")
    if f"sec-websocket-accept: {expected}".lower() not in header_text.lower():
        raise ConnectionError("server returned an invalid Sec-WebSocket-Accept")
    connection.settimeout(None)
    return connection


def send_client_frame(connection: socket.socket, opcode: int, payload: bytes) -> None:
    mask = os.urandom(4)
    header = bytearray([0x80 | (opcode & 0x0F)])
    size = len(payload)
    if size <= 125:
        header.append(0x80 | size)
    elif size <= 0xFFFF:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", size))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", size))
    header.extend(mask)
    masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    connection.sendall(header + masked)


def receive_server_frame(connection: socket.socket) -> tuple[int, bytes]:
    first, second = receive_exact(connection, 2)
    opcode = first & 0x0F
    size = second & 0x7F
    if size == 126:
        size = struct.unpack("!H", receive_exact(connection, 2))[0]
    elif size == 127:
        size = struct.unpack("!Q", receive_exact(connection, 8))[0]
    if second & 0x80:
        mask = receive_exact(connection, 4)
    else:
        mask = None
    payload = receive_exact(connection, size)
    if mask:
        payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    return opcode, payload


def binary_envelope(frame_id: int, timestamp_ms: int, jpeg: bytes) -> bytes:
    return struct.pack("!4sQqI", b"GAF1", frame_id, timestamp_ms, len(jpeg)) + jpeg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8080/ws")
    parser.add_argument("--jpeg", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--token", default=os.getenv("GAMEAGENT_API_TOKEN"))
    args = parser.parse_args()
    if args.fps <= 0 or args.frames <= 0:
        parser.error("--fps and --frames must be positive")

    jpeg = args.jpeg.read_bytes()
    if not jpeg.startswith(b"\xff\xd8"):
        parser.error("--jpeg must point to a JPEG image")
    connection = connect_websocket(args.url, args.token)
    print(f"[CONNECTED] {args.url}")

    frame_id = 1
    next_frame = time.monotonic()
    sent = 0
    received = 0
    sent_at: dict[int, float] = {}
    current_commands: set[int] = set()
    last_decision_id = 0
    try:
        while sent < args.frames or time.monotonic() < next_frame + 0.25:
            now = time.monotonic()
            if sent < args.frames and now >= next_frame:
                timestamp_ms = time.time_ns() // 1_000_000
                send_client_frame(
                    connection, 0x2, binary_envelope(frame_id, timestamp_ms, jpeg)
                )
                sent_at[frame_id] = time.monotonic()
                frame_id += 1
                sent += 1
                next_frame += 1.0 / args.fps

            readable, _, _ = select.select([connection], [], [], 0.005)
            if readable:
                opcode, payload = receive_server_frame(connection)
                if opcode == 0x1:
                    message = json.loads(payload)
                    if message.get("type") == "action":
                        if message.get("protocol_version") != 2:
                            raise RuntimeError("server did not return action protocol version 2")
                        if message.get("action_mode") != "state":
                            raise RuntimeError("server did not return state action mode")
                        decision_id = message.get("decision_id")
                        if not isinstance(decision_id, int) or decision_id <= last_decision_id:
                            raise RuntimeError("server returned a stale or invalid decision_id")
                        commands = message.get("commands")
                        if (
                            not isinstance(commands, list)
                            or len(commands) != len(set(commands))
                            or any(not isinstance(value, int) or value < 0 or value > 9 for value in commands)
                        ):
                            raise RuntimeError("server returned an invalid command state")
                        new_commands = set(commands)
                        message["client_state_transition"] = {
                            "pressed": sorted(new_commands - current_commands),
                            "released": sorted(current_commands - new_commands),
                        }
                        current_commands = new_commands
                        last_decision_id = decision_id
                        received += 1
                        used_frame_id = message.get("frame_id")
                        if used_frame_id in sent_at:
                            message["client_frame_to_action_ms"] = round(
                                (time.monotonic() - sent_at[used_frame_id]) * 1000.0,
                                3,
                            )
                        print(json.dumps(message, separators=(",", ":")))
                    else:
                        print(f"[SERVER] {message}")
                elif opcode == 0x9:
                    send_client_frame(connection, 0xA, payload)
                elif opcode == 0x8:
                    break
        print(f"[DONE] frames_sent={sent} actions_received={received}")
        return 0
    finally:
        try:
            send_client_frame(connection, 0x8, struct.pack("!H", 1000))
        except OSError:
            pass
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
