#!/usr/bin/env python3

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send the same JPEG as a 30-frame window to GameAgent."
    )
    parser.add_argument("--jpeg", required=True, type=Path, help="JPEG file to send")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8080/commands",
        help="GameAgent mock endpoint",
    )
    parser.add_argument("--emulator", default="zsnes")
    parser.add_argument("--sample-fps", type=int, default=10)
    parser.add_argument(
        "--api-token",
        default=os.environ.get("GAMEAGENT_API_TOKEN"),
        help="Bearer token; defaults to GAMEAGENT_API_TOKEN",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jpeg = args.jpeg.read_bytes()
    if len(jpeg) < 4 or jpeg[:2] != b"\xff\xd8":
        raise SystemExit(f"Not a JPEG file: {args.jpeg}")
    if args.sample_fps <= 0:
        raise SystemExit("--sample-fps must be positive")

    encoded = base64.b64encode(jpeg).decode("ascii")
    now_ms = int(time.time() * 1000)
    interval_ms = round(1000 / args.sample_fps)
    frames = [
        {
            "timestamp_ms": now_ms + index * interval_ms,
            "jpeg_base64": encoded,
        }
        for index in range(30)
    ]
    payload = {
        "protocol_version": 1,
        "emulator": args.emulator,
        "window_seconds": 30 / args.sample_fps,
        "sample_fps": args.sample_fps,
        "frames": frames,
    }

    headers = {"Content-Type": "application/json"}
    if args.api_token:
        headers["Authorization"] = f"Bearer {args.api_token}"

    request = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
            print(json.dumps(body, indent=2))
    except urllib.error.HTTPError as error:
        print(error.read().decode("utf-8"))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
