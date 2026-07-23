# GameAgent streaming WebSocket server

The current stage runs a persistent, low-latency WebSocket server. It receives
JPEG frames continuously, decodes them into RGB images, retains only a bounded
window of the newest frames, runs a temporal Sub-Zero mock agent at 20 decisions
per second, and returns command-state vectors through the same connection.

The mock is intentionally isolated behind `MockAgent`. A future BaseAI model can
replace that class without changing the transport, buffering, metrics, or client
protocol.

## Endpoints

- `GET /health`: JSON health check.
- `GET /ws`: WebSocket upgrade and persistent frame/action stream.

The server binds to `0.0.0.0:8080` by default. In RunPod, expose port `8080` as
an HTTP port and connect to:

```text
wss://POD_ID-8080.proxy.runpod.net/ws
```

RunPod terminates TLS at the proxy. The process inside the Pod therefore serves
ordinary WebSocket traffic on port 8080; it does not need a certificate.

## Binary frame protocol (recommended)

Every client-to-server WebSocket binary message contains exactly one JPEG:

| Offset | Size | Encoding | Field |
|---:|---:|---|---|
| 0 | 4 | ASCII | magic `GAF1` |
| 4 | 8 | unsigned big-endian | `frame_id` |
| 12 | 8 | signed big-endian | Unix `timestamp_ms` |
| 20 | 4 | unsigned big-endian | JPEG byte count |
| 24 | N | bytes | JPEG payload |

The complete envelope is equivalent to Python's `struct.pack("!4sQqI", ...)`
followed by the JPEG bytes. WebSocket client frames must use the masking required
by RFC 6455.

For diagnostics, a text WebSocket message is also accepted:

```json
{
  "type": "frame",
  "frame_id": 42,
  "timestamp_ms": 1784740000123,
  "jpeg_base64": "/9j/4AAQSkZJRg..."
}
```

Binary JPEGs avoid Base64's size and CPU overhead and should be used by the real
CatchOS client.

## Action response

Every decision is a text WebSocket message. `commands` remains at the top level,
so the existing GameAgent JSON parser can continue reading it:

```json
{
  "protocol_version": 2,
  "type": "action",
  "action_mode": "state",
  "commands": [7],
  "mock_move": "Congelar",
  "mock_step": 1,
  "mock_steps": 5,
  "mock_step_duration_ms": 50,
  "decision_id": 120,
  "frame_id": 417,
  "frame_ids": [410, 411, 412, 413, 414, 415, 416, 417],
  "timestamp_ms": 1784740000170,
  "control_interval_ms": 50,
  "failsafe_timeout_ms": 200,
  "metrics": {
    "frame_latency_ms": 47,
    "server_frame_age_ms": 18,
    "inference_ms": 0.018,
    "buffer_size": 32,
    "frames_received": 417,
    "frames_dropped": 385,
    "decode_errors": 0,
    "client_message_errors": 0,
    "acknowledgements_received": 119,
    "acknowledgements_rejected": 0,
    "applied_action_history_size": 8,
    "last_applied_decision_id": 119
  }
}
```

Protocol version 2 defines `commands` as the complete controller state. It is
not a list of taps: listed commands remain pressed until a newer decision
replaces the state. An empty array is an intentional neutral action and releases
all controls. A client must also release all controls after the advertised
failsafe timeout or immediately on disconnect.

After applying every valid state, the client confirms what was actually applied
with a text WebSocket message:

```json
{
  "protocol_version": 2,
  "type": "action_ack",
  "decision_id": 120,
  "applied_commands": [9, 3],
  "applied_timestamp_ms": 1784740000182
}
```

The server validates the ID against actions sent on the same connection, records
whether the applied set matches the requested set, and keeps the newest confirmed
states in an in-memory action history. The agent receives this confirmed history
alongside the decoded frame observation. No disk access is involved.

The mock chooses one of six Sub-Zero moves at random and completes its entire
state sequence before choosing another move. At the default 50 ms decision
interval, every row below is one WebSocket action. `[]` releases all controls;
it is deliberately used between repeated button presses and between moves.

With Sub-Zero facing right, the exact sequences are:

| Move | WebSocket `commands` states |
|---|---|
| Congelar (`D, F, B`) | `[7] → [9] → [2] → [] → []` |
| Clone de Gelo (`D, B, B`) | `[7] → [8] → [2] → [] → []` |
| Chuva de Gelo — Frente (`D, F, B, Y`) | `[7] → [9] → [2] → [0] → [] → []` |
| Chuva de Gelo — Perto (`D, B, F, Y`) | `[7] → [8] → [9] → [0] → [] → []` |
| 3 Golpes (`Y, Y, D+B`) | `[0] → [] → [0] → [] → [7,2] → [] → []` |
| 4 Golpes (`X, X, B+X, F+A`) | `[1] → [] → [1] → [] → [8,1] → [] → [9,3] → [] → []` |

Here `B` in the final position of a special move means the B/low-punch button
(`2`); `B` used as a direction means Back. A plus sign means simultaneous
controls in one vector. Commas mean consecutive 50 ms states.

The command mapping remains:

| Value | Command |
|---:|---|
| 0 | Y |
| 1 | X |
| 2 | B |
| 3 | A |
| 4 | L |
| 5 | R |
| 6 | Up |
| 7 | Down |
| 8 | Left |
| 9 | Right |

## Buffer and inference behavior

- The circular buffer holds 32 decoded frames by default.
- When full, inserting a new frame removes the oldest one immediately.
- Duplicate or out-of-order `frame_id` values are rejected as stale.
- The inference loop is independent of frame reception and wakes every 50 ms.
- It selects up to the newest 8 frames and returns one action immediately.
- A disconnect stops only that client's inference loop. The listening process
  remains alive and waits for the next WebSocket connection.

Each action and connection lifecycle event is appended to a JSON Lines metrics
file. Decision records contain frame latency, inference time, discarded frames,
commands, and the exact frame IDs used. `frame_latency_ms` uses the timestamp
sent by the client and therefore requires synchronized client/server clocks.
`server_frame_age_ms` uses only the RunPod clock and reliably measures how long
the selected frame waited between server reception and the decision. Records are flushed in small
batches (at most one second at 20 Hz) to keep persistent-volume I/O off the
critical path; connection and disconnection events are flushed immediately.

JPEG persistence is disabled by default in both the executable and the RunPod
helper. The production data path is therefore only: receive compressed JPEG,
decode once into an RGB image, put that decoded image in the in-memory circular
buffer, and pass references to those images to the agent. No file is written and
no image is re-encoded.

For an explicit debug run, set `GAMEAGENT_SAVE_FRAMES=1`. The original compressed
JPEG received from the client is then written to disk only after its decoded
image has been accepted into the circular buffer. Only the first valid JPEG is
saved for each WebSocket connection; every later frame remains memory-only.
Each capture directory also contains a one-line `frames.jsonl` mapping the
sample filename to its frame ID and timestamps.

## Configuration

| Environment variable | Default | Meaning |
|---|---:|---|
| `GAMEAGENT_HOST` | `0.0.0.0` | bind address |
| `GAMEAGENT_PORT` | `8080` | server port |
| `GAMEAGENT_FRAME_BUFFER_CAPACITY` | `32` | decoded frames retained |
| `GAMEAGENT_OBSERVATION_FRAMES` | `8` | newest frames used by the agent |
| `GAMEAGENT_ACTION_HISTORY_SIZE` | `8` | newest client-confirmed states given to the agent |
| `GAMEAGENT_DECISION_INTERVAL_MS` | `50` | inference loop interval |
| `GAMEAGENT_FAILSAFE_TIMEOUT_MS` | `200` | client safety release timeout advertised in every action |
| `GAMEAGENT_MAX_JPEG_MB` | `8` | maximum JPEG payload |
| `GAMEAGENT_METRICS_PATH` | `metrics.jsonl` | JSONL metrics output |
| `GAMEAGENT_CAPTURE_DIR` | `captures` | optional JPEG inspection output |
| `GAMEAGENT_SAVE_FRAMES` | `0` | save only the first valid JPEG from each connection; zero keeps everything in memory |
| `GAMEAGENT_SUBZERO_FACING` | `right` | maps Forward/Back for a character facing `right` or `left` |
| `GAMEAGENT_API_TOKEN` | unset | Bearer or X-API-Key credential |
| `GAMEAGENT_RANDOM_SEED` | random | deterministic mock seed |

`GAMEAGENT_OBSERVATION_FRAMES` cannot exceed the buffer capacity.

## Build and test locally or in the Pod

BaseAI is expected at `../BaseAI` locally. In the Pod, the helper expects
`/tmp/GameAgent` and `/tmp/BaseAI`:

```bash
cd /tmp/GameAgent
bash scripts/build_runpod.sh
```

The build runs the old batch-protocol compatibility tests plus the new binary
envelope, RFC 6455 handshake, WebSocket framing, and circular-buffer tests.

## Start on RunPod

Create a persistent token once:

```bash
mkdir -p /workspace/gameagent
umask 077
openssl rand -hex 32 > /workspace/gameagent/api_token
```

Start the process in the background:

```bash
cd /tmp/GameAgent

export GAMEAGENT_API_TOKEN="$(< /workspace/gameagent/api_token)"

nohup env \
  GAMEAGENT_API_TOKEN="${GAMEAGENT_API_TOKEN}" \
  bash scripts/run_runpod_server.sh \
  > /workspace/gameagent/server.log 2>&1 &

echo $!
```

Monitor server events and structured metrics:

```bash
tail -f /workspace/gameagent/server.log
tail -f /workspace/gameagent/metrics.jsonl
```

When `GAMEAGENT_SAVE_FRAMES=1`, captured frames are stored under:

```text
/workspace/gameagent/captures/session_PROCESS_ID/connection_N/
├── frame_FRAME_ID.jpg
└── frames.jsonl
```

Normal production launch (no disk writes for frames):

```bash
GAMEAGENT_SAVE_FRAMES=0 bash scripts/run_runpod_server.sh
```

Debug capture, saving only one sample JPEG per WebSocket connection:

```bash
GAMEAGENT_SAVE_FRAMES=1 bash scripts/run_runpod_server.sh
```

Health check from another machine:

```bash
curl https://POD_ID-8080.proxy.runpod.net/health
```

The `/ws` upgrade accepts either of these headers:

```http
Authorization: Bearer YOUR_TOKEN
```

```http
X-API-Key: YOUR_TOKEN
```

For a short test only, `GAMEAGENT_ALLOW_UNAUTHENTICATED=1` disables the script's
token requirement. The RunPod proxy URL is public, so this is unsafe for normal
operation.

## End-to-end diagnostic client

The included client has no third-party Python dependencies. It sends a chosen
JPEG repeatedly using the binary protocol and prints action messages. It adds
`client_frame_to_action_ms` locally using the Mac's monotonic clock; unlike an
absolute timestamp comparison, this measures frame-to-action round-trip latency
without requiring the Mac and Pod clocks to agree.

It also sends an `action_ack` for every validated action, using the state applied
by its local protocol-v2 state machine.

```bash
export GAMEAGENT_API_TOKEN='YOUR_TOKEN'

python3 scripts/send_websocket_frames.py \
  --url wss://POD_ID-8080.proxy.runpod.net/ws \
  --jpeg /path/to/frame.jpg \
  --fps 15 \
  --frames 300
```

Fifteen input frames per second is the recommended starting point for gameplay:
successive samples are about 67 ms apart, an 8-frame observation covers roughly
0.47 seconds, and the independent 20 Hz action loop remains responsive without
requiring two nearly identical JPEGs for every decision.

For a server on the same machine, replace the URL with
`ws://127.0.0.1:8080/ws`.
