# GameAgent streaming WebSocket server

The current stage runs a persistent, low-latency WebSocket server. It receives
JPEG frames continuously, decodes them into RGB images, retains only a bounded
window of the newest frames, runs a temporal Sub-Zero mock agent at 20 decisions
per second, and returns command-state vectors through the same connection.

The mock is intentionally isolated behind `MockAgent`. A future BaseAI model can
replace that class without changing the transport, buffering, metrics, or client
protocol.

## Frozen ResNet50 visual features

`ResNet50FeatureExtractor` reconstructs the validated ImageNet-1000 ResNet50 in
evaluation mode, loads all 320 model tensors from `resnet50_best.ckpt`, and
verifies the FP32 device copies exactly. The wrapper does not construct an
optimizer or GradScaler and never executes backward; the current underlying
BaseAI ResNet50 allocation is reused unchanged.

Each RGB frame is bilinearly resized to 224x224, converted to CHW `[0,1]`, and
normalized with the same ImageNet mean and standard deviation used during
training. The extractor returns the 2048 values after global average pooling
and before the ImageNet classifier:

```text
RGB frame
→ resize 224x224
→ ImageNet normalization
→ frozen ResNet50 backbone
→ global average pooling
→ 2048-dimensional FP16 device vector
```

`extract_device()` is the zero-copy interface intended for the future GPU
temporal model. `extract()` downloads the same vector as FP32 for validation
and the current GameAgent prototype.

To validate the checkpoint against every JPEG recorded by debug mode and
materialize LSTM-ready feature files:

```bash
cd /tmp/GameAgent

GAMEAGENT_RESNET50_CHECKPOINT=/workspace/resnet50_best.ckpt \
GAMEAGENT_CAPTURE_ROOT=/workspace/gameagent/captures \
GAMEAGENT_FEATURE_OUTPUT_DIR=/workspace/gameagent/resnet50_features \
bash scripts/run_resnet50_feature_test.sh
```

The command creates a timestamped output directory containing:

```text
features.jsonl
summary.json
feature_000000.f32
feature_000001.f32
...
```

Every `.f32` file contains exactly 2048 native float32 values. The JSONL
manifest preserves `sequence_id`, `frame_id`, client/server timestamps, source
image path, feature path, shape, inference time and numerical statistics.
Records are ordered by the original client timestamp so later temporal windows
can be formed without relying on lexicographic JPEG filenames.

## Temporal LSTM prototype

GameAgent now contains an inference-only GPU temporal path that can be enabled
without changing the WebSocket protocol:

```text
new decoded frame
→ frozen ResNet50 once
→ cache FP16[2048] on the GPU
→ project 2048 → 256
→ concatenate:
   - 10-value client-confirmed controller state
   - acknowledgement-known flag
   - normalized age of that state
→ LSTM hidden size 256 over the newest 8 cached steps
→ temporal hidden state for the future action head
```

The ResNet50 is not repeated for the complete window on every decision. Each
new frame is embedded once and the bounded GPU ring retains only the newest
features. The controller input comes from `action_ack`, so the temporal model
sees what the client actually applied rather than assuming every requested
state succeeded.

Without a temporal checkpoint, the LSTM uses deterministic initialization only.
With a trained checkpoint, the student predicts in shadow mode while controller
actions still come from the Sub-Zero teacher. The optional temporal metrics
report the sequence, aligned frame/ACK, hidden-state norm, inference time,
student prediction and errors. Student control can therefore be enabled later
without changing transport or acknowledgement semantics. `infer_device()`
already exposes the final FP32 hidden vector on the same CUDA stream; the
current host copy exists only for tests and diagnostic metrics.

## Sub-Zero curriculum and behavior cloning

The teacher now samples a weighted curriculum:

- 50% conditional plans: Freeze, observe, approach and complete a three- or
  four-hit combo;
- 30% known isolated Sub-Zero moves;
- 15% structured combat variations: forward jump/high kick, crouching
  uppercut, and running approaches followed by three- or four-hit combos;
- 5% bounded exploration containing no contradictory directions and explicit
  release states.

The jump and running variations mirror Forward/Back automatically when the
fighter changes side. Every movement and attack is represented by complete
controller states at the 50 ms decision cadence, including explicit releases,
so the existing protocol-v2 client requires no change.

After Freeze, the teacher compares a conservative opponent-side ice-color score
with the pre-move baseline. A clear miss branches to block/retreat; a confirmed
or visually ambiguous result continues through approach and combo. Every
decision records `curriculum_source`, `curriculum_phase`,
`curriculum_branch`, `freeze_visual_score` and `freeze_visual_delta`, so
ambiguous heuristic labels remain auditable.

The first learner is behavior cloning, not reinforcement learning. It joins
timestamped ResNet50 features with the latest client-confirmed controller state
and learns the teacher's next complete 10-button state with
`BCEWithLogitsLoss`. The trainable model is:

```text
8 x FP32[2048] visual features
→ tanh visual projection (256)
+ 10 confirmed buttons + ACK-known + command age
→ one-layer LSTM (hidden 256)
→ 10-logit multi-label action head
```

The trainer uses AdamW, gradient clipping, session-level train/validation
splits, positive-class balancing and checkpoint resume. It writes:

```text
/workspace/gameagent/checkpoints/
├── subzero_temporal_latest.pt   # model + AdamW + epoch/step; resumable
├── subzero_temporal_best.pt
├── subzero_temporal_latest.gat  # compact C++/CUDA inference weights
├── subzero_temporal_best.gat
└── subzero_dataset_summary.json
```

The server loads `.gat` weights in student-shadow mode. The teacher still
controls the client, while `student_shadow_commands`, probabilities and exact
match are logged for safe evaluation.

Run the standalone production-shape test:

```bash
cd /tmp/GameAgent

GAMEAGENT_RESNET50_CHECKPOINT=/workspace/resnet50_best.ckpt \
bash scripts/run_temporal_lstm_test.sh
```

It verifies the complete checkpoint, `512x448 → ResNet50 → LSTM`, use of the
latest confirmed command, command sensitivity, an 8-step circular feature
buffer and deterministic reset.

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
    "last_applied_decision_id": 119,
    "temporal_model_enabled": true,
    "temporal_model_ready": true,
    "temporal_model_errors": 0,
    "temporal_lstm_ms": 1.32,
    "temporal_hidden_l2": 3.04,
    "temporal_sequence_size": 8,
    "temporal_frame_id": 417,
    "temporal_applied_decision_id": 119,
    "temporal_applied_command_known": true
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

The teacher samples the weighted Sub-Zero curriculum and completes the selected
state sequence or conditional recovery branch before choosing another plan. At
the default 50 ms decision
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
| `GAMEAGENT_PUBLIC_POD_ID` | detected from RunPod | optional pod ID override used to print the public proxy URL |
| `GAMEAGENT_PUBLIC_WEBSOCKET_URL` | derived from pod ID | optional complete public WebSocket URL override |
| `GAMEAGENT_PUBLIC_HEALTH_URL` | derived from pod ID | optional complete public health URL override |
| `GAMEAGENT_FRAME_BUFFER_CAPACITY` | `32` | decoded frames retained |
| `GAMEAGENT_OBSERVATION_FRAMES` | `8` | newest frames used by the agent |
| `GAMEAGENT_ACTION_HISTORY_SIZE` | `8` | newest client-confirmed states given to the agent |
| `GAMEAGENT_DECISION_INTERVAL_MS` | `50` | inference loop interval |
| `GAMEAGENT_FAILSAFE_TIMEOUT_MS` | `200` | client safety release timeout advertised in every action |
| `GAMEAGENT_MAX_JPEG_MB` | `8` | maximum JPEG payload |
| `GAMEAGENT_METRICS_PATH` | `metrics.jsonl` | JSONL metrics output |
| `GAMEAGENT_CAPTURE_DIR` | `captures` | optional JPEG inspection output |
| `GAMEAGENT_SAVE_FRAMES` | `0` | save only the first valid JPEG from each connection; zero keeps everything in memory |
| `GAMEAGENT_COLLECT_TRAINING_DATA` | `0` | explicitly save every accepted JPEG and aligned ACK metadata for offline training |
| `GAMEAGENT_TRAINING_CAPTURE_DIR` | `/workspace/gameagent/training_captures` | full-session training capture root |
| `GAMEAGENT_SUBZERO_FACING` | `right` | maps Forward/Back for a character facing `right` or `left` |
| `GAMEAGENT_API_TOKEN` | unset | Bearer or X-API-Key credential |
| `GAMEAGENT_RANDOM_SEED` | random | deterministic mock seed |
| `GAMEAGENT_TEMPORAL_MODEL` | `0` | enable frozen ResNet50 plus GPU LSTM feature path |
| `GAMEAGENT_RESNET50_CHECKPOINT` | `/workspace/resnet50_best.ckpt` | frozen ImageNet ResNet50 checkpoint |
| `GAMEAGENT_TEMPORAL_CHECKPOINT` | unset | trained `.gat` policy loaded in student-shadow mode |
| `GAMEAGENT_DEVICE_ID` | `0` | CUDA device used by ResNet50 and LSTM |
| `GAMEAGENT_LSTM_PROJECTION_SIZE` | `256` | ResNet50 feature projection width |
| `GAMEAGENT_LSTM_HIDDEN_SIZE` | `256` | temporal hidden-state width |

`GAMEAGENT_OBSERVATION_FRAMES` cannot exceed the buffer capacity.

## Build and test locally or in the Pod

BaseAI is expected at `../BaseAI` locally. In the Pod, the helper expects
`/tmp/GameAgent` and `/tmp/BaseAI`:

```bash
cd /tmp/GameAgent
bash scripts/build_runpod.sh
```

On a fresh RunPod image, the build helper checks for CMake 3.28 or newer. If
the system CMake is older, it installs a compatible CMake in
`$HOME/.local/bin` through Python pip and consistently uses the matching
`ctest`. Set `GAMEAGENT_CMAKE_BIN` to override that executable explicitly.

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

Enable the validated temporal feature path while retaining mock actions:

```bash
GAMEAGENT_TEMPORAL_MODEL=1 \
GAMEAGENT_RESNET50_CHECKPOINT=/workspace/resnet50_best.ckpt \
GAMEAGENT_SAVE_FRAMES=0 \
bash scripts/run_runpod_server.sh
```

Collect a complete curriculum session explicitly:

```bash
GAMEAGENT_TEMPORAL_MODEL=1 \
GAMEAGENT_COLLECT_TRAINING_DATA=1 \
GAMEAGENT_TRAINING_CAPTURE_DIR=/workspace/gameagent/training_captures \
GAMEAGENT_SAVE_FRAMES=0 \
bash scripts/run_runpod_server.sh
```

Normal gameplay keeps `GAMEAGENT_COLLECT_TRAINING_DATA=0`; full JPEG persistence
is only active under this training flag.

Convert a collected session to features and train/checkpoint the student:

```bash
GAMEAGENT_PRETRAIN_EPOCHS=30 \
GAMEAGENT_PRETRAIN_BATCH_SIZE=64 \
bash scripts/run_subzero_training_pipeline.sh
```

Resume for another 20 epochs:

```bash
GAMEAGENT_PRETRAIN_RESUME=1 \
GAMEAGENT_PRETRAIN_EPOCHS=20 \
bash scripts/run_subzero_pretraining.sh
```

Load the best trained student in safe shadow mode:

```bash
GAMEAGENT_TEMPORAL_MODEL=1 \
GAMEAGENT_TEMPORAL_CHECKPOINT=/workspace/gameagent/checkpoints/subzero_temporal_best.gat \
GAMEAGENT_SAVE_FRAMES=0 \
bash scripts/run_runpod_server.sh
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
