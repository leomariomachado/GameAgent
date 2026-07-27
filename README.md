# GameAgent streaming WebSocket server

The current stage runs a persistent, low-latency WebSocket server. It receives
JPEG frames continuously, decodes them into RGB images, retains only a bounded
window of the newest frames, and returns command-state vectors through the same
connection. A character-specific scripted teacher or online reinforcement
learner can control active fight frames.

The mock remains isolated behind `MockAgent`; enabling reinforcement learning
does not change transport, buffering, acknowledgement, or client controller
semantics.

The available profiles are Sub-Zero, Scorpion, Noob Saibot, and Sindel. At
connection start, the client identifies the current fighter with a protocol-v2
`session_config`; old clients that send a frame first continue to default to
Sub-Zero. All four profiles use the same architecture. Each character owns an
independent epsilon schedule and action head, while all profiles train one
shared strategy trunk, embedding tables, replay and latest checkpoint.

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
→ one recurrent step per 50 ms decision tick
→ LSTM hidden size 256 over the newest 8 decision steps
→ active/release head + 10-button action head
```

The ResNet50 is not repeated for the complete window on every decision. A
latest-wins worker embeds each selected new frame once. Every 50 ms decision
then reuses the newest cached visual vector and advances the bounded GPU
decision ring. The controller input comes from `action_ack`, so the temporal model
sees what the client actually applied rather than assuming every requested
state succeeded.

Without a temporal checkpoint, the LSTM uses one deterministic initialization
shared by every character. This is the clean default: only the ImageNet
ResNet50 is pretrained, and the shared RL policy starts from scratch. An
optional `.gat` temporal checkpoint can still be supplied for experiments, but
it is not required by reinforcement mode.

## Uniform character curricula

Every profile is built by the same pipeline:

1. load only the declared special moves and combos for that character;
2. add an equal-weight Forward/Back mirror for each unique directional move;
3. preserve explicit neutral states between repeated taps;
4. combine simultaneous controls such as `B+X` in one state vector;
5. add the same 11 generic primitive options for reinforcement learning.

Sub-Zero no longer receives conditional freeze follow-ups, hand-authored
running plans, extra exploration programs, or a pretrained temporal policy.
Its six declared base moves produce nine unique scripted variants after
mirroring. Noob Saibot, Scorpion, and Sindel produce 7, 11, and 12 scripted
variants respectively. Different counts reflect their declared move lists, not
different learning privileges.

## Online reinforcement learning

Reinforcement mode uses the frozen ResNet50 and deterministic temporal encoder
as the state encoder for a shared semi-Markov Double DQN. It adds the
controlled and opponent health fractions, health advantage, HUD confidence,
the latest client-confirmed 10-button state and ACK-known flag to the 256-value
LSTM hidden state.

The state is concatenated with learned 16-value embeddings for the controlled
character and opponent. A shared hidden trunk learns spacing, defense, timing
and fight-state strategy from every character. Each controlled character has a
small action head so specials with different meanings never collide. The
selected head chooses among high-level options:

- bounded primitive holds such as move, block, jump, punch and kick;
- that character's declared specials and combos;
- the unique Forward/Back mirrors of those programs.

An option emits one complete protocol-v2 controller state every 50 ms. Its
discounted reward is accumulated until the option ends, and replay stores its
actual duration. Double-DQN targets therefore use `gamma^duration`, assigning
delayed game damage to the complete move instead of an arbitrary controller
tick.

Exploration is hierarchical. Epsilon first decides exploration versus DQN
exploitation. During exploration, `GAMEAGENT_RL_SCRIPTED_EXPLORATION` chooses
between a valid character-specific sequence and a bounded primitive action.
The default is a 50/50 split. Epsilon decays from 1.0 to 0.05 over 100,000
option selections independently for each character. Replay is shared and
sampled uniformly across non-empty player/opponent matchups so a frequently
used character cannot completely suppress rarer matchups.

### Fight gate and reward

A deterministic detector measures the two health bars in their normalized HUD
regions. It distinguishes these phases:

```text
no_hud -> acquiring -> round_intro -> active_fight -> round_end
```

Only `active_fight` frames are sent through ResNet50/LSTM and allowed to produce
non-empty commands. Menus, stale frames, save-state loads, health refills,
round intros and round ends emit the empty controller state. Health increases
close the old episode and never count as healing reward.

During active play:

```text
reward = 15 * damage_dealt - 11 * damage_taken
round win  = +5
round loss = -5
```

Damage and episode flags accumulate until the 20 Hz decision loop consumes
them, so one camera frame cannot be counted twice. Terminal bonuses are emitted
once. `action_ack` remains the source of truth for commands actually applied.

### Checkpoints and resume

RL format-v4 checkpoints include the online/target shared trunk, learned
embeddings, all four action heads, Adam moments, per-character epsilon and
episode counters, plus matchup-labelled replay memory. They are written
atomically at round end and connection shutdown:

```text
/workspace/gameagent/rl_checkpoints/
├── shared_policy_latest.garl
├── shared_policy_best_subzero.garl
├── shared_policy_best_scorpion.garl
├── shared_policy_best_noob_saibot.garl
└── shared_policy_best_sindel.garl
```

The only required pretrained artifact is:

```text
/workspace/resnet50_best.ckpt
```

Start or resume online training:

```bash
cd /tmp/GameAgent

export GAMEAGENT_API_TOKEN="$(< /workspace/gameagent/api_token)"
GAMEAGENT_RL_RESUME=1 bash scripts/run_character_rl.sh
```

The client then selects the character and opponent in `session_config`. A
different character loads the same shared latest checkpoint and selects its own
head. To start the complete shared policy from scratch in an empty directory:

```bash
GAMEAGENT_RL_RESUME=0 \
GAMEAGENT_RL_CHECKPOINT_DIR=/workspace/gameagent/rl_checkpoints_clean \
bash scripts/run_character_rl.sh
```

Evaluation uses the same checkpoint and disables updates:

```bash
GAMEAGENT_RL_TRAINING=0 bash scripts/run_character_rl.sh
```

Frame persistence remains off by default in RL mode. Health detection, visual
features and decisions operate directly on the decoded in-memory frame.

The character RL launcher uses a 100 ms decision interval (10 controller states
per second) and advertises a 750 ms client failsafe. This matches a 10 FPS
client without changing the generic server defaults. The fight gate still
releases controls when the newest frame is older than three decision intervals,
so the longer client failsafe does not authorize decisions from stale images.
Both values remain configurable:

```bash
GAMEAGENT_DECISION_INTERVAL_MS=100 \
GAMEAGENT_FAILSAFE_TIMEOUT_MS=750 \
bash scripts/run_character_rl.sh
```

The server prints one accumulated learning summary every five seconds without
running an extra inference or synchronizing the GPU again:

```text
[RL_REWARD] reward_gained=0.42 reward_accumulated=1.87 reward_terminal=0 result=ongoing connection=1 episode=4 phase=active_fight damage_dealt=0.08 damage_taken=0.03 hp=0.71/0.54 epsilon=0.93 replay=418 updates=162 loss=0.021
```

An episode ending forces an immediate summary instead of waiting for the next
reporting interval. `reward_terminal` is `+5` for a win, `-5` for a loss and
zero otherwise. `reward_gained` is the reward since the previous summary, while
`reward_accumulated` is the complete episode return, including damage rewards,
damage penalties and the terminal reward. A blank line follows each summary to
make reward events easier to scan.

After `round_lost`, the server releases controls and emits one
`training_reset_state` request with a connection-local `reset_id`. The client
loads its configured training save state and returns `training_reset_ack`.
Wins and double KOs do not trigger this request. This behavior is enabled by
default and can be disabled with `GAMEAGENT_TRAINING_RESET_STATE=0`.

Change only the reporting cadence, in milliseconds, with:

```bash
GAMEAGENT_RL_REWARD_LOG_INTERVAL_MS=3000 bash scripts/run_character_rl.sh
```

Run the standalone production-shape test:

```bash
cd /tmp/GameAgent

GAMEAGENT_RESNET50_CHECKPOINT=/workspace/resnet50_best.ckpt \
bash scripts/run_temporal_lstm_test.sh
```

It verifies the complete checkpoint, `512x448 → ResNet50 → LSTM`, use of the
   latest confirmed command, command sensitivity, an 8-decision circular
   sequence reusing one cached frame, checkpoint v1/v2 loading and deterministic
   reset.

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

## Character selection

Immediately after connecting and before sending the first frame, send:

```json
{
  "protocol_version": 2,
  "type": "session_config",
  "player_character": "noob_saibot",
  "opponent_character": "scorpion",
  "player_side": "left",
  "initial_facing": "right"
}
```

Supported `player_character` values are `subzero`, `scorpion`,
`noob_saibot`, and `sindel`.

`opponent_character` selects one row from a learned embedding table and accepts:

```text
unknown, cyrax, ermac, jade, jax, kabal, kano, kitana, kung_lao,
liu_kang, mileena, motaro, nightwolf, noob_saibot, rain, reptile,
scorpion, sektor, shang_tsung, shao_kahn, sheeva, sindel, smoke,
human_smoke, sonya, stryker, subzero, classic_subzero
```

`unknown` is used when the field is omitted. `sheeva` is retained for arcade
UMK3 compatibility even though she is absent from the SNES port.
`player_side` and `initial_facing` are optional and accept `left` or `right`.
The client must wait for `session_config_ack` before transmitting frames.
Changing either fighter identity requires a reconnect; a later
`session_config` is rejected. If the first data message is already a frame, the
server preserves legacy behavior and selects Sub-Zero versus `unknown` without
an acknowledgement.

The acknowledgement reports the canonical character slug, selected profile,
canonical opponent slug/name and embedding index, number of scripted curriculum
variants, effective side/facing, active agent mode, and whether a compatible
checkpoint is loaded. Unsupported opponent values produce
`session_config_error` with `supported_opponent_characters`.

The latest policy checkpoint is shared across every controlled fighter and
matchup. For example, Sub-Zero experience updates the common strategy used by
Sindel, while their specials remain in distinct action heads. Opponent
embeddings allow the same policy to learn different behavior against Scorpion,
Motaro and Shao Kahn. Reinforcement checkpoints use format version 4.

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
  "player_character": "subzero",
  "opponent_character": "scorpion",
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

The teacher samples the selected character's weighted curriculum and completes
the selected state sequence before choosing another plan. At
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

Scorpion, Noob Saibot, and Sindel use the moves listed in their source
specification, plus an equal-weight Forward/Back mirror for every unique
directional program. Repeated directions contain an explicit neutral state
between taps, aerial variants begin with Up states, and combinations such as
`B+X` or `D+Y` are emitted atomically in one `commands` vector. Their current
curriculum sizes are respectively 11, 7, and 12 variants.

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
- ResNet50 feature extraction runs on a latest-wins background worker. If a
  newer frame arrives while another is pending, the obsolete pending frame is
  replaced instead of creating latency.
- Optional JPEG persistence runs on a separate bounded queue. Disk/network
  volume latency never blocks WebSocket ACK reception or the decision loop.
- A disconnect stops only that client's inference loop. The listening process
  remains alive and waits for the next WebSocket connection.

Each action and connection lifecycle event is appended to a JSON Lines metrics
file. Decision records contain frame latency, inference time, discarded frames,
commands, and the exact frame IDs used. It also records temporal frames
processed/overwritten and persistence frames saved/dropped/errors.
`frame_latency_ms` uses the timestamp
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
image has been accepted into the circular buffer. The first valid JPEG is saved
immediately and later samples are limited by
`GAMEAGENT_DEBUG_CAPTURE_INTERVAL_MS`, which defaults to 3000 ms. All frames
between samples remain memory-only. Each capture directory also contains a
`frames.jsonl` mapping each sample filename to its frame ID and timestamps.

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
| `GAMEAGENT_PERSISTENCE_QUEUE_CAPACITY` | `256` | bounded asynchronous JPEG write queue; oldest pending write is dropped on overflow |
| `GAMEAGENT_DECISION_INTERVAL_MS` | `50` | inference loop interval |
| `GAMEAGENT_FAILSAFE_TIMEOUT_MS` | `200` | client safety release timeout advertised in every action |
| `GAMEAGENT_MAX_JPEG_MB` | `8` | maximum JPEG payload |
| `GAMEAGENT_METRICS_PATH` | `metrics.jsonl` | JSONL metrics output |
| `GAMEAGENT_CAPTURE_DIR` | `captures` | optional JPEG inspection output |
| `GAMEAGENT_SAVE_FRAMES` | `0` | enable periodic debug JPEG samples; zero keeps everything in memory |
| `GAMEAGENT_DEBUG_CAPTURE_INTERVAL_MS` | `3000` | minimum time between debug JPEG samples |
| `GAMEAGENT_COLLECT_TRAINING_DATA` | `0` | enqueue accepted JPEGs and aligned ACK metadata for offline training without blocking reception |
| `GAMEAGENT_TRAINING_CAPTURE_DIR` | `/workspace/gameagent/training_captures` | full-session training capture root |
| `GAMEAGENT_DEFAULT_FACING` | `right` | legacy/default Forward/Back mapping when the client omits `session_config` |
| `GAMEAGENT_API_TOKEN` | unset | Bearer or X-API-Key credential |
| `GAMEAGENT_RANDOM_SEED` | random | deterministic mock seed |
| `GAMEAGENT_TEMPORAL_MODEL` | `0` | enable frozen ResNet50 plus GPU LSTM feature path |
| `GAMEAGENT_RESNET50_CHECKPOINT` | `/workspace/resnet50_best.ckpt` | frozen ImageNet ResNet50 checkpoint |
| `GAMEAGENT_TEMPORAL_CHECKPOINT` | unset | trained `.gat` policy loaded in student-shadow mode |
| `GAMEAGENT_DEVICE_ID` | `0` | CUDA device used by ResNet50 and LSTM |
| `GAMEAGENT_LSTM_PROJECTION_SIZE` | `256` | ResNet50 feature projection width |
| `GAMEAGENT_LSTM_HIDDEN_SIZE` | `256` | temporal hidden-state width |
| `GAMEAGENT_REINFORCEMENT_LEARNING` | `0` | enable shared multi-character Double DQN |
| `GAMEAGENT_RL_CHECKPOINT_DIR` | `/workspace/gameagent/rl_checkpoints` | shared latest policy/replay plus per-character best snapshots |
| `GAMEAGENT_TRAINING_RESET_STATE` | `1` | request a client-side save-state load once after each `round_lost` |

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
is only active under this training flag. If the persistent volume cannot keep
up, the oldest pending disk write is dropped and
`persistence_frames_dropped` is incremented; live control remains prioritized.

The previous Sub-Zero-only behavior-cloning checkpoints are not part of the
runtime path. Fresh reinforcement training needs only the shared ResNet50;
character-specific learning state is created automatically on first use.

Debug capture, saving one JPEG immediately and then at most one every 3 seconds:

```bash
GAMEAGENT_SAVE_FRAMES=1 \
GAMEAGENT_DEBUG_CAPTURE_INTERVAL_MS=3000 \
bash scripts/run_runpod_server.sh
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
