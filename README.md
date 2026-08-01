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

## ResNet50 visual features

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
→ ResNet50 backbone
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

## Trainable ResNet50 and temporal LSTM

The reinforcement path now trains the temporal model together with the
Double-DQN on CUDA, without changing the WebSocket protocol:

```text
new decoded frame
→ ResNet50 once
→ cache/project its 2048 visual values
→ fixed projection 2048 → 256
→ concatenate:
   - 10-value client-confirmed controller state
   - acknowledgement-known flag
   - normalized age of that state
→ one recurrent step per 100 ms decision tick
→ trainable LSTM hidden size 256 over the newest 16 decision steps
→ shared strategy trunk + character-specific Q head
```

The ResNet50 is not repeated for the complete window on every decision. A
latest-wins worker embeds each selected new frame once. Every decision then
reuses the newest cached visual vector and advances the bounded recurrent
decision ring. The controller input comes from `action_ack`, so the temporal model
sees what the client actually applied rather than assuming every requested
state succeeded.

The character RL launcher also fine-tunes ResNet50 `layer4` by default. Earlier
stages, every BatchNorm running statistic and the dimensionality-reduction
projection stay frozen. One visual mini-batch is trained every 16 recurrent
updates, using the compressed JPEG retained in the bounded in-memory replay.
This prevents image persistence and repeated JPEG re-encoding from entering the
decision path.

Backpropagation through time updates the LSTM, shared strategy trunk,
character/opponent embeddings and all four Q heads in GPU batches. The replay
stores bounded sequences and never crosses an episode boundary. The older
inference-only `.gat` path remains available when trainable RL is disabled.
The generic server launcher keeps ResNet fine-tuning disabled unless
`GAMEAGENT_RL_TRAINABLE_RESNET=1` is explicitly set.

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

Reinforcement mode uses the optionally fine-tuned ResNet50 and trainable
temporal encoder as the state encoder for a shared recurrent semi-Markov Double
DQN. It adds the controlled and opponent health fractions, health advantage,
HUD confidence, the latest client-confirmed 10-button state, ACK-known flag and
15 explicit combat-state values to the 256-value LSTM hidden state. The added
values are player/opponent X, distance, grounded/defending/attacking flags,
time since damage, consecutive hits, stationary/corner duration and confidence
for the visual/action estimates.

Player attack and defense come from `action_ack`, so they represent controls
the client actually applied. Health and damage remain deterministic HUD
measurements. Screen position and grounded state come from a bounded temporal
motion tracker below the HUD. Opponent attack/defense are explicitly
confidence-weighted estimates; uncertain estimates are not treated as hard
ground truth.

The state is concatenated with learned 16-value embeddings for the controlled
character and opponent. A shared hidden trunk learns spacing, defense, timing
and fight-state strategy from every character. Each controlled character has a
small action head so specials with different meanings never collide. The
selected head chooses among high-level options:

- bounded primitive holds such as move, block, jump, punch and kick;
- a coordinated forward jump-kick (and its F/B mirror), shared by all
  controlled characters;
- a shared crouching uppercut that holds Down before pressing Down+Y;
- a two-option cooldown after Noob Saibot's disable cloud; both F/B
  variants are masked in mock exploration, DQN exploitation and Double-DQN
  successor targets while the projectile can still turn against its owner;
- close combos masked when normalized fighter distance exceeds `0.22`;
- projectile-first scripted exploration at distance `0.28` or greater, plus
  a bounded Q-value preference for a competitive projectile in DQN mode;
- `Neutro` masked during spatially valid active fights, while remaining in the
  action head for checkpoint compatibility and uncertain fallback states;
- that character's declared specials and combos;
- the unique Forward/Back mirrors of those programs.

An option emits one complete protocol-v2 controller state every 100 ms. Its
discounted reward is accumulated until the option ends, and replay stores its
actual duration. Double-DQN targets therefore use `gamma^duration`, assigning
delayed game damage to the complete move instead of an arbitrary controller
tick. Projectile and teleport transitions also retain two released controller
steps before closing, extending their reward-credit window by roughly 300 ms
at the default 10 decisions per second without changing checkpoint topology.

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
health_reward = 15 * damage_dealt - 11 * damage_taken
combo_reward  = 0.05 * min(consecutive_hits - 1, 5)
defense_reward = 0.05/second while a nearby attack is blocked without damage
idle_penalty  = 0.01/second after 1.5 seconds without activity
corner_penalty = 0.02/second after 1.0 second in a screen corner
neutral_option_penalty = 0.20 whenever fallback policy selects Neutro
reward = health_reward + combo_reward + defense_reward
         - idle_penalty - corner_penalty - neutral_option_penalty
round win  = +5
round loss = -5
```

Damage and episode flags accumulate until the RL decision loop consumes
them, so one camera frame cannot be counted twice. Terminal bonuses are emitted
once. Position, distance, grounded/attack/defense flags and timers are state
features. Only a contextually successful block receives direct defensive
shaping; simply holding block is not rewarded. `action_ack` remains the source
of truth for commands actually applied.

### Checkpoints and resume

The recurrent checkpoint stores online/target LSTM and Double-DQN parameters,
their AdamW state, the fine-tuned ResNet50 and its optimizer when enabled,
update counters and a compact recent replay snapshot. Persisted replay omits
JPEG bytes to keep checkpoints bounded; new frames refill visual replay after
resume. A separate format-v4 metadata checkpoint preserves per-character
epsilon/episode state. Both are written atomically at round end and connection
shutdown:

```text
/workspace/gameagent/rl_checkpoints/
├── shared_recurrent_latest.pt
├── shared_recurrent_metadata_latest.garl
├── shared_recurrent_metadata_best_<character>.garl
└── shared_recurrent_metadata_best_<character>.garl.recurrent.pt
```

Combat-state v2 expands the recurrent context from 15 to 30 values. A
pre-v2 checkpoint is intentionally rejected instead of silently attaching old
weights to different meanings. Start v2 in a new checkpoint directory with
`GAMEAGENT_RL_RESUME=0`; subsequent runs of that directory use
`GAMEAGENT_RL_RESUME=1`.

Once either latest checkpoint exists, WebSocket reconnections always resume
it even if the long-lived server was originally launched with
`GAMEAGENT_RL_RESUME=0`. A deliberately fresh experiment must therefore use a
new checkpoint directory. If recurrent weights contain more updates than the
metadata selection counters, the server reconstructs a conservative counter
floor so epsilon cannot silently jump back to one.

The former `shared_policy_latest.garl` is not overwritten, so it remains an
independent baseline for the inference-only DQN.

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

To start the corrected v8 policy without overwriting v7, use:

```bash
cd /tmp/GameAgent
export GAMEAGENT_API_TOKEN="$(< /workspace/gameagent/api_token)"
bash scripts/run_character_rl_v8.sh
```

On its first run, the launcher reads
`rl_combat_state_v7_calibrated/shared_recurrent_latest.pt`, imports the
ResNet50 stage 4, projection, LSTM, embeddings and strategy representation,
then initializes two fresh independent value heads. New checkpoints and replay
are written only under `/workspace/gameagent/rl_combat_state_v8`. Subsequent
runs resume that v8 checkpoint normally.

The client then selects the character and opponent in `session_config`. A
different character loads the same shared latest checkpoint and selects its own
head. To start the complete shared policy from scratch in an empty directory:

```bash
GAMEAGENT_RL_RESUME=0 \
GAMEAGENT_RL_CHECKPOINT_DIR=/workspace/gameagent/rl_checkpoints_clean \
bash scripts/run_character_rl.sh
```

Pure-DQN evaluation uses an isolated checkpoint snapshot, disables replay,
gradient/Adam updates and checkpoint writes, forces epsilon to zero, and writes
metrics under a timestamped evaluation directory:

```bash
bash scripts/run_character_rl_evaluation.sh
```

After stopping the evaluation server, summarize the latest run:

```bash
python3 scripts/analyze_rl_evaluation.py
```

The analyzer fails if it observes exploration, training, changing update
counters or checkpoint saves. The evaluator snapshots
`shared_recurrent_latest.pt` (and metadata when present) before starting and
verifies that every snapshot remains byte-identical.
Use `GAMEAGENT_EVALUATION_PORT=8081` when evaluating on a separately exposed
port.

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
[RL_REWARD] reward_gained=0.42 reward_accumulated=1.87 reward_terminal=0 result=ongoing connection=1 episode=4 phase=active_fight damage_dealt=0.08 damage_taken=0.03 hp=0.71/0.54 epsilon=0.93 replay=418 updates=162 loss=0.021 q_pred=1.84 td_target=1.77 q_gap=0.07 abs_td=0.31 bootstrap=1.72 target_clipped=0 terminal_batch=0.06 terminal_q_pred=-3.8 terminal_target=-5
```

An episode ending forces an immediate summary instead of waiting for the next
reporting interval. `reward_terminal` is `+5` for a win, `-5` for a loss and
zero otherwise. `reward_gained` is the reward since the previous summary, while
`reward_accumulated` is the complete episode return, including damage rewards,
damage penalties and the terminal reward. A blank line follows each summary to
make reward events easier to scan.

`q_pred` is the replay batch's mean predicted Q value, `td_target` is its mean
clipped Double-DQN target, and `q_gap=q_pred-td_target`. A persistently positive
gap is direct evidence of current overestimation. `abs_td` is the mean absolute
TD error, `bootstrap` is the conservative successor estimate and
`target_clipped` is the fraction constrained by the configured value bound.
`terminal_batch` is the terminal fraction of that replay batch; for those
samples `terminal_target` contains only the observed terminal transition and
never a successor bootstrap, while `terminal_q_pred` shows its calibration.

The same line also reports `shaping`, `combo_reward`, `idle_penalty`,
`corner_penalty`, player/opponent X, distance, combo length, stationary time
and corner time.

During training, after `round_lost`, the server releases controls and emits one
`training_reset_state` request with a connection-local `reset_id`. The client
loads its configured training save state and returns `training_reset_ack`.
Wins and double KOs do not trigger this request. Read-only evaluation instead
requests one clean save-state load after either round result, allowing
repeatable pure-DQN trials. This behavior is enabled by
default and can be disabled with `GAMEAGENT_TRAINING_RESET_STATE=0`.
Independently of a detected result, a recovery watchdog requests the same reset
when the detector remains outside `active_fight` for five continuous seconds.
An active-fight frame clears the watchdog. If the client acknowledges a reset
but the screen is still outside a fight, a new request may be issued only after
another complete timeout window. Set the window with
`GAMEAGENT_TRAINING_RESET_NON_FIGHT_TIMEOUT_MS`.
For compatibility with existing clients, this recovery request keeps the
legacy wire values `reason: "round_lost"` and a positive `episode_id`. The
server-side JSONL metric records the actual cause as
`detected_reason: "non_fight_timeout"`; no client change is required.

RL sessions also compare the decoded RGB bytes of consecutive frames, rather
than trusting an increasing `frame_id`. After 2000 ms of an exactly unchanged
image, the server releases controls and calls the agent with actions disabled,
which prevents replay writes, optimizer updates and epsilon decay while the
emulator is frozen. At 3000 ms it emits one idempotent
`training_reset_state`; its server metric uses
`detected_reason: "repeated_frame_timeout"` while the wire message retains the
legacy `reason: "round_lost"`. Repeated reset requests are not created for the
same frozen streak. Processing resumes only after decoded RGB pixels change;
the temporal sequence is cleared on pause and recovery.
Until acknowledged, one logical reset remains pending, blocks new actions and
is retransmitted every 500 ms with the same id. This keeps the operation
idempotent and prevents a delayed acknowledgement from being displaced by a
new reset id. Ordinary action responses are suspended until that acknowledgement
so clients cannot hide the reset behind one-action-per-frame processing.

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

Validate the trainable visual path, exact BaseAI/PyTorch preprocessing parity
and the production-shape GPU memory budget:

```bash
bash scripts/run_trainable_resnet_test.sh
bash scripts/run_resnet_feature_parity_test.sh
bash scripts/run_joint_training_memory_test.sh
```

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
| `GAMEAGENT_DECISION_INTERVAL_MS` | `50` (`100` in RL launcher) | inference loop interval |
| `GAMEAGENT_FAILSAFE_TIMEOUT_MS` | `200` (`750` in RL launcher) | client safety release timeout advertised in every action |
| `GAMEAGENT_MAX_JPEG_MB` | `8` | maximum JPEG payload |
| `GAMEAGENT_METRICS_PATH` | `metrics.jsonl` | JSONL metrics output |
| `GAMEAGENT_CAPTURE_DIR` | `captures` | optional JPEG inspection output |
| `GAMEAGENT_SAVE_FRAMES` | `0` | enable periodic debug JPEG samples; zero keeps everything in memory |
| `GAMEAGENT_DEBUG_CAPTURE_INTERVAL_MS` | `3000` | minimum time between debug JPEG samples |
| `GAMEAGENT_COLLECT_TRAINING_DATA` | `0` | enqueue accepted JPEGs and aligned ACK metadata for offline training without blocking reception |
| `GAMEAGENT_TRAINING_CAPTURE_DIR` | `/workspace/gameagent/training_captures` | full-session training capture root |
| `GAMEAGENT_DEFAULT_FACING` | `right` | legacy/default Forward/Back mapping when the client omits `session_config` |
| `GAMEAGENT_RL_CHECKPOINT_INTERVAL_EPISODES` | `25` | completed episodes between full online-training checkpoints; graceful disconnect/shutdown still saves the latest state |
| `GAMEAGENT_API_TOKEN` | unset | Bearer or X-API-Key credential |
| `GAMEAGENT_RANDOM_SEED` | random | deterministic mock seed |
| `GAMEAGENT_TEMPORAL_MODEL` | `0` | enable ResNet50 plus GPU temporal feature path |
| `GAMEAGENT_RESNET50_CHECKPOINT` | `/workspace/resnet50_best.ckpt` | initial ImageNet ResNet50 checkpoint |
| `GAMEAGENT_TEMPORAL_CHECKPOINT` | unset | trained `.gat` policy loaded in student-shadow mode |
| `GAMEAGENT_DEVICE_ID` | `0` | CUDA device used by ResNet50 and LSTM |
| `GAMEAGENT_LSTM_PROJECTION_SIZE` | `256` | ResNet50 feature projection width |
| `GAMEAGENT_LSTM_HIDDEN_SIZE` | `256` | temporal hidden-state width |
| `GAMEAGENT_REINFORCEMENT_LEARNING` | `0` | enable shared multi-character recurrent Double DQN |
| `GAMEAGENT_RL_TRAINABLE_LSTM` | `1` | train the LSTM and Q network with BPTT on CUDA |
| `GAMEAGENT_RL_TRAINABLE_RESNET` | `0` (`1` in character RL launcher) | fine-tune ResNet50 `layer4`; earlier stages and BatchNorm statistics stay frozen |
| `GAMEAGENT_RL_GAMMA` | `0.995` | discount per 100 ms decision, giving an approximately 20-second value horizon |
| `GAMEAGENT_RL_CONSERVATIVE_BOOTSTRAP` | `1` | use the lower value from two independent recurrent critics |
| `GAMEAGENT_RL_Q_VALUE_LIMIT` | `25.0` | symmetric TD-target bound based on the attainable combat reward range |
| `GAMEAGENT_RL_TERMINAL_REPLAY_FRACTION` | `0.25` | fraction of every recurrent replay batch reserved for matchup-balanced terminal transitions |
| `GAMEAGENT_RL_HARD_NEGATIVE_REPLAY_FRACTION` | `0.25` | minimum batch fraction reserved for positive-Q options whose n-step return became negative |
| `GAMEAGENT_RL_HARD_NEGATIVE_Q_THRESHOLD` | `1.0` | Q threshold used to classify hard-negative transitions |
| `GAMEAGENT_RL_N_STEP_OPTIONS` | `3` | consecutive semi-Markov options combined into each recurrent TD target |
| `GAMEAGENT_RL_RESNET_LEARNING_RATE` | `0.000001` | AdamW learning rate for ResNet50 `layer4` |
| `GAMEAGENT_RL_RESNET_BATCH_SIZE` | `4` | JPEG observations in each visual update |
| `GAMEAGENT_RL_RESNET_UPDATE_INTERVAL` | `16` | recurrent optimizer updates per visual update |
| `GAMEAGENT_RL_RESNET_REPLAY_IMAGES` | `4096` | newest compressed JPEGs retained only in memory for visual training |
| `GAMEAGENT_RL_DAMAGE_DEALT_REWARD_SCALE` | `15.0` | reward multiplier for normalized damage dealt |
| `GAMEAGENT_RL_DAMAGE_TAKEN_PENALTY_SCALE` | `11.0` | penalty multiplier for normalized damage taken |
| `GAMEAGENT_RL_COMBO_REWARD_SCALE` | `0.05` | incremental bonus for each consecutive hit after the first, capped at five extra hits |
| `GAMEAGENT_RL_DEFENSE_REWARD_PER_SECOND` | `0.05` | reward rate for a confirmed nearby block that prevents damage |
| `GAMEAGENT_RL_NEUTRAL_OPTION_PENALTY` | `0.20` | fallback opportunity cost when `Neutro` is available under uncertain spatial state |
| `GAMEAGENT_RL_SUPPRESS_NEUTRAL` | `1` | mask `Neutro` during spatially valid active fights without changing the action head |
| `GAMEAGENT_RL_COMBO_MAX_DISTANCE` | `0.22` | maximum normalized fighter distance at which close scripted combos are legal |
| `GAMEAGENT_RL_CLOSE_ATTACK_MAX_DISTANCE` | `0.24` | maximum distance for primitive punches, kicks and uppercuts |
| `GAMEAGENT_RL_PROJECTILE_MIN_DISTANCE` | `0.28` | minimum distance for ordinary projectiles and distance at which they are preferred |
| `GAMEAGENT_RL_PROJECTILE_Q_MARGIN` | `1.0` | maximum Q disadvantage accepted when preferring a projectile at distance; halved under an observed attack |
| `GAMEAGENT_RL_DISABLE_CLOUD_MIN_DISTANCE` / `MAX_DISTANCE` | `0.12` / `0.55` | empirically useful range for Noob Saibot's disable cloud |
| `GAMEAGENT_RL_ADVANCE_MIN_DISTANCE` | `0.18` | prevents advancing again after already reaching close range |
| `GAMEAGENT_RL_DEFENSE_MAX_DISTANCE` | `0.32` | prevents distant blocking with no immediate threat |
| `GAMEAGENT_RL_DYNAMIC_FACING` | `1` | derives Forward/Back from live fighter X positions and masks explicit mirrored duplicates |
| `GAMEAGENT_RL_DISABLE_CLOUD_COOLDOWN_OPTIONS` | `2` | complete alternative options required before another Noob Saibot disable cloud; teleports remain unrestricted |
| `GAMEAGENT_RL_MELEE_CREDIT_STEPS` | `2` | post-program steps retained for melee/combo damage credit |
| `GAMEAGENT_RL_TELEPORT_CREDIT_STEPS` | `4` | post-program steps retained for teleport impact credit |
| `GAMEAGENT_RL_PROJECTILE_CREDIT_MIN_STEPS` / `MAX_STEPS` | `4` / `8` | distance-scaled post-program projectile credit window |
| `GAMEAGENT_RL_GREEDY_EPISODE_INTERVAL` | `10` | train one near-greedy episode after every ten episodes |
| `GAMEAGENT_RL_GREEDY_EPISODE_EPSILON` | `0.05` | epsilon used by the near-greedy training episode |
| `GAMEAGENT_RL_MIGRATION_CHECKPOINT` | unset | optional v7 recurrent checkpoint used only to initialize v8 perception and temporal representation |
| `GAMEAGENT_RL_IDLE_PENALTY_PER_SECOND` | `0.01` | small penalty per second after 1.5 seconds without movement, commands or damage |
| `GAMEAGENT_RL_CORNER_PENALTY_PER_SECOND` | `0.02` | small penalty per second after 1.0 second near a screen edge |
| `GAMEAGENT_RL_RECURRENT_SERVICE` | `/tmp/GameAgent/scripts/recurrent_dqn_service.py` | persistent local PyTorch/CUDA recurrent backend |
| `GAMEAGENT_RL_CHECKPOINT_DIR` | `/workspace/gameagent/rl_checkpoints` | shared latest policy/replay plus per-character best snapshots |
| `GAMEAGENT_TRAINING_RESET_STATE` | `1` | request a client-side save-state load once after each `round_lost` |
| `GAMEAGENT_TRAINING_RESET_NON_FIGHT_TIMEOUT_MS` | `5000` | request recovery after this many continuous milliseconds outside `active_fight` |
| `GAMEAGENT_REPEATED_FRAME_PAUSE_MS` | `2000` | release controls and suspend RL when decoded RGB pixels remain identical |
| `GAMEAGENT_REPEATED_FRAME_RESET_MS` | `3000` | request one client state reset for the same unchanged-frame streak |
| `GAMEAGENT_EVALUATION_ROOT` | `/workspace/gameagent/evaluations` | isolated snapshots, logs and metrics produced by the pure-DQN evaluator |

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
`test_gameagent_combat_state_tracker` validates motion-based position,
grounded/action state, damage timing, consecutive-hit reward and one-time
idle/corner reward consumption.

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
