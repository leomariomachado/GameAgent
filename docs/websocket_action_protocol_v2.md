# GameAgent WebSocket action protocol v2

## Session configuration

Immediately after the WebSocket upgrade and before the first frame, the client
should send one unfragmented text message identifying the selected fighter:

```json
{
  "protocol_version": 2,
  "type": "session_config",
  "player_character": "scorpion",
  "opponent_character": "subzero",
  "player_side": "left",
  "initial_facing": "right"
}
```

`player_character` is required and accepts `subzero`, `scorpion`,
`noob_saibot`, or `sindel`.

`opponent_character` should always be sent by a new client and accepts exactly:

```text
unknown, cyrax, ermac, jade, jax, kabal, kano, kitana, kung_lao,
liu_kang, mileena, motaro, nightwolf, noob_saibot, rain, reptile,
scorpion, sektor, shang_tsung, shao_kahn, sheeva, sindel, smoke,
human_smoke, sonya, stryker, subzero, classic_subzero
```

Send `unknown` when the opponent is not known. For compatibility, omission is
also interpreted as `unknown`. `sheeva` covers arcade UMK3; the SNES port does
not contain her. `player_side`
identifies which HUD health bar belongs to the agent; `initial_facing` defines
how character-relative Forward and Back are mapped to Left and Right.

The server answers before accepting gameplay frames:

```json
{
  "protocol_version": 2,
  "type": "session_config_ack",
  "player_character": "scorpion",
  "player_character_name": "Scorpion",
  "opponent_character": "subzero",
  "opponent_character_name": "Sub-Zero",
  "opponent_feature_index": 26,
  "opponent_embedding_index": 26,
  "opponent_feature_count": 28,
  "opponent_vocabulary_size": 28,
  "opponent_embedding_size": 16,
  "player_side": "left",
  "initial_facing": "right",
  "profile": "scorpion_v1",
  "mock_move_count": 11,
  "rl_option_count": 22,
  "agent_mode": "reinforcement_learning",
  "shared_policy": true,
  "training_reset_state_enabled": true,
  "training_reset_trigger": "round_lost",
  "training_reset_ack_type": "training_reset_ack",
  "checkpoint_loaded": false,
  "trainable_lstm": true,
  "recurrent_checkpoint_loaded": false,
  "rl_training_device": "cuda:0",
  "rl_checkpoint": "/workspace/gameagent/rl_checkpoints/shared_recurrent_metadata_latest.garl"
}
```

The client must wait for this acknowledgement before sending the first frame.
On `session_config_error`, it must display/log the error and reconnect only
after correcting the configuration. The error includes
`supported_opponent_characters` when the opponent is invalid. A player or
opponent cannot change during a connection; reconnect and send a new
configuration instead.

## Client control toggle lifecycle

The UI control “Permitir que o agente controle” owns the WebSocket session.

When unchecked, the client must first mark agent control disabled, cancel
automatic reconnects, stop sending frames, release every held control, clear
pending frame/action state, invalidate callbacks from the old connection, and
close the WebSocket. It must not send frames or acknowledgements while
disabled.

When checked, the client must read the current player, opponent, HUD side and
facing selections, open a new WebSocket, send a fresh `session_config`, and
wait for a matching `session_config_ack`. Frames and actions are enabled only
after that acknowledgement.

Changing either character while control is enabled performs the same
release/close/reconnect/configure sequence automatically. Changing a selection
while control is disabled only updates local configuration for the next
connection. Automatic reconnects are allowed only while the toggle remains
checked. Callbacks must carry a local connection generation so late actions
from a closed session cannot press controls again.

This restarts only the WebSocket session. It does not restart the server or
discard reinforcement checkpoints.

For backward compatibility, a connection whose first data message is a frame
uses the server defaults: Sub-Zero, configured player side, and configured
facing. Such a legacy connection receives no configuration acknowledgement.

In reinforcement mode all supported characters use one shared recurrent
policy. The ResNet50 remains frozen; the LSTM, player/opponent embeddings,
shared strategy trunk and one action head per controlled character learn
together with BPTT on CUDA. Replay is shared and sampled across non-empty
player/opponent matchup buckets. Epsilon progress, action-selection counters
and best episode return remain independent for each controlled character.

`shared_recurrent_latest.pt` is the resumable model/AdamW/replay checkpoint.
`shared_recurrent_metadata_latest.garl` preserves the C++ option and epsilon
metadata. Per-character best snapshots use the
`shared_recurrent_metadata_best_<character>.garl.recurrent.pt` suffix. There is
no checkpoint per matchup. The previous `shared_policy_latest.garl` remains an
untouched inference-only baseline.

`opponent_embedding_index` selects the row in the learned opponent table;
`opponent_vocabulary_size` is the current number of recognized identities and
`opponent_embedding_size` is the vector width used by the network. The legacy
aliases `opponent_feature_index` and `opponent_feature_count` remain in the ACK
so existing clients do not break; they no longer describe a one-hot vector.

## Semantics

Version 2 uses full controller-state actions. Every action atomically replaces
the preceding state. The `commands` array is an unordered set of controls that
must remain pressed until the next valid action arrives.

```json
{
  "protocol_version": 2,
  "type": "action",
  "action_mode": "state",
  "decision_id": 123,
  "frame_id": 456,
  "frame_ids": [449, 450, 451, 452, 453, 454, 455, 456],
  "commands": [7],
  "mock_move": "Congelar",
  "mock_step": 1,
  "mock_steps": 5,
  "mock_step_duration_ms": 50,
  "timestamp_ms": 1784740000170,
  "control_interval_ms": 50,
  "failsafe_timeout_ms": 200,
  "metrics": {
    "frame_latency_ms": 47,
    "server_frame_age_ms": 18,
    "inference_ms": 0.018,
    "buffer_size": 32,
    "frames_received": 456,
    "frames_dropped": 424,
    "decode_errors": 0,
    "client_message_errors": 0,
    "acknowledgements_received": 122,
    "acknowledgements_rejected": 0,
    "applied_action_history_size": 8,
    "last_applied_decision_id": 122
  }
}
```

The `mock_*` fields identify the current character-specific test sequence and are
diagnostic metadata. Clients may display or log them, but controller execution
depends only on `commands`, `decision_id`, and the protocol-v2 state rules.
When enabled, `temporal_*` fields under `metrics` describe the server-side
ResNet50/LSTM feature pipeline. They are optional diagnostics and require no
client protocol change.

The scripted teacher also reports optional curriculum metadata:

```json
{
  "curriculum_source": "scripted_move",
  "curriculum_phase": "special"
}
```

When a trained temporal checkpoint is loaded, `metrics` may contain
`student_shadow_commands`, `student_shadow_probabilities` and
`student_shadow_exact_match`. These fields never control the client in shadow
mode; the top-level `commands` remains authoritative.

`commands: [9, 3]` means that Right and A are held together. A later
`commands: [9]` releases A while retaining Right. `commands: []` releases every
control. Repeating the same array is idempotent and must not generate extra key
events.

## Reinforcement-learning diagnostics and fight gate

When reinforcement mode is enabled, an action may also contain:

```json
{
  "agent_mode": "reinforcement_learning",
  "control_enabled": true,
  "cancel_pending_sequence": false,
  "fight_phase": "active_fight",
  "episode_id": 12,
  "player_health": 0.82,
  "opponent_health": 0.61,
  "reward": 0.31,
  "rl_option": "Soco Teleportado",
  "rl_option_index": 12,
  "rl_option_step": 2,
  "rl_option_steps": 5,
  "rl_selection_source": "epsilon_scripted"
}
```

These fields are optional metadata; `commands` remains authoritative. When
`control_enabled` is false, the server sends `commands: []` and
`cancel_pending_sequence: true`. A compliant state-mode client already releases
all controls when applying that empty state and must still send its normal
`action_ack`. No new client-side macro queue is introduced.

Possible `fight_phase` values are `no_hud`, `acquiring`, `round_intro`,
`active_fight` and `round_end`. Only `active_fight` can carry non-empty RL
commands. The client must not try to override this gate locally.

## Training-state reset after a defeat

In reinforcement-learning mode, one lost round is treated as a complete failed
training attempt. The server first emits its normal empty `action` state, then
emits exactly one reset request for that episode:

```json
{
  "protocol_version": 2,
  "type": "training_reset_state",
  "reset_id": 4,
  "episode_id": 19,
  "frame_id": 8821,
  "timestamp_ms": 1784740000400,
  "reason": "round_lost",
  "requested_operation": "load_training_state",
  "player_character": "scorpion",
  "opponent_character": "subzero",
  "release_controls": true,
  "cancel_pending_sequence": true,
  "ack_required": true
}
```

The client must process this message outside the normal action parser. It must
release every control, cancel queued input, suspend action application, load
the locally configured emulator save state, clear stale frame/action state and
then answer:

```json
{
  "protocol_version": 2,
  "type": "training_reset_ack",
  "reset_id": 4,
  "success": true,
  "completed_timestamp_ms": 1784740000710,
  "message": "save state loaded"
}
```

`message` is optional. On failure the client sends `success: false` with a
short diagnostic message and keeps controls released. `reset_id` is monotonic
within one WebSocket connection and makes the operation idempotent: a reset ID
already completed must not load the state a second time.

The server does not have direct access to the emulator and therefore requests,
rather than performs, the load. The fight gate continues emitting empty actions
until fresh post-load HUD frames establish a new active round. `round_won`,
`double_ko`, menus and ordinary round transitions do not request a reset.
`session_config_ack.training_reset_state_enabled` tells the client whether this
extension is active.

## Client state transition

For every action with a `decision_id` greater than the last applied ID:

```text
to_release = current_commands - new_commands
to_press   = new_commands - current_commands

release(to_release)
press(to_press)
current_commands = new_commands
```

The client ignores repeated or older decision IDs. It releases all controls
immediately on disconnect and when no valid action arrives within
`failsafe_timeout_ms`.

## Applied-action acknowledgement

Immediately after applying a valid action state, the client sends one WebSocket
text message:

```json
{
  "protocol_version": 2,
  "type": "action_ack",
  "decision_id": 123,
  "applied_commands": [9, 3],
  "applied_timestamp_ms": 1784740000182
}
```

`applied_commands` is the complete state that the client actually left active,
not a copy made before input injection. It may be empty and its order is
irrelevant. Values must be unique integers from 0 through 9.

The server accepts only monotonically increasing IDs that refer to a decision
sent on the current connection. It retains the latest confirmed states in
memory as multi-hot vectors and provides them to the agent's temporal input.
If the applied set differs from the sent set, the ACK remains valid but is
recorded with `commands_match=false`.

The server records `server_roundtrip_ms` using only its own clock. The client's
`applied_timestamp_ms` is retained for timeline alignment, but cross-machine
latency calculations require synchronized clocks.

No acknowledgement is sent for malformed, stale, or ignored actions. On
reconnection, both decision ordering and applied-action history restart for the
new connection.

## Command mapping

| Value | Control |
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

All values are unique integers from 0 through 9. Array order has no meaning.

## Sequences

Sequences are expressed as consecutive state actions, not as buffered macros.
For a quarter-circle to the right followed by A, the server emits:

```text
t=0 ms    [7]
t=50 ms   [7, 9]
t=100 ms  [9]
t=150 ms  [9, 3]
t=200 ms  []
```

This lets the agent reconsider its next state from recent frames instead of
committing the client to a long, stale action queue. A future timed-sequence
extension may be added if empirical tests show that 50 ms state steps are not
precise enough for a particular game; it is not part of protocol version 2.

## Compatibility

`commands` remains at the response top level, so clients that already parse the
vector can transition incrementally. Such clients must still implement state
retention, state differencing, the empty state, monotonic `decision_id`, and the
failsafe before they are considered protocol-v2 compliant.
