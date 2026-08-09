# GameAgent server protocol V2.1

This document describes the server-side wire contract implemented by
`GameAgent`. Protocol V2.1 keeps the binary JPEG frame envelope from protocol 2
and adds an explicit encounter, runtime-mode and reset state machine.

Every V2.1 text message contains `protocol_version`, `request_id`,
`timestamp_ms` and `type`. Identifiers are positive integers and stages are
zero-based.

## Session configuration

The first client message must be `configure_session`, before any frame:

```json
{
  "protocol_version": 2,
  "request_id": 1,
  "timestamp_ms": 1780000000000,
  "type": "configure_session",
  "player_character": "noob_saibot",
  "player_side": "left",
  "initial_facing": "right",
  "encounter_type": "endurance",
  "opponent_count": 2,
  "current_stage": 0,
  "opponents": ["jax", "nightwolf"]
}
```

`normal` and `boss` require one opponent. `endurance` requires two to four
ordered opponents. The server answers `configure_session_ack`; invalid or V1
checkpoint state is rejected rather than migrated.

JPEG frames continue to use the `GAF1` binary envelope:

```text
4 bytes  magic "GAF1"
8 bytes  frame_id, unsigned big endian
8 bytes  timestamp_ms, signed big endian
4 bytes  JPEG byte count, unsigned big endian
N bytes  JPEG
```

## Encounter lifecycle

The client reports semantic events with `encounter_event`:

```json
{
  "protocol_version": 2,
  "request_id": 20,
  "timestamp_ms": 1780000005000,
  "type": "encounter_event",
  "episode_id": 812,
  "stage_transition_id": 44,
  "event": "stage_completed",
  "encounter_stage": 0
}
```

For Endurance, `stage_completed` advances to the next ordered opponent, adds
the stage bonus once and ignores the apparent 0%-to-100% opponent health refill.
It is not terminal. Only `encounter_won` or `encounter_lost` ends the episode.
`stage_transition_id` makes stage changes idempotent.

`update_encounter_stage` is also accepted when the client already knows the
next stage and opponent:

```json
{
  "protocol_version": 2,
  "request_id": 21,
  "timestamp_ms": 1780000005100,
  "type": "update_encounter_stage",
  "episode_id": 812,
  "stage_transition_id": 44,
  "current_stage": 1,
  "current_opponent": "nightwolf"
}
```

## Runtime mode and exploration

Mode changes normally apply at the next episode boundary:

```json
{
  "protocol_version": 2,
  "request_id": 30,
  "timestamp_ms": 1780000006000,
  "type": "set_agent_mode",
  "mode": "evaluation",
  "reset_evaluation_session": true,
  "apply_on_next_episode": true
}
```

Evaluation always uses epsilon zero and freezes replay writes, optimizer
updates, checkpoint writes and scheduler counters. Returning to `training`
continues the same in-memory learner.

Exploration can be changed without restarting the server:

```json
{
  "protocol_version": 2,
  "request_id": 31,
  "timestamp_ms": 1780000006100,
  "type": "set_exploration",
  "mode": "automatic_from_value",
  "value": 0.35,
  "reset_matchup_counter": true
}
```

Supported modes are `automatic`, `manual_fixed` and
`automatic_from_value`. Evaluation mode still overrides the effective epsilon
to zero.

## Transactional reset

The server never presses F4 itself. After the terminal transition is fully
committed, it sends exactly one logical `reset_episode` identified by
`reset_id`. The client executes F4 and replies:

```json
{
  "protocol_version": 2,
  "request_id": 40,
  "timestamp_ms": 1780000007000,
  "type": "reset_ack",
  "reset_id": 94,
  "episode_id": 812,
  "command_executed": true,
  "executed_timestamp_ms": 1780000006990
}
```

Once the initial state is visually stable, the client sends:

```json
{
  "protocol_version": 2,
  "request_id": 41,
  "timestamp_ms": 1780000007900,
  "type": "reset_ready",
  "reset_id": 94,
  "stable_initial_state": true,
  "first_stable_frame_id": 900
}
```

The server then replies `episode_started` with the new `episode_id`. No action,
reward consumption, replay insertion or epsilon advance occurs while the reset
is pending. Duplicate ACK/ready messages are harmless. Failures use
`reset_failed`; acknowledgement and readiness timeouts are controlled by
`GAMEAGENT_RESET_ACK_TIMEOUT_MS` and `GAMEAGENT_RESET_READY_TIMEOUT_MS`.

The client may request a non-terminal reset using `request_reset` with one of
`manual`, `invalid_game_state`, `stream_stalled` or `stuck`. Such an episode is
aborted and does not count as a win or loss.

## Server output

At 10 Hz the action stream includes encounter context, effective mode,
exploration source, reward breakdown, Q values, learner metrics and reset
state. V2.1 additionally emits:

- `telemetry`: complete instantaneous state;
- `telemetry_sample`: compact chart sample at 10 Hz;
- `telemetry_marker`: stage, terminal, mode, epsilon, reset and checkpoint
  events;
- `episode_summary`: one record after each terminal episode;
- `evaluation_summary`: cumulative read-only evaluation results.

The client must route messages by `type` and correlate request responses by
`request_id`; it must not assume that the next incoming message is necessarily
the response to its most recent request.

## Fresh V2.1 training

V2.1 uses 36 contextual features, including encounter type, stage, opponent
count and remaining opponents. Its default checkpoint directory is
`/workspace/gameagent/rl_v21`, and V1 automatic migration is disabled.

```bash
cd /tmp/GameAgent
export GAMEAGENT_API_TOKEN="$(< /workspace/gameagent/api_token)"
GAMEAGENT_RL_RESUME=0 bash scripts/run_character_rl.sh
```

After V2.1 has produced a checkpoint, resume it with
`GAMEAGENT_RL_RESUME=1`. Do not point V2.1 at a V1 checkpoint directory.
