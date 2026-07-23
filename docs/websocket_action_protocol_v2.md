# GameAgent WebSocket action protocol v2

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
  "commands": [9, 3],
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
    "decode_errors": 0
  }
}
```

`commands: [9, 3]` means that Right and A are held together. A later
`commands: [9]` releases A while retaining Right. `commands: []` releases every
control. Repeating the same array is idempotent and must not generate extra key
events.

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
