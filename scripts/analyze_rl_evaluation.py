#!/usr/bin/env python3

import collections
import json
import math
import pathlib
import sys


DEFAULT_METRICS = pathlib.Path(
    "/workspace/gameagent/evaluations/latest/metrics.jsonl"
)
metrics_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_METRICS
if metrics_path.is_dir():
    metrics_path = metrics_path / "metrics.jsonl"
if not metrics_path.is_file():
    raise SystemExit(f"Evaluation metrics not found: {metrics_path}")


def new_stats():
    return {
        "decisions": 0,
        "options": 0,
        "dqn_options": 0,
        "exploration_options": 0,
        "episodes": 0,
        "wins": 0,
        "losses": 0,
        "other_ends": 0,
        "returns": [],
        "damage_dealt": 0.0,
        "damage_taken": 0.0,
        "training_events": 0,
        "checkpoint_saves": 0,
        "updates_first": None,
        "updates_last": None,
        "epsilon_max": 0.0,
    }


groups = collections.defaultdict(new_stats)
invalid_lines = 0
reset_requests = 0
reset_acks = collections.Counter()

with metrics_path.open("rb") as stream:
    for raw_line in stream:
        try:
            record = json.loads(raw_line)
        except Exception:
            invalid_lines += 1
            continue
        event = record.get("event")
        if event == "training_reset_state_requested":
            reset_requests += 1
            continue
        if event == "training_reset_ack":
            reset_acks["success" if record.get("success") else "failure"] += 1
            continue
        if event != "decision" or record.get("agent_mode") != "reinforcement_learning":
            continue

        character = record.get("player_character", "unknown")
        opponent = record.get("opponent_character", "unknown")
        metrics = record.get("metrics") or {}
        for key in ((character, "all"), (character, opponent)):
            stats = groups[key]
            stats["decisions"] += 1
            epsilon = metrics.get("rl_epsilon")
            if isinstance(epsilon, (int, float)) and math.isfinite(epsilon):
                stats["epsilon_max"] = max(stats["epsilon_max"], float(epsilon))
            updates = metrics.get("rl_training_updates")
            if isinstance(updates, int):
                if stats["updates_first"] is None:
                    stats["updates_first"] = updates
                stats["updates_last"] = updates
            stats["training_events"] += int(
                bool(metrics.get("rl_training_performed"))
            )
            stats["checkpoint_saves"] += int(
                bool(metrics.get("rl_checkpoint_saved"))
            )
            damage_dealt = metrics.get("damage_dealt")
            damage_taken = metrics.get("damage_taken")
            if isinstance(damage_dealt, (int, float)):
                stats["damage_dealt"] += float(damage_dealt)
            if isinstance(damage_taken, (int, float)):
                stats["damage_taken"] += float(damage_taken)

            if record.get("rl_option_step") == 1:
                stats["options"] += 1
                if record.get("rl_selection_source") == "dqn_exploit":
                    stats["dqn_options"] += 1
                else:
                    stats["exploration_options"] += 1

            if record.get("episode_ended"):
                stats["episodes"] += 1
                reason = metrics.get("fight_transition_reason")
                if reason == "round_won":
                    stats["wins"] += 1
                elif reason == "round_lost":
                    stats["losses"] += 1
                else:
                    stats["other_ends"] += 1
                episode_return = metrics.get("rl_episode_return")
                if (
                    isinstance(episode_return, (int, float))
                    and math.isfinite(episode_return)
                ):
                    stats["returns"].append(float(episode_return))


print(f"[EVALUATION] metrics={metrics_path}")
print(
    "character opponent decisions options dqn% episodes wins losses "
    "win% avg_return damage_dealt damage_taken updates training writes epsilon_max"
)

if not groups:
    print(
        "[INCOMPLETE] No reinforcement-learning decisions were found.",
        file=sys.stderr,
    )
    raise SystemExit(3)

unsafe = False
for (character, opponent), stats in sorted(groups.items()):
    terminal = stats["wins"] + stats["losses"]
    dqn_percent = (
        100.0 * stats["dqn_options"] / stats["options"]
        if stats["options"] else 0.0
    )
    win_percent = (
        100.0 * stats["wins"] / terminal if terminal else 0.0
    )
    average_return = (
        sum(stats["returns"]) / len(stats["returns"])
        if stats["returns"] else 0.0
    )
    updates = (
        f"{stats['updates_first']}->{stats['updates_last']}"
        if stats["updates_first"] is not None else "n/a"
    )
    print(
        f"{character} {opponent} {stats['decisions']} {stats['options']} "
        f"{dqn_percent:.2f} {stats['episodes']} {stats['wins']} "
        f"{stats['losses']} {win_percent:.2f} {average_return:.4f} "
        f"{stats['damage_dealt']:.4f} {stats['damage_taken']:.4f} "
        f"{updates} {stats['training_events']} "
        f"{stats['checkpoint_saves']} {stats['epsilon_max']:.6f}"
    )
    if (
        stats["exploration_options"] != 0
        or stats["training_events"] != 0
        or stats["checkpoint_saves"] != 0
        or (
            stats["updates_first"] is not None
            and stats["updates_last"] != stats["updates_first"]
        )
        or stats["epsilon_max"] != 0.0
    ):
        unsafe = True

print(
    f"[RESET] requests={reset_requests} success={reset_acks['success']} "
    f"failure={reset_acks['failure']}"
)
print(f"[JSONL] invalid_or_partial_lines={invalid_lines}")
if unsafe:
    print("[FAIL] Evaluation was not pure read-only DQN.", file=sys.stderr)
    raise SystemExit(2)
print("[PASS] Evaluation used pure DQN and did not train or save checkpoints.")
