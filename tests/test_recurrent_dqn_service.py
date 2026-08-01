#!/usr/bin/env python3
"""End-to-end smoke test for trainable recurrent CUDA policy service."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "scripts" / "recurrent_dqn_service.py"


class Client:
    def __init__(
        self,
        checkpoint: Path,
        training: bool,
        writes: bool,
        resume: bool,
        n_step_options: int = 1,
    ) -> None:
        command = [
            sys.executable,
            "-u",
            str(SERVICE),
            "--checkpoint",
            str(checkpoint),
            "--training",
            "1" if training else "0",
            "--checkpoint-writes",
            "1" if writes else "0",
            "--resume",
            "1" if resume else "0",
            "--visual-size",
            "12",
            "--projection-size",
            "8",
            "--controller-size",
            "3",
            "--temporal-hidden-size",
            "16",
            "--context-size",
            "5",
            "--strategy-hidden-size",
            "16",
            "--sequence-capacity",
            "4",
            "--character-embedding-size",
            "3",
            "--opponent-embedding-size",
            "3",
            "--opponent-count",
            "6",
            "--action-sizes",
            "3,4,2,5",
            "--replay-capacity",
            "16",
            "--minimum-replay-size",
            "2",
            "--batch-size",
            "2",
            "--target-update",
            "2",
            "--n-step-options",
            str(n_step_options),
            "--hard-negative-replay-fraction",
            "0.25",
            "--learning-rate",
            "0.001",
            "--seed",
            "1234",
            "--persist-replay-limit",
            "16",
        ]
        if not torch.cuda.is_available():
            command.append("--allow-cpu")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.ready = self.receive()

    def receive(self) -> dict:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("recurrent service terminated unexpectedly")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "service error"))
        return response

    def request(self, request: dict) -> dict:
        assert self.process.stdin is not None
        self.process.stdin.write(
            json.dumps(request, separators=(",", ":")) + "\n"
        )
        self.process.stdin.flush()
        return self.receive()

    def shutdown(self) -> None:
        self.request({"op": "shutdown"})
        self.process.wait(timeout=30)
        if self.process.returncode != 0:
            raise RuntimeError("recurrent service exited with an error")


def observation(client: Client, seed: int, episode_started: bool) -> None:
    client.request(
        {
            "op": "observe",
            "visual": [
                math.sin(seed * 0.17 + index * 0.11)
                for index in range(12)
            ],
            "controller": [
                float((seed + index) % 2) for index in range(3)
            ],
            "context": [
                math.cos(seed * 0.13 + index * 0.07)
                for index in range(5)
            ],
            "episode_started": episode_started,
        }
    )


def add_transition(
    client: Client,
    seed: int,
    terminal: bool,
    episode_started: bool = True,
) -> dict:
    observation(client, seed, episode_started)
    observation(client, seed + 1, False)
    action_count = (3, 4, 2, 5)[seed % 4]
    q = client.request(
        {
            "op": "q_values",
            "character": seed % 4,
            "opponent": seed % 6,
            "action_count": action_count,
        }
    )
    action = max(
        range(len(q["q_values"])),
        key=lambda index: q["q_values"][index],
    )
    client.request(
        {
            "op": "begin",
            "action": action,
            "character": seed % 4,
            "opponent": seed % 6,
            "selected_q": float(q["q_values"][action]),
        }
    )
    observation(client, seed + 2, False)
    return client.request(
        {
            "op": "close",
            "duration": 3,
            "reward": 0.5 if seed % 2 else -0.25,
            "terminal": terminal,
            "next_action_mask": [True] * action_count,
        }
    )


def main() -> int:
    with tempfile.TemporaryDirectory(
        prefix="gameagent_recurrent_test_"
    ) as temporary:
        directory = Path(temporary)
        checkpoint = directory / "recurrent.pt"
        client = Client(checkpoint, training=True, writes=True, resume=False)
        initial_checksum = float(client.ready["parameter_checksum"])
        result = add_transition(client, 1, False)
        if result["training_updates"] != 0:
            raise RuntimeError("recurrent warmup trained too early")
        result = add_transition(client, 2, False)
        if (
            result["training_updates"] != 1
            or not result["training_performed"]
            or not math.isfinite(float(result["last_loss"]))
            or not all(
                math.isfinite(float(result[name]))
                for name in (
                    "last_predicted_q_mean",
                    "last_td_target_mean",
                    "last_bootstrap_mean",
                    "last_q_target_gap",
                    "last_abs_td_error",
                    "last_q_target_clipped_fraction",
                    "last_terminal_sample_fraction",
                    "last_terminal_predicted_q_mean",
                    "last_terminal_td_target_mean",
                    "last_critic_disagreement",
                    "last_hard_negative_sample_fraction",
                )
            )
            or not 0.0
            <= float(result["last_q_target_clipped_fraction"])
            <= 1.0
            or float(result["parameter_checksum"]) == initial_checksum
            or float(result["last_critic_disagreement"]) <= 0.0
        ):
            raise RuntimeError("BPTT did not update recurrent parameters")
        terminal_result = add_transition(client, 3, True)
        if (
            float(terminal_result["last_terminal_sample_fraction"])
            <= 0.0
            or not math.isclose(
                float(terminal_result["last_terminal_td_target_mean"]),
                0.5,
                abs_tol=1.0e-6,
            )
        ):
            raise RuntimeError(
                "terminal replay did not use the unbootstrapped reward"
            )
        client.request(
            {
                "op": "save",
                "path": str(checkpoint),
                "include_replay": True,
            }
        )
        client.shutdown()
        if not checkpoint.is_file():
            raise RuntimeError("recurrent checkpoint was not written")

        forbidden = directory / "evaluation_must_not_write.pt"
        evaluation = Client(
            checkpoint, training=False, writes=False, resume=True
        )
        before = evaluation.request({"op": "stats"})
        observation(evaluation, 10, True)
        values = evaluation.request(
            {
                "op": "q_values",
                "character": 0,
                "opponent": 1,
                "action_count": 3,
            }
        )["q_values"]
        evaluation.request(
            {
                "op": "save",
                "path": str(forbidden),
                "include_replay": True,
            }
        )
        after = evaluation.request({"op": "stats"})
        evaluation.shutdown()
        if (
            not before["checkpoint_loaded"]
            or before["training_updates"] != after["training_updates"]
            or before["parameter_checksum"] != after["parameter_checksum"]
            or forbidden.exists()
            or len(values) != 3
            or not all(math.isfinite(float(value)) for value in values)
        ):
            raise RuntimeError(
                "read-only recurrent evaluation changed policy state"
            )

        n_step_checkpoint = directory / "n_step.pt"
        n_step = Client(
            n_step_checkpoint,
            training=True,
            writes=False,
            resume=False,
            n_step_options=3,
        )
        first = add_transition(n_step, 20, False, True)
        second = add_transition(n_step, 20, False, False)
        third = add_transition(n_step, 20, False, False)
        terminal = add_transition(n_step, 20, True, False)
        n_step.shutdown()
        if (
            first["replay_size"] != 0
            or second["replay_size"] != 0
            or third["replay_size"] != 1
            or terminal["replay_size"] != 4
            or terminal["training_updates"] <= 0
        ):
            raise RuntimeError("three-option n-step replay is inconsistent")
        print(
            "[INFO] "
            f"device={client.ready['device']} "
            f"updates={result['training_updates']} "
            f"loss={float(result['last_loss']):.6f} "
            f"checksum_delta="
            f"{float(result['parameter_checksum']) - initial_checksum:.6f}"
        )
        print(
            "[PASS] trainable LSTM BPTT/checkpoint/evaluation test passed"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
