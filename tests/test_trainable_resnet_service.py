#!/usr/bin/env python3
"""GPU smoke test for joint ResNet50 + LSTM Double-DQN training."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "scripts" / "recurrent_dqn_service.py"
BASE_CHECKPOINT = Path(
    os.environ.get(
        "GAMEAGENT_RESNET50_CHECKPOINT",
        "/workspace/resnet50_best.ckpt",
    )
)


def jpeg(seed: int) -> str:
    image = Image.new(
        "RGB",
        (512, 448),
        (
            (seed * 47) % 256,
            (seed * 83) % 256,
            (seed * 131) % 256,
        ),
    )
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    return base64.b64encode(output.getvalue()).decode("ascii")


class Client:
    def __init__(
        self,
        checkpoint: Path,
        training: bool,
        writes: bool,
        resume: bool,
    ) -> None:
        command = [
            sys.executable,
            "-u",
            str(SERVICE),
            "--checkpoint",
            str(checkpoint),
            "--device-id",
            os.environ.get("GAMEAGENT_DEVICE_ID", "0"),
            "--training",
            "1" if training else "0",
            "--checkpoint-writes",
            "1" if writes else "0",
            "--resume",
            "1" if resume else "0",
            "--visual-size",
            "2048",
            "--projection-size",
            "16",
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
            "8",
            "--minimum-replay-size",
            "2",
            "--batch-size",
            "2",
            "--target-update",
            "2",
            "--learning-rate",
            "0.001",
            "--trainable-resnet",
            "1",
            "--resnet-checkpoint",
            str(BASE_CHECKPOINT),
            "--resnet-learning-rate",
            "0.00001",
            "--resnet-batch-size",
            "1",
            "--resnet-update-interval",
            "1",
            "--resnet-replay-images",
            "4",
            "--persist-replay-limit",
            "4",
            "--seed",
            "4321",
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.ready = self.request_raw()

    def request_raw(self) -> dict:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("recurrent service stopped unexpectedly")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "service error"))
        return response

    def request(self, value: dict) -> dict:
        assert self.process.stdin is not None
        self.process.stdin.write(
            json.dumps(value, separators=(",", ":")) + "\n"
        )
        self.process.stdin.flush()
        return self.request_raw()

    def shutdown(self) -> None:
        self.request({"op": "shutdown"})
        self.process.wait(timeout=120)
        if self.process.returncode != 0:
            raise RuntimeError("recurrent service failed")


def observe(client: Client, seed: int, started: bool) -> None:
    client.request(
        {
            "op": "observe",
            "jpeg_base64": jpeg(seed),
            "frame_id": seed + 1,
            "controller": [
                float((seed + value) % 2) for value in range(3)
            ],
            "context": [
                math.sin(seed * 0.2 + value * 0.1)
                for value in range(5)
            ],
            "episode_started": started,
        }
    )


def transition(client: Client, seed: int) -> dict:
    observe(client, seed, True)
    observe(client, seed + 1, False)
    client.request(
        {
            "op": "begin",
            "action": 0,
            "character": seed % 4,
            "opponent": seed % 6,
        }
    )
    observe(client, seed + 2, False)
    return client.request(
        {
            "op": "close",
            "duration": 2,
            "reward": 0.75 if seed % 2 else -0.25,
            "terminal": False,
        }
    )


def main() -> int:
    if not BASE_CHECKPOINT.is_file():
        raise RuntimeError(
            f"ResNet50 checkpoint not found: {BASE_CHECKPOINT}"
        )
    with tempfile.TemporaryDirectory(
        prefix="gameagent_trainable_resnet_"
    ) as temporary:
        checkpoint = Path(temporary) / "joint.pt"
        training = Client(checkpoint, True, True, False)
        initial = float(training.ready["resnet_parameter_checksum"])
        transition(training, 1)
        result = transition(training, 2)
        updated = float(result["resnet_parameter_checksum"])
        if (
            result["training_updates"] != 1
            or result["resnet_training_updates"] != 1
            or not result["training_performed"]
            or updated == initial
        ):
            raise RuntimeError("ResNet50 stage 4 did not update")
        training.shutdown()
        if not checkpoint.is_file():
            raise RuntimeError("joint checkpoint was not saved")

        evaluation = Client(checkpoint, False, False, True)
        before = evaluation.request({"op": "stats"})
        observe(evaluation, 20, True)
        q_values = evaluation.request(
            {
                "op": "q_values",
                "character": 0,
                "opponent": 1,
                "action_count": 3,
            }
        )["q_values"]
        after = evaluation.request({"op": "stats"})
        evaluation.shutdown()
        if (
            not before["checkpoint_loaded"]
            or not before["resnet_checkpoint_loaded"]
            or before["resnet_training_updates"]
            != after["resnet_training_updates"]
            or before["resnet_parameter_checksum"]
            != after["resnet_parameter_checksum"]
            or not all(math.isfinite(float(value)) for value in q_values)
        ):
            raise RuntimeError(
                "read-only joint checkpoint evaluation changed state"
            )
        print(
            "[INFO] "
            f"device={training.ready['device']} "
            f"recurrent_updates={result['training_updates']} "
            f"resnet_updates={result['resnet_training_updates']} "
            f"resnet_checksum_delta={updated - initial:.9f}"
        )
        print(
            "[PASS] joint ResNet50 stage-4 + LSTM BPTT training passed"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
