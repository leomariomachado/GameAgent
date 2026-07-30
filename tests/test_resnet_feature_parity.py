#!/usr/bin/env python3
"""Compare BaseAI CUDA and PyTorch reconstructions on the same JPEG."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.recurrent_dqn_service import TrainableVisualEncoder


CHECKPOINT = Path(
    os.environ.get(
        "GAMEAGENT_RESNET50_CHECKPOINT",
        "/workspace/resnet50_best.ckpt",
    )
)
EXTRACTOR = (
    Path(os.environ.get("GAMEAGENT_BUILD_DIR", ROOT / "buildrelease"))
    / "test_gameagent_resnet50_feature_extractor"
)


def main() -> int:
    if not CHECKPOINT.is_file() or not EXTRACTOR.is_file():
        raise RuntimeError("checkpoint or BaseAI feature extractor is missing")
    device_id = int(os.environ.get("GAMEAGENT_DEVICE_ID", "0"))
    device = torch.device(f"cuda:{device_id}")
    with tempfile.TemporaryDirectory(
        prefix="gameagent_resnet_parity_"
    ) as temporary:
        root = Path(temporary)
        capture = root / "captures" / "session_1" / "connection_1"
        output = root / "features"
        capture.mkdir(parents=True)
        y, x = np.mgrid[0:448, 0:512]
        pixels = np.stack(
            (
                (x + 2 * y) % 256,
                (3 * x + y) % 256,
                (5 * x + 7 * y) % 256,
            ),
            axis=2,
        ).astype(np.uint8)
        image_path = capture / "frame_1.jpg"
        Image.fromarray(pixels, mode="RGB").save(
            image_path, format="JPEG", quality=92
        )
        (capture / "frames.jsonl").write_text(
            json.dumps(
                {
                    "file": image_path.name,
                    "frame_id": 1,
                    "client_timestamp_ms": 1,
                    "received_timestamp_ms": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment.update(
            {
                "GAMEAGENT_RESNET50_CHECKPOINT": str(CHECKPOINT),
                "GAMEAGENT_CAPTURE_ROOT": str(root / "captures"),
                "GAMEAGENT_FEATURE_OUTPUT_DIR": str(output),
                "GAMEAGENT_FEATURE_LIMIT": "1",
                "GAMEAGENT_DEVICE_ID": str(device_id),
            }
        )
        subprocess.run(
            [str(EXTRACTOR)],
            env=environment,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        manifests = list(output.glob("run_*/features.jsonl"))
        if len(manifests) != 1:
            raise RuntimeError("BaseAI parity feature manifest is missing")
        record = json.loads(
            manifests[0].read_text(encoding="utf-8").splitlines()[0]
        )
        baseai = np.fromfile(record["feature_path"], dtype=np.float32)
        encoder = TrainableVisualEncoder(CHECKPOINT, device)
        pytorch = (
            encoder.features(
                [image_path.read_bytes()], require_gradient=False
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        difference = pytorch - baseai
        cosine = float(
            np.dot(pytorch, baseai)
            / (np.linalg.norm(pytorch) * np.linalg.norm(baseai))
        )
        relative_l2 = float(
            np.linalg.norm(difference) / np.linalg.norm(baseai)
        )
        maximum = float(np.max(np.abs(difference)))
        mean = float(np.mean(np.abs(difference)))
        print(
            "[PARITY] "
            f"cosine={cosine:.8f} relative_l2={relative_l2:.8f} "
            f"mean_abs={mean:.8f} max_abs={maximum:.8f}"
        )
        if cosine < 0.995 or relative_l2 > 0.12:
            raise RuntimeError(
                "PyTorch reconstruction does not match BaseAI features"
            )
    print("[PASS] BaseAI/PyTorch ResNet50 feature parity passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
