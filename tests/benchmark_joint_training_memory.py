#!/usr/bin/env python3
"""One production-shape joint update with CUDA peak-memory reporting."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.recurrent_dqn_service import Service, Transition


def main() -> int:
    checkpoint = Path(
        os.environ.get(
            "GAMEAGENT_RESNET50_CHECKPOINT",
            "/workspace/resnet50_best.ckpt",
        )
    )
    device_id = int(os.environ.get("GAMEAGENT_DEVICE_ID", "0"))
    with tempfile.TemporaryDirectory(
        prefix="gameagent_joint_memory_"
    ) as temporary:
        arguments = SimpleNamespace(
            checkpoint=Path(temporary) / "unused.pt",
            device_id=device_id,
            training=1,
            checkpoint_writes=0,
            resume=0,
            visual_size=2048,
            projection_size=256,
            controller_size=12,
            temporal_hidden_size=256,
            context_size=30,
            strategy_hidden_size=128,
            sequence_capacity=16,
            character_embedding_size=16,
            opponent_embedding_size=16,
            opponent_count=28,
            action_sizes="20,22,18,23",
            replay_capacity=20_000,
            minimum_replay_size=64,
            batch_size=64,
            target_update=500,
            updates_per_transition=1,
            gamma=0.99,
            learning_rate=1.0e-4,
            trainable_resnet=1,
            resnet_checkpoint=checkpoint,
            resnet_learning_rate=1.0e-6,
            resnet_batch_size=4,
            resnet_update_interval=16,
            resnet_replay_images=4096,
            gradient_clip=5.0,
            seed=12345,
            initial_updates=15,
            persist_replay_limit=0,
            allow_cpu=False,
        )
        service = Service(arguments)
        image = Image.new("RGB", (512, 448), (80, 130, 190))
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=90)
        jpeg = encoded.getvalue()
        random = np.random.default_rng(12345)
        for index in range(64):
            state = random.standard_normal(
                (16, 268), dtype=np.float32
            ).astype(np.float16)
            following = random.standard_normal(
                (16, 268), dtype=np.float32
            ).astype(np.float16)
            service.append_replay(
                Transition(
                    state_sequence=state,
                    state_context=random.standard_normal(
                        30, dtype=np.float32
                    ),
                    next_sequence=following,
                    next_context=random.standard_normal(
                        30, dtype=np.float32
                    ),
                    action=0,
                    duration=3,
                    reward=0.25,
                    terminal=False,
                    character=index % 4,
                    opponent=index % 28,
                    state_jpeg=jpeg,
                )
            )
        torch.cuda.synchronize(device_id)
        torch.cuda.reset_peak_memory_stats(device_id)
        started = time.perf_counter()
        trained = service.train_once()
        torch.cuda.synchronize(device_id)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        peak_allocated = torch.cuda.max_memory_allocated(device_id)
        peak_reserved = torch.cuda.max_memory_reserved(device_id)
        total = torch.cuda.get_device_properties(device_id).total_memory
        print(
            "[BENCHMARK] "
            f"joint_update_ms={elapsed_ms:.3f} "
            f"peak_allocated_gib={peak_allocated / 2**30:.3f} "
            f"peak_reserved_gib={peak_reserved / 2**30:.3f} "
            f"gpu_total_gib={total / 2**30:.3f} "
            f"resnet_updates={service.resnet_training_updates}"
        )
        if (
            not trained
            or service.resnet_training_updates != 1
            or peak_reserved >= total * 0.85
        ):
            raise RuntimeError(
                "production joint update exceeded safe GPU capacity"
            )
    print("[PASS] production-shape joint update fits GPU safely")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
