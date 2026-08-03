#!/usr/bin/env python3
from __future__ import annotations

import copy
import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.convert_recurrent_checkpoint_to_baseai import (  # noqa: E402
    recurrent_tensors,
    write_checkpoint,
)
from scripts.recurrent_dqn_service import (  # noqa: E402
    NetworkConfig,
    RecurrentQNetwork,
)


def main() -> int:
    if len(sys.argv) not in (3, 4):
        raise SystemExit(
            "usage: generator CHECKPOINT FIXTURE [TRAINED_CHECKPOINT]"
        )
    checkpoint_path = Path(sys.argv[1])
    fixture_path = Path(sys.argv[2])
    trained_checkpoint_path = (
        Path(sys.argv[3]) if len(sys.argv) == 4 else None
    )
    torch.manual_seed(90817)
    np.random.seed(90817)
    config = NetworkConfig(
        visual_size=32,
        projection_size=8,
        controller_size=4,
        temporal_hidden_size=12,
        context_size=6,
        strategy_hidden_size=10,
        sequence_capacity=4,
        character_embedding_size=3,
        opponent_embedding_size=4,
        opponent_count=5,
        action_sizes=(3, 4),
    )
    projection = nn.Linear(config.visual_size, config.projection_size)
    nn.init.xavier_uniform_(projection.weight)
    nn.init.zeros_(projection.bias)
    online_a = RecurrentQNetwork(config)
    online_b = RecurrentQNetwork(config)
    target_a = copy.deepcopy(online_a)
    target_b = copy.deepcopy(online_b)
    payload = {
        "service_version": 2,
        "config": dataclasses.asdict(config),
        "projection": projection.state_dict(),
        "online_a": online_a.state_dict(),
        "online_b": online_b.state_dict(),
        "target_a": target_a.state_dict(),
        "target_b": target_b.state_dict(),
        "training_updates": 73,
        "last_loss": 0.125,
    }
    write_checkpoint(
        checkpoint_path,
        recurrent_tensors(payload),
        payload["training_updates"],
        payload["last_loss"],
    )

    visuals: list[np.ndarray] = []
    controllers: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    recurrent_steps: list[torch.Tensor] = []
    with torch.inference_mode():
        for step in range(config.sequence_capacity):
            visual = np.asarray(
                [
                    np.sin(step * 0.19 + index * 0.041)
                    for index in range(config.visual_size)
                ],
                dtype=np.float32,
            )
            controller = np.asarray(
                [
                    1.0 if (step + index) % 3 == 0 else 0.0
                    for index in range(config.controller_size)
                ],
                dtype=np.float32,
            )
            context = np.asarray(
                [
                    (step + index) / 17.0
                    for index in range(config.context_size)
                ],
                dtype=np.float32,
            )
            projected = torch.tanh(
                projection(torch.from_numpy(visual))
            )
            recurrent_input = torch.cat(
                (projected, torch.from_numpy(controller))
            ).to(dtype=torch.float16).to(dtype=torch.float32)
            visuals.append(visual)
            controllers.append(controller)
            contexts.append(context)
            recurrent_steps.append(recurrent_input)
        sequence = torch.stack(recurrent_steps).unsqueeze(0)
        lengths = torch.tensor([config.sequence_capacity])
        context_tensor = torch.from_numpy(contexts[-1]).unsqueeze(0)
        characters = torch.tensor([1])
        opponents = torch.tensor([2])
        latent_a = online_a.latent(
            sequence, lengths, context_tensor, characters, opponents
        )
        latent_b = online_b.latent(
            sequence, lengths, context_tensor, characters, opponents
        )
        q_a = online_a.padded_q(latent_a, characters)[0, :4]
        q_b = online_b.padded_q(latent_b, characters)[0, :4]
        expected = torch.minimum(q_a, q_b).cpu().numpy()

    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    with fixture_path.open("w", encoding="utf-8") as output:
        output.write("32 8 4 12 6 10 4 3 4 5 2 3 4\n")
        output.write("1 2 4\n")
        for visual, controller, context in zip(
            visuals, controllers, contexts
        ):
            output.write(" ".join(map(str, visual.tolist())) + "\n")
            output.write(" ".join(map(str, controller.tolist())) + "\n")
            output.write(" ".join(map(str, context.tolist())) + "\n")
        output.write(" ".join(map(str, expected.tolist())) + "\n")
        next_step = config.sequence_capacity
        next_visual = np.asarray(
            [
                np.sin(next_step * 0.19 + index * 0.041)
                for index in range(config.visual_size)
            ],
            dtype=np.float32,
        )
        next_controller = np.asarray(
            [
                1.0 if (next_step + index) % 3 == 0 else 0.0
                for index in range(config.controller_size)
            ],
            dtype=np.float32,
        )
        next_context = np.asarray(
            [
                (next_step + index) / 17.0
                for index in range(config.context_size)
            ],
            dtype=np.float32,
        )
        output.write(" ".join(map(str, next_visual.tolist())) + "\n")
        output.write(" ".join(map(str, next_controller.tolist())) + "\n")
        output.write(" ".join(map(str, next_context.tolist())) + "\n")
    print(
        f"[PASS] C++ recurrent parity fixture generated: {fixture_path}"
    )
    if trained_checkpoint_path is not None:
        with torch.enable_grad():
            next_projected = torch.tanh(
                projection(torch.from_numpy(next_visual))
            )
            next_input = torch.cat(
                (next_projected, torch.from_numpy(next_controller))
            ).to(dtype=torch.float16).to(dtype=torch.float32)
            state_sequence = torch.stack(recurrent_steps)
            next_sequence = torch.cat(
                (state_sequence[1:], next_input.unsqueeze(0)), dim=0
            )
            batch_state = state_sequence.unsqueeze(0).repeat(4, 1, 1)
            batch_next = next_sequence.unsqueeze(0).repeat(4, 1, 1)
            batch_lengths = torch.full((4,), config.sequence_capacity)
            batch_context = torch.from_numpy(
                np.repeat(contexts[-1][None, :], 4, axis=0)
            )
            batch_next_context = torch.from_numpy(
                np.repeat(next_context[None, :], 4, axis=0)
            )
            batch_characters = torch.full((4,), 1, dtype=torch.int64)
            batch_opponents = torch.full((4,), 2, dtype=torch.int64)
            # Materialize the successor path as well so the fixture exercises
            # the same tensor shapes even though terminal targets do not
            # bootstrap from it.
            for network in (online_a, online_b):
                network.latent(
                    batch_next,
                    batch_lengths,
                    batch_next_context,
                    batch_characters,
                    batch_opponents,
                )
            latent_a = online_a.latent(
                batch_state,
                batch_lengths,
                batch_context,
                batch_characters,
                batch_opponents,
            )
            latent_b = online_b.latent(
                batch_state,
                batch_lengths,
                batch_context,
                batch_characters,
                batch_opponents,
            )
            predicted_a = online_a.padded_q(
                latent_a, batch_characters
            )[:, 0]
            predicted_b = online_b.padded_q(
                latent_b, batch_characters
            )[:, 0]
            targets = torch.ones(4)
            loss = 0.5 * (
                torch.nn.functional.smooth_l1_loss(predicted_a, targets) +
                torch.nn.functional.smooth_l1_loss(predicted_b, targets)
            )
            optimizer = torch.optim.AdamW(
                list(online_a.parameters()) + list(online_b.parameters()),
                lr=1.0e-4,
                weight_decay=1.0e-5,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(online_a.parameters()) + list(online_b.parameters()),
                5.0,
            )
            optimizer.step()
        payload["online_a"] = online_a.state_dict()
        payload["online_b"] = online_b.state_dict()
        payload["training_updates"] = 74
        payload["last_loss"] = float(loss.detach())
        write_checkpoint(
            trained_checkpoint_path,
            recurrent_tensors(payload),
            payload["training_updates"],
            payload["last_loss"],
        )
        print(
            "[PASS] PyTorch one-step BPTT reference generated: "
            f"{trained_checkpoint_path} loss={float(loss.detach()):.8f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
