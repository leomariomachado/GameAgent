#!/usr/bin/env python3
"""Persistent GPU recurrent Double-DQN service for GameAgent.

The C++ server keeps ownership of controller timing and option programs. This
process owns the trainable temporal encoder and Q network:

    frozen ResNet50 feature -> frozen projection -> trainable LSTM
      + fight/controller context + character/opponent embeddings
      -> shared strategy trunk -> character action head

One JSON request and response are exchanged per line on stdin/stdout. Human
readable diagnostics are written only to stderr so they cannot corrupt the
protocol.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


SERVICE_VERSION = 1


@dataclass(frozen=True)
class NetworkConfig:
    visual_size: int
    projection_size: int
    controller_size: int
    temporal_hidden_size: int
    context_size: int
    strategy_hidden_size: int
    sequence_capacity: int
    character_embedding_size: int
    opponent_embedding_size: int
    opponent_count: int
    action_sizes: tuple[int, ...]


@dataclass
class Transition:
    state_sequence: np.ndarray
    state_context: np.ndarray
    next_sequence: np.ndarray
    next_context: np.ndarray
    action: int
    duration: int
    reward: float
    terminal: bool
    character: int
    opponent: int


class RecurrentQNetwork(nn.Module):
    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        recurrent_input = config.projection_size + config.controller_size
        self.lstm = nn.LSTM(
            recurrent_input,
            config.temporal_hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.character_embedding = nn.Embedding(
            len(config.action_sizes), config.character_embedding_size
        )
        self.opponent_embedding = nn.Embedding(
            config.opponent_count, config.opponent_embedding_size
        )
        strategy_input = (
            config.temporal_hidden_size
            + config.context_size
            + config.character_embedding_size
            + config.opponent_embedding_size
        )
        self.strategy = nn.Linear(
            strategy_input, config.strategy_hidden_size
        )
        self.heads = nn.ModuleList(
            nn.Linear(config.strategy_hidden_size, actions)
            for actions in config.action_sizes
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, parameter in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)
        with torch.no_grad():
            hidden = self.config.temporal_hidden_size
            self.lstm.bias_ih_l0[hidden : 2 * hidden].fill_(1.0)
        nn.init.normal_(
            self.character_embedding.weight,
            std=1.0 / math.sqrt(self.config.character_embedding_size),
        )
        nn.init.normal_(
            self.opponent_embedding.weight,
            std=1.0 / math.sqrt(self.config.opponent_embedding_size),
        )
        nn.init.kaiming_uniform_(self.strategy.weight, nonlinearity="relu")
        nn.init.zeros_(self.strategy.bias)
        for head in self.heads:
            nn.init.kaiming_uniform_(head.weight, nonlinearity="linear")
            nn.init.zeros_(head.bias)

    def latent(
        self,
        sequences: torch.Tensor,
        lengths: torch.Tensor,
        context: torch.Tensor,
        characters: torch.Tensor,
        opponents: torch.Tensor,
    ) -> torch.Tensor:
        packed = pack_padded_sequence(
            sequences,
            lengths.to(device="cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        joined = torch.cat(
            (
                hidden[-1],
                context,
                self.character_embedding(characters),
                self.opponent_embedding(opponents),
            ),
            dim=1,
        )
        return torch.relu(self.strategy(joined))

    def padded_q(
        self, latent: torch.Tensor, characters: torch.Tensor
    ) -> torch.Tensor:
        maximum = max(self.config.action_sizes)
        rows: list[torch.Tensor] = []
        for row in range(latent.shape[0]):
            character = int(characters[row].item())
            values = self.heads[character](latent[row : row + 1]).squeeze(0)
            if values.numel() < maximum:
                values = torch.cat(
                    (
                        values,
                        torch.full(
                            (maximum - values.numel(),),
                            -torch.inf,
                            dtype=values.dtype,
                            device=values.device,
                        ),
                    )
                )
            rows.append(values)
        return torch.stack(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--training", type=int, choices=(0, 1), default=1)
    parser.add_argument("--checkpoint-writes", type=int, choices=(0, 1), default=1)
    parser.add_argument("--resume", type=int, choices=(0, 1), default=1)
    parser.add_argument("--visual-size", type=int, default=2048)
    parser.add_argument("--projection-size", type=int, default=256)
    parser.add_argument("--controller-size", type=int, default=12)
    parser.add_argument("--temporal-hidden-size", type=int, default=256)
    parser.add_argument("--context-size", type=int, default=15)
    parser.add_argument("--strategy-hidden-size", type=int, default=128)
    parser.add_argument("--sequence-capacity", type=int, default=16)
    parser.add_argument("--character-embedding-size", type=int, default=16)
    parser.add_argument("--opponent-embedding-size", type=int, default=16)
    parser.add_argument("--opponent-count", type=int, default=28)
    parser.add_argument("--action-sizes", default="20,22,18,23")
    parser.add_argument("--replay-capacity", type=int, default=20000)
    parser.add_argument("--minimum-replay-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-update", type=int, default=500)
    parser.add_argument("--updates-per-transition", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0x524C444E)
    parser.add_argument("--initial-updates", type=int, default=0)
    parser.add_argument("--persist-replay-limit", type=int, default=5000)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="test-only fallback; production recurrent training requires CUDA",
    )
    return parser.parse_args()


class Service:
    def __init__(self, arguments: argparse.Namespace) -> None:
        action_sizes = tuple(
            int(value) for value in arguments.action_sizes.split(",")
        )
        if (
            not action_sizes
            or any(value <= 0 for value in action_sizes)
            or arguments.sequence_capacity <= 0
            or arguments.minimum_replay_size <= 0
            or arguments.batch_size <= 0
        ):
            raise ValueError("invalid recurrent service dimensions")
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{arguments.device_id}")
        elif arguments.allow_cpu:
            self.device = torch.device("cpu")
        else:
            raise RuntimeError(
                "CUDA is required for trainable recurrent RL; "
                "use --allow-cpu only in tests"
            )

        random.seed(arguments.seed)
        np.random.seed(arguments.seed)
        torch.manual_seed(arguments.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(arguments.seed)
        self.arguments = arguments
        self.config = NetworkConfig(
            visual_size=arguments.visual_size,
            projection_size=arguments.projection_size,
            controller_size=arguments.controller_size,
            temporal_hidden_size=arguments.temporal_hidden_size,
            context_size=arguments.context_size,
            strategy_hidden_size=arguments.strategy_hidden_size,
            sequence_capacity=arguments.sequence_capacity,
            character_embedding_size=arguments.character_embedding_size,
            opponent_embedding_size=arguments.opponent_embedding_size,
            opponent_count=arguments.opponent_count,
            action_sizes=action_sizes,
        )
        self.projection = nn.Linear(
            self.config.visual_size,
            self.config.projection_size,
        ).to(self.device)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.projection.requires_grad_(False)
        self.projection.eval()

        self.online = RecurrentQNetwork(self.config).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.target.requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(),
            lr=arguments.learning_rate,
            weight_decay=1.0e-5,
        )
        self.replay: collections.deque[Transition] = collections.deque(
            maxlen=arguments.replay_capacity
        )
        self.matchup_counts: collections.Counter[tuple[int, int]] = (
            collections.Counter()
        )
        self.current_sequence: collections.deque[np.ndarray] = collections.deque(
            maxlen=self.config.sequence_capacity
        )
        self.current_context = np.zeros(
            self.config.context_size, dtype=np.float32
        )
        self.active: dict[str, Any] | None = None
        self.training_updates = int(arguments.initial_updates)
        self.last_loss = 0.0
        self.checkpoint_loaded = False
        self.parameter_checksum = 0.0
        if arguments.resume and arguments.checkpoint.is_file():
            self.load(arguments.checkpoint)
        self.refresh_checksum()

    def refresh_checksum(self) -> None:
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        with torch.no_grad():
            for index, parameter in enumerate(self.online.parameters(), 1):
                total += parameter.to(dtype=torch.float64).sum() * index
        self.parameter_checksum = float(total.cpu())

    def project_observation(
        self, visual: list[float], controller: list[float]
    ) -> np.ndarray:
        if (
            len(visual) != self.config.visual_size
            or len(controller) != self.config.controller_size
        ):
            raise ValueError("recurrent observation shape mismatch")
        visual_tensor = torch.tensor(
            visual, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.inference_mode():
            projected = torch.tanh(self.projection(visual_tensor)).squeeze(0)
        controller_tensor = torch.tensor(
            controller, dtype=torch.float32, device=self.device
        )
        return (
            torch.cat((projected, controller_tensor))
            .to(device="cpu", dtype=torch.float16)
            .numpy()
            .copy()
        )

    def snapshot(self) -> np.ndarray:
        if not self.current_sequence:
            raise RuntimeError("recurrent sequence is empty")
        return np.stack(tuple(self.current_sequence)).astype(
            np.float16, copy=True
        )

    def observe(self, request: dict[str, Any]) -> dict[str, Any]:
        if bool(request.get("episode_started", False)):
            self.current_sequence.clear()
            self.active = None
        context = np.asarray(request["context"], dtype=np.float32)
        if context.shape != (self.config.context_size,):
            raise ValueError("recurrent context shape mismatch")
        recurrent_input = self.project_observation(
            request["visual"], request["controller"]
        )
        self.current_sequence.append(recurrent_input)
        self.current_context = context.copy()
        return self.stats()

    def batch(
        self, transitions: list[Transition], next_state: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        maximum_steps = max(
            (
                transition.next_sequence
                if next_state
                else transition.state_sequence
            ).shape[0]
            for transition in transitions
        )
        input_size = self.config.projection_size + self.config.controller_size
        sequences = np.zeros(
            (len(transitions), maximum_steps, input_size),
            dtype=np.float32,
        )
        lengths = np.empty(len(transitions), dtype=np.int64)
        contexts = np.empty(
            (len(transitions), self.config.context_size), dtype=np.float32
        )
        characters = np.empty(len(transitions), dtype=np.int64)
        opponents = np.empty(len(transitions), dtype=np.int64)
        for row, transition in enumerate(transitions):
            sequence = (
                transition.next_sequence
                if next_state
                else transition.state_sequence
            ).astype(np.float32, copy=False)
            context = (
                transition.next_context
                if next_state
                else transition.state_context
            )
            sequences[row, : sequence.shape[0]] = sequence
            lengths[row] = sequence.shape[0]
            contexts[row] = context
            characters[row] = transition.character
            opponents[row] = transition.opponent
        return (
            torch.from_numpy(sequences).to(self.device),
            torch.from_numpy(lengths),
            torch.from_numpy(contexts).to(self.device),
            torch.from_numpy(characters).to(self.device),
            torch.from_numpy(opponents).to(self.device),
        )

    def latent_for(
        self,
        network: RecurrentQNetwork,
        transitions: list[Transition],
        next_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences, lengths, context, characters, opponents = self.batch(
            transitions, next_state
        )
        return (
            network.latent(
                sequences, lengths, context, characters, opponents
            ),
            characters,
        )

    def balanced_sample(self) -> list[Transition]:
        buckets: dict[tuple[int, int], list[Transition]] = (
            collections.defaultdict(list)
        )
        for transition in self.replay:
            buckets[(transition.character, transition.opponent)].append(
                transition
            )
        active = tuple(buckets.values())
        return [
            random.choice(random.choice(active))
            for _ in range(self.arguments.batch_size)
        ]

    def append_replay(self, transition: Transition) -> None:
        if len(self.replay) == self.replay.maxlen:
            removed = self.replay[0]
            removed_key = (removed.character, removed.opponent)
            self.matchup_counts[removed_key] -= 1
            if self.matchup_counts[removed_key] <= 0:
                del self.matchup_counts[removed_key]
        self.replay.append(transition)
        self.matchup_counts[
            (transition.character, transition.opponent)
        ] += 1

    def train_once(self) -> bool:
        minimum = max(
            self.arguments.minimum_replay_size,
            self.arguments.batch_size,
        )
        if not self.arguments.training or len(self.replay) < minimum:
            return False
        transitions = self.balanced_sample()
        self.online.train()
        current_latent, characters = self.latent_for(
            self.online, transitions, False
        )
        current_q = self.online.padded_q(current_latent, characters)
        actions = torch.tensor(
            [value.action for value in transitions],
            dtype=torch.int64,
            device=self.device,
        )
        predicted = current_q.gather(1, actions[:, None]).squeeze(1)

        with torch.no_grad():
            online_next_latent, next_characters = self.latent_for(
                self.online, transitions, True
            )
            online_next = self.online.padded_q(
                online_next_latent, next_characters
            )
            best_actions = online_next.argmax(dim=1)
            target_next_latent, target_characters = self.latent_for(
                self.target, transitions, True
            )
            target_next = self.target.padded_q(
                target_next_latent, target_characters
            )
            bootstrap = target_next.gather(
                1, best_actions[:, None]
            ).squeeze(1)
            rewards = torch.tensor(
                [value.reward for value in transitions],
                dtype=torch.float32,
                device=self.device,
            )
            durations = torch.tensor(
                [value.duration for value in transitions],
                dtype=torch.float32,
                device=self.device,
            )
            nonterminal = torch.tensor(
                [not value.terminal for value in transitions],
                dtype=torch.float32,
                device=self.device,
            )
            discount = torch.pow(
                torch.full_like(durations, self.arguments.gamma), durations
            )
            expected = rewards + nonterminal * discount * bootstrap

        loss = torch.nn.functional.smooth_l1_loss(predicted, expected)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.arguments.gradient_clip
        )
        self.optimizer.step()
        self.training_updates += 1
        self.last_loss = float(loss.detach().cpu())
        if self.training_updates % self.arguments.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())
        self.online.eval()
        self.refresh_checksum()
        return True

    def q_values(self, request: dict[str, Any]) -> dict[str, Any]:
        character = int(request["character"])
        opponent = int(request["opponent"])
        action_count = int(request["action_count"])
        if (
            character < 0
            or character >= len(self.config.action_sizes)
            or opponent < 0
            or opponent >= self.config.opponent_count
            or action_count != self.config.action_sizes[character]
        ):
            raise ValueError("invalid recurrent policy identity/action count")
        sequence = self.snapshot()
        synthetic = Transition(
            state_sequence=sequence,
            state_context=self.current_context.copy(),
            next_sequence=sequence,
            next_context=self.current_context.copy(),
            action=0,
            duration=1,
            reward=0.0,
            terminal=False,
            character=character,
            opponent=opponent,
        )
        self.online.eval()
        with torch.inference_mode():
            latent, characters = self.latent_for(
                self.online, [synthetic], False
            )
            values = self.online.padded_q(
                latent, characters
            )[0, :action_count]
        response = self.stats()
        response["q_values"] = values.cpu().tolist()
        return response

    def begin(self, request: dict[str, Any]) -> dict[str, Any]:
        action = int(request["action"])
        character = int(request["character"])
        opponent = int(request["opponent"])
        if (
            action < 0
            or action >= self.config.action_sizes[character]
        ):
            raise ValueError("invalid recurrent transition action")
        self.active = {
            "sequence": self.snapshot(),
            "context": self.current_context.copy(),
            "action": action,
            "character": character,
            "opponent": opponent,
        }
        return self.stats()

    def close(self, request: dict[str, Any]) -> dict[str, Any]:
        trained = False
        if self.active is not None and self.arguments.training:
            duration = max(1, int(request["duration"]))
            reward = float(request["reward"])
            if not math.isfinite(reward):
                raise ValueError("non-finite recurrent reward")
            transition = Transition(
                state_sequence=self.active["sequence"],
                state_context=self.active["context"],
                next_sequence=self.snapshot(),
                next_context=self.current_context.copy(),
                action=self.active["action"],
                duration=duration,
                reward=reward,
                terminal=bool(request["terminal"]),
                character=self.active["character"],
                opponent=self.active["opponent"],
            )
            self.append_replay(transition)
            for _ in range(self.arguments.updates_per_transition):
                trained = self.train_once() or trained
        self.active = None
        response = self.stats()
        response["training_performed"] = trained
        return response

    def reset(self) -> dict[str, Any]:
        self.current_sequence.clear()
        self.current_context.fill(0.0)
        self.active = None
        return self.stats()

    def checkpoint_payload(self, include_replay: bool) -> dict[str, Any]:
        replay: list[Transition] = []
        if include_replay and self.arguments.persist_replay_limit > 0:
            replay = list(self.replay)[
                -self.arguments.persist_replay_limit :
            ]
        return {
            "service_version": SERVICE_VERSION,
            "config": asdict(self.config),
            "projection": self.projection.state_dict(),
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "training_updates": self.training_updates,
            "last_loss": self.last_loss,
            "parameter_checksum": self.parameter_checksum,
            "replay": replay,
        }

    def save(self, path: Path, include_replay: bool) -> None:
        if not self.arguments.checkpoint_writes:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save(
                self.checkpoint_payload(include_replay),
                temporary_path,
            )
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def load(self, path: Path) -> None:
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=False
        )
        if (
            int(checkpoint.get("service_version", 0)) != SERVICE_VERSION
            or checkpoint.get("config") != asdict(self.config)
        ):
            raise RuntimeError("recurrent checkpoint architecture mismatch")
        self.projection.load_state_dict(checkpoint["projection"])
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_updates = int(checkpoint["training_updates"])
        self.last_loss = float(checkpoint.get("last_loss", 0.0))
        for transition in checkpoint.get("replay", []):
            self.append_replay(transition)
        self.checkpoint_loaded = True
        self.online.eval()
        self.target.eval()

    def stats(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "training_updates": self.training_updates,
            "replay_size": len(self.replay),
            "matchup_count": len(self.matchup_counts),
            "last_loss": self.last_loss,
            "parameter_checksum": self.parameter_checksum,
            "checkpoint_loaded": self.checkpoint_loaded,
            "sequence_size": len(self.current_sequence),
        }

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("op")
        if operation == "observe":
            return self.observe(request)
        if operation == "q_values":
            return self.q_values(request)
        if operation == "begin":
            return self.begin(request)
        if operation == "close":
            return self.close(request)
        if operation == "reset":
            return self.reset()
        if operation == "save":
            path = Path(request.get("path") or self.arguments.checkpoint)
            self.save(path, bool(request.get("include_replay", False)))
            return self.stats()
        if operation == "stats":
            return self.stats()
        if operation == "shutdown":
            self.save(self.arguments.checkpoint, include_replay=True)
            response = self.stats()
            response["shutdown"] = True
            return response
        raise ValueError(f"unknown recurrent service operation: {operation}")


def send(value: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    arguments = parse_arguments()
    service = Service(arguments)
    print(
        "[RECURRENT] "
        f"device={service.device} training={arguments.training} "
        f"sequence={service.config.sequence_capacity} "
        f"hidden={service.config.temporal_hidden_size} "
        f"replay={len(service.replay)} updates={service.training_updates} "
        f"checkpoint_loaded={int(service.checkpoint_loaded)}",
        file=sys.stderr,
        flush=True,
    )
    ready = service.stats()
    ready.update({"ok": True, "ready": True, "service_version": SERVICE_VERSION})
    send(ready)
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = service.handle(request)
            response["ok"] = True
            send(response)
            if response.get("shutdown"):
                return 0
        except Exception as error:  # protocol boundary
            print(f"[RECURRENT][ERROR] {error}", file=sys.stderr, flush=True)
            send({"ok": False, "error": str(error)})
    if arguments.checkpoint_writes:
        service.save(arguments.checkpoint, include_replay=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
