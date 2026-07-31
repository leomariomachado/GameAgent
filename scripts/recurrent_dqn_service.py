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
import base64
import collections
import copy
import io
import json
import math
import os
import random
import struct
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import resnet50


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
    # Compressed source frame for truncated end-to-end visual credit. Older
    # replay entries remain valid for LSTM/DQN learning when this is absent.
    state_jpeg: bytes | None = None


def read_baseai_resnet_tensors(path: Path) -> dict[str, np.ndarray]:
    """Read only the model section of a versioned BaseAI checkpoint."""

    dtype_map = {
        0: (np.dtype("<f4"), 4),
        1: (np.dtype("<f2"), 2),
        2: (np.dtype("<u2"), 2),
        3: (np.dtype("<i4"), 4),
        4: (np.dtype("u1"), 1),
    }

    def exact(handle: Any, size: int) -> bytes:
        value = handle.read(size)
        if len(value) != size:
            raise RuntimeError("truncated BaseAI checkpoint")
        return value

    def scalar(handle: Any, code: str) -> Any:
        return struct.unpack("<" + code, exact(handle, struct.calcsize(code)))[0]

    def text(handle: Any) -> str:
        size = int(scalar(handle, "Q"))
        if size > 1_048_576:
            raise RuntimeError("invalid BaseAI checkpoint string")
        return exact(handle, size).decode("utf-8")

    tensors: dict[str, np.ndarray] = {}
    with path.open("rb") as handle:
        if exact(handle, 8) != b"BASECKPT":
            raise RuntimeError(
                "trainable ResNet requires a versioned BaseAI checkpoint"
            )
        version = int(scalar(handle, "I"))
        if version not in (2, 3):
            raise RuntimeError(
                f"unsupported BaseAI checkpoint version: {version}"
            )
        scalar(handle, "Q")  # epoch
        scalar(handle, "Q")  # global step
        scalar(handle, "f")  # best accuracy
        scalar(handle, "f")  # best loss
        if version >= 3:
            scalar(handle, "Q")  # best epoch
            scalar(handle, "i")  # early-stopping bad epochs
        count = int(scalar(handle, "Q"))
        if count != 320:
            raise RuntimeError(
                f"expected 320 ResNet tensors, found {count}"
            )
        for _ in range(count):
            name = text(handle)
            dtype_value = int(scalar(handle, "I"))
            elements = int(scalar(handle, "Q"))
            if dtype_value not in dtype_map or elements < 0:
                raise RuntimeError("invalid BaseAI tensor metadata")
            dtype, width = dtype_map[dtype_value]
            raw = exact(handle, elements * width)
            if dtype_value != 0:
                raise RuntimeError(
                    f"ResNet tensor is not Float32: {name}"
                )
            tensors[name] = np.frombuffer(raw, dtype=dtype).copy()
    return tensors


def build_baseai_resnet50(path: Path) -> nn.Module:
    """Reconstruct torchvision's ResNet-50 from BaseAI tensor names."""

    tensors = read_baseai_resnet_tensors(path)
    model = resnet50(weights=None)
    state = model.state_dict()

    def assign(target: str, source: str) -> None:
        if source not in tensors or target not in state:
            raise RuntimeError(
                f"ResNet checkpoint mapping is missing {source} -> {target}"
            )
        values = torch.from_numpy(tensors[source])
        if values.numel() != state[target].numel():
            raise RuntimeError(
                f"ResNet tensor shape mismatch: {source} -> {target}"
            )
        state[target] = values.reshape(state[target].shape)

    def unit(source: str, target: str) -> None:
        assign(target + ".weight", source + ".weight")
        bias = tensors.get(source + ".bias")
        if bias is None or (
            bias.size and float(np.max(np.abs(bias))) > 1.0e-6
        ):
            raise RuntimeError(
                f"BaseAI convolution bias is unexpectedly non-zero: {source}"
            )
        assign(target.replace("conv", "bn") + ".weight", source + ".gamma")
        assign(target.replace("conv", "bn") + ".bias", source + ".beta")
        assign(
            target.replace("conv", "bn") + ".running_mean",
            source + ".running_mean",
        )
        assign(
            target.replace("conv", "bn") + ".running_var",
            source + ".running_var",
        )

    assign("conv1.weight", "stem.weight")
    stem_bias = tensors.get("stem.bias")
    if stem_bias is None or (
        stem_bias.size and float(np.max(np.abs(stem_bias))) > 1.0e-6
    ):
        raise RuntimeError("BaseAI stem bias is unexpectedly non-zero")
    assign("bn1.weight", "stem.gamma")
    assign("bn1.bias", "stem.beta")
    assign("bn1.running_mean", "stem.running_mean")
    assign("bn1.running_var", "stem.running_var")

    stage_sizes = (3, 4, 6, 3)
    global_block = 0
    for stage, count in enumerate(stage_sizes, 1):
        for local_block in range(count):
            source = f"blocks.{global_block}"
            target = f"layer{stage}.{local_block}"
            for convolution in ("conv1", "conv2", "conv3"):
                unit(
                    f"{source}.{convolution}",
                    f"{target}.{convolution}",
                )
            projection = f"{source}.projection"
            if projection + ".weight" in tensors:
                assign(
                    f"{target}.downsample.0.weight",
                    projection + ".weight",
                )
                projection_bias = tensors[projection + ".bias"]
                if (
                    projection_bias.size
                    and float(np.max(np.abs(projection_bias))) > 1.0e-6
                ):
                    raise RuntimeError(
                        f"BaseAI projection bias is non-zero: {projection}"
                    )
                assign(
                    f"{target}.downsample.1.weight",
                    projection + ".gamma",
                )
                assign(
                    f"{target}.downsample.1.bias",
                    projection + ".beta",
                )
                assign(
                    f"{target}.downsample.1.running_mean",
                    projection + ".running_mean",
                )
                assign(
                    f"{target}.downsample.1.running_var",
                    projection + ".running_var",
                )
            global_block += 1
    assign("fc.weight", "classifier.weight")
    assign("fc.bias", "classifier.bias")
    model.load_state_dict(state, strict=True)
    return model


class TrainableVisualEncoder:
    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        self.device = device
        self.model = build_baseai_resnet50(checkpoint).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        # Stage 4 contains the semantic high-level filters. Keeping stages
        # 1-3 and every BatchNorm statistic frozen is stable for online RL.
        for parameter in self.model.layer4.parameters():
            parameter.requires_grad_(True)
        self.trainable_parameters = [
            value for value in self.model.layer4.parameters()
            if value.requires_grad
        ]
        self.mean = torch.tensor(
            (0.485, 0.456, 0.406),
            dtype=torch.float32,
            device=device,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            (0.229, 0.224, 0.225),
            dtype=torch.float32,
            device=device,
        ).view(1, 3, 1, 1)

    def decode(self, values: list[bytes]) -> torch.Tensor:
        images: list[np.ndarray] = []
        for encoded in values:
            with Image.open(io.BytesIO(encoded)) as image:
                array = np.asarray(
                    image.convert("RGB"), dtype=np.uint8
                ).copy()
            if array.shape[:2] != (224, 224):
                array = self.resize_like_baseai(array)
            images.append(array)
        batch = torch.from_numpy(np.stack(images)).to(
            device=self.device, dtype=torch.float32
        )
        batch = batch.permute(0, 3, 1, 2).div_(255.0)
        return (batch - self.mean) / self.std

    @staticmethod
    def resize_like_baseai(image: np.ndarray) -> np.ndarray:
        """Match BaseAI's half-pixel bilinear resize and uint8 truncation."""

        height, width, channels = image.shape
        if channels != 3 or height <= 0 or width <= 0:
            raise ValueError("invalid RGB image for ResNet50")
        target = 224
        scale_x = np.float32(width / target)
        scale_y = np.float32(height / target)
        source_x = (
            (np.arange(target, dtype=np.float32) + np.float32(0.5))
            * scale_x
            - np.float32(0.5)
        )
        source_y = (
            (np.arange(target, dtype=np.float32) + np.float32(0.5))
            * scale_y
            - np.float32(0.5)
        )
        x0 = np.clip(np.floor(source_x).astype(np.int64), 0, width - 1)
        y0 = np.clip(np.floor(source_y).astype(np.int64), 0, height - 1)
        x1 = np.clip(x0 + 1, 0, width - 1)
        y1 = np.clip(y0 + 1, 0, height - 1)
        wx = (source_x - x0.astype(np.float32))[None, :, None]
        wy = (source_y - y0.astype(np.float32))[:, None, None]
        source = image.astype(np.float32, copy=False)
        top = (
            source[y0[:, None], x0[None, :]]
            + (
                source[y0[:, None], x1[None, :]]
                - source[y0[:, None], x0[None, :]]
            )
            * wx
        )
        bottom = (
            source[y1[:, None], x0[None, :]]
            + (
                source[y1[:, None], x1[None, :]]
                - source[y1[:, None], x0[None, :]]
            )
            * wx
        )
        resized = top + (bottom - top) * wy
        return np.clip(resized, 0.0, 255.0).astype(np.uint8)

    def features(
        self, encoded: list[bytes], require_gradient: bool
    ) -> torch.Tensor:
        images = self.decode(encoded)
        with torch.no_grad():
            value = self.model.conv1(images)
            value = self.model.bn1(value)
            value = self.model.relu(value)
            value = self.model.maxpool(value)
            value = self.model.layer1(value)
            value = self.model.layer2(value)
            value = self.model.layer3(value)
        context = (
            torch.enable_grad() if require_gradient else torch.no_grad()
        )
        with context:
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                value = self.model.layer4(value)
                value = self.model.avgpool(value)
                value = torch.flatten(value, 1)
        return value.float()


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
    parser.add_argument("--context-size", type=int, default=30)
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
    parser.add_argument(
        "--trainable-resnet", type=int, choices=(0, 1), default=0
    )
    parser.add_argument("--resnet-checkpoint", type=Path, default=Path())
    parser.add_argument(
        "--resnet-learning-rate", type=float, default=1.0e-6
    )
    parser.add_argument("--resnet-batch-size", type=int, default=4)
    parser.add_argument("--resnet-update-interval", type=int, default=16)
    parser.add_argument("--resnet-replay-images", type=int, default=4096)
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
            or (
                arguments.trainable_resnet
                and (
                    arguments.visual_size != 2048
                    or arguments.resnet_batch_size <= 0
                    or arguments.resnet_update_interval <= 0
                    or arguments.resnet_replay_images <= 0
                    or arguments.resnet_learning_rate <= 0.0
                    or not arguments.resnet_checkpoint.is_file()
                )
            )
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
        if arguments.trainable_resnet and self.device.type != "cuda":
            raise RuntimeError(
                "trainable ResNet50 requires CUDA"
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
        self.visual_encoder: TrainableVisualEncoder | None = None
        self.resnet_optimizer: torch.optim.Optimizer | None = None
        if arguments.trainable_resnet:
            self.visual_encoder = TrainableVisualEncoder(
                arguments.resnet_checkpoint, self.device
            )
            self.resnet_optimizer = torch.optim.AdamW(
                self.visual_encoder.trainable_parameters,
                lr=arguments.resnet_learning_rate,
                weight_decay=1.0e-5,
            )
            if arguments.training:
                # Build the cuDNN stage-4 backward plans before a gameplay
                # connection. No optimizer step is performed, so pretrained
                # weights remain byte-for-byte unchanged.
                warmup_image = Image.new("RGB", (224, 224), (0, 0, 0))
                warmup_buffer = io.BytesIO()
                warmup_image.save(
                    warmup_buffer, format="JPEG", quality=90
                )
                warmup = self.visual_encoder.features(
                    [warmup_buffer.getvalue()]
                    * arguments.resnet_batch_size,
                    require_gradient=True,
                )
                warmup.sum().backward()
                self.resnet_optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(arguments.device_id)

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
        self.current_jpeg: bytes | None = None
        self.current_frame_id = 0
        self.cached_frame_id = 0
        self.cached_projected_visual: torch.Tensor | None = None
        self.active: dict[str, Any] | None = None
        self.training_updates = int(arguments.initial_updates)
        self.last_loss = 0.0
        self.checkpoint_loaded = False
        self.resnet_checkpoint_loaded = False
        self.resnet_training_updates = 0
        self.parameter_checksum = 0.0
        self.resnet_parameter_checksum = 0.0
        if arguments.resume and arguments.checkpoint.is_file():
            self.load(arguments.checkpoint)
        self.refresh_checksum()
        self.refresh_resnet_checksum()

    def refresh_checksum(self) -> None:
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        with torch.no_grad():
            for index, parameter in enumerate(self.online.parameters(), 1):
                total += parameter.to(dtype=torch.float64).sum() * index
        self.parameter_checksum = float(total.cpu())

    def refresh_resnet_checksum(self) -> None:
        if self.visual_encoder is None:
            self.resnet_parameter_checksum = 0.0
            return
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        with torch.no_grad():
            for index, parameter in enumerate(
                self.visual_encoder.trainable_parameters, 1
            ):
                flat = parameter.reshape(-1)
                sample = flat[: min(256, flat.numel())]
                total += sample.to(dtype=torch.float64).sum() * index
        self.resnet_parameter_checksum = float(total.cpu())

    def project_visual_tensor(self, visual: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.projection(visual))

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
            projected = self.project_visual_tensor(
                visual_tensor
            ).squeeze(0)
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
            self.current_jpeg = None
            self.current_frame_id = 0
        context = np.asarray(request["context"], dtype=np.float32)
        if context.shape != (self.config.context_size,):
            raise ValueError("recurrent context shape mismatch")
        if self.visual_encoder is not None:
            frame_id = int(request.get("frame_id", 0))
            encoded = base64.b64decode(
                request["jpeg_base64"], validate=True
            )
            if not encoded or frame_id <= 0:
                raise ValueError("invalid trainable ResNet observation")
            if (
                frame_id != self.cached_frame_id
                or self.cached_projected_visual is None
            ):
                with torch.inference_mode():
                    feature = self.visual_encoder.features(
                        [encoded], require_gradient=False
                    )
                    self.cached_projected_visual = (
                        self.project_visual_tensor(feature)
                        .squeeze(0)
                        .detach()
                    )
                self.cached_frame_id = frame_id
            controller = request["controller"]
            if len(controller) != self.config.controller_size:
                raise ValueError(
                    "recurrent controller observation shape mismatch"
                )
            controller_tensor = torch.tensor(
                controller, dtype=torch.float32, device=self.device
            )
            recurrent_input = (
                torch.cat(
                    (self.cached_projected_visual, controller_tensor)
                )
                .to(device="cpu", dtype=torch.float16)
                .numpy()
                .copy()
            )
            self.current_jpeg = encoded
            self.current_frame_id = frame_id
        else:
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
        replacement_visual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences, lengths, context, characters, opponents = self.batch(
            transitions, next_state
        )
        if replacement_visual is not None:
            count = replacement_visual.shape[0]
            if count <= 0 or count > sequences.shape[0]:
                raise ValueError("invalid ResNet replacement batch")
            rows = torch.arange(count, device=self.device)
            columns = lengths[:count].to(self.device) - 1
            sequences[
                rows,
                columns,
                : self.config.projection_size,
            ] = replacement_visual
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
        if self.visual_encoder is not None:
            with_images = 0
            for value in reversed(self.replay):
                if value.state_jpeg is None:
                    continue
                with_images += 1
                if with_images > self.arguments.resnet_replay_images:
                    value.state_jpeg = None

    def resnet_training_sample(
        self, transitions: list[Transition]
    ) -> tuple[list[Transition], torch.Tensor | None]:
        if (
            self.visual_encoder is None
            or self.resnet_optimizer is None
            or (
                (self.training_updates + 1)
                % self.arguments.resnet_update_interval
            ) != 0
        ):
            return transitions, None
        eligible = [
            value for value in self.replay
            if value.state_jpeg is not None
        ]
        if not eligible:
            return transitions, None
        count = min(
            self.arguments.resnet_batch_size,
            len(transitions),
            len(eligible),
        )
        selected = random.sample(eligible, count)
        transitions = selected + transitions[count:]
        images = [
            value.state_jpeg for value in selected
            if value.state_jpeg is not None
        ]
        features = self.visual_encoder.features(
            images, require_gradient=True
        )
        return transitions, self.project_visual_tensor(features)

    def train_once(self) -> bool:
        minimum = max(
            self.arguments.minimum_replay_size,
            self.arguments.batch_size,
        )
        if not self.arguments.training or len(self.replay) < minimum:
            return False
        transitions = self.balanced_sample()
        transitions, replacement_visual = self.resnet_training_sample(
            transitions
        )
        self.online.train()
        current_latent, characters = self.latent_for(
            self.online,
            transitions,
            False,
            replacement_visual,
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
        if self.resnet_optimizer is not None:
            self.resnet_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online.parameters(), self.arguments.gradient_clip
        )
        self.optimizer.step()
        if (
            replacement_visual is not None
            and self.resnet_optimizer is not None
            and self.visual_encoder is not None
        ):
            torch.nn.utils.clip_grad_norm_(
                self.visual_encoder.trainable_parameters,
                self.arguments.gradient_clip,
            )
            self.resnet_optimizer.step()
            self.resnet_training_updates += 1
            self.refresh_resnet_checksum()
            # The live cache was encoded with the pre-update backbone.
            self.cached_projected_visual = None
            self.cached_frame_id = 0
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
            "state_jpeg": self.current_jpeg,
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
                state_jpeg=self.active.get("state_jpeg"),
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
        self.current_jpeg = None
        self.current_frame_id = 0
        self.cached_projected_visual = None
        self.cached_frame_id = 0
        return self.stats()

    def checkpoint_payload(self, include_replay: bool) -> dict[str, Any]:
        replay: list[Transition] = []
        if include_replay and self.arguments.persist_replay_limit > 0:
            for transition in list(self.replay)[
                -self.arguments.persist_replay_limit :
            ]:
                persisted = copy.copy(transition)
                # Checkpoints remain compact. Resumed transitions still train
                # the recurrent policy; fresh frames refill visual replay.
                persisted.state_jpeg = None
                replay.append(persisted)
        payload: dict[str, Any] = {
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
        if self.visual_encoder is not None:
            payload["resnet"] = self.visual_encoder.model.state_dict()
            payload["resnet_training_updates"] = (
                self.resnet_training_updates
            )
            payload["resnet_parameter_checksum"] = (
                self.resnet_parameter_checksum
            )
            if self.resnet_optimizer is not None:
                payload["resnet_optimizer"] = (
                    self.resnet_optimizer.state_dict()
                )
        return payload

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
        if (
            self.visual_encoder is not None
            and "resnet" in checkpoint
        ):
            self.visual_encoder.model.load_state_dict(
                checkpoint["resnet"], strict=True
            )
            if (
                self.resnet_optimizer is not None
                and "resnet_optimizer" in checkpoint
            ):
                self.resnet_optimizer.load_state_dict(
                    checkpoint["resnet_optimizer"]
                )
            self.resnet_training_updates = int(
                checkpoint.get("resnet_training_updates", 0)
            )
            self.resnet_checkpoint_loaded = True
        for transition in checkpoint.get("replay", []):
            if not hasattr(transition, "state_jpeg"):
                transition.state_jpeg = None
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
            "trainable_resnet": self.visual_encoder is not None,
            "resnet_training_updates": self.resnet_training_updates,
            "resnet_parameter_checksum":
                self.resnet_parameter_checksum,
            "resnet_checkpoint_loaded":
                self.resnet_checkpoint_loaded,
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
        f"checkpoint_loaded={int(service.checkpoint_loaded)} "
        f"trainable_resnet={int(service.visual_encoder is not None)} "
        f"resnet_updates={service.resnet_training_updates} "
        f"resnet_checkpoint_loaded="
        f"{int(service.resnet_checkpoint_loaded)}",
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
