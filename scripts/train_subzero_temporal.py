#!/usr/bin/env python3
"""Offline behavior-cloning pretraining for the GameAgent temporal policy.

The trainer joins timestamped ResNet50 features with the WebSocket decision and
action_ack log. A recurrent step represents one controller decision (20 Hz),
not one camera frame. The most recent visual feature is reused when multiple
decisions happen between two frames. It learns the exact C++ architecture:

  tanh(Linear(2048, projection))
  + 12 controller features
  -> one-layer LSTM
  -> active/release logit + 10 controller-state logits

It writes both a resumable PyTorch/AdamW checkpoint and a compact inference
checkpoint consumed by GameAgent's CUDA implementation.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import concurrent.futures
import glob
import json
import math
import os
import random
import re
import struct
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


MAGIC = b"GATLSTM1"
CHECKPOINT_VERSION = 2
CONTROLLER_COMMANDS = 10
CONTROLLER_FEATURES = 12
VISUAL_FEATURES = 2048
DEFAULT_ACTIVE_THRESHOLD = 0.5
DEFAULT_COMMAND_THRESHOLD = 0.5


@dataclass(frozen=True)
class ModelConfig:
    visual_size: int = VISUAL_FEATURES
    projection_size: int = 256
    controller_size: int = CONTROLLER_FEATURES
    hidden_size: int = 256
    output_size: int = CONTROLLER_COMMANDS
    sequence_length: int = 8


@dataclass
class TemporalSample:
    visual: np.ndarray
    controller: np.ndarray
    target: np.ndarray
    sequence_id: str
    decision_id: int
    frame_id: int
    timestamp_ms: int
    action_source: str
    move_name: str


class TemporalDataset(Dataset):
    def __init__(self, samples: list[TemporalSample]) -> None:
        if not samples:
            raise ValueError("temporal dataset cannot be empty")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        sample = self.samples[index]
        return (
            torch.from_numpy(sample.visual),
            torch.from_numpy(sample.controller),
            torch.from_numpy(sample.target),
        )


class TemporalPolicy(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.visual_projection = nn.Linear(
            config.visual_size, config.projection_size
        )
        self.lstm = nn.LSTM(
            config.projection_size + config.controller_size,
            config.hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.action_head = nn.Linear(config.hidden_size, config.output_size)
        self.active_head = nn.Linear(config.hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.visual_projection.weight)
        nn.init.zeros_(self.visual_projection.bias)
        for name, parameter in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            else:
                nn.init.zeros_(parameter)
        with torch.no_grad():
            hidden = self.config.hidden_size
            self.lstm.bias_ih_l0[hidden : 2 * hidden].fill_(1.0)
        nn.init.xavier_uniform_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        nn.init.xavier_uniform_(self.active_head.weight)
        nn.init.zeros_(self.active_head.bias)

    def forward(
        self, visual: torch.Tensor, controller: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = torch.tanh(self.visual_projection(visual))
        recurrent_input = torch.cat((projected, controller), dim=-1)
        recurrent_output, _ = self.lstm(recurrent_input)
        state = recurrent_output[:, -1, :]
        return self.active_head(state).squeeze(-1), self.action_head(state)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("/workspace/gameagent/metrics.jsonl"),
    )
    parser.add_argument(
        "--features-manifest",
        type=Path,
        default=None,
        help="features.jsonl; default selects the largest valid manifest",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("/workspace/gameagent/resnet50_features"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/gameagent/checkpoints"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--projection-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--feature-load-workers",
        type=int,
        default=16,
        help="parallel readers for small .f32 files on network volumes",
    )
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-4)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if isinstance(value, dict):
                yield value


def choose_manifest(feature_root: Path) -> Path:
    candidates: list[tuple[int, float, Path]] = []
    for raw_path in glob.glob(str(feature_root / "**/features.jsonl"), recursive=True):
        path = Path(raw_path)
        valid = 0
        try:
            for record in iter_jsonl(path):
                if (
                    isinstance(record.get("frame_id"), int)
                    and record.get("feature_size") == VISUAL_FEATURES
                    and bool(record.get("feature_path"))
                ):
                    valid += 1
        except (OSError, RuntimeError):
            continue
        if valid:
            candidates.append((valid, path.stat().st_mtime, path))
    if not candidates:
        raise RuntimeError(f"no valid feature manifest found under {feature_root}")
    candidates.sort()
    return candidates[-1][2]


def load_feature_sequences(
    manifest: Path,
) -> dict[str, list[dict[str, Any]]]:
    sequences: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in iter_jsonl(manifest):
        required = (
            "sequence_id",
            "frame_id",
            "received_timestamp_ms",
            "feature_path",
            "feature_size",
        )
        if any(key not in record for key in required):
            continue
        if record["feature_size"] != VISUAL_FEATURES:
            continue
        sequences[str(record["sequence_id"])].append(record)
    for records in sequences.values():
        records.sort(
            key=lambda value: (
                int(value["received_timestamp_ms"]),
                int(value["frame_id"]),
            )
        )
    return {
        key: value
        for key, value in sequences.items()
        if len(value) >= 8
    }


def load_metrics(
    path: Path,
    time_windows: dict[int, list[tuple[int, int]]] | None = None,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[int, list[dict[str, Any]]],
]:
    decisions: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    acknowledgements: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    def relevant(connection: int, timestamp_ms: int) -> bool:
        if time_windows is None:
            return True
        return any(
            start_ms - 2_000 <= timestamp_ms <= end_ms + 2_000
            for start_ms, end_ms in time_windows.get(connection, [])
        )

    for record in iter_jsonl(path):
        connection = record.get("connection_id")
        if not isinstance(connection, int):
            continue
        if record.get("event") == "decision":
            if (
                isinstance(record.get("timestamp_ms"), int)
                and isinstance(record.get("commands"), list)
                and relevant(
                    connection,
                    int(
                        record.get(
                            "decision_tick_timestamp_ms",
                            record["timestamp_ms"],
                        )
                    ),
                )
            ):
                decisions[connection].append(record)
        elif record.get("event") == "action_ack":
            if (
                isinstance(
                    record.get("acknowledgement_received_timestamp_ms"), int
                )
                and isinstance(record.get("applied_commands"), list)
                and relevant(
                    connection,
                    int(
                        record[
                            "acknowledgement_received_timestamp_ms"
                        ]
                    ),
                )
            ):
                acknowledgements[connection].append(record)
    for values in decisions.values():
        values.sort(key=lambda item: int(item["timestamp_ms"]))
    for values in acknowledgements.values():
        values.sort(
            key=lambda item: int(
                item["acknowledgement_received_timestamp_ms"]
            )
        )
    return decisions, acknowledgements


def commands_to_multihot(commands: list[Any]) -> np.ndarray:
    output = np.zeros(CONTROLLER_COMMANDS, dtype=np.float32)
    for raw_command in commands:
        command = int(raw_command)
        if command < 0 or command >= CONTROLLER_COMMANDS:
            raise RuntimeError(f"command outside 0..9: {command}")
        output[command] = 1.0
    return output


def decision_timestamp(record: dict[str, Any]) -> int:
    return int(
        record.get("decision_tick_timestamp_ms", record["timestamp_ms"])
    )


def sequence_time_windows(
    sequences: dict[str, list[dict[str, Any]]],
) -> dict[int, list[tuple[int, int]]]:
    windows: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for sequence_id, records in sequences.items():
        timestamps = [
            int(record["received_timestamp_ms"]) for record in records
        ]
        windows[connection_from_sequence(sequence_id)].append(
            (min(timestamps), max(timestamps))
        )
    return windows


def connection_from_sequence(sequence_id: str) -> int:
    match = re.search(r"connection_(\d+)$", sequence_id)
    if match is None:
        raise RuntimeError(f"sequence has no connection ID: {sequence_id}")
    return int(match.group(1))


def load_visual_feature(record: dict[str, Any]) -> np.ndarray:
    feature = np.fromfile(record["feature_path"], dtype=np.float32)
    if feature.shape != (VISUAL_FEATURES,) or not np.isfinite(feature).all():
        raise RuntimeError(f"invalid visual feature: {record['feature_path']}")
    return feature


def build_samples(
    sequences: dict[str, list[dict[str, Any]]],
    decisions: dict[int, list[dict[str, Any]]],
    acknowledgements: dict[int, list[dict[str, Any]]],
    config: ModelConfig,
    feature_load_workers: int,
) -> list[TemporalSample]:
    samples: list[TemporalSample] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=feature_load_workers
    ) as feature_pool:
        for sequence_id, frame_records in sorted(sequences.items()):
            connection = connection_from_sequence(sequence_id)
            connection_decisions = decisions.get(connection, [])
            connection_acks = acknowledgements.get(connection, [])
            frame_times = [
                int(record["received_timestamp_ms"])
                for record in frame_records
            ]
            frame_index_by_id = {
                int(record["frame_id"]): index
                for index, record in enumerate(frame_records)
            }
            start_time = frame_times[0]
            end_time = frame_times[-1] + 75
            relevant_decisions = [
                record
                for record in connection_decisions
                if start_time <= decision_timestamp(record) <= end_time
            ]
            if not relevant_decisions:
                continue

            ack_times = [
                int(record["acknowledgement_received_timestamp_ms"])
                for record in connection_acks
            ]
            visual = np.stack(
                list(feature_pool.map(load_visual_feature, frame_records))
            )
            tick_visual: list[np.ndarray] = []
            tick_controller: list[np.ndarray] = []
            tick_decisions: list[dict[str, Any]] = []
            for decision in relevant_decisions:
                decision_time = decision_timestamp(decision)
                temporal_frame_id = (
                    decision.get("metrics", {}).get("temporal_frame_id")
                    if isinstance(decision.get("metrics"), dict)
                    else None
                )
                newest_frame = (
                    frame_index_by_id.get(temporal_frame_id, -1)
                    if isinstance(temporal_frame_id, int)
                    else -1
                )
                if newest_frame < 0:
                    newest_frame = (
                        bisect.bisect_right(frame_times, decision_time) - 1
                    )
                if newest_frame < 0:
                    continue
                controller_state = np.zeros(
                    CONTROLLER_FEATURES, dtype=np.float32
                )
                ack_index = (
                    bisect.bisect_right(ack_times, decision_time) - 1
                )
                if ack_index >= 0:
                    ack = connection_acks[ack_index]
                    controller_state[:CONTROLLER_COMMANDS] = (
                        commands_to_multihot(ack["applied_commands"])
                    )
                    controller_state[10] = 1.0
                    controller_state[11] = min(
                        1.0,
                        max(
                            0.0,
                            (
                                decision_time - ack_times[ack_index]
                            ) / 1000.0,
                        ),
                    )
                tick_visual.append(visual[newest_frame])
                tick_controller.append(controller_state)
                tick_decisions.append(decision)
                if len(tick_decisions) < config.sequence_length:
                    continue
                first_tick = len(tick_decisions) - config.sequence_length
                decision_visual = np.stack(tick_visual[first_tick:])
                decision_controller = np.stack(
                    tick_controller[first_tick:]
                )
                samples.append(
                    TemporalSample(
                        visual=np.ascontiguousarray(
                            decision_visual, dtype=np.float32
                        ),
                        controller=np.ascontiguousarray(
                            decision_controller, dtype=np.float32
                        ),
                        target=commands_to_multihot(
                            decision["commands"]
                        ),
                        sequence_id=sequence_id,
                        decision_id=int(decision.get("decision_id", 0)),
                        frame_id=int(
                            frame_records[newest_frame]["frame_id"]
                        ),
                        timestamp_ms=decision_time,
                        action_source=str(
                            decision.get(
                                "curriculum_source",
                                "legacy_scripted_move",
                            )
                        ),
                        move_name=str(
                            decision.get("mock_move", "unknown")
                        ),
                    )
                )
    return samples


def split_samples(
    samples: list[TemporalSample], seed: int
) -> tuple[list[TemporalSample], list[TemporalSample]]:
    sequence_ids = sorted({sample.sequence_id for sample in samples})
    if len(sequence_ids) >= 2:
        random.Random(seed).shuffle(sequence_ids)
        validation_count = max(1, round(len(sequence_ids) * 0.2))
        validation_ids = set(sequence_ids[-validation_count:])
        train = [
            sample for sample in samples if sample.sequence_id not in validation_ids
        ]
        validation = [
            sample for sample in samples if sample.sequence_id in validation_ids
        ]
    else:
        ordered = sorted(samples, key=lambda sample: sample.timestamp_ms)
        split = max(1, int(len(ordered) * 0.8))
        train = ordered[:split]
        validation = ordered[split:]
    if not train or not validation:
        raise RuntimeError("dataset split produced an empty train or validation set")
    return train, validation


def target_statistics(
    samples: list[TemporalSample],
) -> tuple[np.ndarray, np.ndarray, float]:
    targets = np.stack([sample.target for sample in samples])
    positives = targets.sum(axis=0)
    active = targets.any(axis=1)
    active_count = int(active.sum())
    active_targets = targets[active]
    active_negatives = active_count - active_targets.sum(axis=0)
    # Commands are trained only on active decisions. Preserve the actual
    # imbalance instead of clipping every class to the old value 5.
    command_pos_weight = np.clip(
        active_negatives / np.maximum(positives, 1.0), 1.0, 50.0
    ).astype(np.float32)
    inactive_count = len(samples) - active_count
    active_pos_weight = float(
        np.clip(inactive_count / max(active_count, 1), 0.25, 4.0)
    )
    return positives, command_pos_weight, active_pos_weight


def focal_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    base = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    probability = torch.sigmoid(logits)
    probability_of_target = (
        probability * targets + (1.0 - probability) * (1.0 - targets)
    )
    return ((1.0 - probability_of_target).pow(gamma) * base).mean()


def policy_loss(
    active_logits: torch.Tensor,
    command_logits: torch.Tensor,
    targets: torch.Tensor,
    active_pos_weight: torch.Tensor,
    command_pos_weight: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    active_targets = targets.any(dim=1).to(dtype=torch.float32)
    active_loss = focal_binary_loss(
        active_logits, active_targets, active_pos_weight, gamma
    )
    active_rows = active_targets >= 0.5
    if bool(active_rows.any()):
        command_loss = focal_binary_loss(
            command_logits[active_rows],
            targets[active_rows],
            command_pos_weight,
            gamma,
        )
    else:
        command_loss = command_logits.sum() * 0.0
    return active_loss + command_loss, active_loss, command_loss


def decode_predictions(
    active_probabilities: np.ndarray,
    command_probabilities: np.ndarray,
    active_threshold: float,
    command_threshold: float,
) -> np.ndarray:
    active = active_probabilities >= active_threshold
    predicted = command_probabilities >= command_threshold
    predicted &= active[:, None]
    for row in np.flatnonzero(active & ~predicted.any(axis=1)):
        predicted[row, int(np.argmax(command_probabilities[row]))] = True
    return predicted


def classification_metrics(
    active_probabilities: np.ndarray,
    command_probabilities: np.ndarray,
    targets: np.ndarray,
    active_threshold: float,
    command_threshold: float,
) -> dict[str, float]:
    expected = targets >= 0.5
    expected_active = expected.any(axis=1)
    predicted_active = active_probabilities >= active_threshold
    predicted = decode_predictions(
        active_probabilities,
        command_probabilities,
        active_threshold,
        command_threshold,
    )

    active_tp = int((predicted_active & expected_active).sum())
    active_fp = int((predicted_active & ~expected_active).sum())
    active_fn = int((~predicted_active & expected_active).sum())
    active_precision = active_tp / max(active_tp + active_fp, 1)
    active_recall = active_tp / max(active_tp + active_fn, 1)
    active_f1 = (
        2.0 * active_precision * active_recall
        / max(active_precision + active_recall, 1.0e-12)
    )

    active_rows = expected_active
    command_tp = int((predicted[active_rows] & expected[active_rows]).sum())
    command_fp = int((predicted[active_rows] & ~expected[active_rows]).sum())
    command_fn = int((~predicted[active_rows] & expected[active_rows]).sum())
    command_precision = command_tp / max(command_tp + command_fp, 1)
    command_recall = command_tp / max(command_tp + command_fn, 1)
    command_f1 = (
        2.0 * command_precision * command_recall
        / max(command_precision + command_recall, 1.0e-12)
    )
    exact = (predicted == expected).all(axis=1)
    exact_active = (
        float(exact[active_rows].mean()) if bool(active_rows.any()) else 0.0
    )
    release_rows = ~expected_active
    release_accuracy = (
        float((~predicted_active[release_rows]).mean())
        if bool(release_rows.any())
        else 0.0
    )
    selection_score = (
        0.4 * active_f1 + 0.4 * command_f1 + 0.2 * exact_active
    )
    return {
        "active_precision": active_precision,
        "active_recall": active_recall,
        "active_f1": active_f1,
        "command_precision": command_precision,
        "command_recall": command_recall,
        "command_f1": command_f1,
        "exact_match": float(exact.mean()),
        "exact_active": exact_active,
        "release_accuracy": release_accuracy,
        "selection_score": selection_score,
        "active_threshold": active_threshold,
        "command_threshold": command_threshold,
    }


def evaluate(
    model: TemporalPolicy,
    loader: DataLoader,
    device: torch.device,
    active_pos_weight: torch.Tensor,
    command_pos_weight: torch.Tensor,
    focal_gamma: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_active_loss = 0.0
    total_command_loss = 0.0
    total_samples = 0
    all_active_probabilities: list[np.ndarray] = []
    all_command_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.inference_mode():
        for visual, controller, target in loader:
            visual = visual.to(device, non_blocking=True)
            controller = controller.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            active_logits, command_logits = model(visual, controller)
            loss, active_loss, command_loss = policy_loss(
                active_logits,
                command_logits,
                target,
                active_pos_weight,
                command_pos_weight,
                focal_gamma,
            )
            count = target.shape[0]
            total_loss += float(loss) * count
            total_active_loss += float(active_loss) * count
            total_command_loss += float(command_loss) * count
            total_samples += count
            all_active_probabilities.append(
                torch.sigmoid(active_logits).cpu().numpy()
            )
            all_command_probabilities.append(
                torch.sigmoid(command_logits).cpu().numpy()
            )
            all_targets.append(target.cpu().numpy())

    active_probabilities = np.concatenate(all_active_probabilities)
    command_probabilities = np.concatenate(all_command_probabilities)
    targets = np.concatenate(all_targets)
    best_metrics: dict[str, float] | None = None
    for active_threshold in np.arange(0.25, 0.751, 0.05):
        for command_threshold in np.arange(0.20, 0.701, 0.05):
            metrics = classification_metrics(
                active_probabilities,
                command_probabilities,
                targets,
                float(active_threshold),
                float(command_threshold),
            )
            if (
                best_metrics is None
                or metrics["selection_score"]
                > best_metrics["selection_score"]
            ):
                best_metrics = metrics
    assert best_metrics is not None
    best_metrics.update({
        "loss": total_loss / max(total_samples, 1),
        "active_loss": total_active_loss / max(total_samples, 1),
        "command_loss": total_command_loss / max(total_samples, 1),
        "empty_exact_baseline": float((~targets.astype(bool).any(axis=1)).mean()),
    })
    return best_metrics


def atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_tensor(output: Any, tensor: torch.Tensor) -> None:
    values = (
        tensor.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .numpy()
    )
    output.write(struct.pack("<Q", values.size))
    output.write(values.tobytes(order="C"))


def export_inference_checkpoint(
    model: TemporalPolicy,
    config: ModelConfig,
    path: Path,
    training_step: int,
    epoch: int,
    best_validation_loss: float,
    best_selection_score: float,
    active_threshold: float,
    command_threshold: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(MAGIC)
        temporary.write(
            struct.pack(
                "<6I2Q4f",
                CHECKPOINT_VERSION,
                config.visual_size,
                config.projection_size,
                config.controller_size,
                config.hidden_size,
                config.output_size,
                training_step,
                epoch,
                float(best_validation_loss),
                float(best_selection_score),
                float(active_threshold),
                float(command_threshold),
            )
        )
        write_tensor(temporary, model.visual_projection.weight)
        write_tensor(temporary, model.visual_projection.bias)
        write_tensor(temporary, model.lstm.weight_ih_l0)
        write_tensor(temporary, model.lstm.weight_hh_l0)
        write_tensor(
            temporary,
            model.lstm.bias_ih_l0 + model.lstm.bias_hh_l0,
        )
        write_tensor(temporary, model.active_head.weight)
        write_tensor(temporary, model.active_head.bias)
        write_tensor(temporary, model.action_head.weight)
        write_tensor(temporary, model.action_head.bias)
        temporary.flush()
        os.fsync(temporary.fileno())
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> int:
    arguments = parse_arguments()
    if (
        arguments.epochs <= 0
        or arguments.batch_size <= 0
        or arguments.feature_load_workers <= 0
    ):
        raise ValueError(
            "epochs, batch size and feature-load workers must be positive"
        )
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)

    config = ModelConfig(
        projection_size=arguments.projection_size,
        hidden_size=arguments.hidden_size,
        sequence_length=arguments.sequence_length,
    )
    manifest = arguments.features_manifest or choose_manifest(
        arguments.feature_root
    )
    sequences = load_feature_sequences(manifest)
    if arguments.max_samples > 0 and len(sequences) > 2:
        sequence_ids = list(sequences)
        random.Random(arguments.seed).shuffle(sequence_ids)
        selected: dict[str, list[dict[str, Any]]] = {}
        estimated_ticks = 0
        for sequence_id in sequence_ids:
            selected[sequence_id] = sequences[sequence_id]
            estimated_ticks += len(sequences[sequence_id]) * 2
            if (
                len(selected) >= 2
                and estimated_ticks >= arguments.max_samples * 2
            ):
                break
        sequences = selected
    decisions, acknowledgements = load_metrics(
        arguments.metrics, sequence_time_windows(sequences)
    )
    samples = build_samples(
        sequences,
        decisions,
        acknowledgements,
        config,
        arguments.feature_load_workers,
    )
    if arguments.max_samples > 0 and len(samples) > arguments.max_samples:
        generator = random.Random(arguments.seed)
        samples = generator.sample(samples, arguments.max_samples)
    if len(samples) < 32:
        raise RuntimeError(
            f"only {len(samples)} aligned temporal samples were found"
        )
    train_samples, validation_samples = split_samples(
        samples, arguments.seed
    )
    positives, command_pos_weight, active_pos_weight = target_statistics(
        train_samples
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        TemporalDataset(train_samples),
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        TemporalDataset(validation_samples),
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        pin_memory=pin_memory,
    )
    model = TemporalPolicy(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    command_pos_weight_tensor = torch.from_numpy(
        command_pos_weight
    ).to(device)
    active_pos_weight_tensor = torch.tensor(
        active_pos_weight, dtype=torch.float32, device=device
    )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    latest_pt = arguments.output_dir / "subzero_temporal_latest.pt"
    latest_binary = arguments.output_dir / "subzero_temporal_latest.gat"
    best_pt = arguments.output_dir / "subzero_temporal_best.pt"
    best_binary = arguments.output_dir / "subzero_temporal_best.gat"
    start_epoch = 0
    global_step = 0
    best_validation_loss = math.inf
    best_selection_score = -math.inf
    best_active_threshold = DEFAULT_ACTIVE_THRESHOLD
    best_command_threshold = DEFAULT_COMMAND_THRESHOLD
    epochs_without_improvement = 0
    if arguments.resume and latest_pt.is_file():
        checkpoint = torch.load(
            latest_pt, map_location=device, weights_only=False
        )
        if checkpoint.get("version") != CHECKPOINT_VERSION:
            raise RuntimeError(
                "resume checkpoint predates the active/release head; "
                "start a new v2 training run without --resume"
            )
        if checkpoint.get("config") != asdict(config):
            raise RuntimeError("resume checkpoint model configuration differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        best_selection_score = float(
            checkpoint.get("best_selection_score", -math.inf)
        )
        best_active_threshold = float(
            checkpoint.get(
                "best_active_threshold", DEFAULT_ACTIVE_THRESHOLD
            )
        )
        best_command_threshold = float(
            checkpoint.get(
                "best_command_threshold", DEFAULT_COMMAND_THRESHOLD
            )
        )
        print(
            f"[RESUME] epoch={start_epoch} step={global_step} "
            f"best_val_loss={best_validation_loss:.6f} "
            f"best_score={best_selection_score:.6f}"
        )

    source_counts = collections.Counter(
        sample.action_source for sample in samples
    )
    move_counts = collections.Counter(sample.move_name for sample in samples)
    dataset_summary = {
        "feature_manifest": str(manifest),
        "metrics": str(arguments.metrics),
        "sequences": sorted(sequences),
        "samples": len(samples),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "positive_targets_per_command": positives.tolist(),
        "positive_class_weights": command_pos_weight.tolist(),
        "active_positive_weight": active_pos_weight,
        "sequence_clock": "decision_tick",
        "action_sources": dict(source_counts),
        "moves": dict(move_counts),
        "config": asdict(config),
        "seed": arguments.seed,
    }
    with (arguments.output_dir / "subzero_dataset_summary.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(dataset_summary, output, indent=2, sort_keys=True)
        output.write("\n")
    print(
        f"[DATASET] manifest={manifest} samples={len(samples)} "
        f"train={len(train_samples)} val={len(validation_samples)} "
        f"sequences={len(sequences)} device={device}"
    )
    empty_baseline = sum(
        not bool(sample.target.any()) for sample in validation_samples
    ) / len(validation_samples)
    print(
        f"[BASELINE] always_empty_exact={empty_baseline:.4f}; "
        "checkpoint selection uses active/command F1, not zero-bit accuracy"
    )

    for epoch in range(start_epoch + 1, start_epoch + arguments.epochs + 1):
        model.train()
        total_loss = 0.0
        total_active_loss = 0.0
        total_command_loss = 0.0
        trained_samples = 0
        for visual, controller, target in train_loader:
            visual = visual.to(device, non_blocking=True)
            controller = controller.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            active_logits, command_logits = model(visual, controller)
            loss, active_loss, command_loss = policy_loss(
                active_logits,
                command_logits,
                target,
                active_pos_weight_tensor,
                command_pos_weight_tensor,
                arguments.focal_gamma,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = target.shape[0]
            total_loss += float(loss.detach()) * count
            total_active_loss += float(active_loss.detach()) * count
            total_command_loss += float(command_loss.detach()) * count
            trained_samples += count
            global_step += 1

        train_metrics = {
            "loss": total_loss / max(trained_samples, 1),
            "active_loss": total_active_loss / max(trained_samples, 1),
            "command_loss": total_command_loss / max(trained_samples, 1),
        }
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            active_pos_weight_tensor,
            command_pos_weight_tensor,
            arguments.focal_gamma,
        )
        improved = (
            validation_metrics["selection_score"]
            > best_selection_score + arguments.early_stopping_min_delta
        )
        checkpoint_best_score = (
            validation_metrics["selection_score"]
            if improved
            else best_selection_score
        )
        checkpoint_active_threshold = (
            validation_metrics["active_threshold"]
            if improved
            else best_active_threshold
        )
        checkpoint_command_threshold = (
            validation_metrics["command_threshold"]
            if improved
            else best_command_threshold
        )
        lowest_validation_loss = min(
            best_validation_loss, validation_metrics["loss"]
        )
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "epoch": epoch,
            "global_step": global_step,
            "best_validation_loss": lowest_validation_loss,
            "best_selection_score": checkpoint_best_score,
            "best_active_threshold": checkpoint_active_threshold,
            "best_command_threshold": checkpoint_command_threshold,
            "config": asdict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
            "dataset_summary": dataset_summary,
        }
        atomic_torch_save(checkpoint, latest_pt)
        export_inference_checkpoint(
            model,
            config,
            latest_binary,
            global_step,
            epoch,
            checkpoint["best_validation_loss"],
            validation_metrics["selection_score"],
            validation_metrics["active_threshold"],
            validation_metrics["command_threshold"],
        )
        if improved:
            best_selection_score = validation_metrics["selection_score"]
            best_active_threshold = validation_metrics["active_threshold"]
            best_command_threshold = validation_metrics["command_threshold"]
            epochs_without_improvement = 0
            atomic_torch_save(checkpoint, best_pt)
            export_inference_checkpoint(
                model,
                config,
                best_binary,
                global_step,
                epoch,
                lowest_validation_loss,
                best_selection_score,
                best_active_threshold,
                best_command_threshold,
            )
        else:
            epochs_without_improvement += 1
        best_validation_loss = lowest_validation_loss
        print(
            f"[EPOCH {epoch}] train_loss={train_metrics['loss']:.6f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"active_f1={validation_metrics['active_f1']:.4f} "
            f"command_f1={validation_metrics['command_f1']:.4f} "
            f"active_exact={validation_metrics['exact_active']:.4f} "
            f"release={validation_metrics['release_accuracy']:.4f} "
            f"exact={validation_metrics['exact_match']:.4f} "
            f"empty_baseline={validation_metrics['empty_exact_baseline']:.4f} "
            f"score={validation_metrics['selection_score']:.4f} "
            f"best_score={best_selection_score:.4f} "
            f"thresholds={validation_metrics['active_threshold']:.2f}/"
            f"{validation_metrics['command_threshold']:.2f} "
            f"step={global_step}"
        )
        if (
            arguments.early_stopping_patience > 0
            and epochs_without_improvement
            >= arguments.early_stopping_patience
        ):
            print(
                f"[EARLY_STOP] no selection-score improvement for "
                f"{epochs_without_improvement} epochs"
            )
            break

    print(f"[CHECKPOINT] resumable={latest_pt}")
    print(f"[CHECKPOINT] inference={latest_binary}")
    print("[PASS] Sub-Zero temporal behavior-cloning pretraining completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
