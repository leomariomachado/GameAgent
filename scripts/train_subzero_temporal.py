#!/usr/bin/env python3
"""Offline behavior-cloning pretraining for the GameAgent temporal policy.

The trainer joins timestamped ResNet50 features with the WebSocket decision and
action_ack log. It learns the exact C++ architecture:

  tanh(Linear(2048, projection))
  + 12 controller features
  -> one-layer LSTM
  -> 10 independent controller-state logits

It writes both a resumable PyTorch/AdamW checkpoint and a compact inference
checkpoint consumed by GameAgent's CUDA implementation.
"""

from __future__ import annotations

import argparse
import bisect
import collections
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
CHECKPOINT_VERSION = 1
CONTROLLER_COMMANDS = 10
CONTROLLER_FEATURES = 12
VISUAL_FEATURES = 2048
DECISION_THRESHOLD = 0.6


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

    def forward(
        self, visual: torch.Tensor, controller: torch.Tensor
    ) -> torch.Tensor:
        projected = torch.tanh(self.visual_projection(visual))
        recurrent_input = torch.cat((projected, controller), dim=-1)
        recurrent_output, _ = self.lstm(recurrent_input)
        return self.action_head(recurrent_output[:, -1, :])


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
                    and Path(str(record.get("feature_path", ""))).is_file()
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
        feature_path = Path(record["feature_path"])
        if not feature_path.is_file() or feature_path.stat().st_size != 8192:
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
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[int, list[dict[str, Any]]],
]:
    decisions: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    acknowledgements: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in iter_jsonl(path):
        connection = record.get("connection_id")
        if not isinstance(connection, int):
            continue
        if record.get("event") == "decision":
            if (
                isinstance(record.get("timestamp_ms"), int)
                and isinstance(record.get("commands"), list)
            ):
                decisions[connection].append(record)
        elif record.get("event") == "action_ack":
            if (
                isinstance(
                    record.get("acknowledgement_received_timestamp_ms"), int
                )
                and isinstance(record.get("applied_commands"), list)
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
) -> list[TemporalSample]:
    samples: list[TemporalSample] = []
    for sequence_id, frame_records in sorted(sequences.items()):
        connection = connection_from_sequence(sequence_id)
        connection_decisions = decisions.get(connection, [])
        connection_acks = acknowledgements.get(connection, [])
        frame_times = [
            int(record["received_timestamp_ms"]) for record in frame_records
        ]
        start_time = frame_times[0]
        end_time = frame_times[-1] + 75
        relevant_decisions = [
            record
            for record in connection_decisions
            if start_time <= int(record["timestamp_ms"]) <= end_time
        ]
        if not relevant_decisions:
            continue

        ack_times = [
            int(record["acknowledgement_received_timestamp_ms"])
            for record in connection_acks
        ]
        visual = np.stack(
            [load_visual_feature(record) for record in frame_records]
        )
        controller = np.zeros(
            (len(frame_records), CONTROLLER_FEATURES), dtype=np.float32
        )
        for frame_index, frame_time in enumerate(frame_times):
            ack_index = bisect.bisect_right(ack_times, frame_time) - 1
            if ack_index < 0:
                continue
            ack = connection_acks[ack_index]
            controller[frame_index, :CONTROLLER_COMMANDS] = (
                commands_to_multihot(ack["applied_commands"])
            )
            controller[frame_index, 10] = 1.0
            controller[frame_index, 11] = min(
                1.0, max(0.0, (frame_time - ack_times[ack_index]) / 1000.0)
            )

        for decision in relevant_decisions:
            decision_time = int(decision["timestamp_ms"])
            newest_frame = bisect.bisect_right(frame_times, decision_time) - 1
            if newest_frame + 1 < config.sequence_length:
                continue
            first_frame = newest_frame + 1 - config.sequence_length
            samples.append(
                TemporalSample(
                    visual=np.ascontiguousarray(
                        visual[first_frame : newest_frame + 1], dtype=np.float32
                    ),
                    controller=np.ascontiguousarray(
                        controller[first_frame : newest_frame + 1],
                        dtype=np.float32,
                    ),
                    target=commands_to_multihot(decision["commands"]),
                    sequence_id=sequence_id,
                    decision_id=int(decision.get("decision_id", 0)),
                    frame_id=int(frame_records[newest_frame]["frame_id"]),
                    timestamp_ms=decision_time,
                    action_source=str(
                        decision.get("curriculum_source", "legacy_scripted_move")
                    ),
                    move_name=str(decision.get("mock_move", "unknown")),
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


def target_statistics(samples: list[TemporalSample]) -> tuple[np.ndarray, np.ndarray]:
    targets = np.stack([sample.target for sample in samples])
    positives = targets.sum(axis=0)
    negatives = len(samples) - positives
    pos_weight = np.clip(
        negatives / np.maximum(positives, 1.0), 1.0, 5.0
    ).astype(np.float32)
    return positives, pos_weight


def evaluate(
    model: TemporalPolicy,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_bits = 0
    correct_bits = 0
    exact_matches = 0
    with torch.inference_mode():
        for visual, controller, target in loader:
            visual = visual.to(device, non_blocking=True)
            controller = controller.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            logits = model(visual, controller)
            loss = criterion(logits, target)
            predicted = torch.sigmoid(logits) >= DECISION_THRESHOLD
            expected = target >= 0.5
            count = target.shape[0]
            total_loss += float(loss) * count
            total_samples += count
            total_bits += expected.numel()
            correct_bits += int((predicted == expected).sum())
            exact_matches += int((predicted == expected).all(dim=1).sum())
    return {
        "loss": total_loss / max(total_samples, 1),
        "bit_accuracy": correct_bits / max(total_bits, 1),
        "exact_match": exact_matches / max(total_samples, 1),
    }


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(MAGIC)
        temporary.write(
            struct.pack(
                "<6I2Qf",
                CHECKPOINT_VERSION,
                config.visual_size,
                config.projection_size,
                config.controller_size,
                config.hidden_size,
                config.output_size,
                training_step,
                epoch,
                float(best_validation_loss),
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
    if arguments.epochs <= 0 or arguments.batch_size <= 0:
        raise ValueError("epochs and batch size must be positive")
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
    decisions, acknowledgements = load_metrics(arguments.metrics)
    samples = build_samples(
        sequences, decisions, acknowledgements, config
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
    positives, pos_weight = target_statistics(train_samples)

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
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(pos_weight).to(device)
    )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    latest_pt = arguments.output_dir / "subzero_temporal_latest.pt"
    latest_binary = arguments.output_dir / "subzero_temporal_latest.gat"
    best_pt = arguments.output_dir / "subzero_temporal_best.pt"
    best_binary = arguments.output_dir / "subzero_temporal_best.gat"
    start_epoch = 0
    global_step = 0
    best_validation_loss = math.inf
    if arguments.resume and latest_pt.is_file():
        checkpoint = torch.load(
            latest_pt, map_location=device, weights_only=False
        )
        if checkpoint.get("config") != asdict(config):
            raise RuntimeError("resume checkpoint model configuration differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        print(
            f"[RESUME] epoch={start_epoch} step={global_step} "
            f"best_val_loss={best_validation_loss:.6f}"
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
        "positive_class_weights": pos_weight.tolist(),
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

    for epoch in range(start_epoch + 1, start_epoch + arguments.epochs + 1):
        model.train()
        total_loss = 0.0
        trained_samples = 0
        for visual, controller, target in train_loader:
            visual = visual.to(device, non_blocking=True)
            controller = controller.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(visual, controller)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = target.shape[0]
            total_loss += float(loss.detach()) * count
            trained_samples += count
            global_step += 1

        train_metrics = {
            "loss": total_loss / max(trained_samples, 1)
        }
        validation_metrics = evaluate(
            model, validation_loader, criterion, device
        )
        checkpoint = {
            "version": CHECKPOINT_VERSION,
            "epoch": epoch,
            "global_step": global_step,
            "best_validation_loss": min(
                best_validation_loss, validation_metrics["loss"]
            ),
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
        )
        improved = validation_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = validation_metrics["loss"]
            atomic_torch_save(checkpoint, best_pt)
            export_inference_checkpoint(
                model,
                config,
                best_binary,
                global_step,
                epoch,
                best_validation_loss,
            )
        print(
            f"[EPOCH {epoch}] train_loss={train_metrics['loss']:.6f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"bit_acc={validation_metrics['bit_accuracy']:.4f} "
            f"exact={validation_metrics['exact_match']:.4f} "
            f"best={best_validation_loss:.6f} step={global_step}"
        )

    print(f"[CHECKPOINT] resumable={latest_pt}")
    print(f"[CHECKPOINT] inference={latest_binary}")
    print("[PASS] Sub-Zero temporal behavior-cloning pretraining completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
