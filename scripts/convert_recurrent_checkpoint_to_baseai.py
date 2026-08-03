#!/usr/bin/env python3
"""Convert a GameAgent PyTorch recurrent checkpoint to BaseAI v3.

The runtime C++ backend never imports Python.  This utility is intentionally a
one-time migration tool: it preserves projection, both online/target critics
and (optionally) the fine-tuned ResNet50 stage 4, while resetting optimizer
moments and replay because their Python object representation is not portable.
"""

from __future__ import annotations

import argparse
import os
import struct
import tempfile
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
import torch


MAGIC = b"BASECKPT"
VERSION = 3
FOOTER = 0x42415345454E4421


def exact(handle: BinaryIO, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise RuntimeError("truncated BaseAI checkpoint")
    return value


def scalar(handle: BinaryIO, code: str):
    return struct.unpack("<" + code, exact(handle, struct.calcsize(code)))[0]


def text(handle: BinaryIO) -> str:
    size = int(scalar(handle, "Q"))
    if size > 1_048_576:
        raise RuntimeError("invalid BaseAI checkpoint string")
    return exact(handle, size).decode("utf-8")


def write_scalar(handle: BinaryIO, code: str, value) -> None:
    handle.write(struct.pack("<" + code, value))


def write_text(handle: BinaryIO, value: str) -> None:
    encoded = value.encode("utf-8")
    write_scalar(handle, "Q", len(encoded))
    handle.write(encoded)


def as_float32(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().to(device="cpu", dtype=torch.float32).numpy()
    result = np.asarray(value, dtype="<f4").reshape(-1)
    return np.ascontiguousarray(result)


def write_tensors(
    handle: BinaryIO, tensors: Iterable[tuple[str, np.ndarray]]
) -> None:
    values = [(name, as_float32(value)) for name, value in tensors]
    write_scalar(handle, "Q", len(values))
    for name, value in values:
        write_text(handle, name)
        write_scalar(handle, "I", 0)  # Float32
        write_scalar(handle, "Q", value.size)
        handle.write(value.tobytes(order="C"))


def write_checkpoint(
    destination: Path,
    tensors: list[tuple[str, np.ndarray]],
    global_step: int,
    last_loss: float,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=destination.name + ".",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(MAGIC)
        write_scalar(temporary, "I", VERSION)
        write_scalar(temporary, "Q", 0)  # epoch
        write_scalar(temporary, "Q", max(0, int(global_step)))
        write_scalar(temporary, "f", 0.0)  # best accuracy
        write_scalar(temporary, "f", float(last_loss))
        write_scalar(temporary, "Q", 0)  # best epoch
        write_scalar(temporary, "i", 0)  # early-stopping bad epochs
        write_tensors(temporary, tensors)
        write_text(temporary, "AdamW")
        write_scalar(temporary, "Q", 0)  # optimizer reset is intentional
        write_scalar(temporary, "f", 1.0e-4)
        write_scalar(temporary, "f", 1.0e-5)
        write_scalar(temporary, "f", 0.9)
        write_scalar(temporary, "f", 0.999)
        write_scalar(temporary, "f", 1.0e-8)
        write_tensors(temporary, [])
        write_scalar(temporary, "f", 1.0)
        write_scalar(temporary, "f", 2.0)
        write_scalar(temporary, "f", 0.5)
        write_scalar(temporary, "i", 2000)
        write_scalar(temporary, "i", 0)
        write_scalar(temporary, "B", 0)
        write_scalar(temporary, "B", 0)
        write_text(temporary, "")
        write_scalar(temporary, "f", 0.0)
        write_scalar(temporary, "Q", 0)
        write_scalar(temporary, "Q", FOOTER)
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temporary_path, destination)


def signature(config: dict) -> np.ndarray:
    action_sizes = tuple(int(value) for value in config["action_sizes"])
    return np.asarray(
        (
            1,
            config["visual_size"],
            config["projection_size"],
            config["controller_size"],
            config["temporal_hidden_size"],
            config["context_size"],
            config["strategy_hidden_size"],
            config["sequence_capacity"],
            config["character_embedding_size"],
            config["opponent_embedding_size"],
            config["opponent_count"],
            len(action_sizes),
            *action_sizes,
        ),
        dtype=np.float32,
    )


def padded_heads(
    state: dict[str, torch.Tensor],
    action_sizes: tuple[int, ...],
    strategy_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    maximum = max(action_sizes)
    weights = np.zeros(
        (len(action_sizes), maximum, strategy_size), dtype=np.float32
    )
    biases = np.zeros((len(action_sizes), maximum), dtype=np.float32)
    for character, action_count in enumerate(action_sizes):
        source_weight = as_float32(
            state[f"heads.{character}.weight"]
        ).reshape(action_count, strategy_size)
        source_bias = as_float32(
            state[f"heads.{character}.bias"]
        )
        weights[character, :action_count] = source_weight
        biases[character, :action_count] = source_bias
    return weights, biases


def network_tensors(
    prefix: str,
    state: dict[str, torch.Tensor],
    config: dict,
) -> list[tuple[str, np.ndarray]]:
    action_sizes = tuple(int(value) for value in config["action_sizes"])
    head_weight, head_bias = padded_heads(
        state, action_sizes, int(config["strategy_hidden_size"])
    )
    return [
        (prefix + ".lstm.weight_ih", state["lstm.weight_ih_l0"]),
        (prefix + ".lstm.weight_hh", state["lstm.weight_hh_l0"]),
        (prefix + ".lstm.bias_ih", state["lstm.bias_ih_l0"]),
        (prefix + ".lstm.bias_hh", state["lstm.bias_hh_l0"]),
        (
            prefix + ".character_embedding",
            state["character_embedding.weight"],
        ),
        (
            prefix + ".opponent_embedding",
            state["opponent_embedding.weight"],
        ),
        (prefix + ".strategy.weight", state["strategy.weight"]),
        (prefix + ".strategy.bias", state["strategy.bias"]),
        (prefix + ".heads.weight", head_weight),
        (prefix + ".heads.bias", head_bias),
    ]


def recurrent_tensors(checkpoint: dict) -> list[tuple[str, np.ndarray]]:
    config = checkpoint["config"]
    version = int(checkpoint.get("service_version", 0))
    if version == 2:
        sources = {
            "online_a": checkpoint["online_a"],
            "online_b": checkpoint["online_b"],
            "target_a": checkpoint["target_a"],
            "target_b": checkpoint["target_b"],
        }
    elif version == 1:
        sources = {
            "online_a": checkpoint["online"],
            "online_b": checkpoint["online"],
            "target_a": checkpoint["target"],
            "target_b": checkpoint["target"],
        }
    else:
        raise RuntimeError(f"unsupported recurrent service version: {version}")
    tensors: list[tuple[str, np.ndarray]] = [
        ("cpp.config.signature", signature(config)),
        ("projection.weight", checkpoint["projection"]["weight"]),
        ("projection.bias", checkpoint["projection"]["bias"]),
    ]
    for prefix, state in sources.items():
        tensors.extend(network_tensors(prefix, state, config))
    return tensors


def representation_only_tensors(checkpoint: dict) -> list[tuple[str, np.ndarray]]:
    """Keep perception/temporal representation but reset both Q heads."""
    tensors = recurrent_tensors(checkpoint)
    values = {name: np.asarray(value).copy() for name, value in tensors}
    config = checkpoint["config"]
    action_sizes = tuple(int(value) for value in config["action_sizes"])
    strategy_size = int(config["strategy_hidden_size"])
    maximum_actions = max(action_sizes)
    random = np.random.default_rng(0x524C444E)
    for critic in ("a", "b"):
        weight = np.zeros(
            (len(action_sizes), maximum_actions, strategy_size),
            dtype=np.float32,
        )
        bias = np.zeros(
            (len(action_sizes), maximum_actions), dtype=np.float32
        )
        bound = np.sqrt(3.0 / strategy_size)
        for character, action_count in enumerate(action_sizes):
            weight[character, :action_count] = random.uniform(
                -bound,
                bound,
                size=(action_count, strategy_size),
            ).astype(np.float32)
        online = f"online_{critic}"
        target = f"target_{critic}"
        values[online + ".heads.weight"] = weight
        values[online + ".heads.bias"] = bias
        values[target + ".heads.weight"] = weight.copy()
        values[target + ".heads.bias"] = bias.copy()
    return [(name, values[name]) for name, _ in tensors]


def read_base_model(path: Path) -> tuple[dict, list[tuple[str, np.ndarray]]]:
    dtype_map = {
        0: (np.dtype("<f4"), 4),
        1: (np.dtype("<f2"), 2),
        2: (np.dtype("<u2"), 2),
        3: (np.dtype("<i4"), 4),
        4: (np.dtype("u1"), 1),
    }
    with path.open("rb") as handle:
        if exact(handle, 8) != MAGIC:
            raise RuntimeError("invalid BaseAI ResNet checkpoint")
        version = int(scalar(handle, "I"))
        if version not in (2, 3):
            raise RuntimeError(f"unsupported BaseAI checkpoint version: {version}")
        metadata = {
            "epoch": int(scalar(handle, "Q")),
            "global_step": int(scalar(handle, "Q")),
            "best_accuracy": float(scalar(handle, "f")),
            "best_loss": float(scalar(handle, "f")),
            "best_epoch": 0,
            "bad_epochs": 0,
        }
        if version >= 3:
            metadata["best_epoch"] = int(scalar(handle, "Q"))
            metadata["bad_epochs"] = int(scalar(handle, "i"))
        count = int(scalar(handle, "Q"))
        tensors: list[tuple[str, np.ndarray]] = []
        for _ in range(count):
            name = text(handle)
            dtype_id = int(scalar(handle, "I"))
            elements = int(scalar(handle, "Q"))
            if dtype_id not in dtype_map:
                raise RuntimeError(f"unsupported BaseAI dtype for {name}")
            dtype, width = dtype_map[dtype_id]
            value = np.frombuffer(
                exact(handle, elements * width), dtype=dtype
            ).copy()
            if dtype_id != 0:
                raise RuntimeError(
                    f"expected Float32 ResNet tensor, found {name}"
                )
            tensors.append((name, value))
    return metadata, tensors


def export_resnet_stage4(
    checkpoint: dict,
    base_path: Path,
    destination: Path,
) -> None:
    if "resnet" not in checkpoint:
        raise RuntimeError("source checkpoint has no fine-tuned ResNet")
    metadata, tensors = read_base_model(base_path)
    state = checkpoint["resnet"]
    replacements: dict[str, np.ndarray] = {}

    def unit(source: str, target: str) -> None:
        replacements[source + ".weight"] = as_float32(
            state[target + ".weight"]
        )
        batch_norm = target.replace("conv", "bn")
        replacements[source + ".gamma"] = as_float32(
            state[batch_norm + ".weight"]
        )
        replacements[source + ".beta"] = as_float32(
            state[batch_norm + ".bias"]
        )
        replacements[source + ".running_mean"] = as_float32(
            state[batch_norm + ".running_mean"]
        )
        replacements[source + ".running_var"] = as_float32(
            state[batch_norm + ".running_var"]
        )

    for local_block, global_block in enumerate(range(13, 16)):
        source = f"blocks.{global_block}"
        target = f"layer4.{local_block}"
        for convolution in ("conv1", "conv2", "conv3"):
            unit(f"{source}.{convolution}", f"{target}.{convolution}")
        projection = source + ".projection"
        if local_block == 0:
            replacements[projection + ".weight"] = as_float32(
                state[target + ".downsample.0.weight"]
            )
            replacements[projection + ".gamma"] = as_float32(
                state[target + ".downsample.1.weight"]
            )
            replacements[projection + ".beta"] = as_float32(
                state[target + ".downsample.1.bias"]
            )
            replacements[projection + ".running_mean"] = as_float32(
                state[target + ".downsample.1.running_mean"]
            )
            replacements[projection + ".running_var"] = as_float32(
                state[target + ".downsample.1.running_var"]
            )
    output: list[tuple[str, np.ndarray]] = []
    replaced = set()
    for name, values in tensors:
        if name in replacements:
            replacement = replacements[name]
            if replacement.size != values.size:
                raise RuntimeError(
                    f"ResNet stage-4 shape mismatch for {name}"
                )
            output.append((name, replacement))
            replaced.add(name)
        else:
            output.append((name, values))
    missing = set(replacements) - replaced
    if missing:
        raise RuntimeError(
            "BaseAI ResNet is missing stage-4 tensors: " +
            ", ".join(sorted(missing))
        )
    write_checkpoint(
        destination,
        output,
        int(checkpoint.get("resnet_training_updates", metadata["global_step"])),
        metadata["best_loss"],
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-resnet", type=Path)
    parser.add_argument("--output-resnet", type=Path)
    parser.add_argument(
        "--representation-only",
        action="store_true",
        help="preserve temporal representation but reset Q heads/update count",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    checkpoint = torch.load(
        arguments.input, map_location="cpu", weights_only=False
    )
    tensors = (
        representation_only_tensors(checkpoint)
        if arguments.representation_only
        else recurrent_tensors(checkpoint)
    )
    global_step = (
        0 if arguments.representation_only
        else int(checkpoint.get("training_updates", 0))
    )
    write_checkpoint(
        arguments.output,
        tensors,
        global_step,
        float(checkpoint.get("last_loss", 0.0)),
    )
    print(
        f"[PASS] recurrent checkpoint converted: {arguments.output} "
        f"tensors={len(tensors)} optimizer=reset replay=reset "
        f"heads={'reset' if arguments.representation_only else 'preserved'}"
    )
    if arguments.output_resnet is not None:
        if arguments.base_resnet is None:
            raise RuntimeError("--output-resnet requires --base-resnet")
        if "resnet" in checkpoint:
            export_resnet_stage4(
                checkpoint, arguments.base_resnet, arguments.output_resnet
            )
            print(
                f"[PASS] fine-tuned ResNet stage 4 exported: "
                f"{arguments.output_resnet}"
            )
        else:
            print("[INFO] source has no fine-tuned ResNet stage 4; skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
