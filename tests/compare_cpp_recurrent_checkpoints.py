#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.convert_recurrent_checkpoint_to_baseai import read_base_model  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare EXPECTED ACTUAL")
    expected_meta, expected_values = read_base_model(Path(sys.argv[1]))
    actual_meta, actual_values = read_base_model(Path(sys.argv[2]))
    expected = dict(expected_values)
    actual = dict(actual_values)
    names = sorted(
        name for name in expected
        if name.startswith("online_a.") or name.startswith("online_b.")
    )
    if not names or any(name not in actual for name in names):
        raise RuntimeError("recurrent model tensor set mismatch")
    maximum = 0.0
    maximum_name = ""
    for name in names:
        if expected[name].shape != actual[name].shape:
            raise RuntimeError(f"tensor shape mismatch: {name}")
        difference = float(
            np.max(np.abs(expected[name].astype(np.float64) -
                          actual[name].astype(np.float64)))
        )
        if difference > maximum:
            maximum = difference
            maximum_name = name
    if int(expected_meta["global_step"]) != int(actual_meta["global_step"]):
        raise RuntimeError("training update count mismatch")
    # CUDA reductions/GEMMs need not be bit-identical to PyTorch/cuDNN, but a
    # single update should remain within a tight numerical tolerance.
    if maximum > 5.0e-5:
        raise RuntimeError(
            f"one-step BPTT parity exceeded tolerance: {maximum} ({maximum_name})"
        )
    print(
        "[INFO] PyTorch/C++ one-step maximum_abs="
        f"{maximum:.9g} tensor={maximum_name}"
    )
    print("[PASS] Native LSTM BPTT/AdamW parity passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
