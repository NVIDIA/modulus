# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit weight initialization and controlled training-state transitions."""

import hashlib
from pathlib import Path

import torch


def load_initial_weights(model, source: Path) -> dict:
    """Load local PhysicsNeMo weights strictly and record their immutable source.

    The caller constructs a fresh optimizer, scheduler and scaler. Their states
    and the source run's preprocessing are not part of a model-only checkpoint.
    """
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Initial model checkpoint does not exist: {source}")

    def file_hash() -> str:
        """Hash the source without keeping a second checkpoint in memory."""
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    before = file_hash()
    model.load(source, strict=True)
    if file_hash() != before:
        raise RuntimeError("Initial model checkpoint changed while being loaded")
    state = model.state_dict()
    if any(not torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("Initial model checkpoint contains non-finite weights")
    return {
        "kind": "weights_only",
        "source_path": str(source),
        "source_sha256": before,
        "loaded_model_state_sha256": _state_hash(state),
        "optimizer_restored": False,
        "scheduler_restored": False,
        "scaler_restored": False,
    }


def _state_hash(value) -> str:
    """Hash optimizer structure and dense tensor bytes, including scalar/BF16 state."""
    digest = hashlib.sha256()

    def frame(tag: str, data: bytes) -> None:
        """Use typed, length-delimited frames to distinguish nested structures."""
        digest.update(tag.encode() + b":" + str(len(data)).encode() + b":" + data)

    def add(item) -> None:
        """Serialize the optimizer's containers, tensor bytes and scalar metadata."""
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            frame("tensor", repr((str(tensor.dtype), tuple(tensor.shape))).encode())
            frame("bytes", tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            frame("dict", str(len(item)).encode())
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                add(key)
                add(item[key])
        elif isinstance(item, (tuple, list)):
            frame(type(item).__name__, str(len(item)).encode())
            for child in item:
                add(child)
        elif item is None or isinstance(item, (str, bool, int, float)):
            frame(type(item).__name__, repr(item).encode())
        else:
            raise TypeError(f"Unsupported optimizer state type: {type(item).__name__}")

    add(value)
    return digest.hexdigest()
