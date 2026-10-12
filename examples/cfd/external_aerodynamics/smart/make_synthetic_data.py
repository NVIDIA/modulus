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

"""Generate original smooth fields for pipeline verification, not CFD benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate(
    path: Path, seed: int = 19, train_cases: int = 8, val_cases: int = 3
) -> None:
    """Create separate train/validation clouds with geometry-dependent analytic targets."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Synthetic data output directory must be empty")
    rng = np.random.default_rng(seed)
    for split, count in (("train", train_cases), ("val", val_cases)):
        directory = path / split
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            radius = rng.uniform(0.5, 1.0)
            geometry = rng.normal(size=(64, 3))
            geometry = (
                radius * geometry / np.linalg.norm(geometry, axis=-1, keepdims=True)
            )
            surface = rng.normal(size=(80, 3))
            surface = radius * surface / np.linalg.norm(surface, axis=-1, keepdims=True)
            volume = rng.uniform(-1.5, 1.5, size=(96, 3))
            parameters = rng.uniform(-0.5, 0.5, size=2)
            pressure = 1 + 0.2 * surface[:, :1] + 0.1 * radius + 0.1 * parameters[0]
            velocity = np.stack(
                (
                    1 + 0.2 * volume[:, 0] + 0.1 * radius + 0.2 * parameters[0],
                    0.5 + 0.1 * volume[:, 1] + 0.1 * parameters[1],
                    0.25 + 0.1 * volume[:, 2] + 0.05 * radius,
                ),
                axis=-1,
            )
            sample = {
                "geometry": geometry,
                "surface_queries": surface,
                "volume_queries": volume,
                "parameters": parameters,
                "surface_targets": pressure,
                "volume_targets": velocity,
            }
            np.savez(
                directory / f"case_{index:04d}.npz",
                **{k: v.astype(np.float32) for k, v in sample.items()},
            )


def main() -> None:
    """Generate data in the requested directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
