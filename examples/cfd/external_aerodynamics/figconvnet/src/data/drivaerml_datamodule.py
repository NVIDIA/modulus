# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from collections.abc import Callable, Iterable
import json
from pathlib import Path
from typing import Any

import dgl
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.base_datamodule import BaseDataModule


class DrivAerMLPartitionedDataset(Dataset):
    """DrivAerML partitioned dataset."""

    def __init__(
        self,
        data_path: Path,
        num_points: int = 0,
    ) -> None:
        self.p_files = sorted(data_path.glob("*.bin"))
        self.num_points = num_points

    def __len__(self) -> int:
        return len(self.p_files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(f"Invalid {index = } expected in [0, {len(self)})")

        gs, _ = dgl.load_graphs(str(self.p_files[index]))

        coords = torch.cat([g.ndata["coordinates"] for g in gs], dim=0)
        # Sample indices from the combined graph.
        n_total = coords.shape[0]
        if n_total >= self.num_points:
            indices = np.random.choice(n_total, self.num_points)
        else:
            indices = np.concatenate(
                (
                    np.arange(n_total),
                    np.random.choice(n_total, self.num_points - n_total),
                )
            )
        coords = coords[indices]
        pressure = torch.cat([g.ndata["pressure"] for g in gs], dim=0)[indices]
        shear_stress = torch.cat([g.ndata["shear_stress"] for g in gs], dim=0)[indices]

        return {
            "coordinates": coords,
            "pressure": pressure,
            "shear_stress": shear_stress,
            "design": self.p_files[index].stem.removeprefix("graph_partitions_"),
        }


class DrivAerMLDataModule(BaseDataModule):
    """DrivAerML data module"""

    def __init__(
        self,
        data_path: str | Path,
        num_points: int = 0,
        stats_filename: str = "global_stats.json",
        **kwargs,
    ):
        data_path = Path(data_path)
        self._train_dataset = DrivAerMLPartitionedDataset(
            data_path / "partitions", num_points
        )
        self._val_dataset = DrivAerMLPartitionedDataset(
            data_path / "validation_partitions", num_points
        )
        self._test_dataset = DrivAerMLPartitionedDataset(
            data_path / "test_partitions", num_points
        )

        with open(data_path / stats_filename, "r", encoding="utf-8") as f:
            stats = json.load(f)

        self.mean = {k: torch.tensor(v) for k, v in stats["mean"].items()}
        self.std = {k: torch.tensor(v) for k, v in stats["std_dev"].items()}

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    def encode(self, x: Tensor, name: str):
        return (x - self.mean[name].to(x.device)) / self.std[name].to(x.device)

    def decode(self, x: Tensor, name: str):
        return x * self.std[name].to(x.device) + self.mean[name].to(x.device)
