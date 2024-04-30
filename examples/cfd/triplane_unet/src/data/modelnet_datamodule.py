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

import sys
import glob
import os
import copy
import os.path as osp
from pathlib import Path
from typing import Optional, List, Literal, Callable, Union, Tuple
import numpy as np

import torch
from torch import Tensor
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Dataset

from .mesh_utils import read_off
from .base_datamodule import BaseDataModule
from .components import ComposePreprocessors


class ProcessMesh:
    """
    Process ModelNet mesh files for a specific category
    """

    def __init__(self, dir_path, phase, categories, transform=None):
        self.dir_path = dir_path
        self.phase = phase
        self.transform = transform
        self.categories = categories

    def __call__(self, category):
        data_list = []
        target = self.categories.index(category)
        folder = osp.join(self.dir_path, category, self.phase)
        paths = glob.glob(f"{folder}/{category}_*.off")
        for i, path in enumerate(paths):
            vertices, faces = read_off(path)
            data = {"vertices": vertices, "class": target}
            if self.transform is not None:
                data = self.transform(data)

            # Check validity
            if data["vertices"].shape[0] == 0:
                continue
            if not torch.isfinite(data["vertices"]).all():
                continue
            if data["class"] is None:
                continue

            data_list.append(data)

            # print every 100 files
            if i % 100 == 0:
                print(f"{category} {self.phase}: {i}/{len(paths)}")

        torch.save(data_list, osp.join(self.dir_path, f"{category}_{self.phase}.pt"))


def preprocess(
    dir_path: str,
    phase: Literal["train", "test"],
    num_processes: int = 8,
    out_path: Optional[str] = None,
    transform: Optional[Callable] = None,
) -> None:
    # Assume that the input is unzipped http://modelnet.cs.princeton.edu/ModelNet40.zip
    categories = glob.glob(osp.join(dir_path, "*", ""))
    categories = sorted([x.split(os.sep)[-2] for x in categories])

    # Multiprocessing
    from multiprocessing import Pool

    with Pool(num_processes) as p:
        p.map(
            ProcessMesh(
                dir_path=dir_path,
                phase=phase,
                categories=categories,
                transform=transform,
            ),
            categories,
        )

    # Load all category specific pt files and concatenate them
    all_data = []
    for category in categories:
        data = torch.load(osp.join(dir_path, f"{category}_{phase}.pt"))
        all_data.extend(data)
    torch.save(all_data, osp.join(dir_path, f"{phase}.pt"))


class ModelNet40Dataset(Dataset):
    """The ModelNet40 dataset from http://modelnet.cs.princeton.edu/ModelNet40.zip"""

    def __init__(
        self,
        data_dir: str,
        phase: Literal["train", "test"] = "train",
        preprocessors: Optional[List[Callable]] = None,
    ) -> None:
        Dataset.__init__(self)
        categories = glob.glob(osp.join(data_dir, "*", ""))
        self.categories = sorted([x.split(os.sep)[-2] for x in categories])
        self.phase = phase
        path = osp.join(data_dir, f"{phase}.pt")
        self._data = torch.load(path)
        self.preprocessors = ComposePreprocessors(preprocessors)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        datum = self._data[idx]
        datum = self.preprocessors(datum)
        return datum


class RandomSampler:
    """RandomSampler"""

    def __init__(self, num_points: int = 4096):
        self.num_points = num_points

    def __call__(self, data: dict) -> dict:
        vertices = data["vertices"]
        # Create random indices
        indices = np.random.choice(len(vertices), self.num_points, replace=True)
        data["vertices"] = vertices[indices]
        return data


class NormalizeVertices:
    """NormalizeVertices"""

    def __init__(
        self,
        bbox_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
        bbox_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        bbox_margin: float = 0.1,
    ):
        self.bbox_min = torch.Tensor(bbox_min) + bbox_margin
        self.bbox_max = torch.Tensor(bbox_max) - bbox_margin

    def __call__(self, data: dict) -> dict:
        vertices = data["vertices"]
        min_vertices = vertices.min(0)[0]
        max_vertices = vertices.max(0)[0]
        # Find scale and shift
        scale = 1.0 / (max_vertices - min_vertices).max()
        vertices = scale * (vertices - min_vertices)
        vertices = vertices * (self.bbox_max - self.bbox_min) + self.bbox_min
        data["vertices"] = vertices
        return data


class PerturbVertices:
    """PerturbVertices"""

    def __init__(self, pert_std: float = 0.001):
        self.pert_std = pert_std

    def __call__(self, data: dict) -> dict:
        vertices = data["vertices"]
        vertices += torch.randn_like(vertices) * self.pert_std
        data["vertices"] = vertices
        return data


class ModelNet40DataModule(BaseDataModule):
    """
    ModelNet40DataModule
    """

    def __init__(
        self,
        data_path: Union[Path, str],
        num_points: int = 8192,
        pert_std: float = 0.001,
        bbox_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        bbox_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
        bbox_margin: float = 0.1,
        preprocessors: List[Callable] = None,
    ):
        """
        Args:
            data_path (Union[Path, str]): Path that contains train and test directories
        """
        super().__init__()

        if isinstance(data_path, str):
            data_path = Path(data_path)
        assert (
            data_path.exists() and data_path.is_dir()
        ), f"{data_path} must exist and should be a directory"

        self.data_dir = data_path

        if preprocessors is None:
            preprocessors = []

        preprocessors = [
            RandomSampler(num_points=num_points),
            NormalizeVertices(
                bbox_min=bbox_min, bbox_max=bbox_max, bbox_margin=bbox_margin
            ),
            *preprocessors,
        ]
        self.preprocessors = preprocessors
        self._train_dataset = self._create_dataset("train")
        self._val_dataset = self._create_dataset("test")
        self._test_dataset = self._create_dataset("test")

    def _create_dataset(self, phase: str) -> Dataset:
        # Add PerturbVertices to train dataset
        preprocessors = copy.deepcopy(self.preprocessors)
        if phase == "train":
            preprocessors += [PerturbVertices(pert_std=0.001)]

        dataset = ModelNet40Dataset(
            self.data_dir, phase=phase, preprocessors=preprocessors
        )
        return dataset

    def encode(self, x):
        return self.normalizer.encode(x)

    def decode(self, x):
        return self.normalizer.decode(x)

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    def _create_dataloader(self, dataset: Dataset, **kwargs) -> DataLoader:
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self, **kwargs) -> DataLoader:
        return self._create_dataloader(self.train_dataset, **kwargs)

    def val_dataloader(self, **kwargs) -> DataLoader:
        return self._create_dataloader(self.val_dataset, **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        return self._create_dataloader(self.test_dataset, **kwargs)


if __name__ == "__main__":
    preprocess(sys.argv[1], "train")
    preprocess(sys.argv[1], "test")
    data_module = ModelNet40DataModule(sys.argv[1])
    train_loader = data_module.train_dataloader(batch_size=32)
    for batch in train_loader:
        print(batch["vertices"].shape)
        print(batch["vertices"].max(1)[0])
        print(batch["vertices"].min(1)[0])
        break
