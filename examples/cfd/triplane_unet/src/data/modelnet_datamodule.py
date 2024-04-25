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
import os.path as osp
from typing import Optional, List, Literal, Callable

import torch
from torch import Tensor
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Dataset

from .mesh_utils import read_off


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
    ) -> None:
        super().__init__(self)
        self.phase = phase
        path = osp.join(data_dir, f"{phase}.pt")
        self._data = torch.load(path)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        return self._data[idx]


if __name__ == "__main__":
    preprocess(sys.argv[1], "train")
    preprocess(sys.argv[1], "test")
