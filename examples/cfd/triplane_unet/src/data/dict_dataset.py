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

from typing import Callable, Iterable, Optional

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


class DictDataset(Dataset):
    def __init__(self, data_dict: dict):
        self.data_dict = data_dict
        for k, v in data_dict.items():
            assert len(v) == len(
                data_dict[list(data_dict.keys())[0]]
            ), "All data must have the same length"

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data_dict.items()}

    def __len__(self):
        return len(self.data_dict[list(self.data_dict.keys())[0]])


class DictDatasetWithConstant(DictDataset):
    def __init__(self, data_dict: dict, constant_dict: dict):
        super().__init__(data_dict)
        self.constant_dict = constant_dict

    def __getitem__(self, index):
        return_dict = {k: v[index] for k, v in self.data_dict.items()}
        return_dict.update(self.constant_dict)
        return return_dict


class MappingDatasetWrapper(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        mapping: dict,
        pre_sample_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        num_points: Optional[int] = None,
    ) -> None:
        r"""
        Args:
            dataset (Dataset): The dataset to wrap
            mapping (dict): The mapping from the dataset to the new dataset. If the key is comma separated, it will be interpreted as dataset_key, subkey.
            transform (Optional[Callable], optional): A callable to transform the data. Defaults to None.
            num_points (Optional[int], optional): If not None, only return random sampled points. Defaults to None.
        """
        self.dataset = dataset
        self.mapping = mapping
        self.pre_sample_transform = pre_sample_transform
        self.transform = transform
        self.num_points = num_points

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]

        return_dict = {}
        for k, v in self.mapping.items():
            # If comma doesn't exist, just use the key
            if "," not in k:
                if k not in item:
                    raise KeyError(f"Key {k} not found in dataset")
                return_value = item[k]
            else:
                # Split the key into dataset_key, subkey
                dataset_key, subkey = k.split(",")
                # Get the value from the dataset
                if dataset_key not in item:
                    raise KeyError(f"Key {dataset_key} not found in dataset")
                if subkey not in item[dataset_key]:
                    raise KeyError(f"Key {subkey} not found in dataset[{dataset_key}]")

                return_value = item[dataset_key][subkey]
                return_dict[v] = return_value

        if self.pre_sample_transform is not None:
            return_dict = self.pre_sample_transform(return_dict)

        # Sampling
        _indices = None
        if self.num_points is not None:
            for k, v in return_dict.items():
                if (isinstance(v, Iterable)
                    or isinstance(v, np.ndarray)
                    or isinstance(v, Tensor)
                ):
                    # Select indices
                    if _indices is None:
                        _indices = np.random.choice(len(v), self.num_points, replace=False)
                    v = v[_indices]
                return_dict[v] = v

        if self.transform is not None:
            return_dict = self.transform(return_dict)

        return return_dict
