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

from torch.utils.data import DataLoader, Dataset


class BaseDataModule:
    @property
    def train_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    def val_dataset(self) -> Dataset:
        raise NotImplementedError

    @property
    def test_dataset(self) -> Dataset:
        raise NotImplementedError

    def train_dataloader(self, **kwargs) -> DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        return DataLoader(self.train_dataset, collate_fn=collate_fn, **kwargs)

    def val_dataloader(self, **kwargs) -> DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        return DataLoader(self.val_dataset, collate_fn=collate_fn, **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        return DataLoader(self.test_dataset, collate_fn=collate_fn, **kwargs)
