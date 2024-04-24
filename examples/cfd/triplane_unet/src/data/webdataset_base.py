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

import io
import os
from typing import Dict, List, Optional, Union, Callable

# TODO(akamenev): migration
# import fire
import numpy as np
import webdataset as wds
from torch.utils.data import IterableDataset


class PreprocessingFunctorBase:
    """Base class for a callable preprocess function for the webdataset."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, sample):
        raise NotImplementedError


class NumpyPreprocessingFunctor(PreprocessingFunctorBase):
    """numpy preprocessor."""

    def __init__(self, np_ext: str = "npz", **kwargs):
        super().__init__(**kwargs)
        self.np_ext = np_ext

    def __call__(self, sample):
        np_obj = np.load(io.BytesIO(sample[self.np_ext]), allow_pickle=True)
        return {k: np_obj[k] for k in np_obj.files}


class Webdataset(IterableDataset):
    """Webdataset-based dataset."""

    def __init__(
        self,
        paths: Union[str, List[str]],
        preprocess_fn: Callable,
    ) -> None:
        super().__init__()
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            assert os.path.exists(path), f"Path {path} does not exist"

        self._dataset = wds.WebDataset(paths).map(lambda x: preprocess_fn(x))

    def shuffle(self, buffer_size: int) -> IterableDataset:
        self._dataset = self._dataset.shuffle(buffer_size)
        return self

    def __iter__(self):
        return iter(self._dataset)


def test_webdataset(path: str):
    dataset = Webdataset(path, NumpyPreprocessingFunctor(np_ext="npz"))
    for i, data in enumerate(dataset):
        for k, v in data.items():
            print(i, k, v.shape if isinstance(v, np.ndarray) else v)
        if i > 1:
            break


if __name__ == "__main__":
    import sys

    test_webdataset(sys.argv[1])
