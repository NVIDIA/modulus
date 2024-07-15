# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

from datasets.cwb import (
    _ZarrDataset,
    FilterTime,
    get_zarr_dataset,
)
import torch
import datetime
import numpy as np
import os
import joblib


import pytest

path = os.getenv(
    "CWB_DATA_PATH", "/lustre/fsw/sw_climate_fno/nbrenowitz/2023-01-24-cwb-4years.zarr"
)


@pytest.fixture
def dataset():
    if not os.path.isdir(path):
        pytest.skip()
    else:
        return _ZarrDataset(path)


def test_zarr_dataset(dataset):
    dataset[0]
    assert len(dataset) > 0
    assert dataset[len(dataset) - 1]


def test_zarr_dataset_channels(dataset):
    assert dataset.input_channels()
    assert dataset.output_channels()


def test_zarr_dataset_time(dataset):
    isinstance(dataset.time()[0].year, int)


def test_zarr_dataset_get_valid_time_index(dataset):
    ans = dataset._get_valid_time_index(0)
    assert isinstance(ans, np.int64)


def test_filter_time():
    class MockData(torch.utils.data.Dataset):
        def __getitem__(self, idx):
            return self.time()[idx]

        def time(self):
            return [datetime.datetime(2018, 1, 1), datetime.datetime(1970, 1, 1)]

    data = MockData()
    filtered = FilterTime(data, lambda time: time.year > 1990)
    assert filtered.time() == [datetime.datetime(2018, 1, 1)]
    assert filtered[0]


def hash_array(arr, tol=1e-3):
    intarray = (np.asarray(arr) * tol).astype(int)
    return joblib.hash(intarray)


def test_Zarr_Dataset(regtest):
    # reset regression data with pytest --regtest-reset
    ds = get_zarr_dataset(data_path=path)
    inp, outp, idx = ds[0]
    print("index", idx, file=regtest)
    print("input", hash_array(inp), file=regtest)
    print("output", hash_array(outp), file=regtest)
