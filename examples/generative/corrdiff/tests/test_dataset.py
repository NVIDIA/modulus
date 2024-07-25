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


def test_train_test_split():
    ds = get_zarr_dataset(data_path=path, train=True)
    assert not any(t.year == 2021 for t in ds.time())

    ds = get_zarr_dataset(data_path=path, train=False)
    assert all(t.year == 2021 for t in ds.time())


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


def test_input_normalization():
    ds = get_zarr_dataset(data_path=path)
    input = np.ones([1, len(ds.input_channels()), 3, 3])
    round_trip = ds.denormalize_input(ds.normalize_input(input))
    np.testing.assert_array_almost_equal(round_trip, input)


def test_output_normalization():
    ds = get_zarr_dataset(data_path=path)
    input = np.ones([1, len(ds.output_channels()), 3, 3])
    round_trip = ds.denormalize_output(ds.normalize_output(input))
    np.testing.assert_array_almost_equal(round_trip, input)
