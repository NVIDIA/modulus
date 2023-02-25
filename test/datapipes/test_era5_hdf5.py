# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import torch

from typing import Tuple
from modulus.datapipes.climate import ERA5HDF5Datapipe
from . import common

Tensor = torch.Tensor


@pytest.fixture
def data_dir():
    return "/data/nfs/modulus-data/datasets/hdf5/test/"


@pytest.fixture
def stats_dir():
    return "/data/nfs/modulus-data/datasets/hdf5/stats/"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_era5_hdf5_constructor(data_dir, stats_dir, device):

    # construct data pipe
    datapipe = ERA5HDF5Datapipe(
        data_dir=data_dir,
        stats_dir=None,
        channels=None,
        stride=1,
        num_steps=1,
        patch_size=8,
        num_samples_per_year=None,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        device=torch.device(device),
    )

    # iterate datapipe is iterable
    common.check_datapipe_iterable(datapipe)

    # check for failure from invalid dir
    try:
        # init datapipe with empty path
        # if datapipe throws an IO error then this should pass
        datapipe = ERA5HDF5Datapipe(
            data_dir="/null_path",
            stats_dir="/null_path",
            channels=None,
            stride=1,
            num_steps=1,
            patch_size=None,
            num_samples_per_year=1,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            device=device,
        )
        raise IOError("Failed to raise error given null data path")
    except IOError:
        pass

    # check for failure from invalid dir
    try:
        # init datapipe with empty path
        # if datapipe throws an IO error then this should pass
        datapipe = ERA5HDF5Datapipe(
            data_dir=data_dir,
            stats_dir="/null_path",
            channels=None,
            stride=1,
            num_steps=1,
            patch_size=None,
            num_samples_per_year=1,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            device=device,
        )
        raise IOError("Failed to raise error given null stats path")
    except IOError:
        pass

    # check for failure from invalid num_samples_per_year
    try:
        datapipe = ERA5HDF5Datapipe(
            data_dir=data_dir,
            stats_dir=stats_dir,
            channels=None,
            stride=1,
            num_steps=1,
            patch_size=None,
            num_samples_per_year=100,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            device=device,
        )
        raise ValueError("Failed to raise error given invalid num_samples_per_year")
    except ValueError:
        pass

    # check invalid channel
    try:
        datapipe = ERA5HDF5Datapipe(
            data_dir=data_dir,
            stats_dir=stats_dir,
            channels=[20],
            stride=1,
            num_steps=1,
            patch_size=None,
            num_samples_per_year=1,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            device=device,
        )
        raise ValueError("Failed to raise error given invalid channel id")
    except ValueError:
        pass


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_era5_hdf5_device(data_dir, stats_dir, device):

    # construct data pipe
    datapipe = ERA5HDF5Datapipe(
        data_dir=data_dir,
        stats_dir=stats_dir,
        channels=None,
        stride=1,
        num_steps=1,
        patch_size=None,
        num_samples_per_year=None,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        device=device,
    )

    # test single sample
    for data in datapipe:
        common.check_datapipe_device(data[0]["invar"], device)
        common.check_datapipe_device(data[0]["outvar"], device)
        break


@pytest.mark.parametrize("data_channels", [[0, 1]])
@pytest.mark.parametrize("num_steps", [2])
@pytest.mark.parametrize("patch_size", [None])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_era5_hdf5_shape(
    data_dir, stats_dir, data_channels, num_steps, patch_size, batch_size, device
):

    # construct data pipe
    datapipe = ERA5HDF5Datapipe(
        data_dir=data_dir,
        stats_dir=stats_dir,
        channels=data_channels,
        stride=1,
        num_steps=num_steps,
        patch_size=patch_size,
        num_samples_per_year=None,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        device=device,
    )

    # test single sample
    for data in datapipe:
        input = data[0]["invar"]
        output = data[0]["outvar"]

        # check batch size
        assert common.check_batch_size([input, output], batch_size)

        # check seq length
        assert common.check_seq_length(output, num_steps)

        # check channels
        if data_channels is None:
            nr_channels = 3
        else:
            nr_channels = len(data_channels)
        assert common.check_channels(input, nr_channels, axis=1)
        assert common.check_channels(output, nr_channels, axis=2)

        # check grid dims
        if patch_size is None:
            patch_size = (721, 1440)
        assert common.check_grid(input, patch_size, axis=(2, 3))
        assert common.check_grid(output, patch_size, axis=(3, 4))
        break


@pytest.mark.parametrize("num_steps", [1, 2])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_era5_hdf5_sequence(data_dir, stats_dir, num_steps, stride, device):

    # construct data pipe
    datapipe = ERA5HDF5Datapipe(
        data_dir=data_dir,
        stats_dir=stats_dir,
        channels=None,
        stride=stride,
        num_steps=num_steps,
        patch_size=None,
        num_samples_per_year=None,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        device=device,
    )

    # get single sample
    # TODO generalize tests for sequence type datapipes
    for data in datapipe:
        output = data[0]["outvar"]
        break

    # check if tensor has correct shape
    assert common.check_sequence(
        output, start_index=stride, step_size=stride, seq_length=num_steps, axis=1
    )


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_era5_hdf5_shuffle(data_dir, stats_dir, shuffle, stride, device):

    # construct data pipe
    datapipe = ERA5HDF5Datapipe(
        data_dir=data_dir,
        stats_dir=stats_dir,
        channels=None,
        stride=stride,
        num_steps=1,
        patch_size=None,
        num_samples_per_year=None,
        batch_size=1,
        num_workers=1,
        shuffle=shuffle,
        device=device,
    )

    # get all samples
    # TODO generalize this
    tensors = []
    for data in datapipe:
        tensors.append(data[0]["invar"])

    # check sample order
    assert common.check_shuffle(tensors, shuffle, stride, 8)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_era5_hdf5_cudagraphs(data_dir, stats_dir, device):

    # Preprocess function to convert dataloader output into Tuple of tensors
    def input_fn(data) -> Tuple[Tensor, ...]:
        return (data[0]["invar"], data[0]["outvar"])

    # construct data pipe
    datapipe = ERA5HDF5Datapipe(
        data_dir=data_dir,
        stats_dir=stats_dir,
        channels=None,
        stride=1,
        num_steps=1,
        patch_size=None,
        num_samples_per_year=None,
        batch_size=1,
        num_workers=1,
        device=device,
    )

    assert common.check_cuda_graphs(datapipe, input_fn)
