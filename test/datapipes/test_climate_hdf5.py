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

import os
import pytest
import torch

from typing import Tuple
from . import common
from pytest_utils import import_or_fail

Tensor = torch.Tensor


@pytest.fixture
def data_dir():
    path = "/data/nfs/modulus-data/datasets/hdf5/test/"
    if not os.path.exists(path):
        pytest.skip("NFS volumes not set up. Run `make get-data` from the root directory of the repo")
    return path

@pytest.fixture
def stats_dir():
    path = "/data/nfs/modulus-data/datasets/hdf5/stats/"
    if not os.path.exists(path):
        pytest.skip("NFS volumes not set up. Run `make get-data` from the root directory of the repo")
    return path

@pytest.fixture
def lsm_filename():
    path = "/data/nfs/modulus-data/datasets/hdf5/static/land_sea_mask.nc"
    if not os.path.exists(path):
        pytest.skip("NFS volumes not set up. Run `make get-data` from the root directory of the repo")
    return path

@pytest.fixture
def geopotential_filename():
    path = "/data/nfs/modulus-data/datasets/hdf5/static/geopotential.nc"
    if not os.path.exists(path):
        pytest.skip("NFS volumes not set up. Run `make get-data` from the root directory of the repo")
    return path

@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_climate_hdf5_constructor(
    data_dir, stats_dir, lsm_filename, geopotential_filename, device, pytestconfig
):
    
    import_or_fail("netCDF4", pytestconfig)
    from modulus.experimental.datapipes.climate import ClimateHDF5Datapipe

    # construct data pipe
    datapipe = ClimateHDF5Datapipe(
        data_dir=data_dir,
        stats_dir=stats_dir,
        channels=None,
        stride=1,
        dt=1,
        start_year=2018,
        num_steps=1,
        patch_size=8,
        num_samples_per_year=None,
        batch_size=1,
        lsm_filename=lsm_filename,
        geopotential_filename=geopotential_filename,
        use_cos_zenith=True,
        use_latlon=True,
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
        datapipe = ClimateHDF5Datapipe(
            data_dir="/null_path",
            stats_dir=stats_dir,
            channels=None,
            stride=1,
            dt=1,
            start_year=2018,
            num_steps=1,
            patch_size=8,
            num_samples_per_year=None,
            batch_size=1,
            lsm_filename=lsm_filename,
            geopotential_filename=geopotential_filename,
            use_cos_zenith=True,
            use_latlon=True,
            num_workers=1,
            shuffle=False,
            device=torch.device(device),
        )
        raise IOError("Failed to raise error given null data path")
    except IOError:
        pass

    # check for failure from invalid dir
    try:
        # init datapipe with empty path
        # if datapipe throws an IO error then this should pass
        datapipe = ClimateHDF5Datapipe(
            data_dir=data_dir,
            stats_dir="/null_path",
            channels=None,
            stride=1,
            dt=1,
            start_year=2018,
            num_steps=1,
            patch_size=8,
            num_samples_per_year=None,
            batch_size=1,
            lsm_filename=lsm_filename,
            geopotential_filename=geopotential_filename,
            use_cos_zenith=True,
            use_latlon=True,
            num_workers=1,
            shuffle=False,
            device=torch.device(device),
        )
        raise IOError("Failed to raise error given null stats path")
    except IOError:
        pass

    # check for failure from invalid num_samples_per_year
    try:
        datapipe = ClimateHDF5Datapipe(
            data_dir=data_dir,
            stats_dir=stats_dir,
            channels=None,
            stride=1,
            dt=1,
            start_year=2018,
            num_steps=1,
            patch_size=8,
            num_samples_per_year=100,
            batch_size=1,
            lsm_filename=lsm_filename,
            geopotential_filename=geopotential_filename,
            use_cos_zenith=True,
            use_latlon=True,
            num_workers=1,
            shuffle=False,
            device=torch.device(device),
        )
        raise ValueError("Failed to raise error given invalid num_samples_per_year")
    except ValueError:
        pass

    # check invalid channel
    try:
        datapipe = ClimateHDF5Datapipe(
            data_dir=data_dir,
            stats_dir=stats_dir,
            channels=[20],
            stride=1,
            dt=1,
            start_year=2018,
            num_steps=1,
            patch_size=8,
            num_samples_per_year=None,
            batch_size=1,
            lsm_filename=lsm_filename,
            geopotential_filename=geopotential_filename,
            use_cos_zenith=True,
            use_latlon=True,
            num_workers=1,
            shuffle=False,
            device=torch.device(device),
        )
        raise ValueError("Failed to raise error given invalid channel id")
    except ValueError:
        pass


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_climate_hdf5_device(
    data_dir, stats_dir, lsm_filename, geopotential_filename, device, pytestconfig
):
    
    import_or_fail("netCDF4", pytestconfig)
    from modulus.experimental.datapipes.climate import ClimateHDF5Datapipe

    # construct data pipe
    datapipe = ClimateHDF5Datapipe(
        data_dir=data_dir,
        stats_dir=stats_dir,
        channels=None,
        stride=1,
        dt=1,
        start_year=2018,
        num_steps=1,
        patch_size=8,
        num_samples_per_year=None,
        batch_size=1,
        lsm_filename=lsm_filename,
        geopotential_filename=geopotential_filename,
        use_cos_zenith=True,
        use_latlon=True,
        num_workers=1,
        shuffle=False,
        device=torch.device(device),
    )

    # test single sample
    for data in datapipe:
        common.check_datapipe_device(data[0]["state_seq"], device)
        common.check_datapipe_device(data[0]["timestamps"], device)
        common.check_datapipe_device(data[0]["land_sea_mask"], device)
        common.check_datapipe_device(data[0]["geopotential"], device)
        common.check_datapipe_device(data[0]["latlon"], device)
        common.check_datapipe_device(data[0]["cos_latlon"], device)
        common.check_datapipe_device(data[0]["cos_zenith"], device)
        break


@pytest.mark.parametrize("data_channels", [[0, 1]])
@pytest.mark.parametrize("num_steps", [2])
@pytest.mark.parametrize("patch_size", [None])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_climate_hdf5_shape(
    data_dir,
    stats_dir,
    lsm_filename,
    geopotential_filename,
    data_channels,
    num_steps,
    patch_size,
    batch_size,
    device,
    pytestconfig
):
    
    import_or_fail("netCDF4", pytestconfig)
    from modulus.experimental.datapipes.climate import ClimateHDF5Datapipe

    # construct data pipe
    datapipe = ClimateHDF5Datapipe(
        data_dir=data_dir,
        stats_dir=stats_dir,
        channels=data_channels,
        stride=1,
        dt=1,
        start_year=2018,
        num_steps=num_steps,
        patch_size=patch_size,
        num_samples_per_year=None,
        batch_size=batch_size,
        lsm_filename=lsm_filename,
        geopotential_filename=geopotential_filename,
        use_cos_zenith=True,
        use_latlon=True,
        num_workers=1,
        shuffle=False,
        device=torch.device(device),
    )

    # test single sample
    for data in datapipe:
        state_seq = data[0]["state_seq"]
        timestamps = data[0]["timestamps"]
        land_sea_mask = data[0]["land_sea_mask"]
        geopotential = data[0]["geopotential"]
        latlon = data[0]["latlon"]
        cos_latlon = data[0]["cos_latlon"]
        cos_zenith = data[0]["cos_zenith"]

        # check batch size
        assert common.check_batch_size(
            [
                state_seq,
                timestamps,
                land_sea_mask,
                geopotential,
                latlon,
                cos_latlon,
                cos_zenith,
            ],
            batch_size,
        )

        # check seq length
        assert common.check_seq_length([state_seq, timestamps, cos_zenith], num_steps)

        # check channels
        if data_channels is None:
            nr_channels = 3
        else:
            nr_channels = len(data_channels)
        assert common.check_channels(state_seq, nr_channels, axis=2)

        # check grid dims
        if patch_size is None:
            patch_size = (721, 1440)
        assert common.check_grid([state_seq, cos_zenith], patch_size, axis=(3, 4))
        assert common.check_grid(
            [land_sea_mask, geopotential, latlon, cos_latlon], patch_size, axis=(2, 3)
        )
        break


@pytest.mark.parametrize("num_steps", [1, 2])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_era5_hdf5_sequence(
    data_dir, stats_dir, lsm_filename, geopotential_filename, num_steps, stride, device, pytestconfig
):

    import_or_fail("netCDF4", pytestconfig)
    from modulus.experimental.datapipes.climate import ClimateHDF5Datapipe

    # construct data pipe
    datapipe = ClimateHDF5Datapipe(
        data_dir=data_dir,
        stats_dir=stats_dir,
        channels=None,
        stride=stride,
        dt=1,
        start_year=2018,
        num_steps=num_steps,
        patch_size=None,
        num_samples_per_year=None,
        batch_size=1,
        lsm_filename=lsm_filename,
        geopotential_filename=geopotential_filename,
        use_cos_zenith=True,
        use_latlon=True,
        num_workers=1,
        shuffle=False,
        device=torch.device(device),
    )

    # get single sample
    # TODO generalize tests for sequence type datapipes
    for data in datapipe:
        state_seq = data[0]["state_seq"]
        break

    # check if tensor has correct shape
    print(state_seq[0, 0, 0, 0, 0])
    if num_steps > 1:
        print(state_seq[0, 1, 0, 0, 0])
    assert common.check_sequence(
        state_seq, start_index=0, step_size=stride, seq_length=num_steps, axis=1
    )


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_era5_hdf5_shuffle(
    data_dir, stats_dir, lsm_filename, geopotential_filename, shuffle, stride, device, pytestconfig
):

    import_or_fail("netCDF4", pytestconfig)
    from modulus.experimental.datapipes.climate import ClimateHDF5Datapipe

    # construct data pipe
    datapipe = ClimateHDF5Datapipe(
        data_dir=data_dir,
        stats_dir=stats_dir,
        channels=None,
        stride=stride,
        dt=1,
        start_year=2018,
        num_steps=2,
        patch_size=None,
        num_samples_per_year=None,
        batch_size=1,
        lsm_filename=lsm_filename,
        geopotential_filename=geopotential_filename,
        use_cos_zenith=True,
        use_latlon=True,
        num_workers=1,
        shuffle=shuffle,
        device=torch.device(device),
    )

    # get all samples
    # TODO generalize this
    tensors = []
    for data in datapipe:
        tensors.append(data[0]["state_seq"])

    # check sample order
    assert common.check_shuffle(tensors, shuffle, stride, 8)


@pytest.mark.parametrize("device", ["cuda:0"])
def test_era5_hdf5_cudagraphs(
    data_dir, stats_dir, lsm_filename, geopotential_filename, device, pytestconfig
):
    
    import_or_fail("netCDF4", pytestconfig)
    from modulus.experimental.datapipes.climate import ClimateHDF5Datapipe

    # Preprocess function to convert dataloader output into Tuple of tensors
    def input_fn(data) -> Tensor:
        return data[0]["state_seq"]

    # construct data pipe
    datapipe = ClimateHDF5Datapipe(
        data_dir=data_dir,
        stats_dir=stats_dir,
        channels=None,
        stride=1,
        dt=1,
        start_year=2018,
        num_steps=1,
        patch_size=None,
        num_samples_per_year=None,
        batch_size=1,
        lsm_filename=lsm_filename,
        geopotential_filename=geopotential_filename,
        use_cos_zenith=True,
        use_latlon=True,
        num_workers=1,
        shuffle=False,
        device=torch.device(device),
    )

    assert common.check_cuda_graphs(datapipe, input_fn)
