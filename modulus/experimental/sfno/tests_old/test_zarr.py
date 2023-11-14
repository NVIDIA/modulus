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

import xarray
import numpy as np
import datetime
from modulus.experimental.sfno.utils.dataloaders import zarr_helper


def test_xarray_to_data_loader():

    # create training data
    shape = (2, 32, 12, 10)
    dims = ["time", "channel", "lat", "lon"]
    times = [datetime.datetime(1980, 1, 1), datetime.datetime(1980, 1, 1, 1)]
    fields = np.ones((shape))
    ds = xarray.DataArray(fields, dims=dims, coords={"time": times}).to_dataset(
        name="fields"
    )

    class params:
        in_channels = [0, 1, 2]
        out_channels = [0, 1]

    loader = zarr_helper._xarray_to_dataset(params, ds, train=True)
    len(loader) == 1
    _, _ = loader[0]

    # none of data is in the test period
    loader = zarr_helper._xarray_to_dataset(params, ds, train=False)
    len(loader) == 0


def test_xarray_data_loader():

    # create training data
    shape = (2, 32, 12, 10)
    dims = ["time", "channel", "lat", "lon"]
    times = [datetime.datetime(1980, 1, 1), datetime.datetime(1980, 1, 1, 1)]
    fields = np.ones((shape))
    ds = xarray.DataArray(fields, dims=dims, coords={"time": times})

    dl = zarr_helper.XarrayDataset(ds)
    assert len(dl) == 1
    _, _ = dl[0]


def test_xarray_sel_inclusive():
    """A test of an upstream API to make sure we understand how xarray works and
    protect against future changes"""
    arr = xarray.DataArray(
        np.ones(
            10,
        ),
        dims=["x"],
    )
    arr["x"] = arr.x

    v = arr.sel(x=slice(0, 5))
    assert len(v) == 6


def test_Map():
    l = [1, 2, 3]

    m = zarr_helper.Map(l, lambda x: 2 * x)
    assert len(m) == 3
    assert list(m) == [2, 4, 6]
