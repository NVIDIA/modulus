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

import datetime
import json
import pathlib
from typing import Sequence

import dask.array as da
import h5py
import xarray
from modulus.experimental.sfno.datasets.era5 import time

__all__ = ["open_34_vars"]

METADATA = pathlib.Path(__file__).parent / "data.json"


def open_34_vars(path: str, chunks: Sequence[int] = (1, 1, -1, -1)) -> xarray.DataArray:
    """Open 34Vars hdf5 file

    Args:
        path: local path to hdf5 file
        chunks: the chunks size to use for dask

    Examples:

        >>> import datasets
        >>> path = "/out_of_sample/2018.h5"
        >>> datasets.era5.open_34_vars(path)
        <xarray.DataArray 'fields' (time: 1460, channel: 34, lat: 721, lon: 1440)>
        dask.array<array, shape=(1460, 34, 721, 1440), dtype=float32, chunksize=(1, 1, 721, 1440), chunktype=numpy.ndarray>
        Coordinates:
        * time     (time) datetime64[ns] 2018-01-01 ... 2018-12-31T18:00:00
        * lat      (lat) float64 90.0 89.75 89.5 89.25 ... -89.25 -89.5 -89.75 -90.0
        * lon      (lon) float64 0.0 0.25 0.5 0.75 1.0 ... 359.0 359.2 359.5 359.8
        * channel  (channel) <U5 'u10' 'v10' 't2m' 'sp' ... 'v900' 'z900' 't900'
        Attributes:
            selene_path:  /lustre/fsw/sw_climate_fno/34Var
            description:  ERA5 data at 6 hourly frequency with snapshots at 0000, 060...
            path:         /out_of_sample/2018.h5
    """

    metadata = json.loads(METADATA.read_text())
    dims = metadata["dims"]
    h5_path = metadata["h5_path"]

    f = h5py.File(path)
    array = da.from_array(f[h5_path], chunks=chunks)
    ds = xarray.DataArray(array, name=h5_path, dims=dims)
    year = time.filename_to_year(path)
    n = array.shape[0]
    ds = ds.assign_coords(
        time=time.datetime_range(year, time_step=datetime.timedelta(hours=6), n=n),
        **metadata["coords"]
    )
    ds = ds.assign_attrs(metadata["attrs"], path=path)
    return ds
