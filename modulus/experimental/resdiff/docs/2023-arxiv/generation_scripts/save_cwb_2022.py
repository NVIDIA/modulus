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

# %%
import os
from fcn_mip.initial_conditions import get
from fcn_mip import schema
from fcn_mip.time import convert_to_datetime
import numpy as np
import datetime
import xarray as xr
import matplotlib.pyplot as plt

import sys

import logging

logging.basicConfig(level=logging.DEBUG)

OUTPUT = "/lustre/fsw/nvresearch/nbrenowitz/diffusions/inputs/random-sample-2022.nc"


cwb =xr.open_zarr("/lustre/fsw/nvresearch/nbrenowitz/diffusions/targets/cwb.zarr")
subset = cwb.sel(time=cwb.time.dt.year == 2022)

plt.plot(subset.time, np.ones_like(subset.time), '.')
plt.xticks(rotation=45)


# use seed so this is the same always
r = np.random.default_rng(0)
times = r.choice(subset.time, replace=False, size=10)
# %%
in_channels = [
 'tcwv', 
 'z500', 't500', 'u500', 'v500',
 'z700', 't700', 'u700', 'v700',
 'z850', 't850', 'u850', 'v850',
 'z925', 't925', 'u925', 'v925',
 't2m', 'u10m', 'v10m'
]


download_path = "/lustre/fsw/nvresearch/nbrenowitz/diffusions/targets/era5"

output = []
for time in times:
    dt = convert_to_datetime(time)
    time_path = os.path.join(download_path, dt.isoformat() + '.nc')
    if not os.path.exists(time_path):
        print(f"Downloading {dt.isoformat()} from CDS")
        data = get(0, dt, schema.ChannelSet.var73, schema.InitialConditionSource.cds)
        data.rename("fields").to_netcdf(time_path)

    data = xr.open_dataset(time_path)
    data = data.sel(channel=in_channels)
    output.append(data)

output = xr.concat(output, dim='time')

# run from ngc
xlat = cwb["XLAT"]
xlong = cwb["XLONG"]
interpolated = output.interp(lat=xlat, lon=xlong)
interpolated['target'] = cwb.rename(channel='cwb_channel').fields.sel(time=output.time)
interpolated.attrs['history'] = ' '.join(sys.argv)
interpolated.to_netcdf(OUTPUT, mode="w")
