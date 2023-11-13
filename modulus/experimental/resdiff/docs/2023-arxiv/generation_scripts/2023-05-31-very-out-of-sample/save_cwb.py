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
from fcn_mip.initial_conditions import get
from fcn_mip import schema
import datetime
import xarray as xr

import sys

OUTPUT = sys.argv[1]
print(OUTPUT)
 


times = [
    datetime.datetime(1985, 1, 1, 2),
    datetime.datetime(1995, 4, 20, 10),
    datetime.datetime(1990, 11, 15, 16),
    datetime.datetime(2005, 7, 3, 9),
    datetime.datetime(2008, 12, 25, 20),
    datetime.datetime(1992, 6, 17, 5),
    datetime.datetime(2001, 9, 9, 15),
    datetime.datetime(2012, 3, 10, 14),
    datetime.datetime(1988, 12, 31, 23),
    datetime.datetime(2009, 5, 2, 12),
    # typhoon maraket: https://www.greenpeace.org/eastasia/blog/6975/the-5-biggest-typhoons-to-batter-east-asia-in-recent-history/
    datetime.datetime(2009, 8, 7, 0),
    datetime.datetime(2009, 8, 8, 0),
]

times = sorted(times)



# from here: https://gitlab-master.nvidia.com/jpathak/parallel-data-preprocessing/-/blob/selene/cwb/run.py#L157
# ERA5_CHANNELS = [
#     # CWB as reflectivity instead of TCWV
#  {'variable': 'tcwv'},
#  {'pressure': 500, 'variable': 'geopotential_height'},
#  {'pressure': 500, 'variable': 'temperature'},
#  {'pressure': 500, 'variable': 'eastward_wind'},
#  {'pressure': 500, 'variable': 'northward_wind'},
#  {'pressure': 700, 'variable': 'geopotential_height'},
#  {'pressure': 700, 'variable': 'temperature'},
#  {'pressure': 700, 'variable': 'eastward_wind'},
#  {'pressure': 700, 'variable': 'northward_wind'},
#  {'pressure': 850, 'variable': 'geopotential_height'},
#  {'pressure': 850, 'variable': 'temperature'},
#  {'pressure': 850, 'variable': 'eastward_wind'},
#  {'pressure': 850, 'variable': 'northward_wind'},
#  {'pressure': 925, 'variable': 'geopotential_height'},
#  {'pressure': 925, 'variable': 'temperature'},
#  {'pressure': 925, 'variable': 'eastward_wind'},
#  {'pressure': 925, 'variable': 'northward_wind'},
#  {'variable': 'temperature_2m'},
#  {'variable': 'eastward_wind_10m'},
#  {'variable': 'northward_wind_10m'}
# ]

in_channels = [
 'tcwv', 
 'z500', 't500', 'u500', 'v500',
 'z700', 't700', 'u700', 'v700',
 'z850', 't850', 'u850', 'v850',
 'z925', 't925', 'u925', 'v925',
 't2m', 'u10m', 'v10m'
]



output = []
for time in times:
    data = get(0, time, schema.ChannelSet.var73, schema.InitialConditionSource.era5)
    data = data.sel(channel=in_channels)
    output.append(data)

output = xr.concat(output, dim='time')

# run from ngc
cwb_path = "s3://sw_climate_fno/nbrenowitz/2023-01-24-cwb-4years.zarr"
cwb = xr.open_zarr(cwb_path)
xlat = cwb["XLAT"]
xlong = cwb["XLONG"]
interpolated = output.interp(lat=xlat, lon=xlong)
interpolated.to_netcdf(OUTPUT)