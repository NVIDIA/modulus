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

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# !ldconfig

# %%
# %pip install git+ssh://git@gitlab-master.nvidia.com:12051/earth-2/fcn-mip.git@main

# %%
from fcn_mip.initial_conditions.cds import get
from fcn_mip.schema import  ChannelSet
import os
from fcn_mip import schema, weather_events
from fcn_mip.initial_conditions import get
import datetime

time = datetime.datetime(2021, 9, 10, 0)

kw = dict(time=time, channel_set=schema.ChannelSet.var73, n_history=0)
ic= get(source=weather_events.InitialConditionSource.cds, **kw)
# need to hack around bug in cds source
# https://gitlab-master.nvidia.com/earth-2/fcn-mip/-/issues/56
cds = ic.roll(lon=720)
cds.rename("fields").to_netcdf("ic.nc")

# %%
# %%file config.json
{
    "simulation_length": 12,
    "weather_event": {
      "properties": {
        "name": "Globe",
        "netcdf": "ic.nc"
      },
      "domains": [
        {
          "name": "global",
          "type": "Window",
          "diagnostics": [
            {
              "type": "raw",
              "channels": [
                  "tcwv",
                  "z500",
                  "t500",
                  "u500",
                  "v500",
                  "z700",
                  "t700",
                  "u700",
                  "v700",
                  "z850",
                  "t850",
                  "u850",
                  "v850",
                  "z925",
                  "t925",
                  "u925",
                  "v925",
                  "t2m",
                  "u10m",
                  "v10m"
                ]

            }
          ]
        }
      ]
    },
    "output_path": "output",
    "output_frequency": 1,
    "fcn_model": "sfno_coszen"
}

# %% language="bash"
# export WORLD_SIZE=1
# export LOCAL_CACHE=/workspace/.cache
# rm -rf output/
# python3 -m fcn_mip.inference_ensemble config.json

# %%
import xarray

root = xarray.open_dataset("output/ensemble_out_0.nc")
output = xarray.open_dataset("output/ensemble_out_0.nc", group='global').merge(root)
output

# %%
channels = [
                  "tcwv",
                  "z500",
                  "t500",
                  "u500",
                  "v500",
                  "z700",
                  "t700",
                  "u700",
                  "v700",
                  "z850",
                  "t850",
                  "u850",
                  "v850",
                  "z925",
                  "t925",
                  "u925",
                  "v925",
                  "t2m",
                  "u10m",
                  "v10m"
                ]

cwb_path = "/lustre/fsw/sw_climate_fno/nbrenowitz/2023-01-24-cwb-4years.zarr"
cwb = xarray.open_zarr(cwb_path)
xlat = cwb["XLAT"]
xlong = cwb["XLONG"]
p = output.to_array(dim='channel', name='fields').sel(channel=channels).isel(ensemble=0)
interpolated = p.interp(lat=xlat, lon=xlong)
interpolated = interpolated.transpose("time", "channel", "south_north", "west_east")

dataset = interpolated.to_dataset()
dataset['target'] = cwb.cwb.sel(time=output.time)

dataset.to_netcdf("fcn_outputs.nc")

# %% language="bash"
#
# cd ../../../
# ./test.sh

# %%
# !apt-get update && apt-get install -y imagemagick

# %% language="bash"
#
# cd ../../../
#
# convert -delay 25 -loop 0 generations/netcdf/fcn/singlke/*.sample.png generations/netcdf/sample.gif

# %%
from IPython.display import display, HTML
import base64

def embed_gif(path):
    b64 = base64.b64encode(open(path,'rb').read()).decode('ascii')
    return HTML(f'<img src="data:image/gif;base64,{b64}" />')

display(embed_gif("../../../generations/netcdf/sample.gif"))
