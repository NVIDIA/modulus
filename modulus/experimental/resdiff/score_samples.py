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

"""Score the generated samples


Saves a netCDF of crps and other scores. Depends on time, but space and ensemble have been reduced::

    netcdf scores {
dimensions:
        metric = 4 ;
        time = 205 ;
variables:
        double eastward_wind_10m(metric, time) ;
                eastward_wind_10m:_FillValue = NaN ;
        double maximum_radar_reflectivity(metric, time) ;
                maximum_radar_reflectivity:_FillValue = NaN ;
        double northward_wind_10m(metric, time) ;
                northward_wind_10m:_FillValue = NaN ;
        double temperature_2m(metric, time) ;
                temperature_2m:_FillValue = NaN ;
        int64 time(time) ;
                time:units = "hours since 1990-01-01" ;
                time:calendar = "standard" ;
        string metric(metric) ;
}


"""
# %%
import xskillscore

# TODO install skillscore
import xarray as xr
import sys
import pandas as pd
import dask.diagnostics


def open_samples(f):
    root = xr.open_dataset(f)
    pred = xr.open_dataset(f, group="prediction")
    truth = xr.open_dataset(f, group="truth")

    pred = pred.merge(root)
    truth = truth.merge(root)

    truth = truth.set_coords(["lon", "lat"])
    pred = pred.set_coords(["lon", "lat"])
    return truth, pred, root


path = sys.argv[1]

truth, pred, root = open_samples(path)
pred = pred.chunk(time=1)
truth = truth.chunk(time=1)

dim =["x", "y"]

a = xskillscore.rmse(truth, pred.mean("ensemble"), dim=dim)
b = xskillscore.crps_ensemble(truth, pred, member_dim="ensemble", dim=dim)
c = pred.std('ensemble').mean(dim)
crps_mean = xskillscore.crps_ensemble(truth, pred.mean("ensemble").expand_dims("ensemble"), member_dim="ensemble", dim=dim)

metrics = xr.concat([a, b, c, crps_mean], dim="metric").assign_coords(metric=["rmse", "crps", "std_dev", "mae"])
with dask.diagnostics.ProgressBar():
    metrics.to_netcdf(sys.argv[2], mode='w')

