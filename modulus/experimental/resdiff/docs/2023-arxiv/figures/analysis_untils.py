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
import numpy as np 
import xarray
import fsspec
import tarfile
import config


def load_regression_data():
    url = os.path.join(config.root, "baselines/regression/era5-cwb-v3/validation_big/samples.nc")
    print("Loading regression data from: ", url)
    if config.root.startswith("s3://"):
        fs = fsspec.get_filesystem_class(url)
        if not os.path.exists("reg_samples.nc"):
            fs.get(url, "reg_samples.nc")
            reg = xarray.open_dataset("reg_samples.nc", group="prediction").load()
    else:
        reg = xarray.open_dataset(url, group="prediction").load()

    return reg


def load_windspeed(data):
    northward_wind_10m = data["northward_wind_10m"]
    eastward_wind_10m = data["eastward_wind_10m"]
    windspeed_10m = np.sqrt(np.multiply(northward_wind_10m,northward_wind_10m) + np.multiply(eastward_wind_10m,eastward_wind_10m))
    return windspeed_10m

def compute_curl(data):
    u = data["eastward_wind_10m"]
    v = data["northward_wind_10m"]
    du_dy, du_dx = np.gradient(u, axis=(-2, -1))
    dv_dy, dv_dx = np.gradient(v, axis=(-2, -1))
    curl = dv_dx/4000.0 - du_dy/4000.0
    return curl

def axis_symmetric_mean(x, y, data, i_center, j_center, radii):
    axis_sym_mean = np.zeros((len(radii) - 1, *data.shape[2:]))
    R = np.zeros((len(radii) - 1))
    distances = np.sqrt((x - x[i_center,j_center])**2 + (y - y[i_center,j_center])**2)
    for i in range(len(radii) - 1):
        mask = np.zeros_like(x) * np.nan
        mask[np.where((radii[i] < distances) & (distances <= radii[i+1]))] = 1.0
        axis_sym_mean[i] = np.nanmean(data * mask, axis=(0, 1))
        R[i] = np.nanmean(distances * mask, axis=(0, 1))
    return R, axis_sym_mean

def find_minimum_windspeed(windspeed, i_c, j_c, window_size):
    i_start, i_end = max(0, i_c - window_size), min(windspeed.shape[0], i_c + window_size + 1)
    j_start, j_end = max(0, j_c - window_size), min(windspeed.shape[1], j_c + window_size + 1)
    subarray = windspeed[i_start:i_end, j_start:j_end]
    if isinstance(subarray, xarray.DataArray):
        min_i, min_j = np.unravel_index(np.argmin(subarray.values), subarray.shape)
    else:
        min_i, min_j = np.unravel_index(np.argmin(subarray), subarray.shape)
    
    abs_min_i, abs_min_j = i_start + min_i, j_start + min_j
    return abs_min_i, abs_min_j

def find_maximum_vorticity(vorticity, i_c, j_c, window_size):
    i_start, i_end = max(0, i_c - window_size), min(vorticity.shape[0], i_c + window_size + 1)
    j_start, j_end = max(0, j_c - window_size), min(vorticity.shape[1], j_c + window_size + 1)
    subarray = vorticity[i_start:i_end, j_start:j_end]
    min_i, min_j = np.unravel_index(np.argmax(subarray), subarray.shape)
    abs_min_i, abs_min_j = i_start + min_i, j_start + min_j
    return abs_min_i, abs_min_j

def find_storm_center(vorticity, windspeed, window_size):
    i_c_guess,j_c_guess = np.unravel_index(np.argmax(vorticity, axis=None), vorticity.shape)
    i_c, j_c = find_minimum_windspeed(windspeed, i_c_guess,j_c_guess, window_size)
    return i_c, j_c


def find_storm_center_guess(vorticity, windspeed, window_size, lat, lon, true_lat, true_lon):
    distance = np.power(lat-true_lat,2.0)+np.power(lon-true_lon,2.0)
    i_c_guess, j_c_guess = np.unravel_index(np.argmin(distance, axis=None), distance.shape)
    i_c_guess, j_c_guess = find_maximum_vorticity(vorticity, i_c_guess, j_c_guess, window_size)
    i_c, j_c = find_minimum_windspeed(windspeed, i_c_guess,j_c_guess, window_size)
    return i_c, j_c

def add_windspeed(data, group):
    northward_wind_10m = data["northward_wind_10m"]
    eastward_wind_10m = data["eastward_wind_10m"]
    windspeed_10m = np.sqrt(np.multiply(northward_wind_10m, northward_wind_10m) + np.multiply(eastward_wind_10m, eastward_wind_10m))
    data["windspeed_10m"] = windspeed_10m
    return data