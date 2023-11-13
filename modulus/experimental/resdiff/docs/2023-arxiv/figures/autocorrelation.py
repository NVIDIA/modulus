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
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%

# %%
import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import fsspec
import sys
sys.path.insert(0, "../../../")
import power_spectra
from scipy.fft import fftshift, irfft
import os
import config
import analysis_untils

url = os.path.join(config.root, "generations/era5-cwb-v3/validation_big/samples.zarr")
output = os.path.splitext(__file__)[0]


# %%
coords =  xr.open_zarr(url)
pred = xr.open_zarr(url, group='prediction').merge(coords)
truth = xr.open_zarr(url, group='truth').merge(coords)

reg = analysis_untils.load_regression_data()

# ensemble was generated with the same noise for all ensemble=0 samples, which leads to odd results
# to get independent samples need to select different ensemble members for each time sample
# meaning we can only get 192 independent time samples.
i = xr.Variable(["time"], np.arange(192))
pred = pred.isel(time=slice(192)).isel(ensemble=i)
truth = truth.isel(time=slice(192))

# load
pred = pred.load()
truth = truth.load()

# %%
reg = reg.merge(coords)
# %%
pred_avg = pred.mean(["time"]).load()
truth_avg = truth.mean(["time"]).load()

# %%
field = 'maximum_radar_reflectivity'
bottom = 1e-2


def plot_spectra_and_acf(output, truth, reg, field, bottom):
    y, x  = xr.align(truth[field], reg[field].isel(ensemble=0), join="inner")

    f, pw_y = power_spectra.average_power_spectrum(y, d=2)
    _, pw_anom = power_spectra.average_power_spectrum(y-x, d=2)

    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)

    a.loglog(f, pw_y, label="$x$")
    a.loglog(f, pw_anom, label="$r(y) = x - E[x|y]$")
    a.set_ylim(bottom=bottom)
    a.set_title("a) Power Spectra", loc="left")
    a.set_xlabel("Frequency (1/km)")
    a.grid()
    a.legend()


    acf_y = power_spectra.power_spectra_to_acf(f, pw_y)
    acf_anom = power_spectra.power_spectra_to_acf(f, pw_anom)

    d = 2
    x = np.arange(len(acf_y))
    n = len(acf_y) // 2
    b.plot(x[:n], acf_y[:n], label="x")
    b.plot(x[:n], acf_anom[:n], label="(x-reg(y))")
    b.grid()
    b.set_xlabel("r (km)")
    b.set_title("b) Spatial autocorrelation", loc="left")
    fig.savefig(output, bbox_inches="tight")
    fig.suptitle(field)


os.makedirs(output, exist_ok=True)
for field in ["maximum_radar_reflectivity", "eastward_wind_10m", "northward_wind_10m", "temperature_2m"]:
    path = os.path.join(output, f"{field}.pdf")
    plot_spectra_and_acf(path, truth, reg, field, bottom)

