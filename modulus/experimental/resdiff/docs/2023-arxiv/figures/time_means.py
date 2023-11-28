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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
import os
import config
# %%
url = os.path.join(config.root, "generations/era5-cwb-v3/validation_big/samples.zarr")

# %%
import xarray as xr
import os
import numpy as np
import cartopy.crs
import analysis_untils

# %%
coords =  xr.open_zarr(url)
pred = xr.open_zarr(url, group='prediction').merge(coords)
truth = xr.open_zarr(url, group='truth').merge(coords)
reg = analysis_untils.load_regression_data()
reg = reg.merge(coords)


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
# %%
pred_avg = pred.mean(["time"]).load()
truth_avg = truth.mean(["time"]).load()

# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
plt.style.use('dark_background')


def plot_difference(
    truth_avg, pred_avg, field, diff_kwargs=None, absolute_kwargs=None
):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection=ccrs.PlateCarree()))

    if diff_kwargs is None:
        diff_kwargs = dict(cmap='coolwarm', vmin=-1, vmax=1)
    
    if absolute_kwargs is None:
        absolute_kwargs = dict(cmap='magma', vmin=0, vmax=8)


    def plot(ax, data, title, vmin=0, vmax=8, cmap='magma'):
        im = ax.pcolormesh(
            truth.lon, truth.lat, data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin, vmax=vmax
        )
        ax.coastlines(color='white')
        ax.set_title(title)
        return im

    # Plotting the first two panels
    im1 = plot(ax1, pred_avg[field], 'Prediction', **absolute_kwargs)
    im2 = plot(ax2, truth_avg[field], 'Truth', **absolute_kwargs)

    # Plotting the difference in the third panel
    difference = (pred_avg[field] - truth_avg[field]).assign_coords(lat=truth_avg.lat, lon=truth_avg.lon)
    rms = np.sqrt((difference ** 2).mean()).item()
    im3 = plot(ax3, difference, f'Difference\nRMS: {rms:.2f}', **diff_kwargs)

    # Adding a shared colorbar for the first two panels
    fig.colorbar(im1, ax=[ax1, ax2], orientation='horizontal', label=field, shrink=0.5)

    # Adding a separate colorbar for the third panel
    fig.colorbar(im3, ax=[ax3], orientation='horizontal', label='Difference')

    plt.show()


# %%
os.makedirs("time_means", exist_ok=True)
plot_difference(truth_avg, pred_avg, "maximum_radar_reflectivity")
plt.savefig(os.path.join("time_means", "maximum_radar_reflectivity.png"))

# %%
plot_difference(truth_avg, pred_avg, "temperature_2m",
                absolute_kwargs=dict(cmap='viridis', vmin=273, vmax=305), 
                diff_kwargs=dict(cmap='coolwarm', vmin=-0.2, vmax=0.2))
plt.savefig(os.path.join("time_means", "temperature_2m.png"))
# %%
plot_difference(truth_avg, pred_avg, "eastward_wind_10m",
                absolute_kwargs=dict(cmap='coolwarm', vmin=-5, vmax=5), 
                diff_kwargs=dict(cmap='coolwarm', vmin=-0.2, vmax=0.2))
plt.savefig(os.path.join("time_means", "eastward_wind_10m.png"))
# %%
plot_difference(truth_avg, pred_avg, "northward_wind_10m",
                absolute_kwargs=dict(cmap='coolwarm', vmin=-5, vmax=5), 
                diff_kwargs=dict(cmap='coolwarm', vmin=-0.2, vmax=0.2))
plt.savefig(os.path.join("time_means", "northward_wind_10m.png"))