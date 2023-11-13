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

import numpy as np 
import math
import netCDF4 as nc
import pylab as plt
import datetime
import xarray
import matplotlib as mpl
import pickle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import TwoSlopeNorm
from scipy import interpolate
from analysis_untils import *
from matplotlib.gridspec import GridSpec

path_22 = '/lustre/fsw/sw_climate_fno/yacohen/diffusion/paper/2022.nc'

cmap = mpl.colors.ListedColormap(
        [
            "white",
            "lightskyblue",
            "deepskyblue",
            "dodgerblue",
            "goldenrod",
            "darkorange",
            "orangered",
            "seagreen",
            "teal",
            "crimson",
        ]
    )

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

cmap_name = 'custom_cmap'
colors = [(0, 'navy'), 
          (0.2, 'teal'), 
          (0.4, 'lightseagreen'), 
          (0.7, 'gold'), 
          (0.8, 'orange'), 
          (0.9, 'crimson'), 
          (1, 'maroon')]

customcmap = LinearSegmentedColormap.from_list(cmap_name, colors)

def find_closest_idx(array2D, value):
    distance = np.abs(array2D - value)
    return np.unravel_index(distance.argmin(), distance.shape)

def rotated_winds(lat, lon, ds_):
    temp = ds_['eastward_wind_10m'].values
    y1, x1 = find_closest_idx(lat, 23)
    y2, x2 = find_closest_idx(lat, lat.min())
    x2 = find_closest_idx(lon, 124.9)[1]
    num_points = 100
    y_values_main = np.clip(np.linspace(y1, y2, num_points).astype(int), 0, temp.shape[0] - 1)
    x_values_main = np.clip(np.linspace(x1, x2, num_points).astype(int), 0, temp.shape[1] - 1)
    u_wind = ds_['eastward_wind_10m'].values
    v_wind = ds_['northward_wind_10m'].values

    delta_y = y2 - y1
    delta_x = x2 - x1
    theta = np.arctan2(delta_y, delta_x)

    # Rotate the wind components for the whole domain
    u_wind = ds_['eastward_wind_10m'].values
    v_wind = ds_['northward_wind_10m'].values

    u_rotated = u_wind * np.cos(theta) - v_wind * np.sin(theta)
    v_rotated = u_wind * np.sin(theta) + v_wind * np.cos(theta)

    return u_rotated, v_rotated


def get_mean_cross_section(lat, lon, temp, shape):
    y1, x1 = find_closest_idx(lat, 23)
    y2, x2 = find_closest_idx(lat, lat.min())
    x2 = find_closest_idx(lon, 124.9)[1]
    num_points = 100
    y_values_main = np.clip(np.linspace(y1, y2, num_points).astype(int), 0, shape[0] - 1)
    x_values_main = np.clip(np.linspace(x1, x2, num_points).astype(int), 0, shape[1] - 1)
    parallel_temps = []
    offsets = np.arange(-10, 11)

    for offset in offsets:
        y_values_shifted = y_values_main + offset
        x_values_shifted = x_values_main + offset
        y_values_shifted = np.where((y_values_shifted >= 0) & (y_values_shifted < shape[0]), y_values_shifted, np.nan)
        x_values_shifted = np.where((x_values_shifted >= 0) & (x_values_shifted < shape[1]), x_values_shifted, np.nan)
        temp_values_shifted = [temp[int(y), int(x)] if not np.isnan(y) and not np.isnan(x) else np.nan for y, x in zip(y_values_shifted, x_values_shifted)]
        parallel_temps.append(temp_values_shifted)
    mean_temp_values = np.nanmean(np.array(parallel_temps), axis=0)
    distance = np.sqrt((x_values_main - x1)**2 + (y_values_main - y1)**2)
    return distance, mean_temp_values

def get_mean_cross_section2(lat, lon, ds_, var):
    temp = ds_[var].values
    y1, x1 = find_closest_idx(lat, 23)
    y2, x2 = find_closest_idx(lat, lat.min())
    x2 = find_closest_idx(lon, 124.9)[1]
    num_points = 100
    y_values_main = np.clip(np.linspace(y1, y2, num_points).astype(int), 0, temp.shape[0] - 1)
    x_values_main = np.clip(np.linspace(x1, x2, num_points).astype(int), 0, temp.shape[1] - 1)
    parallel_temps = []
    offsets = np.arange(-10, 11)

    for offset in offsets:
        y_values_shifted = y_values_main + offset
        x_values_shifted = x_values_main + offset
        y_values_shifted = np.where((y_values_shifted >= 0) & (y_values_shifted < temp.shape[0]), y_values_shifted, np.nan)
        x_values_shifted = np.where((x_values_shifted >= 0) & (x_values_shifted < temp.shape[1]), x_values_shifted, np.nan)
        temp_values_shifted = [temp[int(y), int(x)] if not np.isnan(y) and not np.isnan(x) else np.nan for y, x in zip(y_values_shifted, x_values_shifted)]
        parallel_temps.append(temp_values_shifted)
    mean_temp_values = np.nanmean(np.array(parallel_temps), axis=0)
    distance = np.sqrt((x_values_main - x1)**2 + (y_values_main - y1)**2)
    return distance, mean_temp_values


def get_cross_section(lat, lon, ds, var):
    temp = ds[var].values
    distance = find_closest_idx(array2D, value)
    y1, x1 = find_closest_idx(lat, 23)
    y2, x2 = find_closest_idx(lat, 124.9)
    x2 = find_closest_idx(lon, lon.max())[1]
    num_points = 100
    y_values = np.clip(np.linspace(y1, y2, num_points).astype(int), 0, temp.shape[0]-1)
    x_values = np.clip(np.linspace(x1, x2, num_points).astype(int), 0, temp.shape[1]-1)
    temp_values = [temp[y, x] for y, x in zip(y_values, x_values)]
    distance = np.sqrt((x_values - x1)**2 + (y_values - y1)**2)
    return distance, temp_values

ds_22 = xarray.open_dataset(path_22)
lat  = np.array(ds_22.variables['lat'])
lon  = np.array(ds_22.variables['lon'])


ds_22_prediction = xarray.open_dataset(path_22,group='prediction')
ds_22_truth = xarray.open_dataset(path_22,group='truth')
ds_22_input = xarray.open_dataset(path_22,group='input')
ds_22_prediction = ds_22_prediction.assign_coords(time=ds_22['time'], lat=ds_22['lat'], lon=ds_22['lon'])
ds_22_truth = ds_22_truth.assign_coords(time=ds_22['time'], lat=ds_22['lat'], lon=ds_22['lon'])
ds_22_input = ds_22_input.assign_coords(time=ds_22['time'], lat=ds_22['lat'], lon=ds_22['lon'])

u_rotated_truth, v_rotated_truth = rotated_winds(lat, lon, ds_22_truth.isel(time=8))
u_rotated_input, v_rotated_input = rotated_winds(lat, lon, ds_22_input.isel(time=8))
u_rotated_pred, v_rotated_pred = rotated_winds(lat, lon, ds_22_prediction.isel(ensemble=99, time=8))

skip = 40
nlevels = 20
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(3, 5, figure=fig, width_ratios=[1.1, 1.1, 1.1, 0.05, 1])

temp_input = ds_22_input.isel(time=8)['temperature_2m']
temp_pred = ds_22_prediction.isel(ensemble=50, time=8)['temperature_2m'] 
temp_truth = ds_22_truth.isel(time=8)['temperature_2m']
vmin = np.min([temp_input, temp_pred,temp_truth])
vmax = np.max([temp_input, temp_pred,temp_truth])
norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=295.0)
target_lon_start, target_lat_start = 122, 23
target_lon_end, target_lat_end = 124.9, lat.min()
# -----------------------------
i=0
u_wind = ds_22_input.isel(time=8)['eastward_wind_10m'].values
v_wind = ds_22_input.isel(time=8)['northward_wind_10m'].values

distance_perd, values_perd = get_mean_cross_section(lat, lon, temp_pred.values, [448,448])
distance_truth, values_truth = get_mean_cross_section(lat, lon, temp_truth.values, [448,448])
distance_input, values_input = get_mean_cross_section(lat, lon, temp_input.values, [448,448])

ax0 = fig.add_subplot(gs[i, 0])
ax0.contourf(lon, lat, temp_input, cmap=customcmap, norm=norm, levels=nlevels)
ax0.plot([target_lon_start, target_lon_end], [target_lat_start, target_lat_end], color='k', linestyle='--', linewidth=0.5)
ax0.set_ylabel('Latitude')
ax0.set_title('ERA5')
ax0.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u_wind[::skip, ::skip], v_wind[::skip, ::skip],
          scale=200, headlength=4, headwidth=4, width=0.002, color='black')

u_wind = ds_22_prediction.isel(ensemble=99, time=8)['eastward_wind_10m'].values
v_wind = ds_22_prediction.isel(ensemble=99, time=8)['northward_wind_10m'].values

ax1 = fig.add_subplot(gs[i, 1])
ax1.contourf(lon, lat, temp_pred, cmap=customcmap, norm=norm, levels=nlevels)
ax1.plot([target_lon_start, target_lon_end], [target_lat_start, target_lat_end], color='k', linestyle='--', linewidth=0.5)
ax1.set_title('ResDiff from ERA5')
ax1.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u_wind[::skip, ::skip], v_wind[::skip, ::skip],
          scale=200, headlength=4, headwidth=4, width=0.002, color='black')

u_wind = ds_22_truth.isel(time=8)['eastward_wind_10m'].values
v_wind = ds_22_truth.isel(time=8)['northward_wind_10m'].values

temp = ds_22_truth.isel(time=8)['temperature_2m']
ax2 = fig.add_subplot(gs[i, 2])
c = ax2.contourf(lon, lat, temp_truth, cmap=customcmap, norm=norm, levels=nlevels)
ax2.plot([target_lon_start, target_lon_end], [target_lat_start, target_lat_end], color='k', linestyle='--', linewidth=0.5)
ax2.set_title('WRF')
ax2.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u_wind[::skip, ::skip], v_wind[::skip, ::skip],
          scale=200, headlength=4, headwidth=4, width=0.002, color='black')
ax2.contour(lon, lat, temp_truth, colors = 'grey', norm=norm, linewidths = 0.1, levels=nlevels, alpha=0.5)
cbar_ax = fig.add_subplot(gs[i, 3])
plt.colorbar(c, cax=cbar_ax, fraction=0.5)
cbar_ax.set_position([0.7, 0.65, 0.007, 0.23])# [left, bottom, width, height]


ax3 = fig.add_subplot(gs[i, 4])
ax3.plot(distance_perd*2/111.1, values_perd,'orange', label = 'ResDiff')
ax3.plot(distance_truth*2/111.1, values_truth, 'k', label = 'WRF')
ax3.plot(distance_input*2/111.1, values_input, 'crimson',  label = 'ERA5')
ax3.set_title('NW-SE cross section')
ax3.set_ylabel('2m temperature [K]')
plt.legend()
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position("right")


i=1
distance_perd, values_perd = get_mean_cross_section(lat, lon, u_rotated_pred, [448,448])
distance_truth, values_truth = get_mean_cross_section(lat, lon, u_rotated_truth, [448,448])
distance_input, values_input = get_mean_cross_section(lat, lon, u_rotated_input, [448,448])

    
vmin = np.min([u_rotated_input, u_rotated_pred,u_rotated_truth])
vmax = np.max([u_rotated_input, u_rotated_pred,u_rotated_truth])
norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0.0)
ax4 = fig.add_subplot(gs[i, 0])
ax4.contourf(lon, lat, u_rotated_input, cmap=customcmap, norm=norm, levels=nlevels)
ax4.plot([target_lon_start, target_lon_end], [target_lat_start, target_lat_end], color='k', linestyle='--', linewidth=0.5)
ax4.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u_rotated_input[::skip, ::skip], u_rotated_input[::skip, ::skip],
          scale=200, headlength=4, headwidth=4, width=0.002, color='black')
ax4.set_ylabel('Latitude')

ax5 = fig.add_subplot(gs[i, 1])
ax5.contourf(lon, lat, u_rotated_pred, cmap=customcmap, norm=norm, levels=nlevels)
ax5.plot([target_lon_start, target_lon_end], [target_lat_start, target_lat_end], color='k', linestyle='--', linewidth=0.5)
# ax5.contour(lon, lat, u_rotated_pred, colors = 'grey', norm=norm, linewidths = 0.1, levels=nlevels, alpha=0.5)
ax5.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u_rotated_pred[::skip, ::skip], u_rotated_pred[::skip, ::skip],
          scale=200, headlength=4, headwidth=4, width=0.002, color='black')

temp = ds_22_truth.isel(time=8)['temperature_2m']
ax6 = fig.add_subplot(gs[i, 2])
c = ax6.contourf(lon, lat, u_rotated_truth, cmap=customcmap, norm=norm, levels=nlevels)
ax6.plot([target_lon_start, target_lon_end], [target_lat_start, target_lat_end], color='k', linestyle='--', linewidth=0.5)
ax6.quiver(lon[::skip, ::skip], lat[::skip, ::skip], u_rotated_truth[::skip, ::skip], u_rotated_truth[::skip, ::skip],
          scale=200, headlength=4, headwidth=4, width=0.002, color='black')
cbar_ax = fig.add_subplot(gs[i, 3])
plt.colorbar(c, cax=cbar_ax, fraction=0.1)
cbar_ax.set_position([0.7, 0.38, 0.007, 0.23])

ax7 = fig.add_subplot(gs[i, 4])
ax7.plot(distance_perd*2/111.1, values_perd,'orange', label = 'ResDiff')
ax7.plot(distance_truth*2/111.1, values_truth, 'k', label = 'WRF')
ax7.plot(distance_input*2/111.1, values_input, 'crimson',  label = 'ERA5')
ax7.set_ylabel('along front wind [m/s]')
ax7.yaxis.tick_right()
ax7.yaxis.set_label_position("right")

# ---------------------------
i=2
distance_perd, values_perd = get_mean_cross_section(lat, lon, v_rotated_pred, [448,448])
distance_truth, values_truth = get_mean_cross_section(lat, lon, v_rotated_truth, [448,448])
distance_input, values_input = get_mean_cross_section(lat, lon, v_rotated_input, [448,448])

vmin = np.min([v_rotated_input, v_rotated_pred,v_rotated_truth])
vmax = np.max([v_rotated_input, v_rotated_pred,v_rotated_truth])
norm = TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=0.0)
ax8 = fig.add_subplot(gs[i,0])
ax8.contourf(lon, lat, v_rotated_input, cmap=customcmap, norm=norm, levels=nlevels)
ax8.quiver(lon[::skip, ::skip], lat[::skip, ::skip], -v_rotated_input[::skip, ::skip], v_rotated_input[::skip, ::skip],
          scale=200, headlength=4, headwidth=4, width=0.002, color='black')
ax8.plot([target_lon_start, target_lon_end], [target_lat_start, target_lat_end], color='k', linestyle='--', linewidth=0.5)
ax8.set_xlabel('Longitude')
ax8.set_ylabel('Latitude')

ax9 = fig.add_subplot(gs[i,1])
ax9.contourf(lon, lat, v_rotated_pred, cmap=customcmap, norm=norm, levels=nlevels)
ax9.plot([target_lon_start, target_lon_end], [target_lat_start, target_lat_end], color='k', linestyle='--', linewidth=0.5)
ax9.quiver(lon[::skip, ::skip], lat[::skip, ::skip], -v_rotated_pred[::skip, ::skip], v_rotated_pred[::skip, ::skip],
          scale=200, headlength=4, headwidth=4, width=0.002, color='black')
ax9.set_xlabel('Longitude')

ax10 = fig.add_subplot(gs[i,2])
temp = ds_22_truth.isel(time=8)['temperature_2m']
c = ax10.contourf(lon, lat, v_rotated_truth, cmap=customcmap, norm=norm, levels=nlevels)
ax10.quiver(lon[::skip, ::skip], lat[::skip, ::skip], v_rotated_truth[::skip, ::skip], -v_rotated_truth[::skip, ::skip],
          scale=200, headlength=4, headwidth=4, width=0.002, color='black')
ax10.plot([target_lon_start, target_lon_end], [target_lat_start, target_lat_end], color='k', linestyle='--', linewidth=0.5)
ax10.set_xlabel('Longitude')
cbar_ax = fig.add_subplot(gs[i, 3])
plt.colorbar(c, cax=cbar_ax, fraction=0.2)
cbar_ax.set_position([0.7, 0.115, 0.007, 0.22])


ax11 = fig.add_subplot(gs[i,4])
ax11.plot(distance_perd*2/111.1, values_perd,'orange', label = 'ResDiff')
ax11.plot(distance_truth*2/111.1, values_truth, 'k', label = 'WRF')
ax11.plot(distance_input*2/111.1, values_input, 'crimson',  label = 'ERA5')
ax11.set_xlabel('distance from [122E, 23N]')
ax11.set_ylabel('cross front wind [m/s]')
ax11.yaxis.tick_right()
ax11.yaxis.set_label_position("right")
plt.savefig("./diffusion_fig_front.pdf")
plt.show()