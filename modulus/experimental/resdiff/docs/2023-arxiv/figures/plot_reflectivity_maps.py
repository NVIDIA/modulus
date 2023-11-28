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

time = ['2021-09-12T00', '2021-04-02T06', '2021-02-02T12']

path_22 = '/lustre/fsw/sw_climate_fno/yacohen/diffusion/paper/2022.nc'
path = '/lustre/fsw/sw_climate_fno/yacohen/diffusion/paper/2021.nc'

ds = xarray.open_dataset(path)
lat  = np.array(ds.variables['lat'])
lon  = np.array(ds.variables['lon'])
ds_prediction = xarray.open_dataset(path,group='prediction')
ds_truth = xarray.open_dataset(path,group='truth')
ds_input = xarray.open_dataset(path,group='input')
ds_prediction = ds_prediction.assign_coords(time=ds['time'], lat=ds['lat'], lon=ds['lon'])
ds_truth = ds_truth.assign_coords(time=ds['time'], lat=ds['lat'], lon=ds['lon'])
ds_input = ds_input.assign_coords(time=ds['time'], lat=ds['lat'], lon=ds['lon'])

ds_22 = xarray.open_dataset(path_22)
ds_22_prediction = xarray.open_dataset(path_22,group='prediction')
ds_22_truth = xarray.open_dataset(path_22,group='truth')
ds_22_input = xarray.open_dataset(path_22,group='input')
ds_22_prediction = ds_22_prediction.assign_coords(time=ds_22['time'], lat=ds['lat'], lon=ds['lon'])
ds_22_truth = ds_22_truth.assign_coords(time=ds_22['time'], lat=ds['lat'], lon=ds['lon'])
ds_22_input = ds_22_input.assign_coords(time=ds_22['time'], lat=ds['lat'], lon=ds['lon'])

plt.figure(figsize=(15, 10))

nvars  = len(time) + 1
ncolumns = 4
sequential_cmap = plt.get_cmap('magma', 20)
var = 'maximum_radar_reflectivity'
# for t in range(3):
t=2

labels1 = ['(a)','(e)','(i)','(m)']
labels2 = ['(b)','(f)','(j)','(n)']
labels3 = ['(c)','(g)','(k)','(o)']
labels4 = ['(d)','(h)','(l)','(p)']
for i in range(nvars-1):
    ax = plt.subplot(nvars,ncolumns, i * ncolumns + 1, projection=ccrs.PlateCarree())
    ax.annotate(labels1[i], xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='black', alpha=0.0, draw_labels=True, linestyle="None")
    im1 = ax.pcolormesh(lon, lat, ds_prediction[var].clip(min=0).mean(dim = 'ensemble')[i,:,:], cmap=sequential_cmap)
    plt.colorbar(im1, ax=ax, shrink=0.62)
    gl.right_labels=False
    gl.top_labels=False
    ax.coastlines(linewidth = 0.5, color = 'white')
    ax.set_ylabel('latitude')
    if i==nvars-1:
        ax.set_xlabel('longitude')
    else:
        gl.bottom_labels = False

    ax = plt.subplot(nvars,ncolumns, i * ncolumns + 2, projection=ccrs.PlateCarree())
    ax.annotate(labels2[i], xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='black', alpha=0.0, draw_labels=True, linestyle="None")
    if i==nvars-1:
        ax.set_xlabel('longitude')
    else:
        gl.bottom_labels = False
    im1 = ax.pcolormesh(lon, lat, np.sqrt(ds_prediction[var].clip(min=0).var(dim = 'ensemble')[i,:,:].clip(min=0)), cmap=sequential_cmap)
    plt.colorbar(im1, ax=ax, shrink=0.62)
    gl.right_labels=False
    gl.top_labels=False
    ax.coastlines(linewidth = 0.5, color = 'white')
    gl.left_labels = False

    ax = plt.subplot(nvars,ncolumns, i * ncolumns + 3, projection=ccrs.PlateCarree())    
    ax.annotate(labels3[i], xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='black', alpha=0.0, draw_labels=True, linestyle="None")
    if i==nvars-1:
        ax.set_xlabel('longitude')
    else:
        gl.bottom_labels = False
    im1 = ax.pcolormesh(lon, lat, ds_prediction[var][199,i,:,:].clip(min=0), cmap=sequential_cmap)
    plt.colorbar(im1, ax=ax, shrink=0.62)
    gl.right_labels=False
    gl.top_labels=False
    ax.coastlines(linewidth = 0.5, color = 'white')
    gl.left_labels = False


    ax = plt.subplot(nvars,ncolumns, i * ncolumns + 4, projection=ccrs.PlateCarree())
    ax.annotate(labels4[i], xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='black', alpha=0.0, draw_labels=True, linestyle="None")
    if i==nvars-1:
        ax.set_xlabel('longitude')
    else:
        gl.bottom_labels = False
    im1 = ax.pcolormesh(lon, lat, ds_truth[var][i,:,:].clip(min=0), cmap=sequential_cmap)
    plt.colorbar(im1, ax=ax, shrink=0.62)
    gl.right_labels=False
    gl.top_labels=False
    ax.coastlines(linewidth = 0.5, color = 'white')
    gl.left_labels = False

i = i + 1
ax = plt.subplot(nvars,ncolumns, i * ncolumns + 1, projection=ccrs.PlateCarree())
ax.annotate(labels1[i], xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='black', alpha=0.0, draw_labels=True, linestyle="None")
im1 = ax.pcolormesh(lon, lat, ds_22_prediction[var].clip(min=0).mean(dim = 'ensemble')[8,:,:], cmap=sequential_cmap)
plt.colorbar(im1, ax=ax, shrink=0.62)
gl.right_labels=False
gl.top_labels=False
ax.coastlines(linewidth = 0.5, color = 'white')
ax.set_ylabel('latitude')
if i==nvars-1:
    ax.set_xlabel('longitude')
else:
    gl.bottom_labels = False

ax = plt.subplot(nvars,ncolumns, i * ncolumns + 2, projection=ccrs.PlateCarree())
ax.annotate(labels2[i], xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='black', alpha=0.0, draw_labels=True, linestyle="None")
if i==nvars-1:
    ax.set_xlabel('longitude')
else:
    gl.bottom_labels = False
im1 = ax.pcolormesh(lon, lat, np.sqrt(ds_22_prediction[var].clip(min=0).var(dim = 'ensemble')[8,:,:].clip(min=0)), cmap=sequential_cmap) #, vmin=vmin, vmax=vmax)
plt.colorbar(im1, ax=ax, shrink=0.62)
gl.right_labels=False
gl.top_labels=False
ax.coastlines(linewidth = 0.5, color = 'white')
gl.left_labels = False

ax = plt.subplot(nvars,ncolumns, i * ncolumns + 3, projection=ccrs.PlateCarree())    
ax.annotate(labels3[i], xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='black', alpha=0.0, draw_labels=True, linestyle="None")
if i==nvars-1:
    ax.set_xlabel('longitude')
else:
    gl.bottom_labels = False
im1 = ax.pcolormesh(lon, lat, ds_22_prediction[var][99,8,:,:].clip(min=0), cmap=sequential_cmap)
plt.colorbar(im1, ax=ax, shrink=0.62)
gl.right_labels=False
gl.top_labels=False
ax.coastlines(linewidth = 0.5, color = 'white')
gl.left_labels = False


ax = plt.subplot(nvars,ncolumns, i * ncolumns + 4, projection=ccrs.PlateCarree())
ax.annotate(labels4[i], xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
gl = ax.gridlines(crs=ccrs.PlateCarree(), color='black', alpha=0.0, draw_labels=True, linestyle="None")
if i==nvars-1:
    ax.set_xlabel('longitude')
else:
    gl.bottom_labels = False
im1 = ax.pcolormesh(lon, lat, ds_22_truth[var][8,:,:].clip(min=0), cmap=sequential_cmap)
plt.colorbar(im1, ax=ax, shrink=0.62)
gl.right_labels=False
gl.top_labels=False
ax.coastlines(linewidth = 0.5, color = 'white')
gl.left_labels = False
plt.savefig("./reflectivity_maps.png")
plt.show()
