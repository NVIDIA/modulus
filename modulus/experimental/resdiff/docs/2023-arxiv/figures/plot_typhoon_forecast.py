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
import netCDF4 as nc
import pylab as plt
import xarray
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tarfile
import datetime
from analysis_untils import *

Koinu2023_resdiff_path = '/lustre/fsw/sw_climate_fno/yacohen/diffusion/paper/Koinu2023_2023100412.nc'
gfs_folder = '/lustre/fsw/sw_climate_fno/yacohen/diffusion/paper/noiku2023_2023100412_gfs/'
Chanthu2021_path = '/lustre/fsw/sw_climate_fno/yacohen/diffusion/paper/Chanthu2021.nc'
Koinu2023_cwb_path = '/lustre/fsw/sw_climate_fno/yacohen/diffusion/paper/koinu_2023.tar'

dx = 4000.0
dy = 4000.0

date_format = '%Y%m%d%H%M%S'


def parse_channel(dms_key):
    variables = {
        "000": "geopotential_height",
        "100": "temperature",
        "200": "eastward_wind",
        "210": "northward_wind",
        "MOS": "maximum_radar_reflectivity",
        "LAT": "latitude",
        "LON": "longitude",
    }

    height_code = dms_key[:3]
    variable_code = dms_key[3:6]

    variable = variables[variable_code]

    if height_code == "B10":
        return {"variable": variable + "_10m"}
    elif height_code == "B02":
        return {"variable": variable + "_2m"}
    elif height_code == "H00":
        return {"variable": variable, "pressure": 1000}
    elif height_code == "X00":
        return {"variable": variable}
    else:
        return {
            "variable": variable, "pressure": int(height_code)
        }


def parse_file_name(name):
    time_str, channel_code = name.split("/")
    initial_time_str, step_str = time_str[:-2], time_str[-2:]
    initial_time = datetime.datetime.strptime(initial_time_str, date_format)
    channel = parse_channel(channel_code)
    return initial_time, channel, int(step_str)


def open_as_zarr(tar_name,
                dtype="<d",
                shape=(450, 450)):
    """Open a tarball containing CWB forecast data as an xarray dataset

    The files should be like this:
        YYYYMMDDHHMMSSZZ/{channel}

    where {channel} can be parsed by `parse_channel`.

    These files are assumed to be flat binary files containing arrays with ``dtype`` and ``shape``.

    Examples:

    ``tar_name`` needs be a .tar file (not gzipped). To get this be sure to first unzip the .tar.gz file with::

    gunzip koinu.tar.gz

    Then
    """
    import zarr
    import toolz
    data = []


    tf = tarfile.open(tar_name)
    k = 0
    for item in tf:
        if item.isfile():
            initial_time, channel, step = parse_file_name(item.name)
            variable_name = channel['variable']
            data.append((tar_name, channel, item.offset_data, item.size, initial_time, step))
            k+=1

    store = {}
    group = zarr.open_group(store)
    variables = toolz.groupby(lambda x: x[1]['variable'], data)
    times = sorted(set(x[-2] for x in data))
    steps = sorted(set(x[-1] for x in data))
    ds = xarray.Dataset(coords={"time": times, "step": steps})
    ds.to_zarr(store, mode='a')

    shape = list(shape)
    for v, arrs in variables.items():
        group.zeros(v, shape=[len(times), len(steps)]+shape, chunks=[1,1] + shape, dtype=dtype, compressor=None)
        group[v].attrs['_ARRAY_DIMENSIONS'] = ['time', 'step', 'x', 'y']
        for fname, _, offset, size, time , step in arrs:
            i = times.index(time)
            j = steps.index(step)
            store[f"{v}/{i}.{j}.0.0"] = [fname, offset, size]


    reference = {"version": 1, "gen": [], "refs": store}
    g =  zarr.open_group(
        "reference://",
        storage_options=dict(fo=reference)
    )

    zarr.consolidate_metadata(store)
    return xarray.open_zarr("reference://", storage_options=dict(fo=reference))

# get lat lon from older data and derive x,y
diffusion = nc.Dataset(Chanthu2021_path, 'r')
lat  = np.array(diffusion.variables['lat'])
lon  = np.array(diffusion.variables['lon'])
I, J = lon.shape
i_index = np.arange(I)
j_index = np.arange(J)
dx = 4000.0
dy = 4000.0
x, y = np.meshgrid(i_index, j_index, indexing='ij')
x = np.multiply(x,dx)
y = np.multiply(y,dy)


ds_resdiff = xarray.open_zarr(Koinu2023_resdiff_path+'/prediction')
resdiff_windspeed = load_windspeed(ds_resdiff)

ds_gfs = xarray.open_zarr(Koinu2023_resdiff_path+'/input')
gfs_windspeed = load_windspeed(ds_gfs)

ds_cwa = open_as_zarr(Koinu2023_cwb_path)
cwa_windspeed = load_windspeed(ds_cwa)
cwa_windspeed = cwa_windspeed[:,:,1:-1, 1:-1]


file_names = sorted(os.listdir(gfs_folder))  # Sorting ensures time-ordered sequence

lat_xr = xarray.DataArray(lat, dims=('y', 'x'))
lon_xr = xarray.DataArray(lon, dims=('y', 'x'))

datasets = []
times = []

for file_name in file_names:
    if file_name.endswith('.grib2'):
        time_str = file_name.split('.')[-2][1:]  # Extract the time part (after 'f')
        time_value = int(time_str)  # Convert the time string to integer
        times.append(time_value)

        grib_path = os.path.join(gfs_folder, file_name)
        ds = xarray.open_dataset(grib_path, engine='cfgrib',
            filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'atmosphere'})
        datasets.append(ds)

time_coords = xarray.DataArray(times, dims='time', name='time')
combined_ds = xarray.concat(datasets, dim=time_coords)
combined_ds = combined_ds.interp(latitude=lat_xr, longitude=lon_xr)

fix_bin_edges = np.linspace(0.0, 60.0, 30)
combined_values_input = np.zeros(len(fix_bin_edges) - 1)
combined_values_pred = np.zeros(len(fix_bin_edges) - 1)
plt.figure(figsize=(30, 20))
ens = 0
L = 80
ncolumns = 3
t = [0,3,6, 9, 12]
ntimes = len(t)
for tc_time in range(ntimes):
    ax = plt.subplot(ntimes,ncolumns, tc_time * ncolumns + 1, projection=ccrs.PlateCarree())
    if tc_time==0:
        ax.set_title('GFS')
    im1 = ax.contourf(combined_ds.longitude, combined_ds.latitude, combined_ds.refc.clip(min=0)[tc_time,:,:], cmap='magma')
    plt.colorbar(im1, ax=ax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='black', alpha=0.0, draw_labels=True, linestyle="None")
    ax.coastlines(linewidth = 2, color = 'white')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'fontsize': 20}
    gl.ylabel_style = {'fontsize': 20}
    if tc_time<ntimes-1:
        ax.set_xticks([])
        gl.bottom_labels = False
    else:
        ax.set_xlabel('longitude')
        gl.xlabel_style = {'fontsize': 20}

    
    ax = plt.subplot(ntimes,ncolumns, tc_time * ncolumns + 2, projection=ccrs.PlateCarree())
    if tc_time==0:
        ax.set_title('ResDiff from GFS')
    im1 = ax.contourf(lon, lat, ds_resdiff.maximum_radar_reflectivity.clip(min=0)[ens,t[tc_time],:,:], cmap='magma')
    plt.colorbar(im1, ax=ax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='black', alpha=0.0, draw_labels=True, linestyle="None")
    ax.coastlines(linewidth = 2, color  = 'white')
    ax.set_yticks([])
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.ylabel_style = {'fontsize': 20}
    if tc_time<ntimes-1:
        ax.set_xticks([])
        gl.bottom_labels = False
    else:
        ax.set_xlabel('longitude')
        gl.xlabel_style = {'fontsize': 20}

    
    ax = plt.subplot(ntimes,ncolumns, tc_time * ncolumns + 3, projection=ccrs.PlateCarree())
    if tc_time==0:
        ax.set_title('WRF')
    im1 = ax.contourf(lon, lat, ds_cwa.maximum_radar_reflectivity.clip(min=0)[29,t[tc_time],1:-1,1:-1], cmap='magma')
    plt.colorbar(im1, ax=ax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='black', alpha=0.0, draw_labels=True, linestyle="None")
    ax.coastlines(linewidth = 2, color  = 'white')
    ax.set_yticks([])
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    if tc_time<ntimes-1:
        ax.set_xticks([])
        gl.bottom_labels = False
    else:
        ax.set_xlabel('longitude')
        gl.xlabel_style = {'fontsize': 20}
    print("mean wrf ref", ds_cwa.maximum_radar_reflectivity.clip(min=0)[29,t[tc_time],1:-1,1:-1].mean().compute())
    
plt.tight_layout()
plt.savefig("./typhoon_koinu_radar.pdf")


cwa_vorticity  = compute_curl(ds_cwa)
cwa_vorticity = cwa_vorticity[:,:,1:-1, 1:-1]
gfs_vorticity  = compute_curl(ds_gfs)
resdiff_vorticity  = compute_curl(ds_resdiff)

plt.figure(figsize=(30, 20))
ens = 0
L = 80
ntimes  = 4
ncolumns = 8
t = [0,3,6,9,12]
for tc_time in range(ntimes):
    window_size = 10
    i_c_cwa, j_c_cwa = find_storm_center(cwa_vorticity[29, t[tc_time],:,:], cwa_windspeed[29, t[tc_time],:,:], window_size)
    i_c_gfs, j_c_gfs = find_storm_center(gfs_vorticity[t[tc_time],:,:], gfs_windspeed[t[tc_time],:,:], window_size)
    i_c_resdiff, j_c_resdiff = find_storm_center(resdiff_vorticity[ens,t[tc_time],:,:], resdiff_windspeed[ens,t[tc_time],:,:], window_size)

    radii = np.linspace(0,100,101)*dx
    r = radii[:-1]
    shape = np.shape(resdiff_vorticity)
    R_cwa, axi_v_cwa = axis_symmetric_mean(x, y, cwa_windspeed[29, t[tc_time],:,:], i_c_cwa, j_c_cwa, radii)
    shape = np.shape(cwa_windspeed)
    shape = np.shape(resdiff_windspeed)
    axi_v_resdiff = np.zeros((len(radii)-1, shape[0]))
    R_resdiff = np.zeros((len(radii)-1, shape[0]))
    for i in range(shape[0]):
        R_resdiff[:,i], axi_v_resdiff[:,i] = axis_symmetric_mean(
            x,
            y,
            np.squeeze(resdiff_windspeed[i,t[tc_time],:,:]),
            i_c_resdiff,
            j_c_resdiff,
            radii)
    R_gfs, axi_v_gfs = axis_symmetric_mean(x, y, gfs_windspeed[t[tc_time],:,:], i_c_gfs, j_c_gfs, radii)
    mean_axi_v_resdiff = np.mean(axi_v_resdiff, axis=1)
    std_axi_v_resdiff = np.std(axi_v_resdiff, axis=1)

    ax = plt.subplot(ntimes,ncolumns, tc_time * ncolumns + 8)
    ax.set_xlabel('radius [km]')
    ax.set_ylabel('10m windspeed [m/s]')
    im2 = ax.plot(r/1000.0,axi_v_cwa, 'k',label = 'WRF')
    im2 = ax.plot(r/1000.0,axi_v_gfs, label = 'GFS', color = 'crimson')
    im2 = ax.plot(r/1000.0,mean_axi_v_resdiff, linestyle = 'dashdot', label = 'ResDiff from GFS', linewidth = 1, color='orange')
    im2 = ax.fill_between(r/1000.0, mean_axi_v_resdiff - std_axi_v_resdiff, mean_axi_v_resdiff + std_axi_v_resdiff, color='orange', alpha=0.2)
    ax.set_xlim([0,400.0])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if tc_time==0:
        ax.legend()
plt.tight_layout()
plt.savefig("./typhoon_forecasts_axis.pdf")




