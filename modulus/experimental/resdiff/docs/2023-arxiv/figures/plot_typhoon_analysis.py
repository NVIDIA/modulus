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

dx = 4000.0
dy = 4000.0

def Nvidia_cmap():
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["white", "darkgreen"])
    return cmap

true_rmw_list = np.array([60, 60,  70, 70,  50, 50])*1.852
true_vmax_list = np.array([105, 100, 100, 90, 80, 85])*0.514
true_lat_list = np.array([218,228, 238, 252, 262, 276])/10.0
true_lon_list = np.array([1218,1220,1223, 1223, 1226, 1230])/10.0
time_str = ['0911-12' ,'0911-18' ,'912-00' ,'0912-06' , '0912-12', '0912-18']

path = os.path.join(config.root, "paper/Chanthu2021.nc")
print(f"opening {path}")
diffusion = nc.Dataset(path, 'r')

lat  = np.array(diffusion.variables['lat'])
lon  = np.array(diffusion.variables['lon'])
f = 2*7.29*10**-5*np.sin(np.deg2rad(lat))
pred_diffusion_windspeed = load_windspeed(diffusion["prediction"])
truth_diffusion_windspeed = load_windspeed(diffusion["truth"])
input_diffusion_windspeed = load_windspeed(diffusion["input"])
I, J = truth_diffusion_windspeed[0,:,:].shape
i_index = np.arange(I)
j_index = np.arange(J)
dx = 2000.0
dy = 2000.0
x, y = np.meshgrid(i_index, j_index, indexing='ij')
x = np.multiply(x,dx)
y = np.multiply(y,dy)
pred_vorticity  = compute_curl(diffusion["prediction"])
truth_vorticity  = compute_curl(diffusion["truth"])
input_vorticity  = compute_curl(diffusion["input"])

datetime_values = [datetime.datetime(1990, 1, 1, 0, 0, 0) + datetime.timedelta(hours=t) for t in np.array(diffusion['time']).astype(float)]
datetime_values = np.array(datetime_values, dtype='datetime64[ns]')

fix_bin_edges = np.linspace(0.0, 60.0, 30)
combined_values_input = np.zeros(len(fix_bin_edges) - 1)
combined_values_pred = np.zeros(len(fix_bin_edges) - 1)

for tc_time in range(len(true_rmw_list)-1):

    true_rmw = true_rmw_list[tc_time]
    true_vmax = true_vmax_list[tc_time]
    true_lat = true_lat_list[tc_time]
    true_lon = true_lon_list[tc_time]

    i_c_input, j_c_input = find_storm_center_guess(input_vorticity[tc_time,:,:], input_diffusion_windspeed[tc_time,:,:], 10, lat, lon, true_lat, true_lon)
    i_c_pred, j_c_pred   = find_storm_center_guess(pred_vorticity[0,tc_time,:,:], pred_diffusion_windspeed[0,tc_time,:,:], 10, lat, lon, true_lat, true_lon)
    i_c_truth, j_c_truth = find_storm_center_guess(truth_vorticity[tc_time,:,:], truth_diffusion_windspeed[tc_time,:,:], 10, lat, lon, true_lat, true_lon)
    pred_lon = lon[i_c_pred, j_c_pred]
    pred_lat = lat[i_c_pred, j_c_pred]
    input_lon = lon[i_c_input, j_c_input]
    input_lat = lat[i_c_input, j_c_input]
    wrf_lat = lat[i_c_truth, j_c_truth]
    wrf_lon = lon[i_c_truth, j_c_truth]




    radii = np.linspace(0,200,201)*dx
    r = radii[:-1]
    shape = np.shape(pred_diffusion_windspeed)
    axi_v_pred = np.zeros((len(radii)-1, shape[0]))
    R_pred = np.zeros((len(radii)-1, shape[0]))
    for i in range(shape[0]):
        R_pred[:,i], axi_v_pred[:,i] = axis_symmetric_mean(
            x,
            y,
            np.squeeze(pred_diffusion_windspeed[i,tc_time,:,:]),
            i_c_pred,
            j_c_pred,
            radii)
    R_truth, axi_v_truth = axis_symmetric_mean(x, y, truth_diffusion_windspeed[tc_time,:,:], i_c_truth, j_c_truth, radii)
    R_input, axi_v_input = axis_symmetric_mean(x, y, input_diffusion_windspeed[tc_time,:,:], i_c_input, j_c_input, radii)
    mean_axi_v_pred = np.mean(axi_v_pred, axis=1)
    std_axi_v_pred = np.std(axi_v_pred, axis=1)
    rmw_input = R_input[np.argmax(axi_v_input)]
    rmw_pred = R_input[np.argmax(axi_v_pred, axis=0)]
    rmw_wrf = R_input[np.argmax(axi_v_truth, axis=0)]

    ens = 0
    L = 80
    pdf_values_input_tmp, _ = np.histogram(input_diffusion_windspeed[tc_time].flatten(), bins=fix_bin_edges)
    pdf_values_pred_tmp, _ = np.histogram(pred_diffusion_windspeed[ens, tc_time].flatten(), bins=fix_bin_edges)
    combined_values_input += pdf_values_input_tmp
    combined_values_pred += pdf_values_pred_tmp

    i_c_input_min = np.max([i_c_input - L, 0])
    j_c_input_min = np.max([j_c_input - L, 0])
    i_c_pred_min = np.max([i_c_pred - L, 0])
    j_c_pred_min = np.max([j_c_pred - L, 0])
    i_c_truth_min = np.max([i_c_truth - L, 0])
    j_c_truth_min = np.max([j_c_truth - L, 0])
    i_c_input_max = np.min([i_c_input + L, 447])
    j_c_input_max = np.min([j_c_input + L, 447])
    i_c_pred_max = np.min([i_c_pred + L, 447])
    j_c_pred_max = np.min([j_c_pred + L, 447])
    i_c_truth_max = np.min([i_c_truth + L, 447])
    j_c_truth_max = np.min([j_c_truth + L, 447])



    pdf_values_input, bin_edges_input = np.histogram(input_diffusion_windspeed[tc_time].flatten(), bins='auto', density=True)
    bin_centers_input = 0.5 * (bin_edges_input[1:] + bin_edges_input[:-1])
    pdf_values_pred, bin_edges_pred = np.histogram(pred_diffusion_windspeed[ens, tc_time].flatten(), bins='auto', density=True)
    bin_centers_pred = 0.5 * (bin_edges_pred[1:] + bin_edges_pred[:-1])
    pdf_values_truth, bin_edges_truth = np.histogram(truth_diffusion_windspeed[tc_time].flatten(), bins='auto', density=True)
    bin_centers_truth = 0.5 * (bin_edges_truth[1:] + bin_edges_truth[:-1])
    pdf_values_truth_tc, bin_edges_truth_tc = np.histogram(truth_diffusion_windspeed[tc_time, i_c_truth_min:i_c_truth_max, j_c_truth_min:j_c_truth_max].flatten(), bins='auto', density=True)
    bin_centers_truth_tc = 0.5 * (bin_edges_truth_tc[1:] + bin_edges_truth_tc[:-1])

    pdf_values_truth_tc, bin_edges_truth_tc = np.histogram(truth_diffusion_windspeed[tc_time, i_c_input_min:i_c_input_max, j_c_input_min:j_c_input_max].flatten(), bins='auto', density=True)
    bin_centers_truth_tc = 0.5 * (bin_edges_truth_tc[1:] + bin_edges_truth_tc[:-1])
    pdf_values_input_tc, bin_edges_input_tc = np.histogram(input_diffusion_windspeed[tc_time, i_c_input_min:i_c_input_max, j_c_input_min:j_c_input_max].flatten(), bins='auto', density=True)
    bin_centers_input_tc = 0.5 * (bin_edges_input_tc[1:] + bin_edges_input_tc[:-1])
    pdf_values_pred_tc, bin_edges_pred_tc = np.histogram(pred_diffusion_windspeed[ens, tc_time, i_c_input_min:i_c_input_max, j_c_input_min:j_c_input_max].flatten(), bins='auto', density=True)
    bin_centers_pred_tc = 0.5 * (bin_edges_pred_tc[1:] + bin_edges_pred_tc[:-1])

    abs_vmax_input = np.max(input_diffusion_windspeed[tc_time])
    abs_vmax_pred = np.max(pred_diffusion_windspeed[ens, tc_time])
    abs_vmax_wrf = np.max(truth_diffusion_windspeed[tc_time, : ,: ])

    vmin = np.min([input_diffusion_windspeed[tc_time], pred_diffusion_windspeed[ens, tc_time]])
    vmax = np.max([input_diffusion_windspeed[tc_time], pred_diffusion_windspeed[ens, tc_time]])
    sequential_cmap = cmap = Nvidia_cmap() #'viridis'


    plt.figure(figsize=(15, 10))

    ax = plt.subplot(231, projection=ccrs.PlateCarree())
    ax.annotate('(a)', xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
    ax.set_title('ERA5')
    im1 = ax.contourf(lon, lat, input_diffusion_windspeed[tc_time], cmap=sequential_cmap,  vmin=vmin, vmax=vmax)
    ax.plot(pred_lon, pred_lat, 'd', color = 'orange', markersize=10)
    ax.plot(wrf_lon, wrf_lat, 'o', color = 'k', markersize=6)
    ax.plot(input_lon, input_lat, '+', color = 'crimson', markersize=10)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitute')
    ax.set_xlim(lon[i_c_input_min, j_c_input_min], lon[i_c_input_max, j_c_input_max])
    ax.set_ylim(lat[i_c_input_min, j_c_input_min], lat[i_c_input_max, j_c_input_max])
    plt.colorbar(im1, ax=ax, shrink=0.62)
    im1 = ax.contour(lon, lat, input_diffusion_windspeed[tc_time], colors='grey',linewidths = 0.1, vmin=vmin, vmax=vmax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='black', alpha=0.0, draw_labels=True, linestyle="None")
    gl.right_labels=False
    gl.top_labels=False
    ax.coastlines(linewidth = 2, color = 'k')

    ax = plt.subplot(232, projection=ccrs.PlateCarree())
    ax.annotate('(b)', xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
    ax.set_title('ResDiff from ERA5')
    im2 = ax.contourf(lon, lat, pred_diffusion_windspeed[ens, tc_time], cmap=sequential_cmap, vmin=vmin, vmax=vmax)

    ax.plot(pred_lon, pred_lat, 'd', color = 'orange', markersize=10)
    ax.plot(wrf_lon, wrf_lat, 'o', color = 'k', markersize=6)
    ax.plot(input_lon, input_lat, '+', color = 'crimson', markersize=10)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitute')
    ax.set_xlim(lon[i_c_pred_min, j_c_pred_min], lon[i_c_pred_max, j_c_pred_max])
    ax.set_ylim(lat[i_c_pred_min, j_c_pred_min], lat[i_c_pred_max, j_c_pred_max])
    plt.colorbar(im2, ax=ax, shrink=0.62)
    im2 = ax.contour(lon, lat, pred_diffusion_windspeed[ens, tc_time], colors='grey',linewidths = 0.1, vmin=vmin, vmax=vmax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='black', alpha=0.0, draw_labels=True, linestyle="None")
    gl.right_labels=False
    gl.top_labels=False
    ax.coastlines(linewidth = 2, color = 'k')

    ax = plt.subplot(233, projection=ccrs.PlateCarree())
    ax.annotate('(c)', xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
    ax.set_title('WRF')
    im3 = ax.contourf(lon, lat, truth_diffusion_windspeed[tc_time], cmap=sequential_cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitute')
    ax.set_xlim(lon[i_c_truth - L, j_c_truth - L], lon[i_c_truth + L, j_c_truth + L])
    ax.set_ylim(lat[i_c_truth - L, j_c_truth - L], lat[i_c_truth + L, j_c_truth + L])
    ax.plot(pred_lon, pred_lat, 'd', color = 'orange', markersize=10)
    ax.plot(wrf_lon, wrf_lat, 'o', color = 'k', markersize=6)
    ax.plot(input_lon, input_lat, '+', color = 'crimson', markersize=10)

    plt.colorbar(im3, ax=ax, shrink=0.62)
    im3 = ax.contour(lon, lat, truth_diffusion_windspeed[tc_time], colors='grey',linewidths = 0.1, vmin=vmin, vmax=vmax)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), color='black', alpha=0.0, draw_labels=True, linestyle="None")
    gl.right_labels=False
    gl.top_labels=False
    ax.coastlines(linewidth = 2, color = 'k')

    ax = plt.subplot(234)
    ax.annotate('(d)', xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
    ax.plot(bin_centers_truth, pdf_values_truth,'k', linewidth = 1, label = 'WRF')
    ax.fill_between(bin_centers_truth, pdf_values_truth, alpha=0.3, color='grey')
    ax.plot(bin_centers_input, pdf_values_input, linewidth = 1, label = 'ERA5', color = 'crimson')
    ax.fill_between(bin_centers_input, pdf_values_input, alpha=0.3, color='crimson')
    ax.plot(bin_centers_pred, pdf_values_pred, linewidth = 1, label = 'ResDiff from ERA5', color='orange')
    ax.fill_between(bin_centers_pred, pdf_values_pred, alpha=0.3, color='orange')
    ax.set_xlabel('10m windspeed [m/s]')
    ax.set_ylabel('Probability Density')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    ax = plt.subplot(235)
    ax.annotate('(e)', xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
    ax.plot(bin_centers_input_tc, pdf_values_input_tc, linewidth = 1, label = 'ERA5', color = 'crimson')
    ax.fill_between(bin_centers_input_tc, pdf_values_input_tc, alpha=0.3, color='crimson')
    ax.plot(bin_centers_pred_tc, pdf_values_pred_tc, linewidth = 1, label = 'ResDiff from ERA5', color='orange')
    ax.fill_between(bin_centers_pred_tc, pdf_values_pred_tc, alpha=0.3, color='orange')
    ax.plot(bin_centers_truth_tc, pdf_values_truth_tc,'k', linewidth = 1, label = 'WRF')
    ax.fill_between(bin_centers_truth_tc, pdf_values_truth_tc, alpha=0.3, color='grey')

    ax.set_xlabel('10m windspeed [m/s]')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


    ax = plt.subplot(236)
    ax.annotate('(f)', xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
    ax.set_xlabel('radius [km]')
    ax.set_ylabel('10m windspeed [m/s]')
    im2 = ax.plot(r/1000.0,axi_v_truth, 'k',label = 'WRF')
    im2 = ax.plot(r/1000.0,mean_axi_v_pred, label = 'ResDiff from ERA5', linewidth = 1, color='orange')
    im2 = ax.fill_between(r/1000.0, mean_axi_v_pred - std_axi_v_pred, mean_axi_v_pred + std_axi_v_pred, color='orange', alpha=0.2)
    im2 = ax.plot(r/1000.0,axi_v_input, label = 'ERA5', color = 'crimson')
    ax.set_xlim([-2,200.0])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.legend()

    plt.tight_layout()
    plt.savefig(f"typhoon_Chanthu2021_{time_str[tc_time]}.pdf", dpi=100)

