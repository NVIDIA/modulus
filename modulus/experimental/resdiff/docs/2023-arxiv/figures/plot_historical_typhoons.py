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
import pickle
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from analysis_untils import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

storm_record = '/lustre/fsw/sw_climate_fno/yacohen/diffusion/paper/taiwan_TC_storms.txt'
diffusion = nc.Dataset('/lustre/fsw/sw_climate_fno/yacohen/diffusion/paper/historical_typhoons_1980_2016.nc', 'r')

def storm_distance(windspeed, lon, lat, i_center, j_center):
    i_max, j_max = np.unravel_index(np.argmax(windspeed), windspeed.shape)
    lat_max = lat[i_max, j_max]
    lon_max = lon[i_max, j_max]
    lat_center = lat[i_center, j_center]
    lon_center = lon[i_center, j_center]
    delta_lat = lat_max - lat_center
    delta_lon = lon_max - lon_center
    distance = np.sqrt(delta_lat**2 + delta_lon**2)*111100.0
    return distance


def read_storm_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    storm_names = []
    true_lat = []
    true_lon = []
    dates_times = []
    surface_pressures = []
    true_vmax = []
    true_rmw = []
    
    for line in lines:
        parts = line.split()
        storm_names.append(parts[0])
        true_lat.append(float(parts[1]))
        true_lon.append(float(parts[2]))
        date_time_str = parts[3]
        year = int(date_time_str[0:4])
        month = int(date_time_str[4:6])
        day = int(date_time_str[6:8])
        hour = int(date_time_str[8:10])
        dates_times.append(datetime(year, month, day, hour, 0))
        surface_pressures.append(float(parts[4]))
        true_vmax.append(float(parts[5]))
        true_rmw.append(float(parts[6]))
        
    return np.array(storm_names), np.array(true_lat), np.array(true_lon), np.array(dates_times), np.array(surface_pressures), np.array(true_vmax), np.array(true_rmw)

lat  = np.array(diffusion.variables['lat'])
lon  = np.array(diffusion.variables['lon'])
f = 2*7.29*10**-5*np.sin(np.deg2rad(lat))
pred_diffusion_windspeed = load_windspeed(diffusion["prediction"])
input_diffusion_windspeed = load_windspeed(diffusion["input"])
I, J = input_diffusion_windspeed[0,:,:].shape
i_index = np.arange(I)
j_index = np.arange(J)
dx = 2000.0
dy = 2000.0
x, y = np.meshgrid(i_index, j_index, indexing='ij')
x = np.multiply(x,dx)
y = np.multiply(y,dy)
shape = np.shape(pred_diffusion_windspeed)
pred_vorticity  = compute_curl(diffusion["prediction"])
input_vorticity  = compute_curl(diffusion["input"])

storm_names, true_lat, true_lon, dates_times, surface_pressures, true_vmax, true_rmw = read_storm_data(storm_record)

rmw_input = np.zeros(shape[1])
rmw_pred = np.zeros(shape[:2])
vmax_input = np.zeros(shape[1])
vmax_pred = np.zeros(shape[:2])

abs_vmax_input = np.zeros(shape[1])
abs_vmax_pred = np.zeros(shape[:2])

abs_rmw_input = np.zeros(shape[1])
abs_rmw_pred = np.zeros(shape[:2])


fix_bin_edges = np.linspace(0.0, 60.0, 30)
combined_values_input = np.zeros(len(fix_bin_edges) - 1)
combined_values_pred = np.zeros(len(fix_bin_edges) - 1)

for tc_time in range(len(dates_times)):
    i_c_input, j_c_input = find_storm_center_guess(input_vorticity[tc_time,:,:], input_diffusion_windspeed[tc_time,:,:], 20, lat, lon, true_lat[tc_time], true_lon[tc_time])
    i_c_pred, j_c_pred   = find_storm_center_guess(input_vorticity[tc_time,:,:], pred_diffusion_windspeed[0,tc_time,:,:], 20, lat, lon, true_lat[tc_time], true_lon[tc_time])
    pred_lon = lon[i_c_pred, j_c_pred]
    pred_lat = lat[i_c_pred, j_c_pred]
    input_lon = lon[i_c_input, j_c_input]
    input_lat = lat[i_c_input, j_c_input]
    radii = np.linspace(0,150,151)*dx
    r = radii[:-1]
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
    R_input, axi_v_input = axis_symmetric_mean(x, y, input_diffusion_windspeed[tc_time,:,:], i_c_input, j_c_input, radii)
    rmw_input[tc_time] = R_input[np.argmax(axi_v_input)]
    rmw_pred[:,tc_time] = R_pred[np.argmax(axi_v_pred, axis=0)]
    vmax_input[tc_time] = np.max(axi_v_input)
    vmax_pred[:,tc_time] = np.max(axi_v_pred, axis=0)
    mean_axi_v_pred = np.mean(axi_v_pred, axis=1)
    std_axi_v_pred = np.std(axi_v_pred, axis=1)
    
    ens = 0
    L = 100
    i_c_input_min = np.max([i_c_input - L, 0])
    j_c_input_min = np.max([j_c_input - L, 0])
    i_c_pred_min = np.max([i_c_pred - L, 0])
    j_c_pred_min = np.max([j_c_pred - L, 0])
    i_c_input_max = np.min([i_c_input + L, 447])
    j_c_input_max = np.min([j_c_input + L, 447])
    i_c_pred_max = np.min([i_c_pred + L, 447])
    j_c_pred_max = np.min([j_c_pred + L, 447])
    
    pdf_values_input_tmp, _ = np.histogram(input_diffusion_windspeed[tc_time].flatten(), bins=fix_bin_edges)
    pdf_values_pred_tmp, _ = np.histogram(pred_diffusion_windspeed[ens, tc_time].flatten(), bins=fix_bin_edges)
    combined_values_input += pdf_values_input_tmp
    combined_values_pred += pdf_values_pred_tmp
    
    
    pdf_values_input, bin_edges_input = np.histogram(input_diffusion_windspeed[tc_time].flatten(), bins='auto', density=True)
    bin_centers_input = 0.5 * (bin_edges_input[1:] + bin_edges_input[:-1])
    pdf_values_pred, bin_edges_pred = np.histogram(pred_diffusion_windspeed[ens, tc_time].flatten(), bins='auto', density=True)
    bin_centers_pred = 0.5 * (bin_edges_pred[1:] + bin_edges_pred[:-1])
    pdf_values_input_tc, bin_edges_input_tc = np.histogram(input_diffusion_windspeed[tc_time, i_c_input_min: i_c_input_max,j_c_input_min: j_c_input_max].flatten(), bins='auto', density=True)
    bin_centers_input_tc = 0.5 * (bin_edges_input_tc[1:] + bin_edges_input_tc[:-1])
    pdf_values_pred_tc, bin_edges_pred_tc = np.histogram(pred_diffusion_windspeed[ens, tc_time, i_c_input_min: i_c_input_max,j_c_input_min: j_c_input_max].flatten(), bins='auto', density=True)
    bin_centers_pred_tc = 0.5 * (bin_edges_pred_tc[1:] + bin_edges_pred_tc[:-1])
    
    
    abs_vmax_input[tc_time] = np.max(input_diffusion_windspeed[tc_time, i_c_input_min: i_c_input_max,j_c_input_min: j_c_input_max])
    abs_vmax_pred[:,tc_time] = np.max(pred_diffusion_windspeed[ens, tc_time, i_c_input_min: i_c_input_max,j_c_input_min: j_c_input_max])
    
    
    abs_rmw_input[tc_time] = storm_distance(input_diffusion_windspeed[tc_time], lat, lon, i_c_input, j_c_input)
    abs_rmw_pred[ens,tc_time] = storm_distance(pred_diffusion_windspeed[ens, tc_time],  lat, lon, i_c_pred, j_c_pred)

    vmin = np.min([input_diffusion_windspeed[tc_time], pred_diffusion_windspeed[ens, tc_time]])
    vmax = np.max([input_diffusion_windspeed[tc_time], pred_diffusion_windspeed[ens, tc_time]])

combined_bin_centers = 0.5 * (fix_bin_edges[1:] + fix_bin_edges[:-1])

total_input = combined_values_input.sum()
total_pred = combined_values_pred.sum()
if total_input > 0:
    combined_values_input /= total_input
if total_pred > 0:
    combined_values_pred /= total_pred
    

mean_vmax_input = []
mean_vmax_pred = []
mean_vmax_obs = []

std_vmax_input = []
std_vmax_pred = []
std_vmax_obs = []
unique_true_vmax = np.unique(true_vmax)

# Compute mean of x and y for each unique value in z
for val in unique_true_vmax:
    indices = np.where(true_vmax == val)
    mean_vmax_input.append(np.nanmean(abs_vmax_input[indices]))
    mean_vmax_pred.append(np.nanmean(abs_vmax_pred[0,indices]))
    mean_vmax_obs.append(np.nanmean(true_vmax[indices]))
    std_vmax_input.append(np.nanstd(abs_vmax_input[indices]))
    std_vmax_pred.append(np.nanstd(abs_vmax_pred[0,indices]))
    std_vmax_obs.append(np.nanstd(true_vmax[indices]))


mean_rmw_input = []
mean_rmw_pred = []
mean_rmw_obs = []
std_rmw_input = []
std_rmw_pred = []
std_rmw_obs = []
unique_true_rmw = np.unique(true_rmw)

# Compute mean of x and y for each unique value in z
for val in unique_true_rmw:
    indices = np.where(true_rmw == val)
    mean_rmw_input.append(np.nanmean(abs_rmw_input[indices]))
    mean_rmw_pred.append(np.nanmean(abs_rmw_pred[0,indices]))
    mean_rmw_obs.append(np.nanmean(true_rmw[indices]))
    std_rmw_input.append(np.nanstd(abs_rmw_input[indices]))
    std_rmw_pred.append(np.nanstd(abs_rmw_pred[0,indices]))
    std_rmw_obs.append(np.nanstd(true_rmw[indices]))
    
# side figure
fig = plt.figure(figsize=(6, 18))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

ax1 = plt.subplot(gs[0])
ax1.annotate('(g)', xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
ax1.errorbar(np.array(mean_rmw_obs), np.array(mean_rmw_input)/1000.0,yerr=np.array(std_rmw_input)/1000.0, color='crimson', fmt='o',label='ERA5', alpha=0.5)
ax1.errorbar(np.array(mean_rmw_obs), np.array(mean_rmw_pred)/1000.0, yerr=np.array(std_rmw_pred)/1000.0, color='orange', fmt='o', label='ResDiff', alpha=0.5)
ax1.plot([0,350],[0,350],'--',color = 'k', linewidth = 0.5)
ax1.set_xlabel('observed radius of maximum winds [km]')
ax1.set_ylabel('predicted radius of maximum winds [km]')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.legend()

# Third subplot
ax2 = plt.subplot(gs[1])
ax2.annotate('(h)', xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
ax2.errorbar(np.array(mean_vmax_obs), np.array(mean_vmax_input),yerr=np.array(std_vmax_input), color='crimson', fmt='o',label='ERA5', alpha=0.5)
ax2.errorbar(np.array(mean_vmax_obs), np.array(mean_vmax_pred),yerr=np.array(std_vmax_pred), color='orange', fmt='o',label='ResDiff', alpha=0.5)
ax2.plot([0, 100], [0, 100], '--', color='k', linewidth=0.5)
ax2.set_xlim([30, 62])
ax2.set_ylim([0, 62])
ax2.set_xlabel('observed maximum windspeed [m/s]')
ax2.set_ylabel('predicted maximum windspeed [m/s]')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.legend()

ax0 = plt.subplot(gs[2])
ax0.annotate('(i)', xy=(-0.12, 1.05), xycoords='axes fraction', fontsize=12)
ax0.plot(combined_bin_centers, combined_values_input, linewidth=1, label='ERA5', color='crimson')
ax0.fill_between(combined_bin_centers, combined_values_input, alpha=0.3, color='crimson')
ax0.plot(combined_bin_centers, combined_values_pred, linewidth=1, label='ResDiff from ERA5', color='orange')
ax0.fill_between(combined_bin_centers, combined_values_pred, alpha=0.3, color='orange')
ax0.set_xlabel('10m windspeed [m/s]')
ax0.set_ylabel('Probability Density')
ax0.set_xlim([combined_bin_centers[0],60.0])
ax0.set_ylim([0,0.15])
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.legend()

plt.savefig("./typhoon_statistics.pdf")
