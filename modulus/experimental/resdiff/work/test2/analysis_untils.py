import os
import numpy as np 
import xarray
import tarfile

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