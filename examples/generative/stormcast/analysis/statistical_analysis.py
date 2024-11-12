import os
import sys 
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.getcwd())

import argparse
import zarr
import numpy as np
import xarray
from scipy.signal import periodogram
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import glob
from dask.array import concatenate
from dask.array import stack
import xarray as xr
import re
import dask.array as da
import time 
import pandas as pd
from utils.metrics import *


def store_metric(pred, reg, base_path, metric_name):
    paranet_dir = os.path.dirname(base_path)
    model_name = os.path.basename(os.path.normpath(base_path))
    vars_with_levels = [var for var in pred.data_vars if 'levels' in pred[var].dims]
    vars_without_levels = [var for var in pred.data_vars if 'levels' not in pred[var].dims]
    num_levels = len(pred.levels) if 'levels' in pred.dims else 0

    data_for_csv = []

    def save_data(data, var, level=None, label=None, data_list=[]):
        if 'cutoff' in data.dims:
            cutoff_values = data.cutoff.values
            for cutoff_idx in range(len(data.cutoff)):
                # Collect data for CSV
                if level is not None:
                    data_to_save = data[var].isel(levels=level, cutoff=cutoff_idx).mean(dim='ic')
                else:
                    data_to_save = data[var].isel(cutoff=cutoff_idx).mean(dim='ic')
                
                for time, value in zip(data.time.values, data_to_save.values):
                    data_list.append({
                        'Data': metric_name,
                        'Variable': var,
                        'Type': label,
                        'Model': model_name,
                        'Level': level if level is not None else '',
                        'Time': time,
                        'Value': value,
                        'Cutoff': cutoff_values[cutoff_idx]
                    })
        else:
            if level is not None:
                data_to_save = data[var].isel(levels=level).mean(dim='ic')
            else:
                data_to_save = data[var].mean(dim='ic')
            
            for time, value in zip(data.time.values, data_to_save.values):
                data_list.append({
                    'Data': metric_name,
                    'Variable': var,
                    'Type': label,
                    'Model': model_name,
                    'Level': level if level is not None else '',
                    'Time': time,
                    'Value': value
                })
            
        return data_list
    
    for idx, var in enumerate(vars_without_levels):
        data_for_csv = save_data(reg, var, label='regression')
        data_for_csv = save_data(pred, var, label='diffusion')

    for idx, var in enumerate(vars_with_levels):
        for level in range(num_levels):
            data_for_csv = save_data(reg, var, level=level, label='regression')
            data_for_csv = save_data(pred, var, level=level, label='diffusion')

    csv_data = pd.DataFrame(data_for_csv)
    csv_path = os.path.join(paranet_dir, f'{metric_name}_{model_name}.csv')
    csv_data.to_csv(csv_path, index=False)

    
def power_spectrum(data, d):
    _, Pxx = periodogram(data, fs=1/d, axis=-1)
    return np.mean(Pxx, axis=-2) 


def lazy_power_spectrum(da, d):
    return xr.apply_ufunc(
        power_spectrum,
        da,
        d,
        input_core_dims=[['y', 'x'], []],
        output_core_dims=[['frequency']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float]
    )


def compute_ke(ds):
    return 0.5 * (ds['u_comb']**2 + ds['v_comb']**2) #+ ds['w_comb']**2)


def compute_ws(ds):
    return np.sqrt(ds['u_comb']**2 + ds['v_comb']**2)


def compute_surf_ke(ds):
    return 0.5 * (ds['u10m']**2 + ds['v10m']**2)


def compute_surf_ws(ds):
    return np.sqrt(ds['u10m']**2 + ds['v10m']**2)


def load_dist(var, bins):
    if isinstance(var, xarray.DataArray):
        flattened_var = var.values.flatten()
    else:
        flattened_var = np.array(var).flatten()
    pdf_values, bin_edges = np.histogram(np.abs(flattened_var), bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return bin_centers, pdf_values


def concatenate_netcdfs(pred_paths, reg_paths, targ_paths, viz_variable_names, viz_levels):
    prediction_datasets = []
    regression_datasets = []
    target_datasets = []
    ics = []

    first_pred_ds = xr.open_dataset(pred_paths[0])
    latitude = first_pred_ds['latitude']
    longitude = first_pred_ds['longitude']

    for i, (pred_path, reg_path, targ_path) in enumerate(zip(pred_paths, reg_paths, targ_paths)):
        pred_ds = xr.open_dataset(pred_path).sel(levels=viz_levels)
        reg_ds = xr.open_dataset(reg_path).sel(levels=viz_levels)
        targ_ds = xr.open_dataset(targ_path).sel(levels=viz_levels)

        pred_ds = pred_ds.drop_vars(set(pred_ds.variables) - set(viz_variable_names))
        reg_ds = reg_ds.drop_vars(set(reg_ds.variables) - set(viz_variable_names))
        targ_ds = targ_ds.drop_vars(set(targ_ds.variables) - set(viz_variable_names))
            
        prediction_datasets.append(pred_ds)
        regression_datasets.append(reg_ds)
        target_datasets.append(targ_ds)
        ics.append(i)

    combined_predictions = xr.concat(prediction_datasets, pd.Index(ics, name='ic'))
    combined_regressions = xr.concat(regression_datasets, pd.Index(ics, name='ic'))
    combined_targets = xr.concat(target_datasets, pd.Index(ics, name='ic'))
    
    combined_predictions['latitude'] = (['y', 'x'], latitude.values)
    combined_predictions['longitude'] = (['y', 'x'], longitude.values)
    combined_regressions['latitude'] = (['y', 'x'], latitude.values)
    combined_regressions['longitude'] = (['y', 'x'], longitude.values)
    combined_targets['latitude'] = (['y', 'x'], latitude.values)
    combined_targets['longitude'] = (['y', 'x'], longitude.values)
    return combined_predictions, combined_regressions, combined_targets


def find_nc_files(base_path, pattern):
    nc_paths = []
    for root, dirs, files in os.walk(base_path):
        nc_paths.extend(glob.glob(os.path.join(root, pattern)))
    return nc_paths


def save_spectra_dist(predictions, regressions, targets, viz_variable_names, viz_surf_variable_names, viz_levels, base_path):
    spectra_data = []
    distribution_data = []
    paranet_dir = os.path.dirname(base_path)
    model_name = os.path.basename(os.path.normpath(base_path))

    d = 3
    bins = 20
    
    for i, var_name in enumerate(viz_surf_variable_names):
        if var_name == 'Ek10m':
            target = compute_surf_ke(targets)
            prediction = compute_surf_ke(predictions)
            regression = compute_surf_ke(regressions)
        elif var_name == 'u10m' or var_name == 'v10m':
            target = np.abs(targets[var_name])
            prediction = np.abs(predictions[var_name])
            regression = np.abs(regressions[var_name])
        else:
            target = targets[var_name]
            prediction = predictions[var_name]
            regression = regressions[var_name]

        frequencies, _ = periodogram(target, fs=1/d)
        ps_spectra_pd = lazy_power_spectrum(prediction, d).compute()
        ps_spectra_rg = lazy_power_spectrum(regression, d).compute()
        ps_spectra_tr = lazy_power_spectrum(target, d).compute()
        
        spectra_tr_leadtime = np.mean(ps_spectra_tr, axis = (0))
        spectra_pd_leadtime = np.mean(ps_spectra_pd, axis = (0))
        spectra_rg_leadtime = np.mean(ps_spectra_rg, axis = (0))
            
        times = np.arange(spectra_pd_leadtime.shape[0])
        for freq_idx, freq in enumerate(frequencies):
            for time in times:
                pred_value = spectra_pd_leadtime[time, freq_idx].values.item()
                spectra_data.append({
                    'Data': 'spectra',
                    'Variable': var_name,
                    'Type': 'diffusion',
                    'Model': model_name,
                    'Level': '',
                    'Time': time,
                    'Frequency': freq,
                    'Value': pred_value
                })
                reg_value = spectra_rg_leadtime[time, freq_idx].values.item()
                spectra_data.append({
                    'Data': 'spectra',
                    'Variable': var_name,
                    'Type': 'regression',
                    'Model': model_name,
                    'Level': '',
                    'Time': time,
                    'Frequency': freq,
                    'Value': reg_value
                })
                reg_value = spectra_tr_leadtime[time, freq_idx].values.item()
                spectra_data.append({
                    'Data': 'spectra',
                    'Variable': var_name,
                    'Type': 'target',
                    'Model': model_name,
                    'Level': '',
                    'Time': time,
                    'Frequency': freq,
                    'Value': reg_value
                })

        if 'Ek10m' == var_name:
            target = compute_surf_ws(targets)
            prediction = compute_surf_ws(predictions)
            regression = compute_surf_ws(regression)
        
        for time in times:
            bins_pd, dist_pd = load_dist(prediction[:,time,:,:], bins)
            bins_rg, dist_rg = load_dist(regression[:,time,:,:], bins)
            bins_tr, dist_tr = load_dist(target[:,time,:,:], bins)
            
            #  Save the current seterr so that I can aviod warnings from np.log of zero devision
            old_settings = np.seterr(divide='ignore', invalid='ignore')
            log_dist_pd = np.log(dist_pd)
            log_dist_rg = np.log(dist_rg)
            log_dist_tr = np.log(dist_tr)
            np.seterr(**old_settings)

            for bin_center, pdf_value in zip(bins_pd, log_dist_pd):
                distribution_data.append({
                    'Data': 'distribution',
                    'Variable': var_name,
                    'Type': 'diffusion',
                    'Model': model_name,
                    'Level': '',
                    'Time': time,
                    'Bin': bin_center,
                    'Value': pdf_value
                })
            for bin_center, pdf_value in zip(bins_rg, log_dist_rg):
                distribution_data.append({
                    'Data': 'distribution',
                    'Variable': var_name,
                    'Type': 'regression',
                    'Model': model_name,
                    'Level': '',
                    'Time': time,
                    'Bin': bin_center,
                    'Value': pdf_value
                })
            for bin_center, pdf_value in zip(bins_tr, log_dist_tr):
                distribution_data.append({
                    'Data': 'distribution',
                    'Variable': var_name,
                    'Type': 'target',
                    'Model': model_name,
                    'Level': '',
                    'Time': time,
                    'Bin': bin_center,
                    'Value': pdf_value
                })

    for var_name in viz_variable_names:
        for i, level in enumerate(viz_levels):
            if var_name == 'Ek':
                target = compute_ke(targets.isel(levels = i))
                prediction = compute_ke(predictions.isel(levels = i))
                regression = compute_ke(regressions.isel(levels = i))
            elif var_name == 'w' or var_name == 'u' or var_name == 'v':
                # taking the abs of velcoity componets 
                target = np.abs(targets[var_name].isel(levels = i))
                prediction = np.abs(predictions[var_name].isel(levels = i))
                regression = np.abs(regressions[var_name].isel(levels = i))
            else:
                target = targets[var_name].isel(levels = i)
                prediction = predictions[var_name].isel(levels = i)
                regression = regressions[var_name].isel(levels = i)
            frequencies, _ = periodogram(target, fs=1/d)
            ps_spectra_pd = lazy_power_spectrum(prediction, d).compute()
            ps_spectra_rg = lazy_power_spectrum(regression, d).compute()
            ps_spectra_tr = lazy_power_spectrum(target, d).compute()
            spectra_tr_leadtime = np.mean(ps_spectra_tr, axis = (0))
            spectra_pd_leadtime = np.mean(ps_spectra_pd, axis = (0))
            spectra_rg_leadtime = np.mean(ps_spectra_rg, axis = (0))            
            
            for freq_idx, freq in enumerate(frequencies):
                for time in times:
                    pred_value = spectra_pd_leadtime[time, freq_idx].values.item()
                    spectra_data.append({
                        'Data': 'spectra',
                        'Variable': var_name,
                        'Type': 'diffusion',
                        'Model': model_name,
                        'Level': level,
                        'Time': time,
                        'Frequency': freq,
                        'Value': pred_value
                    })
                    reg_value = spectra_rg_leadtime[time, freq_idx].values.item()
                    spectra_data.append({
                        'Data': 'spectra',
                        'Variable': var_name,
                        'Type': 'regression',
                        'Model': model_name,
                        'Level': level,
                        'Time': time,
                        'Frequency': freq,
                        'Value': reg_value
                    })
                    reg_value = spectra_tr_leadtime[time, freq_idx].values.item()
                    spectra_data.append({
                        'Data': 'spectra',
                        'Variable': var_name,
                        'Type': 'target',
                        'Model': model_name,
                        'Level': level,
                        'Time': time,
                        'Frequency': freq,
                        'Value': reg_value
                    })

            if 'Ek' == var_name:
                target = compute_ws(targets.isel(levels = i))
                prediction = compute_ws(predictions.isel(levels = i))
                regression = compute_ws(regressions.isel(levels = i))

            for time in times:
                bins_pd, dist_pd = load_dist(prediction[:,time,:,:], bins)
                bins_rg, dist_rg = load_dist(regression[:,time,:,:], bins)
                bins_tr, dist_tr = load_dist(target[:,time,:,:], bins)

                #  Save the current seterr so that I can aviod warnings from np.log of zero devision
                old_settings = np.seterr(divide='ignore', invalid='ignore')
                log_dist_pd = np.log(dist_pd)
                log_dist_rg = np.log(dist_rg)
                log_dist_tr = np.log(dist_tr)
                np.seterr(**old_settings)

                for bin_center, pdf_value in zip(bins_pd, log_dist_pd):
                    distribution_data.append({
                        'Data': 'distribution',
                        'Variable': var_name,
                        'Type': 'diffusion',
                        'Model': model_name,
                        'Level': level,
                        'Time': time,
                        'Bin': bin_center,
                        'Value': pdf_value
                    })
                for bin_center, pdf_value in zip(bins_rg, log_dist_rg):
                    distribution_data.append({
                        'Data': 'distribution',
                        'Variable': var_name,
                        'Type': 'regression',
                        'Model': model_name,
                        'Level': level,
                        'Time': time,
                        'Bin': bin_center,
                        'Value': pdf_value
                    })
                for bin_center, pdf_value in zip(bins_tr, log_dist_tr):
                    distribution_data.append({
                        'Data': 'distribution',
                        'Variable': var_name,
                        'Type': 'target',
                        'Model': model_name,
                        'Level': level,
                        'Time': time,
                        'Bin': bin_center,
                        'Value': pdf_value
                    })


    spectra_data_df = pd.DataFrame(spectra_data)
    spectra_csv_path = os.path.join(paranet_dir, f'spectra_{model_name}.csv')
    spectra_data_df.to_csv(spectra_csv_path, index=False)
    
    distribution_data_df = pd.DataFrame(distribution_data)
    distribution_csv_path = os.path.join(paranet_dir, f'distributions_{model_name}.csv')
    distribution_data_df.to_csv(distribution_csv_path, index=False)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_path",
        type=str,
        help="full path to the ensemble simulation directory",
    )
    args = parser.parse_args()
    base_path = args.base_path
    
    viz_variable_names = ['q_comb', 't_comb', 'u_comb', 'v_comb', 'Ek']
    viz_surf_variable_names = ['u10m','v10m','refc', 'msl']
    viz_levels = [1,2,3,4,5,6,7,8,9,10,11,13,15,20,25,30]
    
    prediction_files = find_nc_files(base_path, 'ds_pred_edm.nc')
    prediction_files.sort()
    regression_files = find_nc_files(base_path, 'ds_pred_noedm.nc')
    regression_files.sort()
    target_files = find_nc_files(base_path, 'ds_targ.nc')
    target_files.sort()

    if not target_files or not prediction_files:
        print("No netCDF files found in the specified directory: ", base_path)
        return

    predictions, regressions, targets = concatenate_netcdfs(
        prediction_files,
        regression_files,
        target_files,
        viz_variable_names + viz_surf_variable_names, 
        viz_levels,
    )

    pred_rmse = np.sqrt(weighted_mse(predictions,targets))
    reg_rmse = np.sqrt(weighted_mse(regressions,targets))
    store_metric(pred_rmse, reg_rmse, base_path, 'rmse')
    pred_fss = fraction_skill_score(predictions,targets)
    reg_fss = fraction_skill_score(regressions,targets)
    store_metric(pred_fss, reg_fss, base_path, 'fss')
    save_spectra_dist(predictions,
                      regressions,
                      targets,
                      viz_variable_names,
                      viz_surf_variable_names,
                      viz_levels,
                      base_path,
                     )
    print("statisical analysis completed")
    

if __name__ == "__main__":
    main()
