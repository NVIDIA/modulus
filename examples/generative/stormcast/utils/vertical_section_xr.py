import xarray as xr
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from utils.diagnostics import add_diagnostics
#rom utils.diagnostics import calculate_divergence
import matplotlib.colors as colors
import matplotlib.pyplot as plt

def interpolate_vertical(ds, latitude,diagnostics=False):
    ds = ds.sel(levels=slice(0, 20))
    z_levels = np.array([125.0, 150.0, 200.0, 280.0, 400.0, 560.0, 750.0, 970.0, 1210.0, 1500.0,
                         1800.0, 2500.0, 3500.0, 6300.0])

    ds_sub = ds.sel(y=latitude)
    dict_2d = {}

    for var_name in ds_sub.data_vars:
        if var_name not in ds_sub.dims and var_name not in ds_sub.coords:
            if len(ds_sub[var_name].shape) == 3:
                data = ds_sub[var_name].values
                original_levels = ds_sub['z_comb'].values

                z_levels_3d = np.repeat(z_levels[np.newaxis, :, np.newaxis, np.newaxis],
                                        original_levels.shape[0], axis=0)
                z_levels_3d = np.repeat(z_levels_3d, original_levels.shape[2], axis=2)

                interpolated_data = np.empty_like(z_levels_3d)

                for t in range(data.shape[0]):
                    for x in range(data.shape[2]):
                        interpolator = interp1d(original_levels[t, :, x], data[t, :, x],
                                                 kind='cubic', bounds_error=False, fill_value=np.nan)

                        interpolated_data[t, :, x] = interpolator(z_levels_3d[t,:,x])

                dict_2d[var_name] = np.squeeze(interpolated_data)

    dict_2d['z'] = np.squeeze(z_levels_3d)

    ds_vert = xr.Dataset()
    for var_name, var_data in dict_2d.items():
        ds_vert[var_name] = xr.DataArray(var_data, dims=('time', 'z_levels', 'x'),
                                                  coords={'time': ds.time.values,
                                                          'z_levels': z_levels,
                                                          'x': ds.x.values})

    longitude = ds.longitude.isel(y=latitude).values
    ds_vert['longitude'] = xr.DataArray(longitude,dims=('x'),
                                       coords={'x':ds.x.values})
    # if add_diagnostics:
    #     ds_vert = add_diagnostics(ds_vert)
        
    return ds_vert



