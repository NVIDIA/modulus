import xarray as xr
import numpy as np

def save_subset_to_netcdf_fp16(frames,
                               ds_target,
                               ds_edm,
                               ds_noedm,
                               var1,
                               var2,
                               var3,
                               var4,
                               var5,
                               #var6,
                               #var7,
                               var2_lev,
                               var3_lev,
                               var5_lev,
                               out_path,
                               model_name,
                               case_name,
                               left_lon_idx,
                               right_lon_idx,
                               bottom_lat_idx,
                               top_lat_idx,
                              ):
    """
    Modified function to save data in FP16
    """
    
    def get_data_subset(ds, var_name,lev_val=None,level=False):
        data_list = []
        for frame in frames:
            if level==True and lev_val is not None:
                data_subset = ds[var_name].isel(time=frame, levels=level,y=slice(bottom_lat_idx, top_lat_idx + 1), 
                                                x=slice(left_lon_idx, right_lon_idx + 1))
                data_list.append(data_subset.expand_dims('time'))
            elif level==False and lev_val is None:
                data_subset = ds[var_name].isel(time=frame, y=slice(bottom_lat_idx, top_lat_idx + 1), 
                                                x=slice(left_lon_idx, right_lon_idx + 1))
                data_list.append(data_subset.expand_dims('time'))
        concatenated_data = xr.concat(data_list, dim='time')
        return concatenated_data
    
    var_behavior = {
        var1: {'level': False, 'lev_val': None},
        var2: {'level': True, 'lev_val': int(var2_lev)},
        var3: {'level': True, 'lev_val': int(var3_lev)},
        var4: {'level': True, 'lev_val': int(var3_lev)},
        var5: {'level': True, 'lev_val': int(var5_lev)}
    }

    data_vars = {}
    for var in [var1, var2, var3, var4, var5]:
        behavior = var_behavior.get(var)
        data = [get_data_subset(ds, var, lev_val=behavior['lev_val'], level=behavior['level']) for ds in [ds_target, ds_edm, ds_noedm]]
        data_vars[var] = xr.concat(data, dim='dataset')

    lat_2d_subset = ds_target.latitude.isel(y=slice(bottom_lat_idx, top_lat_idx + 1),
                                            x=slice(left_lon_idx, right_lon_idx + 1))
    lon_2d_subset = ds_target.longitude.isel(y=slice(bottom_lat_idx, top_lat_idx + 1),
                                             x=slice(left_lon_idx, right_lon_idx + 1))
    time_values = ds_target['time'].isel(time=frames).values

    ds_new = xr.Dataset(
        data_vars,
        coords={
            'lat': (('y', 'x'), lat_2d_subset.values),
            'lon': (('y', 'x'), lon_2d_subset.values),
            'time': ('time', time_values),
            'dataset': ['target', 'diffusion', 'regression'],
            'model': model_name,
            'case': case_name
        }
    )

    ds_new.to_netcdf(path=out_path, encoding={var: {'dtype': 'int16'} for var in [var1, var2, var3]})
    print(f"Saved to {out_path}")

