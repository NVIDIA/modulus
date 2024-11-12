import os
from realtime_gfs import get_gfs_vars
from realtime_hrrr import get_hrrr
from hrrr_to_zarr import grib2_to_zarr, create_keep_params
import datetime
import machine_info
import sys
import xarray as xr
import pathlib
import numpy as np
sys.path.append(str(pathlib.Path(__file__).parent.absolute().parent))
from hrrr_forecasts.grab_forecasts import load_dataset, ZarrId, create_s3_group_url, create_s3_subgroup_url
from refactored_inference_gfs import run_forecast
from gfs_grib_pipeline import get_forecast

today = datetime.datetime.utcnow()
#inputdatestr = sys.argv[1]
#today = datetime.datetime.strptime(inputdatestr, '%Y%m%d-%H')
#today = datetime.datetime(2024, 5, 28, 6)
day = today.day
month = today.month
year = today.year
hour = today.hour
init_z = int(hour / 6) * 6
initialization_date = datetime.datetime(year, month, day, init_z)
datestr = f'{year}{month:02d}{day:02d}'

savepath_gfs = machine_info.savepath_gfs
savepath_hrrr_grib = machine_info.savepath_hrrr_grib
savepath_hrrr_zarr = machine_info.savepath_hrrr_zarr
savepath_hrrr_forecast = machine_info.savepath_hrrr_forecast

os.makedirs(savepath_gfs, exist_ok=True)
os.makedirs(savepath_hrrr_grib, exist_ok=True)
os.makedirs(savepath_hrrr_zarr, exist_ok=True)
os.makedirs(savepath_hrrr_forecast, exist_ok=True)

for attempt in range(3):
    
    print("attempt", attempt)

    fname = savepath_gfs + f"gfs_{datestr}_{init_z:02d}z.zarr"
    if os.path.exists(fname):
        print(f"Found existing file {fname}, skipping and exiting")
        break
        
    #retval = get_gfs_vars(datestr, init_z, 24, savepath=savepath_gfs)
    retval = get_forecast(datestr, init_z, 24, savepath=savepath_gfs)

    if retval:
        print(f"Successfully downloaded GFS data for {datestr} {init_z:02d}z")
        break
    else:
        initialization_date -= datetime.timedelta(hours=6)
        datestr = initialization_date.strftime('%Y%m%d')
        init_z = initialization_date.hour
        fname = savepath_gfs + f"gfs_{datestr}_{init_z:02d}z.zarr"
        print(f"Failed to download GFS data, checking previous initialization time which is {datestr} {init_z:02d}z")
        if os.path.exists(fname):
            print(f"Found existing file {fname}, skipping and exiting")
            break

#usually the HRRR data is available faster so it should be available
#TODO implement a failsafe if HRRR data is not available at the latest GFS initialization time
filename_sfc = savepath_hrrr_grib + f'hrrr_sfc_{datestr}_{init_z}z_f01.grib2'
filename_nat = savepath_hrrr_grib + f'hrrr_nat_{datestr}_{init_z}z_f01.grib2'
#check if HRRR data is already downloaded
if os.path.exists(filename_sfc) and os.path.exists(filename_nat):
    print(f"Found existing HRRR files {filename_sfc} and {filename_nat}, skipping and exiting")
    retval = True
else:
   retval = get_hrrr(datestr, init_z, savepath=savepath_hrrr_grib)

if retval:

    print(f"Successfully downloaded HRRR data for {datestr} {init_z:02d}z")

    sfc, nat, levels, param_master_list = create_keep_params()

    num_channels = len(param_master_list)

    filename_sfc = savepath_hrrr_grib + f'hrrr_sfc_{datestr}_{init_z}z_f01.grib2'
    filename_nat = savepath_hrrr_grib + f'hrrr_nat_{datestr}_{init_z}z_f01.grib2'

    grib2_to_zarr(sfc, nat, levels, param_master_list, savepath=savepath_hrrr_zarr, datestr=datestr, init_z=init_z, filename_sfc=filename_sfc, filename_nat=filename_nat)

# obtain hrrr forecast data for the same initialization time
X_START= 579
X_END= 1219
Y_START= 273
Y_END= 785

zarr_id = ZarrId(
                    run_hour=initialization_date, #datetime.datetime(2020, 8, 1, 0), # Aug 1, 0Z
                    level_type="sfc",
                    var_level="entire_atmosphere",
                    var_name="REFC",
                    model_type="fcst"
                    )
                
ds = load_dataset([create_s3_group_url(zarr_id), create_s3_subgroup_url(zarr_id)])
ds = ds.isel(x=slice(X_START, X_END), y=slice(Y_START, Y_END)).isel(time=slice(0, 24))

new_ds = xr.Dataset(
    data_vars=dict(
        REFC=(["time", "y", "x"], ds.REFC.values.astype(np.float32)),
    ),
    coords=dict(
        time=ds.time.values,
        longitude=( ["y", "x"], ds.longitude.values),
        latitude=( ["y", "x"], ds.latitude.values),
    ),
)

hrrr_forecast_filename = savepath_hrrr_forecast + f'hrrr_{datestr}_{init_z:02d}z_forecast.zarr'

new_ds.to_zarr(hrrr_forecast_filename, mode='w')


#obtain analysis for the previous day's forecast
anl_date = initialization_date - datetime.timedelta(hours=24)
datestr = anl_date.strftime('%Y%m%d')
init_z = anl_date.hour

anl_arr = []
time_arr = []

for t in range(18):

    anl_date = anl_date + datetime.timedelta(hours=1)
    print(anl_date)

    zarr_id = ZarrId(
                        run_hour=anl_date, 
                        level_type="sfc",
                        var_level="entire_atmosphere",
                        var_name="REFC",
                        model_type="anl"
                        )

    ds = load_dataset([create_s3_group_url(zarr_id), create_s3_subgroup_url(zarr_id)])
    ds = ds.isel(x=slice(X_START, X_END), y=slice(Y_START, Y_END))
    anl_arr.append(ds.REFC.values)
    valid_time = ds.time.values
    time_arr.append(valid_time)

new_ds = xr.Dataset(
    data_vars=dict(
        REFC=(["time", "y", "x"], np.stack(anl_arr, axis=0)),
    ),
    coords=dict(
        time=time_arr,
        longitude=( ["y", "x"], ds.longitude.values),
        latitude=( ["y", "x"], ds.latitude.values),
    ),
)

new_ds.to_zarr(savepath_hrrr_forecast + f'hrrr_{datestr}_{init_z:02d}z_anl.zarr', mode='w')

run_forecast(n_steps=18, initial_time=initialization_date, ensemble_size=7)