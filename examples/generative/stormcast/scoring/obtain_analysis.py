import os
import datetime
import machine_info
import sys
import xarray as xr
import pathlib
import numpy as np
sys.path.append(str(pathlib.Path(__file__).parent.absolute().parent))
from hrrr_forecasts.grab_forecasts import load_dataset, ZarrId, create_s3_group_url, create_s3_subgroup_url



start_date = datetime.datetime(2024, 5, 3, 0)
end_date = datetime.datetime(2024, 5, 17, 0)

timestamps = [start_date + datetime.timedelta(hours=i) for i in range(0, int((end_date - start_date).total_seconds() / 3600), 1)]

#timestamps = timestamps[0:10]

X_START= 579
X_END= 1219
Y_START= 273
Y_END= 785

anl_arr = []
time_arr = []

for timestamp in timestamps:

    anl_date = timestamp
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

new_ds.to_zarr('/data/realtime_hrrr/zarrfiles/analysis/analysis.zarr', mode='w')