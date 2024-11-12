import os
import sys
import xarray as xr
import numpy as np
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.absolute().parent))
from refactored_inference_gfs import run_forecast
from scoring import machine_info
import datetime

use_local_data = False
use_swiftstack = not use_local_data
model_shortname = "diffusion_regression_a2a_v3_1_exclude_w_pstep_pos_embed_v2_noema_16M"
#model_shortname = "diffusion_regression_a2a_v3_1_exclude_w_pstep_pos_embed_noema"

if use_local_data:

    print("Using local data")

    gfs_path = machine_info.gfs_path
    #list all the files in the gfs_path
    gfs_files = os.listdir(gfs_path)

    print("gfs_files", gfs_files)

    for gfs_file in gfs_files:

        datestr = gfs_file.split("_")[1]
        init_z = int(gfs_file.split("_")[2][:2])
        initstr = f"{datestr}-{init_z:02d}"
        datetime_obj = datetime.datetime.strptime(initstr, "%Y%m%d-%H")

        #check if the forecast has already been made
        ml_forecast_path = machine_info.ml_forecasts_path + f"{model_shortname}/"
        ml_forecast_filename = f"ml_{datestr}_{init_z:02d}z.zarr"
        ml_forecast_file = ml_forecast_path + ml_forecast_filename
        print("ml_forecast_file", ml_forecast_file)
        if os.path.exists(ml_forecast_file):
            print(f"Found existing file {ml_forecast_file}, skipping and exiting")
            continue

        run_forecast(
            model_shortname=model_shortname,
            initial_time=datetime_obj,
            n_steps=18,
        )
elif use_swiftstack:

    import s3fs
    import os
    import xarray as xr
    import shutil

    access_key = 'team-earth2-datasets'
    secret_key = os.environ['PDXSECRET']
    endpoint_url = 'https://pdx.s8k.io'

    swiftstack_staging_path = '/data/realtime_hrrr/staging/'

    fs = s3fs.S3FileSystem(key=access_key, secret=secret_key, client_kwargs={'endpoint_url': endpoint_url})

    gfs_files = fs.ls('us_km_ddwp/realtime_hrrr/zarrfiles/gfs/')

    for file_ in gfs_files:

        filename = file_.split("/")[-1]
        datestr = filename.split("_")[1]
        init_z = int(filename.split("_")[2][:2])
        initstr = f"{datestr}-{init_z:02d}"
        datetime_obj = datetime.datetime.strptime(initstr, "%Y%m%d-%H")
        ml_forecast_path = machine_info.ml_forecasts_path + f"{model_shortname}/"
        ml_forecast_filename = f"ml_{datestr}_{init_z:02d}z.zarr"
        ml_forecast_file = ml_forecast_path + ml_forecast_filename
        if os.path.exists(ml_forecast_file):
            print(f"Found existing file {ml_forecast_file}, skipping and exiting")
            continue
        
        gfs_file = swiftstack_staging_path + filename
        if not os.path.exists(gfs_file):
            fs.get(file_, gfs_file, recursive=True)
        hrrr_file_path = f"us_km_ddwp/realtime_hrrr/zarrfiles/hrrr/hrrr_{datestr}_{init_z:02d}z_f01.zarr"
        hrrr_file_name = hrrr_file_path.split("/")[-1]
        hrrr_file = swiftstack_staging_path + hrrr_file_name

        if not os.path.exists(hrrr_file):
            fs.get(hrrr_file_path, hrrr_file, recursive=True)

        try:
            run_forecast(
                model_shortname=model_shortname,
                initial_time=datetime_obj,
                n_steps=18,
                use_swiftstack=True,
                swiftstack_staging_path=swiftstack_staging_path,
                movie=False
                )
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        #delete the gfs and hrrr files from staging
        shutil.rmtree(gfs_file)
        shutil.rmtree(hrrr_file)
        


