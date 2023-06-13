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
import intake
import datetime
import xarray
import numpy as np
import dask
import hydra
import s3fs
import boto3
import tempfile
import h5py
import netCDF4

from dask.diagnostics.progress import ProgressBar
from omegaconf import DictConfig

from e2_datasets.path import catalog_path


def upload_np_array_to_s3(np_array, s3_path, s3_fs):
    # Upload np array to s3
    with tempfile.NamedTemporaryFile() as f:
        np.save(f, np_array)
        f.flush()
        s3_fs.put(f.name, s3_path)


def subsample_era5_4_var(era5_pl, era5_sl, era5_tisr, era5_6h_acc_precip):
    # This function subsamples the ERA5 data to 4 variables and is only used for testing
    # Subsample ERA5 data to 4 variables
    subsampled_era5_4 = [
        era5_pl["T"].sel(level=850),  # 0
        era5_pl["Z"].sel(level=1000),  # 1
        era5_pl["Z"].sel(level=700),  # 2
        era5_pl["Z"].sel(level=500),  # 3
        era5_pl["Z"].sel(level=300),  # 4
        era5_sl.TCWV,  # 5
        era5_sl.VAR_2T,  # 6
    ]
    return subsampled_era5_4


def subsample_era5_34_var(era5_pl, era5_sl, era5_tisr, era5_6h_acc_precip):
    # Subsample ERA5 data to 34 variables
    subsampled_era5_34 = [
        era5_sl.VAR_10U,
        era5_sl.VAR_10V,
        era5_sl.VAR_2T,
        era5_sl.SP,
        era5_sl.MSL,
        era5_pl["T"].sel(level=850),
        era5_pl["U"].sel(level=1000),
        era5_pl["V"].sel(level=1000),
        era5_pl["Z"].sel(level=1000),
        era5_pl["U"].sel(level=850),
        era5_pl["V"].sel(level=850),
        era5_pl["Z"].sel(level=850),
        era5_pl["U"].sel(level=500),
        era5_pl["V"].sel(level=500),
        era5_pl["Z"].sel(level=500),
        era5_pl["T"].sel(level=500),
        era5_pl["Z"].sel(level=50),
        era5_pl["R"].sel(level=500),
        era5_pl["R"].sel(level=850),
        era5_sl.TCWV,
        era5_sl.VAR_100U,
        era5_sl.VAR_100V,
        era5_pl["U"].sel(level=250),
        era5_pl["V"].sel(level=250),
        era5_pl["Z"].sel(level=250),
        era5_pl["T"].sel(level=250),
        era5_pl["U"].sel(level=100),
        era5_pl["V"].sel(level=100),
        era5_pl["Z"].sel(level=100),
        era5_pl["T"].sel(level=100),
        era5_pl["U"].sel(level=900),
        era5_pl["V"].sel(level=900),
        era5_pl["Z"].sel(level=900),
        era5_pl["T"].sel(level=900),
    ]
    return subsampled_era5_34


def create_xarray(subsampled_era5, name):

    # Only subsample era5 data so we only take years 1980-2021
    subsampled_era5 = [
        ch.sel(time=slice("1980-01-01", "2021-01-01")) for ch in subsampled_era5
    ]

    # Chunk every xarray the same way
    common_chunks = {"time": 1, "latitude": 721, "longitude": 1440}
    subsampled_era5 = [ch.chunk(common_chunks) for ch in subsampled_era5]

    # Create xarray dataset
    channel_name = [ch.name for ch in subsampled_era5]
    channel_level = [ch.coords.get("level", np.nan) for ch in subsampled_era5]
    dropped_subsampled_era5 = [
        ch.drop("level", errors="ignore") for ch in subsampled_era5
    ]

    return dropped_subsampled_era5


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Open ERA5 from intake catalog
    catalog = intake.open_catalog(catalog_path)
    with dask.config.set(
        scheduler="processes",
        num_workers=cfg.dask.num_workers,
        threads_per_worker=cfg.dask.threads_per_worker,
    ):
        if cfg.debug:
            era5_pl = catalog.pbss_era5.era5_pl.read(variables=["T", "Z"])
            era5_sl = catalog.pbss_era5.era5_sl.read(variables=["VAR_2T", "TCWV"])
        else:
            era5_pl = catalog.pbss_era5.era5_pl.read(
                variables=["T", "U", "V", "Z", "R"]
            )
            era5_sl = catalog.pbss_era5.era5_sl.read(
                variables=[
                    "VAR_10U",
                    "VAR_10V",
                    "VAR_2T",
                    "SP",
                    "MSL",
                    "VAR_100U",
                    "VAR_100V",
                    "TCWV",
                ]
            )
        era5_tisr = catalog.pbss_era5.era5_tisr.to_dask()
        era5_6h_acc_precip = catalog.pbss_era5.era5_6h_acc_precip.to_dask()
    # Subsample ERA5
    if cfg.debug:
        subsampled_era5 = subsample_era5_4_var(
            era5_pl, era5_sl, era5_tisr, era5_6h_acc_precip
        )
    else:
        if cfg.dataset.name == "era5_34_var":
            subsampled_era5 = subsample_era5_34_var(
                era5_pl, era5_sl, era5_tisr, era5_6h_acc_precip
            )
        else:
            raise ValueError(f"Unknown dataset {cfg.dataset.name}")

    # Create xarray of subsampled ERA5
    era5_xarray = create_xarray(subsampled_era5, cfg.dataset.name)
    # Subset to desired time frequency
    if cfg.dataset.dt != 1:
        times = era5_xarray[0].time.dt.hour % cfg.dataset.dt == 0
        for i in range(len(era5_xarray)):
            print(i)
            era5_xarray[i] = era5_xarray[i].sel(time=times)

    # Set years (please keep this hard set)
    if cfg.debug:
        year_pairs = {
            "train": list(range(1980, 2016)),
            "test": [2016, 2017],
            "out_of_sample": [2018],
        }
    else:
        year_pairs = {
            "train": list(range(1980, 2016)),
            "test": [2016, 2017],
            "out_of_sample": [2018],
        }

    # Make hdf5 files
    for split, years in year_pairs.items():
        # Save each year
        for year in years:
            for i in range(len(era5_xarray)):
                # HDF5 filename
                filename = os.path.join(split, f"{year}_{i}.nc")
                # Save year using dask
                with dask.config.set(
                    scheduler="threads",
                    num_workers=cfg.dask.num_workers,
                    threads_per_worker=cfg.dask.threads_per_worker,
                    **{"array.slicing.split_large_chunks": False},
                ):
                    with ProgressBar():
                        # Get data for the current year
                        year_data = era5_xarray[i].sel(
                            time=era5_xarray[i].time.dt.year == year
                        )
                        # year_data = era5_xarray[i].sel(time=slice('1980-01-01', '1980-01-02'))
                        year_data.to_netcdf(filename, engine="h5netcdf")


if __name__ == "__main__":
    main()
