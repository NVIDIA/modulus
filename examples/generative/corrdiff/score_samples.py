# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""Score the generated samples


Saves a netCDF of crps and other scores. Depends on time, but space and ensemble have been reduced::

    netcdf scores {
dimensions:
        metric = 4 ;
        time = 205 ;
variables:
        double eastward_wind_10m(metric, time) ;
                eastward_wind_10m:_FillValue = NaN ;
        double maximum_radar_reflectivity(metric, time) ;
                maximum_radar_reflectivity:_FillValue = NaN ;
        double northward_wind_10m(metric, time) ;
                northward_wind_10m:_FillValue = NaN ;
        double temperature_2m(metric, time) ;
                temperature_2m:_FillValue = NaN ;
        int64 time(time) ;
                time:units = "hours since 1990-01-01" ;
                time:calendar = "standard" ;
        string metric(metric) ;
}


"""
# %%
import dask.diagnostics
import xarray as xr
import os
import argparse

try:
    import xskillscore
except ImportError:
    raise ImportError("xskillscore not installed. Try `pip install xskillscore`")


def open_samples(f):
    """
    Open prediction and truth samples from a dataset file.

    Parameters:
        f: Path to the dataset file.

    Returns:
        tuple: A tuple containing truth, prediction, and root datasets.
    """
    root = xr.open_dataset(f)
    pred = xr.open_dataset(f, group="prediction")
    truth = xr.open_dataset(f, group="truth")

    pred = pred.merge(root)
    truth = truth.merge(root)

    truth = truth.set_coords(["lon", "lat"])
    pred = pred.set_coords(["lon", "lat"])
    return truth, pred, root


def main(netcdf_file, output_dir):
    truth, pred, root = open_samples(netcdf_file)
    pred = pred.chunk(time=1)
    truth = truth.chunk(time=1)

    dim = ["x", "y"]

    a = xskillscore.rmse(truth, pred.mean("ensemble"), dim=dim)
    b = xskillscore.crps_ensemble(truth, pred, member_dim="ensemble", dim=dim)
    c = pred.std("ensemble").mean(dim)
    crps_mean = xskillscore.crps_ensemble(
        truth,
        pred.mean("ensemble").expand_dims("ensemble"),
        member_dim="ensemble",
        dim=dim,
    )

    metrics = xr.concat([a, b, c, crps_mean], dim="metric").assign_coords(
        metric=["rmse", "crps", "std_dev", "mae"]
    )
    with dask.diagnostics.ProgressBar():
        metrics.to_netcdf(os.path.join(output_dir, 'scores.nc'), mode="w")


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add the positional arguments
    parser.add_argument("--netcdf_file", help="Path to the NetCDF file")
    parser.add_argument("--output_dir", help="Path to the output directory")
    args = parser.parse_args()
    main(args.netcdf_file, args.output_dir)