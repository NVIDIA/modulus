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


import xarray as xr
import numpy as np
import subprocess

# read any arbitrary input
ds = xr.open_dataset("./train/1980_0.nc")
latgrid, longrid = np.meshgrid(ds["longitude"].values, ds["latitude"].values)

# Create DataArrays from the 2D arrays and add dimensions
da_latgrid = xr.DataArray(
    latgrid,
    coords=[ds["latitude"].values, ds["longitude"].values],
    dims=["latitude", "longitude"],
)
da_longrid = xr.DataArray(
    longrid,
    coords=[ds["latitude"].values, ds["longitude"].values],
    dims=["latitude", "longitude"],
)

# Create a Dataset and add the DataArrays
ds_new = xr.Dataset({"latgrid": da_latgrid, "longrid": da_longrid})

# Save the Dataset to NetCDF
ds_new.to_netcdf("latlon_grid_field.nc")

# Remap the grid to cubed sphere
remap_cmd = "ApplyOfflineMap --in_data latlon_grid_field.nc --out_data latlon_grid_field_cs.nc --map ./map_LL721x1440_CS64.nc --var 'latgrid,longrid'"
output = subprocess.run(remap_cmd, shell=True, stdout=subprocess.DEVNULL)

# reshape tempest remap's output to (face, res, res) shape
mapped_filename = "latlon_grid_field_cs.nc"
reshaped_mapped_filename = "latlon_grid_field_rs_cs.nc"
ds = xr.open_dataset(mapped_filename)
list_datasets = []
for key in list(ds.keys()):
    if key == "lat" or key == "lon":
        pass
    else:
        data_var = ds[key]
        col_var = ds["ncol"]

        num = 6
        res = int(np.sqrt(col_var.size / num))

        y_coords = np.arange(res)
        x_coords = np.arange(res)

        data_var_reshaped = data_var.data.reshape((num, res, res))

        # Create a new coordinate for the 'face' dimension
        face_coords = np.arange(num)

        # Create a new DataArray with the reshaped data and updated coordinates
        reshaped_da = xr.DataArray(
            data_var_reshaped,
            dims=[
                "face",
                "y",
                "x",
            ],
            name=key,
        )

        # Add the coordinates to the reshaped DataArray
        reshaped_da["face"] = ("face", face_coords)
        reshaped_da["y"] = ("y", y_coords)
        reshaped_da["x"] = ("x", x_coords)

        # Copy the attributes from the original data variable
        reshaped_da.attrs = data_var.attrs

        list_datasets.append(reshaped_da)

combined = xr.merge(list_datasets)
# Save the dataset to a new file
combined.to_netcdf(reshaped_mapped_filename)
