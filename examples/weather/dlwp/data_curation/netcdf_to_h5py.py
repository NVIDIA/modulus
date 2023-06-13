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
import subprocess
import xarray as xr
import numpy as np
from pathlib import Path


def netcdf_to_h5py(base_path, start_year, end_year, variables_list):
    modified_files_path = base_path[:-1] + "-h5py/"
    Path(modified_files_path).mkdir(parents=True, exist_ok=True)

    for year in range(start_year, end_year):
        print(f"Processing Year: {year}")
        nc = xr.open_dataset(base_path + str(year) + ".nc")
        channel = xr.concat(
            [nc[var] for var in variables_list], dim="channel"
        ).transpose("time", "channel", "face", "y", "x")
        h5_filename = modified_files_path + str(year) + ".h5"
        ds = channel.to_dataset(name="fields")
        ds.to_netcdf(h5_filename, engine="h5netcdf")


netcdf_to_h5py(
    "./train-post-concat/",
    1980,
    2016,
    ["VAR_T_850", "Z_1000", "Z_700", "Z_500", "Z_300", "TCWV", "VAR_T_2"],
)
netcdf_to_h5py(
    "./test-post-concat/",
    2016,
    2018,
    ["VAR_T_850", "Z_1000", "Z_700", "Z_500", "Z_300", "TCWV", "VAR_T_2"],
)
netcdf_to_h5py(
    "./out_of_sample-post-concat/",
    2018,
    2019,
    ["VAR_T_850", "Z_1000", "Z_700", "Z_500", "Z_300", "TCWV", "VAR_T_2"],
)
