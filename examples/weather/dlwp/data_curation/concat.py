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


def concat(base_path, start_year, end_year, variables_list):
    modified_files_path = base_path[:-1] + "-concat/"
    Path(modified_files_path).mkdir(parents=True, exist_ok=True)

    for year in range(start_year, end_year):
        print(f"Processing Year: {year}")
        list_datasets = []
        for i in range(len(variables_list)):
            list_datasets.append(
                xr.open_dataset(base_path + str(year) + "_" + str(i) + "_rs_cs.nc")
            )

        # list_datasets = [ch.drop("level", errors='ignore') for ch in list_datasets]
        # channel = xr.concat(list_datasets, dim='channel').transpose("time", "channel", "face", "y", "x")
        channel = xr.merge(list_datasets)
        # ds = channel.to_dataset(name="fields")

        combined_filename = modified_files_path + str(year) + ".nc"
        channel.to_netcdf(combined_filename)


concat(
    "./train-post/",
    1980,
    2016,
    ["VAR_T_850", "Z_1000", "Z_700", "Z_500", "Z_300", "TCWV", "VAR_T_2"],
)
concat(
    "./test-post/",
    2016,
    2018,
    ["VAR_T_850", "Z_1000", "Z_700", "Z_500", "Z_300", "TCWV", "VAR_T_2"],
)
concat(
    "./out_of_sample-post/",
    2018,
    2019,
    ["VAR_T_850", "Z_1000", "Z_700", "Z_500", "Z_300", "TCWV", "VAR_T_2"],
)
