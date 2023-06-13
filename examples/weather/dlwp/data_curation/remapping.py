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

base_path = "./train/"


def remapping(base_path, start_year, end_year, variables_list, new_var_names, map_file):
    # prepare directory to store the modified data
    modified_files_path = base_path[:-1] + "-post/"
    Path(modified_files_path).mkdir(parents=True, exist_ok=True)

    for year in range(start_year, end_year):
        print(f"Processing Year: {year}")
        for i, var in enumerate(variables_list):
            print(f"Processing Variable: {var}")
            filename = base_path + str(year) + "_" + str(i) + ".nc"
            newfilename = modified_files_path + str(year) + "_" + str(i) + ".nc"

            ds = xr.open_dataset(filename)
            # Clean-up the attributes to avoid errors with tempest-remap
            ds.attrs["image_git_repo"] = "random"
            ds.attrs["image_git_commit_sha"] = "random"
            ds.attrs["SLURM_JOB_USER"] = "random"

            ds.to_netcdf(newfilename)

            cmd1 = "ncatted -a _FillValue,,m,f,9.96921e+36 " + str(newfilename)
            cmd2 = "ncatted -a missing_value,,m,f,9.96921e+36 " + str(newfilename)
            cmd3 = "ncatted -a number_of_significant_digits,,m,f,0 " + str(newfilename)
            cmd4 = "ncatted -a ecmwf_local_table,,m,f,0 " + str(newfilename)
            cmd5 = "ncatted -a ecmwf_parameter,,m,f,0 " + str(newfilename)

            subprocess.run(cmd1, shell=True)
            subprocess.run(cmd2, shell=True)
            subprocess.run(cmd3, shell=True)
            subprocess.run(cmd4, shell=True)
            subprocess.run(cmd5, shell=True)
            print("pre-processing complete, applying remaps")

            # Apply remaps
            mapped_filename = modified_files_path + str(year) + "_" + str(i) + "_cs.nc"
            remap_cmd = (
                "ApplyOfflineMap --in_data "
                + str(newfilename)
                + " --out_data "
                + mapped_filename
                + " --map "
                + str(map_file)
                + " --var "
                + var
            )
            output = subprocess.run(remap_cmd, shell=True, stdout=subprocess.DEVNULL)
            print("applying remaps complete, now reshaping")

            # reshape tempest remap's output to (face, res, res) shape
            reshaped_mapped_filename = (
                modified_files_path + str(year) + "_" + str(i) + "_rs_cs.nc"
            )
            ds = xr.open_dataset(mapped_filename)
            list_datasets = []
            for key in list(ds.keys()):
                if key == "lat" or key == "lon":
                    pass
                else:
                    data_var = ds[key]
                    time_var = ds["time"]
                    col_var = ds["ncol"]

                    num = 6
                    res = int(np.sqrt(col_var.size / num))

                    y_coords = np.arange(res)
                    x_coords = np.arange(res)

                    data_var_reshaped = data_var.data.reshape(
                        (time_var.size, num, res, res)
                    )

                    # Create a new coordinate for the 'face' dimension
                    face_coords = np.arange(num)

                    # Create a new DataArray with the reshaped data and updated coordinates
                    reshaped_da = xr.DataArray(
                        data_var_reshaped,
                        dims=[
                            "time",
                            "face",
                            "y",
                            "x",
                        ],
                        name=new_var_names[i],
                    )

                    # Add the coordinates to the reshaped DataArray
                    reshaped_da["time"] = ("time", ds["time"].data)
                    reshaped_da["face"] = ("face", face_coords)
                    reshaped_da["y"] = ("y", y_coords)
                    reshaped_da["x"] = ("x", x_coords)

                    # Copy the attributes from the original data variable
                    reshaped_da.attrs = data_var.attrs

                    list_datasets.append(reshaped_da)

            combined = xr.merge(list_datasets)
            # Save the dataset to a new file
            combined.to_netcdf(reshaped_mapped_filename)
            print("reshaping complete")

            # Clean-up temp files
            os.remove(newfilename)
            os.remove(mapped_filename)


# remap train data
remapping(
    "./train/",
    1980,
    2016,
    ["T", "Z", "Z", "Z", "Z", "TCWV", "VAR_2T"],
    ["VAR_T_850", "Z_1000", "Z_700", "Z_500", "Z_300", "TCWV", "VAR_T_2"],
    "./map_LL721x1440_CS64.nc",
)

# remap test data
remapping(
    "./test/",
    2016,
    2018,
    ["T", "Z", "Z", "Z", "Z", "TCWV", "VAR_2T"],
    ["VAR_T_850", "Z_1000", "Z_700", "Z_500", "Z_300", "TCWV", "VAR_T_2"],
    "./map_LL721x1440_CS64.nc",
)

# remap out-of-sample data
remapping(
    "./out_of_sample/",
    2018,
    2019,
    ["T", "Z", "Z", "Z", "Z", "TCWV", "VAR_2T"],
    ["VAR_T_850", "Z_1000", "Z_700", "Z_500", "Z_300", "TCWV", "VAR_T_2"],
    "./map_LL721x1440_CS64.nc",
)
