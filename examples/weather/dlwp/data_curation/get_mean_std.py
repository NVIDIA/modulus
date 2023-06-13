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
import glob


def process_file(file, var):
    ds = xr.open_dataset(file)[var]
    mean = ds.mean().compute()
    std = ds.std().compute()
    var = ds.var().compute()
    return mean, std, var


directories = [
    "./train-post-concat",
    "./test-post-concat",
    "./out_of_sample-post-concat",
]
files = []
for directory in directories:
    files.extend(glob.glob(f"{directory}/*.nc"))

var_list = ["VAR_T_850", "Z_1000", "Z_700", "Z_500", "Z_300", "TCWV", "VAR_T_2"]

mean_list = []
std_list = []
for var in var_list:
    file_stats = [process_file(file, var) for file in files]

    global_mean = np.mean([mean for mean, std, var in file_stats])
    global_var = np.average(
        [var for mean, std, var in file_stats],
        weights=[mean for mean, std, var in file_stats],
    )
    global_std = np.sqrt(global_var)

    # Print the results for individual files
    # for i, (mean, std, var) in enumerate(file_stats):
    #     print(f"File {i+1}: Mean = {mean}, Standard Deviation = {std}")

    mean_list.append(global_mean)
    std_list.append(global_std)
    print(
        f"Var = {var}, Global Mean = {global_mean}, Global Standard Deviation = {global_std}"
    )

mean_array = np.expand_dims(
    np.expand_dims(np.expand_dims(np.stack(mean_list, axis=0), -1), -1), 0
)
std_array = np.expand_dims(
    np.expand_dims(np.expand_dims(np.stack(std_list, axis=0), -1), -1), 0
)
with open("global_means.npy", "wb") as f:
    np.save(f, mean_array)
with open("global_stds.npy", "wb") as f:
    np.save(f, std_array)

print(mean_array.shape, std_array.shape)
