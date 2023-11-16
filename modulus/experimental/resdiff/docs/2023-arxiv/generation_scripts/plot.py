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

# %%
import xarray as xr

def open_samples(f):
    root = xr.open_dataset(f)
    pred = xr.open_dataset(f, group="prediction")
    truth = xr.open_dataset(f, group="truth")

    pred = pred.merge(root)
    truth = truth.merge(root)

    truth = truth.set_coords(["lon", "lat"])
    pred = pred.set_coords(["lon", "lat"])
    return truth, pred, root

f = "/lustre/fsw/nvresearch/nbrenowitz/diffusions/samples/87376.nc"
truth, pred, root = open_samples(f)

# import matplotlib.pyplot as plt
# for v in pred:
#     plt.figure()
#     pred[v].plot(col="ensemble", row="time")

# %%
y = truth.assign_coords(ensemble='truth').expand_dims("ensemble")
plotme = xr.concat([pred, y], dim='ensemble')


pred['maximum_radar_reflectivity'][:5, :].plot(col='ensemble', row='time')

# %%
