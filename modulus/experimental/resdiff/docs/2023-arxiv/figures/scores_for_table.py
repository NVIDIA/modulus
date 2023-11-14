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

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
scores = {}
scores["ResDiff"] = "/lustre/fsw/nvresearch/nbrenowitz/diffusions/generations/era5-cwb-v3/validation_big/scores.nc"
scores["Reg"] = "/lustre/fsw/nvresearch/nbrenowitz/diffusions/baselines/regression/era5-cwb-v3/validation_big/scores.nc"
scores["RF"] = "/lustre/fsw/nvresearch/nbrenowitz/diffusions/baselines/rf/era5-cwb-v3/validation_big/scores.nc"
scores["ERA5"] = "/lustre/fsw/nvresearch/nbrenowitz/diffusions/baselines/era5/era5-cwb-v3/validation_big/scores.nc"

# %%
import xarray
import matplotlib.pyplot as plt
import pandas as pd


arrs = [xarray.open_dataset(path) for path in scores.values()]
ds = xarray.concat(arrs, dim=xarray.Variable(["model"], list(scores)))

# %%
df = ds.mean("time").to_dataframe()
idx = pd.IndexSlice
plotme = df.loc[idx[:, ['crps', 'mae']], :]

names = {"eastward_wind_10m": "u10m", "northward_wind_10m": "v10m", "maximum_radar_reflectivity": "Radar", "temperature_2m": "t2m"}

plotme.columns = [names[c] for c in plotme.columns]

styler =  (plotme.style
    .format(precision=2)
    .format_index(escape="latex", axis=1)
    .format_index(escape="latex", axis=0)
)
print(styler.to_latex(hrules=True))

# %%
plotme

# %% [markdown]
# # debug
