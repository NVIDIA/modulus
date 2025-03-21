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
import sys
import os

import numpy as np
import torch
from configs.ord_ph import (
    path_to_hrrr,
    path_to_pretrained,
    station_locations,
    val_station_path,
    isd_path,
)

from sda.score import GaussianScore_from_denoiser, VPSDE_from_denoiser, VPSDE
import matplotlib.pyplot as plt

from utils import find_takeout, find_takeout_random
import xarray as xr
from training.utils.diffusions.networks import get_preconditioned_architecture
import torch


hr = xr.open_zarr(path_to_hrrr, mask_and_scale=False)


device = torch.device("cuda:0")

model = get_preconditioned_architecture(
    name="ddpmpp-cwb-v0",
    resolution=128,
    target_channels=3,
    conditional_channels=0,
    label_dim=0,
)
# print(model)
state = torch.load(path_to_pretrained)
model.load_state_dict(state, strict=False)
model = model.to(device)

u10_mean, u10_std = -0.262, 2.372
v10_mean, v10_std = 0.865, 4.115
logtp_mean, logtp_std = -8.117, 2.489
means, stds = np.array([u10_mean, v10_mean, logtp_mean]), np.array(
    [u10_std, v10_std, logtp_std]
)

ds_regrid = xr.open_dataset(isd_path)

stat_loc = xr.open_dataarray(station_locations)

num_leave = 10
valid = []
for _ in range(num_leave):
    bool_array = stat_loc.values.astype(bool)
    for indices in valid:
        bool_array[indices[0], indices[1]] = False
    valid.append(find_takeout_random(bool_array))

tune = np.zeros_like(bool_array)
for indices in valid:
    tune[indices[0], indices[1]] = True
mses = []
mses_hr = []

for day in range(100):  # 182):
    target_time = slice(day * 24, (day + 1) * 24)
    u10 = ds_regrid.isel(DATE=target_time).u10.values
    v10 = ds_regrid.isel(DATE=target_time).v10.values
    tp = np.log(ds_regrid.isel(DATE=target_time).tp.values + 0.0001)
    obs = np.array([u10, v10, tp])
    obs -= means[:, np.newaxis, np.newaxis, np.newaxis]
    obs /= stds[:, np.newaxis, np.newaxis, np.newaxis]
    obs = obs.transpose(3, 0, 2, 1)
    obs = torch.tensor(obs)

    inf_obs = obs.where(torch.tensor(np.tile(bool_array, (len(obs), 3, 1, 1))), np.nan)
    val_obs = obs.where(torch.tensor(np.tile(tune, (len(obs), 3, 1, 1))), np.nan)
    mask = ~np.isnan(inf_obs).bool()

    def A(x):
        """
        Mask the observations to the valid locations.
        """
        return x[mask]

    y_star = A(inf_obs)

    sde = VPSDE(  # from this we use sample()
        GaussianScore_from_denoiser(  # from this, sample() uses VPSDE.eps() which is GaussianScore_from_denoiser.forward()
            y_star,
            A=A,
            std=0.1,
            gamma=0.001,
            sde=VPSDE_from_denoiser(
                model, shape=()
            ),  # which calls VPSDE_from_denoiser.eps_from_denoiser() which calls the network
        ),
        shape=obs.shape,
    ).cuda()
    x = sde.sample(steps=64, corrections=2, tau=0.3, makefigs=False).cpu()

    u10 = hr.isel(time=target_time).sel(channel="10u").HRRR.values
    v10 = hr.isel(time=target_time).sel(channel="10v").HRRR.values
    tp = np.log(hr.isel(time=target_time).sel(channel="tp").HRRR.values + 0.0001)
    hrrr = np.array([u10, v10, tp])
    hrrr -= means[:, np.newaxis, np.newaxis, np.newaxis]
    hrrr /= stds[:, np.newaxis, np.newaxis, np.newaxis]
    hrrr = hrrr.transpose(1, 0, 2, 3)

    mse = np.nanmean((x - val_obs) ** 2, axis=(0, 2, 3))
    mse_hr = np.nanmean((hrrr - val_obs.numpy()) ** 2, axis=(0, 2, 3))

    mses.append(mse)
    mses_hr.append(mse_hr)

mses = np.array(mses)
print("RMSE = ", np.sqrt(mses.mean(axis=0)))
np.save(f"tuning_std/rands_mse_for_eval", mses)
np.save(f"tuning_std/rands_mse_for_eval_hr", mses_hr)
