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


import numpy as np
import torch


def reshape_fields(
    img,
    inp_or_tar,
    crop_size_x,
    crop_size_y,
    rnd_x,
    rnd_y,
    y_roll,
    train,
    n_history,
    in_channels,
    out_channels,
    min_path,
    max_path,
    global_means_path,
    global_stds_path,
    normalization,
    gridtype,
    N_grid_channels,
    roll,
    normalize=True,
    grid=False,
):
    """
    Takes in np array of size (n_history+1, c, h, w) and returns torch tensor of
    size ((n_channels*(n_history+1), crop_size_x, crop_size_y)
    """

    if len(np.shape(img)) == 3:
        img = np.expand_dims(img, 0)

    if img.shape[3] > 720:
        img = img[:, :, 0:720]  # remove last pixel for era5 data

    n_history = n_history

    img_shape_x = np.shape(img)[-2]
    img_shape_y = np.shape(img)[-1]
    n_channels = np.shape(img)[1]  # this will either be N_in_channels or N_out_channels
    channels = in_channels if inp_or_tar == "inp" else out_channels

    if normalize and train:
        mins = np.load(min_path)[:, channels]
        maxs = np.load(max_path)[:, channels]
        means = np.load(global_means_path)[:, channels]
        stds = np.load(global_stds_path)[:, channels]

    if crop_size_x is None:
        crop_size_x = img_shape_x
    if crop_size_y is None:
        crop_size_y = img_shape_y

    if normalize and train:
        if normalization == "minmax":
            img -= mins
            img /= maxs - mins
        elif normalization == "zscore":
            img -= means
            img /= stds

    if grid:
        if inp_or_tar == "inp":
            if gridtype == "linear":
                if N_grid_channels != 2:
                    raise ValueError(
                        "N_grid_channels must be set to 2 for gridtype linear"
                    )
                x = np.meshgrid(np.linspace(-1, 1, img_shape_x))
                y = np.meshgrid(np.linspace(-1, 1, img_shape_y))
                grid_x, grid_y = np.meshgrid(y, x)
                grid = np.stack((grid_x, grid_y), axis=0)
            elif gridtype == "sinusoidal":
                # print('sinusuidal grid added ......')
                if N_grid_channels != 4:
                    raise ValueError(
                        "N_grid_channels must be set to 4 for gridtype sinusoidal"
                    )
                n_channels = n_channels + N_grid_channels
                x1 = np.meshgrid(np.sin(np.linspace(0, 2 * np.pi, img_shape_x)))
                x2 = np.meshgrid(np.cos(np.linspace(0, 2 * np.pi, img_shape_x)))
                y1 = np.meshgrid(np.sin(np.linspace(0, 2 * np.pi, img_shape_y)))
                y2 = np.meshgrid(np.cos(np.linspace(0, 2 * np.pi, img_shape_y)))
                grid_x1, grid_y1 = np.meshgrid(y1, x1)
                grid_x2, grid_y2 = np.meshgrid(y2, x2)
                grid = np.expand_dims(
                    np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis=0), axis=0
                )
            img = np.concatenate((img, grid), axis=1)

    if roll:
        img = np.roll(img, y_roll, axis=-1)

    if crop_size_x or crop_size_y:  # TODO check if this should be done only in training
        img = img[:, :, rnd_x : rnd_x + crop_size_x, rnd_y : rnd_y + crop_size_y]

    if inp_or_tar == "inp":
        img = np.reshape(img, (n_channels * (n_history + 1), crop_size_x, crop_size_y))
    elif inp_or_tar == "tar":
        img = np.reshape(img, (n_channels, crop_size_x, crop_size_y))

    return torch.as_tensor(img)
