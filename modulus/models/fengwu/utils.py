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

from typing import Tuple

import torch
from torch.nn import ConstantPad3d
from torch.nn.common_types import _size_6_t


def get_earth_position_index(window_size):
    """
    This function construct the position index to reuse symmetrical parameters of the position bias.
    implementation from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        window_size (tuple[int]): [pressure levels, latitude, longitude]

    Returns:
        position_index (torch.Tensor): [win_pl * win_lat * win_lon, win_pl * win_lat * win_lon]
    """
    win_pl, win_lat, win_lon = window_size
    # Index in the pressure level of query matrix
    coords_zi = torch.arange(win_pl)
    # Index in the pressure level of key matrix
    coords_zj = -torch.arange(win_pl) * win_pl

    # Index in the latitude of query matrix
    coords_hi = torch.arange(win_lat)
    # Index in the latitude of key matrix
    coords_hj = -torch.arange(win_lat) * win_lat

    # Index in the longitude of the key-value pair
    coords_w = torch.arange(win_lon)

    # Change the order of the index to calculate the index in total
    coords_1 = torch.stack(torch.meshgrid([coords_zi, coords_hi, coords_w]))
    coords_2 = torch.stack(torch.meshgrid([coords_zj, coords_hj, coords_w]))
    coords_flatten_1 = torch.flatten(coords_1, 1)
    coords_flatten_2 = torch.flatten(coords_2, 1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0).contiguous()

    # Shift the index for each dimension to start from 0
    coords[:, :, 2] += win_lon - 1
    coords[:, :, 1] *= 2 * win_lon - 1
    coords[:, :, 0] *= (2 * win_lon - 1) * win_lat * win_lat

    # Sum up the indexes in three dimensions
    position_index = coords.sum(-1)

    return position_index


def get_earth_position_index_2d(window_size):
    """
    This function construct the position index to reuse symmetrical parameters of the position bias.
    Modify from: https://github.com/198808xc/Pangu-Weather/blob/main/pseudocode.py

    Args:
        window_size (tuple[int]): [latitude, longitude]

    Returns:
        position_index (torch.Tensor): [win_lat * win_lon, win_lat * win_lon]
    """
    win_lat, win_lon = window_size

    # Index in the latitude of query matrix
    coords_hi = torch.arange(win_lat)
    # Index in the latitude of key matrix
    coords_hj = -torch.arange(win_lat) * win_lat

    # Index in the longitude of the key-value pair
    coords_w = torch.arange(win_lon)

    # Change the order of the index to calculate the index in total
    coords_1 = torch.stack(torch.meshgrid([coords_hi, coords_w]))
    coords_2 = torch.stack(torch.meshgrid([coords_hj, coords_w]))
    coords_flatten_1 = torch.flatten(coords_1, 1)
    coords_flatten_2 = torch.flatten(coords_2, 1)
    coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
    coords = coords.permute(1, 2, 0).contiguous()

    # Shift the index for each dimension to start from 0
    coords[:, :, 1] += win_lon - 1
    coords[:, :, 0] *= 2 * win_lon - 1

    # Sum up the indexes in two dimensions
    position_index = coords.sum(-1)

    return position_index


def get_pad3d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): (Pl, Lat, Lon)
        window_size (tuple[int]): (Pl, Lat, Lon)

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = (
        padding_right
    ) = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return (
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
        padding_front,
        padding_back,
    )


def get_pad2d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): Lat, Lon
        window_size (tuple[int]): Lat, Lon

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom)
    """
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[:4]


def crop2d(x: torch.Tensor, resolution):
    """
    Args:
        x (torch.Tensor): B, C, Lat, Lon
        resolution (tuple[int]): Lat, Lon
    """
    _, _, Lat, Lon = x.shape
    lat_pad = Lat - resolution[0]
    lon_pad = Lon - resolution[1]

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left

    return x[
        :, :, padding_top : Lat - padding_bottom, padding_left : Lon - padding_right
    ]


def crop3d(x: torch.Tensor, resolution):
    """
    Args:
        x (torch.Tensor): B, C, Pl, Lat, Lon
        resolution (tuple[int]): Pl, Lat, Lon
    """
    _, _, Pl, Lat, Lon = x.shape
    pl_pad = Pl - resolution[0]
    lat_pad = Lat - resolution[1]
    lon_pad = Lon - resolution[2]

    padding_front = pl_pad // 2
    padding_back = pl_pad - padding_front

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left
    return x[
        :,
        :,
        padding_front : Pl - padding_back,
        padding_top : Lat - padding_bottom,
        padding_left : Lon - padding_right,
    ]


class ZeroPad3d(ConstantPad3d):
    r"""Pads the input tensor boundaries with zero.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in all boundaries. If a 6-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`,
            :math:`\text{padding\_top}`, :math:`\text{padding\_bottom}`,
            :math:`\text{padding\_front}`, :math:`\text{padding\_back}`)

    Shape:
        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or
          :math:`(C, D_{out}, H_{out}, W_{out})`, where

          :math:`D_{out} = D_{in} + \text{padding\_front} + \text{padding\_back}`

          :math:`H_{out} = H_{in} + \text{padding\_top} + \text{padding\_bottom}`

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> m = ZeroPad3d(3)
        >>> input = torch.randn(16, 3, 10, 20, 30)
        >>> output = m(input)
        >>> # using different paddings for different sides
        >>> m = ZeroPad3d((3, 3, 6, 6, 0, 1))
        >>> output = m(input)

    """

    padding: Tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t) -> None:
        super().__init__(padding, 0.0)

    def extra_repr(self) -> str:
        return f"{self.padding}"
