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

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from timm.layers import DropPath, trunc_normal_
from torch import nn
from torch.nn import ZeroPad2d

from ..meta import ModelMetaData
from ..module import Module
from .patch_embed import PatchEmbed2D, PatchRecovery2D
from .shift_window_mask import (
    get_shift_window_mask,
    get_shift_window_mask_2d,
    window_partition,
    window_partition_2d,
    window_reverse,
    window_reverse_2d,
)
from .utils import (
    ZeroPad3d,
    crop2d,
    crop3d,
    get_earth_position_index,
    get_earth_position_index_2d,
    get_pad2d,
    get_pad3d,
)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UpSample_2D(nn.Module):
    """
    Up-sampling operation.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        input_resolution (tuple[int]): [latitude, longitude]
        output_resolution (tuple[int]): [latitude, longitude]
    """

    def __init__(self, in_dim, out_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (B, N, C)
        """
        B, N, C = x.shape
        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution

        x = self.linear1(x)
        x = x.reshape(B, in_lat, in_lon, 2, 2, C // 2).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, in_lat * 2, in_lon * 2, -1)

        pad_h = in_lat * 2 - out_lat
        pad_w = in_lon * 2 - out_lon

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x = x[
            :, pad_top : 2 * in_lat - pad_bottom, pad_left : 2 * in_lon - pad_right, :
        ]
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
        x = self.norm(x)
        x = self.linear2(x)
        return x


class DownSample_2D(nn.Module):
    """
    Down-sampling operation

    Args:
        in_dim (int): Number of input channels.
        input_resolution (tuple[int]): [latitude, longitude]
        output_resolution (tuple[int]): [latitude, longitude]
    """

    def __init__(self, in_dim, input_resolution, output_resolution):
        super().__init__()
        self.linear = nn.Linear(in_dim * 4, in_dim * 2, bias=False)
        self.norm = nn.LayerNorm(4 * in_dim)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution

        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution

        h_pad = out_lat * 2 - in_lat
        w_pad = out_lon * 2 - in_lon

        pad_top = h_pad // 2
        pad_bottom = h_pad - pad_top

        pad_left = w_pad // 2
        pad_right = w_pad - pad_left

        self.pad = ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        in_lat, in_lon = self.input_resolution
        out_lat, out_lon = self.output_resolution
        x = x.reshape(B, in_lat, in_lon, C)

        # Padding the input to facilitate downsampling
        x = self.pad(x.permute(0, -1, 1, 2)).permute(0, 2, 3, 1)
        x = x.reshape(B, out_lat, 2, out_lon, 2, C).permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, out_lat * out_lon, 4 * C)

        x = self.norm(x)
        x = self.linear(x)
        return x


class EarthAttention3D(nn.Module):
    """
    3D window attention with earth position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): [pressure levels, latitude, longitude]
        window_size (tuple[int]): [pressure levels, latitude, longitude]
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wpl, Wlat, Wlon
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.type_of_windows = (input_resolution[0] // window_size[0]) * (
            input_resolution[1] // window_size[1]
        )

        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size[0] ** 2)
                * (window_size[1] ** 2)
                * (window_size[2] * 2 - 1),
                self.type_of_windows,
                num_heads,
            )
        )  # Wpl**2 * Wlat**2 * Wlon*2-1, Npl//Wpl * Nlat//Wlat, nH

        earth_position_index = get_earth_position_index(
            window_size
        )  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
        self.register_buffer("earth_position_index", earth_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.earth_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x: input features with shape of (B * num_lon, num_pl*num_lat, N, C)
            mask: (0/-inf) mask with shape of (num_lon, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon)
        """
        B_, nW_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.type_of_windows,
            -1,
        )  # Wpl*Wlat*Wlon, Wpl*Wlat*Wlon, num_pl*num_lat, nH
        earth_position_bias = earth_position_bias.permute(
            3, 2, 0, 1
        ).contiguous()  # nH, num_pl*num_lat, Wpl*Wlat*Wlon, Wpl*Wlat*Wlon
        attn = attn + earth_position_bias.unsqueeze(0)

        if mask is not None:
            nLon = mask.shape[0]
            attn = attn.view(
                B_ // nLon, nLon, self.num_heads, nW_, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, nW_, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EarthAttention2D(nn.Module):
    """
    2D window attention with earth position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): [latitude, longitude]
        window_size (tuple[int]): [latitude, longitude]
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        input_resolution,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wlat, Wlon
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.type_of_windows = input_resolution[0] // window_size[0]

        self.earth_position_bias_table = nn.Parameter(
            torch.zeros(
                (window_size[0] ** 2) * (window_size[1] * 2 - 1),
                self.type_of_windows,
                num_heads,
            )
        )  # Wlat**2 * Wlon*2-1, Nlat//Wlat, nH

        earth_position_index = get_earth_position_index_2d(
            window_size
        )  # Wlat*Wlon, Wlat*Wlon
        self.register_buffer("earth_position_index", earth_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.earth_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask=None):
        """
        Args:
            x: input features with shape of (B * num_lon, num_lat, N, C)
            mask: (0/-inf) mask with shape of (num_lon, num_lat, Wlat*Wlon, Wlat*Wlon)
        """
        B_, nW_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, nW_, N, 3, self.num_heads, C // self.num_heads)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        earth_position_bias = self.earth_position_bias_table[
            self.earth_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            self.type_of_windows,
            -1,
        )  # Wlat*Wlon, Wlat*Wlon, num_lat, nH
        earth_position_bias = earth_position_bias.permute(
            3, 2, 0, 1
        ).contiguous()  # nH, num_lat, Wlat*Wlon, Wlat*Wlon
        attn = attn + earth_position_bias.unsqueeze(0)

        if mask is not None:
            nLon = mask.shape[0]
            attn = attn.view(
                B_ // nLon, nLon, self.num_heads, nW_, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, nW_, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 3, 1, 4).reshape(B_, nW_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Transformer3DBlock(nn.Module):
    """
    3D Transformer Block
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size [pressure levels, latitude, longitude].
        shift_size (tuple[int]): Shift size for SW-MSA [pressure levels, latitude, longitude].
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=None,
        shift_size=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        window_size = (2, 6, 12) if window_size is None else window_size
        shift_size = (1, 3, 6) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        padding = get_pad3d(input_resolution, window_size)
        self.pad = ZeroPad3d(padding)

        pad_resolution = list(input_resolution)
        pad_resolution[0] += padding[-1] + padding[-2]
        pad_resolution[1] += padding[2] + padding[3]
        pad_resolution[2] += padding[0] + padding[1]

        self.attn = EarthAttention3D(
            dim=dim,
            input_resolution=pad_resolution,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        shift_pl, shift_lat, shift_lon = self.shift_size
        self.roll = shift_pl and shift_lon and shift_lat

        if self.roll:
            attn_mask = get_shift_window_mask(pad_resolution, window_size, shift_size)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor):
        Pl, Lat, Lon = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Pl, Lat, Lon, C)

        # start pad
        x = self.pad(x.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)

        _, Pl_pad, Lat_pad, Lon_pad, _ = x.shape

        shift_pl, shift_lat, shift_lon = self.shift_size
        if self.roll:
            shifted_x = torch.roll(
                x, shifts=(-shift_pl, -shift_lat, -shift_lat), dims=(1, 2, 3)
            )
            x_windows = window_partition(shifted_x, self.window_size)
            # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)
            # B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C

        win_pl, win_lat, win_lon = self.window_size
        x_windows = x_windows.view(
            x_windows.shape[0], x_windows.shape[1], win_pl * win_lat * win_lon, C
        )
        # B*num_lon, num_pl*num_lat, win_pl*win_lat*win_lon, C

        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # B*num_lon, num_pl*num_lat, win_pl*win_lat*win_lon, C

        attn_windows = attn_windows.view(
            attn_windows.shape[0], attn_windows.shape[1], win_pl, win_lat, win_lon, C
        )

        if self.roll:
            shifted_x = window_reverse(
                attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad
            )
            # B * Pl * Lat * Lon * C
            x = torch.roll(
                shifted_x, shifts=(shift_pl, shift_lat, shift_lon), dims=(1, 2, 3)
            )
        else:
            shifted_x = window_reverse(
                attn_windows, self.window_size, Pl_pad, Lat_pad, Lon_pad
            )
            x = shifted_x

        # crop, end pad
        x = crop3d(x.permute(0, 4, 1, 2, 3), self.input_resolution).permute(
            0, 2, 3, 4, 1
        )

        x = x.reshape(B, Pl * Lat * Lon, C)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Transformer2DBlock(nn.Module):
    """
    2D Transformer Block
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size [latitude, longitude].
        shift_size (tuple[int]): Shift size for SW-MSA [latitude, longitude].
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=None,
        shift_size=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        window_size = (6, 12) if window_size is None else window_size
        shift_size = (3, 6) if shift_size is None else shift_size
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        padding = get_pad2d(input_resolution, window_size)
        self.pad = ZeroPad2d(padding)

        pad_resolution = list(input_resolution)
        pad_resolution[0] += padding[2] + padding[3]
        pad_resolution[1] += padding[0] + padding[1]

        self.attn = EarthAttention2D(
            dim=dim,
            input_resolution=pad_resolution,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        shift_lat, shift_lon = self.shift_size
        self.roll = shift_lon and shift_lat

        if self.roll:
            attn_mask = get_shift_window_mask_2d(
                pad_resolution, window_size, shift_size
            )
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor):
        Lat, Lon = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Lat, Lon, C)

        # start pad
        x = self.pad(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        _, Lat_pad, Lon_pad, _ = x.shape

        shift_lat, shift_lon = self.shift_size
        if self.roll:
            shifted_x = torch.roll(x, shifts=(-shift_lat, -shift_lat), dims=(1, 2))
            x_windows = window_partition_2d(shifted_x, self.window_size)
            # B*num_lon, num_lat, win_lat, win_lon, C
        else:
            shifted_x = x
            x_windows = window_partition_2d(shifted_x, self.window_size)
            # B*num_lon, num_lat, win_lat, win_lon, C

        win_lat, win_lon = self.window_size
        x_windows = x_windows.view(
            x_windows.shape[0], x_windows.shape[1], win_lat * win_lon, C
        )
        # B*num_lon, num_lat, win_lat*win_lon, C

        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # B*num_lon, num_lat, win_lat*win_lon, C

        attn_windows = attn_windows.view(
            attn_windows.shape[0], attn_windows.shape[1], win_lat, win_lon, C
        )

        if self.roll:
            shifted_x = window_reverse_2d(
                attn_windows, self.window_size, Lat_pad, Lon_pad
            )
            # B * Lat * Lon * C
            x = torch.roll(shifted_x, shifts=(shift_lat, shift_lon), dims=(1, 2))
        else:
            shifted_x = window_reverse_2d(
                attn_windows, self.window_size, Lat_pad, Lon_pad
            )
            x = shifted_x

        # crop, end pad
        x = crop2d(x.permute(0, 3, 1, 2), self.input_resolution).permute(0, 2, 3, 1)

        x = x.reshape(B, Lat * Lon, C)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class FuserLayer(nn.Module):
    """A basic 3D Transformer layer for one stage

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList(
            [
                Transformer3DBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if i % 2 == 0 else None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, Sequence)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class EncoderLayer(nn.Module):
    """A 2D Transformer Encoder Module for one stage

    Args:
        img_size (tuple[int]): image size(Lat, Lon).
        patch_size (tuple[int]): Patch token size of Patch Embedding.
        in_chans (int): number of input channels of Patch Embedding.
        dim (int): Number of input channels of transformer.
        input_resolution (tuple[int]): Input resolution for transformer before downsampling.
        middle_resolution (tuple[int]): Input resolution for transformer after downsampling.
        depth (int): Number of blocks for transformer before downsampling.
        depth_middle (int): Number of blocks for transformer after downsampling.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(
        self,
        img_size,
        patch_size,
        in_chans,
        dim,
        input_resolution,
        middle_resolution,
        depth,
        depth_middle,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.depth_middle = depth_middle
        if isinstance(drop_path, Sequence):
            drop_path_middle = drop_path[depth:]
            drop_path = drop_path[:depth]
        else:
            drop_path_middle = drop_path
        if isinstance(num_heads, Sequence):
            num_heads_middle = num_heads[1]
            num_heads = num_heads[0]
        else:
            num_heads_middle = num_heads

        self.patchembed2d = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=dim,
        )
        self.blocks = nn.ModuleList(
            [
                Transformer2DBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0) if i % 2 == 0 else None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, Sequence)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.downsample = DownSample_2D(
            in_dim=dim,
            input_resolution=input_resolution,
            output_resolution=middle_resolution,
        )

        self.blocks_middle = nn.ModuleList(
            [
                Transformer2DBlock(
                    dim=dim * 2,
                    input_resolution=middle_resolution,
                    num_heads=num_heads_middle,
                    window_size=window_size,
                    shift_size=(0, 0) if i % 2 == 0 else None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_middle[i]
                    if isinstance(drop_path_middle, Sequence)
                    else drop_path_middle,
                    norm_layer=norm_layer,
                )
                for i in range(depth_middle)
            ]
        )

    def forward(self, x):
        x = self.patchembed2d(x)
        B, C, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        skip = x.reshape(B, Lat, Lon, C)
        x = self.downsample(x)
        for blk in self.blocks_middle:
            x = blk(x)
        return x, skip


class DecoderLayer(nn.Module):
    """A 2D Transformer Decoder Module for one stage

    Args:
        img_size (tuple[int]): image size(Lat, Lon).
        patch_size (tuple[int]): Patch token size of Patch Recovery.
        out_chans (int): number of output channels of Patch Recovery.
        dim (int): Number of input channels of transformer.
        output_resolution (tuple[int]): Input resolution for transformer after upsampling.
        middle_resolution (tuple[int]): Input resolution for transformer before upsampling.
        depth (int): Number of blocks for transformer after upsampling.
        depth_middle (int): Number of blocks for transformer before upsampling.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(
        self,
        img_size,
        patch_size,
        out_chans,
        dim,
        output_resolution,
        middle_resolution,
        depth,
        depth_middle,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.out_chans = out_chans
        self.dim = dim
        self.output_resolution = output_resolution
        self.depth = depth
        self.depth_middle = depth_middle
        if isinstance(drop_path, Sequence):
            drop_path_middle = drop_path[depth:]
            drop_path = drop_path[:depth]
        else:
            drop_path_middle = drop_path
        if isinstance(num_heads, Sequence):
            num_heads_middle = num_heads[1]
            num_heads = num_heads[0]
        else:
            num_heads_middle = num_heads

        self.blocks_middle = nn.ModuleList(
            [
                Transformer2DBlock(
                    dim=dim * 2,
                    input_resolution=middle_resolution,
                    num_heads=num_heads_middle,
                    window_size=window_size,
                    shift_size=(0, 0) if i % 2 == 0 else None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_middle[i]
                    if isinstance(drop_path_middle, Sequence)
                    else drop_path_middle,
                    norm_layer=norm_layer,
                )
                for i in range(depth_middle)
            ]
        )

        self.upsample = UpSample_2D(
            in_dim=dim * 2,
            out_dim=dim,
            input_resolution=middle_resolution,
            output_resolution=output_resolution,
        )

        self.blocks = nn.ModuleList(
            [
                Transformer2DBlock(
                    dim=dim,
                    input_resolution=output_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0) if i % 2 == 0 else None,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, Sequence)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.patchrecovery2d = PatchRecovery2D(img_size, patch_size, 2 * dim, out_chans)

    def forward(self, x, skip):
        B, Lat, Lon, C = skip.shape
        for blk in self.blocks_middle:
            x = blk(x)
        x = self.upsample(x)
        for blk in self.blocks:
            x = blk(x)
        output = torch.concat([x, skip.reshape(B, -1, C)], dim=-1)
        output = output.transpose(1, 2).reshape(B, -1, Lat, Lon)
        output = self.patchrecovery2d(output)
        return output


@dataclass
class MetaData(ModelMetaData):
    name: str = "Fengwu"
    # Optimization
    jit: bool = False  # ONNX Ops Conflict
    cuda_graphs: bool = True
    amp: bool = True
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class Fengwu(Module):
    """
    FengWu PyTorch impl of: `FengWu: Pushing the Skillful Global Medium-range Weather Forecast beyond 10 Days Lead`
    - https://arxiv.org/pdf/2304.02948.pdf

    Args:
        img_size: Image size(Lat, Lon). Default: (721,1440)
        pressure_level: Number of pressure_level. Default: 37
        embed_dim (int): Patch embedding dimension. Default: 192
        patch_size (tuple[int]): Patch token size. Default: (4,4)
        num_heads (tuple[int]): Number of attention heads in different layers.
        window_size (tuple[int]): Window size.
    """

    def __init__(
        self,
        img_size=(721, 1440),
        pressure_level=37,
        embed_dim=192,
        patch_size=(4, 4),
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
    ):
        super().__init__(meta=MetaData())
        drop_path = np.linspace(0, 0.2, 8).tolist()
        drop_path_fuser = [0.2] * 6
        resolution_down1 = (
            math.ceil(img_size[0] / patch_size[0]),
            math.ceil(img_size[1] / patch_size[1]),
        )
        resolution_down2 = (
            math.ceil(resolution_down1[0] / 2),
            math.ceil(resolution_down1[1] / 2),
        )
        resolution = (resolution_down1, resolution_down2)
        self.encoder_surface = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=4,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.encoder_z = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=pressure_level,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.encoder_r = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=pressure_level,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.encoder_u = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=pressure_level,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.encoder_v = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=pressure_level,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.encoder_t = EncoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=pressure_level,
            dim=embed_dim,
            input_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )

        self.fuser = FuserLayer(
            dim=embed_dim * 2,
            input_resolution=(6, resolution[1][0], resolution[1][1]),
            depth=6,
            num_heads=num_heads[1],
            window_size=window_size,
            drop_path=drop_path_fuser,
        )

        self.decoder_surface = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=4,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.decoder_z = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=pressure_level,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.decoder_r = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=pressure_level,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.decoder_u = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=pressure_level,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.decoder_v = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=pressure_level,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )
        self.decoder_t = DecoderLayer(
            img_size=img_size,
            patch_size=patch_size,
            out_chans=pressure_level,
            dim=embed_dim,
            output_resolution=resolution[0],
            middle_resolution=resolution[1],
            depth=2,
            depth_middle=6,
            num_heads=num_heads[:2],
            window_size=window_size[1:],
            drop_path=drop_path,
        )

    def forward(self, surface, z, r, u, v, t):
        """
        Args:
            surface (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=4.
            z (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            r (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            u (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            v (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
            t (torch.Tensor): 2D n_lat=721, n_lon=1440, chans=37.
        """
        surface, skip_surface = self.encoder_surface(surface)
        z, skip_z = self.encoder_z(z)
        r, skip_r = self.encoder_r(r)
        u, skip_u = self.encoder_u(u)
        v, skip_v = self.encoder_v(v)
        t, skip_t = self.encoder_t(t)

        x = torch.concat(
            [
                surface.unsqueeze(1),
                z.unsqueeze(1),
                r.unsqueeze(1),
                u.unsqueeze(1),
                v.unsqueeze(1),
                t.unsqueeze(1),
            ],
            dim=1,
        )
        B, PL, L_SIZE, C = x.shape
        x = x.reshape(B, -1, C)
        x = self.fuser(x)

        x = x.reshape(B, PL, L_SIZE, C)
        surface, z, r, u, v, t = (
            x[:, 0, :, :],
            x[:, 1, :, :],
            x[:, 2, :, :],
            x[:, 3, :, :],
            x[:, 4, :, :],
            x[:, 5, :, :],
        )

        surface = self.decoder_surface(surface, skip_surface)
        z = self.decoder_z(z, skip_z)
        r = self.decoder_r(r, skip_r)
        u = self.decoder_u(u, skip_u)
        v = self.decoder_v(v, skip_v)
        t = self.decoder_t(t, skip_t)
        return surface, z, r, u, v, t
