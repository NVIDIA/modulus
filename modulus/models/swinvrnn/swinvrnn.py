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

from dataclasses import dataclass

import torch
from timm.layers import to_2tuple
from timm.models.swin_transformer import SwinTransformerStage
from torch import nn

from ..meta import ModelMetaData
from ..module import Module
from ..pangu.utils import get_pad2d


class CubeEmbedding(nn.Module):
    """
    3D Image Cube Embedding
    Args:
        img_size (tuple[int]): Image size [T, Lat, Lon].
        patch_size (tuple[int]): Patch token size [T, Lat, Lon].
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: torch.nn.LayerNorm
    """

    def __init__(
        self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        ]

        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, T, Lat, Lon = x.shape
        x = self.proj(x).reshape(B, self.embed_dim, -1).transpose(1, 2)  # B T*Lat*Lon C
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *self.patches_resolution)
        return x


class DownBlock(nn.Module):
    """
    Spatial Down-sampling block
    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        num_groups (int): Number of groups to separate the channels into for group normalization.
        num_residuals (int, optinal): Number of Conv2d operator. Default: 2
    """

    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chans, out_chans, kernel_size=(3, 3), stride=2, padding=1
        )

        blk = []
        for i in range(num_residuals):
            blk.append(
                nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1)
            )
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        return x + shortcut


class UpBlock(nn.Module):
    """
    Spatial Up-sampling block
    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        num_groups (int): Number of groups to separate the channels into for group normalization.
        num_residuals (int, optinal): Number of Conv2d operator. Default: 2
    """

    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)

        blk = []
        for i in range(num_residuals):
            blk.append(
                nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1)
            )
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        return x + shortcut


class ConvBlock(nn.Module):
    """
    Conv2d block
    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        num_groups (int): Number of groups to separate the channels into for group normalization.
        num_residuals (int, optinal): Number of Conv2d operator. Default: 2
    """

    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chans, out_chans, kernel_size=(3, 3), stride=1, padding=1
        )

        blk = []
        for i in range(num_residuals):
            blk.append(
                nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1)
            )
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        return x + shortcut


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


class SwinTransformer(nn.Module):
    """Swin Transformer
    Args:
        embed_dim (int): Patch embedding dimension.
        input_resolution (tuple[int]): Lat, Lon.
        num_heads (int): Number of attention heads in different layers.
        window_size (int | tuple[int]): Window size.
        depth (int): Number of blocks.
    """

    def __init__(self, embed_dim, input_resolution, num_heads, window_size, depth):
        super().__init__()
        window_size = to_2tuple(window_size)
        padding = get_pad2d(input_resolution, to_2tuple(window_size))
        padding_left, padding_right, padding_top, padding_bottom = padding
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)
        input_resolution = list(input_resolution)
        input_resolution[0] = input_resolution[0] + padding_top + padding_bottom
        input_resolution[1] = input_resolution[1] + padding_left + padding_right
        self.layer = SwinTransformerStage(
            dim=embed_dim,
            out_dim=embed_dim,
            input_resolution=input_resolution,
            depth=depth,
            downsample=None,
            num_heads=num_heads,
            window_size=window_size,
        )

    def forward(self, x):
        B, C, Lat, Lon = x.shape
        padding_left, padding_right, padding_top, padding_bottom = self.padding

        # pad
        x = self.pad(x)
        _, _, pad_lat, pad_lon = x.shape

        x = x.permute(0, 2, 3, 1)  # B Lat Lon C
        x = self.layer(x)
        x = x.permute(0, 3, 1, 2)

        # crop
        x = x[
            :,
            :,
            padding_top : pad_lat - padding_bottom,
            padding_left : pad_lon - padding_right,
        ]

        return x


@dataclass
class MetaData(ModelMetaData):
    name: str = "SwinRNN"
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


class SwinRNN(Module):
    """
    Implementation of SwinRNN https://arxiv.org/abs/2205.13158
    Args:
        img_size (Sequence[int], optional): Image size [T, Lat, Lon].
        patch_size (Sequence[int], optional): Patch token size [T, Lat, Lon].
        in_chans (int, optional): number of input channels.
        out_chans (int, optional): number of output channels.
        embed_dim (int, optional): number of embed channels.
        num_groups (Sequence[int] | int, optional): number of groups to separate the channels into.
        num_heads (int, optional): Number of attention heads.
        window_size (int | tuple[int], optional): Local window size.
    """

    def __init__(
        self,
        img_size=(2, 721, 1440),
        patch_size=(2, 4, 4),
        in_chans=70,
        out_chans=70,
        embed_dim=1536,
        num_groups=32,
        num_heads=8,
        window_size=7,
    ):
        super().__init__(meta=MetaData())
        input_resolution = img_size[1:]
        self.cube_embedding = CubeEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.swin_block1 = SwinTransformer(
            embed_dim, input_resolution, num_heads, window_size, depth=2
        )
        self.down1 = DownBlock(embed_dim, embed_dim, num_groups)
        self.down1x = DownBlock(in_chans, in_chans, in_chans)
        self.lin_proj1 = nn.Linear(embed_dim + in_chans, embed_dim)
        self.swin_decoder1 = SwinTransformer(
            embed_dim, input_resolution, num_heads, window_size, depth=12
        )
        input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
        self.swin_block2 = SwinTransformer(
            embed_dim, input_resolution, num_heads, window_size, depth=2
        )
        self.down2 = DownBlock(embed_dim, embed_dim, num_groups)
        self.down2x = DownBlock(in_chans, in_chans, in_chans)
        self.lin_proj2 = nn.Linear(embed_dim + in_chans, embed_dim)
        self.swin_decoder2 = SwinTransformer(
            embed_dim, input_resolution, num_heads, window_size, depth=12
        )
        input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
        self.swin_block3 = SwinTransformer(
            embed_dim, input_resolution, num_heads, window_size, depth=2
        )
        self.down3 = DownBlock(embed_dim, embed_dim, num_groups)
        self.down3x = DownBlock(in_chans, in_chans, in_chans)
        self.lin_proj3 = nn.Linear(embed_dim + in_chans, embed_dim)
        self.swin_decoder3 = SwinTransformer(
            embed_dim, input_resolution, num_heads, window_size, depth=12
        )
        input_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
        self.swin_block4 = SwinTransformer(
            embed_dim, input_resolution, num_heads, window_size, depth=2
        )
        self.lin_proj4 = nn.Linear(embed_dim + in_chans, embed_dim)
        self.swin_decoder4 = SwinTransformer(
            embed_dim, input_resolution, num_heads, window_size, depth=12
        )
        self.up3x = UpBlock(embed_dim, embed_dim, num_groups)
        self.up2x = UpBlock(embed_dim * 2, embed_dim, num_groups)
        self.up1x = UpBlock(embed_dim * 2, embed_dim, num_groups)
        self.pred = ConvBlock(embed_dim * 2, out_chans, out_chans)

        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.out_chans = out_chans
        self.img_size = img_size
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor):
        B, Cin, _, _, _ = x.shape
        _, patch_lat, patch_lon = self.patch_size
        Lat, Lon = self.input_resolution
        xT = x[:, :, -1, :, :]
        x = self.cube_embedding(x).squeeze(2)  # B C Lat Lon
        h1 = self.swin_block1(x)
        x = self.down1(h1)
        h2 = self.swin_block2(x)
        x = self.down2(h2)
        h3 = self.swin_block3(x)
        x = self.down3(h3)
        h4 = self.swin_block4(x)
        B, Cin, H, W = xT.shape
        h1_d = torch.cat(
            [xT.reshape(B, Cin, -1), h1.reshape(B, self.embed_dim, -1)], dim=1
        ).transpose(1, 2)
        h1_d = self.lin_proj1(h1_d).transpose(1, 2).reshape(B, self.embed_dim, H, W)
        h1_d = self.swin_decoder1(h1_d)
        h1 = h1 + h1_d
        x2T = self.down1x(xT)
        B, Cin, H, W = x2T.shape
        h2_d = torch.cat(
            [x2T.reshape(B, Cin, -1), h2.reshape(B, self.embed_dim, -1)], dim=1
        ).transpose(1, 2)
        h2_d = self.lin_proj2(h2_d).transpose(1, 2).reshape(B, self.embed_dim, H, W)
        h2_d = self.swin_decoder2(h2_d)
        h2 = h2 + h2_d
        x3T = self.down2x(x2T)
        B, Cin, H, W = x3T.shape
        h3_d = torch.cat(
            [x3T.reshape(B, Cin, -1), h3.reshape(B, self.embed_dim, -1)], dim=1
        ).transpose(1, 2)
        h3_d = self.lin_proj3(h3_d).transpose(1, 2).reshape(B, self.embed_dim, H, W)
        h3_d = self.swin_decoder3(h3_d)
        h3 = h3 + h3_d
        x4T = self.down3x(x3T)
        B, Cin, H, W = x4T.shape
        h4_d = torch.cat(
            [x4T.reshape(B, Cin, -1), h4.reshape(B, self.embed_dim, -1)], dim=1
        ).transpose(1, 2)
        h4_d = self.lin_proj4(h4_d).transpose(1, 2).reshape(B, self.embed_dim, H, W)
        h4_d = self.swin_decoder4(h4_d)
        h4 = h4 + h4_d
        h4_up = self.up3x(h4)
        h3_up = self.up2x(torch.cat([h3, h4_up], dim=1))
        h2_up = self.up1x(torch.cat([h2, h3_up], dim=1))
        h1_up = self.pred(torch.cat([h1, h2_up], dim=1))
        x_h1 = xT + h1_up
        return x_h1
