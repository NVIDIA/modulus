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

from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO(akamenev): migration
# from fft_conv_pytorch import fft_conv
from torch import Tensor

from src.networks.point_feature_ops import GridFeatures, GridFeaturesMemoryFormat

from .base_model import BaseModule


def to_ntuple(val: Union[int, Iterable[int]], n: int) -> Tuple[int, ...]:
    """Casts to a tuple with length 'n'.  Useful for automatically computing the padding and stride
    for convolutions, where users may only provide an integer.

    Args:
        val: (Union[int, Iterable[int]]) Value to cast into a tuple.
        n: (int) Desired length of the tuple

    Returns:
        (Tuple[int, ...]) Tuple of length 'n'
    """
    if isinstance(val, Iterable):
        out = tuple(val)
        if len(out) == n:
            return out
        else:
            raise ValueError(f"Cannot cast tuple of length {len(out)} to length {n}.")
    else:
        return n * (val,)


class GridFeaturePadToMatch(nn.Module):
    """GridFeaturePadToMatch."""

    def forward(self, ref_grid: GridFeatures, x_grid: GridFeatures) -> GridFeatures:
        assert ref_grid.memory_format == x_grid.memory_format
        assert x_grid.memory_format != GridFeaturesMemoryFormat.b_x_y_z_c

        if x_grid.memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
            x = x_grid.batch_features
            # get height, width, depth of x
            height, width, depth = x_grid.resolution
            # get height, width, depth of ref
            ref_height, ref_width, ref_depth = ref_grid.resolution
            pad_height = ref_height - height
            pad_width = ref_width - width
            pad_depth = ref_depth - depth
            # If pad_XXX is negative, crop. Otherwise, pad 0 at the end
            if pad_height < 0 or pad_width < 0 or pad_depth < 0:
                x = x[:, :, :ref_height, :ref_width, :ref_depth]
            if pad_height > 0 or pad_width > 0 or pad_depth > 0:
                x = F.pad(x, (0, pad_depth, 0, pad_width, 0, pad_height), "constant", 0)
            return GridFeatures(
                vertices=ref_grid.vertices,
                features=x,
                memory_format=x_grid.memory_format,
                grid_shape=ref_grid.grid_shape,
                num_channels=x_grid.num_channels,
            )

        # For planes
        else:
            # Make x to have the same dimension as ref
            x = x_grid.features
            ref = ref_grid.features
            height, width = x.shape[2], x.shape[3]
            ref_height, ref_width = ref.shape[2], ref.shape[3]
            pad_height = ref_height - height
            pad_width = ref_width - width
            # If pad_XXX is negative, crop. Otherwise, pad 0 at the end
            if pad_height < 0 or pad_width < 0:
                x = x[:, :, :ref_height, :ref_width]
            if pad_height > 0 or pad_width > 0:
                x = F.pad(x, (0, pad_width, 0, pad_height), "constant", 0)
            return GridFeatures(
                vertices=ref_grid.vertices,
                features=x,
                memory_format=x_grid.memory_format,
                grid_shape=ref_grid.grid_shape,
                num_channels=x_grid.num_channels,
            )


class GridFeatureConv2d(nn.Module):
    """GridFeatureConv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        compressed_spatial_dim: int = 1,
        stride: Optional[int] = 1,
        up_stride: Optional[int] = None,
        padding: Optional[int] = None,
        output_padding: Optional[int] = None,
        bias: bool = True,
    ):
        """
        Args:
            in_channels: (int) Number of input channels
            out_channels: (int) Number of output channels
            kernel_size: (int) Size of the convolutional kernel
            compressed_spatial_dim: (int) Number of spatial dimensions to compress
                into a single dimension.  For example, if the input is a 3D grid with
                shape (N, C, X, Y, Z), then the output will be a 2D grid with shape
                (N, C, X*Y*Z).  This is useful for reducing the memory footprint of
                the convolutional kernel.
            stride: (Optional[int]) Stride of the convolutional kernel
            up_stride: (Optional[int]) Upsampling stride.  If provided, the
                convolutional kernel will be replaced with a transposed convolution
                with the given stride.
            padding: (Optional[int]) Padding of the convolutional kernel
            output_padding: (Optional[int]) Output padding of the transposed
                convolutional kernel
            bias: (bool) Whether or not to include a bias term
        """
        super().__init__()
        if up_stride is None:
            self.conv = nn.Conv2d(
                in_channels=in_channels * compressed_spatial_dim,
                out_channels=out_channels * compressed_spatial_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding if padding is not None else (kernel_size - 1) // 2,
                bias=bias,
            )
        else:
            # H_out ​=(H_in​−1)×stride[0]−2×padding[0]+output_padding[0]+1
            self.conv = nn.ConvTranspose2d(
                in_channels=in_channels * compressed_spatial_dim,
                out_channels=out_channels * compressed_spatial_dim,
                kernel_size=kernel_size,
                stride=up_stride,
                output_padding=output_padding if output_padding is not None else 0,
                bias=bias,
            )
        self.out_channels = out_channels

    def forward(self, grid_features: GridFeatures) -> GridFeatures:
        assert (
            grid_features.memory_format != GridFeaturesMemoryFormat.b_x_y_z_c
            and grid_features.memory_format != GridFeaturesMemoryFormat.b_c_x_y_z
        )
        plane_view = grid_features.features
        plane_view = self.conv(plane_view)
        # TODO: stride != 1 vertices
        out_grid_features = GridFeatures.from_conv_output(
            plane_view,
            grid_features.vertices,
            grid_features.memory_format,
            grid_features.grid_shape,
            self.out_channels,
        )
        return out_grid_features


class GridFeatureFFTConv2d(nn.Module):
    """GridFeatureFFTConv2d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Iterable[int]],
        compressed_spatial_dim: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = to_ntuple(kernel_size, 2)
        weight = torch.zeros(
            out_channels * compressed_spatial_dim,
            in_channels * compressed_spatial_dim,
            *kernel_size,
        )
        weight[0, 0] = 0.25  # 2D FFT of identity
        self.weight = nn.Parameter(weight)

    def forward(self, grid_features: GridFeatures) -> GridFeatures:
        assert (
            grid_features.memory_format != GridFeaturesMemoryFormat.b_x_y_z_c
            and grid_features.memory_format != GridFeaturesMemoryFormat.b_c_x_y_z
        )
        plane_view = grid_features.features
        # if ndim is 3, expand batch dim
        if plane_view.ndim == 3:
            plane_view = plane_view.unsqueeze(0)
        out_features = fft_conv(
            plane_view,
            self.weight,
            padding="same",
        )
        out_grid_features = GridFeatures.from_conv_output(
            out_features.squeeze(0),  # remove batch dim
            grid_features.vertices,
            grid_features.memory_format,
            grid_features.grid_shape,
            self.out_channels,
        )
        return out_grid_features


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm2d."""

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class LayerNorm3d(nn.LayerNorm):
    """LayerNorm3d."""

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class GridFeatureTransform(BaseModule):
    """GridFeatureTransform."""

    def __init__(self, transform: nn.Module) -> None:
        super().__init__()
        self.feature_transform = transform

    def forward(self, grid_features: GridFeatures) -> GridFeatures:
        assert grid_features.memory_format != GridFeaturesMemoryFormat.b_x_y_z_c

        if grid_features.memory_format == GridFeaturesMemoryFormat.b_c_x_y_z:
            batch_view = grid_features.batch_features
            assert batch_view.ndim == 5
        else:
            batch_view = grid_features.features
            if batch_view.ndim == 3:
                assert False, "This should not happen."
                batch_view = batch_view.unsqueeze(0)

        batch_view = self.feature_transform(batch_view)
        out_grid_features = GridFeatures.from_conv_output(
            batch_view,
            grid_features.vertices,
            grid_features.memory_format,
            grid_features.grid_shape,
            grid_features.num_channels,
        )
        return out_grid_features


class GridFeatureConv2dBlock(nn.Module):
    """GridFeatureConv2dBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        compressed_spatial_dim: int = 1,
        stride: Optional[int] = 1,
        up_stride: Optional[int] = None,
        apply_nonlinear_at_end: bool = True,
    ):
        super().__init__()

        self.conv1 = GridFeatureConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size if up_stride is None else up_stride,
            stride=stride,
            up_stride=up_stride,
            compressed_spatial_dim=compressed_spatial_dim,
        )
        self.conv2 = GridFeatureConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            compressed_spatial_dim=compressed_spatial_dim,
            up_stride=None,
        )
        self.norm1 = GridFeatureTransform(
            LayerNorm2d(out_channels * compressed_spatial_dim)
        )
        self.norm2 = GridFeatureTransform(
            LayerNorm2d(out_channels * compressed_spatial_dim)
        )
        self.apply_nonlinear_at_end = apply_nonlinear_at_end

        if up_stride is None:
            if stride == 1 and in_channels == out_channels:
                self.shortcut = nn.Identity()
            elif stride == 1:
                self.shortcut = GridFeatureConv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    compressed_spatial_dim=compressed_spatial_dim,
                )
            elif stride > 1:
                self.shortcut = GridFeatureConv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=stride,
                    stride=stride,
                    compressed_spatial_dim=compressed_spatial_dim,
                )
        else:
            self.shortcut = GridFeatureConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=up_stride,
                up_stride=up_stride,
                compressed_spatial_dim=compressed_spatial_dim,
            )
        self.pad_to_match = GridFeaturePadToMatch()
        self.nonlinear = GridFeatureTransform(nn.GELU())

    def forward(self, grid_features: GridFeatures) -> GridFeatures:
        out = self.conv1(grid_features)
        out = self.nonlinear(self.norm1(out))
        out = self.norm2(self.conv2(out))
        shortcut = self.shortcut(grid_features)
        shortcut = self.pad_to_match(out, shortcut)
        out = out + shortcut
        if self.apply_nonlinear_at_end:
            out = self.nonlinear(out)
        return out


class GridFeatureMemoryFormatConverter(BaseModule):
    """GridFeatureMemoryFormatConverter."""

    def __init__(self, memory_format: GridFeaturesMemoryFormat) -> None:
        super().__init__()
        self.memory_format = memory_format

    def __repr__(self):
        return f"GridFeatureMemoryFormatConverter(memory_format={self.memory_format})"

    def forward(self, grid_features: GridFeatures) -> GridFeatures:
        return grid_features.to(memory_format=self.memory_format)


# 3D Operations
class GridFeatureConv3d(nn.Module):
    """GridFeatureConv3d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Optional[int] = 1,
        up_stride: Optional[int] = None,
        padding: Optional[int] = None,
        output_padding: Optional[int] = None,
        bias: bool = True,
    ):
        """
        Args:
            in_channels: (int) Number of input channels
            out_channels: (int) Number of output channels
            kernel_size: (int) Size of the convolutional kernel
            stride: (Optional[int]) Stride of the convolutional kernel
            up_stride: (Optional[int]) Upsampling stride.  If provided, the
                convolutional kernel will be replaced with a transposed convolution
                with the given stride.
            padding: (Optional[int]) Padding of the convolutional kernel
            output_padding: (Optional[int]) Output padding of the transposed
                convolutional kernel
            bias: (bool) Whether or not to include a bias term
        """
        super().__init__()
        if up_stride is None:
            self.conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding if padding is not None else (kernel_size - 1) // 2,
                bias=bias,
            )
        else:
            # H_out ​=(H_in​−1)×stride[0]−2×padding[0]+output_padding[0]+1
            self.conv = nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=up_stride,
                output_padding=output_padding if output_padding is not None else 0,
                bias=bias,
            )
        self.out_channels = out_channels

    def forward(self, grid_features: GridFeatures) -> GridFeatures:
        assert grid_features.memory_format == GridFeaturesMemoryFormat.b_c_x_y_z
        plane_view = grid_features.batch_features
        plane_view = self.conv(plane_view)
        # TODO: stride != 1 vertices
        out_grid_features = GridFeatures.from_conv_output(
            plane_view.squeeze(0),
            grid_features.vertices,
            grid_features.memory_format,
            grid_features.grid_shape,
            self.out_channels,
        )
        return out_grid_features


class GridFeatureConv3dBlock(nn.Module):
    """GridFeatureConv3dBlock."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: Optional[int] = 1,
        up_stride: Optional[int] = None,
    ):
        super().__init__()

        self.conv1 = GridFeatureConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size if up_stride is None else up_stride,
            stride=stride,
            up_stride=up_stride,
        )
        self.conv2 = GridFeatureConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            up_stride=None,
        )
        self.norm1 = GridFeatureTransform(LayerNorm3d(out_channels))
        self.norm2 = GridFeatureTransform(LayerNorm3d(out_channels))

        if up_stride is None:
            if stride == 1 and in_channels == out_channels:
                self.shortcut = nn.Identity()
            elif stride == 1:
                self.shortcut = GridFeatureConv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                )
            elif stride > 1:
                self.shortcut = GridFeatureConv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=stride,
                    stride=stride,
                )
        else:
            self.shortcut = GridFeatureConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=up_stride,
                up_stride=up_stride,
            )
        self.pad_to_match = GridFeaturePadToMatch()
        self.nonlinear = GridFeatureTransform(nn.GELU())

    def forward(self, grid_features: GridFeatures) -> GridFeatures:
        assert grid_features.memory_format == GridFeaturesMemoryFormat.b_c_x_y_z
        out = self.conv1(grid_features)
        out = self.nonlinear(self.norm1(out))
        out = self.norm2(self.conv2(out))
        shortcut = self.shortcut(grid_features)
        shortcut = self.pad_to_match(out, shortcut)
        return self.nonlinear(out + shortcut)
