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

from typing import List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .ahmedbody_base import AhmedBodyBase
from .base_model import BaseModel
from .drivaer_base import DrivAerBase
from .net_utils import MLP
from .point_feature_conv import PointFeatureConv, PointFeatureConvBlock
from .point_feature_grid_conv import (
    GridFeatureConv2d,
    GridFeatureConv2dBlock,
    GridFeatureConv3d,
    GridFeatureConv3dBlock,
    GridFeatureMemoryFormatConverter,
    GridFeaturePadToMatch,
)
from .point_feature_grid_ops import (
    GridFeatureToPoint,
    GridFeatureToPointFeature,
    PointFeatureToDistanceGridFeature,
    PointFeatureToGrid,
)
from .point_feature_ops import (
    GridFeatures,
    GridFeaturesMemoryFormat,
    PointFeatures,
    VerticesToPointFeatures,
)

memory_format_to_axis_index = {
    GridFeaturesMemoryFormat.b_xc_y_z: 0,
    GridFeaturesMemoryFormat.b_yc_x_z: 1,
    GridFeaturesMemoryFormat.b_zc_x_y: 2,
    GridFeaturesMemoryFormat.b_x_y_z_c: -1,
}


class GridFeatureUNet(BaseModel):
    """GridFeatureUNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        compressed_spatial_dim: int = 1,
    ):
        BaseModel.__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for level in range(num_levels):
            down_block = GridFeatureConv2dBlock(
                in_channels=hidden_channels[level],
                out_channels=hidden_channels[level + 1],
                kernel_size=kernel_size,
                stride=2,
                compressed_spatial_dim=compressed_spatial_dim,
            )
            self.down_blocks.append(down_block)
            # Add up blocks
            up_block = GridFeatureConv2dBlock(
                in_channels=hidden_channels[level + 1],
                out_channels=hidden_channels[level],
                kernel_size=kernel_size,
                up_stride=2,
                compressed_spatial_dim=compressed_spatial_dim,
            )
            self.up_blocks.append(up_block)

        self.projection = GridFeatureConv2d(
            hidden_channels[0],
            out_channels,
            kernel_size=1,
            compressed_spatial_dim=compressed_spatial_dim,
        )

    def forward(
        self,
        grid_features: GridFeatures,
    ):
        down_grid_features = [grid_features]
        for down_block in self.down_blocks:
            out_features = down_block(down_grid_features[-1])
            down_grid_features.append(out_features)

        for level in reversed(range(self.num_levels)):
            up_grid_features = self.up_blocks[level](down_grid_features[level + 1])
            up_grid_features = up_grid_features + down_grid_features[level]
            down_grid_features[level] = up_grid_features
        return self.projection(down_grid_features[0])


class GridFeature3DUNet(BaseModel):
    """GridFeature3DUNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
    ):
        BaseModel.__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        if isinstance(num_down_blocks, int):
            num_down_blocks = [num_down_blocks] * (num_levels + 1)
        if isinstance(num_up_blocks, int):
            num_up_blocks = [num_up_blocks] * (num_levels + 1)

        # in channel to hidden
        self.in_projection = GridFeatureConv3d(
            in_channels,
            hidden_channels[0],
            kernel_size=1,
        )

        for level in range(num_levels):
            down_block = GridFeatureConv3dBlock(
                in_channels=hidden_channels[level],
                out_channels=hidden_channels[level + 1],
                kernel_size=kernel_size,
                stride=2,
            )
            down_blocks = [down_block]
            for i in range(1, num_down_blocks[level]):
                down_block = GridFeatureConv3dBlock(
                    in_channels=hidden_channels[level + 1],
                    out_channels=hidden_channels[level + 1],
                    kernel_size=kernel_size,
                    stride=1,
                )
                down_blocks.append(down_block)
            down_blocks = nn.Sequential(*down_blocks)
            self.down_blocks.append(down_blocks)
            # Add up blocks
            up_block = GridFeatureConv3dBlock(
                in_channels=hidden_channels[level + 1],
                out_channels=hidden_channels[level],
                kernel_size=kernel_size,
                up_stride=2,
            )
            up_blocks = [up_block]
            for i in range(1, num_up_blocks[level]):
                up_block = GridFeatureConv3dBlock(
                    in_channels=hidden_channels[level],
                    out_channels=hidden_channels[level],
                    kernel_size=kernel_size,
                    up_stride=1,
                )
                up_blocks.append(up_block)
            up_blocks = nn.Sequential(*up_blocks)
            self.up_blocks.append(up_blocks)

        self.pad_to_match = GridFeaturePadToMatch()

        self.projection = GridFeatureConv3d(
            hidden_channels[0],
            out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        grid_features: GridFeatures,
    ):
        grid_features = self.in_projection(grid_features)
        down_grid_features = [grid_features]
        for down_block in self.down_blocks:
            out_features = down_block(down_grid_features[-1])
            down_grid_features.append(out_features)

        for level in reversed(range(self.num_levels)):
            up_grid_features = self.up_blocks[level](down_grid_features[level + 1])
            # PAD or CROP
            padded_down_features = self.pad_to_match(
                up_grid_features, down_grid_features[level]
            )
            up_grid_features = up_grid_features + padded_down_features
            down_grid_features[level] = up_grid_features
        return self.projection(down_grid_features[0])


class PointFeatureToGrid3DUNet(GridFeature3DUNet):
    """PointFeatureToGrid3DUNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
        aabb_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        aabb_min: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        voxel_size: Optional[float] = None,
        resolution: Tuple[int, int, int] = (128, 128, 128),
        pos_encode_dist: bool = True,
        pos_encode_grid: bool = True,
        pos_encode_dim: int = 32,
    ):
        self.voxel_size = voxel_size
        if voxel_size is None:
            self.voxel_size = torch.tensor(
                [
                    (aabb_max[0] - aabb_min[0]) / resolution[0],
                    (aabb_max[1] - aabb_min[1]) / resolution[1],
                    (aabb_max[2] - aabb_min[2]) / resolution[2],
                ]
            ).min()
        point2grid = PointFeatureToDistanceGridFeature(
            grid_resolution=resolution,
            voxel_size=voxel_size,
            aabb_max=aabb_max,
            aabb_min=aabb_min,
            pos_encode_dist=pos_encode_dist,
            pos_encode_grid=pos_encode_grid,
            pos_encode_dim=pos_encode_dim,
        )

        super().__init__(
            point2grid.num_channels,
            hidden_channels[0],
            kernel_size,
            hidden_channels,
            num_levels,
            num_down_blocks=num_down_blocks,
            num_up_blocks=num_up_blocks,
        )

        self.point2grid = point2grid
        self.convert_to_c_x_y_z = GridFeatureMemoryFormatConverter(
            memory_format=GridFeaturesMemoryFormat.b_c_x_y_z
        )
        self.grid2point = GridFeatureToPointFeature(
            in_channels=hidden_channels[0], pos_encode_point=False
        )
        self.point_to_out = MLP(
            [self.grid2point.num_channels, out_channels],
            torch.nn.GELU,
        )

    def forward(
        self,
        point_features: PointFeatures,
    ):
        grid_features = self.point2grid(point_features)
        grid_features = self.convert_to_c_x_y_z(grid_features)
        # UNet
        grid_features = GridFeature3DUNet.forward(self, grid_features)
        point_features = self.grid2point(grid_features, point_features)
        return self.point_to_out(point_features.features)


class PointFeatureToGrid3DUNetAhmedBody(AhmedBodyBase, PointFeatureToGrid3DUNet):
    """PointFeatureToGrid3DUNetAhmedBody."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
        aabb_max: Tuple[float, float, float] = (0, 0.26, 0.45),
        aabb_min: Tuple[float, float, float] = (-1.4, -0.26, 0),
        resolution: Optional[Tuple[int, int, int]] = (280, 104, 90),
        voxel_size: Optional[float] = 0.05,
        pos_encode_dist: bool = True,
        pos_encode_grid: bool = True,
        pos_encode_dim: int = 32,
        # First and Last convs
        use_rel_pos: bool = True,
        knn_k: int = 16,
        # AhmedBodyBase
        use_uniformized_velocity: bool = True,
        velocity_pos_encoding: bool = False,
        normals_as_features: bool = True,
        vertices_as_features: bool = True,
        random_purturb_train: bool = True,
        vertices_purturb_range: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        PointFeatureToGrid3DUNet.__init__(
            self,
            in_channels=hidden_channels[0],
            out_channels=out_channels,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            num_down_blocks=num_down_blocks,
            num_up_blocks=num_up_blocks,
            aabb_max=aabb_max,
            aabb_min=aabb_min,
            voxel_size=voxel_size,
            resolution=resolution,
            pos_encode_dist=pos_encode_dist,
            pos_encode_grid=pos_encode_grid,
            pos_encode_dim=pos_encode_dim,
        )
        AhmedBodyBase.__init__(
            self,
            use_uniformized_velocity=use_uniformized_velocity,
            velocity_pos_encoding=velocity_pos_encoding,
            normals_as_features=normals_as_features,
            vertices_as_features=vertices_as_features,
            pos_encode_dim=pos_encode_dim,
            pos_encode_range=1,
            random_purturb_train=random_purturb_train,
            vertices_purturb_range=vertices_purturb_range,
        )

        # This property is defined in AhmedBodyBase
        in_feat = self.ahmed_input_feature_dim
        self.first_conv = PointFeatureConv(
            in_channels=in_feat,
            out_channels=hidden_channels[0],
            use_rel_pos=use_rel_pos,
            use_rel_pos_encode=False,
            reductions=["sum"],
            pos_encode_dim=pos_encode_dim,
            pos_encode_range=1,
            out_point_feature_type="same",
            neighbor_search_type="knn",
            knn_k=knn_k,
            radius=None,
        )

    def forward(
        self,
        vertices: Float[Tensor, "N 3"],
        features: Float[Tensor, "N C"],
    ) -> Tensor:
        point_feature = PointFeatures(vertices, features)
        point_feature = self.first_conv(point_feature)
        # Downsample points
        down_point_feature = point_feature.voxel_down_sample(self.voxel_size / 2)
        grid_features = self.point2grid(down_point_feature)
        grid_features = self.convert_to_c_x_y_z(grid_features)
        # UNet
        grid_features = GridFeature3DUNet.forward(self, grid_features)
        point_feature = self.grid2point(grid_features, point_feature)
        return self.point_to_out(point_feature.features)


class PointFeatureToGrid3DUNetDrivAer(DrivAerBase, PointFeatureToGrid3DUNet):
    """PointFeatureToGrid3DUNetDrivAer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
        aabb_max: Tuple[float, float, float] = (2.5, 1.5, 1.0),
        aabb_min: Tuple[float, float, float] = (-2.5, -1.5, -1.0),
        resolution: Optional[Tuple[int, int, int]] = (128, 128, 128),
        voxel_size: Optional[float] = 0.05,
        pos_encode_dist: bool = True,
        pos_encode_grid: bool = True,
        pos_encode_dim: int = 32,
    ):
        PointFeatureToGrid3DUNet.__init__(
            self,
            in_channels=hidden_channels[0],
            out_channels=out_channels,
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            num_down_blocks=num_down_blocks,
            num_up_blocks=num_up_blocks,
            aabb_max=aabb_max,
            aabb_min=aabb_min,
            voxel_size=voxel_size,
            resolution=resolution,
            pos_encode_dist=pos_encode_dist,
            pos_encode_grid=pos_encode_grid,
            pos_encode_dim=pos_encode_dim,
        )
        DrivAerBase.__init__(self)

        self.vertex_to_point_features = VerticesToPointFeatures(
            embed_dim=pos_encode_dim,
            out_features=hidden_channels[0],
            use_mlp=True,
            pos_embed_range=aabb_max[0] - aabb_min[0],
        )

    def forward(
        self,
        vertices: Float[Tensor, "N 3"],
        features: Optional[Float[Tensor, "N C"]] = None,
    ) -> Tensor:
        if features is None:
            point_features = self.vertex_to_point_features(vertices)
        else:
            point_features = PointFeatures(vertices, features)
        return PointFeatureToGrid3DUNet.forward(self, point_features)

    def data_dict_to_input(self, data_dict):
        vertices = data_dict["vertices"].squeeze(0)  # (n_in, 3)

        # center vertices
        vertices_max = vertices.max(0)[0]
        vertices_min = vertices.min(0)[0]
        vertices_center = (vertices_max + vertices_min) / 2.0
        vertices = vertices - vertices_center

        return vertices


class PointFeatureToGridUNet(BaseModel):
    """PointFeatureToGridUNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
        aabb_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        aabb_min: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        voxel_size: Optional[float] = None,
        resolution: Optional[Tuple[int, int, int]] = (128, 128, 2),
        grid_memory_format: GridFeaturesMemoryFormat = GridFeaturesMemoryFormat.b_zc_x_y,
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = True,
        pos_embed_dim: int = 32,
    ):
        BaseModel.__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        compressed_axis = memory_format_to_axis_index[grid_memory_format]
        self.compressed_spatial_dim = resolution[compressed_axis]
        self.point_feature_to_grid = PointFeatureToGrid(
            in_channels=in_channels,
            out_channels=hidden_channels[0],
            aabb_max=aabb_max,
            aabb_min=aabb_min,
            voxel_size=voxel_size,
            resolution=resolution,
            use_rel_pos=use_rel_pos,
            use_rel_pos_encode=use_rel_pos_embed,
            pos_encode_dim=pos_embed_dim,
        )
        self.convert_to_compressed = GridFeatureMemoryFormatConverter(
            memory_format=grid_memory_format
        )
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        if isinstance(num_down_blocks, int):
            num_down_blocks = [num_down_blocks] * (num_levels + 1)
        if isinstance(num_up_blocks, int):
            num_up_blocks = [num_up_blocks] * (num_levels + 1)

        for level in range(num_levels):
            down_block = [
                GridFeatureConv2dBlock(
                    in_channels=hidden_channels[level],
                    out_channels=hidden_channels[level + 1],
                    kernel_size=kernel_size,
                    stride=2,
                    compressed_spatial_dim=self.compressed_spatial_dim,
                )
            ]
            for _ in range(1, num_down_blocks[level]):
                down_block.append(
                    GridFeatureConv2dBlock(
                        in_channels=hidden_channels[level + 1],
                        out_channels=hidden_channels[level + 1],
                        kernel_size=kernel_size,
                        stride=1,
                        compressed_spatial_dim=self.compressed_spatial_dim,
                    )
                )
            down_block = nn.Sequential(*down_block)
            self.down_blocks.append(down_block)
            # Add up blocks
            up_block = [
                GridFeatureConv2dBlock(
                    in_channels=hidden_channels[level + 1],
                    out_channels=hidden_channels[level],
                    kernel_size=kernel_size,
                    up_stride=2,
                    compressed_spatial_dim=self.compressed_spatial_dim,
                )
            ]
            for _ in range(1, num_up_blocks[level]):
                up_block.append(
                    GridFeatureConv2dBlock(
                        in_channels=hidden_channels[level],
                        out_channels=hidden_channels[level],
                        kernel_size=kernel_size,
                        up_stride=1,
                        compressed_spatial_dim=self.compressed_spatial_dim,
                    )
                )
            up_block = nn.Sequential(*up_block)
            self.up_blocks.append(up_block)
        self.convert_to_orig = GridFeatureMemoryFormatConverter(
            memory_format=GridFeaturesMemoryFormat.b_x_y_z_c
        )
        self.to_point = GridFeatureToPoint(
            grid_in_channels=hidden_channels[0],
            point_in_channels=in_channels,
            out_channels=out_channels,
            aabb_max=aabb_max,
            aabb_min=aabb_min,
            use_rel_pos=use_rel_pos,
            use_rel_pos_embed=use_rel_pos_embed,
            pos_embed_dim=pos_embed_dim,
        )
        self.pad_to_match = GridFeaturePadToMatch()

    def forward(
        self,
        point_features: PointFeatures,
    ):
        grid_features = self.point_feature_to_grid(point_features)
        grid_features = self.convert_to_compressed(grid_features)
        down_grid_features = [grid_features]
        for down_block in self.down_blocks:
            out_features = down_block(down_grid_features[-1])
            down_grid_features.append(out_features)

        for level in reversed(range(self.num_levels)):
            up_grid_features = self.up_blocks[level](down_grid_features[level + 1])
            padded_down_features = self.pad_to_match(
                up_grid_features, down_grid_features[level]
            )
            up_grid_features = up_grid_features + padded_down_features
            down_grid_features[level] = up_grid_features

        grid_features = self.convert_to_orig(down_grid_features[0])
        return self.to_point(grid_features, point_features)


class PointFeatureToGridUNets(BaseModel):
    """Three PointFeatureToGridUNet with different resolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
        aabb_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        aabb_min: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        resolution_memory_format_pairs: List[
            Tuple[GridFeaturesMemoryFormat, Tuple[int, int, int]]
        ] = [
            (GridFeaturesMemoryFormat.b_xc_y_z, (2, 128, 128)),
            (GridFeaturesMemoryFormat.b_yc_x_z, (128, 2, 128)),
            (GridFeaturesMemoryFormat.b_zc_x_y, (128, 128, 2)),
        ],
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = True,
        pos_embed_dim: int = 32,
        unet_repeat: int = 1,
        unet_reduction: Literal["mul", "sum"] = "mul",
    ):
        BaseModel.__init__(self)
        self.unet_reduction = unet_reduction
        self.unet_repeat = unet_repeat

        unets_all = []
        for i in range(unet_repeat):
            unets = []
            for memory_format, resolution in resolution_memory_format_pairs:
                unet = PointFeatureToGridUNet(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    hidden_channels=hidden_channels,
                    num_levels=num_levels,
                    num_down_blocks=num_down_blocks,
                    num_up_blocks=num_up_blocks,
                    aabb_max=aabb_max,
                    aabb_min=aabb_min,
                    resolution=resolution,
                    grid_memory_format=memory_format,
                    use_rel_pos=use_rel_pos,
                    use_rel_pos_embed=use_rel_pos_embed,
                    pos_embed_dim=pos_embed_dim,
                )
                unets.append(unet)
            unets = nn.ModuleList(unets)
            unets_all.append(unets)
        self.unets_all = nn.ModuleList(unets_all)

    def forward(
        self,
        point_features: PointFeatures,
    ):
        for unets in self.unets_all:
            out_features = []
            for unet in unets:
                out_features.append(unet(point_features))

            # Multiply all features together
            out_feature = out_features[0]
            if self.unet_reduction == "mul":
                for feature in out_features[1:]:
                    out_feature = out_feature * feature
            elif self.unet_reduction == "sum":
                for feature in out_features[1:]:
                    out_feature = out_feature + feature
            else:
                raise ValueError(f"Unknown reduction: {self.unet_reduction}")
            point_features = out_feature

        return out_feature


class PointFeatureToGridUNetsDrivAer(DrivAerBase, PointFeatureToGridUNets):
    """PointFeatureToGridUNetsDrivAer."""

    def __init__(
        # Same as the PointFeatureToGridUNets
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        aabb_max: Tuple[float, float, float] = (2.5, 1.5, 1.0),
        aabb_min: Tuple[float, float, float] = (-2.5, -1.5, -1.0),
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
        # 0.02:  [250, 150, 100]
        # 0.025: [200, 120, 80]
        # 0.04:  [125, 75, 50]
        resolution_memory_format_pairs: List[
            Tuple[GridFeaturesMemoryFormat, Tuple[int, int, int]]
        ] = [
            (GridFeaturesMemoryFormat.b_xc_y_z, (4, 120, 80)),
            (GridFeaturesMemoryFormat.b_yc_x_z, (200, 3, 80)),
            (GridFeaturesMemoryFormat.b_zc_x_y, (200, 120, 2)),
        ],
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = True,
        pos_embed_dim: int = 32,
        pos_embed_range: float = 4,
        unet_reduction: Literal["mul", "sum"] = "mul",
        unet_repeat: int = 1,
    ):
        PointFeatureToGridUNets.__init__(
            self,
            in_channels=hidden_channels[0],
            out_channels=hidden_channels[0],
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            num_down_blocks=num_down_blocks,
            num_up_blocks=num_up_blocks,
            aabb_max=aabb_max,
            aabb_min=aabb_min,
            resolution_memory_format_pairs=resolution_memory_format_pairs,
            use_rel_pos=use_rel_pos,
            use_rel_pos_embed=use_rel_pos_embed,
            pos_embed_dim=pos_embed_dim,
            unet_reduction=unet_reduction,
            unet_repeat=unet_repeat,
        )
        DrivAerBase.__init__(self)
        self.vertex_to_point_features = VerticesToPointFeatures(
            embed_dim=pos_embed_dim,
            out_features=hidden_channels[0],
            use_mlp=True,
            pos_embed_range=pos_embed_range,
        )
        self.projection = MLP(
            [hidden_channels[0], out_channels],
            torch.nn.GELU,
        )

    def forward(
        self,
        vertices: Float[Tensor, "N 3"],
        features: Optional[Float[Tensor, "N C"]] = None,
    ) -> Tensor:
        if features is None:
            point_features = self.vertex_to_point_features(vertices)
        else:
            point_features = PointFeatures(vertices, features)

        out_point_features = PointFeatureToGridUNets.forward(self, point_features)
        return self.projection(out_point_features.features)

    def data_dict_to_input(self, data_dict):
        vertices = data_dict["vertices"].squeeze(0)  # (n_in, 3)

        # center vertices
        vertices_max = vertices.max(0)[0]
        vertices_min = vertices.min(0)[0]
        vertices_center = (vertices_max + vertices_min) / 2.0
        vertices = vertices - vertices_center

        return vertices


class PointFeatureToGridUNetsAhmedBody(AhmedBodyBase, PointFeatureToGridUNets):
    """PointFeatureToGridUNetsAhmedBody."""

    def __init__(
        # Same as the PointFeatureToGridUNets
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        aabb_max: Tuple[float, float, float] = (2.5, 1.5, 1.0),
        aabb_min: Tuple[float, float, float] = (-2.5, -1.5, -1.0),
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
        # 0.02:  [250, 150, 100]
        # 0.025: [200, 120, 80]
        # 0.04:  [125, 75, 50]
        resolution_memory_format_pairs: List[
            Tuple[GridFeaturesMemoryFormat, Tuple[int, int, int]]
        ] = [
            (GridFeaturesMemoryFormat.b_xc_y_z, (4, 120, 80)),
            (GridFeaturesMemoryFormat.b_yc_x_z, (200, 3, 80)),
            (GridFeaturesMemoryFormat.b_zc_x_y, (200, 120, 2)),
        ],
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = True,
        pos_embed_dim: int = 32,
        pos_embed_range: float = 4,
        unet_reduction: Literal["mul", "sum"] = "mul",
        unet_repeat: int = 1,
        # AhmedBodyBase
        use_uniformized_velocity: bool = True,
        velocity_pos_encoding: bool = False,
        normals_as_features: bool = True,
        vertices_as_features: bool = True,
        random_purturb_train: bool = True,
        vertices_purturb_range: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        BaseModel.__init__(self)
        AhmedBodyBase.__init__(
            self,
            use_uniformized_velocity=use_uniformized_velocity,
            velocity_pos_encoding=velocity_pos_encoding,
            normals_as_features=normals_as_features,
            vertices_as_features=vertices_as_features,
            pos_encode_dim=pos_embed_dim,
            pos_encode_range=1,
            random_purturb_train=random_purturb_train,
            vertices_purturb_range=vertices_purturb_range,
        )
        PointFeatureToGridUNets.__init__(
            self,
            in_channels=self.ahmed_input_feature_dim,
            out_channels=hidden_channels[0],
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            num_down_blocks=num_down_blocks,
            num_up_blocks=num_up_blocks,
            aabb_max=aabb_max,
            aabb_min=aabb_min,
            resolution_memory_format_pairs=resolution_memory_format_pairs,
            use_rel_pos=use_rel_pos,
            use_rel_pos_embed=use_rel_pos_embed,
            pos_embed_dim=pos_embed_dim,
            unet_reduction=unet_reduction,
            unet_repeat=unet_repeat,
        )
        self.projection = MLP(
            [hidden_channels[0], out_channels],
            torch.nn.GELU,
        )

    def forward(
        self,
        vertices: Float[Tensor, "N 3"],
        features: Optional[Float[Tensor, "N C"]] = None,
    ) -> Tensor:
        if features is None:
            point_features = self.vertex_to_point_features(vertices)
        else:
            point_features = PointFeatures(vertices, features)

        out_point_features = PointFeatureToGridUNets.forward(self, point_features)
        return self.projection(out_point_features.features)

    def data_dict_to_input(self, data_dict):
        vertices = data_dict["vertices"].squeeze(0)  # (n_in, 3)

        # center vertices
        vertices_max = vertices.max(0)[0]
        vertices_min = vertices.min(0)[0]
        vertices_center = (vertices_max + vertices_min) / 2.0
        vertices = vertices - vertices_center

        return vertices


class PointFeatureUNetWithGridUNets(BaseModel):
    """PointFeatureUNetWithGridUNets."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[int],
        radius_to_voxel_ratio: float = 2.0,
        pos_embed_dim: int = 32,
        pos_embed_range: float = 4,
        num_levels: int = 3,
        unit_voxel_size: float = 0.05,
        inter_level_voxel_ratio: float = 2.0,
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = False,
        reductions: List[Literal["mean", "min", "max", "sum"]] = ["mean"],
        # GridUNet parameters
        grid_unet_hidden_channels: List[int] = [64, 128, 256],
        grid_unet_num_levels: int = 3,
        grid_resolution: Tuple[int, int, int] = (128, 128, 128),
        compressed_grid_dim: int = 2,
        aabb_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        aabb_min: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        BaseModel.__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.radius_to_voxel_ratio = radius_to_voxel_ratio
        self.pos_embed_dim = pos_embed_dim
        self.num_levels = num_levels
        self.vertex_to_point_features = VerticesToPointFeatures(
            embed_dim=pos_embed_dim,
            out_features=hidden_channels[0],
            use_mlp=True,
            pos_embed_range=pos_embed_range,
        )
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for level in range(num_levels):
            # num_levels 3 then the down_voxel_size are 1/4, 1/2, 1
            down_voxel_size = unit_voxel_size / inter_level_voxel_ratio ** (
                num_levels - 1 - level
            )
            radius_size = down_voxel_size * radius_to_voxel_ratio
            print(f"Level {level} down voxel size: {down_voxel_size}")
            down_block = PointFeatureConvBlock(
                in_channels=hidden_channels[level],
                out_channels=hidden_channels[level + 1],
                radius=radius_size,
                pos_encode_dim=pos_embed_dim,
                use_rel_pos=use_rel_pos,
                use_rel_pos_encode=use_rel_pos_embed,
                reductions=reductions,
                pos_encode_range=pos_embed_range,
                out_point_feature_type="downsample",
                downsample_voxel_size=down_voxel_size,
            )
            self.down_blocks.append(down_block)
            # Add up blocks
            up_block = PointFeatureConvBlock(
                in_channels=hidden_channels[level + 1],
                out_channels=hidden_channels[level],
                radius=radius_size,
                pos_encode_dim=pos_embed_dim,
                use_rel_pos=use_rel_pos,
                use_rel_pos_encode=use_rel_pos_embed,
                reductions=reductions,
                pos_encode_range=pos_embed_range,
                out_point_feature_type="provided",
                provided_in_channels=hidden_channels[level],
            )
            self.up_blocks.append(up_block)

        # GridUNets after down blocks
        self.grid_unets = PointFeatureToGridUNets(
            in_channels=hidden_channels[num_levels],
            out_channels=hidden_channels[num_levels],
            kernel_size=3,
            hidden_channels=grid_unet_hidden_channels,
            num_levels=grid_unet_num_levels,
            aabb_max=aabb_max,
            aabb_min=aabb_min,
            resolution_memory_format_pairs=[
                (
                    GridFeaturesMemoryFormat.b_zc_x_y,
                    (grid_resolution[0], grid_resolution[1], compressed_grid_dim),
                ),
                (
                    GridFeaturesMemoryFormat.b_yc_x_z,
                    (grid_resolution[0], compressed_grid_dim, grid_resolution[2]),
                ),
                (
                    GridFeaturesMemoryFormat.b_xc_y_z,
                    (compressed_grid_dim, grid_resolution[1], grid_resolution[2]),
                ),
            ],
            use_rel_pos=use_rel_pos,
            use_rel_pos_embed=use_rel_pos_embed,
            pos_embed_dim=pos_embed_dim,
        )

        self.projection = MLP(
            [hidden_channels[0], out_channels],
            torch.nn.GELU,
        )

    def forward(
        self,
        vertices: Float[Tensor, "N 3"],
        features: Optional[Float[Tensor, "N C"]] = None,
    ):
        if features is None:
            point_features = self.vertex_to_point_features(vertices)
        else:
            point_features = PointFeatures(vertices, features)

        down_point_features = [point_features]
        for down_block in self.down_blocks:
            point_features = down_block(point_features)
            down_point_features.append(point_features)

        # GridUNets
        grid_out = self.grid_unets(down_point_features[-1])
        down_point_features[-1] = down_point_features[-1] + grid_out

        for level in reversed(range(self.num_levels)):
            up_point_features = self.up_blocks[level](
                down_point_features[level + 1], down_point_features[level]
            )
            down_point_features[level] = up_point_features
        return self.projection(down_point_features[0].features)

    def data_dict_to_input(self, data_dict):
        vertices = data_dict["vertices"].squeeze(0)  # (n_in, 3)

        # center vertices
        vertices_max = vertices.max(0)[0]
        vertices_min = vertices.min(0)[0]
        vertices_center = (vertices_max + vertices_min) / 2.0
        vertices = vertices - vertices_center

        return vertices
