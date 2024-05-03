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

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from src.networks.point_feature_conv import PointFeatureConv, PointFeatureTransform
from src.networks.point_feature_grid_conv import (
    GridFeatureMemoryFormatConverter,
)
from src.networks.point_feature_ops import (
    GridFeaturesMemoryFormat,
    PointFeatures,
    VerticesToPointFeatures,
)

from .ahmedbody_base import AhmedBodyBase
from .base_model import BaseModel
from .components.reductions import REDUCTION_TYPES
from .components.mlp import MLP
from .drivaer_base import DrivAerBase
from .grid_feature_group import (
    GridFeatureConv2DBlocksAndIntraCommunication,
    GridFeatureGroup,
    GridFeatureGroupPadToMatch,
    GridFeatureGroupToPoint,
    GridFeatureGroupPool,
)
from .grid_feature_unet import memory_format_to_axis_index
from .point_feature_grid_ops import PointFeatureToGrid


class PointFeatureToGridGroupUNet(BaseModel):
    """PointFeatureToGridGroupUNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        num_up_blocks: Union[int, List[int]] = 1,
        mlp_channels: List[int] = [2048, 2048],
        aabb_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        aabb_min: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        voxel_size: Optional[float] = None,
        resolution_memory_format_pairs: List[
            Tuple[GridFeaturesMemoryFormat, Tuple[int, int, int]]
        ] = [
            (GridFeaturesMemoryFormat.b_xc_y_z, (2, 128, 128)),
            (GridFeaturesMemoryFormat.b_yc_x_z, (128, 2, 128)),
            (GridFeaturesMemoryFormat.b_zc_x_y, (128, 128, 2)),
        ],
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = True,
        pos_encode_dim: int = 32,
        communication_types: List[Literal["mul", "sum"]] = ["sum"],
        to_point_sample_method: Literal["graphconv", "interp"] = "graphconv",
        neighbor_search_type: Literal["knn", "radius"] = "radius",
        knn_k: int = 16,
        reductions: List[REDUCTION_TYPES] = ["mean"],
    ):
        BaseModel.__init__(self)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_levels = num_levels
        compressed_spatial_dims = []
        self.grid_feature_group_size = len(resolution_memory_format_pairs)
        self.point_feature_to_grids = nn.ModuleList()
        self.aabb_length = torch.tensor(aabb_max) - torch.tensor(aabb_min)
        self.min_voxel_edge_length = torch.tensor([np.inf, np.inf, np.inf])
        for mem_fmt, res in resolution_memory_format_pairs:
            compressed_axis = memory_format_to_axis_index[mem_fmt]
            compressed_spatial_dims.append(res[compressed_axis])
            to_grid = nn.Sequential(
                PointFeatureToGrid(
                    in_channels=in_channels,
                    out_channels=hidden_channels[0],
                    aabb_max=aabb_max,
                    aabb_min=aabb_min,
                    voxel_size=voxel_size,
                    resolution=res,
                    use_rel_pos=use_rel_pos,
                    use_rel_pos_encode=use_rel_pos_embed,
                    pos_encode_dim=pos_encode_dim,
                    reductions=reductions,
                    neighbor_search_type=neighbor_search_type,
                    knn_k=knn_k,
                ),
                GridFeatureMemoryFormatConverter(
                    memory_format=mem_fmt,
                ),
            )
            self.point_feature_to_grids.append(to_grid)
            # Compute voxel size
            voxel_size = self.aabb_length / torch.tensor(res)
            self.min_voxel_edge_length = torch.min(
                self.min_voxel_edge_length, voxel_size
            )
        self.compressed_spatial_dims = compressed_spatial_dims

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        if isinstance(num_down_blocks, int):
            num_down_blocks = [num_down_blocks] * (num_levels + 1)
        if isinstance(num_up_blocks, int):
            num_up_blocks = [num_up_blocks] * (num_levels + 1)

        for level in range(num_levels):
            down_block = [
                GridFeatureConv2DBlocksAndIntraCommunication(
                    in_channels=hidden_channels[level],
                    out_channels=hidden_channels[level + 1],
                    kernel_size=kernel_size,
                    stride=2,
                    compressed_spatial_dims=compressed_spatial_dims,
                    communication_types=communication_types,
                )
            ]
            for _ in range(1, num_down_blocks[level]):
                down_block.append(
                    GridFeatureConv2DBlocksAndIntraCommunication(
                        in_channels=hidden_channels[level + 1],
                        out_channels=hidden_channels[level + 1],
                        kernel_size=kernel_size,
                        stride=1,
                        compressed_spatial_dims=compressed_spatial_dims,
                        communication_types=communication_types,
                    )
                )
            down_block = nn.Sequential(*down_block)
            self.down_blocks.append(down_block)
            # Add up blocks
            up_block = [
                GridFeatureConv2DBlocksAndIntraCommunication(
                    in_channels=hidden_channels[level + 1],
                    out_channels=hidden_channels[level],
                    kernel_size=kernel_size,
                    up_stride=2,
                    compressed_spatial_dims=compressed_spatial_dims,
                    communication_types=communication_types,
                )
            ]
            for _ in range(1, num_up_blocks[level]):
                up_block.append(
                    GridFeatureConv2DBlocksAndIntraCommunication(
                        in_channels=hidden_channels[level],
                        out_channels=hidden_channels[level],
                        kernel_size=kernel_size,
                        up_stride=1,
                        compressed_spatial_dims=compressed_spatial_dims,
                        communication_types=communication_types,
                    )
                )
            up_block = nn.Sequential(*up_block)
            self.up_blocks.append(up_block)
        self.convert_to_orig = GridFeatureMemoryFormatConverter(
            memory_format=GridFeaturesMemoryFormat.b_x_y_z_c
        )

        self.grid_pools = GridFeatureGroupPool(
            in_channels=hidden_channels[-1],
            out_channels=mlp_channels[0],
            compressed_spatial_dims=self.compressed_spatial_dims,
        )

        self.mlp = MLP(
            mlp_channels[0] * len(self.compressed_spatial_dims),
            mlp_channels[-1],
            mlp_channels,
            use_residual=True,
            activation=nn.GELU,
        )
        self.mlp_projection = nn.Linear(mlp_channels[-1], 1)
        # nn.Sigmoid(),

        self.to_point = GridFeatureGroupToPoint(
            grid_in_channels=hidden_channels[0],
            point_in_channels=in_channels,
            out_channels=hidden_channels[0] * 2,
            grid_feature_group_size=self.grid_feature_group_size,
            aabb_max=aabb_max,
            aabb_min=aabb_min,
            use_rel_pos=use_rel_pos,
            use_rel_pos_embed=use_rel_pos_embed,
            pos_embed_dim=pos_encode_dim,
            sample_method=to_point_sample_method,
            neighbor_search_type=neighbor_search_type,
            knn_k=knn_k,
            reductions=reductions,
        )
        self.projection = PointFeatureTransform(
            nn.Sequential(
                nn.Linear(hidden_channels[0] * 2, hidden_channels[0] * 2),
                nn.LayerNorm(hidden_channels[0] * 2),
                nn.GELU(),
                nn.Linear(hidden_channels[0] * 2, out_channels),
            )
        )

        self.pad_to_match = GridFeatureGroupPadToMatch()

    def _grid_forward(self, point_features: PointFeatures):
        grid_feature_group = GridFeatureGroup(
            [to_grid(point_features) for to_grid in self.point_feature_to_grids]
        )
        down_grid_feature_groups = [grid_feature_group]
        for down_block in self.down_blocks:
            out_features = down_block(down_grid_feature_groups[-1])
            down_grid_feature_groups.append(out_features)

        # Drag prediction
        pooled_feats = self.grid_pools(down_grid_feature_groups[-1])
        drag_pred = self.mlp_projection(self.mlp(pooled_feats))

        for level in reversed(range(self.num_levels)):
            up_grid_features = self.up_blocks[level](
                down_grid_feature_groups[level + 1]
            )
            padded_down_features = self.pad_to_match(
                up_grid_features, down_grid_feature_groups[level]
            )
            up_grid_features = up_grid_features + padded_down_features
            down_grid_feature_groups[level] = up_grid_features

        grid_features = self.convert_to_orig(down_grid_feature_groups[0])
        return grid_features, drag_pred

    def forward(
        self,
        point_features: PointFeatures,
    ) -> Tuple[PointFeatures, Tensor]:
        grid_features, drag_pred = self._grid_forward(point_features)
        out_point_features = self.to_point(grid_features, point_features)
        out_point_features = self.projection(out_point_features)
        return out_point_features, drag_pred


class PointFeatureToGridGroupUNetDrivAer(DrivAerBase, PointFeatureToGridGroupUNet):
    """PointFeatureToGridGroupUNetDrivAer."""

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
        voxel_size: Optional[float] = None,
        resolution_memory_format_pairs: List[
            Tuple[GridFeaturesMemoryFormat, Tuple[int, int, int]]
        ] = [
            (GridFeaturesMemoryFormat.b_xc_y_z, (2, 128, 128)),
            (GridFeaturesMemoryFormat.b_yc_x_z, (128, 2, 128)),
            (GridFeaturesMemoryFormat.b_zc_x_y, (128, 128, 2)),
        ],
        use_rel_pos: bool = True,
        use_rel_pos_encode: bool = True,
        pos_encode_dim: int = 32,
        communication_types: List[Literal["mul", "sum"]] = ["sum"],
        to_point_sample_method: Literal["graphconv", "interp"] = "graphconv",
        neighbor_search_type: Literal["knn", "radius"] = "knn",
        knn_k: int = 16,
        reductions: List[REDUCTION_TYPES] = ["mean"],
    ):
        DrivAerBase.__init__(self)

        vertex_to_point_features = VerticesToPointFeatures(
            embed_dim=pos_encode_dim,
            out_features=hidden_channels[0],
            use_mlp=True,
            pos_embed_range=aabb_max[0] - aabb_min[0],
        )

        PointFeatureToGridGroupUNet.__init__(
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
            resolution_memory_format_pairs=resolution_memory_format_pairs,
            use_rel_pos=use_rel_pos,
            use_rel_pos_embed=use_rel_pos_encode,
            pos_encode_dim=pos_encode_dim,
            communication_types=communication_types,
            to_point_sample_method=to_point_sample_method,
            neighbor_search_type=neighbor_search_type,
            knn_k=knn_k,
            reductions=reductions,
        )

        self.vertex_to_point_features = vertex_to_point_features

    def data_dict_to_input(self, data_dict):
        return DrivAerBase.data_dict_to_input(self, data_dict)

    def forward(
        self,
        vertices: Float[Tensor, "B N 3"],
        features: Optional[Float[Tensor, "B N C"]] = None,
    ) -> Tensor:
        if features is None:
            point_features = self.vertex_to_point_features(vertices)
        else:
            point_features = PointFeatures(vertices, features)
        out_point_features, drag_pred = PointFeatureToGridGroupUNet.forward(
            self, point_features
        )
        return out_point_features.features, drag_pred


class PointFeatureToGridGroupUNetAhmedBody(AhmedBodyBase, PointFeatureToGridGroupUNet):
    """PointFeatureToGridGroupUNetAhmedBody."""

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
        voxel_size: Optional[float] = None,
        resolution_memory_format_pairs: List[
            Tuple[GridFeaturesMemoryFormat, Tuple[int, int, int]]
        ] = [
            (GridFeaturesMemoryFormat.b_xc_y_z, (6, 104, 90)),
            (GridFeaturesMemoryFormat.b_yc_x_z, (280, 2, 90)),
            (GridFeaturesMemoryFormat.b_zc_x_y, (280, 104, 2)),
        ],
        use_rel_pos: bool = True,
        use_rel_pos_encode: bool = True,
        pos_encode_dim: int = 32,
        communication_types: List[Literal["mul", "sum"]] = ["sum"],
        to_point_sample_method: Literal["graphconv", "interp"] = "graphconv",
        to_point_neighbor_search_type: Literal["knn", "radius"] = "knn",
        to_point_knn_k: int = 16,
        # AhmedBodyBase
        use_uniformized_velocity: bool = True,
        velocity_pos_encoding: bool = False,
        normals_as_features: bool = True,
        vertices_as_features: bool = True,
    ):
        BaseModel.__init__(self)
        PointFeatureToGridGroupUNet.__init__(
            self,
            in_channels=hidden_channels[0],
            out_channels=2 * hidden_channels[0],
            kernel_size=kernel_size,
            hidden_channels=hidden_channels,
            num_levels=num_levels,
            num_down_blocks=num_down_blocks,
            num_up_blocks=num_up_blocks,
            aabb_max=aabb_max,
            aabb_min=aabb_min,
            voxel_size=voxel_size,
            resolution_memory_format_pairs=resolution_memory_format_pairs,
            use_rel_pos=use_rel_pos,
            use_rel_pos_embed=use_rel_pos_encode,
            pos_encode_dim=pos_encode_dim,
            communication_types=communication_types,
            to_point_sample_method=to_point_sample_method,
            neighbor_search_type=to_point_neighbor_search_type,
            knn_k=to_point_knn_k,
        )

        AhmedBodyBase.__init__(
            self,
            use_uniformized_velocity=use_uniformized_velocity,
            velocity_pos_encoding=velocity_pos_encoding,
            normals_as_features=normals_as_features,
            vertices_as_features=vertices_as_features,
            pos_encode_dim=pos_encode_dim,
            pos_encode_range=1,
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
            knn_k=to_point_knn_k,
            radius=None,
        )

        self.to_point = GridFeatureGroupToPoint(
            grid_in_channels=hidden_channels[0],
            point_in_channels=hidden_channels[0],
            out_channels=hidden_channels[0] * 2,
            grid_feature_group_size=self.grid_feature_group_size,
            aabb_max=aabb_max,
            aabb_min=aabb_min,
            use_rel_pos=use_rel_pos,
            use_rel_pos_embed=use_rel_pos_encode,
            pos_embed_dim=pos_encode_dim,
            sample_method=to_point_sample_method,
            neighbor_search_type=to_point_neighbor_search_type,
            knn_k=to_point_knn_k,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels[0] * 2, hidden_channels[0] * 2),
            nn.LayerNorm(hidden_channels[0] * 2),
            nn.GELU(),
            nn.Linear(hidden_channels[0] * 2, out_channels),
        )

    def forward(
        self,
        vertices: Float[Tensor, "B N 3"],
        features: Float[Tensor, "B N C"],
    ) -> Tensor:
        point_feature = PointFeatures(vertices, features)
        point_feature = self.first_conv(point_feature)
        # Downsample points
        down_point_feature = point_feature.voxel_down_sample(
            self.min_voxel_edge_length.min()
        )
        # TriplaneUNet
        grid_features, drag_pred = self._grid_forward(down_point_feature)
        out_point_feature = self.to_point(grid_features, point_feature)
        return self.projection(out_point_feature.features), drag_pred
