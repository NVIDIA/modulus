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

from src.networks.base_model import BaseModule
from src.networks.point_feature_grid_conv import (
    GridFeatureConv2d,
    GridFeatureConv2dBlock,
    GridFeaturePadToMatch,
    GridFeatureTransform,
    LayerNorm2d,
)
from src.networks.point_feature_ops import (
    GridFeatures,
    GridFeaturesMemoryFormat,
    PointFeatures,
)

from .components.reductions import REDUCTION_TYPES
from .point_feature_grid_ops import (
    GridFeatureCat,
    GridFeatureToPoint,
)


class GridFeatureGroup:
    """A group of GridFeatures."""

    grid_features: List[GridFeatures]

    def __init__(self, grid_features: List[GridFeatures]) -> None:
        assert len(grid_features) > 0
        self.grid_features = grid_features

    def to(
        self,
        device: Union[torch.device, str] = None,
        memory_format: GridFeaturesMemoryFormat = None,
    ):
        assert device is not None or memory_format is not None
        if device is not None:
            for grid_features in self.grid_features:
                grid_features.to(device=device)

        if memory_format is not None:
            for grid_features in self.grid_features:
                grid_features.to(memory_format=memory_format)
        return self

    def __getitem__(self, index: int) -> GridFeatures:
        return self.grid_features[index]

    def __len__(self) -> int:
        return len(self.grid_features)

    def __iter__(self):
        return iter(self.grid_features)

    def __repr__(self) -> str:
        out_str = "GridFeaturesGroup("
        for grid_features in self.grid_features:
            out_str += f"\n\t{grid_features}"
        out_str += "\n)"
        return out_str

    def __add__(self, other: "GridFeatureGroup") -> "GridFeatureGroup":
        assert len(self) == len(other)
        grid_features = []
        for i in range(len(self)):
            grid_features.append(self[i] + other[i])
        return GridFeatureGroup(grid_features)


class GridFeaturesGroupIntraCommunication(nn.Module):
    """A module that performs intra-communication between GridFeatures in a GridFeaturesGroup."""

    def __init__(self, communication_type: Literal["sum", "mul"] = "sum") -> None:
        super().__init__()
        self.communication_type = communication_type

    def forward(self, grid_features_group: GridFeatureGroup) -> GridFeatureGroup:
        # convert grid_features in grid_features_group to cxyz format
        orig_memory_formats = []
        for grid_features in grid_features_group:
            orig_memory_formats.append(grid_features.memory_format)
            grid_features.to(memory_format=GridFeaturesMemoryFormat.c_x_y_z)

        # Assert the channel size of all grid_features are the same
        channel_size = grid_features_group[0].features.shape[0]
        for grid_features in grid_features_group:
            assert (
                grid_features.features.shape[0] == channel_size
            ), f"Channel size of grid_features are not the same: {grid_features.features.shape[1]} != {channel_size}"

        # broadcast the features between all pairs of grid_features
        orig_features = [
            torch.clone(grid_features.features) for grid_features in grid_features_group
        ]
        normalized_xyzs = []
        with torch.no_grad():
            for i in range(len(grid_features_group)):
                vertices = grid_features_group[i].vertices
                if (
                    grid_features_group[i].vertices.shape[:3]
                    != orig_features[i].shape[1:]
                ):
                    vertices = grid_features_group[i].strided_vertices(
                        orig_features[i].shape[1:]
                    )

                xyz = vertices.flatten(0, 2)
                xyz_min = torch.min(xyz, dim=1, keepdim=True)[0]
                xyz_max = torch.max(xyz, dim=1, keepdim=True)[0]
                normalized_xyz = (xyz - xyz_min) / (xyz_max - xyz_min) * 2 - 1
                normalized_xyzs.append(normalized_xyz.view(vertices.shape))

        # Add features from orig_featurs j to i
        for i in range(len(grid_features_group)):
            for j in range(len(grid_features_group)):
                if i == j:
                    continue
                sampled_features = torch.nn.functional.grid_sample(
                    orig_features[j].unsqueeze(0),  # BCHWD
                    normalized_xyzs[i].view(1, 1, 1, -1, 3),  # BHWD3
                    align_corners=True,
                )  # BC11N
                # Reshape to CHWD
                # TODO: replace the shape
                sampled_features = sampled_features.reshape(
                    sampled_features.shape[1],
                    normalized_xyzs[i].shape[0],
                    normalized_xyzs[i].shape[1],
                    normalized_xyzs[i].shape[2],
                )
                if self.communication_type == "sum":
                    grid_features_group[i].features += sampled_features
                elif self.communication_type == "mul":
                    grid_features_group[i].features *= sampled_features
                else:
                    raise NotImplementedError

        # convert grid_features in grid_features_group back to original memory format
        for i, grid_features in enumerate(grid_features_group):
            grid_features.to(memory_format=orig_memory_formats[i])

        return grid_features_group


class GridFeatureGroupIntraCommunications(nn.Module):
    """Multiple communication types e.g. ["sum", "mul"]"""

    def __init__(
        self, communication_types: List[Literal["sum", "mul"]] = ["sum"]
    ) -> None:
        super().__init__()
        self.intra_communications = nn.ModuleList()
        self.grid_cat = GridFeatureGroupCat()
        for communication_type in communication_types:
            self.intra_communications.append(
                GridFeaturesGroupIntraCommunication(
                    communication_type=communication_type
                )
            )

    def forward(self, grid_features_group: GridFeatureGroup) -> GridFeatureGroup:
        if len(self.intra_communications) == 1:
            return self.intra_communications[0](grid_features_group)
        elif len(self.intra_communications) == 2:
            # cat features
            return self.grid_cat(
                self.intra_communications[0](grid_features_group),
                self.intra_communications[1](grid_features_group),
            )
        else:
            raise NotImplementedError


class GridFeatureGroupConv2dNorm(BaseModule):
    """GridFeatureGroupConv2dNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        compressed_spatial_dims: Tuple[int],
        stride: int = 1,
        up_stride: Optional[int] = None,
        norm: nn.Module = LayerNorm2d,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for compressed_spatial_dim in compressed_spatial_dims:
            self.convs.append(
                nn.Sequential(
                    GridFeatureConv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        compressed_spatial_dim=compressed_spatial_dim,
                        stride=stride,
                        up_stride=up_stride,
                    ),
                    GridFeatureTransform(norm(out_channels * compressed_spatial_dim)),
                )
            )

    def forward(self, grid_features_group: GridFeatureGroup) -> GridFeatureGroup:
        assert len(grid_features_group) == len(self.convs)
        grid_feats = []
        for grid_feat, conv in zip(grid_features_group, self.convs):
            grid_feats.append(conv(grid_feat))
        return GridFeatureGroup(grid_feats)


class GridFeatureGroupTransform(BaseModule):
    """GridFeatureGroupTransform."""

    def __init__(self, transform: nn.Module, in_place: bool = True) -> None:
        super().__init__()
        self.transform = transform
        self.in_place = in_place

    def forward(self, grid_feature_group: GridFeatureGroup) -> GridFeatureGroup:
        if not self.in_place:
            grid_feature_group = GridFeatureGroup(
                [grid_feature.clone() for grid_feature in grid_feature_group]
            )
        for grid_feature in grid_feature_group:
            grid_feature.features = self.transform(grid_feature.features)
        return grid_feature_group


class GridFeatureConv2DBlocksAndIntraCommunication(nn.Module):
    """GridFeatureConv2DBlocksAndIntraCommunication."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        compressed_spatial_dims: Tuple[int],
        stride: int = 1,
        up_stride: Optional[int] = None,
        communication_types: List[Literal["sum", "mul"]] = ["sum"],
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for compressed_spatial_dim in compressed_spatial_dims:
            self.convs.append(
                GridFeatureConv2dBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    compressed_spatial_dim=compressed_spatial_dim,
                    stride=stride,
                    up_stride=up_stride,
                    apply_nonlinear_at_end=False,
                )
            )
        self.intra_communications = GridFeatureGroupIntraCommunications(
            communication_types=communication_types
        )
        # If len(communication_types) > 1, apply linear projection to reduce the channel size
        if isinstance(communication_types, str):
            communication_types = [communication_types]
        if len(communication_types) > 1:
            self.proj = GridFeatureGroupConv2dNorm(
                in_channels=out_channels * len(communication_types),
                out_channels=out_channels,
                kernel_size=1,
                compressed_spatial_dims=compressed_spatial_dims,
            )
        else:
            self.proj = nn.Identity()
        self.nonlinear = GridFeatureGroupTransform(nn.GELU())

    def forward(self, grid_features_group: GridFeatureGroup) -> GridFeatureGroup:
        assert len(grid_features_group) == len(self.convs)
        grid_feats = []
        for grid_feat, conv in zip(grid_features_group, self.convs):
            grid_feats.append(conv(grid_feat))
        grid_features_group = GridFeatureGroup(grid_feats)
        grid_features_group = self.intra_communications(grid_features_group)
        grid_features_group = self.proj(grid_features_group)
        grid_features_group = self.nonlinear(grid_features_group)
        return grid_features_group


class GridFeatureGroupCat(BaseModule):
    """GridFeatureGroupCat."""

    def __init__(self):
        super().__init__()
        self.grid_cat = GridFeatureCat()

    def forward(
        self, group1: GridFeatureGroup, group2: GridFeatureGroup
    ) -> GridFeatureGroup:
        assert len(group1) == len(group2)
        return GridFeatureGroup(
            [self.grid_cat(g1, g2) for g1, g2 in zip(group1, group2)]
        )


class GridFeatureGroupPadToMatch(BaseModule):
    """GridFeatureGroupPadToMatch."""

    def __init__(self) -> None:
        super().__init__()
        self.match = GridFeaturePadToMatch()

    def forward(
        self,
        grid_features_group_ref: GridFeatureGroup,
        grid_features_group_target: GridFeatureGroup,
    ) -> GridFeatureGroup:
        assert len(grid_features_group_ref) == len(grid_features_group_target)
        grid_features_group_out = []
        for i in range(len(grid_features_group_ref)):
            grid_features_group_out.append(
                self.match(grid_features_group_ref[i], grid_features_group_target[i])
            )
        return GridFeatureGroup(grid_features_group_out)


class GridFeatureGroupToPoint(BaseModule):
    """GridFeatureGroupToPoint."""

    def __init__(
        self,
        grid_in_channels: int,
        point_in_channels: int,
        out_channels: int,
        grid_feature_group_size: int,
        aabb_max: Tuple[float, float, float],
        aabb_min: Tuple[float, float, float],
        use_rel_pos: bool = True,
        use_rel_pos_embed: bool = False,
        pos_embed_dim: int = 32,
        sample_method: Literal["graphconv", "interp"] = "graphconv",
        neighbor_search_type: Literal["radius", "knn"] = "radius",
        knn_k: int = 16,
        reductions: List[REDUCTION_TYPES] = ["mean"],
    ) -> None:
        super().__init__()
        self.conv_list = nn.ModuleList()
        assert out_channels % 2 == 0

        for i in range(grid_feature_group_size):
            self.conv_list.append(
                GridFeatureToPoint(
                    grid_in_channels=grid_in_channels,
                    point_in_channels=point_in_channels,
                    out_channels=out_channels // 2,
                    aabb_max=aabb_max,
                    aabb_min=aabb_min,
                    use_rel_pos=use_rel_pos,
                    use_rel_pos_embed=use_rel_pos_embed,
                    pos_embed_dim=pos_embed_dim,
                    sample_method=sample_method,
                    neighbor_search_type=neighbor_search_type,
                    knn_k=knn_k,
                    reductions=reductions,
                )
            )

    def forward(
        self, grid_features_group: GridFeatureGroup, point_features: PointFeatures
    ) -> PointFeatures:
        assert len(grid_features_group) == len(self.conv_list)
        out_point_features: PointFeatures = self.conv_list[0](
            grid_features_group[0], point_features
        )
        out_point_features_add: PointFeatures = out_point_features
        out_point_features_mul: PointFeatures = out_point_features
        for i in range(1, len(grid_features_group)):
            curr = self.conv_list[i](grid_features_group[i], point_features)
            out_point_features_add += curr
            out_point_features_mul *= curr
        out_point_features = PointFeatures(
            vertices=point_features.vertices,
            features=torch.cat(
                (out_point_features_add.features, out_point_features_mul.features),
                dim=1,
            ),
        )
        return out_point_features