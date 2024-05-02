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

from typing import List, Literal, Optional, Tuple, Union, Any

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
from .base_model import BaseModel, BaseModule
from .components.reductions import REDUCTION_TYPES
from .components.mlp import MLP
from .drivaer_base import DrivAerBase
from .grid_feature_group import (
    GridFeatureConv2DBlocksAndIntraCommunication,
    GridFeatureGroup,
    GridFeatureGroupPadToMatch,
    GridFeatureGroupToPoint,
)
from .grid_feature_unet import memory_format_to_axis_index
from .point_feature_grid_ops import PointFeatureToGrid
from .modelnet_base import ModelNet40Base


class TransfomerBlock(BaseModule):
    """
    Transformer block
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=in_channels,
                nhead=num_heads,
                dim_feedforward=hidden_channels,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

    def forward(self, x: Tensor):
        return self.transformer(x)


class EmbeddingMultiHeadKQVAttention(BaseModule):
    """
    Multi-head attention layer with query embedding
    """

    def __init__(
        self, input_dim, output_dim, num_heads, output_tokens, key_value_dim=None
    ):
        super(EmbeddingMultiHeadKQVAttention, self).__init__()
        self.num_heads = num_heads

        # Dimension of keys/values, default to input_dim if not specified
        self.key_value_dim = key_value_dim if key_value_dim is not None else input_dim

        # Query embedding layer to produce `output_tokens` queries of dimension `self.key_value_dim`
        self.query_embedding = nn.Parameter(
            torch.randn(1, output_tokens, self.key_value_dim), requires_grad=True
        )

        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=self.key_value_dim, num_heads=num_heads, batch_first=True
        )

        # Optional: Output transformation layer
        self.output_transform = (
            nn.Linear(self.key_value_dim, output_dim)
            if self.key_value_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x):
        # x is the input of shape (B, N, C)
        queries = torch.tile(
            self.query_embedding.to(x.device), (x.size(0), 1, 1)
        )  # Shape: (B, output_tokens, key_value_dim)

        # Passing the same x as keys and values
        attn_output, attn_output_weights = self.attention(queries, x, x)

        # Transform the output to desired output dimension C'
        output = self.output_transform(
            attn_output
        )  # Shape: (B, output_tokens, output_dim)

        return output


class MultiHeadAttentionPool(BaseModule):
    """
    Multi-head attention layer
    """

    def __init__(self, input_dim, output_dim, num_heads):
        super(MultiHeadAttentionPool, self).__init__()
        self.num_heads = num_heads

        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )

        # Optional: Output transformation layer
        self.output_transform = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x):
        # x is the input of shape (B, N, C)
        # Create query that is max-pooled over the points
        query = x.max(dim=1).values.unsqueeze(1)  # Shape: (B, 1, C)
        attn_output, attn_output_weights = self.attention(query, x, x)

        # Transform the output to desired output dimension C'
        output = self.output_transform(attn_output)  # Shape: (B, 1, output_dim)

        # Squeeze the output to remove the extra dimension
        output = output.squeeze(1)
        return output


class PointFeatureToFactorizedImplicitGlobalConvNet(BaseModel):
    """FIGConvNet"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        aabb_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        aabb_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
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
        neighbor_search_type: Literal["knn", "radius"] = "radius",
        knn_k: int = 16,
        reductions: List[REDUCTION_TYPES] = ["mean"],
        down_num_points: int = 1024,
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
        self.down_num_points = down_num_points
        for mem_fmt, res in resolution_memory_format_pairs:
            compressed_axis = memory_format_to_axis_index[mem_fmt]
            compressed_spatial_dims.append(res[compressed_axis])
            to_grid = nn.Sequential(
                PointFeatureToGrid(
                    in_channels=in_channels,
                    out_channels=hidden_channels[0],
                    aabb_max=aabb_max,
                    aabb_min=aabb_min,
                    radius=2.0,
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

        self.down_blocks = nn.ModuleList()

        if isinstance(num_down_blocks, int):
            num_down_blocks = [num_down_blocks] * (num_levels + 1)

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

        self.compressed_spatial_dims = compressed_spatial_dims
        self.convert_to_orig = GridFeatureMemoryFormatConverter(
            memory_format=GridFeaturesMemoryFormat.b_x_y_z_c
        )

    def _grid_forward(self, point_features: PointFeatures):
        grid_feature_group = GridFeatureGroup(
            [to_grid(point_features) for to_grid in self.point_feature_to_grids]
        )
        for down_block in self.down_blocks:
            grid_feature_group = down_block(grid_feature_group)

        return grid_feature_group

    def forward(
        self,
        point_features: PointFeatures,
    ):
        grid_features = self._grid_forward(point_features)
        return grid_features


def grid_features_to_sequence(
    grid_feature_group: GridFeatureGroup,
) -> List[Float[Tensor, "B N C"]]:
    """
    Convert grid features to sequence of features
    """
    feature_stack = []
    C = grid_feature_group[0].num_channels
    for grid_feature in grid_feature_group:
        feat = grid_feature.features.flatten(2)
        feat = feat.permute(0, 2, 1)
        feature_stack.append(feat)

    return feature_stack


class FIGConvNetModelNet(ModelNet40Base, PointFeatureToFactorizedImplicitGlobalConvNet):
    """
    FIGConvNet model for ModelNet40 dataset
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        hidden_channels: List[int],
        num_levels: int = 3,
        num_down_blocks: Union[int, List[int]] = 1,
        mlp_channels: List[int] = [2048, 2048],
        aabb_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        aabb_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
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
        neighbor_search_type: Literal["knn", "radius"] = "radius",
        knn_k: int = 16,
        reductions: List[REDUCTION_TYPES] = ["mean"],
        down_num_points: int = 1024,
    ):
        ModelNet40Base.__init__(self)
        PointFeatureToFactorizedImplicitGlobalConvNet.__init__(
            self,
            hidden_channels[0],
            out_channels,
            kernel_size,
            hidden_channels,
            num_levels,
            num_down_blocks,
            aabb_max,
            aabb_min,
            voxel_size,
            resolution_memory_format_pairs,
            use_rel_pos,
            use_rel_pos_embed,
            pos_encode_dim,
            communication_types,
            neighbor_search_type,
            knn_k,
            reductions,
            down_num_points,
        )

        self.vertex_to_point_features = VerticesToPointFeatures(
            embed_dim=pos_encode_dim,
            out_features=hidden_channels[0],
            use_mlp=True,
            pos_embed_range=aabb_max[0] - aabb_min[0],
        )

        # self.grid_pools = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Linear(
        #                 hidden_channels[num_levels] * c, hidden_channels[num_levels]
        #             ),
        #             nn.LayerNorm(hidden_channels[num_levels]),
        #             nn.GELU(),
        #             EmbeddingMultiHeadKQVAttention(
        #                 input_dim=hidden_channels[num_levels],
        #                 output_dim=hidden_channels[num_levels],
        #                 num_heads=4,
        #                 output_tokens=out_channels,
        #                 key_value_dim=hidden_channels[num_levels],
        #             ),
        #         )
        #         for c in self.compressed_spatial_dims
        #     ]
        # )
        # self.projection = nn.Linear(
        #     hidden_channels[num_levels] * self.grid_feature_group_size, 1
        # )

        # self.grid_pools = nn.ModuleList(
        #     [
        #         nn.Sequential(
        #             nn.Linear(
        #                 hidden_channels[num_levels] * c, mlp_channels[0]
        #             ),
        #             nn.LayerNorm(mlp_channels[0]),
        #             MultiHeadAttentionPool(
        #                 input_dim=mlp_channels[0],
        #                 output_dim=mlp_channels[0],
        #                 num_heads=4,
        #             ),
        #         )
        #         for c in self.compressed_spatial_dims
        #     ]
        # )
        self.grid_pools = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        hidden_channels[num_levels] * c, mlp_channels[0], kernel_size=1
                    ),
                    nn.AdaptiveMaxPool2d((1, 1)),
                    nn.Flatten(),
                    nn.LayerNorm(mlp_channels[0]),
                )
                for c in self.compressed_spatial_dims
            ]
        )

        self.mlp = MLP(
            mlp_channels[0] * len(self.compressed_spatial_dims),
            mlp_channels[-1],
            mlp_channels,
            use_residual=True,
            activation=nn.GELU,
        )
        self.projection = nn.Linear(mlp_channels[-1], out_channels)

    def forward(self, points: Float[Tensor, "B N 3"]):
        with torch.no_grad():
            point_features = self.vertex_to_point_features(points)
        grid_features = PointFeatureToFactorizedImplicitGlobalConvNet.forward(
            self, point_features
        )
        # seqs = grid_features_to_sequence(grid_features)
        seqs = [pool(seq.features) for seq, pool in zip(grid_features, self.grid_pools)]
        seqs = torch.cat(seqs, dim=-1)
        return self.projection(self.mlp(seqs))


def test_figconvnet():

    device = torch.device("cuda:0")
    model = FIGConvNetModelNet(
        in_channels=3,
        out_channels=40,
        kernel_size=3,
        hidden_channels=[16, 32, 64, 128],
        num_levels=3,
        num_down_blocks=1,
        aabb_max=(1.0, 1.0, 1.0),
        aabb_min=(-1.0, -1.0, -1.0),
        voxel_size=None,
        resolution_memory_format_pairs=[
            (GridFeaturesMemoryFormat.b_xc_y_z, (2, 128, 128)),
            (GridFeaturesMemoryFormat.b_yc_x_z, (128, 2, 128)),
            (GridFeaturesMemoryFormat.b_zc_x_y, (128, 128, 2)),
        ],
        use_rel_pos=True,
        use_rel_pos_embed=True,
        pos_encode_dim=32,
        communication_types=["sum"],
        neighbor_search_type="radius",
        knn_k=16,
        reductions=["mean"],
        down_num_points=1024,
    ).to(device)

    points = torch.rand(2, 1024, 3).to(device)
    output = model(points)
    assert output.shape == (2, 40)


if __name__ == "__main__":
    test_figconvnet()
