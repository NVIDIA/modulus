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

from typing import List, Dict, Any, Optional
from jaxtyping import Float

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from .base_model import BaseModel
from .ahmedbody_base import AhmedBodyDragRegressionBase
from .drivaer_base import DrivAerDragRegressionBase
from .modelnet_base import ModelNet40Base
from .neighbor_ops import neighbor_knn_search
from .components.mlp import MLP


def batched_self_knn(x, k: int, chunk_size: int = 4096):
    """
    Compute the k-nearest neighbors for each point in a batch of point clouds.
    :param x: (B, N, D) input point cloud
    :param k: number of neighbors
    :return: (B, N, k) indices of the k-nearest neighbors
    """
    neighbor_index = []
    for i in range(x.size(0)):
        x_NC = x[i].transpose(0, 1)
        neighbor_index.append(
            neighbor_knn_search(x_NC, x_NC, k, chunk_size=chunk_size).unsqueeze(0)
        )
    return torch.cat(neighbor_index, dim=0)


class EdgeConv(nn.Module):
    """
    EdgeConv layer that collects features from the KNN neighbors and applying MLP using kernel size 1 convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        negative_slope: float = 0.2,
        knn_k: int = 4,
    ):
        super(EdgeConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.knn_k = knn_k

    def get_graph_feature(self, x, k=20, idx=None):
        """
        Return KNN neighbor features
        """
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        if idx is None:
            idx = batched_self_knn(x, k=k)  # (batch_size, num_points, k)
        idx_base = (
            torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        )

        idx = idx + idx_base
        idx = idx.view(-1)

        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()
        # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return feature

    def forward(self, x: torch.Tensor, idx: Optional[torch.Tensor] = None):
        """
        Forward pass

        :param x: input features (B, C, N)
        :param idx: indices of the k-nearest neighbors (B, N, k)
        """
        x = self.get_graph_feature(x, k=self.knn_k, idx=idx)
        x = self.conv(x)
        return x.max(dim=-1, keepdim=False)[0]


class DGCNN(BaseModel):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        conv_channels: List[int] = [256, 512, 512, 1024],
        pre_mlp_channels: int = 512,
        mlp_channels: List[int] = [1024, 1024],
        knn_k: int = 4,
        knn_search_chunk_size: int = 4096,
    ):
        super(DGCNN, self).__init__()
        self.k = knn_k
        self.knn_search_chunk_size = knn_search_chunk_size

        self.convs = nn.ModuleList()
        self.convs.append(EdgeConv(in_channels, conv_channels[0], knn_k=knn_k))
        for i in range(1, len(conv_channels)):
            self.convs.append(
                EdgeConv(conv_channels[i - 1], conv_channels[i], knn_k=knn_k)
            )

        self.pre_mlp = nn.Conv1d(sum(conv_channels), pre_mlp_channels, kernel_size=1)
        self.mlp = MLP(
            pre_mlp_channels,
            mlp_channels[-1],
            mlp_channels,
            use_residual=True,
            activation=nn.GELU,
        )
        self.projection = nn.Linear(mlp_channels[-1], out_channels)

    def forward(self, x: Float[Tensor, "B C N"]) -> Float[Tensor, "B C2"]:
        batch_size = x.size(0)
        idx = batched_self_knn(x, k=self.k, chunk_size=self.knn_search_chunk_size)

        conv_outs = []
        for conv in self.convs:
            x = conv(x, idx)
            conv_outs.append(x)

        x = torch.cat(conv_outs, dim=1)
        x = self.pre_mlp(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)

        return self.projection(self.mlp(x))


class DGCNNModelNet40(ModelNet40Base, DGCNN):
    """
    DGCNN from https://arxiv.org/abs/1801.07829
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 40,
        conv_channels: List[int] = [256, 512, 512, 1024],
        pre_mlp_channels: int = 1024,
        mlp_channels: List[int] = [1024, 1024],
        knn_k: int = 4,
        knn_search_chunk_size: int = 4096,
    ):
        ModelNet40Base.__init__(self)

        DGCNN.__init__(
            self,
            in_channels,
            out_channels,
            conv_channels,
            pre_mlp_channels,
            mlp_channels,
            knn_k,
            knn_search_chunk_size,
        )

    def data_dict_to_input(self, data_dict, **kwargs) -> Any:
        """Convert data dictionary to appropriate input for the model."""
        # From BxNxC to BxCxN for DGCNN input
        points = data_dict["vertices"].to(self.device).transpose(1, 2)
        label = data_dict["class"].to(self.device)
        return points, label


class DrivAerNet(DrivAerDragRegressionBase, DGCNN):
    """
    DrivAerNet from https://arxiv.org/abs/2403.08055

    This network is a DGCNN with a few modifications to the architecture such as
    network depth, channel sizes.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        conv_channels: List[int] = [256, 512, 512, 1024],
        pre_mlp_channels: int = 512,
        mlp_channels: List[int] = [128, 64, 32, 16],
        knn_k: int = 4,
        point_cloud_sample_size: int = 5000,
        knn_search_chunk_size: int = 4096,
    ):
        DrivAerDragRegressionBase.__init__(self)

        DGCNN.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            conv_channels=conv_channels,
            pre_mlp_channels=pre_mlp_channels,
            mlp_channels=mlp_channels,
            knn_k=knn_k,
            knn_search_chunk_size=knn_search_chunk_size,
        )

        self.point_cloud_sample_size = point_cloud_sample_size

    def data_dict_to_input(self, data_dict) -> torch.Tensor:
        vertices = data_dict["cell_centers"]

        # Sample N random points
        sampled_verts = torch.empty(vertices.size(0), self.point_cloud_sample_size, 3)
        for i in range(vertices.size(0)):
            perm_vert = vertices[i][:, torch.randperm(vertices.size(2))]
            sampled_verts[i] = perm_vert[: self.point_cloud_sample_size]

        return sampled_verts.float().to(self.device).transpose(1, 2)  # B, 3, N
