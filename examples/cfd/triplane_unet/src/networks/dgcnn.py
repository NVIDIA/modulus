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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseModel
from .ahmedbody_base import AhmedBodyDragRegressionBase
from .drivaer_base import DrivAerDragRegressionBase
from .neighbor_ops import neighbor_knn_search


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
        neighbor_index.append(neighbor_knn_search(x_NC, x_NC, k, chunk_size=chunk_size).unsqueeze(0)
    return torch.cat(neighbor_index, dim=0)


def get_graph_feature(x, k=20, idx=None):
    """
    Return KNN neighbor features
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = batched_self_knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device("cuda")
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(
        2, 1
    ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN(BaseModel):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        knn_k: int = 8,
        emb_dims: int = 1024,
        dropout: float = 0.5,
        knn_search_chunk_size: int = 4096,
    ):
        super(DGCNN, self).__init__()
        self.k = knn_k
        self.knn_search_chunk_size = knn_search_chunk_size
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.linear1 = nn.Linear(emb_dims * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, out_channels)

    def forward(self, x):
        batch_size = x.size(0)
        idx = batched_self_knn(x, k=self.k, chunk_size=self.knn_search_chunk_size)
        x = get_graph_feature(x, k=self.k, idx=idx)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k, idx=idx)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k, idx=idx)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k, idx=idx)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class DGCNNDrivAer(DrivAerDragRegressionBase, DGCNN):
    def __init__(
        self,
        knn_k: int,
        emb_dims: int = 1024,
        dropout: float = 0.5,
        knn_search_chunk_size: int = 4096,
    ):
        DrivAerDragRegressionBase.__init__(self)

        DGCNN.__init__(
            self,
            in_channels=3,
            out_channels=1,
            knn_k=knn_k,
            emb_dims=emb_dims,
            dropout=dropout,
            knn_search_chunk_size=knn_search_chunk_size,
        )

    def data_dict_to_input(self, data_dict) -> torch.Tensor:
        vertices = data_dict["cell_centers"]
        return vertices.float().to(self.device).transpose(1, 2)  # B, 3, N
