# ignore_header_test
# ruff: noqa: E402

# © Copyright 2023 HP Development Company, L.P.
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
import torch.nn as nn
import torch_geometric
from torch.nn import Linear as Lin
from torch.nn import Sequential as Seq
from torch_geometric.nn import DynamicEdgeConv, EdgeConv, knn_graph

import modulus  # noqa: F401 for docs

from ..meta import ModelMetaData
from ..module import Module


@dataclass
class MetaData(ModelMetaData):
    name: str = "GraphComPNet"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = True
    amp_cpu: bool = False  # Reflect padding not supported in bfloat16
    amp_gpu: bool = False
    # Inference
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


def MLP(channels, batch_norm=True):
    """
    Set up the MLP layer with NN.linear

    :param channels:    channel[0]:in_features
                        channel[1]:out_features
    :param batch_norm:
    :return: nn.Sequentially structured MLP model
    """
    return nn.Sequential(
        *[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i]), nn.ReLU()
            )  # , nn.BatchNorm1d(channels[i]))
            for i in range(1, len(channels))
        ]
    )


class DynamicEdgeConv2(EdgeConv):
    """
    A modified pytorch-geometric implementation of EdgeConv:
    https://pytorch-geometric.readthedocs.io/en/2.6.0/generated/torch_geometric.nn.conv.EdgeConv.html
    Original Paper: https://arxiv.org/abs/1801.07829
    """

    def __init__(self, nn, k, aggr="max", **kwargs):
        """

        :param nn: network architecture
        :param k:
        :param aggr:
        :param kwargs:
        """
        super(DynamicEdgeConv2, self).__init__(nn=nn, aggr=aggr, **kwargs)

        if knn_graph is None:
            raise ImportError("`DynamicEdgeConv` requires `torch-cluster`.")

        self.k = k

    def forward(self, x, edge_index):
        # edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow, cosine=True)
        return super(DynamicEdgeConv2, self).forward(x, edge_index)

    def __repr__(self):
        return "{}(nn={}, k={})".format(self.__class__.__name__, self.nn, self.k)


class DGCNN(Module):
    """
    A modified of EdgeConv blocks with the DGCNN backbone.
    Applies convolution to the edge features.

    Parameters
    ----------
    1st EdgeConv MLP: input feature channels=3*2, output feature channels=64
    2nd EdgeConv MLP: input feature channels=64*2, output feature channels=128
    3rd EdgeConv MLP: input feature channels=128*2, output feature channels=512

    Example
    -------
    >>> model = modulus.models.dgcnn.DGCNN()
    >>> sample_pts = torch.randn(10000, 3)
    >>> edge_index = torch_geometric.nn.knn_graph(torch.FloatTensor(sample_pts), 10)
    >>> sample_pts_data = torch_geometric.data.Data(x=sample_pts, edge_index=edge_index)
    >>> compensated_output = model(sample_pts_data)
    >>> compensated_output.size()

    Note
    ----
    Reference of DGCNN backbone: https://arxiv.org/pdf/1801.07829
    """

    def __init__(
        self,
        k: int = 20,
        aggr: str = "max",
    ):
        if not (k >= 0):
            raise ValueError("Invalid arch params")
        super().__init__(meta=MetaData(name="dgcnn"))

        self.conv1 = DynamicEdgeConv2(MLP([3 * 2, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv2(MLP([64 * 2, 128]), k, aggr)
        self.conv3 = DynamicEdgeConv2(MLP([128 * 2, 512]), k, aggr)

        self.lin1 = Seq(
            MLP([512, 256]),  #  Dropout(0.2), #MLP([512,256]), Dropout(0.2),
            Lin(256, 3),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.lin1(x3)
        return x + x4


class DGCNN_ocardo(Module):
    """
    Variation of EdgeConv blocks with the DGCNN backbone: https://arxiv.org/pdf/1801.07829
    Model architecture tuned for optimal performance on the Orcardo dataset

    Parameters
    ----------
    1st EdgeConv MLP: input feature channels=3*2, output feature channels=64
    2nd EdgeConv MLP: input feature channels=64*2, output feature channels=64
    3rd EdgeConv MLP: input feature channels=64*2, output feature channels=64
    4th EdgeConv MLP: input feature channels=64*2, output feature channels=64
    5th EdgeConv MLP: input feature channels=64*2, output feature channels=64

    Aggregation func of the last layer: Max

    Note
    ----
    Reference of DGCNN backbone: https://arxiv.org/pdf/1801.07829
    """

    def __init__(
        self,
        k: int = 5,
        aggr: str = "max",
    ):
        if not (k >= 0):
            raise ValueError("Invalid arch params")
        super().__init__(meta=MetaData(name="dgcnn_orcardo"))

        self.conv1 = DynamicEdgeConv2(MLP([3 * 2, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv2(MLP([64 * 2, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv2(MLP([64 * 2, 64]), k, aggr)
        self.conv4 = DynamicEdgeConv2(MLP([64 * 2, 64]), k, aggr)
        self.conv5 = DynamicEdgeConv2(MLP([64 * 2, 64]), k, aggr)

        self.lin1 = Seq(
            MLP([128, 128]),  #  Dropout(0.2), #MLP([512,256]), Dropout(0.2),
            # MLP([256, 256]),#  Dropout(0.2), #MLP([512,256]), Dropout(0.2),
            # Lin(256, 3)
            Lin(128, 3),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        n_pts, _ = x.shape
        x1 = self.conv1(x, edge_index)
        x2 = self.conv2(x1, edge_index)
        x3 = self.conv3(x2, edge_index)
        x4 = self.conv4(x3, edge_index)
        x5 = self.conv5(x4, edge_index)
        # extract global feature
        globals = torch.max(x5, 0, keepdim=True)[0]
        # concat local and global features
        feat = torch.cat([x5, globals.repeat(n_pts, 1)], 1)
        # MLP feature
        x6 = self.lin1(feat)
        # residual connection
        return x + x6
