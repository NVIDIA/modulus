# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import sum_efeat_dgl

try:
    from pylibcugraphops.pytorch.operators import update_efeat_bipartite_e2e
    from pylibcugraphops.pytorch import BipartiteCSC
except:
    update_efeat_bipartite_e2e = None
    BipartiteCSC = None

try:
    from apex.normalization import FusedLayerNorm

    apex_imported = True
except:
    apex_imported = False


class MLP(nn.Module):
    """MLP with normalization in the last layer

    Parameters
    ----------
    input_dim : int
        dimensionality of the input features
    output_dim : int, optional
        dimensionality of the output features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        number of hidden layers, by default 1
    activation_fn : nn.Module, optional
        , by default nn.SiLU()
    norm_type : str, optional
        normalization type, by default "LayerNorm"
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), activation_fn]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
        layers.append(nn.Linear(hidden_dim, output_dim))

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]
            if norm_type == "LayerNorm" and apex_imported:
                norm_layer = FusedLayerNorm
            else:
                norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class TMLPDGL(nn.Module):
    """Truncated MLP where concat+MLP is replaced by MLP+sum

    Parameters
    ----------
    efeat_dim: int
        dimension of the input edge features
    src_dim: int
        dimension of the input src-node features
    dst_dim: int
        dimension of the input dst-node features
    output_dim : int, optional
        dimensionality of the output features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        number of hidden layers, by default 1
    activation_fn : nn.Module, optional
        type of activation function, by default nn.SiLU()
    norm_type : str, optional
        normalization type, by default "LayerNorm"
    bias : bool, optional
        whether to use bias in the MLP, by default True
    """

    def __init__(
        self,
        efeat_dim: int,
        src_dim: int,
        dst_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        bias: bool = True,
    ):
        super().__init__()

        self.efeat_dim = efeat_dim
        self.src_dim = src_dim
        self.dst_dim = dst_dim

        # this should ensure the same sequence of initializations
        # as the original MLP-Layer in combination with a concat operation
        tmp_lin = nn.Linear(efeat_dim + src_dim + dst_dim, hidden_dim, bias=bias)
        # orig_weight has shape (hidden_dim, efeat_dim + src_dim + dst_dim)
        orig_weight = tmp_lin.weight
        w_efeat, w_src, w_dst = torch.split(
            orig_weight, [efeat_dim, src_dim, dst_dim], dim=1
        )
        self.lin_efeat = nn.Parameter(w_efeat)
        self.lin_src = nn.Parameter(w_src)
        self.lin_dst = nn.Parameter(w_dst)

        if bias:
            self.bias = tmp_lin.bias
        else:
            self.bias = None

        layers = [activation_fn]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
        layers.append(nn.Linear(hidden_dim, output_dim))

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]
            if norm_type == "LayerNorm" and apex_imported:
                norm_layer = FusedLayerNorm
            else:
                norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(output_dim))

        self.model = nn.Sequential(*layers)

    def forward(
        self,
        efeat: Tensor,
        src_feat: Tensor,
        dst_feat: Tensor,
        src_idx: Tensor,
        dst_idx: Tensor,
    ) -> Tensor:
        # separate linear layers without bias
        # bias is added to one MLP, as we sum afterwards anyways
        # this adds the bias to the total sum, too
        # having it in one F.linear should allow a fusion of the bias addition
        # while avoiding adding the bias to the "edge-level" result
        mlp_efeat = F.linear(efeat, self.lin_efeat, None)
        mlp_src = F.linear(src_feat, self.lin_src, None)
        mlp_dst = F.linear(dst_feat, self.lin_dst, self.bias)
        mlp_sum = sum_efeat_dgl(mlp_efeat, mlp_src, mlp_dst, src_idx, dst_idx)
        return self.model(mlp_sum)


class TMLPCUGO(nn.Module):
    """Truncated MLP where concat+MLP is replaced
    by MLP+sum

    Parameters
    ----------
    efeat_dim: int
        dimension of the input edge features
    src_dim: int
        dimension of the input src-node features
    dst_dim: int
        dimension of the input dst-node features
    output_dim : int, optional
        dimensionality of the output features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        number of hidden layers, by default 1
    activation_fn : nn.Module, optional
        type of activation function, by default nn.SiLU()
    norm_type : str, optional
        normalization type, by default "LayerNorm"
    bias : bool, optional
        whether to use bias in the MLP, by default True
    """

    def __init__(
        self,
        efeat_dim: int,
        src_dim: int,
        dst_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        bias: bool = True,
    ):
        super().__init__()

        self.efeat_dim = efeat_dim
        self.src_dim = src_dim
        self.dst_dim = dst_dim

        # this should ensure the same sequence of initializations
        # as the original MLP-Layer in combination with a concat operation
        tmp_lin = nn.Linear(efeat_dim + src_dim + dst_dim, hidden_dim, bias=bias)
        # orig_weight has shape (hidden_dim, efeat_dim + src_dim + dst_dim)
        orig_weight = tmp_lin.weight
        w_efeat, w_src, w_dst = torch.split(
            orig_weight, [efeat_dim, src_dim, dst_dim], dim=1
        )
        self.lin_efeat = nn.Parameter(w_efeat)
        self.lin_src = nn.Parameter(w_src)
        self.lin_dst = nn.Parameter(w_dst)

        if bias:
            self.bias = tmp_lin.bias
        else:
            self.bias = None

        layers = [activation_fn]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
        layers.append(nn.Linear(hidden_dim, output_dim))

        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]
            if norm_type == "LayerNorm" and apex_imported:
                norm_layer = FusedLayerNorm
            else:
                norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(output_dim))

        self.model = nn.Sequential(*layers)

    def forward(
        self,
        efeat: Tensor,
        src_feat: Tensor,
        dst_feat: Tensor,
        graph: BipartiteCSC,
    ) -> Tensor:
        # separate linear layers without bias
        mlp_efeat = F.linear(efeat, self.lin_efeat, None)
        mlp_src = F.linear(src_feat, self.lin_src, None)
        mlp_dst = F.linear(dst_feat, self.lin_dst, self.bias)
        mlp_sum = update_efeat_bipartite_e2e(
            mlp_efeat, mlp_src, mlp_dst, graph, mode="sum"
        )
        return self.model(mlp_sum)
