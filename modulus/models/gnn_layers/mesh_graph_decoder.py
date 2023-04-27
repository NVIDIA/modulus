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

import torch
import torch.nn as nn

from typing import Union
from torch import Tensor
from dgl import DGLGraph
from .mesh_graph_mlp import MeshGraphMLP, MeshGraphEdgeMLPConcat, MeshGraphEdgeMLPSum
from .utils import aggregate_and_concat, CuGraphCSC


class MeshGraphDecoder(nn.Module):
    """Decoder used e.g. in GraphCast
       which acts on the bipartite graph connecting a mesh
       (e.g. representing a latent space) to a mostly regular
       grid (e.g. representing the output domain).

    Parameters
    ----------
    aggregation : str, optional
        Message passing aggregation method ("sum", "mean"), by default "sum"
    input_dim_src_nodes : int, optional
        Input dimensionality of the source node features, by default 512
    input_dim_dst_nodes : int, optional
        Input dimensionality of the destination node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim_dst_nodes : int, optional
        Output dimensionality of the destination node features, by default 512
    output_dim_edges : int, optional
        Output dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        Number of hiddel layers, by default 1
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type, by default "LayerNorm"
    do_conat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_src_nodes: int = 512,
        input_dim_dst_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim_dst_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
    ):
        super().__init__()
        self.aggregation = aggregation

        MLP = MeshGraphEdgeMLPSum if do_concat_trick else MeshGraphEdgeMLPConcat
        # edge MLP
        self.edge_mlp = MLP(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_src_nodes,
            dst_dim=input_dim_dst_nodes,
            output_dim=output_dim_edges,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # dst node MLP
        self.node_mlp = MeshGraphMLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    @torch.jit.ignore()
    def forward(
        self,
        m2g_efeat: Tensor,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tensor:
        # update edge features
        efeat = self.edge_mlp(m2g_efeat, (mesh_nfeat, grid_nfeat), graph)
        # aggregate messages (edge features) to obtain updated node features
        cat_feat = aggregate_and_concat(efeat, grid_nfeat, graph, self.aggregation)
        # transformation and residual connection
        dst_feat = self.node_mlp(cat_feat) + grid_nfeat
        return dst_feat
