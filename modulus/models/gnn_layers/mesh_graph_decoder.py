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
from .mesh_graph_mlp import MeshGraphMLP, TruncatedMeshGraphMLP
from .utils import concat_efeat_dgl, agg_concat_dgl, CuGraphCSC

try:
    from pylibcugraphops.pytorch.operators import (
        agg_concat_e2n,
        update_efeat_bipartite_e2e,
    )
except ImportError:
    agg_concat_e2n = None
    update_efeat_bipartite_e2e = None


class MeshGraphDecoderConcat(nn.Module):
    """Decoder used e.g. in GraphCast or MeshGraphNet
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
        recompute_activation: bool = False,
    ):
        super().__init__()
        self.aggregation = aggregation

        # edge MLP
        self.edge_mlp = MeshGraphMLP(
            input_dim=input_dim_src_nodes + input_dim_dst_nodes + input_dim_edges,
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
        # update edge features through concatenating edge and node features
        if isinstance(graph, CuGraphCSC):
            # torch.int64 to avoid indexing overflows due tu current behavior of cugraph-ops
            bipartite_graph = graph.to_bipartite_csc(dtype=torch.int64)
            efeat = update_efeat_bipartite_e2e(
                m2g_efeat, mesh_nfeat, grid_nfeat, bipartite_graph, "concat"
            )
        else:
            efeat = concat_efeat_dgl(m2g_efeat, (mesh_nfeat, grid_nfeat), graph)

        # transform updated edge features
        efeat = self.edge_mlp(efeat)

        # aggregate messages (edge features) to obtain updated node features
        if isinstance(graph, CuGraphCSC):
            static_graph = graph.to_static_csc()
            cat_feat = agg_concat_e2n(grid_nfeat, efeat, static_graph, self.aggregation)
        else:
            cat_feat = agg_concat_dgl(efeat, grid_nfeat, graph, self.aggregation)

        # transformation and residual connection
        dst_feat = self.node_mlp(cat_feat) + grid_nfeat
        return dst_feat


class MeshGraphDecoderSum(nn.Module):
    """Decoder used e.g. in GraphCast or MeshGraphNet
       which acts on the bipartite graph connecting a mesh
       (e.g. representing a latent space) to a mostly regular
       grid (e.g. representing the output domain). This variant
       makes use of the `Concat-Trick` which transforms Concat+MMA
       into MMA+Sum in its first linear layer.

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
        recompute_activation: bool = False,
    ):
        super().__init__()
        self.aggregation = aggregation

        # edge MLP
        self.edge_trunc_mlp = TruncatedMeshGraphMLP(
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
        efeat = self.edge_trunc_mlp(m2g_efeat, mesh_nfeat, grid_nfeat, graph)

        # update edge features and aggregate them to obtain updated node features
        if isinstance(graph, CuGraphCSC):
            static_graph = graph.to_static_csc()
            cat_feat = agg_concat_e2n(grid_nfeat, efeat, static_graph, self.aggregation)

        else:
            efeat = self.edge_trunc_mlp(
                m2g_efeat,
                mesh_nfeat,
                grid_nfeat,
                graph,
            )
            cat_feat = agg_concat_dgl(efeat, grid_nfeat, graph, self.aggregation)

        # transform node features and apply residual connection
        dst_feat = self.node_mlp(cat_feat) + grid_nfeat
        return dst_feat
