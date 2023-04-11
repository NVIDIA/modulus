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
import dgl.function as fn

from typing import Any, Union
from torch import Tensor
from dgl import DGLGraph
from .mlp import MLP, TruncatedMLP, TruncatedMLPCuGraph
from .utils import concat_efeat_dgl_m2g_g2m, agg_concat_dgl, CuGraphCSC

try:
    from pylibcugraphops.pytorch.operators import (
        agg_concat_e2n,
        update_efeat_bipartite_e2e,
    )
except ImportError:
    agg_concat_e2n = None
    update_efeat_bipartite_e2e = None


class DecoderConcat(nn.Module):
    """GraphCast Mesh2Grid decoder

    Parameters
    ----------
    graph : DGLGraph | CuGraphCSC
        Graph structure representing the edges between mesh and grid
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
        graph: Union[DGLGraph, CuGraphCSC],
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
        self.graph = graph
        self.aggregation = aggregation
        self.use_cugraphops = isinstance(graph, CuGraphCSC)

        # edge MLP
        self.edge_mlp = MLP(
            input_dim=input_dim_src_nodes + input_dim_dst_nodes + input_dim_edges,
            output_dim=output_dim_edges,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # dst node MLP
        self.node_mlp = MLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    def forward(
        self, m2g_efeat: Tensor, grid_nfeat: Tensor, mesh_nfeat: Tensor
    ) -> Tensor:
        # update edge features through concatenating edge and node features
        if self.use_cugraphops:
            # torch.int64 to avoid indexing overflows due tu current behavior of cugraph-ops
            bipartite_graph = self.graph.to_bipartite_csc(dtype=torch.int64)
            efeat = update_efeat_bipartite_e2e(
                m2g_efeat, mesh_nfeat, grid_nfeat, bipartite_graph, "concat"
            )
        else:
            efeat = concat_efeat_dgl_m2g_g2m(
                m2g_efeat, mesh_nfeat, grid_nfeat, self.graph
            )

        # transform updated edge features
        efeat = self.edge_mlp(efeat)

        # aggregate messages (edge features) to obtain updated node features
        if self.use_cugraphops:
            static_graph = self.graph.to_static_csc()
            cat_feat = agg_concat_e2n(grid_nfeat, efeat, static_graph, self.aggregation)
        else:
            cat_feat = agg_concat_dgl(efeat, grid_nfeat, self.graph, self.aggregation)

        # transformation and residual connection
        dst_feat = self.node_mlp(cat_feat) + grid_nfeat
        return dst_feat

    def to(self, *args: Any, **kwargs: Any) -> "DecoderConcat":
        """Moves the object to the specified device, dtype, or format.
        This method moves the object and its underlying graph to the specified
        device, dtype, or format, and returns the updated object.

        Parameters
        ----------
        *args : Any
            Positional arguments to be passed to the `torch._C._nn._parse_to` function.
        **kwargs : Any
            Keyword arguments to be passed to the `torch._C._nn._parse_to` function.

        Returns
        -------
        DecoderDGLConcat
            The updated object after moving to the specified device, dtype, or format.
        """
        self = super().to(*args, **kwargs)
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.graph = self.graph.to(device=device)
        return self


class DecoderSum(nn.Module):
    """GraphCast Mesh2Grid decoder

    Parameters
    ----------
    graph : DGLGraph | CuGraphCSC
        Graph structure representing the edges between mesh and grid
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
        graph: Union[DGLGraph, CuGraphCSC],
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
        self.graph = graph
        self.aggregation = aggregation
        self.use_cugraphops = isinstance(graph, CuGraphCSC)

        if self.use_cugraphops:
            self.edge_trunc_mlp = TruncatedMLPCuGraph(
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

        else:
            self.src, self.dst = (item.long() for item in graph.edges())
            # edge MLP
            self.edge_trunc_mlp = TruncatedMLP(
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
        self.node_mlp = MLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    def forward(
        self, m2g_efeat: Tensor, grid_nfeat: Tensor, mesh_nfeat: Tensor
    ) -> Tensor:
        # update edge features and aggregate them to obtain updated node features
        if self.use_cugraphops:
            static_graph = self.graph.to_static_csc()
            bipartite_graph = self.graph.to_bipartite_csc()

            efeat = self.edge_trunc_mlp(
                m2g_efeat, mesh_nfeat, grid_nfeat, bipartite_graph
            )
            cat_feat = agg_concat_e2n(grid_nfeat, efeat, static_graph, self.aggregation)

        else:
            efeat = self.edge_trunc_mlp(
                m2g_efeat, mesh_nfeat, grid_nfeat, self.src, self.dst
            )
            cat_feat = agg_concat_dgl(efeat, grid_nfeat, self.graph, self.aggregation)

        # transform node features and apply residual connection
        dst_feat = self.node_mlp(cat_feat) + grid_nfeat
        return dst_feat

    def to(self, *args: Any, **kwargs: Any) -> "DecoderSum":
        """Moves the object to the specified device, dtype, or format.
        This method moves the object and its underlying graph to the specified
        device, dtype, or format, and returns the updated object.

        Parameters
        ----------
        *args : Any
            Positional arguments to be passed to the `torch._C._nn._parse_to` function.
        **kwargs : Any
            Keyword arguments to be passed to the `torch._C._nn._parse_to` function.

        Returns
        -------
        DecoderDGLSum
            The updated object after moving to the specified device, dtype, or format.
        """
        self = super().to(*args, **kwargs)
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.graph = self.graph.to(device=device)
        return self
