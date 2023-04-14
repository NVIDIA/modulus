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

from torch import Tensor
from dgl import DGLGraph
from typing import Any, Tuple, Union
from .mlp import MLP, TruncatedMLP, TruncatedMLPCuGraph
from .utils import agg_concat_dgl, concat_efeat_dgl_m2g_g2m, CuGraphCSC

try:
    from pylibcugraphops.pytorch.operators import (
        agg_concat_e2n,
        update_efeat_bipartite_e2e,
    )
except:
    agg_concat_e2n = None
    update_efeat_bipartite_e2e = None
    BipartiteCSC = None


class EncoderConcat(nn.Module):
    """GraphCast Grid2Mesh encoder

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
    output_dim_src_nodes : int, optional
        Output dimensionality of the source node features, by default 512
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
        output_dim_src_nodes: int = 512,
        output_dim_dst_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: int = nn.SiLU(),
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

        # src node MLP
        self.src_node_mlp = MLP(
            input_dim=input_dim_src_nodes,
            output_dim=output_dim_src_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # dst node MLP
        self.dst_node_mlp = MLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    def forward(
        self, g2m_efeat: Tensor, grid_nfeat: Tensor, mesh_nfeat: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # update edge features by concatenating node features (both mesh and grid) and existing edger featues
        # torch.int64 to avoid indexing overflows due tu current behavior of cugraph-ops
        if self.use_cugraphops:
            bipartite_graph = self.graph.to_bipartite_csc(dtype=torch.int64)
            efeat = update_efeat_bipartite_e2e(
                g2m_efeat, grid_nfeat, mesh_nfeat, bipartite_graph, mode="concat"
            )
        else:
            efeat = concat_efeat_dgl_m2g_g2m(
                g2m_efeat, grid_nfeat, mesh_nfeat, self.graph
            )

        # transform edge features
        efeat = self.edge_mlp(efeat)

        # aggregate messages (edge features) to obtain updated node features
        if self.use_cugraphops:
            static_graph = self.graph.to_static_csc()
            cat_feat = agg_concat_e2n(mesh_nfeat, efeat, static_graph, self.aggregation)
        else:
            cat_feat = agg_concat_dgl(efeat, mesh_nfeat, self.graph, self.aggregation)

        # update src, dst node features + residual connections
        mesh_nfeat = mesh_nfeat + self.dst_node_mlp(cat_feat)
        grid_nfeat = grid_nfeat + self.src_node_mlp(grid_nfeat)
        return grid_nfeat, mesh_nfeat

    def to(self, *args: Any, **kwargs: Any) -> "EncoderConcat":
        """Moves the object to the specified device, dtype, or format.
        This method moves the object and its underlying graph and graph features to
        the specified device, dtype, or format, and returns the updated object.

        Parameters
        ----------
        *args : Any
            Positional arguments to be passed to the `torch._C._nn._parse_to` function.
        **kwargs : Any
            Keyword arguments to be passed to the `torch._C._nn._parse_to` function.

        Returns
        -------
        EncoderDGLConcat
            The updated object after moving to the specified device, dtype, or format.
        """
        self = super().to(*args, **kwargs)
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.graph = self.graph.to(device=device)
        return self


class EncoderSum(nn.Module):
    """GraphCast Grid2Mesh encoder

    Parameters
    ----------
    graph : DGLGraph | CuGraphCSC
        graph structure representing the edges between mesh and grid
    aggregation : str, optional
        message passing aggregation method ("sum", "mean"), by default "sum"
    input_dim_src_nodes : int, optional
        input dimensionality of the source node features, by default 512
    input_dim_dst_nodes : int, optional
        input dimensionality of the destination node features, by default 512
    input_dim_edges : int, optional
        input dimensionality of the edge features, by default 512
    output_dim_src_nodes : int, optional
        output dimensionality of the source node features, by default 512
    output_dim_dst_nodes : int, optional
        output dimensionality of the destination node features, by default 512
    output_dim_edges : int, optional
        output dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        number of hiddel layers, by default 1
    activation_fn : nn.Module, optional
        type of activation function, by default nn.SiLU()
    norm_type : str, optional
        normalization type, by default "LayerNorm"
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
        output_dim_src_nodes: int = 512,
        output_dim_dst_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: int = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()
        self.graph = graph
        self.aggregation = aggregation
        self.use_cugraphops = isinstance(graph, CuGraphCSC)

        if self.use_cugraphops:
            # edge MLP
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

        # src node MLP
        self.src_node_mlp = MLP(
            input_dim=input_dim_src_nodes,
            output_dim=output_dim_src_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # dst node MLP
        self.dst_node_mlp = MLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    def forward(
        self, g2m_efeat: Tensor, grid_nfeat: Tensor, mesh_nfeat: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.use_cugraphops:
            bipartite_graph = self.graph.to_bipartite_csc()
            static_graph = self.graph.to_static_csc()

            mlp_efeat = self.edge_trunc_mlp(
                g2m_efeat, grid_nfeat, mesh_nfeat, bipartite_graph
            )
            cat_feat = agg_concat_e2n(
                mesh_nfeat, mlp_efeat, static_graph, self.aggregation
            )
        else:
            # update edge features with Truncated MLP
            mlp_efeat = self.edge_trunc_mlp(
                g2m_efeat, grid_nfeat, mesh_nfeat, self.src, self.dst
            )
            # aggregate messages (edge features) to obtain updated node features
            cat_feat = agg_concat_dgl(
                mlp_efeat, mesh_nfeat, self.graph, self.aggregation
            )

        # update src-feat, dst-feat and apply residual connections
        mesh_nfeat = mesh_nfeat + self.dst_node_mlp(cat_feat)
        grid_nfeat = grid_nfeat + self.src_node_mlp(grid_nfeat)
        return grid_nfeat, mesh_nfeat

    def to(self, *args: Any, **kwargs: Any) -> "EncoderSum":
        """Moves the object to the specified device, dtype, or format.
        This method moves the object and its underlying graph and graph features to
        the specified device, dtype, or format, and returns the updated object.

        Parameters
        ----------
        *args : Any
            Positional arguments to be passed to the `torch._C._nn._parse_to` function.
        **kwargs : Any
            Keyword arguments to be passed to the `torch._C._nn._parse_to` function.

        Returns
        -------
        EncoderDGLSum
            The updated object after moving to the specified device, dtype, or format.
        """
        self = super().to(*args, **kwargs)
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.graph = self.graph.to(device=device)
        return self
