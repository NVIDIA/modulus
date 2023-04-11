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
from typing import Any, Tuple
from .mlp import MLP, TMLPDGL, TMLPCUGO
from .utils import agg_concat_dgl, concat_efeat_dgl_m2g_g2m, CuGraphCSC

try:
    from pylibcugraphops.pytorch.operators import (
        agg_concat_e2n,
        update_efeat_bipartite_e2e,
    )
    from pylibcugraphops.pytorch import BipartiteCSC
except:
    agg_concat_e2n = None
    update_efeat_bipartite_e2e = None
    BipartiteCSC = None


class EncoderDGLConcat(nn.Module):
    """GraphCast Grid2Mesh encoder

    Parameters
    ----------
    graph : DGLGraph
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
    """

    def __init__(
        self,
        graph: DGLGraph,
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
    ):
        super().__init__()
        self.graph = graph
        self.aggregation = aggregation

        # edge MLP
        self.edge_MLP = MLP(
            input_dim=input_dim_src_nodes + input_dim_dst_nodes + input_dim_edges,
            output_dim=output_dim_edges,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

        # src node MLP
        self.src_node_MLP = MLP(
            input_dim=input_dim_src_nodes,
            output_dim=output_dim_src_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

        # dst node MLP
        self.dst_node_MLP = MLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(
        self, g2m_efeat: Tensor, grid_nfeat: Tensor, mesh_nfeat: Tensor
    ) -> Tuple[Tensor, Tensor]:
        efeat = concat_efeat_dgl_m2g_g2m(g2m_efeat, grid_nfeat, mesh_nfeat, self.graph)
        efeat = self.edge_MLP(efeat)
        cat_feat = agg_concat_dgl(efeat, mesh_nfeat, self.graph, self.aggregation)

        # update src, dst node features + residual connections
        mesh_nfeat = mesh_nfeat + self.dst_node_MLP(cat_feat)
        grid_nfeat = grid_nfeat + self.src_node_MLP(grid_nfeat)
        # TODO (mnabian) verify edge update is not needed (Eq. A.10)
        return grid_nfeat, mesh_nfeat

    def to(self, *args: Any, **kwargs: Any) -> "EncoderDGLConcat":
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
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.graph = self.graph.to(device=device)
        return self


class EncoderDGLSum(nn.Module):
    """GraphCast Grid2Mesh encoder

    Parameters
    ----------
    graph : DGLGraph
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
    """

    def __init__(
        self,
        graph: DGLGraph,
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
    ):
        super().__init__()
        self.graph = graph
        self.src, self.dst = (item.long() for item in graph.edges())
        self.aggregation = aggregation

        # edge MLP
        self.edge_TMLP = TMLPDGL(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_src_nodes,
            dst_dim=input_dim_dst_nodes,
            output_dim=output_dim_edges,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            bias=True,
        )

        # src node MLP
        self.src_node_MLP = MLP(
            input_dim=input_dim_src_nodes,
            output_dim=output_dim_src_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

        # dst node MLP
        self.dst_node_MLP = MLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(
        self, g2m_efeat: Tensor, grid_nfeat: Tensor, mesh_nfeat: Tensor
    ) -> Tuple[Tensor, Tensor]:
        mlp_efeat = self.edge_TMLP(
            g2m_efeat, grid_nfeat, mesh_nfeat, self.src, self.dst
        )
        cat_feat = agg_concat_dgl(mlp_efeat, mesh_nfeat, self.graph, self.aggregation)
        mesh_nfeat = mesh_nfeat + self.dst_node_MLP(cat_feat)
        grid_nfeat = grid_nfeat + self.src_node_MLP(grid_nfeat)
        # TODO (mnabian) verify edge update is not needed (Eq. A.10)
        return grid_nfeat, mesh_nfeat

    def to(self, *args: Any, **kwargs: Any) -> "EncoderDGLSum":
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


class EncoderCUGOConcat(nn.Module):
    """GraphCast Grid2Mesh encoder

    Parameters
    ----------
    graph : CuGraphCSC
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
    """

    def __init__(
        self,
        graph: CuGraphCSC,
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
    ):  # pragma: no cover
        super().__init__()
        self.graph = graph
        self.static_graph = None
        self.bipartite_graph = None

        self.aggregation = aggregation

        # edge MLP
        self.edge_MLP = MLP(
            input_dim=input_dim_src_nodes + input_dim_dst_nodes + input_dim_edges,
            output_dim=output_dim_edges,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

        # src node MLP
        self.src_node_MLP = MLP(
            input_dim=input_dim_src_nodes,
            output_dim=output_dim_src_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

        # dst node MLP
        self.dst_node_MLP = MLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(
        self, g2m_efeat: Tensor, grid_nfeat: Tensor, mesh_nfeat: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.bipartite_graph is None:
            self.bipartite_graph = self.graph.to_bipartite_csc()
        if self.static_graph is None:
            # torch.int64 to avoid indexing overflows due tu current behavior of cugraph-ops
            self.static_graph = self.graph.to_static_csc(dtype=torch.int64)

        efeat = update_efeat_bipartite_e2e(
            g2m_efeat, grid_nfeat, mesh_nfeat, self.bipartite_graph, mode="concat"
        )
        efeat = self.edge_MLP(efeat)
        cat_feat = agg_concat_e2n(
            mesh_nfeat, efeat, self.static_graph, self.aggregation
        )

        # update src, dst node features + residual connections
        mesh_nfeat = mesh_nfeat + self.dst_node_MLP(cat_feat)
        grid_nfeat = grid_nfeat + self.src_node_MLP(grid_nfeat)
        # TODO (mnabian) verify edge update is not needed (Eq. A.10)
        return grid_nfeat, mesh_nfeat

    def to(self, *args: Any, **kwargs: Any) -> "EncoderCUGOConcat":  # pragma: no cover
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
        EncoderCUGOConcat
            The updated object after moving to the specified device, dtype, or format.
        """
        self = super().to(*args, **kwargs)
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.graph = self.graph.to(device=device)
        return self


class EncoderCUGOSum(nn.Module):
    """GraphCast Grid2Mesh encoder

    Parameters
    ----------
    graph : MfgCsr
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
    """

    def __init__(
        self,
        graph: CuGraphCSC,
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
    ):  # pragma: no cover
        super().__init__()
        self.graph = graph
        self.bipartite_graph = None
        self.static_graph = None
        self.aggregation = aggregation

        # edge MLP
        self.edge_TMLP = TMLPCUGO(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_src_nodes,
            dst_dim=input_dim_dst_nodes,
            output_dim=output_dim_edges,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            bias=True,
        )

        # src node MLP
        self.src_node_MLP = MLP(
            input_dim=input_dim_src_nodes,
            output_dim=output_dim_src_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

        # dst node MLP
        self.dst_node_MLP = MLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(
        self, g2m_efeat: Tensor, grid_nfeat: Tensor, mesh_nfeat: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.bipartite_graph is None:
            self.bipartite_graph = self.graph.to_bipartite_csc()
        if self.static_graph is None:
            self.static_graph = self.graph.to_static_csc()

        mlp_efeat = self.edge_TMLP(
            g2m_efeat, grid_nfeat, mesh_nfeat, self.bipartite_graph
        )
        cat_feat = agg_concat_e2n(
            mesh_nfeat, mlp_efeat, self.static_graph, self.aggregation
        )
        mesh_nfeat = mesh_nfeat + self.dst_node_MLP(cat_feat)
        grid_nfeat = grid_nfeat + self.src_node_MLP(grid_nfeat)
        # TODO (mnabian) verify edge update is not needed (Eq. A.10)
        return grid_nfeat, mesh_nfeat

    def to(self, *args: Any, **kwargs: Any) -> "EncoderCUGOSum":  # pragma: no cover
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
        EncoderCUGOSum
            The updated object after moving to the specified device, dtype, or format.
        """
        self = super().to(*args, **kwargs)
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.graph = self.graph.to(device=device)
        return self
