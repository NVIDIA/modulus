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

from typing import Any
from torch import Tensor
from dgl import DGLGraph
from .mlp import MLP, TMLPDGL, TMLPCUGO
from .utils import concat_efeat_dgl_m2g_g2m, agg_concat_dgl

try:
    from pylibcugraphops.torch.autograd import agg_concat_e2n, update_efeat_e2e
    from pylibcugraphops.typing import MfgCsr
except ImportError:
    agg_concat_e2n = None
    update_efeat_e2e = None
    MfgCsr = None


class DecoderDGLConcat(nn.Module):
    """GraphCast Mesh2Grid decoder

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
        output_dim_dst_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__()
        self.graph = graph
        self.src, self.dst = (item.long() for item in graph.edges())
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

        # dst node MLP
        self.node_MLP = MLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(
        self, m2g_efeat: Tensor, grid_nfeat: Tensor, mesh_nfeat: Tensor
    ) -> Tensor:
        efeat = concat_efeat_dgl_m2g_g2m(m2g_efeat, mesh_nfeat, grid_nfeat, self.graph)
        efeat = self.edge_MLP(efeat)
        cat_feat = agg_concat_dgl(efeat, grid_nfeat, self.graph, self.aggregation)
        dst_feat = self.node_MLP(cat_feat) + grid_nfeat

        return dst_feat

    def to(self, *args: Any, **kwargs: Any) -> DecoderDGLConcat:
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
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        self.graph = self.graph.to(device=device, dtype=dtype)
        self = super().to(*args, **kwargs)
        return self


class DecoderDGLSum(nn.Module):
    """GraphCast Mesh2Grid decoder

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
        output_dim_dst_nodes: int = 512,
        output_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
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

        # dst node MLP
        self.node_MLP = MLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(
        self, m2g_efeat: Tensor, grid_nfeat: Tensor, mesh_nfeat: Tensor
    ) -> Tensor:
        efeat = self.edge_TMLP(m2g_efeat, mesh_nfeat, grid_nfeat, self.src, self.dst)
        cat_feat = agg_concat_dgl(efeat, grid_nfeat, self.graph, self.aggregation)
        dst_feat = self.node_MLP(cat_feat) + grid_nfeat
        return dst_feat

    def to(self, *args: Any, **kwargs: Any) -> DecoderDGLSum:
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
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        self.graph = self.graph.to(device=device, dtype=dtype)
        self = super().to(*args, **kwargs)
        return self


class DecoderCUGOConcat(nn.Module):
    """GraphCast Mesh2Grid decoder

    Parameters
    ----------
    graph : MfgCsr
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
    """

    def __init__(
        self,
        graph: MfgCsr,
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

        # dst node MLP
        self.node_MLP = MLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def custom_forward(
        self, m2g_efeat: Tensor, grid_nfeat: Tensor, mesh_nfeat: Tensor
    ) -> Tensor:
        """DecoderCUGOConcat forward method with support for gradient checkpointing.

        Parameters
        ----------
        m2g_efeat : Tensor
            Edge features for the mesh2grid graph.
        grid_nfeat : Tensor
            Node features for the latitude-longitude grid.
        mesh_nfeat : Tensor
            Node features for the multimesh graph.

        Returns
        -------
        dst_feat: Tensor
            Predicted node features on the latitude-longitude grid.
        """
        efeat = update_efeat_e2e(
            m2g_efeat, mesh_nfeat, grid_nfeat, self.graph, "concat"
        )
        efeat = self.edge_MLP(efeat)
        cat_feat = agg_concat_e2n(grid_nfeat, efeat, self.graph, self.aggregation)
        dst_feat = self.node_MLP(cat_feat) + grid_nfeat
        return dst_feat

    def to(self, *args: Any, **kwargs: Any) -> DecoderCUGOConcat:
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
        DecoderCUGOConcat
            The updated object after moving to the specified device, dtype, or format.
        """
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        self.graph = self.graph.to(device=device, dtype=dtype)
        self = super().to(*args, **kwargs)
        return self


class DecoderCUGOSum(nn.Module):
    """GraphCast Mesh2Grid decoder

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
        graph: MfgCsr,
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
    ):
        super().__init__()
        self.graph = graph
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

        # dst node MLP
        self.node_MLP = MLP(
            input_dim=input_dim_dst_nodes + output_dim_edges,
            output_dim=output_dim_dst_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(
        self, m2g_efeat: Tensor, grid_nfeat: Tensor, mesh_nfeat: Tensor
    ) -> Tensor:
        efeat = self.edge_TMLP(m2g_efeat, mesh_nfeat, grid_nfeat, self.graph)
        cat_feat = agg_concat_e2n(grid_nfeat, efeat, self.graph, self.aggregation)
        dst_feat = self.node_MLP(cat_feat) + grid_nfeat
        return dst_feat

    def to(self, *args: Any, **kwargs: Any) -> DecoderCUGOSum:
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
        DecoderCUGOSum
            The updated object after moving to the specified device, dtype, or format.
        """
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        self.graph = self.graph.to(device=device, dtype=dtype)
        self = super().to(*args, **kwargs)
        return self
