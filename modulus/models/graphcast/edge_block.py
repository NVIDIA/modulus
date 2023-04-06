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

from typing import Any
from torch import Tensor
from dgl import DGLGraph
from .mlp import MLP, TMLPDGL, TMLPCUGO
from .utils import concat_efeat_dgl_mesh, CuGraphCSC

try:
    from pylibcugraphops.pytorch.operators import update_efeat_static_e2e
except ModuleNotFoundError:
    update_efeat_static_e2e = None


class EdgeBlockDGLConcat(nn.Module):
    """Edge block for DGL graphs with concatenation.

    Parameters
    ----------
    graph : DGLGraph
        Graph.
    input_dim_nodes : int, optional
        Input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim : int, optional
        Output dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        _description_, by default 512
    hidden_layers : int, optional
        Number of neurons in each hidden layer, by default 1
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        normalization type, by default "LayerNorm"
    """

    def __init__(
        self,
        graph: DGLGraph,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__()
        self.graph = graph

        self.edge_MLP = MLP(
            input_dim=2 * input_dim_nodes + input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
    ) -> Tensor:
        cat_feat = concat_efeat_dgl_mesh(efeat, nfeat, self.graph)
        efeat_new = self.edge_MLP(cat_feat) + efeat

        return efeat_new, nfeat

    def to(self, *args: Any, **kwargs: Any) -> EdgeBlockDGLConcat:
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
        EdgeBlockDGLConcat
            The updated object after moving to the specified device, dtype, or format.
        """
        self = super().to(*args, **kwargs)
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.graph = self.graph.to(device=device)
        return self


class EdgeBlockDGLSum(nn.Module):
    """Edge block for DGL graphs with summation.

    Parameters
    ----------
    graph : DGLGraph
        Graph.
    input_dim_nodes : int, optional
        Input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim : int, optional
        Output dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        _description_, by default 512
    hidden_layers : int, optional
        Number of neurons in each hidden layer, by default 1
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        normalization type, by default "LayerNorm"
    """

    def __init__(
        self,
        graph: DGLGraph,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__()
        self.graph = graph
        self.static_graph = None
        self.src, self.dst = (item.long() for item in self.graph.edges())

        self.edge_TMLP = TMLPDGL(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_nodes,
            dst_dim=input_dim_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            bias=True,
        )

    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
    ) -> Tensor:
        efeat_new = self.edge_TMLP(efeat, nfeat, nfeat, self.src, self.dst) + efeat
        return efeat_new, nfeat

    def to(self, *args: Any, **kwargs: Any) -> EdgeBlockDGLSum:
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
        EdgeBlockDGLSum
            The updated object after moving to the specified device, dtype, or format.
        """
        self = super().to(*args, **kwargs)
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.graph = self.graph.to(device=device)
        return self


class EdgeBlockCUGOConcat(nn.Module):
    """Edge block for CuGraph graphs with concatenation.

    Parameters
    ----------
    graph : CuGraphCSC
        Graph.
    input_dim_nodes : int, optional
        Input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim : int, optional
        Output dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        _description_, by default 512
    hidden_layers : int, optional
        Number of neurons in each hidden layer, by default 1
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        normalization type, by default "LayerNorm"
    """

    def __init__(
        self,
        graph: CuGraphCSC,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__()
        self.graph = graph

        self.edge_MLP = MLP(
            input_dim=(2 * input_dim_nodes + input_dim_edges),
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
        )

    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
    ) -> Tensor:
        if self.static_graph is None:
            self.static_graph = self.graph.to_static_csc()

        cat_feat = update_efeat_static_e2e(
            efeat,
            nfeat,
            self.static_graph,
            mode="concat",
            use_source_emb=True,
            use_target_emb=True,
        )
        efeat_new = self.edge_MLP(cat_feat) + efeat
        return efeat_new, nfeat

    def to(self, *args: Any, **kwargs: Any) -> EdgeBlockCUGOConcat:
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
        EdgeBlockCUGOConcat
            The updated object after moving to the specified device, dtype, or format.
        """
        self = super().to(*args, **kwargs)
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.graph = self.graph.to(device=device)
        return self


class EdgeBlockCUGOSum(nn.Module):
    """Edge block for CuGraph graphs with summation.

    Parameters
    ----------
    graph : DGLGraph
        Graph.
    input_dim_nodes : int, optional
        Input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim : int, optional
        Output dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        _description_, by default 512
    hidden_layers : int, optional
        Number of neurons in each hidden layer, by default 1
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        normalization type, by default "LayerNorm"
    """

    def __init__(
        self,
        graph: CuGraphCSC,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
    ):
        super().__init__()
        self.graph = graph
        self.bipartite_graph = None

        self.edge_TMLP = TMLPCUGO(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_nodes,
            dst_dim=input_dim_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            bias=True,
        )

    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
    ) -> Tensor:
        if self.bipartite_graph is None:
            self.bipartite_graph = self.graph.to_bipartite_csc()

        efeat_new = self.edge_TMLP(efeat, nfeat, nfeat, self.bipartite_graph) + efeat
        return efeat_new, nfeat

    def to(self, *args: Any, **kwargs: Any) -> EdgeBlockCUGOSum:
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
        EdgeBlockCUGOSum
            The updated object after moving to the specified device, dtype, or format.
        """
        self = super().to(*args, **kwargs)
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.graph = self.graph.to(device=device)
        return self
