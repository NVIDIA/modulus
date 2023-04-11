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

from .mlp import MLP
from .utils import agg_concat_dgl, CuGraphCSC

try:
    from pylibcugraphops.pytorch.operators import agg_concat_e2n
except:
    agg_concat_e2n = None


class NodeBlockDGL(nn.Module):
    """Node block for DGLGraph

    Parameters
    ----------
    graph : DGLGraph
        Graph.
    aggregation : str, optional
        Aggregation method (sum, mean) , by default "sum"
    input_dim_nodes : int, optional
        Input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim : int, optional
        Output dimensionality of the node features, by default 512
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        Number of neurons in each hidden layer, by default 1
    activation_fn : nn.Module, optional
       Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type, by default "LayerNorm"
    """

    def __init__(
        self,
        graph: DGLGraph,
        aggregation: str = "sum",
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
        self.aggregation = aggregation

        self.node_MLP = MLP(
            input_dim=input_dim_nodes + input_dim_edges,
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
    ) -> Tuple[Tensor, Tensor]:
        cat_feat = agg_concat_dgl(efeat, nfeat, self.graph, self.aggregation)
        # update node features + residual connection
        nfeat_new = self.node_MLP(cat_feat) + nfeat

        return efeat, nfeat_new

    def to(self, *args: Any, **kwargs: Any) -> "NodeBlockDGL":
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
        NodeBlockDGL
            The updated object after moving to the specified device, dtype, or format.
        """
        self = super().to(*args, **kwargs)
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.graph = self.graph.to(device=device)
        return self


class NodeBlockCUGO(nn.Module):
    """Node block for CuGraph

    Parameters
        ----------
        graph : CuGraphCSC
            Graph.
        aggregation : str, optional
            Aggregation method (sum, mean) , by default "sum"
        input_dim_nodes : int, optional
            Input dimensionality of the node features, by default 512
        input_dim_edges : int, optional
            Input dimensionality of the edge features, by default 512
        output_dim : int, optional
            Output dimensionality of the node features, by default 512
        hidden_dim : int, optional
            Number of neurons in each hidden layer, by default 512
        hidden_layers : int, optional
            Number of neurons in each hidden layer, by default 1
        activation_fn : nn.Module, optional
           Type of activation function, by default nn.SiLU()
        norm_type : str, optional
            Normalization type, by default "LayerNorm"
    """

    def __init__(
        self,
        graph: CuGraphCSC,
        aggregation: str = "sum",
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
        self.aggregation = aggregation

        self.node_MLP = MLP(
            input_dim=input_dim_nodes + input_dim_edges,
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
    ) -> Tuple[Tensor, Tensor]:
        # aggregate edge features and concat node features
        if self.static_graph is None:
            self.static_graph = self.graph.to_static_csc()

        cat_feat = agg_concat_e2n(nfeat, efeat, self.static_graph, self.aggregation)
        # update node features + residual connection
        nfeat_new = self.node_MLP(cat_feat) + nfeat
        return efeat, nfeat_new

    def to(self, *args: Any, **kwargs: Any) -> "NodeBlockCUGO":
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
        NodeBlockCUGO
            The updated object after moving to the specified device, dtype, or format.
        """
        self = super().to(*args, **kwargs)
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.graph = self.graph.to(device=device)
        return self
