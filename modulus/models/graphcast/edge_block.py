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

from typing import Any, Union
from torch import Tensor
from dgl import DGLGraph
from .mlp import MLP, TruncatedMLP, TruncatedMLPCuGraph
from .utils import concat_efeat_dgl_mesh, CuGraphCSC

try:
    from pylibcugraphops.pytorch.operators import update_efeat_static_e2e
except ModuleNotFoundError:
    update_efeat_static_e2e = None


class EdgeBlockConcat(nn.Module):
    """Edge block with an explicit concatenation of features.

    Parameters
    ----------
    graph : DGLGraph | CuGraphCSC
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
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(
        self,
        graph: Union[DGLGraph, CuGraphCSC],
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()
        self.graph = graph
        self.use_cugraphops = isinstance(graph, CuGraphCSC)

        dim_concat = 2 * input_dim_nodes + input_dim_edges
        self.edge_MLP = MLP(
            input_dim=dim_concat,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
    ) -> Tensor:
        if self.use_cugraphops:
            static_graph = self.graph.to_static_csc()
            cat_feat = update_efeat_static_e2e(
                efeat,
                nfeat,
                static_graph,
                mode="concat",
                use_source_emb=True,
                use_target_emb=True,
            )

        else:
            cat_feat = concat_efeat_dgl_mesh(efeat, nfeat, self.graph)

        efeat_new = self.edge_MLP(cat_feat) + efeat
        return efeat_new, nfeat

    def to(self, *args: Any, **kwargs: Any) -> "EdgeBlockConcat":
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


class EdgeBlockSum(nn.Module):
    """Edge block with truncated MLPs.

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
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(
        self,
        graph: Union[DGLGraph, CuGraphCSC],
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()
        self.graph = graph
        self.use_cugraphops = isinstance(graph, CuGraphCSC)

        if self.use_cugraphops:
            self.edge_trunc_mlp = TruncatedMLPCuGraph(
                efeat_dim=input_dim_edges,
                src_dim=input_dim_nodes,
                dst_dim=input_dim_nodes,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                activation_fn=activation_fn,
                norm_type=norm_type,
                recompute_activation=recompute_activation,
            )

        else:
            self.src, self.dst = (item.long() for item in self.graph.edges())

            self.edge_trunc_mlp = TruncatedMLP(
                efeat_dim=input_dim_edges,
                src_dim=input_dim_nodes,
                dst_dim=input_dim_nodes,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                activation_fn=activation_fn,
                norm_type=norm_type,
                recompute_activation=recompute_activation,
            )

    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
    ) -> Tensor:
        if self.use_cugraphops:
            bipartite_graph = self.graph.to_bipartite_csc()
            efeat_new = (
                self.edge_trunc_mlp(efeat, nfeat, nfeat, bipartite_graph) + efeat
            )

        else:
            efeat_new = (
                self.edge_trunc_mlp(efeat, nfeat, nfeat, self.src, self.dst) + efeat
            )

        return efeat_new, nfeat

    def to(self, *args: Any, **kwargs: Any) -> "EdgeBlockSum":
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
