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
from torch import Tensor
import torch.nn as nn
import modulus

try:
    import dgl
    import dgl.function as fn
    from dgl import DGLGraph
except:
    raise ImportError(
        "Mesh Graph Net requires the DGL library. Install the "
        + "desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
    )
from typing import Union, List
from dataclasses import dataclass

from modulus.models.meta import ModelMetaData
from modulus.models.module import Module

from modulus.models.gnn_layers.utils import CuGraphCSC
from modulus.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from modulus.models.gnn_layers.mesh_edge_block import MeshEdgeBlock
from modulus.models.gnn_layers.mesh_node_block import MeshNodeBlock


@dataclass
class MetaData(ModelMetaData):
    name: str = "MeshGraphNet"
    # Optimization, no JIT as DGLGraph causes trouble
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = True
    auto_grad: bool = True


class MeshGraphNet(Module):
    """MeshGraphNet network architecture

    Parameters
    ----------
    input_dim_nodes : int
        Number of node features
    input_dim_edges : int
        Number of edge features
    output_dim : int
        Number of outputs
    processor_size : int, optional
        Number of message passing blocks, by default 15
    num_layers_node_processor : int, optional
        Number of MLP layers for processing nodes in each message passing block, by default 2
    num_layers_edge_processor : int, optional
        Number of MLP layers for processing edge features in each message passing block, by default 2
    hidden_dim_node_encoder : int, optional
        Hidden layer size for the node feature encoder, by default 128
    num_layers_node_encoder : int, optional
        Number of MLP layers for the node feature encoder, by default 2
    hidden_dim_edge_encoder : int, optional
        Hidden layer size for the edge feature encoder, by default 128
    num_layers_edge_encoder : int, optional
        Number of MLP layers for the edge feature encoder, by default 2
    hidden_dim_node_decoder : int, optional
        Hidden layer size for the node feature decoder, by default 128
    num_layers_node_decoder : int, optional
        Number of MLP layers for the node feature decoder, by default 2
    do_conat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum

    Example
    -------
    >>> model = modulus.models.meshgraphnet.MeshGraphNet(
    ...         input_dim_nodes=4,
    ...         input_dim_edges=3,
    ...         output_dim=2,
    ...     )
    >>> graph = dgl.rand_graph(10, 5)
    >>> node_features = torch.randn(10, 4)
    >>> edge_features = torch.randn(5, 3)
    >>> output = model(node_features, edge_features, graph)
    >>> output.size()
    torch.Size([10, 2])

    Note
    ----
    Reference: Pfaff, Tobias, et al. "Learning mesh-based simulation with graph networks."
    arXiv preprint arXiv:2010.03409 (2020).
    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        processor_size: int = 15,
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: int = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: int = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: int = 2,
        do_concat_trick: bool = False,
    ):
        super().__init__(meta=MetaData())

        self.edge_encoder = MeshGraphMLP(
            input_dim_edges,
            output_dim=hidden_dim_edge_encoder,
            hidden_dim=hidden_dim_edge_encoder,
            hidden_layers=num_layers_edge_encoder - 1,
            activation_fn=nn.ReLU(),
            norm_type="LayerNorm",
            recompute_activation=False,
        )
        self.node_encoder = MeshGraphMLP(
            input_dim_nodes,
            output_dim=hidden_dim_node_encoder,
            hidden_dim=hidden_dim_node_encoder,
            hidden_layers=num_layers_node_encoder - 1,
            activation_fn=nn.ReLU(),
            norm_type="LayerNorm",
            recompute_activation=False,
        )
        self.node_decoder = MeshGraphMLP(
            hidden_dim_node_encoder,
            output_dim=output_dim,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder - 1,
            activation_fn=nn.ReLU(),
            norm_type="LayerNorm",
            recompute_activation=False,
        )
        self.processor = MeshGraphNetProcessor(
            processor_size=processor_size,
            input_dim_node=hidden_dim_node_encoder,
            input_dim_edge=hidden_dim_edge_encoder,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation="sum",
            norm_type="LayerNorm",
            activation_fn=nn.ReLU(),
            do_concat_trick=do_concat_trick,
        )

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: Union[DGLGraph, List[DGLGraph]],
    ) -> Tensor:
        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        x = self.processor(node_features, edge_features, graph)
        x = self.node_decoder(x)
        return x


class MeshGraphNetProcessor(nn.Module):
    def __init__(
        self,
        processor_size: int = 15,
        input_dim_node: int = 128,
        input_dim_edge: int = 128,
        num_layers_node: int = 2,
        num_layers_edge: int = 2,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        activation_fn: nn.Module = nn.ReLU(),
        do_concat_trick: bool = False,
    ):
        super().__init__()
        self.processor_size = processor_size

        edge_block_invars = (
            input_dim_node,
            input_dim_edge,
            input_dim_edge,
            input_dim_edge,
            num_layers_edge - 1,
            activation_fn,
            norm_type,
            do_concat_trick,
            False,
        )
        node_block_invars = (
            aggregation,
            input_dim_node,
            input_dim_edge,
            input_dim_edge,
            input_dim_edge,
            num_layers_node - 1,
            activation_fn,
            norm_type,
            False,
        )

        edge_blocks = []
        node_blocks = []

        for _ in range(self.processor_size):
            edge_blocks.append(MeshEdgeBlock(*edge_block_invars))

        for _ in range(self.processor_size):
            node_blocks.append(MeshNodeBlock(*node_block_invars))

        self.edge_blocks = nn.ModuleList(edge_blocks)
        self.node_blocks = nn.ModuleList(node_blocks)

    @torch.jit.unused
    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: Union[DGLGraph, List[DGLGraph], CuGraphCSC],
    ) -> Tensor:
        for i in range(self.processor_size):
            if isinstance(graph, List):  # in case of neighbor sampling
                edge_features = edge_features[: graph[i].num_edges(), :]  # TODO check
                node_features_src = node_features
                node_features_dst = node_features_src[: graph[i].num_dst_nodes(), :]

                edge_features, _ = self.edge_blocks[i](
                    edge_features, (node_features_src, node_features_dst), graph[i]
                )
                _, node_features = self.node_blocks[i](
                    edge_features, node_features_dst, graph[i]
                )

            else:
                edge_features, _ = self.edge_blocks[i](
                    edge_features, node_features, graph
                )
                _, node_features = self.node_blocks[i](
                    edge_features, node_features, graph
                )

        return node_features
