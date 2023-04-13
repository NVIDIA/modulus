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
from torch.nn import Sequential, ModuleList, Linear, ReLU, LayerNorm
from typing import Union, List
from dataclasses import dataclass

from ..meta import ModelMetaData
from ..module import Module


@dataclass
class MetaData(ModelMetaData):
    name: str = "MeshGraphNet"
    # Optimization
    jit: bool = True
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
    >>> output = model(graph, node_features, edge_features)
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
    ):
        super().__init__(meta=MetaData())

        self.edge_encoder = _Encoder(
            input_dim_edges,
            hidden_dim_edge_encoder,
            num_layers_edge_encoder,
        )
        self.node_encoder = _Encoder(
            input_dim_nodes,
            hidden_dim_node_encoder,
            num_layers_node_encoder,
        )
        self.node_decoder = _Decoder(
            output_dim, hidden_dim_node_decoder, num_layers_node_decoder
        )
        self.processor = _GraphProcessor(
            processor_size=processor_size,
            input_dim_node=hidden_dim_node_encoder,
            num_layers_node=num_layers_node_processor,
            input_dim_edge=hidden_dim_edge_encoder,
            num_layers_edge=num_layers_edge_processor,
        )

    @torch.jit.unused
    def forward(
        self,
        graph: Union[DGLGraph, List[DGLGraph]],
        node_features: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        x = self.processor(graph, node_features, edge_features)
        x = self.node_decoder(x)
        return x


class _Encoder(nn.Module):
    """MeshGraphNet encoder

    Parameters
    ----------
    input_dim : int
        Number of input features
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 128
    num_layers : int, optional
        Number of hidden layers, by default 2
    layer_norm : bool, optional
        Use layer norm in the last layer, by default True
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.mlp = [Linear(input_dim, hidden_dim), ReLU()]
        for _ in range(num_layers - 1):
            self.mlp += [Linear(hidden_dim, hidden_dim), ReLU()]
        self.mlp.append(Linear(hidden_dim, hidden_dim))
        if layer_norm:
            self.mlp.append(LayerNorm(hidden_dim))
        self.mlp = Sequential(*self.mlp)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class _Decoder(nn.Module):
    """MeshGraphNet encoder

    Parameters
    ----------
    output_dim : int
        Number of output features
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 128
    num_layers : int, optional
        Number of hidden layers, by default 2
    layer_norm : bool, optional
        Use layer norm in the last layer, by default False
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.mlp = [Linear(hidden_dim, hidden_dim), ReLU()]
        for _ in range(num_layers - 1):
            self.mlp += [Linear(hidden_dim, hidden_dim), ReLU()]
        self.mlp.append(Linear(hidden_dim, output_dim))
        if layer_norm:
            self.mlp.append(LayerNorm(output_dim))
        self.mlp = Sequential(*self.mlp)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class _EdgeBlock(nn.Module):
    def __init__(
        self,
        input_dim_node: int = 128,
        input_dim_edge: int = 128,
        num_layers: int = 2,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.edge_mlp = [
            Linear(2 * input_dim_node + input_dim_edge, input_dim_edge),
            ReLU(),
        ]
        for _ in range(num_layers - 1):
            self.edge_mlp += [Linear(input_dim_edge, input_dim_edge), ReLU()]
        self.edge_mlp.append(Linear(input_dim_edge, input_dim_edge))
        if layer_norm:
            self.edge_mlp.append(LayerNorm(input_dim_edge))
        self.edge_mlp = Sequential(*self.edge_mlp)

    @torch.jit.unused
    def forward(
        self, graph: DGLGraph, node_features: Tensor, edge_features: Tensor
    ) -> Tensor:
        with graph.local_scope():
            if isinstance(node_features, tuple):
                node_features_src, node_features_dst = node_features
            else:
                node_features_src = node_features_dst = node_features
            graph.srcdata["x"] = node_features_src
            graph.dstdata["x"] = node_features_dst
            graph.edata["x"] = edge_features
            graph.apply_edges(self.concat_message_function)
            return (
                self.edge_mlp(graph.edata["cat_feat"]) + edge_features
            )  # residual connection

    @staticmethod
    def concat_message_function(
        edges,
    ):  # TODO feature concat on edges is not efficient, consider optimizing this
        return {
            "cat_feat": torch.cat(
                (edges.src["x"], edges.dst["x"], edges.data["x"]), dim=1
            )
        }


class _NodeBlock(nn.Module):
    def __init__(
        self,
        input_dim_node: int = 128,
        input_dim_edge: int = 128,
        num_layers: int = 2,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.node_mlp = [
            Linear(input_dim_node + input_dim_edge, input_dim_node),
            ReLU(),
        ]
        for _ in range(num_layers - 1):
            self.node_mlp += [Linear(input_dim_node, input_dim_node), ReLU()]
        self.node_mlp.append(Linear(input_dim_node, input_dim_node))
        if layer_norm:
            self.node_mlp.append(LayerNorm(input_dim_node))
        self.node_mlp = Sequential(*self.node_mlp)

    @torch.jit.unused
    def forward(
        self, graph: DGLGraph, node_features: Tensor, edge_features: Tensor
    ) -> Tensor:
        with graph.local_scope():
            if isinstance(node_features, tuple):
                node_features_src, node_features_dst = node_features
            else:
                node_features_src = node_features_dst = node_features
            graph.srcdata["x"] = node_features_src
            graph.dstdata["x"] = node_features_dst
            graph.edata["x"] = edge_features
            graph.update_all(
                fn.copy_e("x", "m"), fn.sum("m", "h_dest")
            )  # aggregate edge message by target
            graph.dstdata["h_dest"] = torch.cat(
                (graph.dstdata["h_dest"], node_features_dst), -1
            )
            return (
                self.node_mlp(graph.dstdata["h_dest"]) + node_features_dst
            )  # residual connection


class _GraphProcessor(nn.Module):
    def __init__(
        self,
        processor_size: int = 15,
        input_dim_node: int = 128,
        num_layers_node: int = 2,
        input_dim_edge: int = 128,
        num_layers_edge: int = 2,
    ):
        super().__init__()
        self.processor_size = processor_size
        self.edge_blocks = ModuleList(
            [
                _EdgeBlock(
                    input_dim_node,
                    input_dim_edge,
                    num_layers_edge,
                )
                for _ in range(self.processor_size)
            ]
        )
        self.node_blocks = ModuleList(
            [
                _NodeBlock(input_dim_node, input_dim_edge, num_layers_node)
                for _ in range(self.processor_size)
            ]
        )

    @torch.jit.unused
    def forward(
        self,
        graph: Union[DGLGraph, List[DGLGraph]],
        node_features: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        for i in range(self.processor_size):
            if isinstance(graph, List):  # in case of neighbor sampling
                edge_features = edge_features[: graph[i].num_edges(), :]  # TODO check
                node_features_src = node_features
                node_features_dst = node_features_src[: graph[i].num_dst_nodes()]
                edge_features = self.edge_blocks[i](
                    graph[i], (node_features_src, node_features_dst), edge_features
                )
                node_features = self.node_blocks[i](
                    graph[i], (node_features_src, node_features_dst), edge_features
                )
            else:
                edge_features = self.edge_blocks[i](graph, node_features, edge_features)
                node_features = self.node_blocks[i](graph, node_features, edge_features)
        return node_features
