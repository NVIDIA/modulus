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


from .utils import set_checkpoint_fn, CuGraphCSC
from .node_block import NodeBlockDGL, NodeBlockCUGO
from .edge_block import (
    EdgeBlockDGLConcat,
    EdgeBlockDGLSum,
    EdgeBlockCUGOConcat,
    EdgeBlockCUGOSum,
)


class Processor(nn.Module):
    """GraphCast icosahedron processor

    Parameters
    ----------
    graph : DGLGraph | CuGraphCSC
        graph structure representing the edges between mesh and grid
    aggregation : str, optional
        message passing aggregation method ("sum", "mean"), by default "sum"
    processor_layers : int, optional
        number of processor layers, by default 16
    input_dim_nodes : int, optional
        input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        input dimensionality of the edge features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        number of hiddel layers, by default 1
    activation_fn : nn.Module, optional
        type of activation function, by default nn.SiLU()
    norm_type : str, optional
        normalization type, by default "LayerNorm"
    do_conat_trick: : bool, default=False
        whether to replace concat+MLP with MLP+idx+sum
    """

    def __init__(
        self,
        graph: Union[DGLGraph, CuGraphCSC],
        aggregation: str = "sum",
        processor_layers: int = 16,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
    ):
        super().__init__()
        self.graph = graph

        if isinstance(graph, DGLGraph):
            EdgeBlock = EdgeBlockDGLSum if do_concat_trick else EdgeBlockDGLConcat
            NodeBlock = NodeBlockDGL

        else:
            EdgeBlock = EdgeBlockCUGOSum if do_concat_trick else EdgeBlockCUGOConcat
            NodeBlock = NodeBlockCUGO

        edge_block_invars = (
            self.graph,
            input_dim_nodes,
            input_dim_edges,
            input_dim_edges,
            hidden_dim,
            hidden_layers,
            activation_fn,
            norm_type,
        )
        node_block_invars = (
            self.graph,
            aggregation,
            input_dim_nodes,
            input_dim_edges,
            input_dim_nodes,
            hidden_dim,
            hidden_layers,
            activation_fn,
            norm_type,
        )

        layers = []
        for _ in range(processor_layers):
            layers.append(EdgeBlock(*edge_block_invars))
            layers.append(NodeBlock(*node_block_invars))

        self.processor_layers = nn.ModuleList(layers)
        self.num_processor_layers = len(self.processor_layers)
        # per default, no checkpointing
        # one segment for compatability
        self.checkpoint_segments = [(0, self.num_processor_layers)]
        self.checkpoint_fn = set_checkpoint_fn(False)

    def set_checkpoint_segments(self, checkpoint_segments: int):
        if checkpoint_segments > 0:
            if self.num_processor_layers % checkpoint_segments != 0:
                raise ValueError(
                    "Processor layers must be a multiple of checkpoint_segments"
                )
            segment_size = self.num_processor_layers // checkpoint_segments
            self.checkpoint_segments = []
            for i in range(0, self.num_processor_layers, segment_size):
                self.checkpoint_segments.append((i, i + segment_size))

            self.checkpoint_fn = set_checkpoint_fn(True)
        else:
            self.checkpoint_fn = set_checkpoint_fn(False)
            self.checkpoint_segments = [(0, self.num_processor_layers)]

    def run_function(self, segment_start: int, segment_end: int):
        segment = self.processor_layers[segment_start:segment_end]

        def custom_forward(efeat, nfeat):
            for module in segment:
                efeat, nfeat = module(efeat, nfeat)
            return efeat, nfeat

        return custom_forward

    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
    ) -> Tensor:

        for segment_start, segment_end in self.checkpoint_segments:
            efeat, nfeat = self.checkpoint_fn(
                self.run_function(segment_start, segment_end),
                efeat,
                nfeat,
                use_reentrant=False,
                preserve_rng_state=False,
            )

        return efeat, nfeat

    def to(self, *args, **kwargs) -> Processor:
        self = super().to(*args, **kwargs)
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.graph = self.graph.to(device=device)
        return self
