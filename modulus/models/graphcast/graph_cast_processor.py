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

from typing import Union

import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor

from modulus.models.gnn_layers.mesh_edge_block import MeshEdgeBlock
from modulus.models.gnn_layers.mesh_node_block import MeshNodeBlock
from modulus.models.gnn_layers.utils import CuGraphCSC, set_checkpoint_fn


class GraphCastProcessor(nn.Module):
    """Processor block used in GraphCast operating on a latent space
    represented by hierarchy of icosahedral meshes.

    Parameters
    ----------
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
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(
        self,
        aggregation: str = "sum",
        processor_layers: int = 16,
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
    ):
        super().__init__()

        edge_block_invars = (
            input_dim_nodes,
            input_dim_edges,
            input_dim_edges,
            hidden_dim,
            hidden_layers,
            activation_fn,
            norm_type,
            do_concat_trick,
            recompute_activation,
        )
        node_block_invars = (
            aggregation,
            input_dim_nodes,
            input_dim_edges,
            input_dim_nodes,
            hidden_dim,
            hidden_layers,
            activation_fn,
            norm_type,
            recompute_activation,
        )

        layers = []
        for _ in range(processor_layers):
            layers.append(MeshEdgeBlock(*edge_block_invars))
            layers.append(MeshNodeBlock(*node_block_invars))

        self.processor_layers = nn.ModuleList(layers)
        self.num_processor_layers = len(self.processor_layers)
        # per default, no checkpointing
        # one segment for compatability
        self.checkpoint_segments = [(0, self.num_processor_layers)]
        self.checkpoint_fn = set_checkpoint_fn(False)

    def set_checkpoint_segments(self, checkpoint_segments: int):
        """
        Set the number of checkpoint segments

        Parameters
        ----------
        checkpoint_segments : int
            number of checkpoint segments

        Raises
        ------
        ValueError
            if the number of processor layers is not a multiple of the number of
            checkpoint segments
        """
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
        """Custom forward for gradient checkpointing

        Parameters
        ----------
        segment_start : int
            Layer index as start of the segment
        segment_end : int
            Layer index as end of the segment

        Returns
        -------
        function
            Custom forward function
        """
        segment = self.processor_layers[segment_start:segment_end]

        def custom_forward(efeat, nfeat, graph):
            """Custom forward function"""
            for module in segment:
                efeat, nfeat = module(efeat, nfeat, graph)
            return efeat, nfeat

        return custom_forward

    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tensor:
        for segment_start, segment_end in self.checkpoint_segments:
            efeat, nfeat = self.checkpoint_fn(
                self.run_function(segment_start, segment_end),
                efeat,
                nfeat,
                graph,
                use_reentrant=False,
                preserve_rng_state=False,
            )

        return efeat, nfeat
