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

from typing import Any
from dataclasses import dataclass

from modulus.models.gnn_layers.utils import set_checkpoint_fn, CuGraphCSC
from modulus.models.gnn_layers.embedder import (
    GraphCastEncoderEmbedder,
    GraphCastDecoderEmbedder,
)
from modulus.models.gnn_layers.mesh_graph_encoder import MeshGraphEncoder
from modulus.models.gnn_layers.mesh_graph_decoder import MeshGraphDecoder
from modulus.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from modulus.models.module import Module
from modulus.models.meta import ModelMetaData
from modulus.utils.graphcast.graph import Graph
from modulus.utils.graphcast.data_utils import StaticData

from .graph_cast_processor import GraphCastProcessor


@dataclass
class MetaData(ModelMetaData):
    name: str = "GraphCastNet"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class GraphCastNet(Module):
    """GraphCast network architecture

    Parameters
    ----------
    meshgraph_path : str
        Path to the meshgraph file. If not provided, the meshgraph will be created
        using PyMesh.
    static_dataset_path : str
        Path to the static dataset file.
    input_res: Tuple[int, int]
        Input resolution of the latitude-longitude grid
    input_dim_grid_nodes : int, optional
        Input dimensionality of the grid node features, by default 474
    input_dim_mesh_nodes : int, optional
        Input dimensionality of the mesh node features, by default 3
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 4
    output_dim_grid_nodes : int, optional
        Final output dimensionality of the edge features, by default 227
    processor_layers : int, optional
        Number of processor layers, by default 16
    hidden_layers : int, optional
        Number of hiddel layers, by default 1
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    aggregation : str, optional
        Message passing aggregation method ("sum", "mean"), by default "sum"
    activation_fn : nn.Module, optional
        Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type, by default "LayerNorm"
    use_cugraphops_encoder : bool, default=False
        Flag to select cugraphops kernels in encoder
    use_cugraphops_processor : bool, default=False
        Flag to select cugraphops kernels in the processor
    use_cugraphops_decoder : bool, default=False
        Flag to select cugraphops kernels in the decoder
    do_conat_trick: : bool, default=False
        Whether to replace concat+MLP with MLP+idx+sum
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.

    Note
    ----
    Based on these papers:
    - "GraphCast: Learning skillful medium-range global weather forecasting"
        https://arxiv.org/abs/2212.12794
    - "Forecasting Global Weather with Graph Neural Networks"
        https://arxiv.org/abs/2202.07575
    - "Learning Mesh-Based Simulation with Graph Networks"
        https://arxiv.org/abs/2010.03409
    - "MultiScale MeshGraphNets"
        https://arxiv.org/abs/2210.00612
    """

    def __init__(
        self,
        meshgraph_path: str,
        static_dataset_path: str,
        input_res: tuple = (721, 1440),
        input_dim_grid_nodes: int = 474,
        input_dim_mesh_nodes: int = 3,
        input_dim_edges: int = 4,
        output_dim_grid_nodes: int = 227,
        processor_layers: int = 16,
        hidden_layers: int = 1,
        hidden_dim: int = 512,
        aggregation: str = "sum",
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        use_cugraphops_encoder: bool = False,
        use_cugraphops_processor: bool = False,
        use_cugraphops_decoder: bool = False,
        do_concat_trick: bool = False,
        recompute_activation: bool = False,
    ):
        super().__init__(meta=MetaData())

        # create the lat_lon_grid
        self.latitudes = torch.linspace(-90, 90, steps=input_res[0])
        self.longitudes = torch.linspace(-180, 180, steps=input_res[1] + 1)[1:]
        self.lat_lon_grid = torch.stack(
            torch.meshgrid(self.latitudes, self.longitudes, indexing="ij"), dim=-1
        )
        self.has_static_data = static_dataset_path is not None

        # Get the static data
        if self.has_static_data:
            self.static_data = StaticData(
                static_dataset_path, self.latitudes, self.longitudes
            ).get()
            num_static_feat = self.static_data.size(1)
            input_dim_grid_nodes += num_static_feat
        else:
            self.static_data = None
        self.input_dim_grid_nodes = input_dim_grid_nodes
        self.output_dim_grid_nodes = output_dim_grid_nodes
        self.input_res = input_res

        # construct the graph
        try:
            self.graph = Graph(meshgraph_path, self.lat_lon_grid)
        except:
            raise FileNotFoundError(
                "The icospheres_path is corrupted. "
                "Tried using pymesh to generate the graph but could not find pymesh"
            )
        self.mesh_graph = self.graph.create_mesh_graph(verbose=False)
        self.g2m_graph = self.graph.create_g2m_graph(verbose=False)
        self.m2g_graph = self.graph.create_m2g_graph(verbose=False)

        self.g2m_edata = self.g2m_graph.edata["x"]
        self.m2g_edata = self.m2g_graph.edata["x"]
        self.mesh_edata = self.mesh_graph.edata["x"]
        self.mesh_ndata = self.mesh_graph.ndata["x"]

        if use_cugraphops_encoder:
            offsets, indices, edge_ids = self.g2m_graph.adj_tensors("csc")
            n_in_nodes, n_out_nodes = (
                self.g2m_graph.num_src_nodes(),
                self.g2m_graph.num_dst_nodes(),
            )
            self.g2m_graph = CuGraphCSC(
                offsets, indices, n_in_nodes, n_out_nodes, edge_ids
            )

        if use_cugraphops_decoder:
            offsets, indices, edge_ids = self.m2g_graph.adj_tensors("csc")
            n_in_nodes, n_out_nodes = (
                self.m2g_graph.num_src_nodes(),
                self.m2g_graph.num_dst_nodes(),
            )
            self.m2g_graph = CuGraphCSC(
                offsets, indices, n_in_nodes, n_out_nodes, edge_ids
            )

        if use_cugraphops_processor:
            offsets, indices, edge_ids = self.mesh_graph.adj_tensors("csc")
            n_in_nodes, n_out_nodes = (
                self.mesh_graph.num_src_nodes(),
                self.mesh_graph.num_dst_nodes(),
            )
            self.mesh_graph = CuGraphCSC(
                offsets, indices, n_in_nodes, n_out_nodes, edge_ids
            )

        # by default: don't checkpoint at all
        self.model_checkpoint_fn = set_checkpoint_fn(False)
        self.encoder_checkpoint_fn = set_checkpoint_fn(False)
        self.decoder_checkpoint_fn = set_checkpoint_fn(False)

        # initial feature embedder
        self.encoder_embedder = GraphCastEncoderEmbedder(
            input_dim_grid_nodes=input_dim_grid_nodes,
            input_dim_mesh_nodes=input_dim_mesh_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )
        self.decoder_embedder = GraphCastDecoderEmbedder(
            input_dim_edges=input_dim_edges,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

        # grid2mesh encoder
        self.encoder = MeshGraphEncoder(
            aggregation=aggregation,
            input_dim_src_nodes=hidden_dim,
            input_dim_dst_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            output_dim_src_nodes=hidden_dim,
            output_dim_dst_nodes=hidden_dim,
            output_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            recompute_activation=recompute_activation,
        )

        # icosahedron processor
        assert processor_layers > 2, "Expected at least 3 processor layers"
        self.processor_encoder = GraphCastProcessor(
            aggregation=aggregation,
            processor_layers=1,
            input_dim_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            recompute_activation=recompute_activation,
        )
        self.processor = GraphCastProcessor(
            aggregation=aggregation,
            processor_layers=processor_layers - 2,
            input_dim_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            recompute_activation=recompute_activation,
        )
        self.processor_decoder = GraphCastProcessor(
            aggregation=aggregation,
            processor_layers=1,
            input_dim_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            recompute_activation=recompute_activation,
        )

        # mesh2grid decoder
        self.decoder = MeshGraphDecoder(
            aggregation=aggregation,
            input_dim_src_nodes=hidden_dim,
            input_dim_dst_nodes=hidden_dim,
            input_dim_edges=hidden_dim,
            output_dim_dst_nodes=hidden_dim,
            output_dim_edges=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            do_concat_trick=do_concat_trick,
            recompute_activation=recompute_activation,
        )

        # final MLP
        self.finale = MeshGraphMLP(
            input_dim=hidden_dim,
            output_dim=output_dim_grid_nodes,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )

    def set_checkpoint_model(self, checkpoint_flag: bool):
        """Sets checkpoint function for the entire model.

        This function returns the appropriate checkpoint function based on the
        provided `checkpoint_flag` flag. If `checkpoint_flag` is True, the
        function returns the checkpoint function from PyTorch's
        `torch.utils.checkpoint`. In this case, all the other gradient checkpoitings
        will be disabled. Otherwise, it returns an identity function
        that simply passes the inputs through the given layer.

        Parameters
        ----------
        checkpoint_flag : bool
            Whether to use checkpointing for gradient computation. Checkpointing
            can reduce memory usage during backpropagation at the cost of
            increased computation time.

        Returns
        -------
        Callable
            The selected checkpoint function to use for gradient computation.
        """
        # force a single checkpoint for the whole model
        self.model_checkpoint_fn = set_checkpoint_fn(checkpoint_flag)
        if checkpoint_flag:
            self.processor.set_checkpoint_segments(-1)
            self.encoder_checkpoint_fn = set_checkpoint_fn(False)
            self.decoder_checkpoint_fn = set_checkpoint_fn(False)

    def set_checkpoint_processor(self, checkpoint_segments: int):
        """Sets checkpoint function for the processor excluding the first and last
        layers.

        This function returns the appropriate checkpoint function based on the
        provided `checkpoint_segments` flag. If `checkpoint_segments` is positive,
         the function returns the checkpoint function from PyTorch's
        `torch.utils.checkpoint`, with number of checkpointing segments equal to
        `checkpoint_segments`. Otherwise, it returns an identity function
        that simply passes the inputs through the given layer.

        Parameters
        ----------
        checkpoint_segments : int
            Number of checkpointing segments for gradient computation. Checkpointing
            can reduce memory usage during backpropagation at the cost of
            increased computation time.

        Returns
        -------
        Callable
            The selected checkpoint function to use for gradient computation.
        """
        self.processor.set_checkpoint_segments(checkpoint_segments)

    def set_checkpoint_encoder(self, checkpoint_flag: bool):
        """Sets checkpoint function for the embedder, encoder, and the first of
        the processor.

        This function returns the appropriate checkpoint function based on the
        provided `checkpoint_flag` flag. If `checkpoint_flag` is True, the
        function returns the checkpoint function from PyTorch's
        `torch.utils.checkpoint`. Otherwise, it returns an identity function
        that simply passes the inputs through the given layer.

        Parameters
        ----------
        checkpoint_flag : bool
            Whether to use checkpointing for gradient computation. Checkpointing
            can reduce memory usage during backpropagation at the cost of
            increased computation time.

        Returns
        -------
        Callable
            The selected checkpoint function to use for gradient computation.
        """
        self.encoder_checkpoint_fn = set_checkpoint_fn(checkpoint_flag)

    def set_checkpoint_decoder(self, checkpoint_flag: bool):
        """Sets checkpoint function for the last layer of the processor, the decoder,
        and the final MLP.

        This function returns the appropriate checkpoint function based on the
        provided `checkpoint_flag` flag. If `checkpoint_flag` is True, the
        function returns the checkpoint function from PyTorch's
        `torch.utils.checkpoint`. Otherwise, it returns an identity function
        that simply passes the inputs through the given layer.

        Parameters
        ----------
        checkpoint_flag : bool
            Whether to use checkpointing for gradient computation. Checkpointing
            can reduce memory usage during backpropagation at the cost of
            increased computation time.

        Returns
        -------
        Callable
            The selected checkpoint function to use for gradient computation.
        """
        self.decoder_checkpoint_fn = set_checkpoint_fn(checkpoint_flag)

    def encoder_forward(
        self,
        grid_nfeat: Tensor,
    ) -> Tensor:
        """Forward method for the embedder, encoder, and the first of the processor.

        Parameters
        ----------
        grid_nfeat : Tensor
            Node features for the latitude-longitude grid.

        Returns
        -------
        mesh_efeat_processed: Tensor
            Processed edge features for the multimesh.
        mesh_nfeat_processed: Tensor
            Processed node features for the multimesh.
        grid_nfeat_encoded: Tensor
            Encoded node features for the latitude-longitude grid.
        """

        # embedd graph features
        (
            grid_nfeat_embedded,
            mesh_nfeat_embedded,
            g2m_efeat_embedded,
            mesh_efeat_embedded,
        ) = self.encoder_embedder(
            grid_nfeat,
            self.mesh_ndata,
            self.g2m_edata,
            self.mesh_edata,
        )

        # encode lat/lon to multimesh
        grid_nfeat_encoded, mesh_nfeat_encoded = self.encoder(
            g2m_efeat_embedded,
            grid_nfeat_embedded,
            mesh_nfeat_embedded,
            self.g2m_graph,
        )

        # process multimesh graph
        mesh_efeat_processed, mesh_nfeat_processed = self.processor_encoder(
            mesh_efeat_embedded,
            mesh_nfeat_encoded,
            self.mesh_graph,
        )

        return mesh_efeat_processed, mesh_nfeat_processed, grid_nfeat_encoded

    def decoder_forward(
        self,
        mesh_efeat_processed: Tensor,
        mesh_nfeat_processed: Tensor,
        grid_nfeat_encoded: Tensor,
    ) -> Tensor:
        """Forward method for the last layer of the processor, the decoder,
        and the final MLP.

        Parameters
        ----------
        mesh_efeat_processed : Tensor
            Multimesh edge features processed by the processor.
        mesh_nfeat_processed : Tensor
            Multi-mesh node features processed by the processor.
        grid_nfeat_encoded : Tensor
            The encoded node features for the latitude-longitude grid.

        Returns
        -------
        grid_nfeat_finale: Tensor
            The final node features for the latitude-longitude grid.
        """

        # process multimesh graph
        _, mesh_nfeat_processed = self.processor_decoder(
            mesh_efeat_processed,
            mesh_nfeat_processed,
            self.mesh_graph,
        )

        m2g_efeat_embedded = self.decoder_embedder(self.m2g_edata)

        # decode multimesh to lat/lon
        grid_nfeat_decoded = self.decoder(
            m2g_efeat_embedded, grid_nfeat_encoded, mesh_nfeat_processed, self.m2g_graph
        )

        # map to the target output dimension
        grid_nfeat_finale = self.finale(
            grid_nfeat_decoded,
        )

        return grid_nfeat_finale

    def custom_forward(self, grid_nfeat: Tensor) -> Tensor:
        """GraphCast forward method with support for gradient checkpointing.

        Parameters
        ----------
        grid_nfeat : Tensor
            Node features of the latitude-longitude graph.

        Returns
        -------
        grid_nfeat_finale: Tensor
            Predicted node features of the latitude-longitude graph.
        """
        (
            mesh_efeat_processed,
            mesh_nfeat_processed,
            grid_nfeat_encoded,
        ) = self.encoder_checkpoint_fn(
            self.encoder_forward,
            grid_nfeat,
            use_reentrant=False,
            preserve_rng_state=False,
        )

        # checkpoint of processor done in processor itself
        mesh_efeat_processed, mesh_nfeat_processed = self.processor(
            mesh_efeat_processed,
            mesh_nfeat_processed,
            self.mesh_graph,
        )

        grid_nfeat_finale = self.decoder_checkpoint_fn(
            self.decoder_forward,
            mesh_efeat_processed,
            mesh_nfeat_processed,
            grid_nfeat_encoded,
            use_reentrant=False,
            preserve_rng_state=False,
        )

        return grid_nfeat_finale

    def forward(
        self,
        grid_nfeat: Tensor,
    ) -> Tensor:
        invar = self.prepare_input(grid_nfeat)
        outvar = self.model_checkpoint_fn(
            self.custom_forward,
            invar,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        return self.prepare_output(outvar)

    def prepare_input(self, invar: Tensor) -> Tensor:
        """Prepares the input to the model in the required shape.

        Parameters
        ----------
        invar : Tensor
            Input in the shape [N, C, H, W].

        Returns
        -------
        Tensor
            Reshaped input.
        """
        assert invar.size(0) == 1, "GraphCast does not support batch size > 1"
        # concat static data
        if self.has_static_data:
            invar = torch.concat((invar, self.static_data), dim=1)
        invar = invar[0].view(self.input_dim_grid_nodes, -1).permute(1, 0)
        return invar

    def prepare_output(self, outvar: Tensor) -> Tensor:
        """Prepares the output of the model in the shape [N, C, H, W].

        Parameters
        ----------
        outvar : Tensor
            Output of the final MLP of the model.

        Returns
        -------
        Tensor
            The reshaped output of the model.
        """
        outvar = outvar.permute(1, 0)
        outvar = outvar.view(self.output_dim_grid_nodes, *self.input_res)
        outvar = torch.unsqueeze(outvar, dim=0)
        return outvar

    def to(self, *args: Any, **kwargs: Any) -> "GraphCastNet":
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
        GraphCastNet
            The updated object after moving to the specified device, dtype, or format.
        """
        self = super(GraphCastNet, self).to(*args, **kwargs)

        self.g2m_edata = self.g2m_edata.to(*args, **kwargs)
        self.m2g_edata = self.m2g_edata.to(*args, **kwargs)
        self.mesh_ndata = self.mesh_ndata.to(*args, **kwargs)
        self.mesh_edata = self.mesh_edata.to(*args, **kwargs)
        if self.has_static_data:
            self.static_data = self.static_data.to(*args, **kwargs)

        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        self.g2m_graph = self.g2m_graph.to(device)
        self.mesh_graph = self.mesh_graph.to(device)
        self.m2g_graph = self.m2g_graph.to(device)

        return self
