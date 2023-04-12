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
from dgl import DGLGraph
import dgl.function as fn
from typing import Any, Callable, Dict, Optional
from torch.utils.checkpoint import checkpoint


try:
    from pylibcugraphops.pytorch import StaticCSC, BipartiteCSC
except:
    StaticCSC = None
    BipartiteCSC = None


class CuGraphCSC:
    """Constructs a CuGraphCSC object.

    Parameters
    ----------
    offsets : Tensor
        The offsets tensor.
    indices : Tensor
        The indices tensor.
    num_src_nodes : int
        The number of source nodes.
    num_dst_nodes : int
        The number of destination nodes.
    ef_indices : Optional[Tensor], optional
        The edge feature indices tensor, by default None
    reverse_graph_bwd : bool, optional
        Whether to reverse the graph for the backward pass, by default True
    """

    def __init__(
        self,
        offsets: Tensor,
        indices: Tensor,
        num_src_nodes: int,
        num_dst_nodes: int,
        ef_indices: Optional[Tensor] = None,
        reverse_graph_bwd: bool = True,
    ) -> None:

        self.offsets = offsets
        self.indices = indices
        self.num_src_nodes = num_src_nodes
        self.num_dst_nodes = num_dst_nodes
        self.ef_indices = ef_indices
        self.reverse_graph_bwd = reverse_graph_bwd

        self.bipartite_csc = None
        self.static_csc = None

    def to(self, *args: Any, **kwargs: Any) -> "CuGraphCSC":
        """Moves the object to the specified device, dtype, or format and returns the
        updated object.

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
        device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        assert dtype in (
            None,
            torch.int32,
            torch.int64,
        ), f"Invalid dtype, expected torch.int32 or torch.int64, got {dtype}."
        self.offsets = self.offsets.to(device=device, dtype=dtype)
        self.indices = self.indices.to(device=device, dtype=dtype)
        if self.ef_indices is not None:
            self.ef_indices = self.ef_indices.to(device=device, dtype=dtype)

        return self

    def to_bipartite_csc(self, dtype=None, cache_graph: bool = True):
        """Converts the graph to a bipartite CSC graph.

        Parameters
        ----------
        dtype : torch.dtype, optional
            The dtype of the graph, by default None
        cache_graph : bool, optional
            Whether to cache the graph, by default True

        Returns
        -------
        BipartiteCSC
            The bipartite CSC graph.
        """

        assert self.offsets.is_cuda, "Expected the graph structures to reside on GPU."
        if self.bipartite_csc is None or not cache_graph:
            # Occassionally, we have to watch out for the IdxT type
            # of offsets and indices. Technically, they are only relevant
            # for storing node and edge indices. However, they are also used
            # to index pointers in the underlying kernels (for now). This means
            # that depending on the data dimension, one has to rely on int64
            # for the indices despite int32 technically being enough to store the
            # graph. This will be improved in cugraph-ops-23.06. Until then, allow
            # the change of dtype.
            graph_offsets = self.offsets
            graph_indices = self.indices
            graph_ef_indices = self.ef_indices

            if dtype is not None:
                graph_offsets = self.offsets.to(dtype=dtype)
                graph_indices = self.indices.to(dtype=dtype)
                if self.ef_indices is not None:
                    graph_ef_indices = self.ef_indices.to(dtype=dtype)

            graph = BipartiteCSC(
                graph_offsets,
                graph_indices,
                self.num_src_nodes,
                graph_ef_indices,
                reverse_graph_bwd=self.reverse_graph_bwd,
            )
            self.bipartite_csc = graph

        return self.bipartite_csc

    def to_static_csc(self, dtype=None, cache_graph: bool = True):
        """Converts the graph to a static CSC graph.

        Parameters
        ----------
        dtype : torch.dtype, optional
            The dtype of the graph, by default None
        cache_graph : bool, optional
            Whether to cache the graph, by default True

        Returns
        -------
        StaticCSC
            The static CSC graph.
        """

        if self.static_csc is None or not cache_graph:
            # Occassionally, we have to watch out for the IdxT type
            # of offsets and indices. Technically, they are only relevant
            # for storing node and edge indices. However, they are also used
            # to index pointers in the underlying kernels (for now). This means
            # that depending on the data dimension, one has to rely on int64
            # for the indices despite int32 technically being enough to store the
            # graph. This will be improved in cugraph-ops-23.06. Until then, allow
            # the change of dtype.
            graph_offsets = self.offsets
            graph_indices = self.indices
            graph_ef_indices = self.ef_indices

            if dtype is not None:
                graph_offsets = self.offsets.to(dtype=dtype)
                graph_indices = self.indices.to(dtype=dtype)
                if self.ef_indices is not None:
                    graph_ef_indices = self.ef_indices.to(dtype=dtype)

            graph = StaticCSC(
                graph_offsets,
                graph_indices,
                graph_ef_indices,
            )
            self.static_csc = graph

        return self.static_csc


def checkpoint_identity(layer: Callable, *args: Any, **kwargs: Any) -> Any:
    """Applies the identity function for checkpointing.

    This function serves as an identity function for use with model layers
    when checkpointing is not enabled. It simply forwards the input arguments
    to the specified layer and returns its output.

    Parameters
    ----------
    layer : Callable
        The model layer or function to apply to the input arguments.
    *args
        Positional arguments to be passed to the layer.
    **kwargs
        Keyword arguments to be passed to the layer.

    Returns
    -------
    Any
        The output of the specified layer after processing the input arguments.
    """
    return layer(*args)


def set_checkpoint_fn(do_checkpointing: bool) -> Callable:
    """Sets checkpoint function.

    This function returns the appropriate checkpoint function based on the
    provided `do_checkpointing` flag. If `do_checkpointing` is True, the
    function returns the checkpoint function from PyTorch's
    `torch.utils.checkpoint`. Otherwise, it returns an identity function
    that simply passes the inputs through the given layer.

    Parameters
    ----------
    do_checkpointing : bool
        Whether to use checkpointing for gradient computation. Checkpointing
        can reduce memory usage during backpropagation at the cost of
        increased computation time.

    Returns
    -------
    Callable
        The selected checkpoint function to use for gradient computation.
    """
    if do_checkpointing:
        return checkpoint
    else:
        return checkpoint_identity


def concat_message_function(edges: Tensor) -> Dict[str, Tensor]:
    """Concatenates source node, destination node, and edge features.

    Parameters
    ----------
    edges : Tensor
        Edges.

    Returns
    -------
    Dict[Tensor]
        Concatenated source node, destination node, and edge features.
    """
    # concats src node , dst node, and edge features
    cat_feat = torch.cat((edges.data["x"], edges.src["x"], edges.dst["x"]), dim=1)
    return {"cat_feat": cat_feat}


def concat_efeat_dgl_mesh(efeat: Tensor, nfeat: Tensor, graph: DGLGraph) -> Tensor:
    """Concatenates edge features with source and destination node features.
    Use for homogeneous graphs.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor
        Node features.
    graph : DGLGraph
        Graph.

    Returns
    -------
    Tensor
        Concatenated edge features with source and destination node features.
    """
    with graph.local_scope():
        graph.ndata["x"] = nfeat
        graph.edata["x"] = efeat

        graph.apply_edges(concat_message_function)
        return graph.edata["cat_feat"]


def concat_efeat_dgl_m2g_g2m(
    efeat: Tensor, src_feat: Tensor, dst_feat: Tensor, graph: DGLGraph
) -> Tensor:
    """Concatenates edge features with source and destination node features.
    Use for heterogeneous graphs.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    src_feat : Tensor
        Source node features.
    dst_feat : Tensor
        Destination node features.
    graph : DGLGraph
        Graph.

    Returns
    -------
    Tensor
        Concatenated edge features with source and destination node features.
    """
    with graph.local_scope():
        graph.srcdata["x"] = src_feat
        graph.dstdata["x"] = dst_feat
        graph.edata["x"] = efeat

        graph.apply_edges(concat_message_function)
        return graph.edata["cat_feat"]


@torch.jit.script
def sum_efeat_dgl(
    efeat: Tensor, src_feat: Tensor, dst_feat: Tensor, src_idx: Tensor, dst_idx: Tensor
) -> Tensor:
    """Sums edge features with source and destination node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    src_feat : Tensor
        Source node features.
    dst_feat : Tensor
        Destination node features.
    src_idx : Tensor
        Source node indices.
    dst_idx : Tensor
        Destination node indices.

    Returns
    -------
    Tensor
        Sum of edge features with source and destination node features.
    """
    return efeat + src_feat[src_idx] + dst_feat[dst_idx]


def agg_concat_dgl(
    efeat: Tensor, nfeat: Tensor, graph: DGLGraph, aggregation: str
) -> Tensor:
    """Aggregates edge features and concatenates with node features.

    Parameters
    ----------
    efeat : Tensor
        Edge features.
    nfeat : Tensor
        Node features.
    graph : DGLGraph
        Graph.
    aggregation : str
        Aggregation method (sum or mean).

    Returns
    -------
    Tensor
        Aggregated edge features concatenated with node features.

    Raises
    ------
    RuntimeError
        If aggregation method is not sum or mean.
    """
    with graph.local_scope():
        # populate features on graph edges
        graph.edata["x"] = efeat

        # aggregate edge features
        if aggregation == "sum":
            graph.update_all(fn.copy_e("x", "m"), fn.sum("m", "h_dest"))
        elif aggregation == "mean":
            graph.update_all(fn.copy_e("x", "m"), fn.mean("m", "h_dest"))
        else:
            raise RuntimeError("Not a valid aggregation!")

        # concat node & edge features
        cat_feat = torch.cat((graph.dstdata["h_dest"], nfeat), -1)
        return cat_feat
