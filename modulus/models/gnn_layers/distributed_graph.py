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


from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed as dist

from modulus.distributed import (
    DistributedManager,
    all_gather_v,
    gather_v,
    indexed_all_to_all_v,
    scatter_v,
)


@dataclass
class GraphPartition:  # pragma: no cover
    """Class acting as a struct to hold all relevant buffers and variables
    to define a graph partition.
    """

    partition_size: int
    partition_rank: int
    device: torch.device
    # data structures for local graph
    # of this current partition rank
    local_offsets: torch.Tensor = field(init=False)
    local_indices: torch.Tensor = field(init=False)
    num_local_src_nodes: int = 0
    num_local_dst_nodes: int = 0
    num_local_indices: int = 0
    # mapping from local to global ID space
    # for this current partition rank
    partitioned_src_node_ids_to_global: torch.Tensor = field(init=False)
    partitioned_dst_node_ids_to_global: torch.Tensor = field(init=False)
    partitioned_indices_to_global: torch.Tensor = field(init=False)
    # buffers, sizes, and ID counts to support
    # distributed communication primitives
    # number of IDs each rank potentially sends to all other ranks
    sizes: List[List[int]] = field(init=False)
    # local indices of IDs current rank sends to all other ranks
    scatter_indices: List[torch.Tensor] = field(init=False)
    num_src_nodes_in_each_partition: List[int] = field(init=False)
    num_dst_nodes_in_each_partition: List[int] = field(init=False)
    num_indices_in_each_partition: List[int] = field(init=False)

    def __post_init__(self):
        # after partition_size has been set in __init__
        self.sizes = [
            [None for _ in range(self.partition_size)]
            for _ in range(self.partition_size)
        ]
        self.scatter_indices = [None] * self.partition_size
        self.num_src_nodes_in_each_partition = [None] * self.partition_size
        self.num_dst_nodes_in_each_partition = [None] * self.partition_size
        self.num_indices_in_each_partition = [None] * self.partition_size


def partition_graph_nodewise(
    global_offsets: torch.Tensor,
    global_indices: torch.Tensor,
    partition_size: int,
    partition_rank: int,
    device: torch.device,
):  # pragma: no cover
    """Utility function which partitions a global graph given as CSC structure naively
    by splitting both the IDs of source and destination nodes into chunks of equal
    size. Each partition rank then manages its according chunk of both source and
    destination nodes. Indices are assigned to the rank such that each rank manages
    all the incoming edges for all the destination nodes on the corresponding
    partition rank.
    The function performs the partitioning based on a global graph in CPU
    memory for each rank independently. It could be rewritten to e.g. only
    do it one rank and exchange the partitions or to an algorithm that also
    assumes an already distributed global graph, however, we expect global
    graphs to fit in CPU memory. After the partitioning, we can get rid off
    the larger one in CPU memory, only keep the local graphs on each GPU, and
    avoid tedious gather/scatter routines for exchanging partitions in the process.

    Parameters
    ----------
    global_offsets : torch.Tensor
        CSC offsets, can live on the CPU
    global_indices : torch.Tensor
        CSC indices, can live on the CPU
    partition_size : int
        number of process groups across which graph is partitioned,
        i.e. the number of graph partitions
    partition_rank : int
        rank within process group managing the distributed graph, i.e.
        the rank determining which partition the corresponding local rank
        will manage
    device : torch.device
        device connected to the passed partition rank, i.e. the device
        on which the local graph and related buffers will live on
    """

    # initialize graph partition
    graph_partition = GraphPartition(
        partition_size=partition_size, partition_rank=partition_rank, device=device
    )

    # --------------------------------------------------------------
    # initialize temporary variables used in computing the partition
    # global information about node ids and edge ids
    num_global_src_nodes = global_indices.max().item() + 1
    num_global_dst_nodes = global_offsets.size(0) - 1

    # global IDs of destination nodes in this partition
    dst_nodes_in_partition = None
    # global IDs of source nodes in this partition
    src_nodes_in_partition = None

    # get distribution of destination IDs: simply divide them into equal chunks
    dst_nodes_in_partition = (
        num_global_dst_nodes + partition_size - 1
    ) // partition_size
    dst_offsets_in_partition = [
        rank * dst_nodes_in_partition for rank in range(partition_size + 1)
    ]
    dst_offsets_in_partition[-1] = min(
        num_global_dst_nodes, dst_offsets_in_partition[-1]
    )

    # get distribution of source IDs: again simply divide them into equal chunks
    src_nodes_in_partition = (
        num_global_src_nodes + partition_size - 1
    ) // partition_size
    src_offsets_in_partition = [
        rank * src_nodes_in_partition for rank in range(partition_size + 1)
    ]
    src_offsets_in_partition[-1] = min(
        num_global_src_nodes, src_offsets_in_partition[-1]
    )

    for rank in range(partition_size):
        offset_start = dst_offsets_in_partition[rank]
        offset_end = dst_offsets_in_partition[rank + 1]
        offsets = global_offsets[offset_start : offset_end + 1].detach().clone()
        partition_indices = global_indices[offsets[0] : offsets[-1]].detach().clone()
        offsets -= offsets[0].item()

        global_src_ids_per_rank, inverse_mapping = partition_indices.unique(
            sorted=True, return_inverse=True
        )
        local_src_ids_per_rank = torch.arange(
            0,
            global_src_ids_per_rank.size(0),
            dtype=offsets.dtype,
            device=offsets.device,
        )
        global_src_ids_to_gpu = global_src_ids_per_rank // src_nodes_in_partition
        remote_src_ids_per_rank = (
            global_src_ids_per_rank - global_src_ids_to_gpu * src_nodes_in_partition
        )

        indices = local_src_ids_per_rank[inverse_mapping]
        graph_partition.num_indices_in_each_partition[rank] = indices.size(0)

        if rank == partition_rank:
            graph_partition.num_local_indices = indices.size(0)
            graph_partition.num_local_dst_nodes = offsets.size(0) - 1
            graph_partition.num_dst_nodes_in_each_partition = [
                dst_offsets_in_partition[rank + 1] - dst_offsets_in_partition[rank]
                for rank in range(partition_size)
            ]
            graph_partition.num_local_src_nodes = global_src_ids_per_rank.size(0)
            graph_partition.num_src_nodes_in_each_partition = [
                src_offsets_in_partition[rank + 1] - src_offsets_in_partition[rank]
                for rank in range(partition_size)
            ]

            graph_partition.partitioned_src_node_ids_to_global = range(
                src_offsets_in_partition[rank], src_offsets_in_partition[rank + 1]
            )
            graph_partition.partitioned_dst_node_ids_to_global = range(
                dst_offsets_in_partition[rank], dst_offsets_in_partition[rank + 1]
            )
            graph_partition.partitioned_indices_to_global = range(
                global_offsets[offset_start], global_offsets[offset_end]
            )

            graph_partition.local_offsets = offsets.to(device=device)
            graph_partition.local_indices = indices.to(device=device)

        for rank_offset in range(partition_size):
            mask = global_src_ids_to_gpu == rank_offset

            if partition_rank == rank_offset:
                # indices to send to this rank from this rank
                graph_partition.scatter_indices[rank] = (
                    remote_src_ids_per_rank[mask]
                    .detach()
                    .clone()
                    .to(device=device, dtype=torch.int64)
                )

            graph_partition.sizes[rank_offset][rank] = mask.sum().item()

    for r in range(graph_partition.partition_size):
        err_msg = "error in graph partition: list containing sizes of exchanged indices does not match the tensor of indices to be exchanged"
        if (
            graph_partition.sizes[graph_partition.partition_rank][r]
            != graph_partition.scatter_indices[r].numel()
        ):
            raise AssertionError(err_msg)

    return graph_partition


class DistributedGraph:
    def __init__(
        self,
        global_offsets: torch.Tensor,
        global_indices: torch.Tensor,
        partition_size: int,
        graph_partition_group_name: str = None,
        graph_partition: Optional[GraphPartition] = None,
    ):  # pragma: no cover
        """Utility Class representing a distributed graph based on a given
        partitioning of a CSC graph structure. By default, a naive node-wise
        partitioning scheme is applied, see ``partition_graph_nodewise`` for
        details on that. This class then wraps necessary communication primitives
        to access all relevant feature buffers related to the graph.

        Parameters
        ----------
        global_offsets : torch.Tensor
            CSC offsets, can live on the CPU
        global_indices : torch.Tensor
            CSC indices, can live on the CPU
        partition_size : int
            Number of process groups across which graphs are distributed, expected to
            be larger than 1, i.e. an actual partition distributed among multiple ranks.
        partition_group_name : str, default=None
            Name of process group across which graphs are distributed. Passing no process
            group name leads to a parallelism across the default process group.
            Otherwise, the group size of a process group is expected to match partition_size.
        graph_partition : GraphPartition, optional
            Optional graph_partition, if passed as None, the naive
            node-wise partitioning scheme will be applied to global_offsets and global_indices,
            otherwise, these will be ignored and the passed partition will be used instead.
        """

        dist_manager = DistributedManager()
        self.device = dist_manager.device
        self.partition_rank = dist_manager.group_rank(name=graph_partition_group_name)
        self.partition_size = dist_manager.group_size(name=graph_partition_group_name)
        error_msg = f"Passed partition_size does not correspond to size of process_group, got {partition_size} and {self.partition_size} respectively."
        if self.partition_size != partition_size:
            raise AssertionError(error_msg)
        self.process_group = dist_manager.group(name=graph_partition_group_name)

        if graph_partition is None:
            # default partitioning scheme
            self.graph_partition = partition_graph_nodewise(
                global_offsets,
                global_indices,
                self.partition_size,
                self.partition_rank,
                self.device,
            )

        else:
            error_msg = f"Passed graph_partition.partition_size does not correspond to size of process_group, got {graph_partition.partition_size} and {self.partition_size} respectively."
            if graph_partition.partition_size != self.partition_size:
                raise AssertionError(error_msg)
            error_msg = f"Passed graph_partition.device does not correspond to device of this rank, got {graph_partition.device} and {self.device} respectively."
            if graph_partition.device != self.device:
                raise AssertionError(error_msg)
            self.graph_partition = graph_partition

        dist.barrier(self.process_group)

    def get_src_node_features_in_partition(
        self,
        global_node_features: torch.Tensor,
        scatter_features: bool = False,
        src_rank: int = 0,
    ) -> torch.Tensor:  # pragma: no cover
        # if global features only on local rank 0 also scatter, split them
        # according to the partition and scatter them to other ranks
        if scatter_features:
            return scatter_v(
                global_node_features,
                self.graph_partition.num_src_nodes_in_each_partition,
                dim=0,
                src=src_rank,
                group=self.process_group,
            )

        return global_node_features[
            self.graph_partition.partitioned_src_node_ids_to_global, :
        ].to(device=self.device)

    def get_src_node_features_in_local_graph(
        self, partitioned_src_node_features: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover
        # main primitive to gather all necessary src features
        # which are required for a csc-based message passing step
        return indexed_all_to_all_v(
            partitioned_src_node_features,
            indices=self.graph_partition.scatter_indices,
            sizes=self.graph_partition.sizes,
            use_fp32=True,
            dim=0,
            group=self.process_group,
        )

    def get_dst_node_features_in_partition(
        self,
        global_node_features: torch.Tensor,
        scatter_features: bool = False,
        src_rank: int = 0,
    ) -> torch.Tensor:  # pragma: no cover
        # if global features only on local rank 0 also scatter, split them
        # according to the partition and scatter them to other ranks
        if scatter_features:
            return scatter_v(
                global_node_features,
                self.graph_partition.num_dst_nodes_in_each_partition,
                dim=0,
                src=src_rank,
                group=self.process_group,
            )

        return global_node_features[
            self.graph_partition.partitioned_dst_node_ids_to_global, :
        ].to(device=self.device)

    def get_dst_node_features_in_local_graph(
        self,
        partitioned_dst_node_features: torch.Tensor,
    ) -> torch.Tensor:  # pragma: no cover
        # current partitioning scheme assumes that
        # local graph is built from partitioned IDs
        return partitioned_dst_node_features

    def get_edge_features_in_partition(
        self,
        global_edge_features: torch.Tensor,
        scatter_features: bool = False,
        src_rank: int = 0,
    ) -> torch.Tensor:  # pragma: no cover
        # if global features only on local rank 0 also scatter, split them
        # according to the partition and scatter them to other ranks
        if scatter_features:
            return scatter_v(
                global_edge_features,
                self.graph_partition.num_indices_in_each_partition,
                dim=0,
                src=src_rank,
                process_group=self.process_group,
            )
        return global_edge_features[
            self.graph_partition.partitioned_indices_to_global, :
        ].to(device=self.device)

    def get_edge_features_in_local_graph(
        self, partitioned_edge_features: torch.Tensor
    ) -> torch.Tensor:  # pragma: no cover
        # current partitioning scheme assumes that
        # local graph is built from partitioned IDs
        return partitioned_edge_features

    def get_global_src_node_features(
        self,
        partitioned_node_features: torch.Tensor,
        get_on_all_ranks: bool = True,
        dst_rank: int = 0,
    ) -> torch.Tensor:  # pragma: no cover
        error_msg = f"Passed partitioned_node_features.device does not correspond to device of this rank, got {partitioned_node_features.device} and {self.device} respectively."
        if partitioned_node_features.device != self.device:
            raise AssertionError(error_msg)

        if not get_on_all_ranks:
            return gather_v(
                partitioned_node_features,
                self.graph_partition.num_src_nodes_in_each_partition,
                dim=0,
                dst=dst_rank,
                group=self.process_group,
            )

        return all_gather_v(
            partitioned_node_features,
            self.graph_partition.num_src_nodes_in_each_partition,
            dim=0,
            use_fp32=True,
            group=self.process_group,
        )

    def get_global_dst_node_features(
        self,
        partitioned_node_features: torch.Tensor,
        get_on_all_ranks: bool = True,
        dst_rank: int = 0,
    ) -> torch.Tensor:  # pragma: no cover
        error_msg = f"Passed partitioned_node_features.device does not correspond to device of this rank, got {partitioned_node_features.device} and {self.device} respectively."
        if partitioned_node_features.device != self.device:
            raise AssertionError(error_msg)

        if not get_on_all_ranks:
            return gather_v(
                partitioned_node_features,
                self.graph_partition.num_dst_nodes_in_each_partition,
                dim=0,
                dst=dst_rank,
                group=self.process_group,
            )

        return all_gather_v(
            partitioned_node_features,
            self.graph_partition.num_dst_nodes_in_each_partition,
            dim=0,
            use_fp32=True,
            group=self.process_group,
        )

    def get_global_edge_features(
        self,
        partitioned_edge_features: torch.Tensor,
        get_on_all_ranks: bool = True,
        dst_rank: int = 0,
    ) -> torch.Tensor:  # pragma: no cover
        error_msg = f"Passed partitioned_edge_features.device does not correspond to device of this rank, got {partitioned_edge_features.device} and {self.device} respectively."
        if partitioned_edge_features.device != self.device:
            raise AssertionError(error_msg)

        if not get_on_all_ranks:
            return gather_v(
                partitioned_edge_features,
                self.graph_partition.num_indices_in_each_partition,
                dim=0,
                dst=dst_rank,
                group=self.process_group,
            )

        return all_gather_v(
            partitioned_edge_features,
            self.graph_partition.num_indices_in_each_partition,
            dim=0,
            use_fp32=True,
            group=self.process_group,
        )
