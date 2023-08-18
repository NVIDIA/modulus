# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch.distributed as dist

from typing import List, Optional

from modulus.distributed import DistributedManager
from modulus.distributed import (
    gather_v,
    scatter_v,
    all_gather_v,
    indexed_all_to_all_v,
)


class DistributedGraph:
    def __init__(
        self,
        global_offsets: torch.Tensor,
        global_indices: torch.Tensor,
        partition_size: int,
        graph_partition_group_name: str,
    ):
        """Utility Class representing a distributed graph. Based on a CSC
        structure represented by corresponding offsets and indices buffers,
        the global graph is partioned into smaller local graphs.

        Parameters
        ----------
        global_offsets : torch.Tensor
            CSC offsets
        global_indices : torch.Tensor
            CSC indices
        partition_size : int
            number of process groups across which graph is partitioned
        graph_partition_group_name : str
            name of process group of ranks across which graph is partitioned
        """

        # global information about node ids and edge ids
        self.num_global_src_nodes = global_indices.max().item() + 1
        self.num_global_dst_nodes = global_offsets.size(0) - 1
        self.num_global_indices = global_indices.size(0)

        # csc-structure of local graph
        self.local_offsets = None
        self.local_indices = None
        self.num_local_src_nodes = (
            None  # number of unique source nodes of local graph, i.e.
        )
        self.num_local_dst_nodes = None  # number of destination nodes of local graph
        self.num_local_indices = None  # number of indices of local graph

        # source, destination, and edge ids per partition
        self.dst_nodes_in_partition = (
            None  # global IDs of destination nodes in this partition
        )
        self.src_nodes_in_partition = (
            None  # global IDs of source nodes in this partition
        )
        self.num_src_nodes_in_partition = (
            None  # number of partitioned source nodes in the local partition
        )
        self.num_dst_nodes_in_partition = None  # number of partitioned destination nodes in the local partition (=self.num_local_dst_nodes)
        self.num_indices_in_partition = None  # number of partitioned indices in this partition (=self.num_local_indices)
        self.num_src_nodes_in_each_partition = [None] * partition_size
        self.num_indices_in_each_partition = [None] * partition_size
        self.num_dst_nodes_in_each_partition = [None] * partition_size

        # mapping of local IDs to their corresponding global IDs
        self.partitioned_indices_to_global = None
        self.partitioned_src_node_ids_to_global = None
        self.partitioned_dst_node_ids_to_global = None

        # utility variables for torch.distributed
        dist_manager = DistributedManager()
        self.device_id = dist_manager.device
        self.partition_rank = dist_manager.group_rank(name=graph_partition_group_name)
        self.partition_size = dist_manager.group_size(name=graph_partition_group_name)
        error_msg = f"Passed partition_size does not correspond to size of process_group, got {partition_size} and {self.partition_size} respectively."
        assert self.partition_size == partition_size, error_msg
        self.process_group = dist_manager.group(name=graph_partition_group_name)

        # this partitions offsets and indices on each rank in the same fashion
        # it could be rewritten to do it on one rank and exchange the partitions
        # however, as we expect the global graphs not to be too large for one CPU
        # we do it once and then can get rid of it afterwards without going through
        # tedious gather/scatter routines for communicating the partitions

        # get distribution of destination IDs
        dst_nodes_in_partition = (
            self.num_global_dst_nodes + self.partition_size - 1
        ) // self.partition_size
        dst_offsets_in_partition = [
            rank * dst_nodes_in_partition for rank in range(self.partition_size + 1)
        ]
        dst_offsets_in_partition[-1] = min(
            self.num_global_dst_nodes, dst_offsets_in_partition[-1]
        )

        src_nodes_in_partition = (
            self.num_global_src_nodes + self.partition_size - 1
        ) // self.partition_size
        src_offsets_in_partition = [
            rank * src_nodes_in_partition for rank in range(self.partition_size + 1)
        ]
        src_offsets_in_partition[-1] = min(
            self.num_global_src_nodes, src_offsets_in_partition[-1]
        )

        scatter_indices = [None] * self.partition_size
        sizes = [
            [None for _ in range(self.partition_size)] for _ in range(partition_size)
        ]

        for rank in range(self.partition_size):
            offset_start = dst_offsets_in_partition[rank]
            offset_end = dst_offsets_in_partition[rank + 1]
            offsets = global_offsets[offset_start : offset_end + 1].detach().clone()
            partition_indices = (
                global_indices[offsets[0] : offsets[-1]].detach().clone()
            )
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
            self.num_indices_in_each_partition[rank] = indices.size(0)

            if rank == self.partition_rank:
                self.num_local_indices = indices.size(0)
                self.num_indices_in_partition = self.num_local_indices
                self.num_local_dst_nodes = offsets.size(0) - 1
                self.num_dst_nodes_in_each_partition = [
                    dst_offsets_in_partition[rank + 1] - dst_offsets_in_partition[rank]
                    for rank in range(self.partition_size)
                ]
                self.num_dst_nodes_in_partition = self.num_local_dst_nodes
                self.num_local_src_nodes = global_src_ids_per_rank.size(0)
                self.num_src_nodes_in_each_partition = [
                    src_offsets_in_partition[rank + 1] - src_offsets_in_partition[rank]
                    for rank in range(self.partition_size)
                ]
                self.num_src_nodes_in_partition = self.num_src_nodes_in_each_partition[
                    rank
                ]
                self.partitioned_src_node_ids_to_global = range(
                    src_offsets_in_partition[rank], src_offsets_in_partition[rank + 1]
                )
                self.partitioned_dst_node_ids_to_global = range(
                    dst_offsets_in_partition[rank], dst_offsets_in_partition[rank + 1]
                )
                self.partitioned_indices_to_global = range(
                    global_offsets[offset_start], global_offsets[offset_end]
                )

                self.local_offsets = offsets.detach().clone().to(device=self.device_id)
                self.local_indices = indices.detach().clone().to(device=self.device_id)

            for rank_offset in range(self.partition_size):
                mask = global_src_ids_to_gpu == rank_offset

                if self.partition_rank == rank_offset:
                    # indices to send to this rank from this rank
                    scatter_indices[rank] = (
                        remote_src_ids_per_rank[mask]
                        .detach()
                        .clone()
                        .to(device=self.device_id, dtype=torch.int64)
                    )

                sizes[rank_offset][rank] = mask.sum().item()

        self.sizes = sizes
        self.scatter_indices = scatter_indices

        for r in range(self.partition_size):
            assert self.sizes[self.partition_rank][r] == self.scatter_indices[r].numel()

        dist.barrier(self.process_group)

    def get_src_node_features_in_partition(
        self,
        global_node_features: torch.Tensor,
        scatter_features: bool = False,
    ) -> torch.Tensor:
        # if global features only on local rank 0 also scatter, split them
        # according to the partition and scatter them to other ranks
        if scatter_features:
            return scatter_v(
                global_node_features,
                self.num_src_nodes_in_each_partition,
                dim=0,
                src=0,
                group=self.process_group,
            )

        return global_node_features[self.partitioned_src_node_ids_to_global, :].to(
            device=self.device_id
        )

    def get_src_node_features_in_local_graph(
        self, partitioned_src_node_features: torch.Tensor
    ) -> torch.Tensor:
        # main primitive to gather all necessary src features
        # which are required for a csc-based message passing step
        return indexed_all_to_all_v(
            partitioned_src_node_features,
            indices=self.scatter_indices,
            sizes=self.sizes,
            use_fp32=True,
            dim=0,
            group=self.process_group,
        )

    def get_dst_node_features_in_partition(
        self,
        global_node_features: torch.Tensor,
        scatter_features: bool = False,
    ) -> torch.Tensor:
        # if global features only on local rank 0 also scatter, split them
        # according to the partition and scatter them to other ranks
        if scatter_features:
            return scatter_v(
                global_node_features,
                self.num_dst_nodes_in_each_partition,
                dim=0,
                src=0,
                group=self.process_group,
            )

        return global_node_features[self.partitioned_dst_node_ids_to_global, :].to(
            device=self.device_id
        )

    def get_dst_node_features_in_local_graph(
        self,
        partitioned_dst_node_features: torch.Tensor,
    ) -> torch.Tensor:
        # current partitioning scheme assumes that
        # local graph is built from partitioned IDs
        return partitioned_dst_node_features

    def get_edge_features_in_partition(
        self,
        global_edge_features: torch.Tensor,
        scatter_features: bool = False,
    ) -> torch.Tensor:
        # if global features only on local rank 0 also scatter, split them
        # according to the partition and scatter them to other ranks
        if scatter_features:
            return scatter_v(
                global_edge_features,
                self.num_indices_in_each_partition,
                dim=0,
                src=0,
                process_group=self.process_group,
            )
        return global_edge_features[self.partitioned_indices_to_global, :].to(
            device=self.device_id
        )

    def get_edge_features_in_local_grpah(
        self, partitioned_edge_features: torch.Tensor
    ) -> torch.Tensor:
        # current partitioning scheme assumes that
        # local graph is built from partitioned IDs
        return partitioned_edge_features

    def get_global_src_node_features(
        self,
        partitioned_node_features: torch.Tensor,
        get_on_all_ranks: bool = True,
    ) -> torch.Tensor:
        if not get_on_all_ranks:
            return gather_v(
                partitioned_node_features,
                self.num_src_nodes_in_each_partition,
                dim=0,
                dst=0,
                group=self.process_group,
            )

        return all_gather_v(
            partitioned_node_features,
            self.num_src_nodes_in_each_partition,
            dim=0,
            use_fp32=True,
            group=self.process_group,
        )

    def get_global_dst_node_features(
        self, partitioned_node_features: torch.Tensor, get_on_all_ranks: bool = True
    ) -> torch.Tensor:
        if not get_on_all_ranks:
            return gather_v(
                partitioned_node_features,
                self.num_dst_nodes_in_each_partition,
                dim=0,
                dst=0,
                group=self.process_group,
            )

        return all_gather_v(
            partitioned_node_features,
            self.num_dst_nodes_in_each_partition,
            dim=0,
            use_fp32=True,
            group=self.process_group,
        )

    def get_global_edge_features(
        self, partitioned_edge_features: torch.Tensor, get_on_all_ranks: bool = True
    ) -> torch.Tensor:
        if not get_on_all_ranks:
            return gather_v(
                partitioned_edge_features,
                self.num_indices_in_each_partition,
                dim=0,
                dst=0,
                group=self.process_group,
            )

        return all_gather_v(
            partitioned_edge_features,
            self.num_indices_in_each_partition,
            dim=0,
            use_fp32=True,
            group=self.process_group,
        )
