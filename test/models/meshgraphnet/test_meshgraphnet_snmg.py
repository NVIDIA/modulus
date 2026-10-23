# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
# ruff: noqa: E402
import os
import sys

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import numpy as np
import pytest
import torch

from physicsnemo.distributed import DistributedManager, mark_module_as_shared
from test.conftest import requires_module

torch.backends.cuda.matmul.allow_tf32 = False


def run_test_distributed_meshgraphnet(rank, world_size, dtype, partition_scheme):
    from torch_geometric.data import Data

    from physicsnemo.models.meshgraphnet.meshgraphnet import MeshGraphNet
    from physicsnemo.nn.module.gnn_layers import (
        DistributedGraph,
        edge_index_to_csc,
        partition_graph_by_coordinate_bbox,
        partition_graph_nodewise,
        partition_graph_with_id_mapping,
    )

    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)
    local_rank = rank % torch.cuda.device_count()
    os.environ["LOCAL_RANK"] = f"{local_rank}"

    DistributedManager.initialize()

    manager = DistributedManager()
    DistributedManager.create_process_subgroup(
        name="graph_partition",
        size=world_size,
    )
    assert manager.is_initialized() and manager._distributed

    model_kwds = {
        "input_dim_nodes": 32,
        "input_dim_edges": 32,
        "output_dim": 32,
        "processor_size": 2,
        "num_layers_node_processor": 2,
        "num_layers_edge_processor": 2,
        "hidden_dim_node_encoder": 32,
        "num_layers_node_encoder": 2,
        "hidden_dim_edge_encoder": 32,
        "num_layers_edge_encoder": 2,
        "hidden_dim_node_decoder": 32,
        "num_layers_node_decoder": 2,
    }

    # initialize single GPU model for reference
    torch.cuda.manual_seed(753)
    torch.manual_seed(753)
    np.random.seed(753)
    model_single_gpu = MeshGraphNet(**model_kwds).to(device=manager.device, dtype=dtype)

    # initialze distributed model with the same seeds
    torch.cuda.manual_seed(753)
    torch.manual_seed(753)
    np.random.seed(753)

    model_multi_gpu = MeshGraphNet(**model_kwds).to(device=manager.device, dtype=dtype)
    mark_module_as_shared(model_multi_gpu, "graph_partition")

    # initialize data
    torch.cuda.manual_seed(753)
    torch.manual_seed(753)
    np.random.seed(753)
    num_nodes = 256

    nfeat_single_gpu = torch.randn((num_nodes, model_kwds["input_dim_nodes"]))
    nfeat_single_gpu = nfeat_single_gpu.to(device=manager.device, dtype=dtype)
    nfeat_single_gpu = nfeat_single_gpu.requires_grad_(True)

    # random graph with 3 to 6 incoming edges per node, with edges in random
    # order so that the multi-GPU path has to permute edge features
    degree = torch.randint(3, 7, (num_nodes,))
    dst = torch.repeat_interleave(torch.arange(num_nodes), degree)
    src = torch.randint(0, num_nodes, (dst.numel(),))
    src[-1] = num_nodes - 1  # nodewise partitioning infers the number of nodes
    edge_order = torch.randperm(dst.numel())
    edge_index = torch.stack([src, dst])[:, edge_order]

    efeat_single_gpu = torch.randn((edge_index.size(1), model_kwds["input_dim_edges"]))
    efeat_single_gpu = efeat_single_gpu.to(device=manager.device, dtype=dtype)
    efeat_single_gpu = efeat_single_gpu.requires_grad_(True)

    graph_single_gpu = Data(edge_index=edge_index, num_nodes=num_nodes).to(
        manager.device
    )
    offsets, indices, edge_perm = edge_index_to_csc(edge_index, num_nodes)
    process_group = None

    if partition_scheme == "nodewise":
        # nodewise should be default
        graph_partition = partition_graph_nodewise(
            offsets,
            indices,
            world_size,
            manager.rank,
            manager.device,
        )

    elif partition_scheme == "none":
        # check default which should be nodewise
        graph_partition = None

    elif partition_scheme == "process_group":
        # default partitioning across a process group passed directly
        graph_partition = None
        process_group = manager.group("graph_partition")

    elif partition_scheme == "coordinate_bbox":
        src_coordinates = torch.rand((num_nodes, 1), device=offsets.device)
        dst_coordinates = src_coordinates

        step_size = 1.0 / (world_size + 1)
        coordinate_separators_min = [[step_size * p] for p in range(world_size)]
        coordinate_separators_max = [[step_size * (p + 1)] for p in range(world_size)]
        coordinate_separators_min[0] = [None]
        coordinate_separators_max[-1] = [None]

        graph_partition = partition_graph_by_coordinate_bbox(
            offsets,
            indices,
            src_coordinates,
            dst_coordinates,
            coordinate_separators_min,
            coordinate_separators_max,
            world_size,
            manager.rank,
            manager.device,
        )

    elif partition_scheme == "mapping":
        mapping_src_ids_to_ranks = torch.randint(
            0, world_size, (num_nodes,), device=offsets.device
        )
        mapping_dst_ids_to_ranks = mapping_src_ids_to_ranks

        graph_partition = partition_graph_with_id_mapping(
            offsets,
            indices,
            mapping_src_ids_to_ranks,
            mapping_dst_ids_to_ranks,
            world_size,
            manager.rank,
            manager.device,
        )

    else:
        assert False  # only schemes above are supported

    graph_multi_gpu = DistributedGraph(
        offsets,
        indices,
        partition_size=world_size,
        graph_partition_group_name=(
            None if process_group is not None else "graph_partition"
        ),
        graph_partition=graph_partition,
        process_group=process_group,
    )

    nfeat_multi_gpu = nfeat_single_gpu.detach().clone()
    nfeat_multi_gpu = graph_multi_gpu.get_src_node_features_in_partition(
        nfeat_multi_gpu
    ).requires_grad_(True)
    efeat_multi_gpu = efeat_single_gpu.detach().clone()
    efeat_multi_gpu = graph_multi_gpu.get_edge_features_in_partition(
        efeat_multi_gpu[edge_perm]
    ).requires_grad_(True)

    # zero grads
    for param in model_single_gpu.parameters():
        param.grad = None
    for param in model_multi_gpu.parameters():
        param.grad = None

    # forward + backward passes
    out_single_gpu = model_single_gpu(
        nfeat_single_gpu, efeat_single_gpu, graph_single_gpu
    )
    loss = out_single_gpu.sum()
    loss.backward()

    out_multi_gpu = model_multi_gpu(nfeat_multi_gpu, efeat_multi_gpu, graph_multi_gpu)
    loss = out_multi_gpu.sum()
    loss.backward()

    # numeric tolerances based on dtype
    tolerances = {
        torch.float32: (1e-4, 1e-5),
        torch.float16: (1e-2, 1e-3),
    }
    tolerances_weight = {
        torch.float32: (1e-2, 1e-4),
        torch.float16: (1e-1, 1e-2),
    }
    atol, rtol = tolerances[dtype]
    atol_w, rtol_w = tolerances_weight[dtype]

    # compare forward
    out_single_gpu_dist = graph_multi_gpu.get_src_node_features_in_partition(
        out_single_gpu
    )
    diff = out_single_gpu_dist - out_multi_gpu
    diff = torch.abs(diff)
    mask = diff > atol
    assert torch.allclose(out_single_gpu_dist, out_multi_gpu, atol=atol, rtol=rtol), (
        f"{mask.sum()} elements have diff > {atol} \n {out_single_gpu_dist[mask]} \n {out_multi_gpu[mask]}"
    )

    # compare data gradient
    nfeat_grad_single_gpu_dist = graph_multi_gpu.get_src_node_features_in_partition(
        nfeat_single_gpu.grad
    )
    diff = nfeat_grad_single_gpu_dist - nfeat_multi_gpu.grad
    diff = torch.abs(diff)
    mask = diff > atol
    assert torch.allclose(
        nfeat_multi_gpu.grad, nfeat_grad_single_gpu_dist, atol=atol_w, rtol=rtol_w
    ), (
        f"{mask.sum()} elements have diff > {atol} \n {nfeat_grad_single_gpu_dist[mask]} \n {nfeat_multi_gpu.grad[mask]}"
    )

    # compare model gradients (ensure correctness of backward)
    model_multi_gpu_parameters = list(model_multi_gpu.parameters())
    for param_idx, param in enumerate(model_single_gpu.parameters()):
        diff = param.grad - model_multi_gpu_parameters[param_idx].grad
        diff = torch.abs(diff)
        mask = diff > atol_w
        assert torch.allclose(
            param.grad,
            model_multi_gpu_parameters[param_idx].grad,
            atol=atol_w,
            rtol=rtol_w,
        ), (
            f"{mask.sum()} for param[{param_idx}].grad elements have diff > {atol_w} with a avg. diff of {diff[mask].mean().item()} ({diff.mean().item()} overall)"
        )

    # cleanup distributed
    del os.environ["RANK"]
    del os.environ["WORLD_SIZE"]
    del os.environ["MASTER_ADDR"]
    del os.environ["MASTER_PORT"]
    del os.environ["LOCAL_RANK"]

    DistributedManager.cleanup()


@requires_module(["torch_geometric", "torch_scatter"])
@pytest.mark.multigpu_dynamic
@pytest.mark.parametrize(
    "partition_scheme",
    ["mapping", "nodewise", "coordinate_bbox", "none", "process_group"],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_distributed_meshgraphnet(dtype, partition_scheme):
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"
    world_size = num_gpus

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_test_distributed_meshgraphnet,
        args=(world_size, dtype, partition_scheme),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    pytest.main([__file__])
