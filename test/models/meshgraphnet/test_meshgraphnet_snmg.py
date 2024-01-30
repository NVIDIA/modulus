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
# ruff: noqa: E402
import os
import sys

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import numpy as np
import pytest
import torch
from meshgraphnet.utils import get_random_graph
from pytest_utils import import_or_fail
from torch.nn.parallel import DistributedDataParallel

from modulus.distributed import DistributedManager
from modulus.models.gnn_layers import (
    partition_graph_by_coordinate_bbox,
    partition_graph_with_id_mapping,
)


def run_test_distributed_meshgraphnet(rank, world_size, dtype, partition_scheme):
    from modulus.models.gnn_layers.utils import CuGraphCSC
    from modulus.models.meshgraphnet.meshgraphnet import MeshGraphNet

    os.environ["RANK"] = f"{rank}"
    os.environ["LOCAL_RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)

    DistributedManager.initialize()
    DistributedManager.create_process_subgroup(
        name="graph_partition",
        size=world_size,
    )
    manager = DistributedManager()
    assert manager.is_initialized() and manager._distributed

    model_kwds = {
        "input_dim_nodes": 3,
        "input_dim_edges": 4,
        "output_dim": 5,
        "processor_size": 4,
        "num_layers_node_processor": 2,
        "num_layers_edge_processor": 2,
        "hidden_dim_node_encoder": 256,
        "num_layers_node_encoder": 2,
        "hidden_dim_edge_encoder": 256,
        "num_layers_edge_encoder": 2,
        "hidden_dim_node_decoder": 256,
        "num_layers_node_decoder": 2,
    }

    # initialize single GPU model for reference
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    model_single_gpu = MeshGraphNet(**model_kwds).to(device=manager.device, dtype=dtype)
    # initialze distributed model with the same seeds
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    model_multi_gpu = MeshGraphNet(**model_kwds).to(device=manager.device, dtype=dtype)
    model_multi_gpu = DistributedDataParallel(
        model_multi_gpu,
        process_group=manager.group("graph_partition"),
    )

    # initialize data
    num_nodes = 1024
    offsets, indices = get_random_graph(
        num_nodes=num_nodes,
        min_degree=3,
        max_degree=6,
    )

    graph_single_gpu = CuGraphCSC(
        offsets.to(manager.device),
        indices.to(manager.device),
        num_nodes,
        num_nodes,
    )

    graph_partition = None

    if partition_scheme == "nodewise":
        pass  # nodewise is default

    elif partition_scheme == "coordinate_bbox":
        src_coordinates = torch.rand((num_nodes, 1), device=offsets.device)
        dst_coordinates = src_coordinates

        step_size = 1.0 / (world_size + 1)
        coordinate_separators_min = [[step_size * p] for p in range(world_size)]
        coordinate_separators_max = [[step_size * (p + 1)] for p in range(world_size)]

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

    graph_multi_gpu = CuGraphCSC(
        offsets.to(manager.device),
        indices.to(manager.device),
        num_nodes,
        num_nodes,
        partition_size=world_size,
        partition_group_name="graph_partition",
        graph_partition=graph_partition,
    )

    nfeat_single_gpu = (
        torch.randn((num_nodes, model_kwds["input_dim_nodes"]))
        .to(device=manager.device, dtype=dtype)
        .requires_grad_(True)
    )
    efeat_single_gpu = (
        torch.randn((indices.numel(), model_kwds["input_dim_edges"]))
        .to(device=manager.device, dtype=dtype)
        .requires_grad_(True)
    )
    nfeat_multi_gpu = nfeat_single_gpu.detach().clone()
    nfeat_multi_gpu = graph_multi_gpu.get_src_node_features_in_partition(
        nfeat_multi_gpu
    ).requires_grad_(True)
    efeat_multi_gpu = efeat_single_gpu.detach().clone()
    efeat_multi_gpu = graph_multi_gpu.get_edge_features_in_partition(
        efeat_multi_gpu
    ).requires_grad_(True)

    # zero grads
    for param in model_single_gpu.parameters():
        param.data.zero_()
    for param in model_multi_gpu.parameters():
        param.data.zero_()

    # forward + backward passes
    out_single_gpu = model_single_gpu(
        nfeat_single_gpu, efeat_single_gpu, graph_single_gpu
    )
    loss = out_single_gpu.sum()
    loss.backward()

    out_multi_gpu = model_multi_gpu(nfeat_multi_gpu, efeat_multi_gpu, graph_multi_gpu)
    # PyTorch unfortunately averages across all process groups within DistributedDataParallel
    # by default, for tensor-parallel applications like this one, potential solutions
    # are either custom gradient hooks (the best solution for actual training workloads)
    # or multiplying by world_size to cancel out the normalization. As this is just a simple
    # test, we do the easier thing here as we only compare the weight gradients.
    loss = out_multi_gpu.sum() * world_size

    loss.backward()

    # numeric tolerances based on dtype
    tolerances = {
        torch.float32: (1e-3, 1e-6),
        torch.bfloat16: (1e-1, 1e-3),
        torch.float16: (1e-1, 1e-3),
    }
    atol, rtol = tolerances[dtype]

    # compare forward
    out_single_gpu_dist = graph_multi_gpu.get_src_node_features_in_partition(
        out_single_gpu
    )
    diff = out_single_gpu_dist - out_multi_gpu
    diff = torch.abs(diff)
    mask = diff > atol
    assert torch.allclose(
        out_single_gpu_dist, out_multi_gpu, atol=atol, rtol=rtol
    ), f"{mask.sum()} elements have diff > {atol} \n {out_single_gpu_dist[mask]} \n {out_multi_gpu[mask]}"

    # compare model gradients (ensure correctness of backward)
    model_multi_gpu_parameters = list(model_multi_gpu.parameters())
    for param_idx, param in enumerate(model_single_gpu.parameters()):
        diff = param.grad - model_multi_gpu_parameters[param_idx].grad
        diff = torch.abs(diff)
        mask = diff > atol
        assert torch.allclose(
            param.grad, model_multi_gpu_parameters[param_idx].grad, atol=atol, rtol=rtol
        ), f"{mask.sum()} elements have diff > {atol} \n {param.grad[mask]} \n {model_multi_gpu_parameters[param_idx].grad[mask]}"

    # cleanup distributed
    del os.environ["RANK"]
    del os.environ["LOCAL_RANK"]
    del os.environ["WORLD_SIZE"]
    del os.environ["MASTER_ADDR"]
    del os.environ["MASTER_PORT"]

    DistributedManager.cleanup()


@import_or_fail("dgl")
@pytest.mark.multigpu
@pytest.mark.parametrize("partition_scheme", ["nodewise", "coordinate_bbox", "mapping"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_distributed_meshgraphnet(dtype, partition_scheme, pytestconfig):
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"
    world_size = min(
        4, num_gpus
    )  # test-graph is otherwise too small for distribution across more GPUs

    torch.multiprocessing.spawn(
        run_test_distributed_meshgraphnet,
        args=(world_size, dtype, partition_scheme),
        nprocs=world_size,
        start_method="spawn",
    )


if __name__ == "__main__":
    pytest.main([__file__])
