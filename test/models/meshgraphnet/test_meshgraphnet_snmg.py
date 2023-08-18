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

import os
import pytest
import torch
import numpy as np

from torch.nn.parallel import DistributedDataParallel

from modulus.models.gnn_layers.utils import CuGraphCSC
from modulus.models.meshgraphnet.meshgraphnet import MeshGraphNet
from modulus.distributed import DistributedManager


def run_test_distributed_meshgraphnet(rank, world_size, dtype):
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
        "processor_size": 10,
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
    min_degree, max_degree = 3, 6
    num_nodes = 1024
    offsets = torch.empty(num_nodes + 1, dtype=torch.int64)
    offsets[0] = 0
    offsets[1:] = torch.randint(
        min_degree, max_degree + 1, (num_nodes,), dtype=torch.int64
    )
    offsets = offsets.cumsum(dim=0)
    num_indices = offsets[-1].item()
    indices = torch.randint(0, num_nodes, (num_indices,), dtype=torch.int64)

    graph_single_gpu = CuGraphCSC(
        offsets.to(manager.device),
        indices.to(manager.device),
        num_nodes,
        num_nodes,
    )
    graph_multi_gpu = CuGraphCSC(
        offsets.to(manager.device),
        indices.to(manager.device),
        num_nodes,
        num_nodes,
        partition_size=world_size,
        partition_group_name="graph_partition",
    )

    nfeat_single_gpu = (
        torch.randn((num_nodes, model_kwds["input_dim_nodes"]))
        .to(device=manager.device, dtype=dtype)
        .requires_grad_(True)
    )
    efeat_single_gpu = (
        torch.randn((num_indices, model_kwds["input_dim_edges"]))
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
        torch.float32: (1e-2, 1e-4),
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
        diff = param - model_multi_gpu_parameters[param_idx]
        diff = torch.abs(diff)
        mask = diff > atol
        assert torch.allclose(
            param, model_multi_gpu_parameters[param_idx], atol=atol, rtol=rtol
        ), f"{mask.sum()} elements have diff > {atol} \n {param[mask]} \n {model_multi_gpu_parameters[param_idx][mask]}"

    # cleanup distributed
    del os.environ["RANK"]
    del os.environ["LOCAL_RANK"]
    del os.environ["WORLD_SIZE"]
    del os.environ["MASTER_ADDR"]
    del os.environ["MASTER_PORT"]

    DistributedManager.cleanup()


@pytest.mark.multigpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_distributed_meshgraphnet(dtype):
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"
    world_size = num_gpus

    torch.multiprocessing.spawn(
        run_test_distributed_meshgraphnet,
        args=(world_size, dtype),
        nprocs=world_size,
        start_method="spawn",
    )


if __name__ == "__main__":
    pytest.main([__file__])
