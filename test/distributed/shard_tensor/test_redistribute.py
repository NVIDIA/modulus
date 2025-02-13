# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import pytest
import torch
import torch.distributed as dist
from test_initialization import (
    init_dist,
    init_global_shape_and_placements,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

from modulus.distributed import DistributedManager, ShardTensor


def shard_tensor_factory(mesh_names, mesh_sizes, requires_grad=False):
    """
    Generate a shard tensor on the mesh
    """

    dm = DistributedManager()

    # Create a mesh right from the inputs:
    global_mesh = dm.initialize_mesh(mesh_sizes, mesh_names)  # noqa: F841
    domain_mesh, _, placements = init_global_shape_and_placements(
        mesh_names,
    )

    local_shape = [
        10,
    ]

    min_size = 64
    index_stride = 16

    # Using the same size per rank in mesh dimension
    for dim in range(domain_mesh.ndim):
        dim_rank = dist.get_group_rank(domain_mesh.get_group(dim), dm.rank)
        local_shape.append(min_size + dim_rank * index_stride)

    local_shape.append(100)

    raw_data = torch.randn(
        local_shape,
        device=torch.device(f"cuda:{dm.local_rank}"),
        requires_grad=requires_grad,
    )

    st = ShardTensor.from_local(
        raw_data, device_mesh=domain_mesh, placements=placements, infer_shape=True
    )
    return st


def run_shard_tensor_global_reduction(
    rank, num_gpus, mesh_names, mesh_sizes, op, verbose
):

    init_dist(rank, num_gpus)

    shard_tensor = shard_tensor_factory(mesh_names, mesh_sizes)

    # For this test, we're testing that the reduction of the tensor works correctly

    # This means we're calling things like `shard_tensor.max()` or `shard_tensor.mean()`
    # and looking to get the right answers

    # Note that calling `full_tensor` is already checked in the initialize file but that's
    # also, technically, a reduction.

    # Perform the operation partially then reduce:
    partial_result = op(shard_tensor).full_tensor()

    # Replicate, then perform the operation
    full_result = op(shard_tensor.full_tensor())

    # We should have agreement on every rank
    if verbose:
        print(f"Partial first: {partial_result}")
        print(f"All gather first: {full_result}")
    assert torch.allclose(partial_result, full_result)


@pytest.mark.multigpu
@pytest.mark.parametrize("data_parallel_size", [-1])
@pytest.mark.parametrize("domain_H", [2, 4])
@pytest.mark.parametrize("domain_W", [1, 2])
@pytest.mark.parametrize("op", [torch.sum, torch.min, torch.max, torch.mean])
def test_shard_tensor_global_reduction(data_parallel_size, domain_H, domain_W, op):
    """
    This test is meant to ensure ShardTensor can be initialized correctly
    from local data. Checks that reduction operations work correctly.

    Note: Mean reduction is expected to fail since averaging over uneven tensor shapes
    is not yet supported.
    """
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"

    if domain_H == 1 and domain_W == 1:
        pytest.skip("No point testing this without parallelism in the domain axes")

    if op == torch.mean:
        pytest.xfail("Mean reduction not yet supported for uneven tensor shapes")

    remaining_gpus = num_gpus
    mesh_names = ["data_parallel"]
    mesh_sizes = [data_parallel_size]

    if int(remaining_gpus / domain_H) != 0:
        mesh_names.append("domain_H")
        mesh_sizes.append(domain_H)
        remaining_gpus = int(remaining_gpus / domain_H)

    if int(remaining_gpus / domain_W) != 0:
        mesh_names.append("domain_W")
        mesh_sizes.append(domain_W)
        remaining_gpus = int(remaining_gpus / domain_W)

    verbose = False  # Change to True for debug

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_shard_tensor_global_reduction,
        args=(num_gpus, mesh_names, mesh_sizes, op, verbose),
        nprocs=num_gpus,
        join=True,
        daemon=True,
    )


def run_shard_tensor_to_replicate(
    rank, num_gpus, mesh_names, mesh_sizes, dst_placements, verbose
):
    """Test redistribution between different sharding schemes"""

    init_dist(rank, num_gpus)

    # Create initial sharded tensor
    shard_tensor = shard_tensor_factory(mesh_names, mesh_sizes)
    if verbose:
        print(f"shard_tensor placements: {shard_tensor._spec.placements}")

    # Redistribute to new placement
    redistributed = shard_tensor.redistribute(placements=dst_placements)

    # Verify data is preserved after redistribution
    redistributed_data = redistributed.full_tensor()

    # Store original data for validation
    original_data = shard_tensor.full_tensor()

    if verbose:
        print(
            f"Original shape: {shard_tensor.shape} vs Redistributed shape: {redistributed.shape}"
        )
        print(f"Original data: {original_data}")
        print(f"Redistributed data: {redistributed_data}")

    assert torch.allclose(original_data, redistributed_data)


@pytest.mark.multigpu
@pytest.mark.parametrize("data_parallel_size", [-1])
@pytest.mark.parametrize("domain_H", [2, 4])
@pytest.mark.parametrize("domain_W", [1, 2])
@pytest.mark.parametrize(
    "redistribution_case",
    [
        # Test cases for different redistribution scenarios
        ("S1+R", [Shard(1), Replicate()]),
        ("R+S1", [Replicate(), Shard(1)]),
        # ("S2+R", [Shard(2), Replicate()]),
        ("R+S2", [Replicate(), Shard(2)]),
        ("R+R", [Replicate(), Replicate()]),
    ],
)
def test_shard_tensor_to_replicate(
    data_parallel_size, domain_H, domain_W, redistribution_case
):
    """Test different redistribution scenarios for ShardTensor"""

    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"

    if domain_H == 1 and domain_W == 1:
        pytest.skip("No point testing this without parallelism in the domain axes")

    case_name, dst_placements = redistribution_case

    remaining_gpus = num_gpus
    mesh_names = ["data_parallel"]
    mesh_sizes = [data_parallel_size]

    if int(remaining_gpus / domain_H) != 0:
        mesh_names.append("domain_H")
        mesh_sizes.append(domain_H)
        remaining_gpus = int(remaining_gpus / domain_H)

    if int(remaining_gpus / domain_W) != 0:
        mesh_names.append("domain_W")
        mesh_sizes.append(domain_W)
        remaining_gpus = int(remaining_gpus / domain_W)

    # To perform the shard-to-shard test, the mesh needs to have more than 2 dimensions
    if case_name == "shard_to_shard" and len(mesh_sizes) <= 2:
        pytest.skip("Not enough dimensions for shard-to-shard test")

    verbose = False  # Change to True for debug

    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        run_shard_tensor_to_replicate,
        args=(num_gpus, mesh_names, mesh_sizes, dst_placements, verbose),
        nprocs=num_gpus,
        join=True,
        daemon=True,
    )


if __name__ == "__main__":

    cases = [
        ("S1+R", [Shard(1), Replicate()]),
        ("R+S1", [Replicate(), Shard(1)]),
        ("R+S2", [Replicate(), Shard(2)]),
        ("R+R", [Replicate(), Replicate()]),
    ]

    case = cases[2]
    test_shard_tensor_to_replicate(-1, 4, 1, case)
