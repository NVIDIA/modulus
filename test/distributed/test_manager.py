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

from modulus.distributed import DistributedManager, ProcessGroupConfig, ProcessGroupNode


# TODO: Need to figure out how to test parallel set up
def test_manager():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    # Reset class state
    DistributedManager._shared_state = {}
    DistributedManager.initialize()
    print(DistributedManager())

    manager = DistributedManager()

    assert manager.is_initialized()
    assert not manager.distributed, "Manager should be in serial mode"
    assert manager.rank == 0
    assert manager.world_size == 1
    assert manager.local_rank == 0

    del os.environ["RANK"]
    del os.environ["WORLD_SIZE"]


def test_manager_slurm():
    # Test distributed manager with Slurm variables
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["SLURM_PROCID"] = "0"
    os.environ["SLURM_NPROCS"] = "1"
    os.environ["SLURM_LOCALID"] = "0"
    os.environ["SLURM_LAUNCH_NODE_IPADDR"] = "localhost"
    # Reset class state
    DistributedManager._shared_state = {}
    DistributedManager.initialize()

    manager = DistributedManager()

    assert manager.is_initialized()
    assert manager.rank == 0
    assert manager.world_size == 1
    assert manager.local_rank == 0
    DistributedManager._shared_state = {}
    del os.environ["SLURM_PROCID"]
    del os.environ["SLURM_NPROCS"]
    del os.environ["SLURM_LOCALID"]
    del os.environ["SLURM_LAUNCH_NODE_IPADDR"]


def test_manager_ompi():
    # Test distributed manager with openMPI variables
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["OMPI_COMM_WORLD_RANK"] = "0"
    os.environ["OMPI_COMM_WORLD_SIZE"] = "1"
    os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = "0"
    # Reset class state
    DistributedManager._shared_state = {}
    DistributedManager.initialize()

    manager = DistributedManager()

    assert manager.is_initialized()
    assert manager.rank == 0
    assert manager.world_size == 1
    assert manager.local_rank == 0
    DistributedManager._shared_state = {}
    del os.environ["OMPI_COMM_WORLD_RANK"]
    del os.environ["OMPI_COMM_WORLD_SIZE"]
    del os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]


def test_manager_specified_initialization():
    # PyTorch env vars
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"

    # SLURM env vars
    os.environ["SLURM_PROCID"] = "0"
    os.environ["SLURM_NPROCS"] = "1"
    os.environ["SLURM_LOCALID"] = "0"
    os.environ["SLURM_LAUNCH_NODE_IPADDR"] = "localhost"

    # OpenMPI env vars
    os.environ["OMPI_COMM_WORLD_RANK"] = "0"
    os.environ["OMPI_COMM_WORLD_SIZE"] = "1"
    os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = "0"

    # Test SLURM initialization
    os.environ["MODULUS_DISTRIBUTED_INITIALIZATION_METHOD"] = "SLURM"
    DistributedManager._shared_state = {}
    DistributedManager.initialize()
    manager = DistributedManager()
    assert manager.is_initialized()
    assert manager._initialization_method == "slurm"
    assert not manager.distributed, "Manager should be in serial mode"
    assert manager.rank == 0
    assert manager.world_size == 1
    assert manager.local_rank == 0

    # Test OpenMPI initialization
    os.environ["MODULUS_DISTRIBUTED_INITIALIZATION_METHOD"] = "OPENMPI"
    DistributedManager._shared_state = {}
    DistributedManager.initialize()
    manager = DistributedManager()
    assert manager.is_initialized()
    assert manager._initialization_method == "openmpi"
    assert not manager.distributed, "Manager should be in serial mode"
    assert manager.rank == 0
    assert manager.world_size == 1
    assert manager.local_rank == 0

    del os.environ["RANK"]
    del os.environ["WORLD_SIZE"]


def test_manager_singleton():
    # Test distributed manager singleton functions as expected
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "45678"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    # Reset class state
    DistributedManager._shared_state = {}
    DistributedManager.initialize()

    manager_1 = DistributedManager()
    manager_1.broadcast_buffers = True
    manager_1.find_unused_parameters = True
    manager_2 = DistributedManager()

    # Compare attributes
    assert manager_1.rank == manager_2.rank
    assert manager_1.world_size == manager_2.world_size
    assert manager_1.local_rank == manager_2.local_rank
    assert manager_1.device == manager_2.device
    assert manager_1.distributed == manager_2.distributed
    assert manager_1.cuda == manager_2.cuda
    assert manager_1.group_names == manager_2.group_names
    assert manager_1.group() == manager_2.group()
    assert manager_1.group_size() == manager_2.group_size()
    assert manager_1.group_rank() == manager_2.group_rank()
    assert manager_1.group_name() == manager_2.group_name()
    assert manager_1.broadcast_buffers == manager_2.broadcast_buffers
    assert manager_1.find_unused_parameters == manager_2.find_unused_parameters
    DistributedManager._shared_state = {}


def run_process_groups(rank, model_parallel_size, verbose):
    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{model_parallel_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)
    DistributedManager._shared_state = {}

    DistributedManager.initialize()

    # Create model parallel process group
    DistributedManager.create_process_subgroup(
        "model_parallel", int(model_parallel_size), verbose=verbose
    )
    # Create data parallel process group for DDP allreduce
    DistributedManager.create_orthogonal_process_group(
        "data_parallel", "model_parallel", verbose=verbose
    )

    manager = DistributedManager()

    assert manager.rank == rank
    assert manager.rank == manager.group_rank(name="model_parallel")
    assert 0 == manager.group_rank(name="data_parallel")


@pytest.mark.multigpu
def test_process_groups():
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"
    model_parallel_size = 2
    verbose = False  # Change to True for debug

    torch.multiprocessing.spawn(
        run_process_groups,
        args=(model_parallel_size, verbose),
        nprocs=model_parallel_size,
        start_method="spawn",
    )


def run_process_groups_from_config(rank, model_parallel_size, verbose):
    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{model_parallel_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)
    DistributedManager._shared_state = {}

    DistributedManager.initialize()

    # Create world group that contains all processes that are part of this job
    world = ProcessGroupNode("world")

    # Create the process group config with the highest level process group
    config = ProcessGroupConfig(world)

    # Create model and data parallel sub-groups
    config.add_node(ProcessGroupNode("model_parallel"), parent="world")
    config.add_node(ProcessGroupNode("data_parallel"), parent="world")

    # Create spatial and channel parallel sub-groups
    config.add_node(ProcessGroupNode("spatial_parallel"), parent="model_parallel")
    config.add_node(ProcessGroupNode("channel_parallel"), parent="model_parallel")

    # Set leaf group sizes
    group_sizes = {"channel_parallel": 1, "spatial_parallel": 2, "data_parallel": 1}
    config.set_leaf_group_sizes(group_sizes)  # Updates all parent group sizes too

    assert (
        config.get_node("model_parallel").size == 2
    ), "Incorrect size for 'model_parallel' parent node"

    assert config.get_node("world").size == 2, "Incorrect size for 'world' parent node"

    # Create model parallel process group
    DistributedManager.create_groups_from_config(config, verbose=verbose)

    manager = DistributedManager()

    assert manager.rank == rank

    # Test that model_parallel and spatial_parallel span all the processes
    assert manager.rank == manager.group_rank(name="model_parallel")
    assert manager.rank == manager.group_rank(name="spatial_parallel")

    # Test orthogonal data_parallel group, only one total model_parallel group so
    # data_parallel rank should always be 0
    assert 0 == manager.group_rank(name="data_parallel")

    # Test channel_parallel group, group with size 1, so rank must be 0
    assert 0 == manager.group_rank(name="channel_parallel")

    # Cleanup process groups
    DistributedManager.cleanup()


@pytest.mark.multigpu
def test_process_groups_from_config():
    num_gpus = torch.cuda.device_count()
    assert num_gpus == 2, "Not enough GPUs available for test"
    model_parallel_size = 2
    verbose = False  # Change to True for debug

    torch.multiprocessing.spawn(
        run_process_groups_from_config,
        args=(model_parallel_size, verbose),
        nprocs=model_parallel_size,
        start_method="spawn",
    )


if __name__ == "__main__":
    pytest.main([__file__])
