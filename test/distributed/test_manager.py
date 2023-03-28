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
from modulus.distributed import DistributedManager

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
    os.environ["SLURM_LOCALID"] = "1"
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
    os.environ["OMPI_COMM_WORLD_RANK"] = "1"
    os.environ["OMPI_COMM_WORLD_SIZE"] = "1"
    os.environ["OMPI_COMM_WORLD_LOCAL_RANK"] = "1"
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
