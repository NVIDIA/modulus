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
from typing import Optional
import os
import numpy as np

from warnings import warn


class DistributedManager(object):
    """Distributed Manager for setting up distributed training enviroment.

    This is a singleton that creates a persistance class instance for storing parallel
    environment information through out the life time of the program. This should be
    used to help set up Distributed Data Parallel and parallel datapipes.

    Note
    ----
    One should call `DistributedManager.initialize()` prior to constructing a manager
    object

    Example
    -------
    >>> DistributedManager.initialize()
    >>> manager = DistributedManager()
    >>> manager.rank
    0
    >>> manager.world_size
    1
    """

    _shared_state = {}

    def __new__(cls):
        obj = super(DistributedManager, cls).__new__(cls)
        obj.__dict__ = cls._shared_state

        # Set the defaults
        if not hasattr(obj, "_rank"):
            obj._rank = 0
        if not hasattr(obj, "_world_size"):
            obj._world_size = 1
        if not hasattr(obj, "_local_rank"):
            obj._local_rank = 0
        if not hasattr(obj, "_distributed"):
            obj._distributed = False
        if not hasattr(obj, "_device"):
            obj._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not hasattr(obj, "_cuda"):
            obj._cuda = torch.cuda.is_available()
        if not hasattr(obj, "_broadcast_buffers"):
            obj._broadcast_buffers = False
        if not hasattr(obj, "_find_unused_parameters"):
            obj._find_unused_parameters = False
        if not hasattr(obj, "_initialization_method"):
            obj._initialization_method = "None"
        if not hasattr(obj, "_groups"):
            obj._groups = {}
        if not hasattr(obj, "_group_ranks"):
            obj._group_ranks = {}
        if not hasattr(obj, "_group_names"):
            obj._group_names = {}

        return obj

    @property
    def rank(self):
        """Process rank"""
        return self._rank

    @property
    def local_rank(self):
        """Process rank on local machine"""
        return self._local_rank

    @property
    def world_size(self):
        """Number of processes in distributed enviroment"""
        return self._world_size

    @property
    def device(self):
        """Process device"""
        return self._device

    @property
    def distributed(self):
        """Distributed enviroment"""
        return self._distributed

    @property
    def cuda(self):
        """If cuda is available"""
        return self._cuda

    @property
    def group_names(self):
        """
        Returns a list of all named process groups created
        """
        return self._groups.keys()

    def group(self, name=None):
        """
        Returns a process group with the given name
        If name is None, group is also None indicating the default process group
        If named group does not exist, returns None also
        """
        if name in self._groups.keys():
            return self._groups[name]
        else:
            return None

    def group_size(self, name=None):
        """
        Returns the size of named process group
        """
        if name is None:
            return self._world_size
        group = self.group(name)
        return dist.get_world_size(group=group)

    def group_rank(self, name=None):
        """
        Returns the rank in named process group
        """
        if name is None:
            return self._rank
        group = self.group(name)
        if group is None:
            return 0
        else:
            return dist.get_rank(group=group)

    def group_name(self, group=None):
        """
        Returns the name of process group
        """
        if group is None:
            return None
        return self._group_names[group]

    @property
    def broadcast_buffers(self):
        """broadcast_buffers in PyTorch DDP"""
        return self._broadcast_buffers

    @broadcast_buffers.setter
    def broadcast_buffers(self, broadcast: bool):
        """Setter for broadcast_buffers"""
        self._broadcast_buffers = broadcast

    @property
    def find_unused_parameters(self):
        """find_unused_parameters in PyTorch DDP"""
        return self._find_unused_parameters

    @find_unused_parameters.setter
    def find_unused_parameters(self, find_params: bool):
        """Setter for find_unused_parameters"""
        if find_params:
            warn(
                "Setting `find_unused_parameters` in DDP to true, "
                "use only if necessary."
            )
        self._find_unused_parameters = find_params

    def __str__(self):
        output = (
            f"Initialized process {self.rank} of {self.world_size} using "
            f"method '{self._initialization_method}'. Device set to {str(self.device)}"
        )
        return output

    @classmethod
    def is_initialized(cls) -> bool:
        """If manager singleton has been initialized"""
        return len(cls._shared_state) > 0

    @staticmethod
    def get_available_backend():
        """Get communication backend"""
        if torch.cuda.is_available() and torch.distributed.is_nccl_available():
            return "nccl"
        else:
            return "gloo"

    @staticmethod
    def initialize_env():
        """Setup method using generic initialization"""
        rank = int(os.environ.get("RANK"))
        world_size = int(os.environ.get("WORLD_SIZE"))
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ.get("LOCAL_RANK"))
        else:
            local_rank = rank % torch.cuda.device_count()
        # Read env variables
        addr = os.environ.get("MASTER_ADDR")
        port = os.environ.get("MASTER_PORT")

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
        )

    @staticmethod
    def initialize_open_mpi(addr, port):
        """Setup method using OpenMPI initialization"""
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK"))
        world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE"))
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
            method="openmpi",
        )

    @staticmethod
    def initialize_slurm(port):
        """Setup method using SLURM initialization"""
        rank = int(os.environ.get("SLURM_PROCID"))
        world_size = int(os.environ.get("SLURM_NPROCS"))
        local_rank = int(os.environ.get("SLURM_LOCALID"))
        addr = os.environ.get("SLURM_LAUNCH_NODE_IPADDR")

        DistributedManager.setup(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            addr=addr,
            port=port,
            backend=DistributedManager.get_available_backend(),
            method="slurm",
        )

    @staticmethod
    def initialize():
        """
        Initialize distributed manager

        Current supported initialization methods are:
            `ENV`: PyTorch environment variable initialization
                 https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
            `SLURM`: Initialization on SLURM systems.
                   Uses `SLURM_PROCID`, `SLURM_NPROCS`, `SLURM_LOCALID` and
                   `SLURM_LAUNCH_NODE_IPADDR` environment variables.
            `OPENMPI`: Initialization for OpenMPI launchers.
                     Uses `OMPI_COMM_WORLD_RANK`, `OMPI_COMM_WORLD_SIZE` and
                     `OMPI_COMM_WORLD_LOCAL_RANK` environment variables.

        Initialization by default is done using the first valid method in the order
        listed above. Initialization method can also be explicitly controlled using the
        `MODULUS_DISTRIBUTED_INITIALIZATION_METHOD` environment variable and setting it
        to one of the options above.
        """
        if DistributedManager.is_initialized():
            warn("Distributed manager is already intialized")
            return

        addr = os.getenv("MASTER_ADDR", "localhost")
        port = os.getenv("MASTER_PORT", "12355")
        # https://pytorch.org/docs/master/notes/cuda.html#id5
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        initialization_method = os.getenv("MODULUS_DISTRIBUTED_INITIALIZATION_METHOD")
        if initialization_method is None:
            try:
                DistributedManager.initialize_env()
            except:
                if "SLURM_PROCID" in os.environ:
                    DistributedManager.initialize_slurm(port)
                elif "OMPI_COMM_WORLD_RANK" in os.environ:
                    DistributedManager.initialize_open_mpi(addr, port)
        elif initialization_method == "ENV":
            DistributedManager.initialize_env()
        elif initialization_method == "SLURM":
            DistributedManager.initialize_slurm(port)
        elif initialization_method == "OPENMPI":
            DistributedManager.initialize_open_mpi(addr, port)
        else:
            raise RuntimeError(
                "Unknown initialization method "
                f"{initialization_method}. "
                "Supported values for "
                "MODULUS_DISTRIBUTED_INITIALIZATION_METHOD are "
                "ENV, SLURM and OPENMPI"
            )

        # Set per rank numpy random seed for data sampling
        np.random.seed(seed=DistributedManager().rank)

    @staticmethod
    def setup(
        rank=0,
        world_size=1,
        local_rank=None,
        addr="localhost",
        port="12355",
        backend="nccl",
        method="env",
    ):
        """Set up PyTorch distributed process group and update manager attributes"""
        os.environ["MASTER_ADDR"] = addr
        os.environ["MASTER_PORT"] = str(port)

        manager = DistributedManager()

        manager._distributed = (world_size > 1) and torch.distributed.is_available()
        if manager._distributed:
            # Update rank and world_size if using distributed
            manager._rank = rank
            manager._world_size = world_size
            if local_rank is None:
                manager._local_rank = rank % torch.cuda.device_count()
            else:
                manager._local_rank = local_rank

            # Setup distributed process group
            # time.sleep(1)
            dist.init_process_group(
                backend, rank=manager.rank, world_size=manager.world_size
            )

        manager._device = torch.device(
            f"cuda:{manager.local_rank}" if torch.cuda.is_available() else "cpu"
        )
        # Needed for cuda graphs
        if torch.cuda.is_available():
            torch.cuda.set_device(manager.local_rank)

        manager._initialization_method = method

        # Set device for this process and empty cache to optimize memory usage
        torch.cuda.device(manager.device)
        torch.cuda.empty_cache()

    @staticmethod
    def create_process_subgroup(
        name: str, size: int, group_name: Optional[str] = None, verbose: bool = False
    ):
        """
        Create a process subgroup of a parent process group. This must be a collective
        call by all processes participating in this application.

        Parameters
        ----------
        name : str
        Name of the process subgroup to be created.

        size : int
        Size of the process subgroup to be created. This must be an integer factor of
        the parent group's size.

        group_name : Optional[str]
        Name of the parent process group, optional. If None, the default process group
        will be used. Default None.

        verbose : bool
        Print out ranks of each created process group, default False.

        """
        manager = DistributedManager()
        if not manager.distributed:
            return None

        assert name not in manager._groups, f"Group with name {name} already exists"

        # Get parent group's params
        group = manager._group[group_name] if group_name else None
        group_size = dist.get_world_size(group=group)
        num_groups = manager.world_size // group_size

        # Get number of sub-groups per parent group
        assert (
            group_size % size == 0
        ), f"Cannot divide group size {group_size} evenly into subgroups of size {size}"
        num_subgroups = group_size // size

        # Create all the sub-groups
        # Note: all ranks in the job need to create all sub-groups in
        # the same order even if a rank is not part of a sub-group
        manager._group_ranks[name] = []
        for g in range(num_groups):
            for i in range(num_subgroups):
                # Get global ranks that are part of this sub-group
                start = i * size
                end = start + size
                if group_name:
                    ranks = manager._group_ranks[group_name][g][start:end]
                else:
                    ranks = list(range(start, end))
                # Create sub-group and keep track of ranks
                tmp_group = dist.new_group(ranks=ranks)
                manager._group_ranks[name].append(ranks)
                if manager.rank in ranks:
                    # Set group in manager only if this rank is part of the group
                    manager._groups[name] = tmp_group
                    manager._group_names[tmp_group] = name

        if verbose and manager.rank == 0:
            print(f"Process group '{name}':")
            for grp in manager._group_ranks[name]:
                print("    ", grp)

    @staticmethod
    def create_orthogonal_process_group(
        name: str, group_name: str, verbose: bool = False
    ):
        """
        Create a process group that is orthogonal to the specified process group.

        Parameters
        ----------
        name : str
        Name of the process group to be created.

        group_name : str
        Name of the existing process group.

        verbose : bool
        Print out ranks of each created process group, default False.

        """
        manager = DistributedManager()
        if not manager.distributed:
            return None

        assert (
            group_name in manager._groups
        ), f"Group with name {group_name} does not exist"
        assert name not in manager._groups, f"Group with name {name} already exists"

        group_ranks = manager._group_ranks[group_name]
        orthogonal_ranks = [list(i) for i in zip(*group_ranks)]

        for ranks in orthogonal_ranks:
            tmp_group = dist.new_group(ranks=ranks)
            if manager.rank in ranks:
                # Set group in manager only if this rank is part of the group
                manager._groups[name] = tmp_group
                manager._group_names[tmp_group] = name

        manager._group_ranks[name] = orthogonal_ranks

        if verbose and manager.rank == 0:
            print(f"Process group '{name}':")
            for grp in manager._group_ranks[name]:
                print("    ", grp)

    @staticmethod
    def cleanup():
        """Clean up distributed group and singleton"""
        dist.destroy_process_group()
        DistributedManager._shared_state = {}
