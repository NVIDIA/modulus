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
import time
import logging
from modulus.experimental.sfno.utils.logging_utils import disable_logging
import math
import torch
import torch.distributed as dist
import datetime as dt
from typing import Union
import numpy as np

# dummy placeholders
_COMM_LIST = []
_COMM_NAMES = {}
_COMM_ROOTS = {}

# world comm
def get_size(comm_id: Union[str, int]) -> int:
    if isinstance(comm_id, int):
        cid = comm_id
    else:
        cid = _COMM_NAMES[comm_id] if (comm_id in _COMM_NAMES) else len(_COMM_LIST)
    
    if not dist.is_initialized() or (cid >= len(_COMM_LIST)):
        return 1
    else:
        return dist.get_world_size(group=_COMM_LIST[cid])

    
def get_rank(comm_id: Union[str, int]) -> int:
    if isinstance(comm_id, int):
        cid = comm_id
    else:
        cid = _COMM_NAMES[comm_id] if (comm_id in _COMM_NAMES) else len(_COMM_LIST)
        
    if not dist.is_initialized() or (cid >= len(_COMM_LIST)):
        return 0
    else:
        return dist.get_rank(group=_COMM_LIST[cid])

    
def get_group(comm_id: Union[str, int]) -> int:
    if isinstance(comm_id, int):
        cid = comm_id
    else:
        cid = _COMM_NAMES[comm_id] if (comm_id in _COMM_NAMES) else len(_COMM_LIST)

    if not dist.is_initialized() or (cid >= len(_COMM_LIST)):
        raise IndexError(f"Error, comm with id {comm_id} not available.")
    else:
        return _COMM_LIST[cid]

def get_root(comm_id: Union[str, int]) -> int:
    if isinstance(comm_id, str):
        cname = comm_id
        cid = _COMM_NAMES[comm_id] if (comm_id in _COMM_NAMES) else len(_COMM_LIST)
    else:
        cname = _COMM_LIST[comm_id] if comm_id < len(_COMM_LIST) else ""
        cid = comm_id

    if not dist.is_initialized() or (cid >= len(_COMM_LIST)):
        return 0
    else:
        return _COMM_ROOTS[cname]

# specialized routines for world comms
def get_world_size():
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()

    
def get_world_rank():
    if not dist.is_initialized():
        return 0
    else:
        return dist.get_rank()


def get_local_rank():
    if os.getenv('LOCAL_RANK') is not None:
        # Use PyTorch env var if available
        return int(os.getenv('LOCAL_RANK'))

    if not dist.is_initialized():
        return 0
    else:
        num_gpu = int(os.getenv("NGPU_PER_NODE", torch.cuda.device_count()))
        return get_world_rank() % num_gpu
    
def get_names():
    return _COMM_NAMES

def is_distributed(name: str):
    return (name in _COMM_NAMES)
    
    
# get 
def init(params, verbose = False):
    
    init_process_group(info=params.wireup_info, store=params.wireup_store)

    # do individual wireup for model parallel comms:
    model_parallel_sizes = params.get("model_parallel_sizes", [1])
    model_parallel_names = params.get("model_parallel_names", ["model"])

    params.model_parallel_size = init_model_parallel_info(
        names=model_parallel_names, 
        sizes=model_parallel_sizes,
        log_to_screen=params.log_to_screen,
        verbose=verbose
    )


def init_process_group(info: str, store: str):
    """Initial torch distributed process group based on ``info`` and ``store``

    Uses NCCL
    
    Args:
        info: either ``env`` or ``mpi``
        store: either ``file`` or ``tcp``
    
    """
    # set up global and local communicator
    if info == "env":
        world_size = int(os.getenv('WORLD_SIZE', 1))
        world_rank = int(os.getenv('RANK', 0))
        if os.getenv('WORLD_RANK') is not None:
            # Use WORLD_RANK if available for backwards compatibility
            world_rank = int(os.getenv('WORLD_RANK'))
        port = int(os.getenv('MASTER_PORT', 0))
        master_address = os.getenv('MASTER_ADDR')
        if os.getenv('MASTER_ADDRESS') is not None:
            # Use MASTER_ADDRESS if available for backwards compatibility
            master_address = int(os.getenv('MASTER_ADDRESS'))
    elif info == "mpi":
        import socket
        from mpi4py import MPI
        mpi_comm = MPI.COMM_WORLD.Dup()
        world_size = mpi_comm.Get_size()
        world_rank = mpi_comm.Get_rank()
        my_host = socket.gethostname()
        port = 29500
        master_address = None
        if world_rank == 0:
            master_address_info = socket.getaddrinfo(my_host, port, family=socket.AF_INET, proto=socket.IPPROTO_TCP)
            master_address = master_address_info[0][-1][0]
        master_address = mpi_comm.bcast(master_address, root=0)
        os.environ["MASTER_ADDRESS"] = master_address
        os.environ["MASTER_PORT"] = str(port)
    else:
        raise ValueError(f"Error, wireup-info {info} not supported")
    # set local rank to 0 if env var not available
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    
    if world_size > 1:
        with disable_logging():
            if store == "file":
                wireup_file_path = os.getenv('WIREUP_FILE_PATH')
                store = dist.FileStore(wireup_file_path, world_size)
            elif store == "tcp":
                # create tcp store
                store = dist.TCPStore(host_name = master_address,
                                             port = port,
                                             world_size = world_size,
                                             is_master = (world_rank == 0),
                                             timeout = dt.timedelta(seconds=900))
            else:
                store = None
                
            # initialize process groups
            dist.init_process_group(backend = 'nccl',
                                    rank = world_rank,
                                    world_size = world_size,
                                    store = store)


def init_model_parallel_info(names, sizes, log_to_screen=False, verbose=False):
    """Create communicators for model parallelism _COMM_LIST, _COMM_NAMES, and
    _COMM_ROOTS"""
    # get sizes
    world_size = get_world_size()
    world_rank = get_world_rank()
    local_rank = get_local_rank()

    model_parallel_names = names
    model_parallel_sizes = sizes

    assert(len(model_parallel_names) == len(model_parallel_sizes)), "Please specify names for your communicators"
    model_parallel_size = math.prod(model_parallel_sizes)

    assert ( (world_size % model_parallel_size == 0) ), \
        "Error, please make sure that the product of model parallel ranks evenly divides the total number of ranks"

    # we set this to be orthogonal to the MP groups
    # we can play tricks with the ddp_group later, in case if all the weights are shared
    data_parallel_size = world_size // model_parallel_size
    
    # create orthogonal communicators first
    global _COMM_LIST
    global _COMM_NAMES
    if log_to_screen:
        logging.info("Starting Wireup")
        
    if world_size > 1:

        # set up the strides:
        model_parallel_sizes_reversed = model_parallel_sizes[::-1]
        model_grid = np.reshape(np.arange(0, model_parallel_size), model_parallel_sizes[::-1])
        perm = np.roll(np.arange(0,len(model_parallel_sizes)), 1).tolist()
        ranks_lookup = {}

        comm_count = 0
        for mpname in model_parallel_names:
            base_group = np.reshape(model_grid, (-1, model_grid.shape[-1]))
            model_groups = []
            for goffset in range(0, world_size, model_parallel_size):
                model_groups += sorted((goffset + base_group).tolist())

            if verbose and world_rank == 0:
                print(f"Creating comm groups for id {mpname}: {model_groups}")
                
            for grp in model_groups:
                if len(grp) > 1:
                    tmp_group = dist.new_group(ranks = grp)
                    if world_rank in grp:
                        _COMM_LIST.append(tmp_group)
                        _COMM_NAMES[mpname] = comm_count
                        _COMM_ROOTS[mpname] = min(grp)
                        comm_count += 1
            ranks_lookup[mpname] = model_groups
                
            # go for the next step
            model_grid = np.transpose(model_grid, perm)

        # helper routine for creating meta comms
        def merge_comms(comm_count, ranks_lookup, comm_name_1, comm_name_2, merge_name):
            
            if ((get_size(comm_name_1) == 1) and (get_size(comm_name_2) > 1)):
                if verbose and world_rank == 0:
                    print(f'Creating comm groups for id {merge_name}: {ranks_lookup[comm_name_2]}')
                _COMM_LIST.append(get_group(comm_name_2))
                _COMM_NAMES[merge_name] = comm_count
                for grp in ranks_lookup[comm_name_2]:
                    if world_rank in grp:
                        _COMM_ROOTS[merge_name] = min(grp)
                comm_count += 1
            elif ((get_size(comm_name_1) > 1) and (get_size(comm_name_2) == 1)):
                if verbose and world_rank == 0:
                    print(f'Creating comm groups for id {merge_name}: {ranks_lookup[comm_name_1]}')
                _COMM_LIST.append(get_group(comm_name_1))
                _COMM_NAMES[merge_name] = comm_count
                for grp	in ranks_lookup[comm_name_1]:
                    if world_rank in grp:
                        _COMM_ROOTS[merge_name] = min(grp)
                comm_count += 1
            elif ((get_size(comm_name_1) > 1) and (get_size(comm_name_2) > 1)):
                # fuse the lists:
                def merge_ranks(list1, list2):
                    coll = list1 + list2
                    pooled = [set(subList) for subList in coll]
                    merging = True
                    while merging:
                        merging=False
                        for i,group in enumerate(pooled):
                            merged = next((g for g in pooled[i+1:] if g.intersection(group)),None)
                            if not merged: continue
                            group.update(merged)
                            pooled.remove(merged)
                            merging = True
                    return [list(x) for x in pooled]
    
                model_groups = merge_ranks(ranks_lookup[comm_name_1], ranks_lookup[comm_name_2])
                if verbose and world_rank == 0:
                    print(f'Creating comm groups for id {merge_name}: {model_groups}')
                for grp in model_groups:
                    tmp_group = dist.new_group(ranks = grp)
                    if world_rank in grp:
                        _COMM_LIST.append(tmp_group)
                        _COMM_NAMES[merge_name] = comm_count
                        _COMM_ROOTS[merge_name] = min(grp)
                        comm_count += 1

            return comm_count

        # merge spatial
        comm_count = merge_comms(comm_count, ranks_lookup, "h", "w", "spatial")

        # merge matmul
        comm_count = merge_comms(comm_count, ranks_lookup, "fin", "fout", "matmul")
                    
        # now the data and model comm:
        model_groups = np.reshape(np.arange(0, world_size), (-1, model_parallel_size)).tolist()
        for grp in model_groups:
            if len(grp) > 1:
                tmp_group = dist.new_group(ranks = grp)
                if world_rank in grp:
                    _COMM_LIST.append(tmp_group)
                    _COMM_NAMES["model"] = comm_count
                    _COMM_ROOTS["model"] = min(grp)
                    comm_count += 1
        
        if data_parallel_size == world_size:
            if verbose and world_rank == 0:
                print(f"Creating comm groups for id data: {[list(range(0, world_size))]}")
            
            _COMM_LIST.append(None)
            _COMM_NAMES["data"] = comm_count
            _COMM_ROOTS["data"] = 0
        else:
            data_groups = [sorted(list(i)) for i in zip(*model_groups)]

            if verbose and world_rank == 0:
                print(f"Creating comm groups for id data: {data_groups}")
            
            for grp in data_groups:
                tmp_group = dist.new_group(ranks = grp)
                if world_rank in grp:
                    _COMM_LIST.append(tmp_group)
                    _COMM_NAMES["data"] = comm_count
                    _COMM_ROOTS["data"] = min(grp)     
    
    if log_to_screen:
        logging.info("Finished Wireup")
    
    return model_parallel_size
