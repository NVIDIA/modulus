import torch

import torch
from time import perf_counter
import argparse


from typing import Tuple, Optional, List

from torch import nn
from torch import Tensor

torch.manual_seed(1234)

import torch.distributed as dist

from modulus.distributed import DistributedManager
from modulus.distributed import ShardTensor

from torch.distributed.tensor.placement_types import (
    Placement,
    Replicate,
    Shard
)

if __name__ == "__main__":
    
    # Set up the local ranks:
    DistributedManager.initialize()
    dm = DistributedManager()
    
    # Use the distributed manager to generate a mesh
    # Passing -1 for an argument will infer the size based on the world_size
    # (similar to reshaping a tensor with one axis as -1)
    dm.initialize_mesh((1, -1), ("world", "domain"))
    
    # Above, we're assuming that whatever you're doing with distributed tensors,
    # There is at least one access of large scale out ("replication") wiht DDP 
    # or similar.  We fill the remaining axes in with our domain sharding.
    
    # Access the rank easily through the manager:
    mesh = dm.global_mesh
    
    
    size = dm.world_size
    rank = dm.rank


    # Make the chunks different but with predictable values:
    local_chunk = torch.arange(50, dtype=torch.float32)  + 0.1*rank
    local_chunk = local_chunk.to(f"cuda:{rank}").contiguous()
    
    print(f"Rank {rank} local_tensor: {local_chunk}")
    dist.barrier()
    
    # Create the mesh, per usual:
    domain_mesh = mesh["domain"]

    
    # # First, slice the local input based on domain mesh rank:
    # domain_group = domain_mesh.get_group()
    # domain_rank = dist.get_group_rank(domain_group, rank)
    # domain_size = len(dist.get_process_group_ranks(domain_group))

    stensor = ShardTensor.from_local(
        local_chunk, 
        device_mesh = domain_mesh, 
        placements = (Shard(0),),
    )
    
    
    print(f"Rank {rank} DTensor: {stensor} has shape {stensor.shape}")
    dist.barrier()
    global_tensor = stensor.full_tensor()
    
    print(f"Rank {rank} full tensor: {global_tensor} has shape {stensor.shape}")
    
    dm.cleanup()