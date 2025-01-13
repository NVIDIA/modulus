import torch

import torch
from time import perf_counter
import argparse


from typing import Tuple, Optional, List

from torch import nn
from torch import Tensor

torch.manual_seed(1234)

import torch.distributed as dist
from torch.distributed import tensor as dist_tensor

from modulus.distributed import DistributedManager



if __name__ == "__main__":
    
    # Set up the local ranks:
    DistributedManager.initialize()
    dm = DistributedManager()
    
    # Use the distributed manager to generate a mesh
    # Passing -1 for an argument will infer the size based on the world_size
    # (similar to reshaping a tensor with one axis as -1)
    dm.initialize_mesh((-1, 2, 2), ("world", "X", "Y"))
    
    # Above, we're assuming that whatever you're doing with distributed tensors,
    # There is at least one access of large scale out ("replication") wiht DDP 
    # or similar.  We fill the remaining axes in with our domain sharding.
    
    # Access the rank easily through the manager:
    mesh = dm.global_mesh
    
    
    size = dm.world_size
    rank = dm.rank

    # Here, let's extract our rank along X and Y dimensions:
    x_mesh = mesh['X']
    y_mesh = mesh['Y']
    
    print(f"X_mesh: {x_mesh}\n")
    dist.barrier()
    
    print(f"Y_mesh: {y_mesh}\n")
    dist.barrier()
    
    x_rank = dist.get_group_rank(x_mesh.get_group(), rank)
    y_rank = dist.get_group_rank(y_mesh.get_group(), rank)
    
    print(f"Hello from rank {rank} which is {x_rank}.{y_rank} on the X.Y mesh!\n")
    
    
    # Make the chunks different but with predictable values:
    # (You'll be able to identify the mesh rank with the two decimal places in a printout!)
    local_chunk = torch.arange(4, dtype=torch.float32)  + 0.1*x_rank + 0.01*y_rank
    local_chunk = local_chunk.reshape(2,2)
    local_chunk = local_chunk.to(f"cuda:{rank}")
    local_chunk.requires_grad_(True)
    
    print(f"Rank {rank} local_tensor: {local_chunk}")
    dist.barrier()
    
    # This time, create the mesh with TWO dimensions:
    domain_mesh = mesh["X","Y"]

    

    dtensor = dist_tensor.DTensor.from_local(
        local_chunk, 
        device_mesh = domain_mesh, 
        placements = (dist_tensor.Shard(0),dist_tensor.Shard(1)),
        run_check=True
    )
    # Above, we shard the placements along BOTH axes
    
    print(f"Rank {rank} DTensor: {dtensor} has shape {dtensor.shape}")
    dist.barrier()
    
    # This will compute the sum on each local chunk
    # The output placement will be `Partial(sum)`
    # Representing a tensor that is awaiting a reduction over the sum.
    local_sum = dtensor.sum()
    print(f"Local sum is: {local_sum}")
    
    # This line right here triggers a reduction of the sum!
    full_sum = local_sum.full_tensor()
    
    full_sum.backward()
    print(f"Input grads: {local_chunk.grad}")
    
    # # Does this?
    # try:
    #     local_max = torch.argmax(dtensor)
    #     printf("local max: {local_max}")
    # except:
    #     # No!  It's not implemented!
    print(f"Full sum: {full_sum} (type: {type(full_sum)})")
    
    
    global_tensor = dtensor.full_tensor()
    global_sum = global_tensor.sum()
    print(f"Global sum is: {global_sum}")
    
    print(f"Rank {rank} full tensor: {global_tensor} has shape {dtensor.shape}")
    
    dm.cleanup()
    
    
    # As expected, this produced a 4x4 matrix where each 2x2 corner was the original data.