import torch


torch.manual_seed(1234)

import torch.distributed as dist
from torch.distributed import tensor as dist_tensor
from torch.distributed.tensor import debug

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
    dm.initialize_mesh((-1, 2, 2), ("world", "A", "B"))
    

    
    # Access the rank easily through the manager:
    mesh = dm.global_mesh
    size = dm.world_size
    rank = dm.rank

    # In this example, though it's just a naming convention, we'll
    # Expect the first axis of all tensors to be A and the second to be
    # B.  

    # Here, let's extract our rank along X and Y dimensions:
    a_mesh = mesh['A']
    b_mesh = mesh['B']
    
    print(f"A_mesh: {a_mesh}\n")
    dist.barrier()
    
    print(f"B_mesh: {b_mesh}\n")
    dist.barrier()
    
    a_rank = dist.get_group_rank(a_mesh.get_group(), rank)
    b_rank = dist.get_group_rank(b_mesh.get_group(), rank)
    
    print(f"Hello from rank {rank} which is {a_rank}.{b_rank} on the A.B mesh!\n")
    
    
    # Make the chunks different but with predictable values:
    # (You'll be able to identify the mesh rank with the two decimal places in a printout!)
    this_shape = (2 + a_rank, 2 + b_rank);
    numel = this_shape[0]*this_shape[1]
    local_chunk = torch.arange(numel, dtype=torch.float32)  + 0.1*a_rank + 0.01*b_rank
    local_chunk = local_chunk.reshape(this_shape)
    local_chunk = local_chunk.to(f"cuda:{rank}")
    local_chunk.requires_grad_(True)
    
    print(f"Rank {rank} ({a_rank},{b_rank}) local_tensor: {local_chunk} (shape: {local_chunk.shape})")
    dist.barrier()
    
    domain_mesh = mesh["A", "B"]
    
    # Now, we can create the shard_tensor properly:
    shard_tensor = ShardTensor.from_local(
        local_chunk, 
        device_mesh = domain_mesh, 
        placements = (dist_tensor.Shard(0), dist_tensor.Shard(1),),
    )
    
    
    print(f"Rank {rank} has {shard_tensor}\n")

    
    full = shard_tensor.full_tensor()
    
    if rank == 0:
        print(f"Rank {rank} has global tensor: {full} of shape {full.shape}\n")
    
    loss = full.sum()
    
    loss.backward()
    
    print(local_chunk.grad)
    
    dm.cleanup()