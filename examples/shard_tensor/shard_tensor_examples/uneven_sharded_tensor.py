import torch


torch.manual_seed(1234)

import torch.distributed as dist
from torch.distributed import tensor as dist_tensor
from torch.distributed.tensor import debug

from modulus.distributed import ShardTensor

from torch.distributed.tensor.placement_types import (
    Placement,
    Replicate,
    Shard
)

if __name__ == "__main__":
    
    mesh_shape = [4,]
    mesh_dim_names = ["domain"]
    
    mesh = dist.device_mesh.init_device_mesh(
            "cuda",
            mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )
    
    # Access the rank easily through the manager:
    rank = dist.get_rank()
    
    
    


    # Make the chunks uneven and with different but predictable values:
    local_chunk = torch.arange(2*(2 + rank), dtype=torch.float32)  + 0.1*rank
    # To make this example not complelely trivial, we have 2D tensors split along one axis
    local_chunk = local_chunk.reshape((2,-1)).cuda().contiguous()
    local_chunk = local_chunk.to(f"cuda:{rank}")
    
    # Create the mesh, per usual:
    domain_mesh = mesh["domain"]

    
    # First, slice the local input based on domain mesh rank:
    domain_group = domain_mesh.get_group()
    domain_rank = dist.get_group_rank(domain_group, rank)
    domain_size = len(dist.get_process_group_ranks(domain_group))


    shape  = (2, 14)
    stride = (14, 1)
    
    
    # Now, we can create the shard_tensor properly:
    shard_tensor = ShardTensor.from_local(
        local_chunk, 
        device_mesh = domain_mesh, 
        placements = (dist_tensor.Shard(1),),
    )
    
    print(dist_tensor.Shard._local_shard_size_on_dim(26, 4, 0))
    
    print(f"Rank {rank} has {shard_tensor}\n")

    
    full = shard_tensor.full_tensor()
    
    # Now, sum the whole tensor