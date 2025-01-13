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
    dm.initialize_mesh((-1,), ("domain",))
    
    
    mesh_dim_names = ["domain"]
    
    
    # Access the rank easily through the manager:
    rank = dm.rank
    
    
    


    # Make the chunks uneven and with different but predictable values:
    local_chunk = torch.arange(2*(2 + rank), dtype=torch.float32)  + 0.1*rank
    # To make this example not complelely trivial, we have 2D tensors split along one axis
    local_chunk = local_chunk.reshape((2,-1)).cuda().contiguous()
    local_chunk = local_chunk.to(f"cuda:{rank}")
    
    local_chunk.requires_grad_(True)
    
    # Create the mesh, per usual:
    domain_mesh = dm.global_mesh["domain"]

    
    # First, slice the local input based on domain mesh rank:
    domain_group = domain_mesh.get_group()
    domain_rank = dist.get_group_rank(domain_group, rank)
    domain_size = len(dist.get_process_group_ranks(domain_group))

    
    # Now, we can create the shard_tensor properly:
    shard_tensor = ShardTensor.from_local(
        local_chunk, 
        device_mesh = domain_mesh, 
        placements = (dist_tensor.Shard(1),),
    )
    print(f"Rank {rank} has {shard_tensor}\n")
    
    # # This doesn't work 
    # # Compute the "loss" locally and back prop:
    local_sum = shard_tensor.sum()
    print(f"Local sum is: {local_sum} of type {type(local_sum)}")
    # local_sum.backward()
    # print("Input gradients: {local_chunk.grads}")
    
    # # print(dist_tensor.Shard._local_shard_size_on_dim(26, 4, 0))
    
    # This DOES work: it forces a reduction over the whole tensor:
    full_tensor = shard_tensor.full_tensor()

    print(f"Full_tensor: {full_tensor}")

    summed = full_tensor.sum()
    print(f"Full sum: {summed}")    
    summed.backward()
    print(f"Input gradients: {local_chunk.grad}")
    
    # print(summed)

    # print(f"full_tensor: {full_tensor}")
    # copy_of_full_tensor = torch.tensor(full_tensor.clone(), requires_grad=True)
    # copy_of_full_tensor.requires_grad_(True)
    # loss_single_device = copy_of_full_tensor.sum()
    # loss_single_device.backward()
    # print(f"Copy grads: {copy_of_full_tensor.grad}")