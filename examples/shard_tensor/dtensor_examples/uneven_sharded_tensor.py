import torch


torch.manual_seed(1234)

import torch.distributed as dist
from torch.distributed import tensor as dist_tensor
from torch.distributed.tensor import debug



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
    
    
    # Now, we can create the dtensor properly:
    dtensor = dist_tensor.DTensor.from_local(
        local_chunk, 
        device_mesh = domain_mesh, 
        placements = (dist_tensor.Shard(1),),
        run_check=True,
        shape=shape,
        stride=stride
    )
    
    print(dist_tensor.Shard._local_shard_size_on_dim(26, 4, 0))
    
    print(f"Rank {rank} has {dtensor}\n")

    debug.visualize_sharding(dtensor)
    
    full = dtensor.full_tensor()
    
    if rank == 0:
        print(f"Rank {rank} has global tensor: {full} of shape {full.shape}\n")
    
    
    # Correct implementation:
    size_list = [0,]*domain_size
    # Gather the sizes:
    dist.all_gather_object(size_list, local_chunk.shape)
    # Create buffers:
    output_tensor = [torch.empty(s, device=local_chunk.device) for s in size_list]
    # Gather up:
    dist.all_gather(output_tensor, local_chunk, group = domain_group)
    #Concat:
    output_tensor = torch.cat(output_tensor, dim=1)
    if rank == 0:
        print(f"Correct output tensor: {output_tensor}")