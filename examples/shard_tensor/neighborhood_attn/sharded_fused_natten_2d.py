import torch

from types import SimpleNamespace

import torch.distributed as dist
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.placement_types import Shard

from natten.functional import na2d_qk, na2d_av, na2d

from einops import rearrange

from modulus.distributed import DistributedManager, ShardTensor
from modulus.distributed.shard_utils.halo2 import HaloPaddingND

import time


def create_data(args, batch=1, dtype=torch.float32, device=torch.device(f"cuda")):
    
    shape = [batch, args.nheads, args.height, args.width, args.head_dim]
    
    input_data = torch.rand(shape, dtype=dtype, device=device)
    
    return input_data


def run_unfused_single_device(q, k, v, kernel_size, dilation=1):
    
    attn_scale = 1.0
    
    # Self attn: attn = q @ k.transpose(-2, -1)
    attn = na2d_qk(q, k, kernel_size=kernel_size, dilation=dilation)

    attn = (attn * attn_scale).softmax(dim=-1)

    # Self attn: output = attn @ v
    output = na2d_av(attn, v, kernel_size=kernel_size, dilation=dilation)

    return output

def run_fused_single_device(q, k, v, kernel_size, dilation=1):

    return na2d(q, k, v, kernel_size=kernel_size, dilation=dilation)


def run_fused_sharded(q,k,v, kernel_size, dilation=1):
    
    halo = [3,3]
    edge_padding_t = "none"
    edge_padding_s = [0,0]
    
    # TODO - check that q k v are sharded the same way and on the same mesh
    
    # Use the halo layer to compute halos:
    local_padded_q = HaloPaddingND.apply(
        q,
        halo,
        edge_padding_t,
        edge_padding_s,
    ) 
    
    local_padded_k = HaloPaddingND.apply(
        k,
        halo,
        edge_padding_t,
        edge_padding_s,
    ) 

    local_padded_v = HaloPaddingND.apply(
        v,
        halo,
        edge_padding_t,
        edge_padding_s,
    ) 

    
    # perform the local operation:
    output = run_fused_single_device(local_padded_q, local_padded_k, local_padded_v, kernel_size, dilation)
    
    # Slice off the unneed pieces.  Need to clean this up to take the same args as halo padding!
    # Meaning: the below code is a hack to get a prototype off the ground :)
    
    mesh = q._spec.mesh
    mesh_dims = range(mesh.ndim)
    tensor_dims = [ p.dim for p in q.placements]
    
    for mesh_dim, tensor_dim in zip(mesh_dims, tensor_dims): 
        #     # The local group can come right from the mesh, which the tensor knows already.
        # Here's its implicit mesh is 1D so we don't pass a mesh_dim
        local_group = mesh.get_group(mesh_dim)
        local_rank  = mesh.get_local_rank(mesh_dim)
        local_size  = dist.get_world_size(group=local_group)
        
        dim_shape = output.shape[tensor_dim]
        
        start = halo[mesh_dim]
        end = dim_shape - halo[mesh_dim]
        
        if local_rank == 0:
            start = 0
        if local_rank == local_size -1:
            end = dim_shape
        
        indices = torch.arange(start, end).to(output.device)
        
        output = output.index_select(tensor_dim, indices)
    
    # Make sure the output is contiguous
    output = output.contiguous()
    
    
    # Rebuild the shard tensor from the output:
    out_shard = ShardTensor.from_local(
        output,
        mesh,
        q.placements,
    )
    
    return out_shard


    
def benchmark(args):    
    # print(f"Benchmarking with the following parameters: {args}")
    
    dm = DistributedManager()
    
    device = torch.device(f"cuda:{dm.rank}")
    
    q = create_data(args,device=device)
    k = create_data(args,device=device)
    v = create_data(args,device=device)

    # Using dimensions 1 for length and 2 for height

    q = rearrange(q, 'b h l w hd -> b l w h hd')
    k = rearrange(k, 'b h l w hd -> b l w h hd')
    v = rearrange(v, 'b h l w hd -> b l w h hd')
    
    fused_output = run_fused_single_device(q, k, v, args.window_size)
    
    
    # Now, shard the original inputs and create a halo'd layer:


    
    # Access the rank easily through the manager:
    mesh = dm.global_mesh
    size = dm.world_size
    rank = dm.rank

    # In this example, though it's just a naming convention, we'll
    # Expect the first axis of all tensors to be A and the second to be
    # B.  

    # Here, let's extract our rank along X and Y dimensions:
    h_mesh = mesh['H']
    w_mesh = mesh['W']
    
    
    h_rank = dist.get_group_rank(h_mesh.get_group(), rank)
    w_rank = dist.get_group_rank(w_mesh.get_group(), rank)
    
    # Create the mesh and placements: 
    domain_mesh = mesh["H", "W"]
    placements  = (Shard(1), Shard(2),)
    
    # TODO - this needs to be simpler, but for now chunking and broadcasting with DTensor:
    shard_q = ShardTensor._from_dtensor(
        distribute_tensor(
            q,
            device_mesh = domain_mesh,
            placements  = placements
        )
    )
    
    shard_k = ShardTensor._from_dtensor(
        distribute_tensor(
            k,
            device_mesh = domain_mesh,
            placements  = placements
        )
    )

    shard_v = ShardTensor._from_dtensor(
        distribute_tensor(
            v,
            device_mesh = domain_mesh,
            placements  = placements
        )
    )
    
    # Apply the sharded halo computation:
    sharded_output = run_fused_sharded(shard_q, shard_k, shard_v, kernel_size=args.window_size, dilation=1)
    
    full_output = sharded_output.full_tensor()
    
    
    
    # if dm.rank == 0:
        
    #     print(f"Output shapes agree? {full_output.shape == fused_output.shape}")
    
    #     print(f"Results agree? {torch.allclose(full_output, fused_output)}")
    
    # # collect timings:
    
    times =  []
    for i in range(25):
        start = time.perf_counter()
        sharded_output = run_fused_sharded(shard_q, shard_k, shard_v, kernel_size=args.window_size, dilation=1)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end-start)
        
    times = torch.tensor(times[5:])
        
    # if dm.rank == 0:
    #     print(times)
    #     print(f"Best time: {times.min()}")
    #     print(f"Mean time: {times.mean()}")
    #     print(f"Std time: {times.std()}")
    
    return times.min(), times.mean(), times.std()



if __name__ == "__main__":
    
    # Set up the local ranks:
    DistributedManager.initialize()
    dm = DistributedManager()
    dm.initialize_mesh((1, 2, -1), ("world", "H", "W"))
    
    if dm.rank == 0:
        print(f"height,width,heads,head_dim,window_size,stride,best,mean,std")
    window_size = 7
    stride  = 1
    # Run over a suite of parameters:
    for h_shape, w_shape in [(512, 512), (512,1024), (1024,1536), (1536, 1024)]:
        for heads, head_dim in [(4, 64), (16, 48)]:
            args = SimpleNamespace(
                width = w_shape,
                height = h_shape,
                nheads = heads,
                head_dim = head_dim,
                nchannels=heads*head_dim,
                window_size=7,
                stride = 1
            )
            best, mean, std = benchmark(args)
            if dm.rank == 0:
                print(f"{h_shape},{w_shape},{heads},{head_dim},{window_size},{stride},{best:.4f},{mean:.4f},{std:.4f}")
            
    dm.cleanup()