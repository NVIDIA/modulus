import torch
from types import SimpleNamespace
import time

from stormcast_attn import Attention

import torch.distributed as dist
from modulus.distributed import DistributedManager, ShardTensor
from modulus.distributed._shard_tensor_spec import _stride_from_contiguous_shape_C_style



from torch.distributed.tensor import distribute_tensor, distribute_module, TensorMeta
from torch.distributed.tensor.placement_types import Shard, Replicate


from torch.distributed.tensor._ops.utils import register_prop_rule
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding, OpStrategy


aten = torch.ops.aten

import pdb

@register_prop_rule(aten.unbind.int)
def unbind_rules(op_schema : OpSchema) -> OutputSharding:
    """
    Need to add rules for unbinding for stormcast and attention in general
    """
    
    # We need to get the dimension of the slice.  0 is default.
    
    args_schema = op_schema.args_schema
    print(f"Args Schema: {args_schema}")

    if len(args_schema) > 1:
        dim = args_schema[-1]
    else:
        dim = 0
        
    # if the chunking dimension is along a dimension that is sharded, we have to handle that.
    # If it's along an unsharded dimension, there is nearly nothing to do.  
   
    # Outpu
   
    input_spec = args_schema[0]
    print(f"Input strategy: {input_spec} of type {type(input_spec)}")    
    
    input_placements = input_spec.placements
    print(f"Input placements: {input_placements}")

    if dim in [ i.dim for i in input_placements]:
        raise Exception("No implementation for unbinding along sharding axis yet.")
    
    else:
        # We are reducing tensor rank and returning one sharding per tensor:
        original_shape = list(input_spec.shape)
        unbind_dim_shape = original_shape.pop(dim)
        print(f"Output shape: {original_shape}")
        output_stride = _stride_from_contiguous_shape_C_style(original_shape)

        # Need to create a new global meta:
        new_meta = TensorMeta(
            torch.size(tuple(original_shape)),
            stride = output_stride,
            dtype = input_spec.tensor_meta.dtype,
        )
        # new_meta = 
        new_spec = type(input_spec)(
            mesh = input_spec.mesh,
            placements = input_spec.placements,
            tensor_meta = input_spec.tensor_meta
        )
        return OutputSharding(
            [ new_spec for _ in range(unbind_dim_shape) ]
        )





def main():
    
        # Set up the local ranks:
    DistributedManager.initialize()
    dm = DistributedManager()
    dm.initialize_mesh((1, -1), ("world", "H"))
    
    window_size = 7
    stride = 1
    
    print(f"height,width,heads,head_dim,window_size,stride,n_params,best,mean,std")
    # Run over a suite of parameters:
    # for h_shape, w_shape in [(512, 512), (512,1024), (1024,1536), (1536, 1024)]:
    for h_shape, w_shape in [(512, 512), ]:
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
            best, mean, std, n_params = benchmark(args)
            print(f"{h_shape},{w_shape},{heads},{head_dim},{window_size},{stride},{n_params},{best:.4f},{mean:.4f},{std:.4f}")
                
                
def benchmark(args):
    
    # Pretend inputs: 
    x = create_data(args)
    
    # For the NAtten implementation, we have to shape this properly...
    
    attn = Attention(
        dim = args.nheads * args.head_dim,
        num_heads = args.nheads,
        attn_kernel = args.window_size,
    ).to(x.device)
    
    output = attn(x, latent_hw = [args.height, args.width])
    
    # Compute the number of parameter:
    n_params = 0
    for name, p in attn.named_parameters():
        this_params_count = 1
        for dim in p.shape:
            this_params_count *= dim
        n_params += this_params_count
    
      # Access the rank easily through the manager:
    dm = DistributedManager()
    mesh = dm.global_mesh
    size = dm.world_size
    rank = dm.rank

    # In this example, though it's just a naming convention, we'll
    # Expect the first axis of all tensors to be A and the second to be
    # B.  

    # Here, let's extract our rank along X and Y dimensions:
    h_mesh = mesh['H']
    
    
    h_rank = dist.get_group_rank(h_mesh.get_group(), rank)
    
    # Create the mesh and placements: 
    domain_mesh = mesh["H",]
    placements  = (Shard(1),)
    
    # TODO - this needs to be simpler, but for now chunking and broadcasting with DTensor:
    shard_x = ShardTensor._from_dtensor(
        distribute_tensor(
            x,
            device_mesh = domain_mesh,
            placements  = placements
        )
    )
    
    # # Parallelize the module with replication:
    # placement = [Shard(0)]
    # for name, param in attn.named_parameters():
    #     print(f"Original Parameter: {name}, Shape: {param.shape}")

    #     dt = distribute_tensor(
    #         param.data,
    #         device_mesh = domain_mesh,
    #         placements = placement,
    #     )

    #     print(f"Type dt: {type(dt)}, dt: {dt}")
    #     param.data = dt
    #     # attn.named_parameters()[name] = param
    #     print(f"Distributed Parameter: {name}, Placement: {placement}, Shape: {param.data.shape}")


        
    attn = distribute_module(
        attn, domain_mesh
    )
    
    for name, param in attn.named_parameters():
        print(f"{name}: {type(param.data)}")
        
    test_output = attn(shard_x, latent_hw = [args.height // 2, args.width ])
    return None

    

    times =  []
    for i in range(50):
        start = time.perf_counter()
        _ = attn(shard_x, latent_hw = [args.height // 2, args.width])
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end-start)
        
    times = torch.tensor(times[5:])

    return times.min(), times.mean(), times.std(), n_params


def create_data(args, batch=1, dtype=torch.float32, device=torch.device(f"cuda")):
    
    shape = [batch, args.height * args.width, args.nheads * args.head_dim]
    
    input_data = torch.rand(shape, dtype=dtype, device=device)
    
    return input_data


if __name__ == "__main__":
    main()