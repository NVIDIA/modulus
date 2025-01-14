import torch
from types import SimpleNamespace
import time

from stormcast_attn import Attention

import torch.distributed as dist
from modulus.distributed import DistributedManager, ShardTensor
from modulus.distributed._shard_tensor_spec import _stride_from_contiguous_shape_C_style



from torch.distributed.tensor import distribute_tensor, distribute_module
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor.placement_types import Shard, Replicate


from torch.distributed.tensor._ops.utils import register_prop_rule
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding, OpStrategy

from torch.distributed.device_mesh import _mesh_resources, DeviceMesh

# from stormcast_attn import unbind_rules
from einops import rearrange


from torch.distributed.tensor._ops.utils import register_prop_rule, register_op_strategy
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding, OpStrategy, RuntimeSchemaInfo

from natten.functional import na2d

aten = torch.ops.aten


@register_prop_rule(aten.unbind.int, schema_info = RuntimeSchemaInfo(1))
def unbind_rules(op_schema : OpSchema) -> OutputSharding:
    """
    Need to add rules for unbinding for stormcast and attention in general
    """
    
    # We need to get the dimension of the slice.  0 is default.
    
    args_schema = op_schema.args_schema

    if len(args_schema) > 1:
        dim = args_schema[-1]
    else:
        dim = 0
        
    # if the chunking dimension is along a dimension that is sharded, we have to handle that.
    # If it's along an unsharded dimension, there is nearly nothing to do.  
   
    # Outpu
   
    input_spec = args_schema[0]
    
    input_placements = input_spec.placements

    if dim in [ i.dim for i in input_placements]:
        raise Exception("No implementation for unbinding along sharding axis yet.")
    
    else:
        # We are reducing tensor rank and returning one sharding per tensor:
        original_shape = list(input_spec.shape)
        unbind_dim_shape = original_shape.pop(dim)

        output_stride = _stride_from_contiguous_shape_C_style(original_shape)

        # Need to create a new global meta:
        new_meta = TensorMeta(
            torch.Size(tuple(original_shape)),
            stride = output_stride,
            dtype = input_spec.tensor_meta.dtype,
        )

        # The placements get adjusted too
        new_placements = []
        for p in input_spec.placements:
            if isinstance(p, Replicate):
                new_placements.append(p)
            elif isinstance(p, Shard):
                if p.dim > dim:
                    new_placements.append(Shard(p.dim - 1))
                else:
                    new_placements.append(p)
            elif isinstance(p, Partial):
                raise Exception("Partial placement not supported yet for unbind")
            
        output_spec_list = [
            DTensorSpec(
                mesh        = input_spec.mesh,
                placements  = tuple(new_placements),
                tensor_meta = new_meta,
            ) for _ in range(unbind_dim_shape)
        ]
        return OutputSharding(output_spec_list) 
        

from torch.distributed.tensor.experimental import register_sharding

@register_sharding(na2d)
def na2d_sharded_strategy(mesh : DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    raise Exception("na2d op strategy failing")



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
                
                
def test_inputs(unsharded_x, sharded_x):
    
    # First test, making sure inputs agree on rank 0:
    
    coalesced_x = sharded_x.full_tensor()
    
    dm = DistributedManager()
    if dm.rank == 0:
        assert torch.allclose(unsharded_x, coalesced_x)

    return True

def test_linear(unsharded_x, sharded_x):
    
    nc = unsharded_x.shape[-1]
    
    qkv_op = torch.nn.Linear(nc, 3*nc).to(unsharded_x.device)
    
    unsharded_out = qkv_op(unsharded_x)
    
    mesh = sharded_x._spec.mesh
    
    # Cast the linear to replicated:
    qkv_op = distribute_module(
        qkv_op, mesh
    )
    
    sharded_out = qkv_op(sharded_x)
    
    full_out = sharded_out.full_tensor()
    
    dm = DistributedManager()
    if dm.rank == 0:
        assert torch.allclose(unsharded_out, full_out)
    
    return unsharded_out, sharded_out

def test_unbind(unsharded_out, sharded_out, dim):
    
    num_heads = 8
    head_dim  = dim // num_heads
    
    B = unsharded_out.shape[0]
    N = unsharded_out.shape[1]
    unsharded_out = unsharded_out.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)

    # post shape is [3, B, num_heads, N, head_dim]
    q, k, v = unsharded_out.unbind(0)
    
    
    B = sharded_out.shape[0]
    N = sharded_out.shape[1]
    sharded_out = sharded_out.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)

    
    sq, sk, sv = sharded_out.unbind(0)
    
    dm = DistributedManager()
    
    sqf = sq.full_tensor()
    skf = sk.full_tensor()
    svf = sv.full_tensor()
    
    
    if dm.rank == 0:
        assert torch.allclose(q, sqf)
        assert torch.allclose(k, skf)
        assert torch.allclose(v, svf)

    return (q, k, v,), (sq, sk, sv)

def test_rearrange(q, shard_q, h):
    
    
    q = rearrange(q, "b head (h w) c -> b h w head c", h=h)
    
    shard_q = rearrange(shard_q, "b head (h w) c -> b h w head c", h=h)
    
    sqf = shard_q.full_tensor()

    dm = DistributedManager()
    
    
    if dm.rank == 0:
        assert torch.allclose(q, sqf)

    return q, shard_q
    
    
def test_na2d(single_device_inputs, sharded_inputs, kernel_size=7):
    
    (k, q, v) = single_device_inputs
    (sk, sq, sv) = sharded_inputs
    
    x = na2d(q, k, v, kernel_size=kernel_size)
    
    sx = na2d(sq, sk, sv, kernel_size = kernel_size)
    
    sxf = sx.full_tensor()
    
    dm = DistributedManager()
    
    print(f"Rank {dm.rank} difference: {torch.sum(torch.abs(x - sxf))}")
    
    if dm.rank == 0:
        assert(torch.allclose(x, sxf))
    
    return x, sx
    
def benchmark(args):
    
    # Pretend inputs: 
    x = create_data(args)
    
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
    
    
    test_inputs(x, shard_x)
    unsharded_out, sharded_out = test_linear(x, shard_x)
    
    unsharded_qkv, sharded_qkv = test_unbind(unsharded_out, sharded_out, dim=args.nheads * args.head_dim)
    
    q, k, v = unsharded_qkv
    sq, sk, sv = sharded_qkv
    
    q, sq = test_rearrange(q, sq, args.height)
    k, sk = test_rearrange(k, sk, args.height)
    v, sv = test_rearrange(v, sv, args.height)
    
    print(f"Rank {dm.rank} passed rearrange")
    
    
    x, sq = test_na2d((k, q, v), (sk, sq, sv), kernel_size=7)
    
    
    dist.barrier()
    
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


    return None
    # attn = distribute_module(
    #     attn, domain_mesh
    # )
    
    # for name, param in attn.named_parameters():
    #     print(f"{name}: {type(param.data)}")

    # test_output = attn(shard_x, latent_hw = [args.height // 2, args.width ])
    # return None

    

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