import wrapt

import torch

import torch.distributed as dist

from modulus.distributed import ShardTensor
from . halo import HaloPaddingND, UnSliceHaloND

class UndeterminedShardingError(Exception):
    pass

class MissingShardPatch(NotImplementedError):
    pass

def compute_halo_from_kernel_and_dilation(kernel_size, dilation):
    """
    Given a kernel and dilation for neighborhood attention, return the halo size
    """
    
    # Currently, reject even kernel_sizes and dilation != 1:
    if kernel_size % 2 == 0:
        raise MissingShardPatch("Neighborhood Attention is not implemented for even kernels")
    if dilation != 1:
        raise MissingShardPatch("Neighborhood Attention is not implemented for dilation != 1")
    
    halo = int(kernel_size // 2)
    
    return halo
        

def shard_to_haloed_local(q, k, v, kernel_size, dilation=1):

    assert q._spec.mesh == k._spec.mesh, "Mismatched mesh not supported in na2d"
    assert q._spec.mesh == v._spec.mesh, "Mismatched mesh not supported in na2d"

    # How big of a halo do we need?
    halo_size = compute_halo_from_kernel_and_dilation(kernel_size, dilation)
    
    # First, determine the mesh dimension which decides the halo dimension:
    mesh = q._spec.mesh
    
    halo = [ halo_size, ] * mesh.ndim
    edge_padding_s = [0,] * mesh.ndim
    edge_padding_t = "none"
    
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
    
    return (local_padded_q, local_padded_k, local_padded_v), halo


    

def na2d_with_halo(q,k,v, kernel_size, dilation=1):
    

    
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

# NAtten patches!
@wrapt.patch_function_wrapper('natten.functional', 'na2d')
def na2d_wrapper(wrapped, instance, args, kwargs):
    """
    Perform na2d on sharded tensors.  Expected that q, k, v
    are sharded the same way and on the same mesh.  And that
    they have the same shape, locally and globally.
    """
    
    # args[0], args[1], and args[2] should all be shard tensors to do the wrapping
    # If they are regular torch tensors, do nothing unusual!

    def fetch_qkv(q, k, v, *args, **kwargs):
        return q, k, v

    q, k, v = fetch_qkv(*args)


    # Not using isinstance here because ShardTensor inherits DTensor inherits torch.Tensor.
    
    try: 
        dilation = kwargs['dilation']
    except:
        dilation = 1
        
    kernel_size = kwargs['kernel_size']
    

    if all( [ type(_t) == torch.Tensor for _t in (q, k, v) ]):
        return wrapped(*args, **kwargs)
    elif all( [ type(_t) ==  ShardTensor for _t in (q, k, v) ]):
        # This applies a halo layer and returns local torch tensors:
        (lq, lk, lv), halo = shard_to_haloed_local(q, k, v, kernel_size, dilation)
        # This applies the native, underlying na2d:
        x = wrapped(lq, lk, lv, kernel_size, dilation)
        # This slices off any extra bits and reforms the output into a shard tensor
        # x = slice_output_by_halo_and_mesh(x, halo, q._spec.mesh, q._spec.placements)
        x = UnSliceHaloND.apply(x, halo, q._spec.mesh, q._spec.placements)
        return x
        # raise NotImplementedError("This is the section to do!")
    else:
        raise UndeterminedShardingError("q, k, and v must all be the same types (torch.Tensor or ShardTensor)")
    