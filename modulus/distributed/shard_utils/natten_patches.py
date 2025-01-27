import wrapt
from typing import Tuple

import torch

import torch.distributed as dist

from modulus.distributed import ShardTensor


from modulus.distributed.shard_utils.patch_core import (
    UndeterminedShardingError,
    MissingShardPatch,
)
from torch.distributed.tensor.placement_types import (
    Shard
)

__all__ = [
    "na2d_wrapper"
]

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


from . halo import halo_padding_1d, halo_unpadding_1d


class HaloPaddingND(torch.autograd.Function):
    """
    Autograd Wrapper for a distributed HaloPadding primitive.
    It is based on the torch DTensor concept which presents as a
    sharded tensor + device Mesh.  In the forward pass, the adjacent regions
    are gathered from next-door devices and concatenated into one output tensor
    
    In the backward pass, the gradients are distributed outward to neighboring tensors.
    
    This halo can accommodate multiple dimensions of halo passing, but requires the 
    mesh and halo parameters to be compatible.  In this case, the backwards pass
    distributes gradients in the reverse order as the forward pass.
    """

    @staticmethod
    def forward(
        ctx,
        stensor: ShardTensor,
        halo: Tuple[int],
        edge_padding_t : str,
        edge_padding_s : Tuple[int]
    ) -> torch.Tensor:  # pragma: no cover
        """forward pass of the Distributed Halo primitive"""
        mesh = stensor.device_mesh

        assert len(halo) == mesh.ndim, f"Halo size ({len(halo)} must match mesh rank ({mesh.ndim}))"

        placements = stensor.placements
        
        local_tensor = stensor.to_local()
        for mesh_dim in range(mesh.ndim):
            if isinstance(placements[mesh_dim], Shard):
                tensor_dim = placements[mesh_dim].dim
                local_tensor = halo_padding_1d(local_tensor, mesh, mesh_dim, tensor_dim, halo[mesh_dim], edge_padding_t, edge_padding_s[0])
        



        # padded_tensor = halo_padding_1d(stensor.to_local(), mesh, halo[0], edge_padding_t, edge_padding_s[0])
        ctx.halo = halo
        ctx.spec = stensor._spec
        ctx.requires_input_grad = stensor.requires_grad

        return local_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> "ShardTensor":  # pragma: no cover
        """backward pass of the of the Distributed HaloPadding primitive"""

        print(f"got grad output of shape: {grad_output.shape}")
        print(f"got grad output of type: {type(grad_output)}")

        spec = ctx.spec
        mesh = spec.mesh
        placements = spec.placements
        halo = ctx.halo
        
        
        for mesh_dim in range(mesh.ndim):
            if isinstance(placements[mesh_dim], Shard):
                tensor_dim = placements[mesh_dim].dim
                grad_output = halo_unpadding_1d(grad_output, mesh, mesh_dim, tensor_dim, halo[mesh_dim])
    

        # And, wrap it into a shard tensor:
        grad_tensor = ShardTensor(
            grad_output,
            spec,
            requires_grad=grad_output.requires_grad,
        )

        return grad_tensor, None, None, None


class UnSliceHaloND(torch.autograd.Function):
    
    """
    Class to trim off unnecessary sections of a tensor that has had a halo computation applied to it
    Used in cases such as: neighborhood attention, when the central halos require trimming after the application
    of the operation.
    """
    
    @staticmethod
    def forward(
        ctx,
        local_tensor : torch.Tensor,
        halo,
        mesh,
        placements,
    ) -> "ShardTensor":
        ctx.halo = halo
        ctx.mesh = mesh
        ctx.placements = placements
        
        assert len(halo) == mesh.ndim, f"Halo size ({len(halo)} must match mesh rank ({mesh.ndim}))"
        
        for mesh_dim in range(mesh.ndim):
            if isinstance(placements[mesh_dim], Shard):
                tensor_dim = placements[mesh_dim].dim
                local_tensor = halo_unpadding_1d(local_tensor, mesh, mesh_dim, tensor_dim, halo[mesh_dim])
        
        # Cast to shard tensor:
        stensor = ShardTensor.from_local(local_tensor, mesh, placements)
        print(f"FORWARD: local_tensor shape and local shape {stensor.shape}, {stensor._local_tensor.shape}")
        return stensor
    
    @staticmethod
    def backward(
        ctx,
        grad_output
    ) -> torch.Tensor:
        
        print(f"Grad output shape and local shape: {grad_output.shape}, {grad_output._local_tensor.shape}")
        
        # padded_tensor = halo_padding_1d(stensor.to_local(), mesh, halo[0], edge_padding_t, edge_padding_s[0])
        mesh = ctx.mesh
        halo = ctx.halo
        placements = ctx.placements
        # the gradient of the slicing is the halo operation that inverts it
        
        edge_padding_s = [0,] * len(halo)
        edge_padding_t = "none"
        
        local_tensor = grad_output.to_local()
        for mesh_dim in range(mesh.ndim):
            tensor_dim = placements[mesh_dim].dim
            local_tensor = halo_padding_1d(local_tensor, mesh, mesh_dim, tensor_dim, halo[mesh_dim], edge_padding_t, edge_padding_s[0])
        


        return local_tensor, None, None, None

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

# Make sure the module exists before importing it:
import importlib.util

natten_spec = importlib.util.find_spec("natten")
if natten_spec is not None:
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
        else:
            raise UndeterminedShardingError("q, k, and v must all be the same types (torch.Tensor or ShardTensor)")

else:
    def na2d_wrapper(*args, **kwargs):
        raise Exception("na2d_wrapper not supported because module 'natten' not installed")
    
    
