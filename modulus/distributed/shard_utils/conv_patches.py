from typing import Tuple, Optional

import torch

import torch.distributed as dist

from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import (
    Shard
)


from modulus.distributed import ShardTensor

from . patch_core import promote_to_iterable

import wrapt


__all__ = [
    "conv2d_wrapper"
]





from modulus.distributed.shard_utils.patch_core import (
    UndeterminedShardingError,
    MissingShardPatch,
)

__all__ = [
    "conv2d_wrapper"
]

def conv_output_shape(L_in, p, s, k, d):
    L_out = ( L_in + 2*p - d*(k-1) - 1) / s + 1
    return int(L_out) 




def compute_halo_from_kernel_stride_and_dilation(kernel_size, stride, dilation):
    """
    Compute the single-dimension halo size for a convolution kernel

    Parameters
    ----------
    kernel_size : int
        convolution kernel size (along this axis)
    stride : int
        Convolution stride (along this axis)
    dilation : int
        Convolution's dilation parameter

    Returns
    -------
    int
        Symmetrical Halo size on each side of a chunk of data
        
    Raises
    -------
    MissingShardPatch
        Exception for unsupported shapes in sharding.
    """

    # If the kernel is even, and matches the stride, and dilation is 1, no halo:


    if kernel_size % 2 == 0:
        if kernel_size == stride and dilation == 1:
            return 0
        else:
            raise MissingShardPatch("Sharded Convolution is not implemented for even kernels without matching stride")
    
    
    if dilation != 1:
        raise MissingShardPatch("Sharded Convolution is not implemented for dilation != 1")
    
    
    
    # The receptive field is how far in the input a pixel in the output can see
    # It's used to calculate how large the halo computation has to be
    receptive_field = dilation * (kernel_size - 1)  + 1
    # receptive_field = kernel + (kernel - 1) * (stride - 1) - 1
        
        
    # The number of halo pixels is the casting `int(receptive field/2)`
    # Why?  Assuming a filter in the output image is centered in the input image,
    # we have only half of it's filter to the left.
    # Even kernels:
    if kernel_size % 2 == 0:
        halo_size =  int(receptive_field / 2 - 1) 
    else:
        halo_size =  int(receptive_field / 2 ) 
        
        
    return halo_size
        


def shard_to_haloed_local_for_conv2d(input, kernel_shape, stride, padding, dilation, groups):

    mesh = input._spec.mesh
    placements = input._spec.placements

    # Extract the full kernel size.  Assuming B C H (W) (D)
    _, _, *full_kernel = kernel_shape

    full_stride, full_padding, full_dilation = tuple( promote_to_iterable(p, full_kernel) for p in (stride, padding, dilation))

    # We only care about dimensions that are sharded:
    h_kernel = []
    h_stride = []
    h_padding = []
    h_dilation = []
    for p in placements:
        if not isinstance(p, Shard): continue
        tensor_dim = p.dim
        
        # Extract the kernel on this dim:
        assert tensor_dim not in [0,1], "Can not compute a domain-parallel convolution on data sharded in batch or channel dimension"
        assert tensor_dim < len(kernel_shape), "Can not use a tensor dim for a rank beyond the weight rank."
        
        h_kernel.append(full_kernel[tensor_dim - 2])
        h_stride.append(full_stride[tensor_dim - 2])
        h_padding.append(full_padding[tensor_dim - 2])
        h_dilation.append(full_dilation[tensor_dim - 2])
            
    
    
    # We need to make sure stride, padding and dilation are iterable of the same length:
    # How big of a halo do we need?
    halo_size = tuple(
        compute_halo_from_kernel_stride_and_dilation(k, s, d)
        for k, s, d in zip(h_kernel, h_stride, h_dilation)
    )
    
    # Padding here is pretty much always 0s.
    # TODO - check and fix any edge cases
    edge_padding_t = "none"
    
    
    # Use the halo layer to compute halos:
    local_input = HaloPaddingND.apply(
        input,
        halo_size,
        edge_padding_t,
        padding,
    ) 

    
    return local_input


    
from . halo import halo_padding_1d

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
                local_tensor = halo_padding_1d(local_tensor, mesh, mesh_dim, tensor_dim, halo[mesh_dim], edge_padding_t, edge_padding_s[mesh_dim])
        
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
        
        
        # for mesh_dim in range(mesh.ndim):
        #     if isinstance(placements[mesh_dim], Shard):
        #         tensor_dim = placements[mesh_dim].dim
        #         grad_output = halo_unpadding_1d(grad_output, mesh, mesh_dim, tensor_dim, halo[mesh_dim])
    

        # And, wrap it into a shard tensor:
        grad_tensor = ShardTensor(
            grad_output,
            spec,
            requires_grad=grad_output.requires_grad,
        )

        return grad_tensor, None, None, None


class PartialConv2D(torch.autograd.Function):
    """
    This function exists to pass gradients through, properly, a 2D conv.
    """
    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weights : torch.nn.Parameter,
        bias: Optional[torch.nn.Parameter],
        stride,
        padding,
        dilation,
        groups,
        base_func,
        output_spec, 
    ) -> "ShardTensor":  # pragma: no cover
        """forward pass of the Distributed Conv2d primitive"""
        ctx.spec = output_spec
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.base_func = base_func

        ctx.weight_spec = weights._spec
        ctx.bias_spec   = bias._spec

        # Converting weights and bias to local tensors.  Cast back to DTensor in the backward pass
        weights = weights.to_local()
        
        # This applies the native, underlying na2d:
        if bias is not None: 
            bias = bias.to_local()

        local_chunk =  base_func(inputs, weights, bias, stride, padding, dilation, groups)

        ctx.save_for_backward(inputs, weights, bias)

        output = ShardTensor.from_local(
            local_chunk,
            output_spec.mesh,
            output_spec.placements
        )
        

        return output

    @staticmethod
    def backward(ctx, grad_output: "ShardTensor") -> "ShardTensor":  # pragma: no cover
        """backward pass of the of the Distributed HaloPadding primitive"""

        spec = ctx.spec
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        base_func = ctx.base_func
        
        local_chunk, weight, bias = ctx.saved_tensors
    
    
        local_grad_output = grad_output._local_tensor

        # Rotate the weights for the grad input:
        grad_input = base_func(local_grad_output, weight.flip(-1,2))

        # Cast grad_input to shard tensor for further backward pass
        grad_input = ShardTensor.from_local(
            grad_input,
            grad_output._spec.mesh,
            grad_output._spec.placements,
        )

        # Compute weights gradient:
        grad_weight = base_func(
            local_chunk.permute(1,0,2,3), 
            local_grad_output.permute(1,0,2,3), 
            stride=stride, padding=padding, 
            groups=groups, dilation=dilation
        )
        grad_weight = grad_weight.permute(1,0,2,3).contiguous()
        # Sync weight group:
        weight_group = ctx.weight_spec.mesh.get_group()
        dist.all_reduce(grad_weight, group=weight_group)
        # Cast back to DTensor
        grad_weight = DTensor.from_local(
            grad_weight,
            ctx.weight_spec.mesh,
            ctx.weight_spec.placements,
        )
        
        
        if bias is not None:
            # Compute bias grad and cast back to DTensor
            grad_bias = local_grad_output.sum((0,2,3))
            bias_group = ctx.bias_spec.mesh.get_group()
            dist.all_reduce(grad_bias, group=bias_group)
            grad_bias = DTensor.from_local(
                grad_bias,
                ctx.bias_spec.mesh,
                ctx.bias_spec.placements,
            )
            
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None



# Convolution patches
# @wrapt.patch_function_wrapper(aten,'convolution.default')
@wrapt.patch_function_wrapper('torch.nn.functional', 'conv2d')
def conv2d_wrapper(wrapped, instance, args, kwargs):
    """
    Perform na2d on sharded tensors.  Expected that q, k, v
    are sharded the same way and on the same mesh.  And that
    they have the same shape, locally and globally.
    """
    
    # args[0], args[1], and args[2] should all be shard tensors to do the wrapping
    # If they are regular torch tensors, do nothing unusual!


    def unpack_key_arguments(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *args, **kwargs):
        return input, weight, bias, stride, padding, dilation, groups, args

    input, weight, bias, stride, padding, dilation, groups, remaining_args = unpack_key_arguments(*args, **kwargs)


    # mixing type and isinstance here because ShardTensor inherits DTensor inherits torch.Tensor.

    # Allow bias to be None
    if type(input) == torch.Tensor and \
        type(weight) == torch.nn.parameter.Parameter and \
        (bias is None or type(bias) == torch.nn.parameter.Parameter):
        return wrapped(*args, **kwargs)
    elif type(input) == ShardTensor and \
        isinstance(weight, (ShardTensor, DTensor)) and \
        (bias is None or isinstance(bias, (ShardTensor, DTensor))):
        
        # This applies a halo layer and returns local torch tensors:
        local_input = shard_to_haloed_local_for_conv2d(input, weight.shape, stride, padding, dilation, groups)

        spec = input._spec


        # local_weight = weight._local_tensor
        local_weight = weight.to_local()
        
        # This applies the native, underlying na2d:
        if bias is not None: 
            local_bias = bias.to_local()
            # local_bias = bias._local_tensor

        x = PartialConv2D.apply(local_input, weight, bias, stride, padding, dilation, groups, wrapped, spec)
        
        return x
        
    else:
        raise UndeterminedShardingError("input, weight, bias (if not None) must all be the same types (torch.Tensor or ShardTensor)")
    
