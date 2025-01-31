from typing import Tuple, Optional, Union

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
    "conv1d_wrapper",
    "conv2d_wrapper",
    "conv3d_wrapper",
]

def conv_output_shape(L_in, p, s, k, d):
    L_out = ( L_in + 2*p - d*(k-1) - 1) / s + 1
    return int(L_out) 




def compute_halo_from_kernel_stride_and_dilation(kernel_size: int, stride: int, dilation: int):
    """
    Genericially compute the single-dimension halo size for a convolution kernel

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
        


def shard_to_haloed_local_for_convNd(
        input: ShardTensor, 
        kernel_shape: Tuple[int], 
        stride: Union[int, Tuple[int]], 
        padding: Union[int, Tuple[int]], 
        dilation: Union[int, Tuple[int]], 
        groups: Optional[int] = 1,
        **extra_kwargs,
    ) -> torch.Tensor:
    """
        Take a shard tensor as well as the corresponding convolution parameters
    and convert it to a local tensor to apply a convolution on.


    Parameters
    ----------
    input : ShardTensor
        A Modulus ShardedTensor.  Contains a mesh and sharding spec that will be used to
        coordinate halo operations.
    kernel_shape : Tuple[int]
        The convolutional kernel shape (aka, weights.shape).  Used to compute 
        halo padding in each sharded dimension.
    stride : Union[int, Tuple[int]]
        The convolution stride, as either a single int or tuple of ints.
        If a tuple, it must match the size of the kernel.
    padding : Union[int, Tuple[int]]
        The convolution padding, as either a single int or tuple of ints.
        If a tuple, it must match the size of the kernel.
    dilation : Union[int, Tuple[int]]
        The convolution dilation, as either a single int or tuple of ints.
        If a tuple, it must match the size of the kernel.
        Note: Dilation sizes that do not equal 1 are not currently supported
        but if you need them, open a ticket.
    groups : Optional[int], optional
        Number of groups for the convolution, by default 1

    Returns
    -------
    torch.Tensor
        A torch.Tensor with the appropriate halo regions added for this convolution.
        When the convolution is applied to this local tensor, it will produce the 
        local output as if the convolution had been applied globally, and then
        sharded.
    """


    mesh = input._spec.mesh
    placements = input._spec.placements

    assert mesh.ndim == len(placements)


    # We only care about dimensions that are sharded.
    # h here stands for halo, not height!
    h_kernel = []
    h_stride = []
    h_padding = []
    h_dilation = []
    for p in placements:
        if not isinstance(p, Shard): continue
        tensor_dim = p.dim

        # Extract the kernel on this dim:
        assert tensor_dim not in [0,1], "Can not compute a domain-parallel convolution on data sharded in batch or channel dimension"
        # We add two here because of Batch and Channel dims
        assert tensor_dim < len(kernel_shape) + 2, "Can not use a tensor dim for a rank beyond the weight rank."
        
        # Assuming the first two dims are N, C
        h_kernel.append(kernel_shape[tensor_dim - 2])
        h_stride.append(stride[tensor_dim - 2])
        h_padding.append(padding[tensor_dim - 2])
        h_dilation.append(dilation[tensor_dim - 2])
            
    
    
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
    local_input = HaloPaddingConvND.apply(
        input,
        halo_size,
        edge_padding_t,
        padding,
    ) 
    
    
    return local_input



    
from . halo import (
    halo_padding_1d, 
    halo_unpadding_1d, 
    perform_halo_collective,
    apply_grad_halo
)

class HaloPaddingConvND(torch.autograd.Function):
    """
    Autograd Wrapper for a distributed Convolution-centric HaloPadding primitive.
    It is based on the modulus ShardTensor concept which presents as a
    local_tensor + device Mesh + shard placements.  In the forward pass, the adjacent regions
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
                local_tensor = halo_padding_1d(
                    local_tensor, 
                    mesh, 
                    mesh_dim, 
                    tensor_dim, 
                    halo[mesh_dim], 
                    edge_padding_t, 
                    edge_padding_s[mesh_dim]
                )
        
        # padded_tensor = halo_padding_1d(stensor.to_local(), mesh, halo[0], edge_padding_t, edge_padding_s[0])
        ctx.halo = halo
        ctx.spec = stensor._spec
        ctx.requires_input_grad = stensor.requires_grad

        return local_tensor


    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> "ShardTensor":  # pragma: no cover
        """backward pass of the of the Distributed HaloPadding primitive"""


        spec = ctx.spec
        mesh = spec.mesh
        placements = spec.placements
        halo = ctx.halo
        
        
        for mesh_dim in range(mesh.ndim):
            if isinstance(placements[mesh_dim], Shard):
                tensor_dim = placements[mesh_dim].dim
                grad_input, grad_halos = halo_unpadding_1d(grad_output, mesh, mesh_dim, tensor_dim, halo[mesh_dim], return_slices=True)

                # The gradient halos for the backward pass with a general convolution need to be be 
                # sent to their original locations and _added_, not concatenated
                all_to_all_dest = perform_halo_collective(mesh, mesh_dim, *grad_halos)
                
                grad_input = apply_grad_halo(mesh, mesh_dim, tensor_dim, grad_input, all_to_all_dest)
                
        # And, wrap it into a shard tensor:
        grad_tensor = ShardTensor(
            grad_input,
            spec,
            requires_grad=grad_input.requires_grad,
        )

        return grad_tensor, None, None, None

aten = torch.ops.aten
class PartialConvND(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        weights : torch.nn.Parameter,
        bias: Optional[torch.nn.Parameter],
        output_spec : "ShardTensorSpec", 
        conv_kwargs : dict,

    ) -> "ShardTensor":  # pragma: no cover
        """forward pass of the Distributed Conv2d primitive"""
        ctx.spec = output_spec
        ctx.conv_kwargs = conv_kwargs

            
        # Save the local versions of weights, bias, otherwise
        # It will dispatch to the tensor parallel conv ...
        ctx.save_for_backward(inputs, weights, bias)

        # Call the generic pytorch layer for convolutions:
        local_chunk =  aten.convolution.default(inputs, weights, bias, **conv_kwargs)

        output = ShardTensor.from_local(
            local_chunk,
            output_spec.mesh,
            output_spec.placements
        )
        
        ctx.requires_input_grad = inputs.requires_grad

        return output

    @staticmethod
    def backward(ctx, grad_output: "ShardTensor") -> "ShardTensor":  # pragma: no cover
        """backward pass of the of the Distributed HaloPadding primitive"""

        spec = ctx.spec
        conv_kwargs = ctx.conv_kwargs
        
        local_chunk, weight, bias = ctx.saved_tensors
    
        output_mask = (
            ctx.requires_input_grad,
            True,
            bias is not None,
        )
    
        local_grad_output = grad_output._local_tensor
        grad_input, grad_weight, grad_bias = \
            aten.convolution_backward(
                local_grad_output,
                local_chunk,
                weight, 
                bias,
                output_mask=output_mask,
                **conv_kwargs,
            )
            
       
        # We have to make sure the weights and biases agree on the domain (a local allreduce)
        group = spec.mesh.get_group()
        dist.all_reduce(grad_weight, group=group)
        if grad_bias is not None:
            dist.all_reduce(grad_bias, group=group)
       

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

    


@wrapt.patch_function_wrapper('torch.nn.functional', 'conv1d')
def conv1d_wrapper(wrapped, instance, args, kwargs):
    
    return generic_conv_nd_wrapper(wrapped, instance, args, kwargs)    
    
@wrapt.patch_function_wrapper('torch.nn.functional', 'conv2d')
def conv2d_wrapper(wrapped, instance, args, kwargs):
    
    return generic_conv_nd_wrapper(wrapped, instance, args, kwargs)    

@wrapt.patch_function_wrapper('torch.nn.functional', 'conv3d')
def conv3d_wrapper(wrapped, instance, args, kwargs):
    
    return generic_conv_nd_wrapper(wrapped, instance, args, kwargs)    

def generic_conv_nd_wrapper(wrapped, instance, args, kwargs):
    
    # Extract the tensor inputs as well as the conv parameters:
    input, weight, bias, conv_kwargs = repackage_conv_args(*args, **kwargs)



    # mixing type and isinstance here because ShardTensor inherits DTensor inherits torch.Tensor.

    # Allow bias to be None
    if type(input) == torch.Tensor and \
        type(weight) == torch.nn.parameter.Parameter and \
        (bias is None or type(bias) == torch.nn.parameter.Parameter):
        return wrapped(*args, **kwargs)
    elif type(input) == ShardTensor:
        # Dynamically collect weights as needed.
        # If they are sharded (DTensor), gather them locally.
        if isinstance(weight, (ShardTensor, DTensor)):
            weight = weight.full_tensor()
        
        if isinstance(bias, (ShardTensor, DTensor)):
            bias = bias.full_tensor()


        # (bias is None or isinstance(bias, (ShardTensor, DTensor))):
        #     elif type(input) == ShardTensor and \
        # isinstance(weight, (ShardTensor, DTensor)) and \
        # (bias is None or isinstance(bias, (ShardTensor, DTensor))):
        kernel_shape = weight.shape[2:]

        promotables = ["stride", "padding", "dilation", "output_padding"]
        # For the relevant args, promote to the same size as the kernel:
        conv_kwargs = {
            key : promote_to_iterable(p, kernel_shape) if key in promotables else p
            for key, p in conv_kwargs.items()
        }
        
        #         # Converting weights and bias to local tensors.  Cast back to DTensor in the backward pass
        # weights = weights.to_local()
        
        # # This applies the native, underlying na2d:
        # if bias is not None: 
        #     bias = bias.to_local()
        
        # This applies a halo layer and returns local torch tensors:
        local_input = shard_to_haloed_local_for_convNd(input, kernel_shape, **conv_kwargs)

        output_spec = input._spec

        x = PartialConvND.apply(local_input, weight, bias, output_spec, conv_kwargs)
        
        return x
        
    else:
        
        msg = "input, weight, bias (if not None) must all be the valid types " \
            "(torch.Tensor or ShardTensor), but got " \
            f"{type(input)}, " \
            f"{type(weight)}, " \
            f"{type(bias)}, "
        raise UndeterminedShardingError(msg)
    

def repackage_conv_args(
        input:          Union[torch.Tensor, ShardTensor],
        weight:         Union[torch.Tensor, DTensor],
        bias:           Union[torch.Tensor, DTensor, None],
        stride:         Union[int, Tuple[int]] =1,
        padding:        Union[int, Tuple[int]] =0,
        dilation:       Union[int, Tuple[int]] =1,
        groups:         int = 1,
        transposed:     bool = False,
        output_padding: Union[int, Tuple[int]] =0,
        *args,
        **kwargs,
    ):
    """
    Take a set of arguments to a generic convolution (or transposed)
    and extract the non-input, non-weight, and non-bias parameters
    into a kwargs dictionary.
    """
    
    return_kwargs = {
        "stride"         : stride,
        "padding"        : padding,
        "dilation"       : dilation,
        "transposed"     : transposed,
        "output_padding" : output_padding,
        "groups"         : groups,
    }
    
    return input, weight, bias, return_kwargs