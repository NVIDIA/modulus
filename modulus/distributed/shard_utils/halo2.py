from typing import Tuple, Optional

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from modulus.distributed.shard_tensor import ShardTensor

from modulus.distributed import DistributedManager


def compute_halo_size(stride : int, kernel: int, dilation: int, ) -> int:
    """
    Compute the single-dimension halo size for a convolution kernel

    Parameters
    ----------
    stride : int
        Convolution stride (along this axis)
    kernel : int
        convolution kernel size (along this axis)
    dilation : int
        Convolution's dilation parameter    

    Returns
    -------
    int
        Symmetrical Halo size on each side of a chunk of data

    """
        
    # To calculate the halo needed, we need to first determine the receptive field.

    # Dilation != 1 is not handled yet
    # TODO
    if dilation != 1: 
        raise NotImplementedError("Dilation different from 1 is not supported in halo computations.")

    # The receptive field is how far in the input a pixel in the output can see
    # It's used to calculate how large the halo computation has to be
    receptive_field = dilation * (kernel - 1)  + 1
    # receptive_field = kernel + (kernel - 1) * (stride - 1) - 1
        
        
    # The number of halo pixels is the casting `int(receptive field/2)`
    # Why?  Assuming a filter in the output image is centered in the input image,
    # we have only half of it's filter to the left.
    # Even kernels:
    if kernel % 2 == 0:
        halo_size =  int(receptive_field / 2 - 1) 
    else:
        halo_size =  int(receptive_field / 2 ) 
        
        
    return halo_size

def halo_padding_1d(
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        tensor_dim: int,
        halo_t : int,
        edge_padding_t : Optional[str] = "zeros", 
        edge_padding_s : Optional[int] = 0,
    ) -> torch.Tensor:  # pragma: no cover
    """
    Forward pass of the Distributed Halo Padding in 1D
    Can be chained to build a halo padding in multiple dimensions, if necessary
    
    Parameters
    ----------
    local_tensor : dist.tensor.DTensor
        Torch Tensor (Tensor) containing the local chunk 
    mesh : torch.distributed.device_mesh.DeviceMesh
        Torch DeviceMesh containing the information for the sharding of this tensor
    mesh_dim : int
        Dimension of mesh to use for this 1D padding operation
    tensor_dim : int, optional
        Dimension for the halo reduction.  Mandatory
    halo_t : int
        halo padding size in this operation, assumed symmetrical by default
    edge_padding_t : Optional[str]
        What to do in the edge padding on the mesh borders.  Valid options correspond to 
        the same options as convolutions in pytorch.
    edge_padding_s : Optional[int]
        How much edge padding to use, if using edge padding.  Only valid if using 
        zeros for the edge padding 
    Returns
    -------
    torch.Tensor
        Tensor with padding applied locally to each chunk.  Note that coalescing this tensor
        directly, without the operation meant to consume the halo, will produce garbage results.
    """

    assert edge_padding_t in ["zeros", "reflect", "replicate", "circular", "none"], f"Invalid edge padding detected: {edge_padding_t}"
    
    if edge_padding_s != 0 and edge_padding_t != "zeros":
        err_msg = f"Sharded convolution with edge_padding != 0 " \
                    f"(got {edge_padding_t} is only supported " \
                    f"if the edge padding type is \"zeros\" (got {edge_padding_t})."
        raise NotImplementedError(err_msg)
    
    # If the dim is None, ensure 1D mesh:
    if mesh_dim is None:
        assert mesh.ndim == 1, f"Halo padding requires `dim` to be set for mesh size greater than 1 (got shape {mesh.shape})"
        dim = 0
    
    # # Check the dimension fits the mesh:
    # if mesh.ndim != 0:
    #     assert dim < mesh.ndim, f"Halo padding can not pad dimension {dim} for mesh of shape {mesh.shape} (size {mesh.ndim})."
    
    
    # The local group can come right from the mesh, which the tensor knows already.
    # Here's its implicit mesh is 1D so we don't pass a mesh_dim
    local_group = mesh.get_group(mesh_dim)
    local_rank  = mesh.get_local_rank(mesh_dim)
    local_size  = dist.get_world_size(group=local_group)

    # Initialize peer2peer buffers to be empty
    # They will be left empty unless there is a message to exchange
    all_to_all_source = [ torch.empty(0, device=local_tensor.device, dtype=local_tensor.dtype) for _ in range(local_size)]
    all_to_all_dest   = [ torch.empty(0, device=local_tensor.device, dtype=local_tensor.dtype) for _ in range(local_size)]
        
    # This is the tensor axis we'll target:    
    target_dim = tensor_dim
        
    # TERMINOLOGY: The halo is implemented with "left" and "right" but it can be any axis.
    # Left means "to the rank one lower than this rank" along the mesh axis.
    # (And vice-versa for right)
    # This can be any physical device, depending on the mesh layout.
    
    # To make padding easier to maintain, always set the indices and select the left and right boundaries:
    left_indices  = torch.arange(0, halo_t).to(local_tensor.device)
    # index_select doesn't accept negative indices, so get the real values:
    max_index = local_tensor.shape[target_dim]
    right_indices = max_index - 1 - left_indices
    # (Need to flip them to complete the mirror)
    right_indices = torch.flip(right_indices, (0,))

    
    halo_to_left  = local_tensor.index_select(target_dim, left_indices).contiguous()
    halo_to_right = local_tensor.index_select(target_dim, right_indices).contiguous()
    
    # Outgoing Halo, send from this rank but index is destination rank
    # These two if blocks are for non-edge ranks:
    if local_rank != 0:
        # Send one left (don't include the bias - it's already accounted for!)
        all_to_all_source[local_rank - 1] = halo_to_left
        # Receive one left (need to initialize an empty buffer of the right size):
        all_to_all_dest[local_rank - 1] = torch.zeros_like(halo_to_left).contiguous() 
    if local_rank != local_size - 1: 
        # Send one right:
        all_to_all_source[local_rank + 1] = halo_to_right
        # Receive one right:
        all_to_all_dest[local_rank + 1] = torch.zeros_like(halo_to_right).contiguous()
        
    # This handles the edge rank for circular padding, which requires the halo pass:
    if edge_padding_t == "circular":
        # Then we send across the edges in a circular fashion:
        if local_rank == 0:
            # Send one left that wraps around
            all_to_all_source[local_size - 1] = halo_to_left
            all_to_all_dest[local_size - 1] = torch.zeros_like(halo_to_left).contiguous()
        if local_rank == local_size - 1:
            # Send from N-1 to 0:
            all_to_all_source[local_rank + 1] = halo_to_right
            # Receive one right:
            all_to_all_dest[local_rank + 1] = torch.zeros_like(halo_to_right).contiguous()
            
    # Do the collective: Scatter and gather the halo objects:
    dist.all_to_all(all_to_all_dest, all_to_all_source, group=local_group)
        
    
    # Build a list of tensors to concatenate (with care for the edge cases):
    padded_output = []
    if local_rank != 0:
        # From one left
        padded_output.append(all_to_all_dest[local_rank - 1])
    else:
        # Deal with padding on the first entry.
        # Using "halo_to_left" as base shape
        if edge_padding_t == "zeros":
            if edge_padding_t is None:
                padded_output.append(torch.zeros_like(halo_to_left))
            else:
                shape = list(halo_to_left.shape)
                shape[target_dim] = edge_padding_s 
                zeros = torch.zeros(shape, device=halo_to_left.device, dtype=halo_to_left.dtype)
                padded_output.append(zeros)
        elif edge_padding_t == "reflect":
            padded_output.append(halo_to_left.flip(target_dim))
        elif edge_padding_t == "replicate":
            raise NotImplementedError("Need to implement replcate padding")
            # TODO            
        elif edge_padding_t == "circular":
            padded_output.append(all_to_all_dest[-1])
        elif edge_padding_t == "none":
            pass
    
    # Central tensor:
    padded_output.append(local_tensor)
    
    if local_rank != local_size - 1: 
        # From one right:
        padded_output.append(all_to_all_dest[local_rank + 1])
    else:
        # Deal with padding on the last entry.
        # Using "halo_to_right" as base shape
        if edge_padding_t == "zeros":
            padded_output.append(torch.zeros_like(halo_to_right))
        elif edge_padding_t == "reflect":
            padded_output = halo_to_right.flip(target_dim)
        elif edge_padding_t == "replicate":
            raise NotImplementedError("Need to implement replcate padding")
            # TODO        
        elif edge_padding_t == "circular":
            padded_output.append(all_to_all_dest[0])
        elif edge_padding_t == "none":
            pass
            
    # Finish up:
    return torch.cat(padded_output, dim=target_dim)
    


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
            tensor_dim = placements[mesh_dim].dim
            local_tensor = halo_padding_1d(local_tensor, mesh, mesh_dim, tensor_dim, halo[mesh_dim], edge_padding_t, edge_padding_s[0])
        



        # padded_tensor = halo_padding_1d(stensor.to_local(), mesh, halo[0], edge_padding_t, edge_padding_s[0])
        ctx.mesh = mesh
        ctx.halo = halo

        return local_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # pragma: no cover
        """backward pass of the of the Distributed HaloPadding primitive"""

        print(f"got grad output: {grad_output}")

        grad_tensor = all_gather_v_bwd_wrapper(
            grad_output,
            ctx.sizes,
            dim=ctx.dim,
            use_fp32=ctx.use_fp32,
            group=ctx.group,
        )

        if not ctx.needs_input_grad[0]:
            grad_tensor = None

        return grad_tensor, None, None, None, None
