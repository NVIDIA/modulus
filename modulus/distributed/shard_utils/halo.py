from typing import Optional

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from modulus.distributed.shard_tensor import ShardTensor

from modulus.distributed import DistributedManager

from torch.distributed.tensor.placement_types import (
    Partial, 
    Placement,
    Replicate,
    Shard
)




def halo_unpadding_1d(
        local_tensor : torch.tensor,
        mesh : DeviceMesh,
        mesh_dim: int,
        tensor_dim: int,
        halo_t : int,
        edge_padding_t : Optional[str] = "zeros", 
        edge_padding_s : Optional[int] = 0,
    ) -> torch.Tensor:
    """
    Backward pass of the Distributed Halo Padding in 1D
    Can be chained to un-build a halo padding in multiple dimensions, if necessary
    
    Parameters
    ----------
    local_tensor : dist.tensor.DTensor
        Torch Tensor (Tensor) containing the local chunk 
    mesh : torch.distributed.device_mesh.DeviceMesh
        Torch DeviceMesh containing the information for the sharding of this tensor
    mesh_dim : int
        Mesh dimension to use for this 1D un-padding operation
    tensor_dim : int, optional
        Tensor dimension for the unslicing.  Mandatory
    halo_t : int
        halo padding size in this operation, assumed symmetrical by default
    edge_padding_t : Optional[str]
        What to do in the slicing on the mesh borders.  Valid options correspond to 
        the same options as convolutions in pytorch.  Currently Unused.
    edge_padding_s : Optional[int]
        How much edge slicing to use, if using edge slicing.  Only valid if using 
        zeros for the edge padding.  Currently Unused.
    Returns
    -------
    torch.Tensor
        Tensor with slicing applied locally to each chunk.  For some operations,
        like NAtten, coalescing after a sharded computation only makes sense after this
        operation is performed
    """

    # assert edge_padding_t in ["zeros", "reflect", "replicate", "circular", "none"], f"Invalid edge padding detected: {edge_padding_t}"
    
    # if edge_padding_s != 0 and edge_padding_t != "zeros":
    #     err_msg = f"Sharded convolution with edge_padding != 0 " \
    #                 f"(got {edge_padding_t} is only supported " \
    #                 f"if the edge padding type is \"zeros\" (got {edge_padding_t})."
    #     raise NotImplementedError(err_msg)
    
    # If the dim is None, ensure 1D mesh:
    if mesh_dim is None:
        assert mesh.ndim == 1, f"Halo padding requires `dim` to be set for mesh size greater than 1 (got shape {mesh.shape})"
        mesh_dim = 0
        
    # The local group can come right from the mesh.
    local_group = mesh.get_group(mesh_dim)
    local_rank  = mesh.get_local_rank(mesh_dim)
    local_size  = dist.get_world_size(group=local_group)

    # Select off the appropriate tensor dim:
    dim_shape = local_tensor.shape[tensor_dim]
        
    start = halo_t
    end = dim_shape - halo_t
        
    # Make corrections for edge effects:
    if local_rank == 0:
        start = 0
    if local_rank == local_size -1:
        end = dim_shape
        
    # Do the slicing:
    indices = torch.arange(start, end).to(local_tensor.device)
        
    local_tensor = local_tensor.index_select(tensor_dim, indices).contiguous()
    
    return local_tensor


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
            if edge_padding_s is None:
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
            if edge_padding_s is None:
                padded_output.append(torch.zeros_like(halo_to_right))
            else:
                shape = list(halo_to_right.shape)
                shape[target_dim] = edge_padding_s 
                zeros = torch.zeros(shape, device=halo_to_right.device, dtype=halo_to_right.dtype)
                padded_output.append(zeros)
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
    
