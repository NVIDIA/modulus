from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import (
    Partial, 
    Placement,
    Replicate,
    Shard,
)

from torch.distributed.tensor._dtensor_spec import (
    TensorMeta,
    DTensorSpec,
)

from typing import NamedTuple, Optional, Tuple



@dataclass(kw_only=True)
class ShardTensorSpec(DTensorSpec):
    # Adding an optional component to store the tensor meta
    # For all other shards of this tensor.
    # Information is only tracked for shards along the same axes of this particular shard!
    # Futher, it's assumed the stride layout and dtype is homogenous, so only track tensor sizes
    
    _local_shape: Optional[torch.Size]
    _sharding_sizes: Optional[Tuple[Tuple[torch.Size, ...]]] = None
    
    """
    Inherit from the DTensorSpec dataclass to keep all it's goodness,
    but extend functionality to include information about global placements
    of shards.  This is useful if the tensor is distributed in an uneven, 
    unexpected way.
    """
    
    def offsets(self,):
        """
        Compute and return the global offets of this tensor along each axis
        """
        raise NotImplementedError("TODO")

    def _hash_impl(self) -> int:
        # Extending the parent hash implementation to include
        # information about the all-shard meta
        if self.tensor_meta is not None and self._sharding_sizes is not None:
            return hash(
                (
                    self.mesh,
                    self.placements,
                    self.tensor_meta.shape,
                    self.tensor_meta.stride,
                    self.tensor_meta.dtype,
                    self.sharding_sizes,
                )
            )
        return hash((self.mesh, self.placements))

    def __hash__(self) -> int:
        # We lazily cache the spec to avoid recomputing the hash upon each
        # use, where we make sure to update the hash when the `tensor_meta`
        # changes by overriding `__setattr__`. This must be lazy so that Dynamo
        # does not try to hash non-singleton `SymInt`s for the stride.
        if self._hash is None:
            self._hash = self._hash_impl()
        return self._hash
    
    @property
    def sharding_sizes(self, mesh_dim: Optional[int] =None) -> Tuple[torch.Size]:
        if self._sharding_sizes is None:
            shard_shapes_by_dim, global_shape = _all_gather_shard_shapes(self._local_shape, self.placements, self.mesh)
            self._sharding_sizes = shard_shapes_by_dim
            self.tensor_meta._replace(shape = global_shape)
        if mesh_dim is not None:
            if mesh_dim < len(self._sharding_sizes): return self._sharding_sizes[mesh_dim]
        return self._sharding_sizes

    @property
    def local_shape(self):
        # Lazily get the local shape, if we don't have it:
        if self._local_shape is None:
            raise Exception("Missing local shape!")
        
        return self._local_shape
    
    @local_shape.setter
    def local_shape(self, value):
        print(f"Calling setter ... {value}")
        if not isinstance(value, torch.Size):
            raise TypeError("Local shape must be instance of torch.Size")
        self._local_shape = value
        print(f"Value set? {self._local_shape}")

    def offset(self, mesh_dim : Optional[int] = None) -> Tuple[int]:
        """
        Return the effective offset of this tensor along a sharded dim, as if it
        was all collected into one device and you wanted to slice it
        to recover the local slice.
        
        """
        
        # if mesh_dim is not None:
        offsets = []
        for mesh_dim in range(self.mesh.ndim):
            coord = self.mesh.get_coordinate()[mesh_dim]
            placement = self.placements[mesh_dim]
            tensor_dim = placement.dim
            shards = self.sharding_sizes[mesh_dim]
            # If the placement is not shard, offset is 0:
            if isinstance(placement, Shard):
                o = sum( [s[tensor_dim] for s in shards[:coord]])
                offsets.append(o)
            else:
                offsets.append(0)
            # Sum the shards along the tensor dim

        if mesh_dim is not None:
            return offsets[mesh_dim]

        return offsets


def _stride_from_contiguous_shape_C_style(shape : Tuple[int,]) -> Tuple[int]:
    """
    Compute and return the stride from a tensor shape, assuming it is 
    both contiguous and laid out in C-style

    Parameters
    ----------
    shape : Tuple[int,]
        input shape as Tuple or torch.Size

    Returns
    -------
    Tuple[int]
        list of strides of same length as input
    """
    # To compute strides, we make the assumption that the tensors are in the "C" style layout (default)
    # So, all strides at the deepest level are 1.
    stride = [1,]
    for axis_len in reversed(shape[1:]):
        next_stride = stride[-1] * axis_len
        stride.append(next_stride)
        
    stride = tuple(reversed(stride))
    return stride


def _all_gather_shard_shapes(
    local_shape : Tuple[int],
    placements : Tuple[Placement],
    target_mesh : DeviceMesh,
):
    
    shard_shapes_by_dim = []
    global_shape = [0 for s in local_shape]
    for mesh_axis, placement in enumerate(placements):
        
        tensor_dim = placement.dim
        
        if isinstance(placement, Shard):
            
            local_group = target_mesh.get_group(mesh_axis)
            
            local_size  = dist.get_world_size(group=local_group)
            
            all_shapes = [torch.Size()] * local_size
        
            # First, allgather the dimensions of each tensor to each rank:
            # Possible collective of CPU-based objects!  Could be slow if using separate hosts!
            dist.all_gather_object(all_shapes, local_shape, local_group)
            
            # Check that all shapes are the same rank:
            assert all([len(local_shape) == len(all_s) for all_s in all_shapes]), \
                "Rank mismatch detected when attempting to infer shapes and sizes"
        
            # Every dimension must be equal for this list, along the sharded axis
            for d in range(len(local_shape)):
                if d == tensor_dim: continue # skip the sharded dimension
                assert all([ local_shape[d] == all_s[d] for all_s in all_shapes])
    
            # Build a list of local torch.Size on this axis for each shard to store:
            local_meta = tuple(
                # torch.Size(tuple(s)) for s in zip(all_shapes)
                all_shapes
            )
        
            shard_shapes_by_dim.append(local_meta)
        
            # To infer the global shape _for this axis_, 
            # we have to loop over each axis in the rank list
            # To check what placement is there.
            # This assumes full sharding:
            global_shape[tensor_dim] = sum([all_s[tensor_dim] for all_s in all_shapes])

        else:
            # We're assuming that, during creation, this is replication along this axis and 
            # We need to use this axis shape as the global shape:
            shard_shapes_by_dim = (local_shape,)
            global_shape[tensor_dim] = local_shape[tensor_dim]
    
    return shard_shapes_by_dim, global_shape

def _infer_shard_tensor_spec_from_local_chunks(
    local_chunk : torch.Tensor, 
    target_mesh : DeviceMesh,
    placements : Tuple[Placement, ...]
) -> ShardTensorSpec:
    """
    Use local sizes, target mesh, and specified placements to build a 
    ShardTensorSpec.  Performs checks that all local tensors are compatible

    Parameters
    ----------
    local_chunk : torch.Tensor
        local tensor to be used as a shard of a global tensor
    target_mesh : DeviceMesh
        Device mesh object to build this ShardTensor on
    placements : Tuple[Placement, ...]
        Specified placements of this tensor

    Returns
    -------
    ShardTensorSpec
        Specification to be used in creating a ShardTensor.  Key feature
        of this spec is that each ShardTensor knows the shape and size of 
        other shards, and can compute global offsets and reductions properly
    """
    
    # Only accept contiguous local_chunk:
    assert local_chunk.is_contiguous(), \
        "ShardTensor can only be constructed from contiguous tensors"
    
    # Only accept sharding placements (not replications or partial (aka pending))
    assert all([ p.is_shard() for p in placements]), \
        "Shard Tensor will only infer shape and strides for sharded tensors," \
        "for replication use DTensor"
    
    # Need to infer the placements on each dimension of the mesh.
    
    assert len(placements) == target_mesh.ndim, "Mesh dimension must match placements length"
    
    
    local_shape = local_chunk.shape
    # Implicitly, assume sharding only happens over specified placements
    # We need all placements, however, to be less than or equal to the local rank:
    assert(len(placements) <= len(local_shape)), \
        f"Too many placements detected ({len(placements)}) for tensor of rank {len(local_shape)}"
    
        
    shard_shapes_by_dim, global_shape = _all_gather_shard_shapes(local_shape, placements, target_mesh,)

    stride = _stride_from_contiguous_shape_C_style(global_shape)

    # # Finally, build a tensor spec to return:
    global_meta = TensorMeta(shape=tuple(global_shape), stride=stride, dtype=local_chunk.dtype)
    # all_shard_meta = local_meta

    sharding_sizes = tuple(tuple(s) for s in shard_shapes_by_dim)

    return ShardTensorSpec(
        mesh           = target_mesh,
        placements     = placements,
        tensor_meta    = global_meta,
        _local_shape   = local_shape,
        _sharding_sizes = sharding_sizes,
    )
    
    
    