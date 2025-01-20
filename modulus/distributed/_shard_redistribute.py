
from typing import Tuple, cast, List

import torch
from torch.distributed.device_mesh import DeviceMesh
import torch.distributed._functional_collectives as funcol
from torch.distributed.tensor._dtensor_spec import (
    TensorMeta,
)
from torch.distributed.tensor.placement_types import (
    Partial, 
    Placement,
    Replicate,
    Shard,
)

from torch.distributed.tensor._redistribute import (
    _gen_transform_infos,
    # _gen_transform_infos_non_cached,
    _TransformInfo,
)

import modulus.distributed.shard_tensor as shard_tensor
from modulus.distributed import (
    all_gather_v,
)
from modulus.distributed._shard_tensor_spec import ShardTensorSpec

# Worker functions for the collectives specific to uneven shaped tensors:
def _to_replicate_tensor(
    local_tensor: torch.Tensor, 
    device_mesh: DeviceMesh, 
    mesh_dim: int, 
    tensor_dim: int,
    # current_logical_shape: List[int],
    current_spec: ShardTensorSpec,
) -> torch.Tensor:

    # Get the mesh for the group:
    mesh = current_spec.mesh
    group = mesh.get_group(mesh_dim)

    # Get all sizes:
    sizes = current_spec.sharding_sizes
    this_sizes = [s[tensor_dim] for s in sizes[mesh_dim]]
    
    # We can implement this with a straightforward allgather_v
    local_tensor = all_gather_v(local_tensor, sizes=this_sizes, dim=tensor_dim, group=group)
    
    return local_tensor
    
def _select_slice_from_replicate(
    local_tensor: torch.Tensor,
    target_spec: ShardTensorSpec,
    mesh_dim : int,
    mesh_coord : int,
) -> torch.Tensor:
    
    # Get the offsets for this tensor:
    offsets = target_spec.offset(mesh_dim)
    
    # We really only need the sizes from this dimension:
    sizes = target_spec.sharding_sizes
    tensor_dim = target_spec.placements[mesh_dim].dim
    this_sizes = [s[tensor_dim] for s in sizes[mesh_dim]]
    
    
    # Split the tensor:
    chunks = torch.split(local_tensor, this_sizes, dim=tensor_dim)
    
    
    return chunks[mesh_coord]
    
def redistribute_local_shard_tensor(
    local_tensor: torch.Tensor,
    current_spec: ShardTensorSpec,
    target_spec: ShardTensorSpec,
    *,
    async_op: bool = False,
    is_backward: bool = False,
) -> torch.Tensor:
    """
    This redistribute the local tensor (torch.Tensor) from the current ShardTensorSpec to
    the target ShardTensorSpec, which involves the necessary collective calls to transform
    the local shard of the ShardTensor from its current spec to the target spec.
    
    The collective operations are implemented in the Placement classes, which we avoid
    modifying.  To get around that, we mimic the logic from pytorch's original redistribute.
    But, in cases where a tensor is Sharded and the shards are uneven (spec.is_uneven)
    we intercept and replace the collectives:
    
    ``Shard(dim)`` -> ``Replicate()``: ``all_gather_v`` instead of ``all_gather``
    ``Shard(src_dim)`` -> ``Shard(dst_dim)``: remains all_to_all but reimplemented to handle sizes correctly
    ``Replicate()`` -> ``Shard(dim)``: local chunking is **unchanged** but return value is ShardTensorSpec instead.
    ``Partial()`` -> ``Replicate()``: ``all_reduce``needs to become a weighted ``all_reduce``, depending on operation.
    ``Partial()`` -> ``Shard(dim)``: ``reduce_scatter`` needs to become a weighted ``reduce_scatter``, depending on operation
    """

    if current_spec.mesh != target_spec.mesh:
        # TODO: alltoall/permute reshuffling to change device_mesh if they are not the same
        raise NotImplementedError("Cross device mesh comm not supported yet!")

    new_local_tensor = None
    device_mesh = current_spec.mesh

    my_coordinate = device_mesh.get_coordinate()

    if my_coordinate is None:
        # if rank is not part of mesh, we skip redistribute and simply return local_tensor,
        # which should be an empty tensor
        return local_tensor


    # if has_symints:
    #     transform_infos = _gen_transform_infos_non_cached(current_spec, target_spec)
    # else:
    transform_infos = _gen_transform_infos(current_spec, target_spec)

    for transform_info in transform_infos:
        i = transform_info.mesh_dim
        current, target = transform_info.src_dst_placements
        device_mesh.size(mesh_dim=i)

        if current == target:
            # short cut, just use the original local tensor
            new_local_tensor = local_tensor
            continue

        # logger.debug("redistribute from %s to %s on mesh dim %s", current, target, i)
        if target.is_replicate():
            # Case 1: target is Replicate
            if current.is_partial():
                partial_spec = cast(Partial, current)
                print("About to call reduce value!")
                new_local_tensor = partial_spec._reduce_value(
                    local_tensor, device_mesh, i
                )
                print("Called reduce value!")
            elif current.is_shard():
                current_placement = cast(Shard, current)
                new_local_tensor = _to_replicate_tensor(
                    local_tensor, 
                    device_mesh,  
                    mesh_dim = i,
                    tensor_dim = current_placement.dim,
                    # transform_info.logical_shape, 
                    current_spec = current_spec,
                )
            else:
                raise RuntimeError(
                    f"redistribute from {current} to {target} not supported yet"
                )
        elif target.is_shard():
            # Case 2: target is Shard
            target_placement = cast(Shard, target)
            if current.is_partial():
                partial_spec = cast(Partial, current)
                new_local_tensor = partial_spec._reduce_shard_value(
                    local_tensor, device_mesh, i, target_placement
                )
            elif current.is_replicate():
                # split the tensor and return the corresponding cloned local shard
                # new_local_tensor = target_placement._replicate_to_shard(
                #     local_tensor, device_mesh, i, my_coordinate[i]
                # )
                new_local_tensor = _select_slice_from_replicate(
                    local_tensor, target_spec, i, my_coordinate[i]
                )
            else:
                assert (
                    current.is_shard()
                ), f"Current placement should be shard but found {current}"
                shard_spec = cast(Shard, current)
                if shard_spec.dim != target_placement.dim:
                    new_local_tensor = shard_spec._to_new_shard_dim(
                        local_tensor,
                        device_mesh,
                        i,
                        transform_info.logical_shape,
                        target_placement.dim,
                    )
        elif target.is_partial():
            if current.is_replicate():
                partial_spec = cast(Partial, target)
                # skip the replicate to partial transformation when we are in backward pass
                # In this case we keep the grad as replicate, this is because we don't
                # want to convert the replicated gradients back to partial, although
                # that's logically conform with the same layout, converting the gradients
                # back to partial is actually useless as you would have to do reduce later
                # which would be more expensive than keeping it replicate! For this reason,
                # we keep the replicate grad here.
                new_local_tensor = (
                    partial_spec._partition_value(local_tensor, device_mesh, i)
                    if not is_backward
                    else local_tensor
                )
            elif current.is_shard():
                if not is_backward:
                    raise RuntimeError(
                        f"redistribute from {current} to {target} not supported yet"
                    )
                # for backward shard -> partial, we just need to convert the shard to replicate
                current_placement = cast(Shard, current)
                # TODO - resolve sharding to partials?
                new_local_tensor = current_placement._to_replicate_tensor(
                    local_tensor, device_mesh, i, transform_info.logical_shape
                )
            else:
                # partial -> partial no op, should never hit
                new_local_tensor = local_tensor

        assert new_local_tensor is not None
        local_tensor = new_local_tensor

    assert new_local_tensor is not None, "redistribute failed!"

    if not async_op and isinstance(new_local_tensor, funcol.AsyncCollectiveTensor):
        new_local_tensor = new_local_tensor.wait()

    return new_local_tensor


class ShardRedistribute(torch.autograd.Function):
    """
    This is a ShardTensor enhanced version of redistribute.  It extends 
    the functionality in DTensor to allow redistribution of sharded tensors.

    Parameters
    ----------
    torch : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    @staticmethod
    def forward(  # type: ignore[override]
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        input: "ShardTensor",
        device_mesh: DeviceMesh,
        placements: Tuple[Placement, ...],
        async_op: bool = False,
    ) -> "shard_tensor.ShardTensor":

        current_spec = input._spec

        
        ctx.current_spec = current_spec
        ctx.async_op = async_op

        if current_spec.placements != placements:
            target_spec = ShardTensorSpec(
                device_mesh, placements, tensor_meta=input._spec.tensor_meta, _local_shape=input._spec.local_shape
            )

            local_tensor = input._local_tensor
            output = redistribute_local_shard_tensor(
                local_tensor, current_spec, target_spec, async_op=async_op
            )
        else:
            # use the same local tensor if placements are the same.
            output = input._local_tensor
            target_spec = current_spec

        # print("FORWARD in SHARD REDISTRIBUTE")
        # print(f"Input shape: {input.shape} with local shape {input._local_tensor.shape}")
        # print(f"Input placements: {input._spec.placements}")
        # print(f"OUTPUT shape: {output.shape}")
        # print(f"OUTPUT spec: {target_spec}")

        return shard_tensor.ShardTensor(
            output,
            target_spec,
            requires_grad=input.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad_output: "ShardTensor"):  # type: ignore[override]

        previous_spec = ctx.current_spec
        current_spec = grad_output._spec

        async_op = ctx.async_op

        local_tensor = grad_output._local_tensor
        
        output = redistribute_local_shard_tensor(
            local_tensor,
            current_spec,
            previous_spec,
            async_op=async_op,
            is_backward=True,
        )

        # normalize the target placement to replicate if it is partial
        normalized_placements: List[Placement] = []
        for previous_placement in previous_spec.placements:
            if previous_placement.is_partial():
                # keep target placement to replicate instead of partial in this case
                normalized_placements.append(Replicate())
            else:
                normalized_placements.append(previous_placement)

        spec = ShardTensorSpec(
            previous_spec.device_mesh,
            tuple(normalized_placements),
            tensor_meta=TensorMeta(
                shape=grad_output.shape,
                stride=grad_output.stride(),
                dtype=grad_output.dtype,
            ),
            _local_shape = output.shape
        )
        output_shard_tensor = shard_tensor.ShardTensor(
            output,
            spec,
            requires_grad=grad_output.requires_grad,
        )

        return (
            output_shard_tensor,
            None,
            None,
            None,
        )