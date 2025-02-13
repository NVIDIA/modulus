# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, cast

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import (
    TensorMeta,
)
from torch.distributed.tensor._redistribute import (
    _gen_transform_infos,
)
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
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
    current_spec: ShardTensorSpec,
) -> torch.Tensor:
    """
    Converts a sharded tensor to a replicated tensor by gathering all shards.

    Args:
        local_tensor (torch.Tensor): The local shard of the tensor to replicate
        device_mesh (DeviceMesh): The device mesh containing process groups
        mesh_dim (int): The mesh dimension along which to gather
        tensor_dim (int): The tensor dimension along which data is sharded
        current_spec (ShardTensorSpec): Specification of current sharding scheme

    Returns:
        torch.Tensor: The fully replicated tensor on this rank

    Note:
        This function handles uneven sharding by using all_gather_v instead of regular all_gather
    """

    # Get the mesh for the group:
    mesh = current_spec.mesh
    group = mesh.get_group(mesh_dim)

    # Get all sizes:
    sizes = current_spec.sharding_sizes()

    this_sizes = [s[tensor_dim] for s in sizes[mesh_dim]]

    # # Ensure contiguous data for the reduction:
    local_tensor = local_tensor.contiguous()
    # We can implement this with a straightforward allgather_v
    local_tensor = all_gather_v(
        local_tensor, sizes=this_sizes, dim=tensor_dim, group=group
    )

    return local_tensor


def _select_slice_from_replicate(
    local_tensor: torch.Tensor,
    target_spec: ShardTensorSpec,
    mesh_dim: int,
    mesh_coord: int,
) -> torch.Tensor:
    """
    Selects the appropriate slice from a replicated tensor to create a shard.

    Args:
        local_tensor (torch.Tensor): The replicated tensor to slice from
        target_spec (ShardTensorSpec): Specification of target sharding scheme
        mesh_dim (int): The mesh dimension along which to shard
        mesh_coord (int): The coordinate of this rank in the mesh dimension

    Returns:
        torch.Tensor: The selected slice that will become this rank's shard

    Note:
        This function handles uneven sharding by using the sharding sizes from the target spec
        to split the tensor into potentially uneven chunks
    """

    # TODO - This needs a rework to enable caching of shapes for a grad pass.

    # We really only need the sizes from this dimension:
    tensor_dim = target_spec.placements[mesh_dim].dim
    mesh_size = target_spec.mesh.size(mesh_dim=mesh_dim)

    # Split the tensor:
    chunks = torch.tensor_split(local_tensor, mesh_size, dim=tensor_dim)

    return chunks[mesh_coord]


def _shard_reduce(
    local_tensor: torch.Tensor,
    target_spec: ShardTensorSpec,
    mesh_dim: int,
    mesh_coord: int,
):
    pass


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
                new_local_tensor = partial_spec._reduce_value(
                    local_tensor, device_mesh, i
                )
            elif current.is_shard():
                current_placement = cast(Shard, current)
                new_local_tensor = _to_replicate_tensor(
                    local_tensor,
                    device_mesh,
                    mesh_dim=i,
                    tensor_dim=current_placement.dim,
                    current_spec=current_spec,
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
                if not current.is_shard():
                    raise RuntimeError(
                        f"Current placement should be shard but found {current}"
                    )
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

        if new_local_tensor is None:
            raise RuntimeError(
                "Failed to create new local tensor during redistribution"
            )
        local_tensor = new_local_tensor

    if new_local_tensor is None:
        raise RuntimeError("redistribute failed!")

    if not async_op and isinstance(new_local_tensor, funcol.AsyncCollectiveTensor):
        new_local_tensor = new_local_tensor.wait()

    return new_local_tensor


class ShardRedistribute(torch.autograd.Function):
    """
    This is a ShardTensor enhanced version of redistribute. It extends
    the functionality in DTensor to allow redistribution of sharded tensors.

    This autograd function handles both forward and backward passes for redistributing
    sharded tensors between different sharding schemes.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: "shard_tensor.ShardTensor",
        device_mesh: DeviceMesh,
        placements: Tuple[Placement, ...],
        async_op: bool = False,
    ) -> "shard_tensor.ShardTensor":
        """
        Forward pass for redistributing a sharded tensor.

        Args:
            ctx: Autograd context for saving tensors/variables for backward
            input: Input sharded tensor to redistribute
            device_mesh: Target device mesh for redistribution
            placements: Target placement scheme for redistribution
            async_op: Whether to perform redistribution asynchronously

        Returns:
            Redistributed sharded tensor with new placement scheme
        """
        current_spec = input._spec

        ctx.current_spec = current_spec
        ctx.async_op = async_op

        if current_spec.placements != placements:

            target_spec = ShardTensorSpec(
                device_mesh,
                placements,
                tensor_meta=input._spec.tensor_meta,
                _local_shape=input._spec.local_shape,
            )

            local_tensor = input._local_tensor
            output = redistribute_local_shard_tensor(
                local_tensor, current_spec, target_spec, async_op=async_op
            )
        else:
            # use the same local tensor if placements are the same.
            output = input._local_tensor
            target_spec = current_spec

        return shard_tensor.ShardTensor(
            output,
            target_spec,
            requires_grad=input.requires_grad,
        )

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: "shard_tensor.ShardTensor",
    ) -> Tuple["shard_tensor.ShardTensor", None, None, None]:
        """
        Backward pass for redistributing a sharded tensor.

        Args:
            ctx: Autograd context containing saved tensors/variables from forward
            grad_output: Gradient output tensor to redistribute back

        Returns:
            Tuple containing:
            - Redistributed gradient tensor
            - None for device_mesh gradient (not needed)
            - None for placements gradient (not needed)
            - None for async_op gradient (not needed)
        """
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
            _local_shape=output.shape,
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
