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

from typing import cast, Optional, Tuple, Sequence
from collections.abc import Iterable

import torch


import torch.distributed.tensor._dispatch as op_dispatch

from torch.distributed.tensor.placement_types import (
    Partial, 
    Placement,
    Replicate,
    Shard
)

import torch.distributed as dist

from modulus.distributed._shard_tensor_spec import (
    ShardTensorSpec,
    _infer_shard_tensor_spec_from_local_chunks
)

from modulus.distributed._shard_redistribute import (
    ShardRedistribute
)
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh


from modulus.distributed import DistributedManager

class _ToTorchTensor(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx,
        input : "ShardTensor",
        grad_placements: Optional[Sequence[Placement]] = None
    ) -> torch.Tensor:
        
        ctx.shard_tensor_spec = input._spec
        ctx.grad_placements = grad_placements
        local_tensor = input._local_tensor

        # JUST LIKE DTENSOR:
        # We need to return a fresh Tensor object there as autograd metadata
        # will be inplaced into it. So we don't want to pollute the Tensor
        # object stored in the _local_tensor of this ShardTensor.
        return local_tensor.view_as(local_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> "ShardTensor":
        
        shard_tensor_spec = ctx.shard_tensor_spec
        mesh = shard_tensor_spec.mesh
        
        
        grad_placements = ctx.grad_placements or shard_tensor_spec.placements
        
        # Generate a spec based on grad outputs and the expected placements:
        grad_tensor_spec = _infer_shard_tensor_spec_from_local_chunks(grad_output, mesh, grad_placements)
        
        return (
            ShardTensor(
                grad_output,
                grad_tensor_spec,
                requires_grad=grad_output.requires_grad
            ),
            None
        )
        
        
    
    

class _FromTorchTensor(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx,
        local_input: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: Tuple[Placement, ...],
    ) -> "ShardTensor":
        ctx.previous_placement = placements
        ctx.previous_mesh = device_mesh
        # This function is simpler than the corresponding DTensor implementation on the surface
        # because under the hood, we always do checks here.
        
        shard_tensor_spec = _infer_shard_tensor_spec_from_local_chunks(local_input, device_mesh, placements)

        shard_tensor = ShardTensor(
            local_input.view_as(local_input),
            shard_tensor_spec,
            requires_grad=local_input.requires_grad,
        )
        
        return shard_tensor
    
    @staticmethod
    def backward(
        ctx,
        grad_output: "ShardTensor",
    ) -> torch.Tensor:

        previous_placement = ctx.previous_placement
        previous_mesh = ctx.previous_mesh
        
        assert grad_output.placements == previous_placement, \
            "Resharding gradients not yet implemented"
            
        
        return grad_output.to_local(), None, None


from torch.distributed.tensor import DTensor

class ShardTensor(DTensor):
    """
    A class similar to pytorch's native DTensor but with more 
    flexibility for uneven data sharding.
    
    Leverages very similar API to DTensor (identical, where possible)
    but deliberately tweaking routines to avoid implicit assumptions 
    about tensor sharding.
    """    
    
    
    _local_tensor: torch.Tensor
    _spec: ShardTensorSpec
    __slots__ = ["_local_tensor", "_spec"]

    # Exactly the same as ShardTensor
    # _op_dispatcher instance as a class attribute to handle runtime dispatching logic
    _op_dispatcher: op_dispatch.OpDispatcher = op_dispatch.OpDispatcher()

    @staticmethod
    def __new__(
        cls, 
        local_tensor: torch.Tensor,
        spec: ShardTensorSpec,
        *,
        requires_grad: bool,
    ) -> "ShardTensor":
        """
        Construct a new Shard Tensor from a local tensor, device mesh, and placement.
        
        Note that unlike DTensor, ShardTensor will automatically collect the Shard size 
        information from all participating devices.  This is to enable uneven and
        dynamic sharding.
        
        Heavily derived from torch DTensor

        Parameters
        ----------
        local_tensor : torch.Tensor
            _description_
        spec : ShardTensorSpec
            _description_
        requires_grad : bool
            _description_

        Returns
        -------
        ShardTensor
            _description_
        """
        if local_tensor.requires_grad and not requires_grad:
            warnings.warn(
                "To construct a new ShardTensor from torch.Tensor, "
                "it's recommended to use local_tensor.detach() and "
                "make requires_grad consistent."
            )
            
        assert spec.tensor_meta is not None, "TensorMeta should not be None!"
        
        # Check the sharding information is known:
        ret = torch.Tensor._make_wrapper_subclass(
            cls,
            spec.tensor_meta.shape,
            strides       = spec.tensor_meta.stride,
            dtype         = local_tensor.dtype,
            device        = local_tensor.device,
            layout        = local_tensor.layout,
            requires_grad = requires_grad
        )
        
        ret._spec = spec
        ret._local_tensor = local_tensor
        
        return ret
    
    def __repr__(self):
        return f"ShardTensor(local_tensor={self._local_tensor}, device_mesh={self._spec.mesh}, placements={self._spec.placements})"
    
    @classmethod
    def _from_dtensor(cls, dtensor : DTensor) -> "ShardTensor":
        spec = ShardTensorSpec(
            mesh            = dtensor._spec.mesh,
            placements      = dtensor._spec.placements,
            tensor_meta     = dtensor._spec.tensor_meta,
            _sharding_sizes = None, # Leave this to none for a lazy init and assume it's not breaking to make this cast.
            _local_shape    = dtensor._local_tensor.shape
        )
        return ShardTensor.__new__(
            cls,
            local_tensor = dtensor._local_tensor,
            spec         = spec,
            requires_grad = dtensor.requires_grad
        )
    
    #TODO - methods from DTensor that might be necessary
    # def __tensor_flatten__(self) (https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/_api.py#L293)
    # @staticmethod
    # def __tensor_unflatten__(input_tensors, flatten_spec, outer_size, outer_stride) (https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/_api.py#L301)
    # def __coerce_tangent_metadata__(self) (https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/_api.py#L323)
    # def __coerce_same_metadata_as_tangent__(self) (https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/_api.py#L331)
    
    @classmethod
    @torch._disable_dynamo
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # Leverage DTensor Dispatch as much as possible, but, enable
        # the ability to operate on this output in the future:

        dispatch_res =  DTensor.__torch_dispatch__(func, types, args, kwargs)
        
        #TODO - for ``Partial`` specs, in SOME cases, we need to include a "weight"
        # For example, taking the average of an unevenly-sharded tensor, the weight
        # must be proportial to _local_tensor.

        
        
        # Return a shard tensor instead of a dtensor.
        # ShardTensor inherits from DTensor and can lazy-init from for efficiency 
        if isinstance(dispatch_res, DTensor):
            return ShardTensor._from_dtensor(dispatch_res)
        
        if isinstance(dispatch_res, Iterable):
            return type(dispatch_res)(
                ShardTensor._from_dtensor(d) if isinstance(d, DTensor) else d for d in dispatch_res
            )
        
        return dispatch_res
        
        
    def from_local(
        local_tensor: torch.Tensor,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        infer_shape: Optional[bool] = False,
    ) -> "ShardTensor":
        """
        Generate a new ShardTensor from local torch tensors.  Uses 
        device mesh and placements to infer global tensor properties.
        
        No restriction is made on forcing tensors to have equal shapes 
        locally.  Instead, the requirement is that tensor shapes could
        be concatenated into a single tensor according to the placements.

        Parameters
        ----------
        local_tensor : torch.Tensor
            Local chunk of tensor.  All participating tensors must be 
            of the same rank and concatable across the mesh dimensions
        device_mesh : Optional[DeviceMesh], optional
            Target Device Mesh, if not specified will use the current mesh, by default None
        placements : Optional[Sequence[Placement]], optional
            Target placements, must have the same number of elements as ``device_mesh.ndim`` , by default None
        infer_shape: Optional[bool]
            If infer_shape is False, from_local assumes an _even_ distirbution of the local tensor
            across each axis.  It will essentially create a DTensor and promote
        Returns
        -------
        ShardTensor
            A Shard Tensor Object
        """
        
        if infer_shape:
                        
            # This implementation follows the pytorch DTensor Implementation Closely.
            device_mesh = device_mesh or _mesh_resources.get_current_mesh()
            device_type = device_mesh.device_type
            
            # convert the local tensor to desired device base on device mesh's device_type
            if device_type != local_tensor.device.type and not local_tensor.is_meta:
                local_tensor = local_tensor.to(device_type)
            
            # set default placements to replicated if not specified
            if placements is None:
                placements = [Replicate() for _ in range(device_mesh.ndim)]
            else:
                placements = list(placements)
                for idx, placement in enumerate(placements):
                    # normalize shard dim to be positive
                    if placement.is_shard():
                        placement = cast(Shard, placement)
                        if placement.dim < 0:
                            placements[idx] = Shard(placement.dim + local_tensor.ndim)

            # `from_local` is differentiable, and the gradient of the dist tensor this function
            # created should flow back the gradients to the local_tensor, so we call an autograd
            # function to construct the dist tensor instead.
            return _FromTorchTensor.apply(  # pyre-ignore[16]: autograd func
                local_tensor,
                device_mesh,
                tuple(placements),
            )
        else:
            return ShardTensor._from_dtensor(
                DTensor.from_local(local_tensor, device_mesh, placements)
            )

        
    def redistribute(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        * ,
        async_op: bool = False):
        """
        This is just like DTensor redistribute except that we use a custom layer for
        shard redistribution.  Otherwise, the backwards pass will not evaluate correctly.

        Parameters
        ----------
        device_mesh : Optional[DeviceMesh], optional
            Mesh to use, by default None
        placements : Optional[Sequence[Placement]], optional
            Target Placements, by default None, will error if not supplied  
        async_op : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        RuntimeError
            _description_
        RuntimeError
            _description_
        """
        
        # if device_mesh is not specified, use the current device_mesh
        device_mesh = device_mesh or self.device_mesh
        # raise error if new placements not specified
        if placements is None:
            raise RuntimeError("placements is needed for redistribute!")

        placements = list(placements)
        for i, placement in enumerate(placements):
            if placement.is_partial():
                raise RuntimeError(
                    "Can not redistribute to Partial, redistributing to Partial is for internal use only!"
                )
            elif isinstance(placement, Shard) and placement.dim < 0:
                # normalize shard dim to be positive
                placements[i] = Shard(placement.dim + self.ndim)
        placements = tuple(placements)
        
        
        # pyre-fixme[16]: `Redistribute` has no attribute `apply`.
        return ShardRedistribute.apply(self, device_mesh, placements, async_op)
    
    
    def to_local(
        self, *, grad_placements: Optional[Sequence[Placement]] = None
    ) -> torch.Tensor:
        """
        Return the local tensor of this ShardTensor.  For Sharded
        tensors, there is no assurance this will be the same shape
        on each rank.

        Parameters
        ----------
        grad_placements : Optional[Sequence[Placement]], optional
            future layout of any gradientss from this function, by default None

        Returns
        -------
        torch.Tensor
            
        """
        
        if not torch.is_grad_enabled():
            return self._local_tensor
        
        if grad_placements is not None and not isinstance(grad_placements, tuple):
            grad_placements = tuple(grad_placements)
        
        return _ToTorchTensor.apply(
            self, grad_placements
        )
        

    
    