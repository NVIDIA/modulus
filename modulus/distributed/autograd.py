# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, List
from .manager import DistributedManager
from .utils import (
    gather_v_wrapper,
    scatter_v_wrapper,
    all_gather_v_wrapper,
    all_reduce_v_wrapper,
    indexed_all_gather_wrapper,
    indexed_all_gather_wrapper_bwd,
)


class AllGatherVAutograd(torch.autograd.Function):
    """Autograd Wrapper for Distributed AllGatherV"""

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        sizes: List[int],
        dim: int = 0,
        use_fp32: bool = True,
        group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """forward pass of the Distributed AllGatherV primitive"""

        gathered_tensor = all_gather_v_wrapper(tensor, sizes, dim, group)
        ctx.sizes = sizes
        ctx.group = group
        ctx.dim = dim
        ctx.use_fp32 = use_fp32
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """backward pass of the Distributed AllGatherV primitive"""

        grad_tensor = None
        needs_grad = ctx.needs_input_grad[0]

        if needs_grad:
            grad_tensor = all_reduce_v_wrapper(
                grad_output, ctx.sizes, ctx.dim, use_fp32=ctx.use_fp32, group=ctx.group
            )

        return grad_tensor, None, None, None, None


class GatherVAutograd(torch.autograd.Function):
    """Autograd Wrapper for Distributed GatherV"""

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        sizes: List[int],
        dim: int = 0,
        dst: int = 0,
        group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """forward pass of the Distributed GatherV primitive"""

        gathered_tensor = gather_v_wrapper(tensor, sizes, dim, dst, group)
        ctx.sizes = sizes
        ctx.dim = dim
        ctx.dst = dst
        ctx.group = group
        return gathered_tensor

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        """forward pass of the Distributed GatherV primitive"""

        grad_tensor = None
        needs_grad = ctx.needs_input_grad[0]

        if needs_grad:
            grad_tensor = scatter_v_wrapper(
                grad_output, ctx.sizes, ctx.dim, ctx.dst, ctx.group
            )

        return grad_tensor, None, None, None, None


class ScatterVAutograd(torch.autograd.Function):
    """Autograd Wrapper for Distributed ScatterV"""

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        sizes: List[int],
        dim: int = 0,
        src: int = 0,
        group=Optional[dist.ProcessGroup],
    ) -> torch.Tensor:
        """forward pass of the Distributed ScatterV primitive"""

        scattered_tensor = scatter_v_wrapper(tensor, sizes, dim, src, group)
        ctx.tensor = tensor
        ctx.sizes = sizes
        ctx.dim = dim
        ctx.src = src
        ctx.group = group
        return scattered_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """backward pass of the Distributed ScatterV primitive"""

        grad_tensor = None
        needs_grad = ctx.needs_input_grad[0]

        if needs_grad:
            grad_tensor = gather_v_wrapper(
                grad_output, ctx.sizes, ctx.dim, ctx.src, ctx.group
            )

        return grad_tensor, None, None, None, None


class IndexedAllGatherAutograd(torch.autograd.Function):
    """Autograd Wrapper for an Indexed AllGather"""

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        scatter_indices: List[torch.Tensor],
        recv_sizes: List[int],
        send_sizes: List[int],
        use_fp32: bool = True,
        group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """forward pass of the Distributed IndexedAllGather primitive"""

        tensor_to_recv = indexed_all_gather_wrapper(
            tensor,
            scatter_indices,
            recv_sizes,
            send_sizes,
            group,
        )

        ctx.recv_sizes = recv_sizes
        ctx.send_sizes = send_sizes
        ctx.use_fp32 = use_fp32
        ctx.group = group
        ctx.tensor_dim0 = tensor.size(0)
        ctx.scatter_indices = scatter_indices

        return tensor_to_recv

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> torch.Tensor:
        """backward pass of the Distributed IndexedAllGather primitive"""

        needs_grad = ctx.needs_input_grad[0]
        grad_tensor = None

        if needs_grad:
            grad_tensor = indexed_all_gather_wrapper_bwd(
                grad_output,
                ctx.scatter_indices,
                ctx.recv_sizes,
                ctx.send_sizes,
                ctx.tensor_dim0,
                ctx.use_fp32,
                ctx.group,
            )

        return grad_tensor, None, None, None, None, None, None


def all_gather_v(
    tensor: torch.Tensor,
    sizes: List[int],
    dim: int = 0,
    use_fp32: bool = True,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Performs a distributed all_gather of tensors of varying sizes
    across all partipitating ranks within the specified process group.

    Parameters
    ----------
    tensor : torch.Tensor
        tensor to be gathered
    sizes : List[int]
        list of the size along the primary dimension to be gathered for each rank
    dim : int, default=0
        primary tensor dimension along which gathered tensors are concatenated
    use_fp32 : bool, default=True
        whether to use FP32 precision for the reduce operation in the backward pass.
    group : torch.distributed.ProcessGroup, optional
        process group across which distributed operation is performed, if passed as
        None, the default world group is used.
    """

    return AllGatherVAutograd.apply(tensor, sizes, dim, use_fp32, group)


def gather_v(
    tensor: torch.Tensor,
    sizes: List[int],
    dim: int = 0,
    dst: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Performs a distributed gather of tensors of varying sizes
    across all partipitating ranks within the specified process group.

    Parameters
    ----------
    tensor : torch.Tensor
        tensor to be gathered
    sizes : List[int]
        list of the size along the primary dimension to be gathered for each rank
    dim : int, default=0
        primary tensor dimension along which gathered tensors are concatenated
    dst : int, default=0
        destination rank of this operation; returned tensor only valid on this rank
    group : torch.distributed.ProcessGroup, optional
        process group across which distributed operation is performed, if passed as
        None, the default world group is used.
    """

    return GatherVAutograd.apply(tensor, sizes, dim, dst, group)


def scatter_v(
    tensor: torch.Tensor,
    sizes: List[int],
    dim: int = 0,
    src: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Performs a distributed scatter of tensors of varying sizes
    across all partipitating ranks within the specified process group.

    Parameters
    ----------
    tensor : torch.Tensor
        tensor to be gathered
    sizes : List[int]
        list of the size along the primary dimension to be scattered for each rank
    dim : int, default=0
        primary tensor dimension along which gathered tensors are concatenated
    src : int, default=0
        source rank of this operation, all ranks receive corresponding portion from this rank
    group : torch.distributed.ProcessGroup, optional
        process group across which distributed operation is performed, if passed as
        None, the default world group is used.
    """

    return ScatterVAutograd.apply(tensor, sizes, dim, src, group)


def indexed_all_gather(
    tensor: torch.Tensor,
    scatter_indices: List[torch.Tensor],
    recv_sizes: List[int],
    send_sizes: List[int],
    use_fp32: bool = True,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Performs a distributed indexed all_gather of tensors of varying sizes
    across all partipitating ranks within the specified process group.

    Parameters
    ----------
    tensor : torch.Tensor
        tensor to be gathered
    scatter_indices : List[torch.Tensor]
        list of tensors specifying the data rows which are sent from each
        rank to each other rank
    recv_sizes : List[int]
        list of number of data rows to be received from each rank
    send_sizes : List[int]
        list of number of data rows to be sent to each rank
    use_fp32 : bool, default=True
        whether to use FP32 precision for the reduce operation in the backward pass.
    group : torch.distributed.ProcessGroup, optional
        process group across which distributed operation is performed, if passed as
        None, the default world group is used.
    """

    return IndexedAllGatherAutograd.apply(
        tensor,
        scatter_indices,
        recv_sizes,
        send_sizes,
        use_fp32,
        group,
    )
