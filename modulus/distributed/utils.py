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

# TODO this also needs more docstrings
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .manager import DistributedManager


def get_memory_format(tensor):
    """Gets format for tensor"""
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    else:
        return torch.contiguous_format


def pad_helper(tensor, dim, new_size, mode="zero"):
    """Util for padding tensors"""
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    ndim_pad = ndim - dim
    output_shape = [0 for _ in range(2 * ndim_pad)]
    orig_size = tensor.shape[dim]
    output_shape[1] = new_size - orig_size
    tensor_pad = F.pad(tensor, output_shape, mode="constant", value=0.0)

    if mode == "conj":
        lhs_slice = [
            slice(0, x) if idx != dim else slice(orig_size, new_size)
            for idx, x in enumerate(tensor.shape)
        ]
        rhs_slice = [
            slice(0, x) if idx != dim else slice(1, output_shape[1] + 1)
            for idx, x in enumerate(tensor.shape)
        ]
        tensor_pad[lhs_slice] = torch.flip(
            torch.conj(tensor_pad[rhs_slice]), dims=[dim]
        )

    return tensor_pad


def truncate_helper(tensor, dim, new_size):
    """Util for truncating"""
    input_format = get_memory_format(tensor)
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    output_slice = [
        slice(0, x) if idx != dim else slice(0, new_size)
        for idx, x in enumerate(tensor.shape)
    ]
    tensor_trunc = tensor[output_slice].contiguous(memory_format=input_format)

    return tensor_trunc


def split_tensor_along_dim(tensor, dim, num_chunks):
    """splits tensor along specific dim"""
    if not (dim < tensor.dim()):
        raise AssertionError(
            f"Error, tensor dimension is {tensor.dim()} which cannot be"
            f"split along {dim}"
        )
    if not (tensor.shape[dim] % num_chunks == 0):
        raise AssertionError(
            f"Error, cannot split dim {dim} evenly. Dim size is \
        {tensor.shape[dim]} and requested numnber of splits is {num_chunks}"
        )
    chunk_size = tensor.shape[dim] // num_chunks
    tensor_list = torch.split(tensor, chunk_size, dim=dim)

    return tensor_list


@torch.no_grad()
def gather_loss(loss: float, dst_rank: int = 0, mean: bool = True):  # pragma: no cover
    """Gathers loss from all processes to one for logging

    Parameters
    ----------
    loss : float
        loss value
    dst_rank : int, Optional
        destination rank to gather to, by default 0
    mean : bool, Optional
        Calculate the mean of the losses gathered, by default True

    Raises
    ------
    Exception
        If DistributedManager has yet to be initialized
    """
    if not DistributedManager.is_initialized():
        raise Exception(
            "Distributed manager should be initialized when using gather_loss"
        )

    distmng = DistributedManager()
    loss = torch.Tensor([loss])

    # For serial runs, just return the current loss!
    if distmng.world_size == 1:
        return float(loss)

    # Gather using PyTorch distributed function
    gather_list = None
    if distmng.rank == dst_rank:
        gather_list = [
            torch.zeros(1).to(distmng.device) for i in range(distmng.world_size)
        ]
    dist.gather(loss.to(distmng.device), gather_list, dst_rank)

    # Return loss if dst_rank, None otherwise
    if distmng.rank == dst_rank:
        loss = torch.sum(torch.cat(gather_list))
        if mean:
            loss = loss / distmng.world_size
        return float(loss.cpu())
    else:
        return None


# distributed primitives
def distributed_transpose(tensor, dim0, dim1, group=None, async_op=False):
    """Perform distributed transpose of tensor to switch sharding dimension"""
    # get input format
    input_format = get_memory_format(tensor)

    # get comm params
    comm_size = dist.get_world_size(group=group)

    # split and local transposition
    split_size = tensor.shape[dim0] // comm_size
    x_send = [
        y.contiguous(memory_format=input_format)
        for y in torch.split(tensor, split_size, dim=dim0)
    ]
    x_recv = [torch.empty_like(x_send[0]) for _ in range(comm_size)]

    # global transposition
    req = dist.all_to_all(x_recv, x_send, group=group, async_op=async_op)

    return x_recv, req


def _reduce(input_, use_fp32=True, group=None):  # pragma: no cover
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_

    # All-reduce.
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.float()
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        dist.all_reduce(input_, group=group)

    return input_


def _split(input_, dim_, group=None):  # pragma: no cover
    """Split the tensor along its last dimension and keep the corresponding slice."""
    # get input format
    input_format = get_memory_format(input_)

    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous(memory_format=input_format)

    return output


def _gather(input_, dim_, group=None):  # pragma: no cover
    """Gather tensors and concatenate along the specified dimension."""
    # get input format
    input_format = get_memory_format(input_)

    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size == 1:
        return input_

    # sanity checks
    if not (dim_ < input_.dim()):
        raise AssertionError(
            f"Error, cannot gather along {dim_} for tensor with {input_.dim()} "
            "dimensions."
        )

    # Size and dimension.
    comm_rank = dist.get_rank(group=group)

    tensor_list = [torch.empty_like(input_) for _ in range(comm_size)]
    tensor_list[comm_rank] = input_
    dist.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim_).contiguous(memory_format=input_format)

    return output


def all_gather_v_wrapper(
    tensor: torch.Tensor,
    sizes: List[int],
    dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Implements a distributed AllGatherV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive gathers all local tensors from each rank into the
    full global tensor onto each rank.

    Parameters
    ----------
    tensor : "torch.Tensor"
        local tensor on each rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        full global tensor, valid on each rank
    """

    comm_size = dist.get_world_size(group=group)
    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()

    if comm_size == 1:
        return tensor

    tensor_shape = list(tensor.shape)
    tensor_list = [None] * comm_size

    for src in range(comm_size):
        tensor_shape[dim] = sizes[src]
        tensor_list[src] = torch.empty(
            tensor_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    dist.all_gather(tensor_list, tensor, group=group)

    output = torch.cat(tensor_list, dim=dim)

    return output


def all_reduce_v_wrapper(
    tensor: torch.Tensor,
    sizes: List[int],
    dim: int = 0,
    use_fp32: bool = True,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Implements a distributed AllReduceV primitive. It is based
    on the idea of a single global tensor which which can be distributed
    along a specified dimension into chunks of variable size.
    This primitive assumes different global tensors of the same shape on each
    rank. It then re-distributes chunks of all these tensors such that each rank
    receives all corresponding parts of a global tensor. Each rank then sums up
    the chunks after receiving it. By design, this primitive thus implements the
    backward pass of the "all_gather_v" primitive. In this case, the result would
    be a single global gradient tensor distributed onto different ranks.

    Parameters
    ----------
    tensor : torch.Tensor
        global tensor on each rank (different one on each rank)
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    use_fp32 : bool, optional
        flag to specify FP32 precision for the redcution, by default True
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        local tensor, i.e. result of reduction of all corresponding chunks
        from all global tensors for each rank separately
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()

    tensor_shape = list(tensor.shape)
    tensor_shape[dim] = sizes[rank]
    tmp = [
        torch.empty(
            tensor_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        for _ in range(comm_size)
    ]
    scatter_list = list(torch.split(tensor, sizes, dim=dim))

    dist.all_to_all(tmp, scatter_list, group=group)
    stack_dim = tensor.dim()
    tmp = torch.stack(tmp, dim=stack_dim)

    if use_fp32:
        # cast to float before sum and return float, then cast back
        output = tmp.sum(dim=stack_dim, dtype=torch.float32)
        output = output.to(dtype=tensor.dtype)
    else:
        # else: just do sum in native dtype
        output = tmp.sum(dim=stack_dim)

    return output


def gather_v_wrapper(
    tensor: torch.Tensor,
    sizes: List[int],
    dim: int = 0,
    dst: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Implements a distributed GatherV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive assumes such a distributed tensor and gathers all
    local tensors from each rank into the full global tensor valid
    on the specified destination rank.

    Parameters
    ----------
    tensor : torch.Tensor
        local tensor on each rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    dst : int, optional
        destination rank which contains the full global tensor after the
        operation, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        full global tensor, valid on destination rank
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if not (0 <= dst < comm_size):
        raise ValueError()
    if tensor.size(dim) != sizes[rank]:
        raise ValueError()

    if comm_size == 1:
        return tensor

    gather_list = [None] * comm_size
    tensor_shape = list(tensor.shape)

    for r in range(comm_size):
        tensor_shape[dim] = sizes[r]
        gather_list[r] = torch.empty(
            tensor_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    # dist.scatter doesn't support tensors of different shape
    # so this implementation is using explicit send/recv combinations
    if rank == dst:
        req_list = [None] * comm_size
        for r in range(comm_size):
            if r == dst:
                gather_list[r] = tensor
            else:
                req_list[r] = dist.irecv(gather_list[r], src=r, group=group)

        for r in range(comm_size):
            if r != dst:
                req_list[r].wait()

    else:
        req = dist.isend(tensor, dst=dst, group=group)
        req.wait()

    output = torch.cat(gather_list, dim=dim)

    return output


def scatter_v_wrapper(
    tensor: torch.Tensor,
    sizes: List[int],
    dim: int = 0,
    src: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Implements a distributed ScatterV primitive. It is based
    on the idea of a single global tensor which is distributed along
    a specified dimension into chunks of variable size.
    This primitive scatters the global tensor from a specified source rank
    into local chunks onto each other rank.

    Parameters
    ----------
    tensor : torch.Tensor
        global tensor, valid on source rank
    sizes : List[int]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set each rank
    dim : int, optional
        dimension along which global tensor is distributed, by default 0
    src : int, optional
        source rank of primitive, i.e. rank of original full global tensor, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        corresponding local part of the global tensor on each rank
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if not (0 <= src < comm_size):
        raise ValueError()

    tensor_shape = list(tensor.shape)
    tensor_shape[dim] = sizes[rank]
    output = torch.empty(
        tensor_shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )

    # dist.scatter doesn't support tensors of different shape
    # so this implementation is using explicit send/recv combinations
    scatter_list = None
    if rank == src:
        scatter_list = torch.split(tensor, sizes, dim=dim)
        req_list = [None] * comm_size
        for r in range(comm_size):
            tensor_to_scatter_to_r = scatter_list[r]
            if r == src:
                output = tensor_to_scatter_to_r
            else:
                req_list[r] = dist.isend(tensor_to_scatter_to_r, dst=r, group=group)

        for r in range(comm_size):
            if r != src:
                req_list[r].wait()

    else:
        req = dist.irecv(output, src=src, group=group)
        req.wait()

    return output


def indexed_all_to_all_v_wrapper(
    tensor: torch.Tensor,
    indices: List[torch.Tensor],
    sizes: List[List[int]],
    dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Implements an indexed version of a distributed AllToAllV
    primitive. It is based on the idea of a single global tensor which
    is distributed along a specified dimension into chunks of variable size.
    This primitive assumes a set of indices into this dimension which indicate
    the corresponding slices sent to each other rank forming an indexed version
    of an AllToAllV primitive.

    Parameters
    ----------
    tensor : torch.Tensor
        local part of global tensor on each rank
    indices : List[torch.Tensor]
        list of indices on each rank of slices being sent to
        each other rank from this rank
    sizes : List[List[int]]
        number of indices each rank sends to each other rank,
        valid and set on each rank, e.g. sizes[0][3] corresponds
        to the number of slices rank 0 sends to rank 3
    dim : int
        dimension along which global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        local result of primitive corresponding to indexed global tensor
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if len(sizes[rank]) != comm_size:
        raise ValueError()
    if len(indices) != comm_size:
        raise ValueError()

    indices = torch.cat(indices, dim=0)
    tensor_to_send = torch.index_select(tensor, dim=dim, index=indices)

    recv_list = [None] * comm_size
    for r in range(comm_size):
        recv_list[r] = scatter_v_wrapper(
            tensor_to_send,
            sizes=sizes[r],
            src=r,
            dim=dim,
            group=group,
        )
    tensor_to_recv = torch.cat(recv_list, dim=dim)

    return tensor_to_recv


def indexed_all_to_all_v_wrapper_bwd(
    tensor: torch.Tensor,
    indices: List[torch.Tensor],
    sizes: List[List[int]],
    tensor_size_along_dim: int,
    use_fp32: bool = True,
    dim: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:  # pragma: no cover
    """
    Implements the backward pass to the indexed version of a distributed
    AllToAllV primitive.

    Parameters
    ----------
    tensor : torch.Tensor
        local tensor, i.e. gradient on resulting tensor from forward pass
    indices : List[torch.Tensor]
        list of indices on each rank of slices being sent to
        each other rank from this rank
    sizes : List[List[int]]
        list of the sizes of each chunk on each rank along distributed dimension,
        valid and set on each rank
    tensor_size_along_dim : int
        size of original local tensor along specified dimension,
        i.e. from the corresponding forward pass
    use_fp32 : bool, optional
        flag to specify FP32 precision, by default True
    dim : int, optional
        dimension along with global tensor is distributed, by default 0
    group : Optional[dist.ProcessGroup], optional
        process group along which global tensor is shared, by default None

    Returns
    -------
    torch.Tensor
        result of primitive corresponding to indexed global tensor
    """

    comm_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    if len(sizes) != comm_size:
        raise ValueError()
    if dim >= tensor.dim():
        raise ValueError()
    if len(sizes[rank]) != comm_size:
        raise ValueError()
    if len(indices) != comm_size:
        raise ValueError()

    indices = torch.cat(indices, dim=0)
    tensor_shape = list(tensor.shape)

    # scatter gradients, roles reversed compared to forward pass
    recv_list = [None] * comm_size
    for r in range(comm_size):
        recv_sizes = [sizes[i][r] for i in range(comm_size)]
        recv_list[r] = scatter_v_wrapper(
            tensor, recv_sizes, dim=dim, src=r, group=group
        )
    tensor_to_recv = torch.cat(recv_list, dim=dim)

    # sum up gathered gradients and taking
    # care of precision handling as specified
    # by boolean flag
    tensor_shape[dim] = tensor_size_along_dim
    if use_fp32:
        out = torch.zeros(
            tensor_shape,
            dtype=torch.float32,
            device=tensor.device,
        )
        tensor_to_recv = tensor_to_recv.to(dtype=torch.float32)
    else:
        out = torch.zeros(
            tensor_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    out.index_add_(source=tensor_to_recv, index=indices, dim=dim)
    if use_fp32:
        out = out.to(tensor.dtype)

    return out
