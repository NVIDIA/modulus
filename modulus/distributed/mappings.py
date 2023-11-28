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

from modulus.distributed.manager import DistributedManager
from modulus.distributed.utils import _gather, _reduce, _split


class _CopyToParallelRegion(torch.autograd.Function):
    """Pass the input to the parallel region"""

    @staticmethod
    def symbolic(graph, input_, group_):  # pragma: no cover
        return input_

    @staticmethod
    def forward(ctx, input_, group_):  # pragma: no cover
        ctx.group = group_
        return input_

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return _reduce(grad_output, group=DistributedManager().group(ctx.group)), None


class _ReduceFromParallelRegion(torch.autograd.Function):
    """All-reduce the input from the parallel region"""

    @staticmethod
    def symbolic(graph, input_, group_):  # pragma: no cover
        return _reduce(input_, group=DistributedManager().group(group_))

    @staticmethod
    def forward(ctx, input_, group_):  # pragma: no cover
        return _reduce(input_, group=DistributedManager().group(group_))

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return grad_output, None


class _ScatterToParallelRegion(torch.autograd.Function):
    """Split the input and keep only the chunk corresponding to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim_, group_):  # pragma: no cover
        return _split(input_, dim_, group=DistributedManager().group(group_))

    @staticmethod
    def forward(ctx, input_, dim_, group_):  # pragma: no cover
        ctx.dim = dim_
        ctx.group = group_
        return _split(input_, dim_, group=DistributedManager().group(group_))

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return (
            _gather(grad_output, ctx.dim, group=DistributedManager().group(ctx.group_)),
            None,
            None,
        )


class _GatherFromParallelRegion(torch.autograd.Function):
    """Gather the input from parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_, dim_, group_):  # pragma: no cover
        return _gather(input_, dim_, group=DistributedManager().group(group_))

    @staticmethod
    def forward(ctx, input_, dim_, group_):  # pragma: no cover
        ctx.dim = dim_
        ctx.group = group_
        return _gather(input_, dim_, group=DistributedManager().group(group_))

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        return (
            _split(grad_output, ctx.dim, group=DistributedManager().group(ctx.group)),
            None,
            None,
        )


class _GatherWithinParallelRegion(torch.autograd.Function):
    """
    Gather the input within parallel region and concatenate.
    The same forward method as _GatherFromParallelRegion, the difference is only in the
    backward pass. This method performs a reduction of the gradients before the split in
    the backward pass while the other version only performs a split
    """

    @staticmethod
    def symbolic(graph, input_, dim_, group_):  # pragma: no cover
        return _gather(input_, dim_, group=DistributedManager().group(group_))

    @staticmethod
    def forward(ctx, input_, dim_, group_):  # pragma: no cover
        ctx.dim = dim_
        ctx.group = group_
        return _gather(input_, dim_, group=DistributedManager().group(group_))

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        red = _reduce(grad_output, group=DistributedManager().group(ctx.group_))
        return (
            _split(red, ctx.dim, group=DistributedManager().group(ctx.group)),
            None,
            None,
        )


# -----------------
# Helper functions.
# -----------------
def copy_to_parallel_region(input, group):  # pragma: no cover
    """Copy input"""
    return _CopyToParallelRegion.apply(input, group)


def reduce_from_parallel_region(input, group):  # pragma: no cover
    """All-reduce the input from the matmul parallel region."""
    return _ReduceFromParallelRegion.apply(input, group)


def scatter_to_parallel_region(input, dim, group):  # pragma: no cover
    """Split the input and keep only the corresponding chuck to the rank."""
    return _ScatterToParallelRegion.apply(input, dim, group)


def gather_from_parallel_region(input, dim, group):  # pragma: no cover
    """Gather the input from matmul parallel region and concatenate."""
    return _GatherFromParallelRegion.apply(input, dim, group)


def gather_within_parallel_region(input, dim, group):  # pragma: no cover
    """
    Gather the input within parallel region and concatenate.
    The same forward method as gather_from_parallel_region, the difference is only in
    the backward pass. This method performs a reduction of the gradients before the
    split in the backward pass while the other version only performs a split
    """
    return _GatherWithinParallelRegion.apply(input, dim, group)
