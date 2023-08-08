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
from modulus.distributed.utils import _reduce
from modulus.distributed.utils import _split
from modulus.distributed.utils import _gather


# matmul parallel
class _CopyToMatmulParallelRegion(torch.autograd.Function):
    """Pass the input to the matmul parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, group=DistributedManager().group("model_parallel"))


class _ReduceFromMatmulParallelRegion(torch.autograd.Function):
    """All-reduce the input from the matmul parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_, group=DistributedManager().group("model_parallel"))

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_, group=DistributedManager().group("model_parallel"))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToMatmulParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, dim_):
        return _split(input_, dim_, group=DistributedManager().group("model_parallel"))

    @staticmethod
    def forward(ctx, input_, dim_):
        ctx.dim = dim_
        return _split(input_, dim_, group=DistributedManager().group("model_parallel"))

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _gather(
                grad_output, ctx.dim, group=DistributedManager().group("model_parallel")
            ),
            None,
        )


class _GatherFromMatmulParallelRegion(torch.autograd.Function):
    """Gather the input from matmul parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, dim_):
        return _gather(input_, dim_, group=DistributedManager().group("model_parallel"))

    @staticmethod
    def forward(ctx, input_, dim_):
        ctx.dim = dim_
        return _gather(input_, dim_, group=DistributedManager().group("model_parallel"))

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _split(
                grad_output, ctx.dim, group=DistributedManager().group("model_parallel")
            ),
            None,
        )


class _GatherWithinMatmulParallelRegion(torch.autograd.Function):
    """Gather the input from matmul parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, dim_):
        return _gather(input_, dim_, group=DistributedManager().group("model_parallel"))

    @staticmethod
    def forward(ctx, input_, dim_):
        ctx.dim = dim_
        return _gather(input_, dim_, group=DistributedManager().group("model_parallel"))

    @staticmethod
    def backward(ctx, grad_output):
        red = _reduce(grad_output, group=DistributedManager().group("model_parallel"))
        return (
            _split(red, ctx.dim, group=DistributedManager().group("model_parallel")),
            None,
        )


# -----------------
# Helper functions.
# -----------------
# matmul parallel
def copy_to_matmul_parallel_region(input_):
    """Copy input"""
    return _CopyToMatmulParallelRegion.apply(input_)


def reduce_from_matmul_parallel_region(input_):
    """All-reduce the input from the matmul parallel region."""
    return _ReduceFromMatmulParallelRegion.apply(input_)


def scatter_to_matmul_parallel_region(input_, dim):
    """Split the input and keep only the corresponding chuck to the rank."""
    return _ScatterToMatmulParallelRegion.apply(input_, dim)


def gather_from_matmul_parallel_region(input_, dim):
    """Gather the input from matmul parallel region and concatenate."""
    return _GatherFromMatmulParallelRegion.apply(input_, dim)


def gather_within_matmul_parallel_region(input_, dim):
    """Gather the input from matmul parallel region and concatenate."""
    return _GatherWithinMatmulParallelRegion.apply(input_, dim)
