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
from modulus.distributed.utils import _transpose, pad_helper, truncate_helper
from modulus.distributed.mappings import gather_from_spatial_parallel_region
from modulus.distributed.mappings import scatter_to_spatial_parallel_region


def conj_pad_helper_2d(tensor, pad_dim, other_dim, new_size):
    ndim = tensor.ndim
    pad_dim = (pad_dim + ndim) % ndim
    other_dim = (other_dim + ndim) % ndim

    # pad with conj
    orig_size = tensor.shape[pad_dim]
    tensor_pad = pad_helper(tensor, pad_dim, new_size, mode="conj")

    # gather
    tensor_pad_gather = gather_from_spatial_parallel_region(tensor_pad, dim=other_dim)

    # flip dims
    flip_slice = [
        slice(0, x)
        if ((idx != pad_dim) and (idx != other_dim))
        else slice(orig_size, new_size)
        if (idx == pad_dim)
        else slice(1, x)
        for idx, x in enumerate(tensor_pad_gather.shape)
    ]
    tensor_pad_gather[flip_slice] = torch.flip(
        tensor_pad_gather[flip_slice], dims=[other_dim]
    )

    # truncate:
    result = scatter_to_spatial_parallel_region(tensor_pad_gather, dim=other_dim)

    return result


# fft layers:
# input: dim[0] contig, dim[1] split
# output: dim[0] split, dim[1] contig
class distributed_rfft2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, dim, norm="ortho"):
        # NVTX marker
        torch.cuda.nvtx.range_push("distributed_rfft2.forward")

        # save:
        ctx.s = s
        ctx.dim = dim
        ctx.norm = norm

        torch.cuda.nvtx.range_push("distributed_rfft2.forward.fft1")
        # assume last dim is split (second to last is contiguous):
        x1 = torch.fft.fft(x, n=s[0], dim=dim[0], norm=norm)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("distributed_rfft2.forward.transpose")
        # transpose
        x1_recv, _ = _transpose(
            x1,
            dim[0],
            dim[1],
            group=DistributedManager().group("spatial_parallel"),
            async_op=False,
        )
        x1_tran = torch.cat(x1_recv, dim=dim[1])
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("distributed_rfft2.forward.fft2")
        # another fft:
        x2 = torch.fft.fft(x1_tran, n=s[1], dim=dim[1], norm=norm)
        torch.cuda.nvtx.range_pop()

        # truncate in last dim:
        ctx.last_dim_size = x2.shape[dim[1]]
        last_dim_size_trunc = ctx.last_dim_size // 2 + 1
        output = truncate_helper(x2, dim[1], last_dim_size_trunc)

        # pop range
        torch.cuda.nvtx.range_pop()

        return output

    @staticmethod
    def backward(ctx, grad_output):

        # load
        dim = ctx.dim
        norm = ctx.norm
        s = ctx.s
        last_dim_size = ctx.last_dim_size

        # pad the input to perform the backward fft
        g_pad = pad_helper(grad_output, dim[1], last_dim_size)

        # do fft
        g1 = torch.fft.ifft(g_pad, n=s[1], dim=dim[1], norm=norm)

        # transpose
        g1_recv, _ = _transpose(
            g1,
            dim[1],
            dim[0],
            group=DistributedManager().group("spatial_parallel"),
            async_op=False,
        )
        g1_tran = torch.cat(g1_recv, dim=dim[0])

        # now do the BW fft:
        grad_input = torch.real(torch.fft.ifft(g1_tran, n=s[0], dim=dim[0], norm=norm))

        return grad_input, None, None, None


# input: dim[0] split, dim[1] contig
# output: dim[0] contig, dim[1] split
class distributed_irfft2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, dim, norm="ortho"):

        # NVTX marker
        torch.cuda.nvtx.range_push("distributed_irfft2.forward")

        # save:
        ctx.s = s
        ctx.dim = dim
        ctx.norm = norm
        ctx.orig_dim_size = x.shape[dim[1]]

        if s is not None:
            first_dim_size = s[0]
            ctx.last_dim_size = s[1]
        else:
            first_dim_size = x.shape[dim[0]]
            ctx.last_dim_size = 2 * (ctx.orig_dim_size - 1)

        # fft in contig contig dim
        x_pad = conj_pad_helper_2d(x, dim[1], dim[0], ctx.last_dim_size)
        x1 = torch.fft.ifft(x_pad, n=ctx.last_dim_size, dim=dim[1], norm=norm)

        # transpose
        x1_recv, _ = _transpose(
            x1,
            dim[1],
            dim[0],
            group=DistributedManager().group("spatial_parallel"),
            async_op=False,
        )
        x1_tran = torch.cat(x1_recv, dim=dim[0])

        # ifft in contig dim
        x2 = torch.fft.ifft(x1_tran, n=first_dim_size, dim=dim[0], norm=norm)

        # take real part
        output = torch.real(x2).contiguous()

        # pop range
        torch.cuda.nvtx.range_pop()

        return output

    @staticmethod
    def backward(ctx, grad_output):

        # load
        dim = ctx.dim
        norm = ctx.norm
        last_dim_size = ctx.last_dim_size
        orig_dim_size = ctx.orig_dim_size

        # do fft
        g1 = torch.fft.fft(grad_output, dim=dim[0], norm=norm)

        # transpose
        g1_recv, _ = _transpose(
            g1,
            dim[0],
            dim[1],
            group=DistributedManager().group("spatial_parallel"),
            async_op=False,
        )
        g1_tran = torch.cat(g1_recv, dim=dim[1])

        # now do the BW fft:
        x2 = torch.fft.fft(g1_tran, dim=dim[1], norm=norm)

        # truncate
        grad_input = truncate_helper(x2, dim[1], orig_dim_size)

        return grad_input, None, None, None
