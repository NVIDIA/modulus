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

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modulus.utils.sfno.distributed import comm
from modulus.utils.sfno.distributed.helpers import _transpose

# matmul parallel
from modulus.utils.sfno.distributed.mappings import (
    copy_to_parallel_region,
    gather_from_parallel_region,
    reduce_from_parallel_region,
)


class distributed_transpose_w(torch.autograd.Function):  # pragma: no cover
    """Distributed transpose"""

    @staticmethod
    def forward(ctx, x, dim):  # pragma: no cover
        xlist, _ = _transpose(x, dim[0], dim[1], group=comm.get_group("w"))
        x = torch.cat(xlist, dim=dim[1])
        ctx.dim = dim
        return x

    @staticmethod
    def backward(ctx, go):  # pragma: no cover
        dim = ctx.dim
        gilist, _ = _transpose(go, dim[1], dim[0], group=comm.get_group("w"))
        gi = torch.cat(gilist, dim=dim[0])
        return gi, None


class distributed_transpose_h(torch.autograd.Function):
    """Distributed transpose"""

    @staticmethod
    def forward(ctx, x, dim):  # pragma: no cover
        xlist, _ = _transpose(x, dim[0], dim[1], group=comm.get_group("h"))
        x = torch.cat(xlist, dim=dim[1])
        ctx.dim = dim
        return x

    @staticmethod
    def backward(ctx, go):  # pragma: no cover
        dim = ctx.dim
        gilist, _ = _transpose(go, dim[1], dim[0], group=comm.get_group("h"))
        gi = torch.cat(gilist, dim=dim[0])
        return gi, None


class DistributedRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):  # pragma: no cover
        super(DistributedRealFFT2, self).__init__()

        # get the comms grid:
        self.comm_size_h = comm.get_size("h")
        self.comm_size_w = comm.get_size("w")
        self.comm_rank_w = comm.get_rank("w")

        # dimensions
        self.nlat = nlat
        self.nlon = nlon
        self.lmax = min(lmax or self.nlat, self.nlat)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.nlon // 2 + 1)

        # compute half modes
        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

        # frequency paddings
        ldist = (self.lmax + self.comm_size_h - 1) // self.comm_size_h
        self.lpad = ldist * self.comm_size_h - self.lmax
        mdist = (self.mmax + self.comm_size_w - 1) // self.comm_size_w
        self.mpad = mdist * self.comm_size_w - self.mmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover

        # we need to ensure that we can split the channels evenly
        if not x.shape[1] % self.comm_size_h == 0:
            raise AssertionError
        if not x.shape[1] % self.comm_size_w == 0:
            raise AssertionError

        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_w > 1:
            xt = distributed_transpose_w.apply(x, (1, -1))
        else:
            xt = x

        # do first FFT
        xtf = torch.fft.rfft(xt, n=self.nlon, dim=-1, norm="ortho")

        # truncate
        xtft = xtf[..., : self.mmax]

        # pad the dim to allow for splitting
        xtfp = F.pad(xtft, [0, self.mpad], mode="constant")

        # transpose: after this, m is split and c is local
        if self.comm_size_w > 1:
            y = distributed_transpose_w.apply(xtfp, (-1, 1))
        else:
            y = xtfp

        # transpose: after this, c is split and h is local
        if self.comm_size_h > 1:
            yt = distributed_transpose_h.apply(y, (1, -2))
        else:
            yt = y

        # the input data might be padded, make sure to truncate to nlat:
        ytt = yt[..., : self.nlat, :].contiguous()

        # do second FFT:
        yo = torch.fft.fft(ytt, n=self.nlat, dim=-2, norm="ortho")

        # apply mode truncation:
        yot = torch.cat(
            [yo[..., : self.lmax_high, :], yo[..., -self.lmax_low :, :]], dim=-2
        )

        # pad if required, truncation is implicit
        yop = F.pad(yot, (0, 0, 0, self.lpad), mode="constant")

        # transpose: after this, l is split and c is local
        if self.comm_size_h > 1:
            y = distributed_transpose_h.apply(yop, (-2, 1))
        else:
            y = yop

        return y


class DistributedInverseRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):  # pragma: no cover
        super(DistributedInverseRealFFT2, self).__init__()

        # get the comms grid:
        self.comm_size_h = comm.get_size("h")
        self.comm_size_w = comm.get_size("w")
        self.comm_rank_w = comm.get_rank("w")

        # dimensions
        self.nlat = nlat
        self.nlon = nlon
        self.lmax = min(lmax or self.nlat, self.nlat)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.nlon // 2 + 1)

        # compute half modes
        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

        # spatial paddings
        latdist = (self.nlat + self.comm_size_h - 1) // self.comm_size_h
        self.latpad = latdist * self.comm_size_h - self.nlat
        londist = (self.nlon + self.comm_size_w - 1) // self.comm_size_w
        self.lonpad = londist * self.comm_size_w - self.nlon

        # frequency paddings
        ldist = (self.lmax + self.comm_size_h - 1) // self.comm_size_h
        self.lpad = ldist * self.comm_size_h - self.lmax
        mdist = (self.mmax + self.comm_size_w - 1) // self.comm_size_w
        self.mpad = mdist * self.comm_size_w - self.mmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover

        # we need to ensure that we can split the channels evenly
        if not x.shape[1] % self.comm_size_h == 0:
            raise AssertionError
        if not x.shape[1] % self.comm_size_w == 0:
            raise AssertionError

        # transpose: after that, channels are split, l is local:
        if self.comm_size_h > 1:
            xt = distributed_transpose_h.apply(x, (1, -2))
        else:
            xt = x

        # truncate: assumes that data was padded at the END!
        # this is compatible with the forward transform and easier to handle
        # in a distributed setting
        xtt = xt[..., : self.lmax, :]

        # we should pad the middle here manually, so that the inverse FFT is correct
        # EXPERIMENTAL
        if self.lmax < self.nlat:
            xtth = xtt[..., : self.lmax_high, :]
            xttl = xtt[..., -self.lmax_low, :]
            xtthp = F.pad(xtth, (0, 0, 0, self.nlat - self.lmax), mode="constant")
            xtt = torch.cat([xtthp, xttl], dim=-2)

        # do first fft
        xf = torch.fft.ifft(xtt, n=self.nlat, dim=-2, norm="ortho")

        # transpose: after this, l is split and channels are local
        xfp = F.pad(xf, (0, 0, 0, self.latpad), mode="constant")

        if self.comm_size_h > 1:
            y = distributed_transpose_h.apply(xfp, (-2, 1))
        else:
            y = xfp

        # transpose: after this, channels are split and m is local
        if self.comm_size_w > 1:
            yt = distributed_transpose_w.apply(y, (1, -1))
        else:
            yt = y

        # truncate
        ytt = yt[..., : self.mmax]

        # apply the inverse (real) FFT
        x = torch.fft.irfft(ytt, n=self.nlon, dim=-1, norm="ortho")

        # pad before we transpose back
        xp = F.pad(x, (0, self.lonpad), mode="constant")

        # transpose: after this, m is split and channels are local
        if self.comm_size_w > 1:
            out = distributed_transpose_w.apply(xp, (-1, 1))
        else:
            out = xp

        return out


class _DistMatmulHelper(torch.autograd.Function):
    """Distributed matrix multiply helper"""

    @staticmethod
    def forward(
        ctx, X, weight, bias, inp_group_name, out_group_name
    ):  # pragma: no cover

        # store some variables
        ctx.save_for_backward(X, weight, bias)
        ctx.out_group_name = out_group_name

        # matrix multiplication
        xconv = F.conv2d(X, weight, bias=None)

        # reduce
        if comm.get_size(inp_group_name) > 1:
            dist.all_reduce(xconv, group=comm.get_group(inp_group_name))

        # add bias
        if bias is not None:
            xconvbias = xconv + bias
        else:
            xconvbias = xconv

        return xconvbias

    @staticmethod
    def backward(ctx, grad_out):  # pragma: no cover
        X, weight, bias = ctx.saved_tensors
        gname = ctx.out_group_name

        # do the bwd pass on dgrad
        grad_input = F.conv_transpose2d(grad_out, weight, bias=None)

        # reduce across nodes
        if comm.get_size(gname) > 1:
            dgrad_handle = dist.all_reduce(
                grad_input, group=comm.get_group(gname), async_op=True
            )

        # weight grad
        grad_weight = F.conv2d(
            X.transpose(0, 1), grad_out.transpose(0, 1), bias=None
        ).transpose(0, 1)

        if bias is not None:
            grad_bias = torch.sum(grad_out, dim=(0, 2, 3), keepdim=True)
        else:
            grad_bias = None

        if comm.get_size(gname) > 1:
            dgrad_handle.wait()

        return grad_input, grad_weight, grad_bias, None, None


class DistributedMatmul(nn.Module):
    """Distributed Matrix Multiply"""

    def __init__(
        self,
        inp_dim,
        out_dim,
        kernel_size=1,
        comm_inp_name="fin",
        comm_out_name="fout",
        bias=True,
    ):  # pragma: no cover
        super(DistributedMatmul, self).__init__()

        # get sizes
        self.comm_inp_name = comm_inp_name
        self.comm_out_name = comm_out_name
        comm_inp_size = comm.get_size(self.comm_inp_name)
        comm_out_size = comm.get_size(self.comm_out_name)

        # split:
        if not (kernel_size == 1):
            raise ValueError("Error, only pointwise operations are currently supported")
        if not (inp_dim % comm_inp_size == 0):
            raise ValueError(
                f"Error, the size of input feature dim ({inp_dim}) has to be \
                evenly divisible by the input feature comm dim ({comm_inp_size})"
            )
        if not (out_dim % comm_out_size == 0):
            raise ValueError(
                f"Error, the size of output feature dim ({out_dim}) has to be \
                evenly divisible by the output feature comm dim ({comm_out_size})"
            )

        # compute reduced dims
        inp_dim_local = inp_dim // comm_inp_size
        out_dim_local = out_dim // comm_out_size

        # parameters
        self.weight = nn.Parameter(
            torch.ones(out_dim_local, inp_dim_local, kernel_size, kernel_size)
        )
        self.weight.is_shared_mp = ["spatial"]
        self.weight.sharded_dims_mp = [
            self.comm_out_name,
            self.comm_inp_name,
            None,
            None,
        ]
        if bias:
            self.bias = nn.Parameter(torch.ones(1, out_dim_local, 1, 1))
            self.bias.is_shared_mp = ["spatial"]
            self.bias.sharded_dims_mp = [None, self.comm_out_name, None, None]

    def forward(self, x):  # pragma: no cover
        x_cp = copy_to_parallel_region(x, self.comm_out_name)
        x_loc = F.conv2d(x_cp, self.weight, bias=None)
        x_out = reduce_from_parallel_region(x_loc, self.comm_inp_name)
        if hasattr(self, "bias"):
            x_out = x_out + self.bias

        return x_out


# distributed encoder/decoder
class DistributedEncoderDecoder(nn.Module):
    """Distributed Encoder/Decoder"""

    def __init__(
        self,
        num_layers,
        input_dim,
        output_dim,
        hidden_dim,
        act,
        comm_inp_name="fin",
        comm_out_name="fout",
    ):  # pragma: no cover
        super(DistributedEncoderDecoder, self).__init__()

        # get list of modules
        encoder_modules = []
        current_dim = input_dim
        comm_inp_name_tmp = comm_inp_name
        comm_out_name_tmp = comm_out_name
        for i in range(num_layers - 1):
            encoder_modules.append(
                DistributedMatmul(
                    current_dim,
                    hidden_dim,
                    1,
                    comm_inp_name=comm_inp_name_tmp,
                    comm_out_name=comm_out_name_tmp,
                    bias=True,
                )
            )
            encoder_modules.append(act())
            current_dim = hidden_dim
            comm_inp_name_tmp, comm_out_name_tmp = (
                comm_out_name_tmp,
                comm_inp_name_tmp,
            )

        # final layer
        encoder_modules.append(
            DistributedMatmul(
                current_dim,
                output_dim,
                1,
                comm_inp_name=comm_inp_name_tmp,
                comm_out_name=comm_out_name_tmp,
                bias=False,
            )
        )

        # create fwd sequence
        self.fwd = nn.Sequential(*encoder_modules)

        # store the comm names for in and out so that they can be queried
        self.comm_inp_name = comm_inp_name
        self.comm_out_name = comm_out_name_tmp

    def forward(self, x):  # pragma: no cover
        return self.fwd(x)


# more complicated layers
class DistributedMLP(nn.Module):
    """Distributed MLP layer"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        output_bias=True,
        comm_inp_name="fin",
        comm_hidden_name="fout",
        act_layer=nn.GELU,
        drop_rate=0.0,
        checkpointing=False,
    ):  # pragma: no cover

        super(DistributedMLP, self).__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = DistributedMatmul(
            in_features,
            hidden_features,
            1,
            comm_inp_name=comm_inp_name,
            comm_out_name=comm_hidden_name,
            bias=True,
        )

        self.fc2 = DistributedMatmul(
            hidden_features,
            out_features,
            1,
            comm_inp_name=comm_hidden_name,
            comm_out_name=comm_inp_name,
            bias=output_bias,
        )

        self.act = act_layer()
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def fwd(self, x):  # pragma: no cover
        # do the mlp
        # first layer
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        # second layer
        x = self.fc2(x)
        x = self.drop(x)

        return x

    @torch.jit.ignore
    def _checkpoint_forward(self, x):  # pragma: no cover
        return checkpoint(self.fwd, x)

    def forward(self, x):  # pragma: no cover
        if self.checkpointing:
            return self._checkpoint_forward(x)
        else:
            return self.fwd(x)


class DistributedPatchEmbed(nn.Module):
    """Distributed patch embedding layer"""

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=(16, 16),
        in_chans=3,
        embed_dim=768,
        input_is_matmul_parallel=False,
        output_is_matmul_parallel=True,
    ):  # pragma: no cover

        super(DistributedPatchEmbed, self).__init__()

        # store params
        self.input_parallel = input_is_matmul_parallel
        self.output_parallel = output_is_matmul_parallel

        # get comm sizes:
        matmul_comm_size = comm.get_size("matmul")
        spatial_comm_size = comm.get_size("spatial")

        # compute parameters
        if (img_size[1] // patch_size[1]) % spatial_comm_size != 0:
            raise ValueError(
                "Error, make sure that the spatial comm size evenly divides patched W"
            )
        num_patches = ((img_size[1] // patch_size[1]) // spatial_comm_size) * (
            img_size[0] // patch_size[0]
        )
        self.img_size = (img_size[0], img_size[1] // spatial_comm_size)
        self.patch_size = patch_size
        self.num_patches = num_patches

        # get effective embedding size:
        if self.output_parallel:
            if not (embed_dim % matmul_comm_size == 0):
                ValueError(
                    "Error, the embed_dim needs to be divisible by matmul_parallel_size"
                )
            out_chans_local = embed_dim // matmul_comm_size
        else:
            out_chans_local = embed_dim

        # the weights  of this layer is shared across spatial parallel ranks
        self.proj = nn.Conv2d(
            in_chans, out_chans_local, kernel_size=patch_size, stride=patch_size
        )

        # make sure we reduce them across rank
        self.proj.weight.is_shared_mp = ["spatial"]
        self.proj.bias.is_shared_mp = ["spatial"]

    def forward(self, x):  # pragma: no cover
        if self.input_parallel:
            x = gather_from_parallel_region(x, 1, "matmul")

        if self.output_parallel:
            x = copy_to_parallel_region(x, "matmul")

        B, C, H, W = x.shape
        if not (H == self.img_size[0] and W == self.img_size[1]):
            raise ValueError(
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            )
        # new: B, C, H*W
        x = self.proj(x).flatten(2)
        return x


@torch.jit.script
def compl_mul_add_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """complex multiplication and addition"""
    tmp = torch.einsum("bkixys,kiot->stbkoxy", a, b)
    res = (
        torch.stack(
            [tmp[0, 0, ...] - tmp[1, 1, ...], tmp[1, 0, ...] + tmp[0, 1, ...]], dim=-1
        )
        + c
    )
    return res


@torch.jit.script
def compl_mul_add_fwd_c(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """Performs a complex multiplication and addition operation on three tensors"""
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    cc = torch.view_as_complex(c)
    tmp = torch.einsum("bkixy,kio->bkoxy", ac, bc)
    res = tmp + cc
    return torch.view_as_real(res)
