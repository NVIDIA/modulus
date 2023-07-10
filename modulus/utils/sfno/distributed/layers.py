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
import torch.nn as nn
import torch.nn.functional as F
from modulus.utils.sfno.distributed import comm

# matmul parallel
from modulus.utils.sfno.distributed.mappings import copy_to_matmul_parallel_region
from modulus.utils.sfno.distributed.mappings import reduce_from_matmul_parallel_region
from modulus.utils.sfno.distributed.mappings import scatter_to_matmul_parallel_region
from modulus.utils.sfno.distributed.mappings import gather_from_matmul_parallel_region

# spatial parallel
from modulus.utils.sfno.distributed.mappings import gather_from_spatial_parallel_region
from modulus.utils.sfno.distributed.mappings import scatter_to_spatial_parallel_region

from modulus.utils.sfno.distributed.helpers import _transpose

from modulus.models.sfno.initialization import trunc_normal_


class distributed_transpose_w(torch.autograd.Function):
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
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

        # frequency paddings
        ldist = (self.lmax + self.comm_size_h - 1) // self.comm_size_h
        self.lpad = ldist * self.comm_size_polar - self.lmax
        mdist = (self.mmax + self.comm_size_w - 1) // self.comm_size_w
        self.mpad = mdist * self.comm_size_w - self.mmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover

        # we need to ensure that we can split the channels evenly
        assert x.shape[1] % self.comm_size_h == 0
        assert x.shape[1] % self.comm_size_w == 0

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
        # ytt = yt[..., :self.nlat, :]

        # do second FFT:
        yo = torch.fft.fft(yt, n=self.nlat, dim=-2, norm="ortho")

        # pad if required, truncation is implicit
        yop = F.pad(yo, [0, 0, 0, self.lpad], mode="constant")

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
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

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
        assert x.shape[1] % self.comm_size_h == 0
        assert x.shape[1] % self.comm_size_w == 0

        # transpose: after that, channels are split, l is local:
        if self.comm_size_h > 1:
            xt = distributed_transpose_h.apply(x, (1, -2))
        else:
            xt = x

        # truncate
        xtt = xt[..., : self.lmax, :]

        # do first fft
        xf = torch.fft.ifft(xtt, n=self.nlat, dim=-2, norm="ortho")

        # transpose: after this, l is split and channels are local
        xfp = F.pad(xf, [0, 0, 0, self.latpad])

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
        xp = F.pad(x, [0, self.lonpad])

        # transpose: after this, m is split and channels are local
        if self.comm_size_w > 1:
            out = distributed_transpose_w.apply(xp, (-1, 1))
        else:
            out = xp

        return out


# more complicated layers
class DistributedMLP(nn.Module):
    """Distributed MLP layer"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        output_bias=True,
        act_layer=nn.GELU,
        drop_rate=0.0,
        checkpointing=False,
    ):  # pragma: no cover

        super(DistributedMLP, self).__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # get effective embedding size:
        comm_size = comm.get_size("matmul")
        assert (
            hidden_features % comm_size == 0
        ), "Error, hidden_features needs to be divisible by matmul_parallel_size"
        hidden_features_local = hidden_features // comm_size

        # first set of hp
        self.w1 = nn.Parameter(torch.ones(hidden_features_local, in_features, 1, 1))
        self.b1 = nn.Parameter(torch.zeros(hidden_features_local))

        # second set of hp
        self.w2 = nn.Parameter(torch.ones(out_features, hidden_features_local, 1, 1))

        if output_bias:
            self.b2 = nn.Parameter(torch.zeros(out_features))

        self.act = act_layer()
        self.drop = nn.Dropout(drop) if drop_rate > 0.0 else nn.Identity()

        # the weights are shared spatially
        self.w1.is_shared_mp = ["h", "w"]
        self.b1.is_shared_mp = ["h", "w"]
        self.w2.is_shared_mp = ["h", "w"]
        if output_bias:
            self.b2.is_shared_mp = [
                "matmul",
                "h",
                "w",
            ]  # this one is shared between all ranks

        # init weights
        self._init_weights()

    def _init_weights(self):  # pragma: no cover
        trunc_normal_(self.w1, std=0.02)
        nn.init.constant_(self.b1, 0.0)
        trunc_normal_(self.w2, std=0.02)
        if hasattr(self, "b2"):
            nn.init.constant_(self.b2, 0.0)

    def fwd(self, x):  # pragma: no cover
        """Forward function."""
        # we need to prepare paralellism here
        # spatial parallelism
        x = scatter_to_spatial_parallel_region(x, dim=-1)

        # prepare the matmul parallel part
        x = copy_to_matmul_parallel_region(x)

        # do the mlp
        x = F.conv2d(x, self.w1, bias=self.b1)
        x = self.act(x)
        x = self.drop(x)
        x = F.conv2d(x, self.w2, bias=None)
        x = reduce_from_matmul_parallel_region(x)
        if hasattr(self, "b2"):
            x = x + torch.reshape(self.b2, (1, -1, 1, 1))
        x = self.drop(x)

        # gather from spatial parallel region
        x = gather_from_spatial_parallel_region(x, dim=-1)

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
        assert (
            img_size[1] // patch_size[1]
        ) % spatial_comm_size == 0, (
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
            assert (
                embed_dim % matmul_comm_size == 0
            ), "Error, the embed_dim needs to be divisible by matmul_parallel_size"
            out_chans_local = embed_dim // matmul_comm_size
        else:
            out_chans_local = embed_dim

        # the weights  of this layer is shared across spatial parallel ranks
        self.proj = nn.Conv2d(
            in_chans, out_chans_local, kernel_size=patch_size, stride=patch_size
        )

        # make sure we reduce them across rank
        self.proj.weight.is_shared_mp = ["h", "w"]
        self.proj.bias.is_shared_mp = ["h", "w"]

    def forward(self, x):  # pragma: no cover
        if self.input_parallel:
            x = gather_from_matmul_parallel_region(x, dim=1)

        if self.output_parallel:
            x = copy_to_matmul_parallel_region(x)

        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
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


class DistributedAFNO2Dv2(nn.Module):
    """Distributed AFNO"""

    def __init__(
        self,
        hidden_size,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
        input_is_matmul_parallel=False,
        output_is_matmul_parallel=False,
        use_complex_kernels=False,
    ):  # pragma: no cover
        """Distributed AFNO2Dv2"""
        super(DistributedAFNO2Dv2, self).__init__()
        assert (
            hidden_size % num_blocks == 0
        ), f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        # get comm sizes:
        matmul_comm_size = comm.get_size("matmul")
        self.spatial_comm_size = comm.get_size("spatial")

        # select fft function handles
        if self.spatial_comm_size > 1:
            self.fft_handle = distributed_rfft2.apply
            self.ifft_handle = distributed_irfft2.apply
        else:
            self.fft_handle = torch.fft.rfft2
            self.ifft_handle = torch.fft.irfft2

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        assert (
            self.num_blocks % matmul_comm_size == 0
        ), "Error, num_blocks needs to be divisible by matmul_parallel_size"
        self.num_blocks_local = self.num_blocks // matmul_comm_size
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.mult_handle = (
            compl_mul_add_fwd_c if use_complex_kernels else compl_mul_add_fwd
        )

        # model paralellism
        self.input_is_matmul_parallel = input_is_matmul_parallel
        self.output_is_matmul_parallel = output_is_matmul_parallel

        # new
        # these weights need to be synced across all spatial ranks!
        self.w1 = nn.Parameter(
            self.scale
            * torch.randn(
                self.num_blocks_local,
                self.block_size,
                self.block_size * self.hidden_size_factor,
                2,
            )
        )
        self.b1 = nn.Parameter(
            self.scale
            * torch.randn(
                self.num_blocks_local,
                self.block_size * self.hidden_size_factor,
                1,
                1,
                2,
            )
        )
        self.w2 = nn.Parameter(
            self.scale
            * torch.randn(
                self.num_blocks_local,
                self.block_size * self.hidden_size_factor,
                self.block_size,
                2,
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.randn(self.num_blocks_local, self.block_size, 1, 1, 2)
        )

        # make sure we reduce them across rank
        self.w1.is_shared_mp = ["h", "w"]
        self.b1.is_shared_mp = ["h", "w"]
        self.w2.is_shared_mp = ["h", "w"]
        self.b2.is_shared_mp = ["h", "w"]

    def forward(self, x):  # pragma: no cover
        if not self.input_is_matmul_parallel:
            # distribute data
            x = scatter_to_matmul_parallel_region(x, dim=1)

        # bias
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W_local = x.shape
        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        H_local = H // self.spatial_comm_size
        W = W_local * self.spatial_comm_size
        x = self.fft_handle(x, (H, W), (-2, -1), "ortho")
        x = x.view(B, self.num_blocks_local, self.block_size, H_local, W // 2 + 1)

        # new
        x = torch.view_as_real(x)
        o2 = torch.zeros(x.shape, device=x.device)

        o1 = F.relu(
            self.mult_handle(
                x[
                    :,
                    :,
                    :,
                    total_modes - kept_modes : total_modes + kept_modes,
                    :kept_modes,
                    :,
                ],
                self.w1,
                self.b1,
            )
        )
        o2[
            :, :, :, total_modes - kept_modes : total_modes + kept_modes, :kept_modes, :
        ] = self.mult_handle(o1, self.w2, self.b2)

        # finalize
        x = F.softshrink(o2, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, H_local, W // 2 + 1)
        x = self.ifft_handle(x, (H, W), (-2, -1), "ortho")
        x = x.type(dtype) + bias

        # gather
        if not self.output_is_matmul_parallel:
            x = gather_from_matmul_parallel_region(x, dim=1)

        return x
