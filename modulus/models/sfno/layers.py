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
import torch.fft
from torch.nn.modules.container import Sequential
from torch.utils.checkpoint import checkpoint
from torch.cuda import amp
import math

from torch_harmonics import *
from modulus.models.sfno.contractions import *
from modulus.models.sfno.activations import *
from modulus.models.sfno.initialization import trunc_normal_
from modulus.models.layers import get_activation


@torch.jit.script
def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:  # pragma: no cover
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).
    This is the same as the DropConnect impl for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in
    a separate paper. See discussion:
        https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    We've opted for changing the layer and argument names to 'drop path' rather than
    mix DropConnect as a layer name and use 'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2d ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual
    blocks).
    """

    def __init__(self, drop_prob=None):  # pragma: no cover
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):  # pragma: no cover
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    Divides the input image into patches and embeds them into a specified dimension
    using a convolutional layer.
    """

    def __init__(
        self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768
    ):  # pragma: no cover
        super(PatchEmbed, self).__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):  # pragma: no cover
        # gather input
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # new: B, C, H*W
        x = self.proj(x).flatten(2)
        return x


class EncoderDecoder(nn.Module):
    """
    Basic Encoder/Decoder
    """

    def __init__(
        self,
        num_layers,
        input_dim,
        output_dim,
        hidden_dim,
        act,
    ):  # pragma: no cover
        super(EncoderDecoder, self).__init__()

        encoder_modules = []
        current_dim = input_dim
        for i in range(num_layers):
            encoder_modules.append(nn.Conv2d(current_dim, hidden_dim, 1, bias=True))
            encoder_modules.append(get_activation(act))
            current_dim = hidden_dim
        encoder_modules.append(nn.Conv2d(current_dim, output_dim, 1, bias=False))
        self.fwd = nn.Sequential(*encoder_modules)

    def forward(self, x):
        return self.fwd(x)


class MLP(nn.Module):
    """
    Basic CNN with support for gradient checkpointing
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer="gelu",
        output_bias=True,
        drop_rate=0.0,
        checkpointing=0,
        **kwargs,
    ):  # pragma: no cover
        super(MLP, self).__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
        act = get_activation(act_layer)
        fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=output_bias)
        if drop_rate > 0.0:
            drop = nn.Dropout(drop_rate)
            self.fwd = nn.Sequential(fc1, act, drop, fc2, drop)
        else:
            self.fwd = nn.Sequential(fc1, act, fc2)

    @torch.jit.ignore
    def checkpoint_forward(self, x):  # pragma: no cover
        """Forward method with support for gradient checkpointing"""
        return checkpoint(self.fwd, x)

    def forward(self, x):  # pragma: no cover
        if self.checkpointing >= 2:
            return self.checkpoint_forward(x)
        else:
            return self.fwd(x)


class RealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):  # pragma: no cover
        super(RealFFT2, self).__init__()

        # use local FFT here
        self.fft_handle = torch.fft.rfft2

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

        self.truncate = True
        if (self.lmax == self.nlat) and (self.mmax == (self.nlon // 2 + 1)):
            self.truncate = False

        # self.num_batches = 1
        assert self.lmax % 2 == 0

    def forward(self, x):  # pragma: no cover
        y = self.fft_handle(x, (self.nlat, self.nlon), (-2, -1), "ortho")

        if self.truncate:
            y = torch.cat(
                (
                    y[..., : math.ceil(self.lmax / 2), : self.mmax],
                    y[..., -math.floor(self.lmax / 2) :, : self.mmax],
                ),
                dim=-2,
            )

        return y


class InverseRealFFT2(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None):  # pragma: no cover
        super(InverseRealFFT2, self).__init__()

        # use local FFT here
        self.ifft_handle = torch.fft.irfft2

        self.nlat = nlat
        self.nlon = nlon
        self.lmax = lmax or self.nlat
        self.mmax = mmax or self.nlon // 2 + 1

    def forward(self, x):  # pragma: no cover
        out = self.ifft_handle(x, (self.nlat, self.nlon), (-2, -1), "ortho")

        return out


class SpectralConv2d(nn.Module):
    """
    Spectral Convolution as utilized in
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        scale="auto",
        hard_thresholding_fraction=1,
        compression=None,
        rank=0,
        bias=False,
    ):  # pragma: no cover
        super(SpectralConv2d, self).__init__()

        if scale == "auto":
            scale = 1 / (in_channels * out_channels)

        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.contract_handle = _contract_diagonal

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.output_dims = (self.inverse_transform.nlat, self.inverse_transform.nlon)
        modes_lat = self.inverse_transform.lmax
        modes_lon = self.inverse_transform.mmax
        self.modes_lat = int(modes_lat * self.hard_thresholding_fraction)
        self.modes_lon = int(modes_lon * self.hard_thresholding_fraction)

        self.scale_residual = (
            self.forward_transform.nlat != self.inverse_transform.nlat
        ) or (self.forward_transform.nlon != self.inverse_transform.nlon)
        # new simple linear layer
        self.w = nn.Parameter(
            scale
            * torch.randn(in_channels, out_channels, self.modes_lat, self.modes_lon, 2)
        )
        # optional bias
        if bias:
            self.b = nn.Parameter(
                scale * torch.randn(1, out_channels, *self.output_dims)
            )

    def forward(self, x):  # pragma: no cover

        dtype = x.dtype
        B, C, H, W = x.shape

        if not self.scale_residual:
            residual = x

        with amp.autocast(enabled=False):
            x = x.to(torch.float32)
            x = self.forward_transform(x)
            if self.scale_residual:
                x = x.contiguous()
                residual = self.inverse_transform(x)
                residual = residual.to(dtype)
            x = torch.view_as_real(x)
            x = x.to(dtype)

        # do spectral conv
        modes = self.contract_handle(x, self.w)

        with amp.autocast(enabled=False):
            x = x.to(torch.float32)
            x = torch.view_as_complex(x)
            x = x.contiguous()
            x = self.inverse_transform(x)
            x = x.to(dtype)

        if hasattr(self, "b"):
            x = x + self.b

        return x, residual


class SpectralAttention2d(nn.Module):
    """
    2d Spectral Attention layer
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        sparsity_threshold=0.0,
        hidden_size_factor=2,
        use_complex_network=True,
        use_complex_kernels=False,
        complex_activation="real",
        bias=False,
        spectral_layers=1,
        drop_rate=0.0,
    ):  # pragma: no cover
        super(SpectralAttention2d, self).__init__()

        self.embed_dim = embed_dim
        self.sparsity_threshold = sparsity_threshold
        self.hidden_size = int(hidden_size_factor * self.embed_dim)
        self.scale = 0.02
        self.spectral_layers = spectral_layers
        self.mul_add_handle = (
            compl_muladd2d_fwd_c if use_complex_kernels else compl_muladd2d_fwd
        )
        self.mul_handle = compl_mul2d_fwd_c if use_complex_kernels else compl_mul2d_fwd

        self.modes_lat = forward_transform.lmax
        self.modes_lon = forward_transform.mmax

        # only storing the forward handle to be able to call it
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        assert inverse_transform.lmax == self.modes_lat
        assert inverse_transform.mmax == self.modes_lon

        self.scale_residual = (
            self.forward_transform.nlat != self.inverse_transform.nlat
        ) or (self.forward_transform.nlon != self.inverse_transform.nlon)

        # weights
        w = [self.scale * torch.randn(self.embed_dim, self.hidden_size, 2)]
        # w = [self.scale * torch.randn(self.embed_dim + 2*self.embed_freqs, self.hidden_size, 2)]
        # w = [self.scale * torch.randn(self.embed_dim + 4*self.embed_freqs, self.hidden_size, 2)]
        for l in range(1, self.spectral_layers):
            w.append(self.scale * torch.randn(self.hidden_size, self.hidden_size, 2))
        self.w = nn.ParameterList(w)

        if bias:
            self.b = nn.ParameterList(
                [
                    self.scale * torch.randn(self.hidden_size, 1, 2)
                    for _ in range(self.spectral_layers)
                ]
            )

        self.wout = nn.Parameter(
            self.scale * torch.randn(self.hidden_size, self.embed_dim, 2)
        )

        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self.activation = ComplexReLU(
            mode=complex_activation, bias_shape=(self.hidden_size, 1, 1)
        )

    def forward_mlp(self, xr):  # pragma: no cover
        """forward method for the MLP part of the network"""
        for l in range(self.spectral_layers):
            if hasattr(self, "b"):
                xr = self.mul_add_handle(
                    xr, self.w[l].to(xr.dtype), self.b[l].to(xr.dtype)
                )
            else:
                xr = self.mul_handle(xr, self.w[l].to(xr.dtype))
            xr = torch.view_as_complex(xr)
            xr = self.activation(xr)
            xr = self.drop(xr)
            xr = torch.view_as_real(xr)

        xr = self.mul_handle(xr, self.wout)

        return xr

    def forward(self, x):  # pragma: no cover

        dtype = x.dtype

        if not self.scale_residual:
            residual = x

        # FWD transform
        with amp.autocast(enabled=False):
            x = x.to(torch.float32)
            x = x.contiguous()
            x = self.forward_transform(x)
            if self.scale_residual:
                x = x.contiguous()
                residual = self.inverse_transform(x)
                residual = residual.to(dtype)
            x = torch.view_as_real(x)

        # MLP
        x = self.forward_mlp(x)

        # BWD transform
        with amp.autocast(enabled=False):
            x = torch.view_as_complex(x)
            x = x.contiguous()
            x = self.inverse_transform(x)
            x = x.to(dtype)

        return x, residual


class SpectralAttentionS2(nn.Module):
    """
    geometrical Spectral Attention layer
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        sparsity_threshold=0.0,
        hidden_size_factor=2,
        use_complex_network=True,
        complex_activation="real",
        bias=False,
        spectral_layers=1,
        drop_rate=0.0,
    ):  # pragma: no cover
        super(SpectralAttentionS2, self).__init__()

        self.embed_dim = embed_dim
        self.sparsity_threshold = sparsity_threshold
        self.hidden_size = int(hidden_size_factor * self.embed_dim)
        self.scale = 0.02
        # self.mul_add_handle = compl_muladd1d_fwd_c if use_complex_kernels else compl_muladd1d_fwd
        self.mul_add_handle = compl_muladd2d_fwd
        # self.mul_handle = compl_mul1d_fwd_c if use_complex_kernels else compl_mul1d_fwd
        self.mul_handle = compl_mul2d_fwd
        self.spectral_layers = spectral_layers

        self.modes_lat = forward_transform.lmax
        self.modes_lon = forward_transform.mmax

        # only storing the forward handle to be able to call it
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        assert inverse_transform.lmax == self.modes_lat
        assert inverse_transform.mmax == self.modes_lon

        self.scale_residual = (
            (self.forward_transform.nlat != self.inverse_transform.nlat)
            or (self.forward_transform.nlon != self.inverse_transform.nlon)
            or (self.forward_transform.grid != self.inverse_transform.grid)
        )
        # weights
        w = [self.scale * torch.randn(self.embed_dim, self.hidden_size, 2)]
        for l in range(1, self.spectral_layers):
            w.append(self.scale * torch.randn(self.hidden_size, self.hidden_size, 2))
        self.w = nn.ParameterList(w)

        if bias:
            self.b = nn.ParameterList(
                [
                    self.scale * torch.randn(2 * self.hidden_size, 1, 1, 2)
                    for _ in range(self.spectral_layers)
                ]
            )

        self.wout = nn.Parameter(
            self.scale * torch.randn(self.hidden_size, self.embed_dim, 2)
        )

        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self.activation = ComplexReLU(
            mode=complex_activation, bias_shape=(self.hidden_size, 1, 1)
        )

    def forward_mlp(self, xr):  # pragma: no cover
        """forward method for the MLP part of the network"""
        for l in range(self.spectral_layers):
            if hasattr(self, "b"):
                xr = self.mul_add_handle(
                    xr, self.w[l].to(xr.dtype), self.b[l].to(xr.dtype)
                )
            else:
                xr = self.mul_handle(xr, self.w[l].to(xr.dtype))
            xr = torch.view_as_complex(xr)
            xr = self.activation(xr)
            xr = self.drop(xr)
            xr = torch.view_as_real(xr)

        # final MLP
        xr = self.mul_handle(xr, self.wout)

        return xr

    def forward(self, x):  # pragma: no cover

        dtype = x.dtype

        if not self.scale_residual:
            residual = x

        # FWD transform
        with amp.autocast(enabled=False):
            x = x.to(torch.float32)
            x = x.contiguous()
            x = self.forward_transform(x)
            if self.scale_residual:
                x = x.contiguous()
                residual = self.inverse_transform(x)
                residual = residual.to(dtype)
            x = torch.view_as_real(x)

        # MLP
        x = self.forward_mlp(x)

        # BWD transform
        with amp.autocast(enabled=False):
            x = torch.view_as_complex(x)
            x = x.contiguous()
            x = self.inverse_transform(x)
            x = x.to(dtype)

        return x, residual
