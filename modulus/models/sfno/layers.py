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
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modulus.models.layers.activations import get_activation


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
        if not (H == self.img_size[0] and W == self.img_size[1]):
            raise ValueError(
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            )
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

    def forward(self, x):  # pragma: no cover
        return self.fwd(x)


class MLP(nn.Module):
    """Basic CNN with support for gradient checkpointing."""

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
        self.lmax = min(lmax or self.nlat, self.nlat)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.nlon // 2 + 1)

        self.truncate = True
        if (self.lmax == self.nlat) and (self.mmax == (self.nlon // 2 + 1)):
            self.truncate = False

        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

    def forward(self, x):  # pragma: no cover
        y = self.fft_handle(
            x,
            s=(self.nlat, self.nlon),
            dim=(-2, -1),
            norm="ortho",
        )

        if self.truncate:
            y = torch.cat(
                (
                    y[..., : self.lmax_high, : self.mmax],
                    y[..., -self.lmax_low :, : self.mmax],
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
        self.lmax = min(lmax or self.nlat, self.nlat)
        self.mmax = min(mmax or self.nlon // 2 + 1, self.nlon // 2 + 1)

        self.truncate = True
        if (self.lmax == self.nlat) and (self.mmax == (self.nlon // 2 + 1)):
            self.truncate = False

        self.lmax_high = math.ceil(self.lmax / 2)
        self.lmax_low = math.floor(self.lmax / 2)

    def forward(self, x):  # pragma: no cover
        # truncation is implicit but better do it manually
        xt = x[..., : self.mmax]

        if self.truncate:
            # pad
            xth = xt[..., : self.lmax_high, :]
            xtl = xt[..., -self.lmax_low :, :]
            xthp = F.pad(xth, (0, 0, 0, self.nlat - self.lmax))
            xt = torch.cat([xthp, xtl], dim=-2)

        out = torch.fft.irfft2(xt, s=(self.nlat, self.nlon), dim=(-2, -1), norm="ortho")

        return out
