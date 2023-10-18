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

"""
Model architecture layers used in the paper "Elucidating the Design Space of 
Diffusion-Based Generative Models".
"""

from typing import Any, Dict, List

import numpy as np
import torch
from torch.nn.functional import silu

from modulus.models.diffusion import weight_init


class Linear(torch.nn.Module):
    """
    A fully connected (dense) layer implementation. The layer's weights and biases can
    be initialized using custom initialization strategies like "kaiming_normal",
    and can be further scaled by factors `init_weight` and `init_bias`.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        The biases of the layer. If set to `None`, the layer will not learn an additive
        bias. By default True.
    init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases. Supported modes
        are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
        By default "kaiming_normal".
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_mode: str = "kaiming_normal",
        init_weight: int = 1,
        init_bias: int = 0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(
            weight_init([out_features, in_features], **init_kwargs) * init_weight
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias)
            if bias
            else None
        )

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class Conv2d(torch.nn.Module):
    """
    A custom 2D convolutional layer implementation with support for up-sampling,
    down-sampling, and custom weight and bias initializations. The layer's weights
    and biases canbe initialized using custom initialization strategies like
    "kaiming_normal", and can be further scaled by factors `init_weight` and
    `init_bias`.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel : int
        Size of the convolving kernel.
    bias : bool, optional
        The biases of the layer. If set to `None`, the layer will not learn an
        additive bias. By default True.
    up : bool, optional
        Whether to perform up-sampling. By default False.
    down : bool, optional
        Whether to perform down-sampling. By default False.
    resample_filter : List[int], optional
        Filter to be used for resampling. By default [1, 1].
    fused_resample : bool, optional
        If True, performs fused up-sampling and convolution or fused down-sampling
        and convolution. By default False.
    init_mode : str, optional (default="kaiming_normal")
        init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases. Supported modes
        are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
        By default "kaiming_normal".
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.0.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        bias: bool = True,
        up: bool = False,
        down: bool = False,
        resample_filter: List[int] = [1, 1],
        fused_resample: bool = False,
        init_mode: str = "kaiming_normal",
        init_weight: float = 1.0,
        init_bias: float = 0.0,
    ):
        if up and down:
            raise ValueError("Both 'up' and 'down' cannot be true at the same time.")

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = (
            self.resample_filter.to(x.dtype)
            if self.resample_filter is not None
            else None
        )
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            x = torch.nn.functional.conv2d(
                x,
                f.tile([self.out_channels, 1, 1, 1]),
                groups=self.out_channels,
                stride=2,
            )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class GroupNorm(torch.nn.Module):
    """
    A custom Group Normalization layer implementation.

    Group Normalization (GN) divides the channels of the input tensor into groups and
    normalizes the features within each group independently. It does not require the
    batch size as in Batch Normalization, making itsuitable for batch sizes of any size
    or even for batch-free scenarios.

    Parameters
    ----------
    num_channels : int
        Number of channels in the input tensor.
    num_groups : int, optional
        Desired number of groups to divide the input channels, by default 32.
        This might be adjusted based on the `min_channels_per_group`.
    min_channels_per_group : int, optional
        Minimum channels required per group. This ensures that no group has fewer
        channels than this number. By default 4.
    eps : float, optional
        A small number added to the variance to prevent division by zero, by default
        1e-5.

    Notes
    -----
    If `num_channels` is not divisible by `num_groups`, the actual number of groups
    might be adjusted to satisfy the `min_channels_per_group` condition.
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        min_channels_per_group: int = 4,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )
        return x


class AttentionOp(torch.autograd.Function):
    """
    Attention weight computation, i.e., softmax(Q^T * K).
    Performs all computation using FP32, but uses the original datatype for
    inputs/outputs/gradients to conserve memory.
    """

    @staticmethod
    def forward(ctx, q, k):
        w = (
            torch.einsum(
                "ncq,nck->nqk",
                q.to(torch.float32),
                (k / np.sqrt(k.shape[1])).to(torch.float32),
            )
            .softmax(dim=2)
            .to(q.dtype)
        )
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(
            grad_output=dw.to(torch.float32),
            output=w.to(torch.float32),
            dim=2,
            input_dtype=torch.float32,
        )
        dq = torch.einsum("nck,nqk->ncq", k.to(torch.float32), db).to(
            q.dtype
        ) / np.sqrt(k.shape[1])
        dk = torch.einsum("ncq,nqk->nck", q.to(torch.float32), db).to(
            k.dtype
        ) / np.sqrt(k.shape[1])
        return dq, dk


class UNetBlock(torch.nn.Module):
    """
    Unified U-Net block with optional up/downsampling and self-attention. Represents
    the union of all features employed by the DDPM++, NCSN++, and ADM architectures.

    Parameters:
    -----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    emb_channels : int
        Number of embedding channels.
    up : bool, optional
        If True, applies upsampling in the forward pass. By default False.
    down : bool, optional
        If True, applies downsampling in the forward pass. By default False.
    attention : bool, optional
        If True, enables the self-attention mechanism in the block. By default False.
    num_heads : int, optional
        Number of attention heads. If None, defaults to `out_channels // 64`.
    channels_per_head : int, optional
        Number of channels per attention head. By default 64.
    dropout : float, optional
        Dropout probability. By default 0.0.
    skip_scale : float, optional
        Scale factor applied to skip connections. By default 1.0.
    eps : float, optional
        Epsilon value used for normalization layers. By default 1e-5.
    resample_filter : List[int], optional
        Filter for resampling layers. By default [1, 1].
    resample_proj : bool, optional
        If True, resampling projection is enabled. By default False.
    adaptive_scale : bool, optional
        If True, uses adaptive scaling in the forward pass. By default True.
    init : dict, optional
        Initialization parameters for convolutional and linear layers.
    init_zero : dict, optional
        Initialization parameters with zero weights for certain layers. By default
        {'init_weight': 0}.
    init_attn : dict, optional
        Initialization parameters specific to attention mechanism layers.
        Defaults to 'init' if not provided.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        up: bool = False,
        down: bool = False,
        attention: bool = False,
        num_heads: int = None,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        skip_scale: float = 1.0,
        eps: float = 1e-5,
        resample_filter: List[int] = [1, 1],
        resample_proj: bool = False,
        adaptive_scale: bool = True,
        init: Dict[str, Any] = dict(),
        init_zero: Dict[str, Any] = dict(init_weight=0),
        init_attn: Any = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not attention
            else num_heads
            if num_heads is not None
            else out_channels // channels_per_head
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            **init,
        )
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel=3, **init_zero
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                **init_zero,
            )

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(
            torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        )
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(
                    x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1
                )
                .unbind(2)
            )
            w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x


class PositionalEmbedding(torch.nn.Module):
    """
    A module for generating positional embeddings based on timesteps.
    This embedding technique is employed in the DDPM++ and ADM architectures.

    Parameters:
    -----------
    num_channels : int
        Number of channels for the embedding.
    max_positions : int, optional
        Maximum number of positions for the embeddings, by default 10000.
    endpoint : bool, optional
        If True, the embedding considers the endpoint. By default False.

    """

    def __init__(
        self, num_channels: int, max_positions: int = 10000, endpoint: bool = False
    ):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FourierEmbedding(torch.nn.Module):
    """
    Generates Fourier embeddings for timesteps, primarily used in the NCSN++
    architecture.

    This class generates embeddings by first multiplying input tensor `x` and
    internally stored random frequencies, and then concatenating the cosine and sine of
    the resultant.

    Parameters:
    -----------
    num_channels : int
        The number of channels in the embedding. The final embedding size will be
        2 * num_channels because of concatenation of cosine and sine results.
    scale : int, optional
        A scale factor applied to the random frequencies, controlling their range
        and thereby the frequency of oscillations in the embedding space. By default 16.
    """

    def __init__(self, num_channels: int, scale: int = 16):
        super().__init__()
        self.register_buffer("freqs", torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
