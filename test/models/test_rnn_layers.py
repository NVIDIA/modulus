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

import pytest
import torch

from modulus.models.rnn.layers import (
    _ConvLayer,
    _ConvResidualBlock,
    _TransposeConvLayer,
)


@pytest.mark.parametrize("activation_fn", [torch.nn.ReLU(), torch.nn.Identity()])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_conv_layer(activation_fn, stride, dimension):
    """Test conv layer"""

    in_channels = 16
    out_channels = 16
    kernel_size = 3

    layer = _ConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dimension=dimension,
        activation_fn=activation_fn,
    )

    bsize = 2
    fig_size = 18
    input_size = (bsize, in_channels) + dimension * (fig_size,)
    invar = torch.randn(size=input_size)
    outvar = layer(invar)

    if stride == 1:
        size_out = fig_size
    else:
        size_out = (fig_size - 1 * (kernel_size - 1) - 1) / stride + 1

    assert outvar.shape == (bsize, out_channels) + dimension * (size_out,)


@pytest.mark.parametrize("activation_fn", [torch.nn.ReLU(), torch.nn.Identity()])
@pytest.mark.parametrize("stride", [1, 3])
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_transconv_layer(activation_fn, stride, dimension):
    """Test transpose conv layer"""

    in_channels = 16
    out_channels = 16
    kernel_size = 3

    layer = _TransposeConvLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dimension=dimension,
        activation_fn=activation_fn,
    )

    bsize = 2
    fig_size = 18
    input_size = (bsize, in_channels) + dimension * (fig_size,)
    invar = torch.randn(size=input_size)
    outvar = layer(invar)

    if stride == 1:
        size_out = fig_size
    else:
        size_out = (fig_size - 1) * stride + 1 * (kernel_size - 1) + 1

    assert outvar.shape == (bsize, out_channels) + dimension * (size_out,)


@pytest.mark.parametrize("activation_fn", [torch.nn.ReLU(), torch.nn.Identity()])
@pytest.mark.parametrize("begin_activation_fn", [True, False])
@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("layer_normalization", [True, False])
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_residual_block_layer(
    activation_fn,
    begin_activation_fn,
    gated,
    layer_normalization,
    dimension,
):
    """Test residual block"""

    stride = 1
    in_channels = 16
    out_channels = 16

    # Just test constructor
    layer = _ConvResidualBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        dimension=dimension,
        gated=gated,
        layer_normalization=layer_normalization,
        activation_fn=activation_fn,
        begin_activation_fn=begin_activation_fn,
        stride=stride,
    )

    bsize = 2
    fig_size = 18
    input_size = (bsize, in_channels) + dimension * (fig_size,)
    invar = torch.randn(size=input_size)
    outvar = layer(invar)

    size_out = fig_size

    assert outvar.shape == (bsize, out_channels) + dimension * (size_out,)
