# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import common
import pytest
import torch

from modulus.models.dlwp_healpix_layers import (
    BasicConvBlock,  # for the output layer
    ConvGRUBlock,  # for the recurrent layer
    ConvNeXtBlock,  # for convolutional layer
    MaxPool,  # for downsampling
    TransposedConvUpsample,  # for upsampling
    UNetDecoder,
    UNetEncoder,
)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_UNetEncoder_initialize(device):
    channels = 2
    n_channels = (16, 32, 64)

    # Dicts for block configs used by encoder
    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": channels,
    }
    down_sampling_block = {
        "_target_": MaxPool,
        "pooling": 2,
    }

    encoder = UNetEncoder(
        conv_block=conv_block,
        down_sampling_block=down_sampling_block,
        n_channels=n_channels,
        input_channels=channels,
    ).to(device)
    assert isinstance(encoder, UNetEncoder)

    # with dilations
    encoder = UNetEncoder(
        conv_block=conv_block,
        down_sampling_block=down_sampling_block,
        n_channels=n_channels,
        input_channels=channels,
        dilations=(1, 1, 1),
    ).to(device)
    assert isinstance(encoder, UNetEncoder)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_UNetEncoder_forward(device):
    channels = 2
    hw_size = 16
    b_size = 12
    n_channels = (16, 32, 64)

    # Dicts for block configs used by encoder
    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": channels,
    }
    down_sampling_block = {
        "_target_": MaxPool,
        "pooling": 2,
    }

    encoder = UNetEncoder(
        conv_block=conv_block,
        down_sampling_block=down_sampling_block,
        n_channels=n_channels,
        input_channels=channels,
    ).to(device)

    tensor_size = [b_size, channels, hw_size, hw_size]
    invar = torch.rand(tensor_size).to(device)
    outvar = encoder(invar)

    # doesn't do anything
    encoder.reset()

    # outvar is a module list
    for idx, out_tensor in enumerate(outvar):
        # verify the channels and h dim are correct
        assert out_tensor.shape[1] == n_channels[idx]
        # default behaviour is to half the h/w size after first
        assert out_tensor.shape[2] == tensor_size[2] // (2**idx)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_UNetEncoder_reset(device):
    channels = 2
    n_channels = (16, 32, 64)

    # Dicts for block configs used by encoder
    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": channels,
    }
    down_sampling_block = {
        "_target_": MaxPool,
        "pooling": 2,
    }

    encoder = UNetEncoder(
        conv_block=conv_block,
        down_sampling_block=down_sampling_block,
        n_channels=n_channels,
        input_channels=channels,
    ).to(device)

    # doesn't do anything
    encoder.reset()
    assert isinstance(encoder, UNetEncoder)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_UNetDecoder_initilization(device):
    in_channels = 2
    out_channels = 1
    n_channels = (64, 32, 16)

    # Dicts for block configs used by decoder
    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": in_channels,
    }

    up_sampling_block = {
        "_target_": TransposedConvUpsample,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "upsampling": 2,
    }

    output_layer = {
        "_target_": BasicConvBlock,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1,
    }

    recurrent_block = {
        "_target_": ConvGRUBlock,
        "in_channels": 2,
        "kernel_size": 1,
    }

    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=recurrent_block,
        n_channels=n_channels,
    ).to(device)

    assert isinstance(decoder, UNetDecoder)

    # without the recurrent block and with dilations
    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=None,
        n_channels=n_channels,
        dilations=(1, 1, 1),
    ).to(device)
    assert isinstance(decoder, UNetDecoder)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_UNetDecoder_forward(device):
    in_channels = 2
    out_channels = 1
    hw_size = 32
    b_size = 12
    n_channels = (64, 32, 16)

    # Dicts for block configs used by decoder
    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": in_channels,
    }

    up_sampling_block = {
        "_target_": TransposedConvUpsample,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "upsampling": 2,
    }

    output_layer = {
        "_target_": BasicConvBlock,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1,
    }

    recurrent_block = {
        "_target_": ConvGRUBlock,
        "in_channels": 2,
        "kernel_size": 1,
    }

    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=recurrent_block,
        n_channels=n_channels,
    ).to(device)

    expected_size = torch.Size([b_size, out_channels, hw_size, hw_size])

    # build the list of tensors for the decoder
    invars = []
    # decoder has an algorithm that goes back to front
    for idx in range(len(n_channels) - 1, -1, -1):
        tensor_size = [b_size, n_channels[idx], hw_size, hw_size]
        invars.append(torch.rand(tensor_size).to(device))
        hw_size = hw_size // 2

    outvar = decoder(invars)
    assert outvar.shape == expected_size

    # make sure history is taken into account with ConvGRU
    outvar_hist = decoder(invars)
    assert not common.compare_output(outvar, outvar_hist)

    # check with no recurrent
    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=None,
        n_channels=n_channels,
        dilations=(1, 1, 1),
    ).to(device)

    outvar = decoder(invars)
    assert outvar.shape == expected_size


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_UNetDecoder_reset(device):
    in_channels = 2
    out_channels = 1
    hw_size = 32
    b_size = 12
    n_channels = (64, 32, 16)

    # Dicts for block configs used by decoder
    conv_block = {
        "_target_": ConvNeXtBlock,
        "in_channels": in_channels,
    }

    up_sampling_block = {
        "_target_": TransposedConvUpsample,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "upsampling": 2,
    }

    output_layer = {
        "_target_": BasicConvBlock,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": 1,
        "dilation": 1,
        "n_layers": 1,
    }

    recurrent_block = {
        "_target_": ConvGRUBlock,
        "in_channels": 2,
        "kernel_size": 1,
    }

    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=recurrent_block,
        n_channels=n_channels,
    ).to(device)

    # build the list of tensors for the decoder
    invars = []
    # decoder has an algorithm that goes back to front
    for idx in range(len(n_channels) - 1, -1, -1):
        tensor_size = [b_size, n_channels[idx], hw_size, hw_size]
        invars.append(torch.rand(tensor_size).to(device))
        hw_size = hw_size // 2

    outvar = decoder(invars)

    # make sure history is taken into account with ConvGRU
    outvar_hist = decoder(invars)
    assert not common.compare_output(outvar, outvar_hist)

    # make sure after reset we get the same result
    decoder.reset()
    outvar_reset = decoder(invars)
    assert common.compare_output(outvar, outvar_reset)

    # test reset without recurrent block
    decoder = UNetDecoder(
        conv_block=conv_block,
        up_sampling_block=up_sampling_block,
        output_layer=output_layer,
        recurrent_block=None,
        n_channels=n_channels,
    ).to(device)

    outvar = decoder(invars)

    # without the recurrent block should be the same
    outvar_hist = decoder(invars)
    assert common.compare_output(outvar, outvar_hist)

    # make sure after reset we get the same result
    decoder.reset()
    outvar_reset = decoder(invars)
    assert common.compare_output(outvar, outvar_reset)
