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
    AvgPool,
    BasicConvBlock,
    ConvGRUBlock,
    ConvNeXtBlock,
    DoubleConvNeXtBlock,
    Interpolate,
    MaxPool,
    SymmetricConvNeXtBlock,
    TransposedConvUpsample,  #
)


@pytest.fixture
def test_data():
    # create dummy data
    def generate_test_data(faces=12, channels=2, img_size=16, device="cpu"):
        test_data = torch.eye(img_size).to(device)
        test_data = test_data[(None,) * 2]
        test_data = test_data.expand([faces, channels, -1, -1])

        return test_data

    return generate_test_data


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_ConvGRUBlock_initialization(device, test_data):
    in_channels = 2
    conv_gru_func = ConvGRUBlock(in_channels=in_channels).to(device)
    assert isinstance(conv_gru_func, ConvGRUBlock)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_ConvGRUBlock_forward(device, test_data):
    in_channels = 2
    tensor_size = 16
    conv_gru_func = ConvGRUBlock(in_channels=in_channels).to(device)

    invar = test_data(img_size=tensor_size, device=device)

    out_shape = torch.Size([12, in_channels, tensor_size, tensor_size])

    outvar = conv_gru_func(invar)
    assert outvar.shape == out_shape

    # check if tracking history
    outvar_hist = conv_gru_func(invar)
    assert not common.compare_output(outvar_hist, outvar)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_ConvNeXtBlock_initialization(device):
    in_channels = 2
    convnext_block = ConvNeXtBlock(in_channels=in_channels).to(device)
    assert isinstance(convnext_block, ConvNeXtBlock)

    in_channels = 2
    out_channels = 2
    convnext_block = ConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert isinstance(convnext_block, ConvNeXtBlock)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_ConvNeXtBlock_forward(device, test_data):
    in_channels = 2
    out_channels = 1
    tensor_size = 16
    convnext_block = ConvNeXtBlock(in_channels=in_channels).to(device)

    invar = test_data(img_size=tensor_size, device=device)

    out_shape = torch.Size([12, 1, tensor_size, tensor_size])

    outvar = convnext_block(invar)
    assert outvar.shape == out_shape

    out_channels = 2
    convnext_block = ConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert outvar.shape == out_shape


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_DoubleConvNeXtBlock_initialization(device):
    in_channels = 2
    out_channels = 1
    latent_channels = 1
    doubleconvnextblock = DoubleConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
    ).to(device)
    assert isinstance(doubleconvnextblock, DoubleConvNeXtBlock)

    latent_channels = 2
    doubleconvnextblock = DoubleConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert isinstance(doubleconvnextblock, DoubleConvNeXtBlock)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_DoubleConvNeXtBlock_forward(device, test_data):
    in_channels = 2
    out_channels = 1
    latent_channels = 1
    tensor_size = 16
    doubleconvnextblock = DoubleConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
    ).to(device)

    invar = test_data(img_size=tensor_size, device=device)

    out_shape = torch.Size([12, 1, tensor_size, tensor_size])

    outvar = doubleconvnextblock(invar)
    assert outvar.shape == out_shape

    latent_channels = 2
    doubleconvnextblock = DoubleConvNeXtBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
    ).to(device)

    outvar = doubleconvnextblock(invar)
    assert outvar.shape == out_shape


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_SymmetricConvNeXtBlock_initialization(device):
    in_channels = 2
    latent_channels = 1
    symmetric_convnextblock = SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
    ).to(device)
    assert isinstance(symmetric_convnextblock, SymmetricConvNeXtBlock)

    latent_channels = 2
    symmetric_convnextblock = SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert isinstance(symmetric_convnextblock, SymmetricConvNeXtBlock)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_SymmetricConvNeXtBlock_forward(device, test_data):
    in_channels = 2
    latent_channels = 1
    tensor_size = 16
    symmetric_convnextblock = SymmetricConvNeXtBlock(
        in_channels=in_channels,
        latent_channels=latent_channels,
    ).to(device)

    invar = test_data(img_size=tensor_size, device=device)

    out_shape = torch.Size([12, 1, tensor_size, tensor_size])

    outvar = symmetric_convnextblock(invar)
    assert outvar.shape == out_shape

    latent_channels = 2
    outvar = symmetric_convnextblock(invar)
    assert outvar.shape == out_shape


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_BasicConvBlock_initialization(device):
    in_channels = 3
    out_channels = 1
    latent_channels = 2
    conv_block = BasicConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
    ).to(device)
    assert isinstance(conv_block, BasicConvBlock)

    # test w/ activation and latent channels
    conv_block = BasicConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        latent_channels=latent_channels,
        activation=torch.nn.ReLU(),
    ).to(device)
    assert isinstance(conv_block, BasicConvBlock)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_BasicConvBlock_forward(device, test_data):
    in_channels = 3
    out_channels = 1
    tensor_size = 16
    conv_block = BasicConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
    ).to(device)

    invar = test_data(
        channels=in_channels, faces=24, img_size=tensor_size, device=device
    )

    outvar = conv_block(invar)
    out_shape = torch.Size([24, out_channels, tensor_size, tensor_size])

    assert outvar.shape == out_shape


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_MaxPool_initialization(device):
    pooling = 2
    maxpool_block = MaxPool(pooling=pooling).to(device)
    assert isinstance(maxpool_block, MaxPool)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_MaxPool_forward(device, test_data):
    pooling = 2
    size = 16
    channels = 4
    maxpool_block = MaxPool(pooling=pooling).to(device)

    invar = test_data(
        faces=1, channels=channels, img_size=(size * pooling), device=device
    )
    outvar = test_data(faces=1, channels=channels, img_size=size, device=device)

    assert common.compare_output(outvar, maxpool_block(invar))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_AvgPool_initialization(device):
    pooling = 2
    avgpool_block = AvgPool(pooling=pooling).to(device)
    assert isinstance(avgpool_block, AvgPool)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_AvgPool_forward(device, test_data):
    pooling = 2
    size = 32
    channels = 4
    avgpool_block = AvgPool(pooling=pooling).to(device)

    invar = test_data(
        faces=1, channels=channels, img_size=(size * pooling), device=device
    )
    outvar = test_data(faces=1, channels=channels, img_size=size, device=device)

    # averaging across 1,0
    outvar = outvar * 0.5

    assert common.compare_output(outvar, avgpool_block(invar))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_TransposedConvUpsample_initialization(device):
    transposed_conv_upsample_block = TransposedConvUpsample().to(device)
    assert isinstance(transposed_conv_upsample_block, TransposedConvUpsample)

    transposed_conv_upsample_block = TransposedConvUpsample(
        activation=torch.nn.ReLU()
    ).to(device)
    assert isinstance(transposed_conv_upsample_block, TransposedConvUpsample)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_TransposedConvUpsample_forward(device, test_data):
    in_channels = 2
    out_channels = 1
    size = 16

    transposed_conv_upsample_block = TransposedConvUpsample(
        in_channels=in_channels,
        out_channels=out_channels,
    ).to(device)

    invar = test_data(faces=1, channels=in_channels, img_size=size, device=device)
    outsize = torch.Size([1, out_channels, size * 2, size * 2])

    outvar = transposed_conv_upsample_block(invar)
    assert outvar.shape == outsize

    transposed_conv_upsample_block = TransposedConvUpsample(
        activation=torch.nn.ReLU()
    ).to(device)

    invar = test_data(faces=1, channels=(in_channels + 1), img_size=size, device=device)
    outvar = transposed_conv_upsample_block(invar)
    assert outvar.shape == outsize


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_Interpolate_initialization(device):
    scale = 2
    mode = "linear"
    interpolation_block = Interpolate(scale_factor=scale, mode=mode).to(device)
    assert isinstance(interpolation_block, Interpolate)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_Interpolate_forward(device):
    scale = 2
    mode = "linear"
    interpolation_block = Interpolate(scale_factor=scale, mode=mode).to(device)

    tensor_size = torch.randint(low=2, high=4, size=(3,)).tolist()
    invar = torch.rand(tensor_size).to(device)

    outvar = torch.nn.functional.interpolate(
        invar,
        scale_factor=scale,
        mode=mode,
    ).to(device)

    assert common.compare_output(outvar, interpolation_block(invar))
