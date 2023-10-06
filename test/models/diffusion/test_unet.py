# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ruff: noqa: E402
import os
import random
import sys

import numpy as np
import pytest
import torch

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import common

from modulus.models.diffusion import SongUNet as UNet


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_song_unet_forward(device):
    torch.manual_seed(0)
    # Construct the DDM++ UNet model
    model = UNet(img_resolution=64, in_channels=2, out_channels=2).to(device)
    input_image = torch.ones([1, 2, 64, 64])
    noise_labels = noise_labels = torch.randn([1])
    class_labels = torch.randint(0, 1, (1, 1))

    assert common.validate_forward_accuracy(
        model,
        (input_image, noise_labels, class_labels),
        file_name=f"ddmpp_unet_output.pth",
        atol=1e-3,
    )

    torch.manual_seed(0)
    # Construct the NCSN++ UNet model
    model = UNet(
        img_resolution=64,
        in_channels=2,
        out_channels=2,
        embedding_type="fourier",
        channel_mult_noise=2,
        encoder_type="residual",
        resample_filter=[1, 3, 3, 1],
    ).to(device)

    assert common.validate_forward_accuracy(
        model,
        (input_image, noise_labels, class_labels),
        file_name=f"ncsnpp_unet_output.pth",
        atol=1e-3,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_song_unet_constructor(device):
    """Test the Song Unet constructor options"""

    # DDM++
    img_resolution = 16
    in_channels = 2
    out_channels = 2
    model = UNet(
        img_resolution=img_resolution,
        in_channels=in_channels,
        out_channels=out_channels,
    ).to(device)
    noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    input_image = torch.ones([1, 2, 16, 16]).to(device)
    output_image = model(input_image, noise_labels, class_labels)
    assert output_image.shape == (1, out_channels, img_resolution, img_resolution)

    # NCSN++
    model = UNet(
        img_resolution=img_resolution,
        in_channels=in_channels,
        out_channels=out_channels,
        embedding_type="fourier",
        channel_mult_noise=2,
        encoder_type="residual",
        resample_filter=[1, 3, 3, 1],
    ).to(device)
    noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    input_image = torch.ones([1, 2, 16, 16]).to(device)
    output_image = model(input_image, noise_labels, class_labels)
    assert output_image.shape == (1, out_channels, img_resolution, img_resolution)

    # Also test failure cases
    try:
        model = UNet(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            embedding_type=None,
        ).to(device)
        raise AssertionError("Failed to error for invalid argument")
    except ValueError:
        pass

    try:
        model = UNet(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_type=None,
        ).to(device)
        raise AssertionError("Failed to error for invalid argument")
    except ValueError:
        pass

    try:
        model = UNet(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            decoder_type=None,
        ).to(device)
        raise AssertionError("Failed to error for invalid argument")
    except ValueError:
        pass


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_song_unet_optims(device):
    """Test Song UNet optimizations"""

    def setup_model():
        model = UNet(
            img_resolution=64,
            in_channels=2,
            out_channels=2,
            embedding_type="fourier",
            channel_mult_noise=2,
            encoder_type="residual",
            resample_filter=[1, 3, 3, 1],
        ).to(device)
        noise_labels = torch.randn([1]).to(device)
        class_labels = torch.randint(0, 1, (1, 1)).to(device)
        input_image = torch.ones([1, 2, 16, 16]).to(device)

        return model, [input_image, noise_labels, class_labels]

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (*invar,))

    # Check JIT
    model, invar = setup_model()
    assert common.validate_jit(model, (*invar,))
    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (*invar,))
    # Check Combo
    model, invar = setup_model()
    assert common.validate_combo_optims(model, (*invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_fno_checkpoint(device):
    """Test Song UNet checkpoint save/load"""
    # Construct FNO models
    model_1 = UNet(
        img_resolution=64,
        in_channels=2,
        out_channels=2,
    ).to(device)

    model_2 = UNet(
        img_resolution=64,
        in_channels=2,
        out_channels=2,
    ).to(device)

    noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    input_image = torch.ones([1, 2, 16, 16]).to(device)
    assert common.validate_checkpoint(
        model_1, model_2, (*[input_image, noise_labels, class_labels],)
    )


@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_fnodeploy(device):
    """Test Song Unet deployment support"""
    model = UNet(
        img_resolution=64,
        in_channels=2,
        out_channels=2,
        embedding_type="fourier",
        channel_mult_noise=2,
        encoder_type="residual",
        resample_filter=[1, 3, 3, 1],
    ).to(device)

    noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    input_image = torch.ones([1, 2, 16, 16]).to(device)

    assert common.validate_onnx_export(
        model, (*[input_image, noise_labels, class_labels],)
    )
    assert common.validate_onnx_runtime(
        model, (*[input_image, noise_labels, class_labels],)
    )
