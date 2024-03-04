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
# ruff: noqa: E402
import os
import sys

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
    input_image = torch.ones([1, 2, 64, 64]).to(device)
    noise_labels = noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)

    assert common.validate_forward_accuracy(
        model,
        (input_image, noise_labels, class_labels),
        file_name="ddmpp_unet_output.pth",
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
        file_name="ncsnpp_unet_output.pth",
        atol=1e-3,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_song_unet_constructor(device):
    """Test the Song UNet constructor options"""

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
            img_resolution=16,
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
def test_song_unet_checkpoint(device):
    """Test Song UNet checkpoint save/load"""
    # Construct FNO models
    model_1 = UNet(
        img_resolution=16,
        in_channels=2,
        out_channels=2,
    ).to(device)

    model_2 = UNet(
        img_resolution=16,
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
def test_son_unet_deploy(device):
    """Test Song UNet deployment support"""
    model = UNet(
        img_resolution=16,
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


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_song_unet_grad_checkpointing(device):
    channels = 2
    img_resolution = 64

    # fix random seeds
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    input_image = torch.ones([1, channels, img_resolution, img_resolution]).to(device)
    noise_labels = noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)

    # Construct the DDM++ UNet model
    model = UNet(
        img_resolution=img_resolution, in_channels=channels, out_channels=channels
    ).to(device)
    y_pred = model(input_image, noise_labels, class_labels)

    # dummy loss
    loss = y_pred.sum()

    # compute gradients
    loss.backward()
    computed_grads = {}
    for name, param in model.named_parameters():
        computed_grads[name] = param.grad.clone()

    # fix random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    input_image = torch.ones([1, channels, img_resolution, img_resolution]).to(device)
    noise_labels = noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)

    # Model with checkpointing enabled
    model_checkpointed = UNet(
        img_resolution=img_resolution,
        in_channels=channels,
        out_channels=channels,
        checkpoint_level=4,
    ).to(device)
    y_pred_checkpointed = model_checkpointed(input_image, noise_labels, class_labels)

    # dummy loss
    loss = y_pred_checkpointed.sum()

    # compute gradients
    loss.backward()
    computed_grads_checkpointed = {}
    for name, param in model.named_parameters():
        computed_grads_checkpointed[name] = param.grad.clone()

    # Check that the results are the same
    assert torch.allclose(
        y_pred_checkpointed, y_pred
    ), "Outputs do not match. Checkpointing failed!"

    # Compare the gradients
    for name in computed_grads:
        torch.allclose(
            computed_grads_checkpointed[name], computed_grads[name]
        ), "Gradient do not match. Checkpointing failed!"
