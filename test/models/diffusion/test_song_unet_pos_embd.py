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

from modulus.models.diffusion import SongUNetPosEmbd as UNet


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_song_unet_forward(device):
    torch.manual_seed(0)
    N_pos = 4
    # Construct the DDM++ UNet model
    model = UNet(img_resolution=64, in_channels=2 + N_pos, out_channels=2).to(device)
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
        in_channels=2 + N_pos,
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
def test_song_unet_global_indexing(device):
    torch.manual_seed(0)
    N_pos = 2
    batch_shape_x = 32
    batch_shape_y = 64
    # Construct the DDM++ UNet model
    model = UNet(
        img_resolution=128,
        in_channels=2 + N_pos,
        out_channels=2,
        gridtype="test",
        N_grid_channels=N_pos,
    ).to(device)
    input_image = torch.ones([1, 2, batch_shape_x, batch_shape_y]).to(device)
    noise_labels = noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    idx_x = torch.arange(45, 45 + batch_shape_x)
    idx_y = torch.arange(12, 12 + batch_shape_y)
    mesh_x, mesh_y = torch.meshgrid(idx_x, idx_y)
    global_index = torch.stack((mesh_x, mesh_y), dim=0)[
        None,
    ].to(device)

    output_image = model(input_image, noise_labels, class_labels, global_index)
    pos_embed = model.positional_embedding_indexing(input_image, global_index)
    assert output_image.shape == (1, 2, batch_shape_x, batch_shape_y)
    assert torch.equal(pos_embed, global_index)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_song_unet_constructor(device):
    """Test the Song UNet constructor options"""

    # DDM++
    img_resolution = 16
    in_channels = 2
    out_channels = 2
    N_pos = 4
    model = UNet(
        img_resolution=img_resolution,
        in_channels=in_channels + N_pos,
        out_channels=out_channels,
    ).to(device)
    noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    input_image = torch.ones([1, 2, 16, 16]).to(device)
    output_image = model(input_image, noise_labels, class_labels)
    assert output_image.shape == (1, out_channels, img_resolution, img_resolution)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_song_unet_position_embedding(device):
    # build unet
    img_resolution = 16
    in_channels = 2
    out_channels = 2
    # NCSN++
    N_pos = 100
    model = UNet(
        img_resolution=img_resolution,
        in_channels=in_channels + N_pos,
        out_channels=out_channels,
        embedding_type="fourier",
        channel_mult_noise=2,
        encoder_type="residual",
        resample_filter=[1, 3, 3, 1],
        gridtype="learnable",
        N_grid_channels=N_pos,
    ).to(device)
    noise_labels = torch.randn([1]).to(device)
    class_labels = torch.randint(0, 1, (1, 1)).to(device)
    input_image = torch.ones([1, 2, 16, 16]).to(device)
    output_image = model(input_image, noise_labels, class_labels)
    assert output_image.shape == (1, out_channels, img_resolution, img_resolution)
    assert model.pos_embd.shape == (100, img_resolution, img_resolution)

    model = UNet(
        img_resolution=img_resolution,
        in_channels=in_channels,
        out_channels=out_channels,
        N_grid_channels=40,
    ).to(device)
    assert model.pos_embd.shape == (40, img_resolution, img_resolution)


def test_fails_if_grid_is_invalid():
    """Test the positional embedding options. "linear" gridtype only support 2 channels, and N_grid_channels in "sinusoidal" should be a factor of 4"""
    img_resolution = 16
    in_channels = 2
    out_channels = 2

    with pytest.raises(ValueError):
        UNet(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            gridtype="linear",
            N_grid_channels=20,
        )

    with pytest.raises(ValueError):
        UNet(
            img_resolution=img_resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            gridtype="sinusoidal",
            N_grid_channels=11,
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_song_unet_optims(device):
    """Test Song UNet optimizations"""

    def setup_model():
        model = UNet(
            img_resolution=16,
            in_channels=6,
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
        in_channels=6,
        out_channels=2,
    ).to(device)

    model_2 = UNet(
        img_resolution=16,
        in_channels=6,
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
        in_channels=6,
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
