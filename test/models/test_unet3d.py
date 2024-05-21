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

import random

import pytest
import torch

from modulus.models.unet import UNet3D

from . import common


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_unet3d_forward(device):
    """Test unet3d forward pass"""
    torch.manual_seed(0)
    # Construct unet3d model
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        model_depth=3,
        feature_map_channels=[8, 8, 16, 16, 32, 32],
        num_conv_blocks = 2,
    ).to(device)

    bsize = 2
    invar = torch.randn(bsize, 1, 16, 16, 16).to(device)
    assert common.validate_forward_accuracy(model, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_unet3d_constructor(device):
    """Test unet3d constructor options"""
    # Define dictionary of constructor args
    arg_list = []
    for pooling_type in ["MaxPool3d", "AvgPool3d"]:
        arg_list += [
            {
                "in_channels": random.randint(1, 3),
                "out_channels": random.randint(1, 3),
                "model_depth": 3,
                "feature_map_channels": [8, 8, 16, 16, 32, 32],
                "num_conv_blocks": 2,
                "pooling_type": pooling_type,
            },
            {
                "in_channels": random.randint(1, 3),
                "out_channels": random.randint(1, 3),
                "model_depth": 3,
                "feature_map_channels": [8, 8, 16, 16, 32, 32],
                "num_conv_blocks": 2,
                "pooling_type": pooling_type,
            }
        ]
    for _, kw_args in enumerate(arg_list):
        # Construct model
        model = UNet3D(**kw_args).to(device)
        bsize = random.randint(1, 16)
        invar = torch.randn(bsize, kw_args["in_channels"], 16, 16, 16).to(device)
        outvar = model(invar)
        assert outvar.shape == (bsize, kw_args["out_channels"], *invar.shape[2:])


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_unet3d_optims(device):
    """Test unet3d optimizations"""

    def setup_model():
        """Set up fresh model and inputs for each optim test"""
        # Construct unet3d model
        model = UNet3D(
            in_channels=1,
            out_channels=1,
            model_depth=3,
            feature_map_channels=[4, 4, 8, 8, 16, 16],
            num_conv_blocks = 2,
        ).to(device)

        bsize = 4
        invar = torch.randn(bsize, 1, 16, 16, 16).to(device)
        return model, invar

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (invar,))
    # Check JIT
    model, invar = setup_model()
    assert common.validate_jit(model, (invar,))
    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (invar,))
    # Check Combo
    model, invar = setup_model()
    assert common.validate_combo_optims(model, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_unet3d_checkpoint(device):
    """Test unet3d checkpoint save/load"""
    # Construct unet3d model
    model_1 = UNet3D(
        in_channels=1,
        out_channels=1,
        model_depth=3,
        feature_map_channels=[8, 8, 16, 16, 32, 32],
        num_conv_blocks = 2,
    ).to(device)

    model_2 = UNet3D(
        in_channels=1,
        out_channels=1,
        model_depth=3,
        feature_map_channels=[8, 8, 16, 16, 32, 32],
        num_conv_blocks = 2,
    ).to(device)

    bsize = random.randint(1, 4)
    invar = torch.randn(bsize, 1, 16, 16, 16).to(device)
    assert common.validate_checkpoint(model_1, model_2, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_unet3d_deploy(device):
    """Test unet3d deployment support"""
    # Construct unet3d model
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        model_depth=3,
        feature_map_channels=[8, 8, 16, 16, 32, 32],
        num_conv_blocks = 2,
    ).to(device)

    bsize = random.randint(1, 8)
    invar = torch.randn(bsize, 1, 32, 32, 32).to(device)
    assert common.validate_onnx_export(model, (invar,))
    assert common.validate_onnx_runtime(model, (invar,))