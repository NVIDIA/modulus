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

from modulus.models.afno import ModAFNO

from . import common


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_modafno_forward(device):
    """Test AFNO forward pass"""
    torch.manual_seed(0)
    model = ModAFNO(
        inp_shape=[32, 32],
        in_channels=2,
        out_channels=1,
        patch_size=[8, 8],
        embed_dim=16,
        depth=2,
        num_blocks=2,
    ).to(device)

    bsize = 2
    invar = torch.randn(bsize, 2, 32, 32).to(device)
    time = torch.full((bsize, 1), 0.5).to(device)
    # Check output size
    assert common.validate_forward_accuracy(model, (invar, time))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_modafno_constructor(device):
    """Test AFNO constructor options"""
    # Define dictionary of constructor args
    arg_list = [
        {
            "inp_shape": [32, 32],
            "in_channels": random.randint(1, 4),
            "out_channels": random.randint(1, 4),
            "patch_size": [8, 8],
            "embed_dim": 4,
            "depth": 2,
            "num_blocks": 2,
        },
        {
            "inp_shape": [8, 16],
            "in_channels": random.randint(1, 4),
            "out_channels": random.randint(1, 4),
            "patch_size": [4, 4],
            "embed_dim": 6,
            "depth": 4,
            "mlp_ratio": 2.0,
            "drop_rate": 0.1,
            "num_blocks": 1,
            "sparsity_threshold": 0.05,
            "hard_thresholding_fraction": 0.9,
        },
    ]
    for kw_args in arg_list:
        # Construct FC model
        model = ModAFNO(**kw_args).to(device)

        bsize = random.randint(1, 16)
        invar = torch.randn(
            bsize,
            kw_args["in_channels"],
            kw_args["inp_shape"][0],
            kw_args["inp_shape"][1],
        ).to(device)
        time = torch.full((bsize, 1), 0.5).to(device)
        outvar = model(invar, time)
        assert outvar.shape == (
            bsize,
            kw_args["out_channels"],
            kw_args["inp_shape"][0],
            kw_args["inp_shape"][1],
        )

    # Also test failure case
    try:
        model = ModAFNO(
            inp_shape=[32, 32],
            in_channels=2,
            out_channels=1,
            patch_size=[8, 8],
            embed_dim=7,
            depth=1,
            num_blocks=4,
        ).to(device)
        raise AssertionError("Failed to error for invalid embed and block number")
    except ValueError:
        pass


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_modafno_optims(device):
    """Test AFNO optimizations"""

    def setup_model():
        """Setups up fresh AFNO model and inputs for each optim test"""
        model = ModAFNO(
            inp_shape=[32, 32],
            in_channels=2,
            out_channels=2,
            patch_size=[8, 8],
            embed_dim=16,
            depth=2,
            num_blocks=2,
        ).to(device)

        bsize = random.randint(1, 5)
        invar = torch.randn(bsize, 2, 32, 32).to(device)
        time = torch.full((bsize, 1), 0.5).to(device)
        return model, invar, time

    # Ideally always check graphs first
    model, invar, time = setup_model()
    assert common.validate_cuda_graphs(model, (invar, time))
    # Check JIT
    model, invar, time = setup_model()
    assert common.validate_jit(model, (invar, time))
    # Check AMP
    model, invar, time = setup_model()
    assert common.validate_amp(model, (invar, time))
    # Check Combo
    model, invar, time = setup_model()
    assert common.validate_combo_optims(model, (invar, time))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_modafno_checkpoint(device):
    """Test AFNO checkpoint save/load"""
    # Construct AFNO models
    model_1 = ModAFNO(
        inp_shape=[32, 32],
        in_channels=2,
        out_channels=2,
        patch_size=[8, 8],
        embed_dim=8,
        depth=2,
        num_blocks=2,
    ).to(device)

    model_2 = ModAFNO(
        inp_shape=[32, 32],
        in_channels=2,
        out_channels=2,
        patch_size=[8, 8],
        embed_dim=8,
        depth=2,
        num_blocks=2,
    ).to(device)

    bsize = random.randint(1, 5)
    invar = torch.randn(bsize, 2, 32, 32).to(device)
    time = torch.full((bsize, 1), 0.5).to(device)
    assert common.validate_checkpoint(model_1, model_2, (invar, time))


@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_modafno_deploy(device):
    """Test AFNO deployment support"""
    # Construct AFNO model
    model = ModAFNO(
        inp_shape=[16, 16],
        in_channels=2,
        out_channels=2,
        patch_size=[8, 8],
        embed_dim=4,
        depth=1,  # Small depth for onnx export speed
        num_blocks=2,
    ).to(device)

    bsize = random.randint(1, 5)
    invar = torch.randn(bsize, 2, 16, 16).to(device)
    time = torch.full((bsize, 1), 0.5).to(device)
    assert common.validate_onnx_export(model, (invar, time))
    assert common.validate_onnx_runtime(model, (invar, time))
