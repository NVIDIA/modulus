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

from dataclasses import dataclass

import pytest
import torch

from modulus.metrics.climate.loss import MSE_SSIM, SSIM


@dataclass
class Model(torch.nn.Module):
    """Minimal torch.nn.Module to test MSE_SSIM"""

    output_channels = 2
    output_time_dim = 1
    input_time_dim = 1
    output_variables = ["tcwv", "t2m"]


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_MSE_SSIM(device):
    model = Model()

    ssim_params = {"time_series_forecasting": True}

    mse_ssim_loss = MSE_SSIM(ssim_params=ssim_params)

    # test for fail case of invalid dims (h != w)
    shape = [1, 1, 1, 12, 128, 64]

    ones = torch.ones(shape).to(device)
    try:
        mse_ssim_loss(ones, ones, model)
        assert False, "Failed to error for incorrect number of dimensions"
    except AssertionError:
        pass

    shape = [1, 1, 2, 12, 128, 128]
    print(device)
    ones = torch.ones(shape).to(device)
    zeros = torch.zeros(shape).to(device)

    assert mse_ssim_loss(ones, ones, model) == 0
    assert mse_ssim_loss(ones, zeros, model) == 1

    invar = torch.ones(shape).to(device)
    invar[0, 0, 1, ...] = zeros[0, 0, 0, ...]
    assert mse_ssim_loss(invar, zeros, model) == 0.5


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_SSIM(device):
    """Test SSIM loss in loss"""
    ssim_loss = SSIM()

    shape = [1, 1, 720, 720]

    # Test for exact match
    rand = torch.randn(shape).to(device)

    assert ssim_loss(rand, rand) == 1.0

    # Test for differences
    ones = torch.ones(shape).to(device)
    zeros = torch.zeros(shape).to(device)

    # testing to make sure this runs
    assert ssim_loss(ones, zeros) < 1.0e-4

    # Test window
    # Since SSIM looks over a window rolling will only cause a small dropoff
    eye = torch.eye(720).to(device)
    eye = eye[None, None, ...]

    loss = ssim_loss(eye, torch.roll(eye, 1, -1))  # ~0.9729
    assert 0.97 < loss < 0.98

    # Test fail case for too few dimensions
    var = torch.randn([32])

    try:
        loss = ssim_loss(var, var)
        assert False, "Failed to error for insufficient number of dimensions"
    except IndexError:
        pass
