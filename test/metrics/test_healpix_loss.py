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

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pytest
import torch

from modulus.metrics.climate.healpix_loss import BaseMSE, WeightedMSE


@pytest.fixture
def test_data():
    # create dummy data

    # We'll pretend h,w are a lat lon grid instead of healpix
    # so the test works the same
    # Set lat/lon in terms of degrees (for use with _compute_lat_weights)
    def generate_test_data(channels=2, img_shape=(768, 768)):
        x = np.linspace(-180, 180, img_shape[1], dtype=np.float32)
        y = np.linspace(-90, 90, img_shape[0], dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        pred_tensor_np = np.cos(2 * np.pi * yv / (180))
        targ_tensor_np = np.cos(np.pi * yv / (180))

        return channels, pred_tensor_np, targ_tensor_np

    return generate_test_data


@dataclass
class trainer_helper:
    """helper class for setup with the MSE testers"""

    output_variables: Sequence
    device: str


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_BaseMSE(device, test_data, rtol: float = 1e-3, atol: float = 1e-3):
    mse_func = BaseMSE()
    mse_func.setup(None)  # for coverage
    channels, pred_tensor_np, targ_tensor_np = test_data()

    pred_tensor = torch.from_numpy(pred_tensor_np).to(device).expand(channels, -1, -1)
    targ_tensor = torch.from_numpy(targ_tensor_np).to(device).expand(channels, -1, -1)

    # expand out to 6 dimensions
    pred_tensor = pred_tensor[(None,) * 3]
    targ_tensor = targ_tensor[(None,) * 3]

    # test for insufficient dimensions
    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        mse_func(torch.zeros((10,), device=device), targ_tensor)

    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        mse_func(targ_tensor, torch.zeros((10,), device=device))

    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        mse_func(torch.zeros((10,), device=device), torch.zeros((10,), device=device))

    # test for 0 loss
    error = mse_func(targ_tensor, targ_tensor)
    assert torch.allclose(
        error,
        torch.zeros([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    # int( cos(x)^2 - cos(2x)^2 )dx, x = 0...2*pi = pi/4
    # So MSE should be pi/4 / (pi) = 0.25
    error = mse_func(pred_tensor**2, targ_tensor**2)
    assert torch.allclose(
        error,
        0.25 * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    # test for non averaged channesl
    # make the last channel of prediction and target the same
    tensor_size = pred_tensor.shape[-2:]
    ones = torch.ones(tensor_size, device=device)

    pred_tensor = pred_tensor.contiguous()
    targ_tensor = targ_tensor.contiguous()
    pred_tensor[0, 0, 0, -1, ...] = ones[...]
    targ_tensor[0, 0, 0, -1, ...] = ones[...]

    error = mse_func(pred_tensor**2, targ_tensor**2, average_channels=False)

    expected_err = 0.25 * torch.ones(error.shape, dtype=torch.float32, device=device)
    expected_err[-1] = 0

    assert torch.allclose(
        error,
        expected_err,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_WeightedMSE(device, test_data, rtol: float = 1e-3, atol: float = 1e-3):
    num_channels = 3
    channels, pred_tensor_np, targ_tensor_np = test_data(channels=num_channels)

    # first two channels will be as BaseMSE above, the last channel will be 0 loss
    # so per channel loss will be [0.25, 0.25, 0]
    # Giving the last channel half the weight results in a per loss of:
    # [0.25*0.5, 0.25*0.25, 0.0*0.25] == [0.125,0.0625,0]
    # and an average weighted loss of 0.0625
    channel_weights = [0.5, 0.25, 0.25]
    channel_weighted_mse = torch.Tensor([0.125, 0.0625, 0]).to(device)
    mean_weighted_mse = channel_weighted_mse.mean()
    weighted_mse_func = WeightedMSE(channel_weights)

    trainer = trainer_helper(
        output_variables=["a", "b", "ones"],
        device=device,
    )
    weighted_mse_func.setup(trainer)

    # test setup fail case if number of variables doesn't match number of weights
    trainer = trainer_helper(
        output_variables=["a", "b"],
        device=device,
    )
    with pytest.raises(
        ValueError, match="Length of outputs and loss_weights is not the same!"
    ):
        weighted_mse_func.setup(trainer)

    pred_tensor = torch.from_numpy(pred_tensor_np).to(device).expand(channels, -1, -1)
    targ_tensor = torch.from_numpy(targ_tensor_np).to(device).expand(channels, -1, -1)

    tensor_size = pred_tensor.shape[-2:]
    ones = torch.ones(tensor_size, device=device)

    # make the last channel of prediction and target the same
    pred_tensor = pred_tensor.contiguous()
    targ_tensor = targ_tensor.contiguous()
    pred_tensor[-1, ...] = ones[...]
    targ_tensor[-1, ...] = ones[...]

    # expand out to 6 dimensions
    pred_tensor = pred_tensor[(None,) * 3]
    targ_tensor = targ_tensor[(None,) * 3]

    # test for insufficient dimensions
    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        weighted_mse_func(torch.zeros((10,), device=device), targ_tensor)

    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        weighted_mse_func(targ_tensor, torch.zeros((10,), device=device))

    with pytest.raises(
        AssertionError, match="Expected predictions to have 6 dimensions"
    ):
        weighted_mse_func(
            torch.zeros((10,), device=device), torch.zeros((10,), device=device)
        )

    # test for 0 loss
    error = weighted_mse_func(
        pred_tensor**2, targ_tensor**2, average_channels=False
    )
    assert torch.allclose(
        error,
        channel_weighted_mse,
        rtol=rtol,
        atol=atol,
    )

    # test for 0 loss
    error = weighted_mse_func(pred_tensor**2, targ_tensor**2, average_channels=True)
    assert torch.allclose(
        error,
        mean_weighted_mse,
        rtol=rtol,
        atol=atol,
    )
