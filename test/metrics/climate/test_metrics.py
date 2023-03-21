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
import numpy as np
import pathlib

from modulus.metrics.climate.metrics import Metrics

Tensor = torch.Tensor


@pytest.fixture
def test_data(channels=2, img_shape=(721, 1440)):
    # create dummy data
    time_means = np.zeros((1, channels, img_shape[0], img_shape[1]), dtype=np.float32)
    time_means_path = pathlib.Path(__file__).parent.resolve() / "test_time_means.npy"
    np.save(time_means_path, time_means)

    x = np.linspace(0, 2 * np.pi, img_shape[1], dtype=np.float32)
    y = np.linspace(0, 2 * np.pi, img_shape[0], dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    pred_tensor_np = np.sin(xv)
    targ_tensor_np = np.cos(xv)

    return channels, img_shape, pred_tensor_np, targ_tensor_np, time_means_path


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_climate_metrics(test_data, device):

    channels, img_shape, pred_tensor_np, targ_tensor_np, time_means_path = test_data

    metrics = Metrics(
        img_shape=img_shape,
        clim_mean_path=time_means_path,
        device=torch.device(device),
    )

    pred_tensor = torch.from_numpy(pred_tensor_np).expand(channels, -1, -1).to(device)
    targ_tensor = torch.from_numpy(targ_tensor_np).expand(channels, -1, -1).to(device)

    # sine and cosine no correlation, ACC 0
    assert torch.allclose(
        metrics.weighted_acc(pred_tensor, targ_tensor), torch.zeros(channels).to(device)
    )
    # sine and cosine out of phase, RMSE 1
    assert torch.allclose(
        metrics.weighted_rmse(pred_tensor, targ_tensor), torch.ones(channels).to(device)
    )

    # clean-up the data
    pathlib.Path(time_means_path).unlink(missing_ok=False)
