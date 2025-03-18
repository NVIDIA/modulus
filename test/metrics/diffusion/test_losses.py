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

import pytest
import torch

from modulus.metrics.diffusion import (
    EDMLoss,
    RegressionLossCE,
    ResidualLoss,
    VELoss,
    VELoss_dfsr,
    VPLoss,
)
from modulus.utils.patching import RandomPatching2D

# VPLoss tests


def test_vploss_initialization():
    loss_func = VPLoss()
    assert loss_func.beta_d == 19.9
    assert loss_func.beta_min == 0.1
    assert loss_func.epsilon_t == 1e-5

    loss_func = VPLoss(beta_d=10.0, beta_min=0.5, epsilon_t=1e-4)
    assert loss_func.beta_d == 10.0
    assert loss_func.beta_min == 0.5
    assert loss_func.epsilon_t == 1e-4


def test_sigma_method():
    loss_func = VPLoss()

    # Scalar input
    sigma_val = loss_func.sigma(1.0)
    assert isinstance(sigma_val, torch.Tensor)
    assert sigma_val.item() > 0

    # Tensor input
    t = torch.tensor([1.0, 2.0])
    sigma_vals = loss_func.sigma(t)
    assert sigma_vals.shape == t.shape


def fake_net(y, sigma, labels, augment_labels=None):
    return torch.tensor([1.0])


def fake_condition_net(y, sigma, condition, class_labels=None, augment_labels=None):
    return torch.tensor([1.0])


def test_call_method_vp():
    loss_func = VPLoss()

    images = torch.tensor([[[[1.0]]]])
    labels = None

    # Without augmentation
    loss_value = loss_func(fake_net, images, labels)
    assert isinstance(loss_value, torch.Tensor)

    # With augmentation
    def mock_augment_pipe(imgs):
        return imgs, None

    loss_value_with_augmentation = loss_func(
        fake_net, images, labels, mock_augment_pipe
    )
    assert isinstance(loss_value_with_augmentation, torch.Tensor)


# VELoss tests


def test_veloss_initialization():
    loss_func = VELoss()
    assert loss_func.sigma_min == 0.02
    assert loss_func.sigma_max == 100.0

    loss_func = VELoss(sigma_min=0.01, sigma_max=50.0)
    assert loss_func.sigma_min == 0.01
    assert loss_func.sigma_max == 50.0


def test_call_method_ve():
    loss_func = VELoss()

    images = torch.tensor([[[[1.0]]]])
    labels = None

    # Without augmentation
    loss_value = loss_func(fake_net, images, labels)
    assert isinstance(loss_value, torch.Tensor)

    # With augmentation
    def mock_augment_pipe(imgs):
        return imgs, None

    loss_value_with_augmentation = loss_func(
        fake_net, images, labels, mock_augment_pipe
    )
    assert isinstance(loss_value_with_augmentation, torch.Tensor)


# EDMLoss tests


def test_edmloss_initialization():
    loss_func = EDMLoss()
    assert loss_func.P_mean == -1.2
    assert loss_func.P_std == 1.2
    assert loss_func.sigma_data == 0.5

    loss_func = EDMLoss(P_mean=-2.0, P_std=2.0, sigma_data=0.3)
    assert loss_func.P_mean == -2.0
    assert loss_func.P_std == 2.0
    assert loss_func.sigma_data == 0.3


def test_call_method_edm():
    loss_func = EDMLoss()

    img = torch.tensor([[[[1.0]]]])
    labels = None

    # Without augmentation or conditioning
    loss_value = loss_func(fake_net, img, labels)
    assert isinstance(loss_value, torch.Tensor)

    # With conditioning
    condition = torch.tensor([[[[0.0]]]])
    loss_value = loss_func(fake_condition_net, img, condition=condition, labels=labels)
    assert isinstance(loss_value, torch.Tensor)

    # With augmentation
    def mock_augment_pipe(imgs):
        return imgs, None

    loss_value_with_augmentation = loss_func(fake_net, img, labels, mock_augment_pipe)
    assert isinstance(loss_value_with_augmentation, torch.Tensor)


# RegressionLoss tests


# def test_regressionloss_initialization():
#     loss_func = RegressionLoss()
#     assert loss_func.P_mean == -1.2
#     assert loss_func.P_std == 1.2
#     assert loss_func.sigma_data == 0.5

#     loss_func = RegressionLoss(P_mean=-2.0, P_std=2.0, sigma_data=0.3)
#     assert loss_func.P_mean == -2.0
#     assert loss_func.P_std == 2.0
#     assert loss_func.sigma_data == 0.3


# def fake_net(input, y_lr, sigma, labels, augment_labels=None):
#     return torch.tensor([1.0])


# def test_call_method():
#     loss_func = RegressionLoss()

#     img_clean = torch.tensor([[[[1.0]]]])
#     img_lr = torch.tensor([[[[0.5]]]])
#     labels = None

#     # Without augmentation
#     loss_value = loss_func(fake_net, img_clean, img_lr, labels)
#     assert isinstance(loss_value, torch.Tensor)

#     # With augmentation
#     def mock_augment_pipe(imgs):
#         return imgs, None

#     loss_value_with_augmentation = loss_func(
#         fake_net, img_clean, img_lr, labels, mock_augment_pipe
#     )
#     assert isinstance(loss_value_with_augmentation, torch.Tensor)

# RegressionLossCE tests


def test_regressionlossce_initialization():
    loss_func = RegressionLossCE()
    assert loss_func.prob_channels == [4, 5, 6, 7, 8]

    loss_func = RegressionLossCE(prob_channels=[1, 2, 3, 4])
    assert loss_func.prob_channels == [1, 2, 3, 4]


def leadtime_fake_net(input, y_lr, lead_time_label=None, augment_labels=None):
    return torch.zeros(1, 4, 29, 29)


def test_call_method():
    prob_channels = [0, 2]
    loss_func = RegressionLossCE(prob_channels=prob_channels)

    img_clean = torch.zeros(1, 4, 29, 29)
    img_lr = torch.zeros(1, 4, 29, 29)
    labels = None
    lead_time_label = None

    # Without augmentation
    loss_value = loss_func(
        leadtime_fake_net, img_clean, img_lr, lead_time_label, labels
    )
    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.shape == (1, 3, 29, 29)

    # With augmentation
    def mock_augment_pipe(imgs):
        return imgs, None

    loss_value_with_augmentation = loss_func(
        leadtime_fake_net, img_clean, img_lr, lead_time_label, labels, mock_augment_pipe
    )
    assert isinstance(loss_value_with_augmentation, torch.Tensor)
    assert loss_value.shape == (1, 3, 29, 29)


# ResidualLoss tests


def test_residualloss_initialization():
    # Mock regression network
    regression_net = torch.nn.Linear(1, 1)

    # Test default parameters
    loss_func = ResidualLoss(
        regression_net=regression_net, img_shape_y=32, img_shape_x=16
    )
    assert loss_func.P_mean == 0.0
    assert loss_func.P_std == 1.2
    assert loss_func.sigma_data == 0.5
    assert loss_func.img_shape_y == 32
    assert loss_func.img_shape_x == 16
    assert loss_func.hr_mean_conditioning is False

    # Test custom parameters
    loss_func = ResidualLoss(
        regression_net=regression_net,
        img_shape_y=32,
        img_shape_x=64,
        P_mean=1.0,
        P_std=2.0,
        sigma_data=0.3,
        hr_mean_conditioning=True,
    )
    assert loss_func.P_mean == 1.0
    assert loss_func.P_std == 2.0
    assert loss_func.sigma_data == 0.3
    assert loss_func.img_shape_y == 32
    assert loss_func.img_shape_x == 64
    assert loss_func.hr_mean_conditioning is True


def fake_residual_net(
    x,
    img_lr,
    sigma,
    labels=None,
    global_index=None,
    embedding_selector=None,
    augment_labels=None,
):
    return torch.zeros_like(x)


def test_residualloss_call_method():
    # Mock regression network that returns scaled input
    class DummyRegNet(torch.nn.Module):
        def forward(self, x, *args, **kwargs):
            return 0.9 * x

    regression_net = DummyRegNet()
    loss_func = ResidualLoss(
        regression_net=regression_net, img_shape_y=32, img_shape_x=32
    )

    # Create test inputs
    batch_size = 2
    channels = 3
    img_clean = torch.randn(batch_size, channels, 32, 32)
    img_lr = torch.randn(batch_size, channels, 32, 32)

    # Test without patching or augmentation
    loss_value = loss_func(fake_residual_net, img_clean, img_lr)
    assert isinstance(loss_value, torch.Tensor)
    assert loss_value.shape == (batch_size, channels, 32, 32)

    # Test with augmentation
    def mock_augment_pipe(imgs):
        return imgs, None

    loss_value_with_augmentation = loss_func(
        fake_residual_net, img_clean, img_lr, augment_pipe=mock_augment_pipe
    )
    assert isinstance(loss_value_with_augmentation, torch.Tensor)
    assert loss_value_with_augmentation.shape == (batch_size, channels, 32, 32)

    # Test with patching
    patch_num = 4
    patch_shape = (16, 16)
    patching = RandomPatching2D(
        img_shape=(32, 32), patch_shape=patch_shape, patch_num=patch_num
    )
    loss_value_with_patching = loss_func(
        fake_residual_net, img_clean, img_lr, patching=patching
    )
    assert isinstance(loss_value_with_patching, torch.Tensor)
    # Shape should be (batch_size * patch_num, channels, patch_shape_y, patch_shape_x)
    expected_shape = (batch_size * patch_num, channels, patch_shape[0], patch_shape[1])
    assert loss_value_with_patching.shape == expected_shape

    # Test error on invalid patching object
    with pytest.raises(ValueError):
        loss_func(
            fake_residual_net, img_clean, img_lr, patching="invalid patching object"
        )


# MixtureLoss tests


# def test_mixtureloss_initialization():
#     loss_func = MixtureLoss()
#     assert loss_func.P_mean == -1.2
#     assert loss_func.P_std == 1.2
#     assert loss_func.sigma_data == 0.5

#     loss_func = MixtureLoss(P_mean=-2.0, P_std=2.0, sigma_data=0.3)
#     assert loss_func.P_mean == -2.0
#     assert loss_func.P_std == 2.0
#     assert loss_func.sigma_data == 0.3


# def fake_net(latent, y_lr, sigma, labels, augment_labels=None):
#     return torch.tensor([1.0])


# def test_call_method():
#     loss_func = MixtureLoss()

#     img_clean = torch.tensor([[[[1.0]]]])
#     img_lr = torch.tensor([[[[0.5]]]])
#     labels = None

#     # Without augmentation
#     loss_value = loss_func(fake_net, img_clean, img_lr, labels)
#     assert isinstance(loss_value, torch.Tensor)

#     # With augmentation
#     def mock_augment_pipe(imgs):
#         return imgs, None

#     loss_value_with_augmentation = loss_func(
#         fake_net, img_clean, img_lr, labels, mock_augment_pipe
#     )
#     assert isinstance(loss_value_with_augmentation, torch.Tensor)


# ResLoss tests


# def test_resloss_initialization():
#     # Mock the model loading
#     ResLoss.unet = torch.nn.Linear(1, 1).cuda()

#     loss_func = ResLoss()
#     assert loss_func.P_mean == 0.0
#     assert loss_func.P_std == 1.2
#     assert loss_func.sigma_data == 0.5

#     loss_func = ResLoss(P_mean=-2.0, P_std=2.0, sigma_data=0.3)
#     assert loss_func.P_mean == -2.0
#     assert loss_func.P_std == 2.0
#     assert loss_func.sigma_data == 0.3


# def fake_net(latent, y_lr, sigma, labels, augment_labels=None):
#     return torch.tensor([1.0])


# def test_call_method():
#     # Mock the model loading
#     ResLoss.unet = torch.nn.Linear(1, 1).cuda()

#     loss_func = ResLoss()

#     img_clean = torch.tensor([[[[1.0]]]])
#     img_lr = torch.tensor([[[[0.5]]]])
#     labels = None

#     # Without augmentation
#     loss_value = loss_func(fake_net, img_clean, img_lr, labels)
#     assert isinstance(loss_value, torch.Tensor)

#     # With augmentation
#     def mock_augment_pipe(imgs):
#         return imgs, None

#     loss_value_with_augmentation = loss_func(
#         fake_net, img_clean, img_lr, labels, mock_augment_pipe
#     )
#     assert isinstance(loss_value_with_augmentation, torch.Tensor)


# VELoss_dfsr tests


def test_veloss_dfsr_initialization():
    loss_func = VELoss_dfsr()
    assert loss_func.beta_schedule == "linear"
    assert loss_func.beta_start == 0.0001
    assert loss_func.beta_end == 0.02
    assert loss_func.num_diffusion_timesteps == 1000
    assert loss_func.num_timesteps == loss_func.betas.shape[0]

    loss_func = VELoss_dfsr(
        beta_start=0.0002, beta_end=0.01, num_diffusion_timesteps=500
    )
    assert loss_func.beta_start == 0.0002
    assert loss_func.beta_end == 0.01


def test_get_beta_schedule_method():
    loss_func = VELoss_dfsr()

    beta_schedule = "linear"
    beta_start = 0.0001
    beta_end = 0.02
    num_diffusion_timesteps = 1000

    betas = loss_func.get_beta_schedule(
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        num_diffusion_timesteps=num_diffusion_timesteps,
    )
    betas = torch.from_numpy(betas).float()
    assert num_diffusion_timesteps == betas.shape[0]


def test_call_method_ve_dfsr():
    loss_func = VELoss_dfsr()

    images = torch.tensor([[[[1.0]]]])
    labels = None

    # Without augmentation
    loss_value = loss_func(fake_net, images, labels)
    assert isinstance(loss_value, torch.Tensor)
