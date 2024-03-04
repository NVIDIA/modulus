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

import torch

from modulus.models.diffusion.preconditioning import EDMPrecondSRV2, _ConditionalPrecond


def test__ConditionalPrecond():

    b, c_target, x, y = 1, 3, 8, 8
    c_cond = 4

    def forward(x, sigma, *, class_labels):
        assert x.shape[1] == c_target + c_cond
        # add mean of full array and sigma, so that changing the scaling will
        # break the regression check
        return x[:, :c_target] + torch.mean(x, dim=1, keepdim=True) + sigma

    preconditioned_model = _ConditionalPrecond(
        model=forward, img_channels=c_target, img_resolution=8
    )

    latents = torch.ones((b, c_target, x, y))
    image_conditioning = torch.arange(b * c_cond * x * y).reshape((b, c_cond, x, y))
    sigma = 10.0
    output = preconditioned_model(
        latents,
        condition=image_conditioning,
        sigma=preconditioned_model.round_sigma(sigma),
    )
    assert output.shape == latents.shape

    # this expected value is a regression check...if you have made an
    # intentional change, feel free to change it
    expected = 45.7331
    assert torch.allclose(torch.tensor(expected), torch.max(output))


def test_EDMPrecondSRV2():
    module = EDMPrecondSRV2(8, 1, 1)
    assert isinstance(module, _ConditionalPrecond)
