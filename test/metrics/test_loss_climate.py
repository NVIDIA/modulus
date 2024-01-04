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

import numpy as np
import pytest
import torch

from modulus.metrics.climate.loss import MSE, SSIM

@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_SSIM(device):
    """Test SSIM loss in loss"""
    ssim_loss = SSIM()
    
    shape = [1, 1, 720, 720]
    
    # Test for exact match
    rand = torch.randn(shape)

    assert ssim_loss(rand,rand) == 1.0

    # Test for differences
    ones = torch.ones(shape)
    zeros = torch.zeros(shape)
    
    assert ssim_loss(ones, zeros) < 1.0e-4

    # Test window
    # Since SSIM looks over a window rolling will only cause a small dropoff
    eye = torch.eye(720)
    eye = eye[None,None,...]

    loss = ssim_loss(eye, torch.roll(eye, 1, -1)) # ~0.9729
    assert 0.97 < loss < 0.98 

    # Test fail case for too few dimensions
    var = torch.randn([32])

    try:
        loss = ssim_loss(var,var)
        assert False, "Failed to error for insufficient number of dimensions"
    except IndexError:
        pass






