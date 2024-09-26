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
import random
import sys

import numpy as np
import pytest
import torch

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import common
from pytest_utils import import_or_fail

dgl = pytest.importorskip("dgl")


@import_or_fail("dgl")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_topodiff_forward(device, pytestconfig):
    """Test topodiff forward pass"""

    from modulus.models.topodiff import TopoDiff

    torch.manual_seed(0)
    dgl.seed(0)
    np.random.seed(0)
    # Construct Topodiff Model
    model = TopoDiff(img_resolution=64, 
                     in_channels=6, 
                     out_channels=1).to(device)

    bsize = 2
    nsteps = 1000 # diffusion steps 
    tops = torch.randn(bsize, 1, 64, 64).to(device)
    cons = torch.randn(bsize, 5, 64, 64).to(device)
    timesteps = torch.randint(0, nsteps, (bsize,)).to(device)
    out = model(tops, cons, timesteps)
    
    assert out.shape == (bsize, 1, 64, 64)

