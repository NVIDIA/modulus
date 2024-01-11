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
# ruff: noqa: E402
from modulus.model.dlwp_healpix_layers import *
import pytest
import torch
import math
import numpy as np
from . import common

class MulX(torch.nn.Module):
    """ Helper class that just multiplies the values of an input tensor """
    def __init__(self, mulitplier: int = 1):
        super(MulX, self).__init__()
        self.multiplier = multiplier

    def forward(self, x):
        return x * multiplier


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_HEALPixFoldFaces(device):
    fold_func = HEALPixFoldFaces()

    tensor_size = torch.randint(low=1, high=4, size=(5,)).tolist()
    output_size = (tensor_size[0] * tensor_size[1], *tensor_size[2:])
    invar = torch.ones(*tensor_size, device=device)

    
    outvar = fold_func(invar)
    assert outvar.shape == output_size

@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_HEALPixFoldFaces(device):
    unfold_func = HEALPixUnfoldFaces()

    tensor_size = torch.randint(low=1, high=4, size=(4,)).tolist()
    output_size = (tensor_size[0], tensor_size[1], 12, *tensor_size[2:])
    invar = torch.ones(*tensor_size, device=device)

    
    outvar = unfold_func(invar)
    assert outvar.shape == output_size

@pytest.mark.parametrize("device", ["cuda:0", "cpu"], "padding", [2,3,4])
def test_HEALPixPadding(device, padding):

    pad_func = HEALPixPadding(padding)
    
    hw_size = torch.randint(low=4, high=24, size=(1,)).tolist()
    hw_size = np.asarray(hw_size + hw_size)
    # dimes are F, H, W
    # F = 12, and H = W
    tensor_size = (2,12, *hw_size)
    invar = torch.rand(tesnor_size, device=device)

    # Healpix scales as ~N^1/2
    scale = math.ceil(padding**0.5)
    out_size = (2,12, *(hw_size*scale))

    outvar = pad_func(invar)
    assert outvar.size == out_size

@pytest.mark.parametrize("device", ["cuda:0", "cpu"], "multiplier", [2,3,4])
def test_HEALPixLayer(device, multiplier):
    layer = HEALPixLayer(layer=MulX, multiplier=multiplier)

    tensor_size = torch.randint(low=1, high=4, size=(2,)).tolist()
    tensor_size = [2,12,*tensor_size]
    invar = torch.rand(tesnor_size, device=device)

    assert common.compare_output(layer(invar), invar * multiplier)
