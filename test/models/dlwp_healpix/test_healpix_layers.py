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
import sys

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import common
import numpy as np
import pytest
import torch

from modulus.models.dlwp_healpix_layers import (
    HEALPixFoldFaces,
    HEALPixLayer,
    HEALPixPadding,
    HEALPixUnfoldFaces,
)


class MulX(torch.nn.Module):
    """Helper class that just multiplies the values of an input tensor"""

    def __init__(self, multiplier: int = 1):
        super(MulX, self).__init__()
        self.multiplier = multiplier

    def forward(self, x):
        return x * self.multiplier


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_HEALPixFoldFaces_initialization(device):
    fold_func = HEALPixFoldFaces()
    assert isinstance(fold_func, HEALPixFoldFaces)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_HEALPixFoldFaces_forward(device):
    fold_func = HEALPixFoldFaces()

    tensor_size = torch.randint(low=2, high=4, size=(5,)).tolist()
    output_size = (tensor_size[0] * tensor_size[1], *tensor_size[2:])
    invar = torch.ones(*tensor_size, device=device)

    outvar = fold_func(invar)
    assert outvar.shape == output_size

    fold_func = HEALPixFoldFaces(enable_nhwc=True)
    assert fold_func(invar).shape == outvar.shape
    assert fold_func(invar).stride() != outvar.stride()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_HEALPixUnfoldFaces_initialization(device):
    unfold_func = HEALPixUnfoldFaces()
    assert isinstance(unfold_func, HEALPixUnfoldFaces)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_HEALPixUnfoldFaces_forward(device):
    num_faces = 12
    unfold_func = HEALPixUnfoldFaces()

    tensor_size = torch.randint(low=1, high=4, size=(4,)).tolist()
    output_size = (tensor_size[0], num_faces, *tensor_size[1:])

    # first dim is B * num_faces
    tensor_size[0] *= num_faces
    invar = torch.ones(*tensor_size, device=device)

    outvar = unfold_func(invar)
    assert outvar.shape == output_size


HEALPixPadding_testdata = [
    ("cuda:0", 2),
    ("cuda:0", 3),
    ("cuda:0", 4),
    ("cpu", 2),
    ("cpu", 3),
    ("cpu", 4),
]


@pytest.mark.parametrize("device,padding", HEALPixPadding_testdata)
def test_HEALPixPadding_initialization(device, padding):
    pad_func = HEALPixPadding(padding)
    assert isinstance(pad_func, HEALPixPadding)


@pytest.mark.parametrize("device,padding", HEALPixPadding_testdata)
def test_HEALPixPadding_forward(device, padding):
    num_faces = 12  # standard for healpix
    batch_size = 2
    pad_func = HEALPixPadding(padding)

    # test invalid padding size
    with pytest.raises(
        ValueError, match=("invalid value for 'padding', expected int > 0 but got 0")
    ):
        pad_func = HEALPixPadding(0)

    hw_size = torch.randint(low=4, high=24, size=(1,)).tolist()
    c_size = torch.randint(low=3, high=7, size=(1,)).tolist()
    hw_size = np.asarray(hw_size + hw_size)

    # dims are B * F, C, H, W
    # F = 12, and H == W
    # HEALPixPadding expects a folded tensor so fold dims here
    tensor_size = (batch_size * num_faces, *c_size, *hw_size)
    invar = torch.rand(tensor_size, device=device)

    # Healpix adds the padding size to each side
    hw_padded_size = hw_size + (2 * padding)
    out_size = (batch_size * num_faces, *c_size, *hw_padded_size)

    outvar = pad_func(invar)
    assert outvar.shape == out_size


HEALPixLayer_testdata = [
    ("cuda:0", 2),
    ("cuda:0", 3),
    ("cuda:0", 4),
    ("cpu", 2),
    ("cpu", 3),
    ("cpu", 4),
]


@pytest.mark.parametrize("device,multiplier", HEALPixLayer_testdata)
def test_HEALPixLayer_initialization(device, multiplier):
    layer = HEALPixLayer(layer=MulX, multiplier=multiplier)
    assert isinstance(layer, HEALPixLayer)


@pytest.mark.parametrize("device,multiplier", HEALPixLayer_testdata)
def test_HEALPixLayer_forward(device, multiplier):
    layer = HEALPixLayer(layer=MulX, multiplier=multiplier)

    tensor_size = torch.randint(low=2, high=4, size=(1,)).tolist()
    tensor_size = [24, 4, *tensor_size, *tensor_size]
    invar = torch.rand(tensor_size, device=device)
    outvar = layer(invar)

    assert common.compare_output(outvar, invar * multiplier)

    # test nhwc mode and dilation
    layer = HEALPixLayer(
        layer=torch.nn.Conv2d,
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        device=device,
        # dilation=4,
    )

    outvar = layer(invar)

    layer = HEALPixLayer(
        layer=torch.nn.Conv2d,
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        device=device,
        dilation=1,
        enable_healpixpad=True,
        enable_nhwc=True,
    )

    assert outvar.shape == layer(invar).shape
    assert outvar.stride() != layer(invar).stride()

    del layer, outvar, invar
    torch.cuda.empty_cache()
