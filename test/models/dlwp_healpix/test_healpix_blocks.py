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
def test_HEALPixFoldFaces(device):
    fold_func = HEALPixFoldFaces()

    tensor_size = torch.randint(low=1, high=4, size=(5,)).tolist()
    output_size = (tensor_size[0] * tensor_size[1], *tensor_size[2:])
    invar = torch.ones(*tensor_size, device=device)

    outvar = fold_func(invar)
    assert outvar.shape == output_size


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_HEALPixUnfoldFaces(device):
    num_faces = 12
    unfold_func = HEALPixUnfoldFaces()

    tensor_size = torch.randint(low=1, high=4, size=(4,)).tolist()
    output_size = (tensor_size[0], num_faces, *tensor_size[1:])
    # first dim is B * num_faces
    tensor_size[0] *= num_faces
    invar = torch.ones(*tensor_size, device=device)
    print(
        f"tensor size {tensor_size} output size {output_size} invar shape {invar.shape}"
    )

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
def test_HEALPixPadding(device, padding):
    print(f"TESTING padding {padding}")
    num_faces = 12  # standard for healpix
    batch_size = 2
    pad_func = HEALPixPadding(padding)

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
def test_HEALPixLayer(device, multiplier):
    layer = HEALPixLayer(layer=MulX, multiplier=multiplier)

    tensor_size = torch.randint(low=1, high=4, size=(2,)).tolist()
    tensor_size = [2, 12, *tensor_size]
    invar = torch.rand(tensor_size, device=device)

    assert common.compare_output(layer(invar), invar * multiplier)
